/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.github.io
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

// implementation of the SWS (species weighting scheme) particle
// weighting method: static per-species weights (Species::specwt) with
// Boyd splitting-merging for unequal-weight collisions and
// probabilistic creation/deletion of reaction products
//
// this file holds the SWS methods of both Collide and CollideVSS,
// following the multi-file-class idiom used by collide_swpm.cpp and
// collide_reduce.cpp; declarations live in collide.h / collide_vss.h

#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "collide.h"
#include "collide_vss.h"
#include "grid.h"
#include "update.h"
#include "particle.h"
#include "mixture.h"
#include "react.h"
#include "comm.h"
#include "random_knuth.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

#include <algorithm>

using namespace SPARTA_NS;
using namespace MathConst;

enum{NONE,DISCRETE,SMOOTH};            // several files
enum{CONSTANT,VARIABLE};

#define DELTADELETE 1024
#define DELTAELECTRON 128
#define EPSZERO 1.0e-14

/* ----------------------------------------------------------------------
   SWS - weighted collision attempt count for a single group
   called by attempt_collision(icell,np,volume) when SWS is active
   sws_attempt_wi and sws_maxwi are set per cell by the collision loops
   conventional DSMC (sws=0): Ncoll = 1/2 N fnum (N-1)
   SWS (sws=1): Ncoll = 1/2 count_wi fnum (N-1)
   SWSmax (sws=2): Ncoll = 1/2 N wi_max fnum (N-1)
   the weighting artificially increases the number of trace-species
   numerical particles, so N under SWS is higher than conventional DSMC
------------------------------------------------------------------------- */

double CollideVSS::sws_attempt_one(int icell, int np, double volume)
{
  double fnum = update->fnum;
  double dt = update->dt;

  double nattempt;

  int sws = particle->sws;
  if (sws==1) {
    if (remainflag) {
      nattempt = 0.5 * sws_attempt_wi * (np-1) *
        vremax[icell][0][0] * dt * fnum / volume + remain[icell][0][0];
      remain[icell][0][0] = nattempt - static_cast<int> (nattempt);
    } else {
      nattempt = 0.5 * sws_attempt_wi * (np-1) *
        vremax[icell][0][0] * dt * fnum / volume + random->uniform();
    }
  } else {
    if (remainflag) {
      nattempt = 0.5 * np * sws_maxwi * (np-1) *
        vremax[icell][0][0] * dt * fnum / volume + remain[icell][0][0];
      remain[icell][0][0] = nattempt - static_cast<int> (nattempt);
    } else {
      nattempt = 0.5 * np * sws_maxwi * (np-1) *
        vremax[icell][0][0] * dt * fnum / volume + random->uniform();
    }
  }
  return nattempt;
}

/* ----------------------------------------------------------------------
   SWS - weighted pair count for a group pair
   called by attempt_collision(icell,igroup,jgroup,volume) under SWS
   count_wi_group[] and maxwigr[] are filled per cell by
   sws_group_weights()
------------------------------------------------------------------------- */

double CollideVSS::sws_group_npairs(int igroup, int jgroup)
{
  int sws = particle->sws;

  double npairs;
  if (igroup == jgroup) {
    if (sws==1) npairs = 0.5 * count_wi_group[igroup] * (ngroup[igroup]-1);
    else npairs = 0.5 * ngroup[igroup] * maxwigr[igroup] * (ngroup[igroup]-1);
  } else {
    if (sws==1) npairs = count_wi_group[igroup] * (ngroup[jgroup]);
    else npairs = ngroup[igroup] * maxwigr[igroup] * (ngroup[jgroup]);
  }
  return npairs;
}

/* ----------------------------------------------------------------------
   SWS - acceptance-rejection scale for the SWSmax variant (sws=2)
   pair candidates are less likely to be selected if both particles
   have low weights; goes along with Ncoll = 1/2 N wi_max fnum (N-1)
------------------------------------------------------------------------- */

double CollideVSS::sws_test_scale(int ispecies, int jspecies)
{
  Particle::Species *species = particle->species;
  double w_ipart = species[ispecies].specwt;
  double w_jpart = species[jspecies].specwt;
  return MAX(w_ipart,w_jpart)/sws_maxwi;
}

/* ----------------------------------------------------------------------
   SWS - Ewilost re-injection for setup_collision()
   returns the pooled split-merge energy (draining the pool) when both
   collision partners carry the max species weight, else 0.0
   sws_species_maxwt (the run-constant max over all species) is resolved
   once in Collide::setup()
------------------------------------------------------------------------- */

double CollideVSS::sws_ewilost_take(int isp, int jsp)
{
  Particle::Species *species = particle->species;

  double w_i = species[isp].specwt;
  double w_j = species[jsp].specwt;

  if ((w_i==sws_species_maxwt) && (w_j==sws_species_maxwt)) {
    double etake = Ewilost;
    Ewilost = 0.0;
    return etake;
  }
  return 0.0;
}

/* ----------------------------------------------------------------------
   SWS - pre-collision capture for perform_collision()
   copies the reactants (react->attempt() may change their species) and
   resets the sws_n_* product multiplicities to the no-reaction values
------------------------------------------------------------------------- */

void CollideVSS::sws_perform_prep(Particle::OnePart *ip, Particle::OnePart *jp)
{
  sws_ip_pre = *ip;
  sws_jp_pre = *jp;
  sws_n_i = 1;
  sws_n_j = sws_n_k = sws_n_pre = 0;
}

/* ----------------------------------------------------------------------
   SWS - probabilistic split of the max-weight reactant after a reaction
   draws n_pre; if 1, appends a copy of the pre-reaction max-weight
   reactant (its un-reacted portion) to the master particle list
   also computes the product survival ratios phi_i/phi_j from the min
   PRE-reaction weight and the POST-reaction species weights
------------------------------------------------------------------------- */

void CollideVSS::sws_pre_split(Particle::OnePart *&ip, Particle::OnePart *&jp)
{
  double x[3],v[3];
  Particle::Species *species = particle->species;

  double w_i = species[sws_ip_pre.ispecies].specwt;
  double w_j = species[sws_jp_pre.ispecies].specwt;
  double w_min = std::min(w_i, w_j);
  double w_max = std::max(w_i, w_j);

  // number of split particles (0 or 1)

  if (w_i == w_j) sws_n_pre = 0;
  else sws_n_pre = ((((w_max-w_min)/w_max)/random->uniform()>1)?1:0);

  // particle creation of the un-reacted portion of the major reactant

  if (sws_n_pre == 1) {
    Particle::OnePart *maxp_pre = ((w_i > w_j) ? &sws_ip_pre : &sws_jp_pre);
    int id = MAXSMALLINT*random->uniform();
    Particle::OnePart *particles = particle->particles;
    memcpy(x,maxp_pre->x,3*sizeof(double));
    memcpy(v,maxp_pre->v,3*sizeof(double));
    int reallocflag =
      particle->add_particle(id,maxp_pre->ispecies,maxp_pre->icell,x,v,
                             maxp_pre->erot,maxp_pre->evib);
    if (reallocflag) {
      ip = particle->particles + (ip - particles);
      jp = particle->particles + (jp - particles);
    }
  }

  sws_w_min = w_min;
  sws_phi_i = w_min/species[ip->ispecies].specwt;
  sws_phi_j = w_min/species[jp->ispecies].specwt;
}

/* ----------------------------------------------------------------------
   SWS - single product-multiplicity draw from survival ratio phi
------------------------------------------------------------------------- */

int CollideVSS::sws_draw_count(double phi)
{
  if (phi < 1.0) return (((phi)/random->uniform()>1)?1:0);
  return int(phi)+(((phi-int(phi))/random->uniform()>1)?1:0);
}

/* ----------------------------------------------------------------------
   SWS - draw the product multiplicities for a reaction
   nprod = 1: recombination (single product I)
   nprod = 2: exchange or associative ionization (products I,J)
   nprod = 3: dissociation or impact ionization (products I,J,K)
   the i,j,k draw order is pinned by the original implementation
------------------------------------------------------------------------- */

void CollideVSS::sws_draw_products(int nprod, Particle::OnePart *kp)
{
  Particle::Species *species = particle->species;

  sws_n_i = sws_draw_count(sws_phi_i);
  if (nprod >= 2) sws_n_j = sws_draw_count(sws_phi_j);
  else sws_n_j = 0;
  if (nprod == 3)
    sws_n_k = sws_draw_count(sws_w_min/species[kp->ispecies].specwt);
  else sws_n_k = 0;
}

/* ----------------------------------------------------------------------
   SWS - splitting-merging velocity blend after two-body scattering
   the higher-weight particle keeps only a phi fraction of its
   post-collision velocity change (only that fraction of the molecules
   it represents took part); the unconserved kinetic energy is pooled
   in Ewilost for later re-injection by sws_ewilost_take()
   with equally weighted partners phi = 1 and the blend is a no-op
------------------------------------------------------------------------- */

void CollideVSS::sws_scatter_merge(Particle::OnePart *ip,
                                   Particle::OnePart *jp,
                                   const double *vi_pre,
                                   const double *vj_pre)
{
  Particle::Species *species = particle->species;
  double *vi = ip->v;
  double *vj = jp->v;
  int isp = ip->ispecies;
  int jsp = jp->ispecies;
  double mass_i = species[isp].mass;
  double mass_j = species[jsp].mass;

  double w_i = species[isp].specwt;
  double w_j = species[jsp].specwt;
  double phi = 1.0;

  double vi_post[3];
  double vj_post[3];

  if ((w_i>0) && (w_j>0)){
    if (w_i>w_j){
      phi = w_j/w_i;
    } else {
      phi = w_i/w_j;
    }
  }

    if (w_i>w_j){
      vj_post[0] = vj[0];
      vj_post[1] = vj[1];
      vj_post[2] = vj[2];
      
      vi_post[0] = (1-phi)*vi_pre[0]+phi*vi[0];
      vi_post[1] = (1-phi)*vi_pre[1]+phi*vi[1];
      vi_post[2] = (1-phi)*vi_pre[2]+phi*vi[2];
  
      Ewilost += w_i*0.5*mass_i*phi*(1-phi)*(
      pow((vi_pre[0]-vi[0]),2.0)+
      pow((vi_pre[1]-vi[1]),2.0)+
      pow((vi_pre[2]-vi[2]),2.0)
      );
  
      vi[0]=vi_post[0];
      vi[1]=vi_post[1];
      vi[2]=vi_post[2];
  
    } else {
      vi_post[0] = vi[0];
      vi_post[1] = vi[1];
      vi_post[2] = vi[2];
      
      vj_post[0] = (1-phi)*vj_pre[0]+phi*vj[0];
      vj_post[1] = (1-phi)*vj_pre[1]+phi*vj[1];
      vj_post[2] = (1-phi)*vj_pre[2]+phi*vj[2];
  
      Ewilost += w_j*0.5*mass_j*phi*(1-phi)*(
      pow((vj_pre[0]-vj[0]),2.0)+
      pow((vj_pre[1]-vj[1]),2.0)+
      pow((vj_pre[2]-vj[2]),2.0)
      );
  
      vj[0]=vj_post[0];
      vj[1]=vj_post[1];
      vj[2]=vj_post[2];
    }
}

/* ----------------------------------------------------------------------
   SWS - per-particle blend factor for the energy-exchange kernel
   phi = weight ratio of the pair (1.0 if either weight is <= 0)
   return 1 if P is the higher-weight (major) particle of the pair
------------------------------------------------------------------------- */

int CollideVSS::sws_eexchange_phi(Particle::OnePart *p, Particle::OnePart *p2,
                                  double &phi)
{
  Particle::Species *species = particle->species;

  double w_p = species[p->ispecies].specwt;
  double w_p2 = species[p2->ispecies].specwt;

  phi = 1.0;
  if ((w_p>0) && (w_p2>0)) {
    if (w_p>w_p2) phi = w_p2/w_p;
    else phi = w_p/w_p2;
  }
  return (w_p > w_p2);
}



/* ----------------------------------------------------------------------
   SWS - per-cell setup for the single-group and ambipolar-one loops
   sws_attempt_wi = weighted count used by the attempt formula
   sws_maxwi = max species weight over the NP heavy particles in plist
   (both read by attempt_collision()/test_collision() under SWS)
------------------------------------------------------------------------- */

void Collide::sws_cell_prep(int np, double count_wi)
{
  Particle::Species *species = particle->species;
  Particle::OnePart *particles = particle->particles;

  double maxwi = 0.0;
  for (int m = 0; m < np; m++)
    maxwi = std::max(species[particles[plist[m]].ispecies].specwt,maxwi);

  sws_maxwi = maxwi;
  sws_attempt_wi = count_wi;
}


/* ----------------------------------------------------------------------
   SWS - per-group weight sums and maxima for the N particles in plist
   fills count_wi_group[] and maxwigr[] (sized ngroups), read by the
   group attempt formula; sets sws_maxwi = cell max species weight
------------------------------------------------------------------------- */

void Collide::sws_group_weights(int n)
{
  Particle::Species *species = particle->species;
  Particle::OnePart *particles = particle->particles;
  int *species2group = mixture->species2group;

  for (int igr = 0; igr < ngroups; igr++) {
    count_wi_group[igr] = 0.0;
    maxwigr[igr] = 0.0;
  }

  for (int m = 0; m < n; m++) {
    int isp = particles[plist[m]].ispecies;
    int igroup = species2group[isp];
    double wi = species[isp].specwt;
    count_wi_group[igroup] += wi;
    maxwigr[igroup] = std::max(wi,maxwigr[igroup]);
  }

  double maxwi = 0.0;
  for (int igr = 0; igr < ngroups; igr++)
    maxwi = std::max(maxwigr[igr],maxwi);
  sws_maxwi = maxwi;
}

/* ----------------------------------------------------------------------
   SWS - bookkeeping for particles created/destroyed by a reaction in the
   single-group and near-neighbor collision loops
   perform_collision() under SWS appended new particles to the master
   particle list in this order:
     p_pre (if n_pre) = un-reacted portion of the max-weight reactant
     kp    (if kpart) = 3rd product particle (dissociation/ionization)
   n_i/n_j/n_k = # of copies of the I/J/K products to keep (0 = delete)
   i,j = plist slots of the two reactants
   nearcp = 1 if the near-neighbor arrays must be maintained
   return 1 if fewer than 2 particles remain in the cell (caller breaks)
------------------------------------------------------------------------- */

int Collide::sws_products_one(int &np, int i, int j,
                              Particle::OnePart *ipart,
                              Particle::OnePart *jpart,
                              Particle::OnePart *kpart,
                              int n_i, int n_j, int n_k, int n_pre,
                              int nearcp)
{
  int i_loop;
  double x[3],v[3];
  Particle::OnePart *particles = particle->particles;

  // indices in the master particle list of the appended particles

  int kp_index = -1;
  int pre_index = -1;
  if (kpart) {
    kp_index = particle->nlocal-1;
    if (n_pre) pre_index = particle->nlocal-2;
  } else if (n_pre) pre_index = particle->nlocal-1;

  // add kp to plist if kept, else flag it for deletion
  // kp was never added to plist, so deletion cannot touch plist

  if (kpart) {
    if (n_k) {
      if (np == npmax) {
        npmax += DELTAPART;
        memory->grow(plist,npmax,"collide:plist");
      }
      if (nearcp) set_nn(np);
      plist[np++] = kp_index;
      particles = particle->particles;
    } else {
      if (ndelete == maxdelete) {
        maxdelete += DELTADELETE;
        memory->grow(dellist,maxdelete,"collide:dellist");
      }
      dellist[ndelete++] = kp_index;
      kpart = NULL;
    }
  }

  // p_pre (un-reacted portion of max-weight reactant) is always kept

  if (n_pre) {
    if (np == npmax) {
      npmax += DELTAPART;
      memory->grow(plist,npmax,"collide:plist");
    }
    if (nearcp) set_nn(np);
    plist[np++] = pre_index;
    particles = particle->particles;
  }

  // delete reactant I from plist if destroyed by probability
  // if the last plist entry swapped into slot i is reactant J,
  // update j to follow it

  if (!n_i) {
    if (ndelete == maxdelete) {
      maxdelete += DELTADELETE;
      memory->grow(dellist,maxdelete,"collide:dellist");
    }
    dellist[ndelete++] = plist[i];
    np--;
    plist[i] = plist[np];
    if (nearcp) nn_last_partner[i] = nn_last_partner[np];
    if (j == np) j = i;
  }

  // delete reactant J from plist if destroyed
  // by probability or recombination

  if (!n_j) {
    if (ndelete == maxdelete) {
      maxdelete += DELTADELETE;
      memory->grow(dellist,maxdelete,"collide:dellist");
    }
    dellist[ndelete++] = plist[j];
    np--;
    plist[j] = plist[np];
    if (nearcp) nn_last_partner[j] = nn_last_partner[np];
  }

  // create the n-1 extra copies of each surviving product

  if (ipart) {
    for (i_loop = 0; i_loop < n_i-1 ; i_loop++) {
      int id = MAXSMALLINT*random->uniform();
      memcpy(x,ipart->x,3*sizeof(double));
      memcpy(v,ipart->v,3*sizeof(double));
      int reallocflag =
      particle->add_particle(id,ipart->ispecies,ipart->icell,x,v,ipart->erot,ipart->evib);
      if (reallocflag) {
        if(ipart) ipart = particle->particles + (ipart - particles);
        if(jpart) jpart = particle->particles + (jpart - particles);
        if(kpart) kpart = particle->particles + (kpart - particles);
      }
      if (np == npmax) {
        npmax += DELTAPART;
        memory->grow(plist,npmax,"collide:plist");
      }
      if (nearcp) set_nn(np);
      plist[np++] = particle->nlocal-1;
      particles = particle->particles;
    }
  }

  if (jpart) {
    for (i_loop = 0; i_loop < n_j-1 ; i_loop++)  {
      int id = MAXSMALLINT*random->uniform();
      memcpy(x,jpart->x,3*sizeof(double));
      memcpy(v,jpart->v,3*sizeof(double));
      int reallocflag =
      particle->add_particle(id,jpart->ispecies,jpart->icell,x,v,jpart->erot,jpart->evib);
      if (reallocflag) {
        if(ipart) ipart = particle->particles + (ipart - particles);
        if(jpart) jpart = particle->particles + (jpart - particles);
        if(kpart) kpart = particle->particles + (kpart - particles);
      }
      if (np == npmax) {
        npmax += DELTAPART;
        memory->grow(plist,npmax,"collide:plist");
      }
      if (nearcp) set_nn(np);
      plist[np++] = particle->nlocal-1;
      particles = particle->particles;
    }
  }

  if (kpart) {
    for (i_loop = 0; i_loop < n_k-1 ; i_loop++) {
      int id = MAXSMALLINT*random->uniform();
      memcpy(x,kpart->x,3*sizeof(double));
      memcpy(v,kpart->v,3*sizeof(double));
      int reallocflag =
      particle->add_particle(id,kpart->ispecies,kpart->icell,x,v,kpart->erot,kpart->evib);
      if (reallocflag) {
        if(ipart) ipart = particle->particles + (ipart - particles);
        if(jpart) jpart = particle->particles + (jpart - particles);
        if(kpart) kpart = particle->particles + (kpart - particles);
      }
      if (np == npmax) {
        npmax += DELTAPART;
        memory->grow(plist,npmax,"collide:plist");
      }
      if (nearcp) set_nn(np);
      plist[np++] = particle->nlocal-1;
      particles = particle->particles;
    }
  }

  if (np < 2) return 1;
  return 0;
}


/* ----------------------------------------------------------------------
   SWS - bookkeeping for particles created/destroyed by a reaction in the
   single-group ambipolar collision loop
   np/nelectron are the caller's live counts (updated here)
   i,j = indices of the two reactants in the combined plist+elist space
   npstart = np at pair-selection time (elist offset for j)
   jspecies = J species before collision chemistry
   caller must refresh its particles/ionambi/velambi pointers on return
------------------------------------------------------------------------- */

void Collide::sws_products_one_ambipolar(int &np, int &nelectron,
                                         int i, int j, int npstart,
                                         int jspecies,
                                         Particle::OnePart *ipart,
                                         Particle::OnePart *jpart,
                                         Particle::OnePart *kpart,
                                         int n_i, int n_j, int n_k, int n_pre)
{
  int i_loop;
  int kpindex = -1;
  double x[3],v[3];
  int nbytes = sizeof(Particle::OnePart);
  Particle::OnePart *ep;

  Particle::OnePart *particles = particle->particles;
  int *ionambi = particle->eivec[particle->ewhich[index_ionambi]];
  double **velambi = particle->edarray[particle->ewhich[index_velambi]];

  // SWS - indices in the master particle list of particles appended by
  // perform_collision() under SWS: p_pre first (if n_pre), then kp (if kpart)

  int kp_index = -1;
  int pre_index = -1;
  if (kpart) {
    kp_index = particle->nlocal-1;
    if (n_pre) pre_index = particle->nlocal-2;
  } else if (n_pre) pre_index = particle->nlocal-1;

  if (kpart || n_pre) {
    particles = particle->particles;
    ionambi = particle->eivec[particle->ewhich[index_ionambi]];
    velambi = particle->edarray[particle->ewhich[index_velambi]];
  }

  // kp handling:
  // heavy kp kept (n_k): add to plist, remember its particle index in k
  // heavy kp discarded (!n_k): flag for deletion, never entered plist
  // electron kp: create n_k copies in elist, then remove kp from the
  //   master particle list (it is the last entry, so nlocal-- is safe;
  //   pre_index = nlocal-2 remains valid)

  if (kpart) {
    if (kpart->ispecies != ambispecies) {
      if (n_k) {
        if (np == npmax) {
          npmax += DELTAPART;
          memory->grow(plist,npmax,"collide:plist");
        }
        plist[np++] = kp_index;
        // save particle index of kp for the copy loop below
        kpindex = kp_index;
      } else {
        if (ndelete == maxdelete) {
          maxdelete += DELTADELETE;
          memory->grow(dellist,maxdelete,"collide:dellist");
        }
        dellist[ndelete++] = kp_index;
        kpart = NULL;
      }
    } else {
      for (i_loop = 0; i_loop < n_k; i_loop++) {
        if (nelectron == maxelectron) {
          maxelectron += DELTAELECTRON;
          elist = (Particle::OnePart *)
            memory->srealloc(elist,maxelectron*nbytes,"collide:elist");
        }
        ep = &elist[nelectron];
        memcpy(ep,kpart,nbytes);
        ep->ispecies = ambispecies;
        nelectron++;
      }
      particle->nlocal--;
      kpart = NULL;
    }
  }

  // p_pre = un-reacted portion of the max-weight reactant, always kept
  // heavy: add to plist as a neutral
  // electron: copy to elist, flag master-list entry for deletion

  if (n_pre) {
    Particle::OnePart *p_pre = &particle->particles[pre_index];
    if (p_pre->ispecies != ambispecies) {
      if (np == npmax) {
        npmax += DELTAPART;
        memory->grow(plist,npmax,"collide:plist");
      }
      plist[np++] = pre_index;
      ionambi[pre_index] = 0;
    } else {
      if (nelectron == maxelectron) {
        maxelectron += DELTAELECTRON;
        elist = (Particle::OnePart *)
          memory->srealloc(elist,maxelectron*nbytes,"collide:elist");
      }
      ep = &elist[nelectron];
      memcpy(ep,p_pre,nbytes);
      ep->ispecies = ambispecies;
      nelectron++;
      if (ndelete == maxdelete) {
        maxdelete += DELTADELETE;
        memory->grow(dellist,maxdelete,"collide:dellist");
      }
      dellist[ndelete++] = pre_index;
    }
  }

  // if jpart exists, was originally not an electron, now is an electron:
  //   ionization reaction converted 2 neutrals to one ion
  //   add to elist, remove from plist, flag J for deletion
  // if jpart exists, was originally an electron, now is not an electron:
  //   exchange reaction converted ion + electron to two neutrals
  //   add neutral J to master particle list, remove from elist, add to plist
  // if jpart destroyed, was an electron:
  //   recombination reaction converted ion + electron to one neutral
  //   remove electron from elist
  // else if jpart destroyed:
  //   non-ambipolar recombination reaction
  //   remove from plist, flag J for deletion

  //==================================================================
  // first delete i and k particle if number of them will be zero.
  // secondary, i, j, k particle is added by copy paste 
  // here, in ambipolar, treatment of j is a bit complicated,
  // therefore, j particle delete and add part is same.
  // 
  // Be careful when respecifying a pointer when reallocflag is 1.
  //================================================================== 
  
  // delete reactant I from plist if destroyed by probability (n_i = 0)
  // if the last plist entry swapped into slot i is reactant J,
  // update j to follow it

  if (!n_i && ipart ) {
    if (ndelete == maxdelete) {
      maxdelete += DELTADELETE;
      memory->grow(dellist,maxdelete,"collide:dellist");
    }
    dellist[ndelete++] = plist[i];
    np--;
    plist[i] = plist[np];
    // j indexes plist only when j < npstart (else it indexes elist)
    if (j < npstart && j == np) j = i;
  }

  // copy paste i particle 
  // i is always non ambipolar particle because of reactoin style limitation
  if (ipart) {
    // printf("!!check cp i \n");
    for (i_loop = 0; i_loop < n_i-1 ; i_loop++) {   
      particles = particle->particles;    
      int id = MAXSMALLINT*random->uniform();
      memcpy(x,ipart->x,3*sizeof(double));
      memcpy(v,ipart->v,3*sizeof(double));
      int reallocflag = 
      particle->add_particle(id,ipart->ispecies,ipart->icell,x,v,ipart->erot,ipart->evib); 
      if (np == npmax) {
        npmax += DELTAPART;
        memory->grow(plist,npmax,"collide:plist");
      }
      if (reallocflag) {
        ionambi = particle->eivec[particle->ewhich[index_ionambi]];
        velambi = particle->edarray[particle->ewhich[index_velambi]];
        if(ipart) ipart = particle->particles + (ipart - particles);
        if(jpart) jpart = particle->particles + (jpart - particles);
        if(kpart) kpart = particle->particles + (kpart - particles);
      }          
      plist[np++] = particle->nlocal-1;
      particles = particle->particles;
      // ionambi is set when paticle copy pasted
      ionambi[particle->nlocal-1] = ionambi[plist[i]];
    }
  }

  // Particle j is not treated the same as i and k
  // j Particle plist elist move, delete, and copy-paste of 
  // particle of ambipolar-involved reaction is done here.

  // if jpart exists, was originally not an electron, now is an electron:
  //   ionization reaction converted 2 neutrals to one ion
  //   add to elist, remove from plist, flag J for deletion
  // if jpart exists, was originally an electron, now is not an electron:
  //   exchange reaction converted ion + electron to two neutrals
  //   add neutral J to master particle list, remove from elist, add to plist
  // if jpart destroyed, was an electron:
  //   recombination reaction converted ion + electron to one neutral
  //   remove electron from elist
  // else if jpart destroyed:
  //   non-ambipolar recombination reaction
  //   remove from plist, flag J for deletion

  // need to save the information of  jpart ,ambipolar electron
  // because it will be deleted. to reproduce correctly, use jp.
  // jpart is NULL if the reaction destroyed it (e.g. recombination)

  Particle::OnePart jp;
  if (jpart) jp = *jpart;

  if (jpart) {
      // printf("!!check process j \n");
      if (jspecies != ambispecies && jpart->ispecies == ambispecies) { 
        for (i_loop = 0; i_loop < n_j; i_loop++) { 
          // loop is added to create additional electron because jpart will be NULL 
	            if (nelectron == maxelectron) {
	              maxelectron += DELTAELECTRON;
	              elist = (Particle::OnePart *)
	                memory->srealloc(elist,maxelectron*nbytes,"collide:elist");
	            }
	            ep = &elist[nelectron];
	            memcpy(ep,jpart,nbytes);
	            ep->ispecies = ambispecies;
	            nelectron++;
        }
	          jpart = NULL;
	        } else if (jspecies == ambispecies && jpart->ispecies != ambispecies) {
        // even particle is not created, delete ambi particle here
        //if (nelectron-1 != j-np) memcpy(&elist[j-np],&elist[nelectron-1],nbytes);
        
        // np can be changed by other reactions, npstart is used
        if (nelectron-1 != j-npstart) memcpy(&elist[j-npstart],&elist[nelectron-1],nbytes);
        nelectron--;
        for (i_loop = 0; i_loop < n_j; i_loop++) {
          // loop is added to create additional j particle
          int id = MAXSMALLINT*random->uniform();
          memcpy(x,jp.x,3*sizeof(double));
          memcpy(v,jp.v,3*sizeof(double));
          int reallocflag = particle->add_particle(id,jp.ispecies,jp.icell,x,v,jp.erot,jp.evib);
	            //int reallocflag = particle->add_particle();
	            if (reallocflag) {
	              particles = particle->particles;
	              ionambi = particle->eivec[particle->ewhich[index_ionambi]];
	              velambi = particle->edarray[particle->ewhich[index_velambi]];
            if(ipart) ipart = particle->particles + (ipart - particles);
            if(jpart) jpart = particle->particles + (jpart - particles);
            if(kpart) kpart = particle->particles + (kpart - particles);
	            }
	            int index = particle->nlocal-1;
	            // memcpy(&particles[index],jpart,nbytes);
	            // particles[index].id = MAXSMALLINT*random->uniform();
	            ionambi[index] = 0;
          //if (i_loop == 0) {
	            //  if (nelectron-1 != j-np) memcpy(&elist[j-np],&elist[nelectron-1],nbytes);
	            //  nelectron--;
          //}
	            if (np == npmax) {
	              npmax += DELTAPART;
	              memory->grow(plist,npmax,"collide:plist");
	            }
	            plist[np++] = index;
	        }
    }
  }

  // remove product major particle with the probability
  // with current assumption, electron cannot be a major
  // thus only neutral is considered
  if ((jpart && !n_j) && (jpart->ispecies != ambispecies && jspecies != ambispecies)) {
    if (ndelete == maxdelete) {
      maxdelete += DELTADELETE;
      memory->grow(dellist,maxdelete,"collide:dellist");
    }
    dellist[ndelete++] = plist[j];
    np--;
    plist[j] = plist[np];        
  }   

  if (!jpart && jspecies == ambispecies) {
    //if (nelectron-1 != j-np) memcpy(&elist[j-np],&elist[nelectron-1],nbytes);
    if (nelectron-1 != j-npstart) memcpy(&elist[j-npstart],&elist[nelectron-1],nbytes);
    nelectron--;
  } else if (!jpart) {
    if (ndelete == maxdelete) {
      maxdelete += DELTADELETE;
      memory->grow(dellist,maxdelete,"collide:dellist");
    }
    dellist[ndelete++] = plist[j];
    plist[j] = plist[np-1];
    np--;
  }
  
  // copy and paste of j particles after reactions not involving j ambispecies 
  // now j is not ambipolar electron 
  // particle j is already exist 
  if (jpart) {
    // printf("!!check cp j \n");
    if (jpart->ispecies != ambispecies && jspecies != ambispecies) { 
      for ( i_loop = 0; i_loop < n_j-1 ; i_loop++)  {
          int id = MAXSMALLINT*random->uniform();
          memcpy(x,jpart->x,3*sizeof(double));
          memcpy(v,jpart->v,3*sizeof(double));
          int reallocflag = 
          particle->add_particle(id,jpart->ispecies,jpart->icell,x,v,jpart->erot,jpart->evib); 
          if (np == npmax) {
            npmax += DELTAPART;
            memory->grow(plist,npmax,"collide:plist");
          }
          if (reallocflag) {
            ionambi = particle->eivec[particle->ewhich[index_ionambi]];
            velambi = particle->edarray[particle->ewhich[index_velambi]];
            if(ipart) ipart = particle->particles + (ipart - particles);
            if(jpart) jpart = particle->particles + (jpart - particles);
            if(kpart) kpart = particle->particles + (kpart - particles);                
          }                        
          plist[np++] = particle->nlocal-1;
          particles = particle->particles;
          ionambi[particle->nlocal-1] = ionambi[plist[j]];
        }
      }
    }      


  
  // copy paste kpart particle
  // electron kp copies were all created in elist above (kpart = NULL),
  // so only heavy kp copies are created here
  if (kpart) {
    for (i_loop = 0; i_loop < n_k-1 ; i_loop++) {
      particles = particle->particles;
      int id = MAXSMALLINT*random->uniform();
      memcpy(x,kpart->x,3*sizeof(double));
      memcpy(v,kpart->v,3*sizeof(double));
      int reallocflag =
      particle->add_particle(id,kpart->ispecies,kpart->icell,x,v,kpart->erot,kpart->evib);
      if (reallocflag) {
        ionambi = particle->eivec[particle->ewhich[index_ionambi]];
        velambi = particle->edarray[particle->ewhich[index_velambi]];
        kpart = particle->particles + (kpart - particles);
      }
      if (np == npmax) {
        npmax += DELTAPART;
        memory->grow(plist,npmax,"collide:plist");
      }
      plist[np++] = particle->nlocal-1;
      particles = particle->particles;
      // k = particle index of the original kp
      ionambi[particle->nlocal-1] = ionambi[kpindex];
    }
  }

}

/* ----------------------------------------------------------------------
   explicit template instantiations for the SWS collision loops
------------------------------------------------------------------------- */

namespace SPARTA_NS {
}
