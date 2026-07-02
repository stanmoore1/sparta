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
#define MAXLINE 1024
#define EPSZERO 1.0e-14

// ========================================================================
// Add the new functions for SWS collisions: 
// collisions_one_SWS()
// collisions_group_SWS()
// collisions_one_ambipolar_SWS()
// collisions_group_ambipolar_SWS()
// ========================================================================
/* ----------------------------------------------------------------------
   NTC algorithm for a single group using Species Weighting Scheme
------------------------------------------------------------------------- */

template < int NEARCP > void Collide::collisions_one_SWS()
{
  int i,j,k,n,ip,np;
  int nattempt,reactflag;
  double attempt,volume;
  Particle::OnePart *ipart,*jpart,*kpart;
  double count_wi;   // SWS
  Particle::Species *species = particle->species;   // SWS
  int n_i,n_j,n_k,n_pre,i_loop;   // SWS
  double x[3],v[3];               // SWS

  // loop over cells I own

  Grid::ChildInfo *cinfo = grid->cinfo;

  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;

  for (int icell = 0; icell < nglocal; icell++) {
    np = cinfo[icell].count;

    count_wi = cinfo[icell].count_wi;     // SWS
    Ewilost = ewilost_cell[icell];                        // SWS
    double maxwi = 0.0;                   // SWS

    if (np <= 1) continue;

    if (NEARCP) {
      if (np > max_nn) realloc_nn(np,nn_last_partner);
      memset(nn_last_partner,0,np*sizeof(int));
    }

    ip = cinfo[icell].first;
    volume = cinfo[icell].volume / cinfo[icell].weight;
    if (volume == 0.0) error->one(FLERR,"Collision cell volume is zero");

    // setup particle list for this cell

    if (np > npmax) {
      while (np > npmax) npmax += DELTAPART;
      memory->destroy(plist);
      memory->create(plist,npmax,"collide:plist");
    }

    n = 0;
    while (ip >= 0) {
      maxwi = std::max(species[particles[ip].ispecies].specwt,maxwi);  // SWS
      plist[n++] = ip;
      ip = next[ip];
    }

    // attempt = exact collision attempt count for all particles in cell
    // nattempt = rounded attempt with RN
    // if no attempts, continue to next grid cell

    attempt = attempt_collision_SWS(icell,np,volume,count_wi,maxwi);   // SWS

    nattempt = static_cast<int> (attempt);

    if (!nattempt) continue;
    nattempt_one += nattempt;

    // perform collisions
    // select random pair of particles, cannot be same
    // test if collision actually occurs

    for (int iattempt = 0; iattempt < nattempt; iattempt++) {
      i = np * random->uniform();
      if (NEARCP) j = find_nn(i,np);
      else {
        j = np * random->uniform();
        while (i == j) j = np * random->uniform();
      }

      ipart = &particles[plist[i]];
      jpart = &particles[plist[j]];

      // test if collision actually occurs
      // continue to next collision if no reaction

      if (!test_collision_SWS(icell,0,0,ipart,jpart,maxwi)) continue;   // SWS

      if (NEARCP) {
        nn_last_partner[i] = j+1;
        nn_last_partner[j] = i+1;
      }

      // if recombination reaction is possible for this IJ pair
      // pick a 3rd particle to participate and set cell number density
      // unless boost factor turns it off, or there is no 3rd particle

      if (recombflag && recomb_ijflag[ipart->ispecies][jpart->ispecies]) {
        if (random->uniform() > react->recomb_boost_inverse)
          react->recomb_species = -1;
        else if (np <= 2)
          react->recomb_species = -1;
        else {
          k = np * random->uniform();
          while (k == i || k == j) k = np * random->uniform();
          react->recomb_part3 = &particles[plist[k]];
          react->recomb_species = react->recomb_part3->ispecies;
          react->recomb_density = count_wi * update->fnum / volume;    // SWS
        }
      }

      // perform collision and possible reaction

      setup_collision_SWS(ipart,jpart);   // SWS

      n_i = 1;                   // SWS
      n_j = n_k = n_pre = 0;     // SWS
      reactflag = perform_collision_SWS(ipart,jpart,kpart,n_i,n_j,n_k,n_pre);   // SWS
      
      ncollide_one++;
      if (reactflag) nreact_one++;
      else continue;

      // SWS - bookkeeping for particles created/destroyed by the reaction
      // perform_collision_SWS() appended new particles to the master
      // particle list in this order:
      //   p_pre (if n_pre) = un-reacted portion of the max-weight reactant
      //   kp    (if kpart) = 3rd product particle (dissociation/ionization)
      // n_i/n_j/n_k = # of copies of the I/J/K products to keep (0 = delete)

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
          if (NEARCP) set_nn(np);
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
        if (NEARCP) set_nn(np);
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
        if (NEARCP) nn_last_partner[i] = nn_last_partner[np];
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
        if (NEARCP) nn_last_partner[j] = nn_last_partner[np];
      }

      // copy paste ipart particle 
      if (ipart) {
        for (i_loop = 0; i_loop < n_i-1 ; i_loop++) {  
          //printf("!!check cp i \n");       
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
          if (NEARCP) set_nn(np);
          plist[np++] = particle->nlocal-1;
          particles = particle->particles;
        }
      }

      // copy paste jpart particle 
      if (jpart) {
        for (i_loop = 0; i_loop < n_j-1 ; i_loop++)  {
          //printf("!!check cp j \n");         
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
          if (NEARCP) set_nn(np);
          plist[np++] = particle->nlocal-1;
          particles = particle->particles;
        }
      }

      // copy paste kpart particle 
      if (kpart) {
        for (i_loop = 0; i_loop < n_k-1 ; i_loop++) {  
          //printf("!!check cp k \n");       
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
          if (NEARCP) set_nn(np);
          plist[np++] = particle->nlocal-1;
          particles = particle->particles;
        }
      }

      // exit attempt loop if less than 2 particles left in cell

      if (np < 2) break;
    }

    // SWS - store residual split-merge energy for this cell

    ewilost_cell[icell] = Ewilost;
  }
}

/* ----------------------------------------------------------------------
   NTC algorithm for multiple groups using Species Weighting Scheme,
   loop over pairs of groups pre-compute # of attempts per group pair
------------------------------------------------------------------------- */

template < int NEARCP > void Collide::collisions_group_SWS()
{
  double wi;        // SWS
  double count_wi;  // SWS
  double maxwi;     // SWS
  int i,j,k,n,ii,jj,ip,np,isp,ng;
  int pindex,ipair,igroup,jgroup,newgroup,ngmax;
  int nattempt,reactflag;
  int *ni,*nj,*ilist,*jlist;
  int *nn_igroup,*nn_jgroup;
  double attempt,volume;
  Particle::OnePart *ipart,*jpart,*kpart;
  int n_i,n_j,n_k,n_pre,i_loop;  // SWS

  // loop over cells I own

  Grid::ChildInfo *cinfo = grid->cinfo;

  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;
  int *species2group = mixture->species2group;

  for (int icell = 0; icell < nglocal; icell++) {
    count_wi = cinfo[icell].count_wi;   // SWS
    Ewilost = ewilost_cell[icell];                      // SWS
    np = cinfo[icell].count;
    if (np <= 1) continue;
    ip = cinfo[icell].first;
    volume = cinfo[icell].volume / cinfo[icell].weight;
    if (volume == 0.0) error->one(FLERR,"Collision cell volume is zero");

    // reallocate plist and p2g if necessary

    if (np > npmax) {
      while (np > npmax) npmax += DELTAPART;
      memory->destroy(plist);
      memory->create(plist,npmax,"collide:plist");
      memory->destroy(p2g);
      memory->create(p2g,npmax,2,"collide:p2g");
    }

    // plist = particle list for entire cell
    // glist[igroup][i] = index in plist of Ith particle in Igroup
    // ngroup[igroup] = particle count in Igroup
    // p2g[i][0] = Igroup for Ith particle in plist
    // p2g[i][1] = index within glist[igroup] of Ith particle in plist

    for (i = 0; i < ngroups; i++) {    // SWS
      ngroup[i] = 0;
      count_wi_group[i] = 0;
      maxwigr[i] = 0.0;
    }

    n = 0;

    while (ip >= 0) {
      isp = particles[ip].ispecies;
      igroup = species2group[isp];
      wi = particle->species[isp].specwt;    // SWS
      if (ngroup[igroup] == maxgroup[igroup]) {
        maxgroup[igroup] += DELTAPART;
        memory->grow(glist[igroup],maxgroup[igroup],"collide:glist");
      }
      ng = ngroup[igroup];
      glist[igroup][ng] = n;
      p2g[n][0] = igroup;
      p2g[n][1] = ng;
      plist[n] = ip;
      ngroup[igroup]++;
      n++;
      ip = next[ip];
      count_wi_group[igroup]+=wi;                    // SWS
      maxwigr[igroup]=std::max(wi,maxwigr[igroup]);  // SWS
    }

    if (NEARCP) {
      ngmax = 0;
      for (i = 0; i < ngroups; i++) ngmax = MAX(ngmax,ngroup[i]);
      if (ngmax > max_nn) {
        realloc_nn(ngmax,nn_last_partner_igroup);
        realloc_nn(ngmax,nn_last_partner_jgroup);
      }
    }

    // SWS - maxwi = max species weight over all particles in this cell
    // must NOT reuse the particle-pair loop index below

    maxwi = 0.0;
    for (int igr = 0; igr < ngroups; igr++)
      maxwi = std::max(maxwigr[igr],maxwi);

    // attempt = exact collision attempt count for a pair of groups
    // double loop over N^2 / 2 pairs of groups
    // nattempt = rounded attempt with RN
    // NOTE: not using RN for rounding of nattempt
    // gpair = list of group pairs when nattempt > 0

    npair = 0;
    for (igroup = 0; igroup < ngroups; igroup++)
      for (jgroup = igroup; jgroup < ngroups; jgroup++) {
        attempt = attempt_collision_SWS(icell,igroup,jgroup,volume);
        nattempt = static_cast<int> (attempt);

        if (nattempt) {
          gpair[npair][0] = igroup;
          gpair[npair][1] = jgroup;
          gpair[npair][2] = nattempt;
          nattempt_one += nattempt;
          npair++;
        }
      }

    // perform collisions for each pair of groups in gpair list
    // select random particle in each group
    // if igroup = jgroup, cannot be same particle
    // test if collision actually occurs
    // if chemistry occurs, move output I,J,K particles to new group lists
    // if chemistry occurs, exit attempt loop if group counts become too small
    // Ni and Nj are pointers to value in ngroup vector
    //   b/c need to stay current as chemistry occurs
    // NOTE: OK to use pre-computed nattempt when Ngroup may have changed via react?

    for (ipair = 0; ipair < npair; ipair++) {
      igroup = gpair[ipair][0];
      jgroup = gpair[ipair][1];
      nattempt = gpair[ipair][2];

      ni = &ngroup[igroup];
      nj = &ngroup[jgroup];
      ilist = glist[igroup];
      jlist = glist[jgroup];

      // re-test for no possible attempts
      // could have changed due to reactions in previous group pairs

      if (*ni == 0 || *nj == 0) continue;
      if (igroup == jgroup && *ni == 1) continue;

      if (NEARCP) {
        nn_igroup = nn_last_partner_igroup;
        if (igroup == jgroup) nn_jgroup = nn_last_partner_igroup;
        else nn_jgroup = nn_last_partner_jgroup;
        memset(nn_igroup,0,(*ni)*sizeof(int));
        if (igroup != jgroup) memset(nn_jgroup,0,(*nj)*sizeof(int));
      }

      for (int iattempt = 0; iattempt < nattempt; iattempt++) {
	    i = *ni * random->uniform();
            if (NEARCP) j = find_nn_group(i,ilist,*nj,jlist,plist,nn_igroup,nn_jgroup);
            else {
              j = *nj * random->uniform();
              if (igroup == jgroup)
                while (i == j) j = *nj * random->uniform();
            }
      
	    ipart = &particles[plist[ilist[i]]];
	    jpart = &particles[plist[jlist[j]]];

        // test if collision actually occurs
        // continue to next collision if no reaction

        if (!test_collision_SWS(icell,igroup,jgroup,ipart,jpart,maxwi)) continue;  // SWS

        if (NEARCP) {
          nn_igroup[i] = j+1;
          nn_jgroup[j] = i+1;
        }

        // if recombination reaction is possible for this IJ pair
        // pick a 3rd particle to participate and set cell number density
        // unless boost factor turns it off, or there is no 3rd particle

        if (recombflag && recomb_ijflag[ipart->ispecies][jpart->ispecies]) {
          if (random->uniform() > react->recomb_boost_inverse)
            react->recomb_species = -1;
          else if (np <= 2)
            react->recomb_species = -1;
          else {
            ii = ilist[i];
            jj = jlist[j];
            k = np * random->uniform();
            while (k == ii || k == jj) k = np * random->uniform();
            react->recomb_part3 = &particles[plist[k]];
            react->recomb_species = react->recomb_part3->ispecies;
            react->recomb_density = count_wi * update->fnum / volume;  // SWS
          }
        }

        // perform collision and possible reaction

        setup_collision_SWS(ipart,jpart);  // SWS
        reactflag = perform_collision_SWS(ipart,jpart,kpart,n_i,n_j,n_k,n_pre);  // SWS
        ncollide_one++;
        if (reactflag) nreact_one++;
        else continue;

        // ipart may now be in different group
        // reset ilist,jlist after addgroup() in case it realloced glist

        newgroup = species2group[ipart->ispecies];
        if (newgroup != igroup) {
          addgroup(newgroup,ilist[i]);
          delgroup(igroup,i);
          ilist = glist[igroup];
          jlist = glist[jgroup];
          // this line needed if jgroup=igroup and delgroup() moved J particle
          if (jgroup == igroup && j == *ni) j = i;
        }

        // jpart may now be in different group or destroyed
        // if new group: reset ilist,jlist after addgroup() in case it realloced glist
        // if destroyed: delete from plist and group, add particle to deletion list

        if (jpart) {
          newgroup = species2group[jpart->ispecies];
          if (newgroup != jgroup) {
            addgroup(newgroup,jlist[j]);
            delgroup(jgroup,j);
            ilist = glist[igroup];
            jlist = glist[jgroup];
          }

        } else {
          if (ndelete == maxdelete) {
            maxdelete += DELTADELETE;
            memory->grow(dellist,maxdelete,"collide:dellist");
          }
          pindex = jlist[j];
          dellist[ndelete++] = plist[pindex];

          delgroup(jgroup,j);

          plist[pindex] = plist[np-1];
          p2g[pindex][0] = p2g[np-1][0];
          p2g[pindex][1] = p2g[np-1][1];
          if (pindex < np-1) glist[p2g[pindex][0]][p2g[pindex][1]] = pindex;
          np--;

          if (NEARCP) nn_jgroup[j] = nn_jgroup[*nj];
        }

        // if kpart created, add to plist and group list
        // kpart was just added to particle list, so index = nlocal-1
        // reset ilist,jlist after addgroup() in case it realloced
        // particles data struct may also have been realloced

        if (kpart) {
          newgroup = species2group[kpart->ispecies];

          if (NEARCP) {
            if (newgroup == igroup || newgroup == jgroup) {
              n = ngroup[newgroup];
              set_nn_group(n);
              nn_igroup = nn_last_partner_igroup;
              if (igroup == jgroup) nn_jgroup = nn_last_partner_igroup;
              else nn_jgroup = nn_last_partner_jgroup;
              nn_igroup[n] = 0;
              nn_jgroup[n] = 0;
            }
          }

          if (np == npmax) {
            npmax += DELTAPART;
            memory->grow(plist,npmax,"collide:plist");
            memory->grow(p2g,npmax,2,"collide:p2g");
          }
          plist[np++] = particle->nlocal-1;

          addgroup(newgroup,np-1);
          ilist = glist[igroup];
          jlist = glist[jgroup];
          particles = particle->particles;
        }

        // test to exit attempt loop due to groups becoming too small

        if (*ni <= 1) {
          if (*ni == 0) break;
          if (igroup == jgroup) break;
        }
        if (*nj <= 1) {
          if (*nj == 0) break;
          if (igroup == jgroup) break;
        }
      }
    }

    // SWS - store residual split-merge energy for this cell

    ewilost_cell[icell] = Ewilost;
  }
}

/* ----------------------------------------------------------------------
   NTC algorithm for a single group with ambipolar approximation
   using Species Weighting Scheme
------------------------------------------------------------------------- */

void Collide::collisions_one_ambipolar_SWS()
{
  int i,j,k,n,ip,np,nelectron,nptotal,jspecies,tmp;
  int nattempt,reactflag;
  double attempt,volume;

  int n_i,n_j,n_k,n_pre,i_loop;  // SWS
  double x[3],v[3];              // SWS
  int np_pre;                    // SWS

  double count_wi;           // SWS
  double count_wi_electron;  // SWS

  Particle::Species *species = particle->species;
  Particle::OnePart *ipart,*jpart,*kpart,*p,*ep;

  // ambipolar vectors

  int *ionambi = particle->eivec[particle->ewhich[index_ionambi]];
  double **velambi = particle->edarray[particle->ewhich[index_velambi]];

  // loop over cells I own

  Grid::ChildInfo *cinfo = grid->cinfo;

  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;
  int nbytes = sizeof(Particle::OnePart);

  for (int icell = 0; icell < nglocal; icell++) {
    count_wi = cinfo[icell].count_wi;   // SWS
    Ewilost = ewilost_cell[icell];                      // SWS
    double maxwi = 0.0;                 // SWS
    np = cinfo[icell].count;
    if (np <= 1) continue;
    ip = cinfo[icell].first;
    volume = cinfo[icell].volume / cinfo[icell].weight;
    if (volume == 0.0) error->one(FLERR,"Collision cell volume is zero");

    // setup particle list for this cell

    if (np > npmax) {
      while (np > npmax) npmax += DELTAPART;
      memory->destroy(plist);
      memory->create(plist,npmax,"collide:plist");
    }

    n = 0;
    while (ip >= 0) {
      maxwi = std::max(species[particles[ip].ispecies].specwt,maxwi);  // SWS
      plist[n++] = ip;
      ip = next[ip];
    }

    // setup elist of ionized electrons for this cell
    // create them in separate array since will never become real particles

    if (np >= maxelectron) {
      while (maxelectron < np) maxelectron += DELTAELECTRON;
      memory->sfree(elist);
      elist = (Particle::OnePart *)
        memory->smalloc(maxelectron*nbytes,"collide:elist");
    }

    // create electrons for ambipolar ions

    nelectron = 0;

    count_wi_electron = 0.0;    // SWS
    Particle::Species *species = particle->species;    // SWS

    for (i = 0; i < np; i++) {
      if (ionambi[plist[i]]) {
        p = &particles[plist[i]];
        ep = &elist[nelectron];
        memcpy(ep,p,nbytes);
        memcpy(ep->v,velambi[plist[i]],3*sizeof(double));
        ep->ispecies = ambispecies;
        count_wi_electron += species[ep->ispecies].specwt;  // SWS
        nelectron++;
      }
    }

    // attempt = exact collision attempt count for all particles in cell
    // nptotal = includes neutrals, ions, electrons
    // nattempt = rounded attempt with RN

    nptotal = np + nelectron;
    double count_wi_total;     // SWS
    count_wi_total = count_wi + count_wi_electron;    // SWS
    attempt = attempt_collision_SWS(icell,nptotal,volume,count_wi_total,maxwi);    // SWS
    nattempt = static_cast<int> (attempt);

    if (!nattempt) continue;
    nattempt_one += nattempt;

    // perform collisions
    // select random pair of particles, cannot be same
    // test if collision actually occurs
    // if chemistry occurs, exit attempt loop if group count goes to 0

    for (int iattempt = 0; iattempt < nattempt; iattempt++) {
      i = nptotal * random->uniform();
      j = nptotal * random->uniform();
      while (i == j) j = nptotal * random->uniform();

      // ipart,jpart = heavy particles or electrons

      if (i < np) ipart = &particles[plist[i]];
      else ipart = &elist[i-np];
      if (j < np) jpart = &particles[plist[j]];
      else jpart = &elist[j-np];

      // check for e/e pair
      // count as collision, but do not perform it

      if (ipart->ispecies == ambispecies && jpart->ispecies == ambispecies) {
        ncollide_one++;
        continue;
      }

      // if particle I is electron
      // swap with J, since electron must be 2nd in any ambipolar reaction
      // just need to swap i/j, ipart/jpart
      // don't have to worry if an ambipolar ion is I or J

      if (ipart->ispecies == ambispecies) {
        tmp = i;
        i = j;
        j = tmp;
        p = ipart;
        ipart = jpart;
        jpart = p;
      }

      // test if collision actually occurs
      if (!test_collision_SWS(icell,0,0,ipart,jpart,maxwi)) continue;   // SWS

      // if recombination reaction is possible for this IJ pair
      // pick a 3rd particle to participate and set cell number density
      // unless boost factor turns it off, or there is no 3rd particle
      // 3rd particle cannot be an electron, so select from Np

      if (recombflag && recomb_ijflag[ipart->ispecies][jpart->ispecies]) {
        if (random->uniform() > react->recomb_boost_inverse)
          react->recomb_species = -1;
        else if (np == 1)
          react->recomb_species = -1;
        else if (np == 2 && jpart->ispecies != ambispecies)
          react->recomb_species = -1;
        else {
          k = np * random->uniform();
          while (k == i || k == j) k = np * random->uniform();
          react->recomb_part3 = &particles[plist[k]];
          react->recomb_species = react->recomb_part3->ispecies;
          react->recomb_density = count_wi * update->fnum / volume;  // SWS
        }
      }

      // perform collision
      // ijspecies = species before collision chemistry
      // continue to next collision if no reaction

      jspecies = jpart->ispecies;
      setup_collision_SWS(ipart,jpart);  // SWS
      // ========================================================================
      // SWS - from here to the end of the loop, changes are made to take into 
      // account reactions in the event of a collision.
      // n_i variable is the number of particles after the reaction.
      // ipart is the particle pointer before and after the reaction, like the baseline.
      // define the number of particles generated with perform collison, and 
      // then, add particle to list in this function.
      // However, note that the main particles and k particles when there is a reaction 
      // are added within perform_collide.
      // ========================================================================
      // parameters to count particle after the reaction      
      n_i = 1; 
      n_j = n_k = n_pre = 0;      
      // save number of particle before collision
      // to point electron in elist   
      np_pre = np;    
      reactflag = perform_collision_SWS(ipart,jpart,kpart,n_i,n_j,n_k,n_pre);

      ncollide_one++;
      if (reactflag) nreact_one++;
      else continue;

      // reset ambipolar ion flags due to collision
      // must do now before particle count reset below can break out of loop
      // first reset ionambi if kpart was added since ambi_reset() uses it

      if (kpart) ionambi = particle->eivec[particle->ewhich[index_ionambi]];
      if (jspecies == ambispecies)
        ambi_reset(plist[i],-1,jspecies,ipart,jpart,kpart,ionambi);
      else
        ambi_reset(plist[i],plist[j],jspecies,ipart,jpart,kpart,ionambi);
      
      //==================================================================
      // add particles witch added in perform_collide
      // if kpart created:
      // particles and custom data structs may have been realloced by kpart
      // add kpart to plist or elist
      // kpart was just added to particle list, so index = nlocal-1
      // must come before jpart code below since it modifies nlocal
      //==================================================================

      // SWS - indices in the master particle list of particles appended by
      // perform_collision_SWS(): p_pre first (if n_pre), then kp (if kpart)

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
            k = kp_index;
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
        // j indexes plist only when j < np_pre (else it indexes elist)
        if (j < np_pre && j == np) j = i;
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
            
            // np can be changed by other reactions, np_pre is used
            if (nelectron-1 != j-np_pre) memcpy(&elist[j-np_pre],&elist[nelectron-1],nbytes);
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
        if (nelectron-1 != j-np_pre) memcpy(&elist[j-np_pre],&elist[nelectron-1],nbytes);
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
          ionambi[particle->nlocal-1] = ionambi[k];
        }
      }

      // update particle counts
      // quit if no longer enough particles for another collision

      nptotal = np + nelectron;
      if (nptotal < 2) break;
    }

    // done with collisions/chemistry for one grid cell
    // recombine ambipolar ions with their matching electrons
    //   by copying electron velocity into velambi
    // which ion is combined with which electron does not matter
    // error if ion count does not match electron count

    int melectron = 0;
    for (n = 0; n < np; n++) {
      i = plist[n];
      if (ionambi[i]) {
        if (melectron < nelectron) {
          ep = &elist[melectron];
          memcpy(velambi[i],ep->v,3*sizeof(double));
        }
        melectron++;
      }
    }
    if (melectron != nelectron) {  // SWS
      error->one(FLERR,"Collisions in cell did not conserve electron count now **Currently only equal weight electrons and ions are supported.");
    }    

    // SWS - store residual split-merge energy for this cell

    ewilost_cell[icell] = Ewilost;
  }
}

/* ----------------------------------------------------------------------
   NTC algorithm for multiple groups with ambipolar approximation
   loop over pairs of groups, pre-compute # of attempts per group pair
   using Species Weighting Scheme
------------------------------------------------------------------------- */

void Collide::collisions_group_ambipolar_SWS()
{
  double wi;  // SWS
  double count_wi;  // SWS
  double maxwi;  // SWS
  int i,j,k,n,ii,jj,ip,np,isp,ng;
  int pindex,ipair,igroup,jgroup,newgroup,jspecies,tmp;
  int nattempt,reactflag,nelectron;
  int *ni,*nj,*ilist,*jlist,*tmpvec;
  double attempt,volume;
  Particle::OnePart *ipart,*jpart,*kpart,*p,*ep;
  int n_i,n_j,n_k,n_pre,i_loop;   // SWS

  // ambipolar vectors

  int *ionambi = particle->eivec[particle->ewhich[index_ionambi]];
  double **velambi = particle->edarray[particle->ewhich[index_velambi]];

  // loop over cells I own

  Grid::ChildInfo *cinfo = grid->cinfo;

  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;
  int nbytes = sizeof(Particle::OnePart);
  int *species2group = mixture->species2group;
  int egroup = species2group[ambispecies];

  for (int icell = 0; icell < nglocal; icell++) {
    count_wi = cinfo[icell].count_wi;   // SWS
    Ewilost = ewilost_cell[icell];   // SWS
    np = cinfo[icell].count;
    if (np <= 1) continue;
    ip = cinfo[icell].first;
    volume = cinfo[icell].volume / cinfo[icell].weight;
    if (volume == 0.0) error->one(FLERR,"Collision cell volume is zero");

    // reallocate plist and p2g if necessary

    if (np > npmax) {
      while (np > npmax) npmax += DELTAPART;
      memory->destroy(plist);
      memory->create(plist,npmax,"collide:plist");
      memory->destroy(p2g);
      memory->create(p2g,npmax,2,"collide:p2g");
    }

    // setup elist of ionized electrons for this cell
    // create them in separate array since will never become real particles

    if (np >= maxelectron) {
      while (maxelectron < np) maxelectron += DELTAELECTRON;
      memory->sfree(elist);
      elist = (Particle::OnePart *)
        memory->smalloc(maxelectron*nbytes,"collide:elist");
    }

    // plist = particle list for entire cell
    // glist[igroup][i] = index in plist of Ith particle in Igroup
    // ngroup[igroup] = particle count in Igroup
    // p2g[i][0] = Igroup for Ith particle in plist
    // p2g[i][1] = index within glist[igroup] of Ith particle in plist
    // also populate elist with ionized electrons, now separated from ions
    // ngroup[egroup] = nelectron

    for (i = 0; i < ngroups; i++) {   // SWS
      ngroup[i] = 0;
      count_wi_group[i] = 0;
      maxwigr[i] = 0.0;
    }

    n = 0;
    nelectron = 0;

    while (ip >= 0) {
      isp = particles[ip].ispecies;
      igroup = species2group[isp];
      wi = particle->species[isp].specwt;  // SWS
      if (ngroup[igroup] == maxgroup[igroup]) {
        maxgroup[igroup] += DELTAPART;
        memory->grow(glist[igroup],maxgroup[igroup],"collide:glist");
      }
      ng = ngroup[igroup];
      glist[igroup][ng] = n;
      p2g[n][0] = igroup;
      p2g[n][1] = ng;
      plist[n] = ip;
      ngroup[igroup]++;
      count_wi_group[igroup]+=wi;  // SWS
      maxwigr[igroup]=std::max(wi,maxwigr[igroup]);  // SWS

      if (ionambi[ip]) {
        p = &particles[ip];
        ep = &elist[nelectron];
        memcpy(ep,p,nbytes);
        memcpy(ep->v,velambi[ip],3*sizeof(double));
        ep->ispecies = ambispecies;
        nelectron++;

        if (ngroup[egroup] == maxgroup[egroup]) {
          maxgroup[egroup] += DELTAPART;
          memory->grow(glist[egroup],maxgroup[egroup],"collide:grouplist");
        }
        ng = ngroup[egroup];
        glist[egroup][ng] = nelectron-1;
        ngroup[egroup]++;
      }

      n++;
      ip = next[ip];
    }

    // SWS - maxwi = max species weight over all particles in this cell
    // must NOT reuse the particle-pair loop index below

    maxwi = 0.0;
    for (int igr = 0; igr < ngroups; igr++)
      maxwi = std::max(maxwigr[igr],maxwi);

    // attempt = exact collision attempt count for a pair of groups
    // double loop over N^2 / 2 pairs of groups
    // temporarily include nelectrons in count for egroup
    // nattempt = rounded attempt with RN
    // NOTE: not using RN for rounding of nattempt
    // gpair = list of group pairs when nattempt > 0
    //         flip igroup/jgroup if igroup = egroup
    // egroup/egroup collisions are not included in gpair

    npair = 0;
    for (igroup = 0; igroup < ngroups; igroup++)
      for (jgroup = igroup; jgroup < ngroups; jgroup++) {
        if (igroup == egroup && jgroup == egroup) continue;
        attempt = attempt_collision_SWS(icell,igroup,jgroup,volume);
        nattempt = static_cast<int> (attempt);

        if (nattempt) {
          if (igroup == egroup) {
              gpair[npair][0] = jgroup;
              gpair[npair][1] = igroup;
            } else {
              gpair[npair][0] = igroup;
              gpair[npair][1] = jgroup;
            }
          gpair[npair][2] = nattempt;
          nattempt_one += nattempt;
          npair++;
        }
      }

    // perform collisions for each pair of groups in gpair list
    // select random particle in each group
    // if igroup = jgroup, cannot be same particle
    // test if collision actually occurs
    // if chemistry occurs, move output I,J,K particles to new group lists
    // if chemistry occurs, exit attempt loop if group counts become too small
    // Ni and Nj are pointers to value in ngroup vector
    //   b/c need to stay current as chemistry occurs
    // NOTE: OK to use pre-computed nattempt when Ngroup may have changed via react?

    for (ipair = 0; ipair < npair; ipair++) {
      igroup = gpair[ipair][0];
      jgroup = gpair[ipair][1];
      nattempt = gpair[ipair][2];

      ni = &ngroup[igroup];
      nj = &ngroup[jgroup];
      ilist = glist[igroup];
      jlist = glist[jgroup];

      // re-test for no possible attempts
      // could have changed due to reactions in previous group pairs

      if (*ni == 0 || *nj == 0) continue;
      if (igroup == jgroup && *ni == 1) continue;

      for (int iattempt = 0; iattempt < nattempt; iattempt++) {
	      i = *ni * random->uniform();
        j = *nj * random->uniform();
        if (igroup == jgroup)
          while (i == j) j = *nj * random->uniform();

	// ipart/jpart can be from particles or elist

	 if (igroup == egroup) ipart = &elist[i];
	 else ipart = &particles[plist[ilist[i]]];
	 if (jgroup == egroup) jpart = &elist[j];
	 else jpart = &particles[plist[jlist[j]]];

        // NOTE: unlike single group, no possibility of e/e collision
        //       means collision stats may be different

        //if (ipart->ispecies == ambispecies && jpart->ispecies == ambispecies) {
        //  ncollide_one++;
        //  continue;
        //}

        // test if collision actually occurs

        if (!test_collision_SWS(icell,igroup,jgroup,ipart,jpart,maxwi)) continue;    // SWS

        // if recombination reaction is possible for this IJ pair
        // pick a 3rd particle to participate and set cell number density
        // unless boost factor turns it off, or there is no 3rd particle
        // 3rd particle will never be an electron since plist has no electrons
        // if jgroup == egroup, no need to check k for match to jj

        if (recombflag && recomb_ijflag[ipart->ispecies][jpart->ispecies]) {
          if (random->uniform() > react->recomb_boost_inverse)
            react->recomb_species = -1;
          else if (np <= 2)
            react->recomb_species = -1;
          else {
            ii = ilist[i];
            if (jgroup == egroup) jj = -1;
            else jj = jlist[j];
            k = np * random->uniform();
            while (k == ii || k == jj) k = np * random->uniform();
            react->recomb_part3 = &particles[plist[k]];
            react->recomb_species = react->recomb_part3->ispecies;
            react->recomb_density = count_wi * update->fnum / volume;    // SWS
          }
        }

        // perform collision
        // ijspecies = species before collision chemistry
        // continue to next collision if no reaction

        jspecies = jpart->ispecies;
        setup_collision_SWS(ipart,jpart);  // SWS
        reactflag = perform_collision_SWS(ipart,jpart,kpart,n_i,n_j,n_k,n_pre);  // SWS
        ncollide_one++;
        if (reactflag) nreact_one++;
        else continue;

        // reset ambipolar ion flags due to reaction
        // must do now before group reset below can break out of loop
        // first reset ionambi if kpart was added since ambi_reset() uses it

        if (kpart) ionambi = particle->eivec[particle->ewhich[index_ionambi]];
        if (jgroup == egroup)
          ambi_reset(plist[ilist[i]],-1,jspecies,ipart,jpart,kpart,ionambi);
        else
          ambi_reset(plist[ilist[i]],plist[jlist[j]],jspecies,
                     ipart,jpart,kpart,ionambi);

        // ipart may now be in different group
        // reset ilist,jlist after addgroup() in case it realloced glist

        newgroup = species2group[ipart->ispecies];
        if (newgroup != igroup) {
          addgroup(newgroup,ilist[i]);
          delgroup(igroup,i);
          ilist = glist[igroup];
          jlist = glist[jgroup];
          // this line needed if jgroup=igroup and delgroup() moved J particle
          if (jlist == ilist && j == *ni) j = i;
        }

        // if kpart created:
        // particles and custom data structs may have been realloced by kpart
        // add kpart to plist or elist and to group
        // kpart was just added to particle list, so index = nlocal-1
        // must come before jpart code below since it modifies nlocal

        if (kpart) {
          particles = particle->particles;
          ionambi = particle->eivec[particle->ewhich[index_ionambi]];
          velambi = particle->edarray[particle->ewhich[index_velambi]];

          newgroup = species2group[kpart->ispecies];

          if (newgroup != egroup) {
            if (np == npmax) {
              npmax += DELTAPART;
              memory->grow(plist,npmax,"collide:plist");
              memory->grow(p2g,npmax,2,"collide:p2g");
            }
            plist[np++] = particle->nlocal-1;
            addgroup(newgroup,np-1);
            ilist = glist[igroup];
            jlist = glist[jgroup];

          } else {
            if (nelectron == maxelectron) {
              maxelectron += DELTAELECTRON;
              elist = (Particle::OnePart *)
                memory->srealloc(elist,maxelectron*nbytes,"collide:elist");
            }
            ep = &elist[nelectron];
            memcpy(ep,kpart,nbytes);
            ep->ispecies = ambispecies;
            nelectron++;
            particle->nlocal--;

            if (ngroup[egroup] == maxgroup[egroup]) {
              maxgroup[egroup] += DELTAPART;
              memory->grow(glist[egroup],maxgroup[egroup],"collide:grouplist");
            }
            ng = ngroup[egroup];
            glist[egroup][ng] = nelectron-1;
            ngroup[egroup]++;
          }
        }

        // jpart may now be in a different group or destroyed
        // if jpart exists, now in a different group, neither group is egroup:
        //   add/del group, reset ilist,jlist after addgroup() in case glist realloced
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
        //   remove from plist and group, add particle to deletion list

        if (jpart) {
          newgroup = species2group[jpart->ispecies];

          if (newgroup == jgroup) {
            // nothing to do

          } else if (jgroup != egroup && newgroup != egroup) {
            addgroup(newgroup,jlist[j]);
            delgroup(jgroup,j);
            ilist = glist[igroup];
            jlist = glist[jgroup];

          } else if (jgroup != egroup && jpart->ispecies == ambispecies) {
            if (nelectron == maxelectron) {
              maxelectron += DELTAELECTRON;
              elist = (Particle::OnePart *)
                memory->srealloc(elist,maxelectron*nbytes,"collide:elist");
            }
            ep = &elist[nelectron];
            memcpy(ep,jpart,nbytes);
            ep->ispecies = ambispecies;
            nelectron++;

            if (ngroup[egroup] == maxgroup[egroup]) {
              maxgroup[egroup] += DELTAPART;
              memory->grow(glist[egroup],maxgroup[egroup],"collide:grouplist");
            }
            ng = ngroup[egroup];
            glist[egroup][ng] = nelectron-1;
            ngroup[egroup]++;

            jpart = NULL;

          } else if (jgroup == egroup && jpart->ispecies != ambispecies) {
            int reallocflag = particle->add_particle();
            if (reallocflag) {
              particles = particle->particles;
              ionambi = particle->eivec[particle->ewhich[index_ionambi]];
              velambi = particle->edarray[particle->ewhich[index_velambi]];
            }

            int index = particle->nlocal-1;
            memcpy(&particles[index],jpart,nbytes);
            particles[index].id = MAXSMALLINT*random->uniform();
            ionambi[index] = 0;

            if (nelectron-1 != j) memcpy(&elist[j],&elist[nelectron-1],nbytes);
            nelectron--;
            ngroup[egroup]--;

            if (np == npmax) {
              npmax += DELTAPART;
              memory->grow(plist,npmax,"collide:plist");
              memory->grow(p2g,npmax,2,"collide:p2g");
            }
            plist[np++] = index;
            addgroup(newgroup,np-1);
            ilist = glist[igroup];
            jlist = glist[jgroup];
          }
        }

        if (!jpart && jspecies == ambispecies) {
          if (nelectron-1 != j) memcpy(&elist[j],&elist[nelectron-1],nbytes);
          nelectron--;
          ngroup[egroup]--;

        } else if (!jpart) {
          if (ndelete == maxdelete) {
            maxdelete += DELTADELETE;
            memory->grow(dellist,maxdelete,"collide:dellist");
          }
          pindex = jlist[j];
          dellist[ndelete++] = plist[pindex];

          delgroup(jgroup,j);

          plist[pindex] = plist[np-1];
          p2g[pindex][0] = p2g[np-1][0];
          p2g[pindex][1] = p2g[np-1][1];
          if (pindex < np-1) glist[p2g[pindex][0]][p2g[pindex][1]] = pindex;
          np--;
        }

        // test to exit attempt loop due to groups becoming too small

        if (*ni <= 1) {
          if (*ni == 0) break;
          if (igroup == jgroup) break;
        }
        if (*nj <= 1) {
          if (*nj == 0) break;
          if (igroup == jgroup) break;
        }
      }
    }

    // done with collisions/chemistry for one grid cell
    // recombine ambipolar ions with their matching electrons
    //   by copying electron velocity into velambi
    // which ion is combined with which electron does not matter
    // error if do not use all nelectrons in cell

    int melectron = 0;
    for (n = 0; n < np; n++) {
      i = plist[n];
      if (ionambi[i]) {
        if (melectron < nelectron) {
          ep = &elist[melectron];
          memcpy(velambi[i],ep->v,3*sizeof(double));
        }
        melectron++;
      }
    }
    if (melectron != nelectron)
      error->one(FLERR,"Collisions in cell did not conserve electron count");

    // SWS - store residual split-merge energy for this cell

    ewilost_cell[icell] = Ewilost;
  }
}

/* ---------------------------------------------------------------------- */
// ========================================================================
// Modify the number of attempted collision to account for the species
// weight, leading to an artificial increase of the number of 
// trace species numerical particles. Thus, in below equations,
// N from the conventional DSMC is lower than N from SWS methods. 
// Conventional DSMC (sws=0): Ncoll = 1/2 N fnum (N-1)
// SWS (sws=1): Ncoll = 1/2 count_wi fnum (N-1)
// SWSmax (sws=2): Ncoll = 1/2 N fnum wi_max (N-1)
// ========================================================================
double CollideVSS::attempt_collision_SWS(int icell, int np, double volume, double count_wi, double maxwi)
{
  double fnum = update->fnum;
  double dt = update->dt;

  double nattempt;

  int sws = particle->sws;  // SWS
  if (sws==1) {             // SWS
    if (remainflag) {
      nattempt = 0.5 * count_wi * (np-1) *
        vremax[icell][0][0] * dt * fnum / volume + remain[icell][0][0];
      remain[icell][0][0] = nattempt - static_cast<int> (nattempt);
    } else {
      nattempt = 0.5 * count_wi * (np-1) *
        vremax[icell][0][0] * dt * fnum / volume + random->uniform();
    }
  } else if (sws==2) {      // SWS
    if (remainflag) {
      nattempt = 0.5 * np * maxwi * (np-1) *
        vremax[icell][0][0] * dt * fnum / volume + remain[icell][0][0];
      remain[icell][0][0] = nattempt - static_cast<int> (nattempt);
    } else {
      nattempt = 0.5 * np * maxwi * (np-1) *
        vremax[icell][0][0] * dt * fnum / volume + random->uniform();
    }
  } else {
    if (remainflag) {
      nattempt = 0.5 * np * (np-1) *
        vremax[icell][0][0] * dt * fnum / volume + remain[icell][0][0];
      remain[icell][0][0] = nattempt - static_cast<int> (nattempt);
    } else {
      nattempt = 0.5 * np * (np-1) *
        vremax[icell][0][0] * dt * fnum / volume + random->uniform();
    }
  }
  return nattempt;
}

/* ---------------------------------------------------------------------- */
// ========================================================================
// SWS keyword: use n = count_wi * fnum instead of n = np * fnum 
// to obtain the correct number of attempted collisions when using 
// different species weight.
// SWSmax keyword: use n = np * fnum * max(wi)
// ========================================================================
double CollideVSS::attempt_collision_SWS(int icell, int igroup, int jgroup,
                                     double volume)
{
 double fnum = update->fnum;
 double dt = update->dt;

 double nattempt;

 // return 2x the value for igroup != jgroup, since no J,I pairing

 double npairs;

 int sws = particle->sws;  // SWS
 if (igroup == jgroup) {   
  if (sws==1) npairs = 0.5 * count_wi_group[igroup] * (ngroup[igroup]-1);  // SWS
  else if (sws==2) npairs = 0.5 * ngroup[igroup] * maxwigr[igroup] * (ngroup[igroup]-1);  // SWS
  else npairs = 0.5 * ngroup[igroup] * (ngroup[igroup]-1);
 }
 else {
  if (sws==1) npairs = count_wi_group[igroup] * (ngroup[jgroup]);  // SWS
  else if (sws==2) npairs = ngroup[igroup] * maxwigr[igroup] * (ngroup[jgroup]);  // SWS
  else npairs = ngroup[igroup] * (ngroup[jgroup]);
 }

 nattempt = npairs * vremax[icell][igroup][jgroup] * dt * fnum / volume;

 if (remainflag) {
   nattempt += remain[icell][igroup][jgroup];
   remain[icell][igroup][jgroup] = nattempt - static_cast<int> (nattempt);
 } else nattempt += random->uniform();

 return nattempt;
}

/* ---------------------------------------------------------------------- */
// ========================================================================
// Modify the acceptance-rejection method so that particle pair
// candidates are less likely to be selected if both particles
// have low weights. 
// This modification goes along with: Ncoll = 1/2 N fnum w_imax (N-1),
// N being increased by SWS.
// This method provides the best accuracy for physical particles species 
// collision rates with the cost of a slight increase in the total number of 
// attempted collision.
// Keyword to use this method: SWSmax
// ========================================================================
int CollideVSS::test_collision_SWS(int icell, int igroup, int jgroup,
  Particle::OnePart *ip, Particle::OnePart *jp, double maxwi)
{
double *vi = ip->v;
double *vj = jp->v;
int ispecies = ip->ispecies;
int jspecies = jp->ispecies;
double du  = vi[0] - vj[0];
double dv  = vi[1] - vj[1];
double dw  = vi[2] - vj[2];
double vr2 = du*du + dv*dv + dw*dw;

// prevent division by zero

if (vr2 < EPSZERO && params[ispecies][jspecies].omega >= 1.0)
  return 0;

double vro  = pow(vr2,1.0-params[ispecies][jspecies].omega);

// although the vremax is calculated for the group,
// the individual collisions calculated species dependent vre

double vre = vro*prefactor[ispecies][jspecies];
vremax[icell][igroup][jgroup] = MAX(vre,vremax[icell][igroup][jgroup]);

int sws = particle->sws;  // SWS
if (sws==2) {             // SWS
  Particle::Species *species = particle->species;   // SWS
  double w_ipart = species[ispecies].specwt;   // SWS
  double w_jpart = species[jspecies].specwt;   // SWS
  if ((vre/vremax[icell][igroup][jgroup])*(MAX(w_ipart,w_jpart)/maxwi) < random->uniform()) return 0;  // SWS
} else {
  if (vre/vremax[icell][igroup][jgroup] < random->uniform()) return 0;
}
precoln.vr2 = vr2;
return 1;
}

/* ---------------------------------------------------------------------- */
// ========================================================================
// Set the weights according to the colliding particle species and 
// determine the maximum weight over all the simulated particles.
// Add the stored energy lost due to differently weighted
// collision to the translation energy of the current collision if
// the colliding particles are major species.
// ========================================================================
void CollideVSS::setup_collision_SWS(Particle::OnePart *ip, Particle::OnePart *jp)
{
  Particle::Species *species = particle->species;

  int isp = ip->ispecies;
  int jsp = jp->ispecies;

  double w_i = species[isp].specwt;  // SWS
  double w_j = species[jsp].specwt;  // SWS
  int nspecies = particle->nspecies; // SWS
  double w_max = 0.0;                // SWS

  for (int i = 0; i < nspecies; i++){  // SWS
    w_max = std::max(species[i].specwt,w_max);
  }

  precoln.vr = sqrt(precoln.vr2);

  precoln.ave_rotdof = 0.5 * (species[isp].rotdof + species[jsp].rotdof);
  precoln.ave_vibdof = 0.5 * (species[isp].vibdof + species[jsp].vibdof);
  precoln.ave_dof = (precoln.ave_rotdof  + precoln.ave_vibdof)/2.;

  double imass = precoln.imass = species[isp].mass;
  double jmass = precoln.jmass = species[jsp].mass;

   if ((w_i==w_max) && (w_j==w_max)){  // SWS
    precoln.etrans = 0.5 * params[isp][jsp].mr * precoln.vr2 + Ewilost;
    Ewilost = 0.0;
  } else {
    precoln.etrans = 0.5 * params[isp][jsp].mr * precoln.vr2;
  }
  
  precoln.erot = ip->erot + jp->erot;
  precoln.evib = ip->evib + jp->evib;

  precoln.eint   = precoln.erot + precoln.evib;
  precoln.etotal = precoln.etrans + precoln.eint;

  // COM velocity calculated using reactant masses

  double divisor = 1.0 / (imass+jmass);
  double *vi = ip->v;
  double *vj = jp->v;
  precoln.ucmf = ((imass*vi[0])+(jmass*vj[0])) * divisor;
  precoln.vcmf = ((imass*vi[1])+(jmass*vj[1])) * divisor;
  precoln.wcmf = ((imass*vi[2])+(jmass*vj[2])) * divisor;

  postcoln.etrans = precoln.etrans;
  postcoln.erot = 0.0;
  postcoln.evib = 0.0;
  postcoln.eint = 0.0;
  postcoln.etotal = precoln.etotal;
}

int CollideVSS::perform_collision_SWS(Particle::OnePart *&ip,
                                  Particle::OnePart *&jp,
                                  Particle::OnePart *&kp,
                                  int &n_i,
                                  int &n_j,
                                  int &n_k,
                                  int &n_pre)
{
  int reactflag,kspecies;
  double x[3],v[3];
  Particle::OnePart *p3;
  Particle::OnePart *p_pre;   // SWS
  
  int i;   // SWS
  Particle::Species *species = particle->species;   // SWS

  //Additional parameters for Species weighting scheme
  //All of particles are pre-collision parameters
  // NOTE: kp is an output arg with no meaningful value on entry,
  //       so ksp/w_k are only set once a K particle is created below
  int isp = ip->ispecies;
  int jsp = jp->ispecies;
  int ksp;
  Particle::OnePart ip_pre = *ip;
  Particle::OnePart jp_pre = *jp;
  Particle::OnePart maxp_pre = *jp;
  // SWS - variables for weighting scheme
  double w_i = species[isp].specwt;
  double w_j = species[jsp].specwt;
  double w_k;
  double w_min = std::min(w_i, w_j);
  double w_max = std::max(w_i, w_j);
  double phi_i;
  double phi_j;
  double phi_k;

  // if gas-phase chemistry defined, attempt and perform reaction
  // if a 3rd particle is created, its kspecies >= 0 is returned
  // if 2nd particle is removed, its jspecies is set to -1

  if (react)
    reactflag = react->attempt(ip,jp,
                               precoln.etrans,precoln.erot,
                               precoln.evib,postcoln.etotal,kspecies);
  else reactflag = 0;

  // repartition energy and perform velocity scattering for I,J,K particles
  // reaction may have changed species of I,J particles
  // J,K particles may have been removed or created by reaction

  kp = NULL;

  if (reactflag) {
    // compute the number of split particle（0 or 1）
    if (w_i == w_j) n_pre = 0;         // SWS
    else n_pre = ((((w_max-w_min)/w_max)/random->uniform()>1)?1:0);   // SWS
    maxp_pre = ((w_i > w_j) ? ip_pre : jp_pre);      // SWS

    // particle creation of major reactant. This should also be taken account.
    // p_pre is the pointer for a reactant particle that continue to exist
    if ( n_pre == 1) {   // SWS
       int id = MAXSMALLINT*random->uniform();
       Particle::OnePart *particles = particle->particles;
       memcpy(x,maxp_pre.x,3*sizeof(double));
       memcpy(v,maxp_pre.v,3*sizeof(double));
       int reallocflag = 
       particle->add_particle(id,maxp_pre.ispecies,maxp_pre.icell,x,v,maxp_pre.erot,maxp_pre.evib);
       if (reallocflag) {
        ip = particle->particles + (ip - particles);
        jp = particle->particles + (jp - particles);
      }
       p_pre = &particle->particles[particle->nlocal-1];
     }          
    isp = ip->ispecies;  // SWS
    jsp = jp->ispecies;  // SWS
    w_i = species[isp].specwt;  // SWS
    w_j = species[jsp].specwt;  // SWS
    phi_i = w_min/w_i;  // SWS
    phi_j = w_min/w_j;  // SWS
    
    // add 3rd K particle if reaction created it
    // index of new K particle = nlocal-1
    // if add_particle() performs a realloc:
    //   make copy of x,v, then repoint ip,jp to new particles data struct
    //   unless electron

    if (kspecies >= 0) {
      int id = MAXSMALLINT*random->uniform();

      Particle::OnePart *particles = particle->particles;
      memcpy(x,ip->x,3*sizeof(double));
      memcpy(v,ip->v,3*sizeof(double));
      int ielectron_flag = (ambiflag && ip->ispecies == ambispecies);
      int jelectron_flag = (ambiflag && jp->ispecies == ambispecies);
      int reallocflag =
        particle->add_particle(id,kspecies,ip->icell,x,v,0.0,0.0);
      if (reallocflag) {
        if (!ielectron_flag)
          ip = particle->particles + (ip - particles);
        if (!jelectron_flag)
          jp = particle->particles + (jp - particles);
      }

      kp = &particle->particles[particle->nlocal-1];
      
      ksp = kp->ispecies;  // SWS
      w_k = species[ksp].specwt;  // SWS
      phi_k = w_min/w_k;  // SWS
      
      // !! if the reaction is impact ionization, 
      // If the ramdom number [0:1] is lower than the creteria made by two weight,
      // integer will be 1 , else 0
      // we can have number of additional product here
      if (phi_i < 1.0) n_i = (((phi_i)/random->uniform()>1)?1:0);  // SWS
      else if (phi_i >= 1.0) n_i = int(phi_i)+(((phi_i-int(phi_i))/random->uniform()>1)?1:0);  // SWS  
      if (phi_j < 1.0) n_j = (((phi_j)/random->uniform()>1)?1:0);  // SWS
      else if (phi_j >= 1.0) n_j = int(phi_j)+(((phi_j-int(phi_j))/random->uniform()>1)?1:0);  // SWS
      if (phi_k < 1.0) n_k = (((phi_k)/random->uniform()>1)?1:0);  // SWS
      else if (phi_k >= 1.0) n_k = int(phi_k)+(((phi_k-int(phi_k))/random->uniform()>1)?1:0);  // SWS
        
      EEXCHANGE_ReactingEDisposal(ip,jp,kp);
      SCATTER_ThreeBodyScattering(ip,jp,kp);

    // remove 2nd J particle if recombination reaction removed it
    // p3 is 3rd particle participating in energy exchange

    } else if (jp->ispecies < 0) {
      double *vi = ip->v;
      double *vj = jp->v;

      double divisor = 1.0 / (precoln.imass + precoln.jmass);
      double ucmf = ((precoln.imass*vi[0]) + (precoln.jmass*vj[0])) * divisor;
      double vcmf = ((precoln.imass*vi[1]) + (precoln.jmass*vj[1])) * divisor;
      double wcmf = ((precoln.imass*vi[2]) + (precoln.jmass*vj[2])) * divisor;

      vi[0] = ucmf;
      vi[1] = vcmf;
      vi[2] = wcmf;

      jp = NULL;
      p3 = react->recomb_part3;

      // properly account for 3rd body energy with another call to setup_collision()
      // it needs relative velocity of recombined species and 3rd body

      double *vp3 = p3->v;
      double du  = vi[0] - vp3[0];
      double dv  = vi[1] - vp3[1];
      double dw  = vi[2] - vp3[2];
      double vr2 = du*du + dv*dv + dw*dw;
      precoln.vr2 = vr2;

      // internal energy of ip particle is already included
      //   in postcoln.etotal returned from react->attempt()
      // but still need to add 3rd body internal energy

      double partial_energy =  postcoln.etotal + p3->erot + p3->evib;

      ip->erot = 0;
      ip->evib = 0;
      p3->erot = 0;
      p3->evib = 0;

      // returned postcoln.etotal will increment only the
      //   relative translational energy between recombined species and 3rd body
      // add back partial_energy to get full total energy

      setup_collision_SWS(ip,p3);  // SWS
      postcoln.etotal += partial_energy;
      
      if (phi_i < 1.0) n_i = (((phi_i)/random->uniform()>1)?1:0);  // SWS
      else if (phi_i >= 1.0) n_i = int(phi_i)+(((phi_i-int(phi_i))/random->uniform()>1)?1:0);   // SWS
      // because this reaction produce 1 particles, number of j,k particle is 0
      n_j = n_k = 0;    // SWS

      if (precoln.ave_dof > 0.0) EEXCHANGE_ReactingEDisposal(ip,p3,jp);

      // Add reactflag to scattering routine as 
      // splitting-merging should not be used when reaction
      // with differently weighted reactant and/or product occur.
      // Instead, the mass, momentum and energy are conserved
      // through the probability of creation/production of the
      // involved species.
      SCATTER_TwoBodyScattering_SWS(ip,p3,reactflag);  // SWS
    } else {
      // exchange reaction or associative ionization
      // compute number of particle after the reaction
      // !! if the reaction is charge exchange or associative ionization
      // !! reactant and product charge should be same.
      if (phi_i < 1.0) n_i = (((phi_i)/random->uniform()>1)?1:0);  // SWS
      else n_i = int(phi_i)+(((phi_i-int(phi_i))/random->uniform()>1)?1:0);  // SWS
      if (phi_j < 1.0) n_j = (((phi_j)/random->uniform()>1)?1:0);  // SWS
      else n_j = int(phi_j)+(((phi_j-int(phi_j))/random->uniform()>1)?1:0);  // SWS

      // because this reaction produce 2 particles, number of k particle is 0
      n_k = 0;  // SWS
      EEXCHANGE_ReactingEDisposal(ip,jp,kp);
      SCATTER_TwoBodyScattering_SWS(ip,jp,reactflag);  // SWS
    }

  } else {
    // no reaction
    // if reaction is not triggered, particle creation part is skipped
    if (precoln.ave_dof > 0.0) EEXCHANGE_NonReactingEDisposal_SWS(ip,jp);  // SWS
    SCATTER_TwoBodyScattering_SWS(ip,jp,reactflag); // SWS
  }

  return reactflag;
}

/* ---------------------------------------------------------------------- */
// ========================================================================
// Compute the post-collision velocity according using the 
// splitting-merging method.
// Compute and store the energy lost due to non conservation of the energy
// during differently weighted particles collision.
// If reaction occurs, the conservation is ensured by the 
// probobility of creation/deletion of the particles. Thus,
// the splitting-merging method is not used.
// Remark : if equally weighted particles collide phi=1.
// ========================================================================
void CollideVSS::SCATTER_TwoBodyScattering_SWS(Particle::OnePart *ip,
					   Particle::OnePart *jp, int reactflag)
{
  double ua,vb,wc;
  double vrc[3];

  Particle::Species *species = particle->species;
  double *vi = ip->v;
  double *vj = jp->v;
  int isp = ip->ispecies;
  int jsp = jp->ispecies;
  double mass_i = species[isp].mass;
  double mass_j = species[jsp].mass;
  
  // SWS - variables
  double w_i = species[isp].specwt;  
  double w_j = species[jsp].specwt; 
  double phi = 1.0; 

  double vi_pre[3];  
  double vi_post[3]; 
  double vj_pre[3];
  double vj_post[3];

  vi_pre[0]=vi[0];
  vi_pre[1]=vi[1];
  vi_pre[2]=vi[2];

  vj_pre[0]=vj[0];
  vj_pre[1]=vj[1];
  vj_pre[2]=vj[2];

  if ((w_i>0) && (w_j>0)){
    if (w_i>w_j){
      phi = w_j/w_i;
    } else {
      phi = w_i/w_j;
    } 
  }

  double alpha_r = 1.0 / params[isp][jsp].alpha;

  double eps = random->uniform() * 2*MY_PI;
  if (fabs(alpha_r - 1.0) < 0.001) {
    double vr = sqrt(2.0 * postcoln.etrans / params[isp][jsp].mr);
    double cosX = 2.0*random->uniform() - 1.0;
    double sinX = sqrt(1.0 - cosX*cosX);
    ua = vr*cosX;
    vb = vr*sinX*cos(eps);
    wc = vr*sinX*sin(eps);
  } else {
    double scale = sqrt((2.0 * postcoln.etrans) / (params[isp][jsp].mr * precoln.vr2));
    double cosX = 2.0*pow(random->uniform(),alpha_r) - 1.0;
    double sinX = sqrt(1.0 - cosX*cosX);
    vrc[0] = vi[0]-vj[0];
    vrc[1] = vi[1]-vj[1];
    vrc[2] = vi[2]-vj[2];
    double d = sqrt(vrc[1]*vrc[1]+vrc[2]*vrc[2]);
    if (d > 1.0e-6) {
      ua = scale * ( cosX*vrc[0] + sinX*d*sin(eps) );
      vb = scale * ( cosX*vrc[1] + sinX*(precoln.vr*vrc[2]*cos(eps) -
                                         vrc[0]*vrc[1]*sin(eps))/d );
      wc = scale * ( cosX*vrc[2] - sinX*(precoln.vr*vrc[1]*cos(eps) +
                                         vrc[0]*vrc[2]*sin(eps))/d );
    } else {
      ua = scale * ( cosX*vrc[0] );
      vb = scale * ( sinX*vrc[0]*cos(eps) );
      wc = scale * ( sinX*vrc[0]*sin(eps) );
    }
  }

  // new velocities for the products

  double divisor = 1.0 / (mass_i + mass_j);
  vi[0] = precoln.ucmf + (mass_j*divisor)*ua;
  vi[1] = precoln.vcmf + (mass_j*divisor)*vb;
  vi[2] = precoln.wcmf + (mass_j*divisor)*wc;
  vj[0] = precoln.ucmf - (mass_i*divisor)*ua;
  vj[1] = precoln.vcmf - (mass_i*divisor)*vb;
  vj[2] = precoln.wcmf - (mass_i*divisor)*wc;

  if (!(reactflag)){     // SWS
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
}

/* ---------------------------------------------------------------------- */
// ========================================================================
// Compute the post-collision rotational and vibrational energy
// using the splitting-merging method.
// ========================================================================
void CollideVSS::EEXCHANGE_NonReactingEDisposal_SWS(Particle::OnePart *ip,
                                                Particle::OnePart *jp)
{

  double State_prob,Fraction_Rot,Fraction_Vib,E_Dispose;
  int i,rotdof,vibdof,max_level,ivib;

  Particle::OnePart *p, *p2;
  Particle::Species *species = particle->species;

  double AdjustFactor = 0.99999999;
  postcoln.erot = 0.0;
  postcoln.evib = 0.0;
  double pevib = 0.0;

  // handle each kind of energy disposal for non-reacting reactants

  if (precoln.ave_dof == 0) {
    ip->erot = 0.0;
    jp->erot = 0.0;
    ip->evib = 0.0;
    jp->evib = 0.0;

  } else {
    E_Dispose = precoln.etrans;

    for (i = 0; i < 2; i++) {
      if (i == 0) {
        p = ip;
        p2 = jp;
      }
      else {
        p = jp;
        p2 = ip;
      }  

      // Two different methods are used:
      // 1) Exclusive method:
      // Allow all exchange scenario but the one that imply internal
      // energy exchange of a major particle when colliding with a 
      // minor particle. In this case the energy exchanged is negligible 
      // (proportional to phi). Particles still exchange internal energy 
      // when colliding with identically weighted particles or 
      // more highly weighted particles.
      // 2) Explicit method:
      // Use the splitting merging principle: E''_i=phi*E'_i+(1-phi)*E_i
      // Explicit conservation of the internal energy when a major
      // particle trigger internal energy exchange.
      // NOTE: 
      // The exclusive method is only used for discrete
      // vibrational energy exchange of major particles, as the discrete  
      // energy mode are proportional to integers (ivib) that 
      // are not mergeable without loosing this integer
      // property. All the other exchanges are handled via spitting
      // merging method.

      // SWS - variables
      int psp = p->ispecies;
      int p2sp = p2->ispecies;
      double w_p = species[psp].specwt;
      double w_p2 = species[p2sp].specwt;
      double phi=1.0;
      if ((w_p>0) && (w_p2>0)){ 
        if (w_p>w_p2){
          phi = w_p2/w_p;
        } else {
          phi = w_p/w_p2;
        } 
      }

      int sp = p->ispecies;
      rotdof = species[sp].rotdof;
      double rotn_phi = species[sp].rotrel;

      if (rotdof) {
        if (relaxflag == VARIABLE) rotn_phi = rotrel(sp,E_Dispose+p->erot);
        if (rotn_phi >= random->uniform()) {
          if (w_p>w_p2) {  // SWS
            if (rotstyle == NONE) {
              p->erot = 0.0;
            } else if (rotstyle != NONE && rotdof == 2) {
              double erot_pre = p->erot;
              E_Dispose += p->erot;
              Fraction_Rot =
               1- pow(random->uniform(),
		        (1/(2.5-params[ip->ispecies][jp->ispecies].omega)));
              double erot_post = Fraction_Rot * E_Dispose;
              E_Dispose -= erot_post;
              p->erot = phi * erot_post + (1-phi) * erot_pre;
            } else {
              double erot_pre = p->erot;
              E_Dispose += p->erot;
              double erot_post = E_Dispose *
                sample_bl(random,0.5*species[sp].rotdof-1.0,
                           1.5-params[ip->ispecies][jp->ispecies].omega);
              E_Dispose -= erot_post;
              p->erot = phi * erot_post + (1-phi) * erot_pre; // SWS
            } 
          } else {
            if (rotstyle == NONE) {
              p->erot = 0.0;
            } else if (rotstyle != NONE && rotdof == 2) {
              E_Dispose += p->erot;
              Fraction_Rot =
               1- pow(random->uniform(),
		        (1/(2.5-params[ip->ispecies][jp->ispecies].omega)));
              p->erot = Fraction_Rot * E_Dispose;
              E_Dispose -= p->erot;
            } else {
              E_Dispose += p->erot;
              p->erot = E_Dispose *
                sample_bl(random,0.5*species[sp].rotdof-1.0,
                           1.5-params[ip->ispecies][jp->ispecies].omega);
              E_Dispose -= p->erot;
            }
          }
        }
      }
      postcoln.erot += p->erot;

      vibdof = species[sp].vibdof;
      double vibn_phi = species[sp].vibrel[0];

      if (vibdof) {
        if (relaxflag == VARIABLE) vibn_phi = vibrel(sp,E_Dispose+p->evib);
        if (vibn_phi >= random->uniform()) {
          if (vibstyle == NONE) {
            p->evib = 0.0;

          } else if (vibdof == 2) {
            if (vibstyle == SMOOTH) {
              double e_vib_pre =  p->evib; // SWS
              E_Dispose += p->evib;
              Fraction_Vib =
                1.0 - pow(random->uniform(),
			            (1.0/(2.5-params[ip->ispecies][jp->ispecies].omega)));
              if (w_p>w_p2) {  // SWS
                double e_vib_post = Fraction_Vib * E_Dispose;
                E_Dispose -= e_vib_post;
                p->evib = phi * e_vib_post + (1-phi) * e_vib_pre;  // SWS
              } else {
                p->evib= Fraction_Vib * E_Dispose;
                E_Dispose -= p->evib;
              }

            } else if (vibstyle == DISCRETE) {
              // Added condition (Exclusive mode):
              if (!(w_p>w_p2)) {  // SWS
              E_Dispose += p->evib;
              max_level = static_cast<int>
                (E_Dispose / (update->boltz * species[sp].vibtemp[0]));
              do {
                ivib = static_cast<int>
                  (random->uniform()*(max_level+AdjustFactor));
                p->evib = ivib * update->boltz * species[sp].vibtemp[0];
                State_prob = pow((1.0 - p->evib / E_Dispose),
                                 (1.5 - params[ip->ispecies][jp->ispecies].omega));
              } while (State_prob < random->uniform());
              E_Dispose -= p->evib;
            }
            }

          } else if (vibdof > 2) {
            if (vibstyle == SMOOTH) {
              double e_vib_pre =  p->evib;  // SWS
              E_Dispose += p->evib;
              if (w_p>w_p2) {  // SWS
                double e_vib_post = E_Dispose *
                  sample_bl(random,0.5*species[sp].vibdof-1.0,
                            1.5-params[ip->ispecies][jp->ispecies].omega);
                E_Dispose -= e_vib_post;
                p->evib = phi * e_vib_post + (1-phi) * e_vib_pre;  // SWS
              } else {
                p->evib = E_Dispose *
                  sample_bl(random,0.5*species[sp].vibdof-1.0,
                            1.5-params[ip->ispecies][jp->ispecies].omega);
                E_Dispose -= p->evib;
              }

            } else if (vibstyle == DISCRETE) {
              // Added condition (Exclusive mode):
              if (!(w_p>w_p2)) {  // SWS
              p->evib = 0.0;
              
              int nmode = particle->species[sp].nvibmode;
              int **vibmode =
                particle->eiarray[particle->ewhich[index_vibmode]];
              int pindex = p - particle->particles;
              
              for (int imode = 0; imode < nmode; imode++) {
                ivib = vibmode[pindex][imode];
                E_Dispose += ivib * update->boltz *
                  particle->species[sp].vibtemp[imode];
                max_level = static_cast<int>
                  (E_Dispose / (update->boltz * species[sp].vibtemp[imode]));
              
                do {
                  ivib = static_cast<int>
                    (random->uniform()*(max_level+AdjustFactor));
                  pevib = ivib * update->boltz * species[sp].vibtemp[imode];
                  State_prob = pow((1.0 - pevib / E_Dispose),
                                   (1.5 - params[ip->ispecies][jp->ispecies].omega));
                } while (State_prob < random->uniform());
              
                vibmode[pindex][imode] = ivib;
                p->evib += pevib;
                E_Dispose -= pevib;
              }
              }
            }
          } // end of vibstyle/vibdof if
        }
        postcoln.evib += p->evib;
      } // end of vibdof if
    }
  }

  // compute portion of energy left over for scattering

  postcoln.eint = postcoln.erot + postcoln.evib;
  postcoln.etrans = E_Dispose;
}

/* ----------------------------------------------------------------------
   explicit template instantiations for the SWS collision loops
------------------------------------------------------------------------- */

namespace SPARTA_NS {
template void Collide::collisions_one_SWS<0>();
template void Collide::collisions_one_SWS<1>();
template void Collide::collisions_group_SWS<0>();
template void Collide::collisions_group_SWS<1>();
}
