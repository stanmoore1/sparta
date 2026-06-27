/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.sandia.gov
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Nagoya University Species Weighting Scheme (SWS) collision drivers.
   These Collide methods are defined here (rather than in collide.cpp) to keep
   collide.cpp manageable, mirroring how the SWPM drivers live in
   collide_swpm.cpp / collide_reduce.cpp.
------------------------------------------------------------------------- */

#include "math.h"
#include "math_extra.h"
#include "string.h"
#include "collide.h"
#include "particle.h"
#include "mixture.h"
#include "update.h"
#include "grid.h"
#include "comm.h"
#include "react.h"
#include "modify.h"
#include "fix.h"
#include "fix_ambipolar.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "memory.h"
#include "error.h"
#include <algorithm>

using namespace SPARTA_NS;

#define DELTADELETE 1024
#define DELTAELECTRON 128

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
    Ewilost = 0.0;                        // SWS
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

      // if jpart destroyed: delete from plist, add particle to deletion list
      // exit attempt loop if only single particle left
      
      ///////////==============================================
      // Here, we add the particles generated in perform_collide to plist.
      // To avoid updating nlocal changes, add them to the plist first.
      ///////////==============================================

      int i_add = 0; // to count 
      if (n_k) {
        i_add++;
        if (np == npmax) {
          npmax += DELTAPART;
          memory->grow(plist,npmax,"collide:plist");
        }
        if (NEARCP) set_nn(np);
        plist[np++] = particle->nlocal-i_add;
        particles = particle->particles;
      }      
      if (n_pre) {
        i_add++;
        if (np == npmax) {
          npmax += DELTAPART;
          memory->grow(plist,npmax,"collide:plist");
        }
        if (NEARCP) set_nn(np);
        plist[np++] = particle->nlocal-i_add;
        particles = particle->particles;
      }           
      ///////////==============================================

      // delete from plist if i is destroyed by probability
      //if (!ipart) {
      if (!n_i) {
        //printf("!!check del i \n");
        if (ndelete == maxdelete) {
          maxdelete += DELTADELETE;
          memory->grow(dellist,maxdelete,"collide:dellist");
        }
        dellist[ndelete++] = plist[i];
        np--;
        plist[i] = plist[np];
        if (NEARCP) nn_last_partner[i] = nn_last_partner[np];
        if (np < 2) break;
      }

      // delete from plist if j is destroyed by probability or recombination
      //if (!jpart) {
      if (!n_j) {
        //printf("!!check del j \n");       
        if (ndelete == maxdelete) {
          maxdelete += DELTADELETE;
          memory->grow(dellist,maxdelete,"collide:dellist");
        }
        dellist[ndelete++] = plist[j];
        np--;
        plist[j] = plist[np];
        if (NEARCP) nn_last_partner[j] = nn_last_partner[np];
        if (np < 2) break;
      }       

      // If a 3rd (k) particle was created by the reaction but is not kept by
      // the weighting probability (n_k == 0), delete it.  kpart was created in
      // perform_collision_SWS and was NOT added to plist (the plist add at the
      // top of this block is guarded by "if (n_k)"), so delete it by its actual
      // particle index and do not touch plist here.  (The previous code indexed
      // plist[k], but k is only set for recombination pairs and is otherwise
      // stale, which corrupted dellist and crashed compress_reactions.)
      if (!n_k && kpart) {
        if (ndelete == maxdelete) {
          maxdelete += DELTADELETE;
          memory->grow(dellist,maxdelete,"collide:dellist");
        }
        dellist[ndelete++] = kpart - particle->particles;
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
    }
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
    Ewilost = 0.0;                      // SWS
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

  double maxwi=0.0;   // SWS
  for (i = 0; i < ngroups; i++) {  // SWS
    maxwi = std::max(maxwigr[i],maxwi);
  }
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
    Ewilost = 0.0;                      // SWS
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

      // counter of 
      int i_add = 0; // number of particle set as normal major particle 
      int i_add_ele = 0; // number of particle set as electron 
      if (n_k) {
        i_add++;
        particles = particle->particles;
        ionambi = particle->eivec[particle->ewhich[index_ionambi]];
        velambi = particle->edarray[particle->ewhich[index_velambi]];
        if (kpart->ispecies != ambispecies) {
          if (np == npmax) {
            npmax += DELTAPART;
            memory->grow(plist,npmax,"collide:plist");
          }
          plist[np++] = particle->nlocal-i_add;
          // to save index of k to use later for particle creation
          k = particle->nlocal-i_add;
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
          i_add_ele++;
        }
      }  

      // major particle which created through major-minor is added
      if (n_pre) {
        // pointer for new particle created in vss collide
        Particle::OnePart *p_pre;
        i_add++;
        particles = particle->particles;
        ionambi = particle->eivec[particle->ewhich[index_ionambi]];
        velambi = particle->edarray[particle->ewhich[index_velambi]];
        p_pre = &particle->particles[particle->nlocal-i_add];
        if (p_pre->ispecies != ambispecies) {
          if (np == npmax) {
            npmax += DELTAPART;
            memory->grow(plist,npmax,"collide:plist");
          }
          plist[np++] = particle->nlocal-i_add+i_add_ele;
          ionambi[particle->nlocal-i_add+i_add_ele]=0; 

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
          i_add_ele++;
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
      
      // delete needless i particle 
      // if i particle is destoryed because of probability(n_i = 0), 
      if (!n_i && ipart ) {
        //printf("!!check del i \n");
        if (ndelete == maxdelete) {
          maxdelete += DELTADELETE;
          memory->grow(dellist,maxdelete,"collide:dellist");
        }
        dellist[ndelete++] = plist[i];
        np--;
        plist[i] = plist[np];
        ionambi[plist[i]]=ionambi[plist[np]];
      }      

      // delete needless k particle 
      // If k particles are created, but become unnecessary with some probability and are deleted
      // In reaction not envolving third speices, kpart = NULL
      // therefore, n_k = 0 and kpart is not NULL is the condition
      if (!n_k && kpart) {
        //printf("!!check del k \n");
        if (ndelete == maxdelete) {
          maxdelete += DELTADELETE;
          memory->grow(dellist,maxdelete,"collide:dellist");
        }
        dellist[ndelete++] = plist[k];
        np--;
        plist[k] = plist[np];
        ionambi[plist[k]]=ionambi[plist[np]];
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
      
      // copy paste kpart particle 
      // k is heavy or electron
      // a k paricle is already added. rest of them is added
      if (kpart) {
        // printf("!!check cp k \n");
        if (kpart->ispecies == ambispecies) { 
          // for ambipolar electron
          // if n_k = 1 or 0 , the particle is already created and not going into for loop
          for (i_loop = 0; i_loop < n_k-1 ; i_loop++) {      
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
        } else {
          // for heavy particle 
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
              if(ipart) ipart = particle->particles + (ipart - particles);
              if(jpart) jpart = particle->particles + (jpart - particles);
              if(kpart) kpart = particle->particles + (kpart - particles);
              kpart = particle->particles + (kpart - particles);
          }                      
            if (np == npmax) {
              npmax += DELTAPART;
              memory->grow(plist,npmax,"collide:plist");
            }
            plist[np++] = particle->nlocal-1;
            particles = particle->particles;
            ionambi[particle->nlocal-1] = ionambi[plist[k]];
          }
        }
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
      Particle::OnePart jp = *jpart;
      
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
      // k is heavy or electron
      // a k paricle is already added. rest of them is added
      if (kpart) {
        // printf("!!check cp k \n");
        if (kpart->ispecies == ambispecies) { 
          // for ambipolar electron
          // if n_k = 1 or 0 , the particle is already created and not going into for loop
          for (i_loop = 0; i_loop < n_k-1 ; i_loop++) {      
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
        } else {
          // for heavy particle 
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
            ionambi[particle->nlocal-1] = ionambi[plist[k]];
          }
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
    Ewilost = 0.0;   // SWS
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

  double maxwi=0.0;    // SWS
  for (i = 0; i < ngroups; i++) {    // SWS
    maxwi = std::max(maxwigr[i],maxwi);
  }
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
  }
}

/* ----------------------------------------------------------------------
   explicit template instantiations (definitions live in this translation
   unit, but the templates are dispatched from collide.cpp)
------------------------------------------------------------------------- */

namespace SPARTA_NS {
template void Collide::collisions_one_SWS<0>();
template void Collide::collisions_one_SWS<1>();
template void Collide::collisions_group_SWS<0>();
template void Collide::collisions_group_SWS<1>();
}
