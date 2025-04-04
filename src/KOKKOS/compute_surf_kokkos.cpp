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

#include "string.h"
#include "compute_surf_kokkos.h"
#include "particle_kokkos.h"
#include "mixture.h"
#include "surf_kokkos.h"
#include "grid.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "memory_kokkos.h"
#include "error.h"
#include "sparta_masks.h"
#include "kokkos.h"

using namespace SPARTA_NS;

#define VAL_1(X) X
#define VAL_2(X) VAL_1(X), VAL_1(X)

/* ---------------------------------------------------------------------- */

ComputeSurfKokkos::ComputeSurfKokkos(SPARTA *sparta, int narg, char **arg) :
  ComputeSurf(sparta, narg, arg),
  sr_kk_global_copy{VAL_2(KKCopy<SurfReactGlobalKokkos>(sparta))},
  sr_kk_prob_copy{VAL_2(KKCopy<SurfReactProbKokkos>(sparta))}
{
  kokkos_flag = 1;
  d_which = DAT::t_int_1d("surf:which",nvalue);
}

ComputeSurfKokkos::ComputeSurfKokkos(SPARTA *sparta) :
  ComputeSurf(sparta),
  sr_kk_global_copy{VAL_2(KKCopy<SurfReactGlobalKokkos>(sparta))},
  sr_kk_prob_copy{VAL_2(KKCopy<SurfReactProbKokkos>(sparta))}
{
  copy = 1;
}

/* ---------------------------------------------------------------------- */

ComputeSurfKokkos::~ComputeSurfKokkos()
{
  if (uncopy) {
    for (int i = 0; i < KOKKOS_MAX_SURF_REACT_PER_TYPE; i++) {
      sr_kk_global_copy[i].uncopy();
      sr_kk_prob_copy[i].uncopy();
    }
  }

  if (copy) return;

  memoryKK->destroy_kokkos(k_tally2surf,tally2surf);
  memoryKK->destroy_kokkos(k_array_surf_tally,array_surf_tally);
}

/* ---------------------------------------------------------------------- */

void ComputeSurfKokkos::init()
{
  ComputeSurf::init();

  auto h_which = Kokkos::create_mirror_view(d_which);
  for (int n=0; n<nvalue; n++)
    h_which(n) = which[n];
  Kokkos::deep_copy(d_which,h_which);
}

/* ----------------------------------------------------------------------
   set normflux for all surfs I store
   all: just nlocal
   distributed: nlocal + nghost
   called by init before each run (in case dt or fnum has changed)
   called whenever grid changes
------------------------------------------------------------------------- */

void ComputeSurfKokkos::init_normflux()
{
  ComputeSurf::init_normflux();

  int nsurf = surf->nlocal + surf->nghost;

  d_normflux = DAT::t_float_1d("surf:normflux",nsurf);
  auto h_normflux = Kokkos::create_mirror_view(d_normflux);
  for (int n=0; n<nsurf; n++)
    h_normflux(n) = normflux[n];
  Kokkos::deep_copy(d_normflux,h_normflux);

  // Cannot realloc inside a Kokkos parallel region, so size tally2surf as nsurf
  memoryKK->grow_kokkos(k_tally2surf,tally2surf,nsurf,"surf:tally2surf");
  d_tally2surf = k_tally2surf.d_view;
  d_surf2tally = DAT::t_int_1d("surf:surf2tally",nsurf);
  Kokkos::deep_copy(d_surf2tally,-1);

  memoryKK->grow_kokkos(k_array_surf_tally,array_surf_tally,nsurf,ntotal,"surf:array_surf_tally");
  d_array_surf_tally = k_array_surf_tally.d_view;
}

/* ---------------------------------------------------------------------- */

void ComputeSurfKokkos::clear()
{
  // reset all set surf2tally values to -1
  // called by Update at beginning of timesteps surf tallying is done

  combined = 0;
  Kokkos::deep_copy(d_array_surf_tally,0);

  Kokkos::deep_copy(d_surf2tally,-1);
}

/* ---------------------------------------------------------------------- */

void ComputeSurfKokkos::pre_surf_tally()
{
  mvv2e = update->mvv2e;

  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Device,SPECIES_MASK);
  d_species = particle_kk->k_species.d_view;
  d_s2g = particle_kk->k_species2group.d_view;

  SurfKokkos* surf_kk = (SurfKokkos*) surf;
  surf_kk->sync(Device,ALL_MASK);
  d_lines = surf_kk->k_lines.d_view;
  d_tris = surf_kk->k_tris.d_view;

  need_dup = sparta->kokkos->need_dup<DeviceType>();
  if (need_dup)
    dup_array_surf_tally = Kokkos::Experimental::create_scatter_view<typename Kokkos::Experimental::ScatterSum, typename Kokkos::Experimental::ScatterDuplicated>(d_array_surf_tally);
  else
    ndup_array_surf_tally = Kokkos::Experimental::create_scatter_view<typename Kokkos::Experimental::ScatterSum, typename Kokkos::Experimental::ScatterNonDuplicated>(d_array_surf_tally);

  if (surf->nsr > KOKKOS_MAX_TOT_SURF_REACT)
    error->all(FLERR,"Kokkos currently supports two instances of each surface reaction method");

  if (surf->nsr > 0) {
    int nglob,nprob;
    nglob = nprob = 0;
    for (int n = 0; n < surf->nsr; n++) {
      if (!surf->sr[n]->kokkosable)
        error->all(FLERR,"Must use Kokkos-enabled surface reaction method with Kokkos");
      if (strcmp(surf->sr[n]->style,"global") == 0) {
        sr_kk_global_copy[nglob].copy((SurfReactGlobalKokkos*)(surf->sr[n]));
        sr_kk_global_copy[nglob].obj.pre_react();
        sr_type_list[n] = 0;
        sr_map[n] = nprob;
        nglob++;
      } else if (strcmp(surf->sr[n]->style,"prob") == 0) {
        sr_kk_prob_copy[nprob].copy((SurfReactProbKokkos*)(surf->sr[n]));
        sr_kk_prob_copy[nprob].obj.pre_react();
        sr_type_list[n] = 1;
        sr_map[n] = nprob;
        nprob++;
      } else {
        error->all(FLERR,"Unknown Kokkos surface reaction method");
      }
    }

    if (nglob > KOKKOS_MAX_SURF_REACT_PER_TYPE || nprob > KOKKOS_MAX_SURF_REACT_PER_TYPE)
      error->all(FLERR,"Kokkos currently supports two instances of each surface reaction method");
  }
}

/* ---------------------------------------------------------------------- */

void ComputeSurfKokkos::post_surf_tally()
{
  if (need_dup) {
    Kokkos::Experimental::contribute(d_array_surf_tally, dup_array_surf_tally);
    dup_array_surf_tally = {}; // free duplicated memory
  }

  k_tally2surf.modify_device();
  k_array_surf_tally.modify_device();
}

/* ----------------------------------------------------------------------
   return ptr to norm vector used by column N
------------------------------------------------------------------------- */

int ComputeSurfKokkos::tallyinfo(surfint *&ptr)
{
  k_tally2surf.sync_host();
  ptr = tally2surf;

  k_array_surf_tally.sync_host();
  auto h_surf2tally = Kokkos::create_mirror_view(d_surf2tally);
  Kokkos::deep_copy(h_surf2tally,d_surf2tally);

  // compress array_surf_tally

  int nsurf = surf->nlocal + surf->nghost;
  int istart = 0;
  int iend = nsurf-1;

  while (1) {
    while (h_surf2tally[istart] != -1 && istart < nsurf-2) istart++;
    while (h_surf2tally[iend] == -1 && iend > 0) iend--;
    if (istart >= iend) {
      ntally = istart;
      break;
    }
    for (int k = 0; k < ntotal; k++) {
      array_surf_tally[istart][k] = array_surf_tally[iend][k];
    }
    h_surf2tally[istart] = h_surf2tally[iend];
    h_surf2tally[iend] = -1;
    tally2surf[istart] = tally2surf[iend];
  }

  return ntally;
}

/* ---------------------------------------------------------------------- */

void ComputeSurfKokkos::grow_tally()
{
  // Cannot realloc inside a Kokkos parallel region, so size tally2surf the
  //  same as surf2tally

  int nsurf = surf->nlocal + surf->nghost;

  memoryKK->grow_kokkos(k_tally2surf,tally2surf,nsurf,"surf:tally2surf");
  d_tally2surf = k_tally2surf.d_view;

  memoryKK->grow_kokkos(k_array_surf_tally,array_surf_tally,nsurf,ntotal,"surf:array_surf_tally");
  d_array_surf_tally = k_array_surf_tally.d_view;
}

