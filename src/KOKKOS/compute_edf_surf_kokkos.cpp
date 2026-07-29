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
#include "compute_edf_surf_kokkos.h"
#include "particle_kokkos.h"
#include "mixture.h"
#include "surf_kokkos.h"
#include "grid.h"
#include "update.h"
#include "memory_kokkos.h"
#include "error.h"
#include "sparta_masks.h"
#include "kokkos.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

ComputeEDFSurfKokkos::ComputeEDFSurfKokkos(SPARTA *sparta, int narg, char **arg) :
  ComputeEDFSurf(sparta, narg, arg)
{
  kokkos_flag = 1;
}

ComputeEDFSurfKokkos::ComputeEDFSurfKokkos(SPARTA *sparta) :
  ComputeEDFSurf(sparta)
{
  copy = 1;
  uncopy = 0;
}

/* ---------------------------------------------------------------------- */

ComputeEDFSurfKokkos::~ComputeEDFSurfKokkos()
{
  if (copy || copymode) return;

  memoryKK->destroy_kokkos(k_tally2surf,tally2surf);
  memoryKK->destroy_kokkos(k_array_surf_tally,array_surf_tally);
}

/* ---------------------------------------------------------------------- */

void ComputeEDFSurfKokkos::init()
{
  ComputeEDFSurf::init();

  mvv2e = update->mvv2e;
  useweight = weightflag && cellweightflag;
}

/* ----------------------------------------------------------------------
   size the tally arrays by surf count, since a Kokkos kernel cannot realloc
   called by init() and whenever the grid changes
------------------------------------------------------------------------- */

void ComputeEDFSurfKokkos::allocate_tally()
{
  int nsurf = surf->nlocal + surf->nghost;

  memoryKK->grow_kokkos(k_tally2surf,tally2surf,nsurf,"edf/surf:tally2surf");
  d_tally2surf = k_tally2surf.view_device();

  d_surf2tally = DAT::t_int_1d("edf/surf:surf2tally",nsurf);
  Kokkos::deep_copy(d_surf2tally,-1);

  memoryKK->grow_kokkos(k_array_surf_tally,array_surf_tally,nsurf,ntotal,
                        "edf/surf:array_surf_tally");
  d_array_surf_tally = k_array_surf_tally.view_device();
}

/* ----------------------------------------------------------------------
   the tally arrays are sized by surf count, and nghost changes when the grid
     changes if surfs are distributed, so resize them
   called by Grid::notify_changed() whenever the grid changes
------------------------------------------------------------------------- */

void ComputeEDFSurfKokkos::reallocate()
{
  allocate_tally();
}

/* ---------------------------------------------------------------------- */

void ComputeEDFSurfKokkos::clear()
{
  // reset all tallies and surf2tally flags
  // called by Update at beginning of timesteps surf tallying is done

  Kokkos::deep_copy(d_array_surf_tally,0);
  Kokkos::deep_copy(d_surf2tally,-1);

  ntally = 0;
  combined = 0;
  compressed = 0;
}

/* ---------------------------------------------------------------------- */

void ComputeEDFSurfKokkos::pre_surf_tally()
{
  mvv2e = update->mvv2e;
  useweight = weightflag && cellweightflag;

  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Device,SPECIES_MASK);
  d_species = particle_kk->k_species.view_device();
  d_s2g = particle_kk->k_species2group.view_device();

  SurfKokkos* surf_kk = (SurfKokkos*) surf;
  surf_kk->sync(Device,ALL_MASK);
  d_lines = surf_kk->k_lines.view_device();
  d_tris = surf_kk->k_tris.view_device();

  need_dup = sparta->kokkos->need_dup<DeviceType>();
  if (need_dup)
    dup_array_surf_tally = Kokkos::Experimental::create_scatter_view<typename Kokkos::Experimental::ScatterSum, typename Kokkos::Experimental::ScatterDuplicated>(d_array_surf_tally);
  else
    ndup_array_surf_tally = Kokkos::Experimental::create_scatter_view<typename Kokkos::Experimental::ScatterSum, typename Kokkos::Experimental::ScatterNonDuplicated>(d_array_surf_tally);
}

/* ---------------------------------------------------------------------- */

void ComputeEDFSurfKokkos::post_surf_tally()
{
  if (need_dup) {
    Kokkos::Experimental::contribute(d_array_surf_tally, dup_array_surf_tally);
    dup_array_surf_tally = {}; // free duplicated memory
  }

  k_tally2surf.modify_device();
  k_array_surf_tally.modify_device();
}

/* ----------------------------------------------------------------------
   return # of tallies and their surf IDs
   the device kernel tallies into row isurf, so bring the arrays back to the
     host and compress out the untallied rows
   compressing is destructive, so only do it once per clear() cycle
------------------------------------------------------------------------- */

int ComputeEDFSurfKokkos::tallyinfo(surfint *&ptr)
{
  if (compressed) {
    ptr = tally2surf;
    return ntally;
  }
  compressed = 1;

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

/* ----------------------------------------------------------------------
   sum tally values to owning surfs
   callers such as dump surf invoke this without calling tallyinfo() first,
     so make sure the device tallies have been brought over and compressed
------------------------------------------------------------------------- */

void ComputeEDFSurfKokkos::post_process_surf()
{
  if (combined) return;

  surfint *ptr;
  tallyinfo(ptr);

  ComputeEDFSurf::post_process_surf();
}

/* ---------------------------------------------------------------------- */

void ComputeEDFSurfKokkos::grow_tally()
{
  // cannot realloc inside a Kokkos parallel region, so the tally arrays are
  //   already sized by surf count in allocate_tally()

  allocate_tally();
}
