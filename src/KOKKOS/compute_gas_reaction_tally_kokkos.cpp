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

#include "compute_gas_reaction_tally_kokkos.h"
#include "particle_kokkos.h"
#include "grid_kokkos.h"
#include "update.h"
#include "domain.h"
#include "mixture.h"
#include "memory_kokkos.h"
#include "sparta_masks.h"
#include "error.h"

using namespace SPARTA_NS;

#define DELTA 4096

/* ---------------------------------------------------------------------- */

ComputeGasReactionTallyKokkos::ComputeGasReactionTallyKokkos(SPARTA *sparta,
                                                                int narg, char **arg) :
  ComputeGasReactionTally(sparta, narg, arg)
{
  kokkos_flag = 1;

  d_ntally = DAT::t_int_scalar("gas/reaction/tally/kk:ntally");
  h_ntally = HAT::t_int_scalar("gas/reaction/tally/kk:ntally_mirror");

  // flatten the value list once; it never changes after construction

  DAT::tdual_int_1d k_which("gas/reaction/tally/kk:which",nvalue);
  for (int m = 0; m < nvalue; m++) k_which.view_host()[m] = which[m];
  k_which.modify_host();
  k_which.sync_device();
  d_which = k_which.view_device();

  maxtally = DELTA;
  MemKK::realloc_kokkos(k_array_tally,"gas/reaction/tally/kk:array_tally",
                        maxtally,nvalue);
  d_array_tally = k_array_tally.view_device();
}

/* ---------------------------------------------------------------------- */

ComputeGasReactionTallyKokkos::ComputeGasReactionTallyKokkos(SPARTA *sparta) :
  ComputeGasReactionTally(sparta)
{
  copy = 1;
}

/* ---------------------------------------------------------------------- */

ComputeGasReactionTallyKokkos::~ComputeGasReactionTallyKokkos()
{
  if (copy || copymode) return;

  // the host base class frees array_tally, which this class never allocated

  memory->destroy(array_tally);
  array_tally = NULL;
  maxtally = 0;
}

/* ---------------------------------------------------------------------- */

void ComputeGasReactionTallyKokkos::clear()
{
  ntally = 0;
}

/* ----------------------------------------------------------------------
   called by CollideVSSKokkos before the collision kernel
------------------------------------------------------------------------- */

void ComputeGasReactionTallyKokkos::pre_gas_tally()
{
  GridKokkos* grid_kk = (GridKokkos*) grid;
  grid_kk->sync(Device,CELL_MASK|CINFO_MASK);
  d_cells = grid_kk->k_cells.view_device();
  d_cinfo = grid_kk->k_cinfo.view_device();

  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Device,SPECIES_MASK);
  d_s2g = particle_kk->k_species2group.view_device();

  Kokkos::deep_copy(d_ntally,0);
}

/* ----------------------------------------------------------------------
   called by CollideVSSKokkos after the collision kernel
   bring the row count and the rows themselves to the host, where
     dump tally and Compute::tallyinfo() read them
------------------------------------------------------------------------- */

void ComputeGasReactionTallyKokkos::post_gas_tally()
{
  Kokkos::deep_copy(h_ntally,d_ntally);
  ntally = h_ntally();

  // an overflowed attempt is discarded and repeated by CollideVSSKokkos, so do
  //   not publish its partial rows

  if (ntally > (int) d_array_tally.extent(0)) return;

  k_array_tally.modify_device();
  k_array_tally.sync_host();

  // the host base class hands out array_tally; point it at the host mirror

  if (ntally) {
    memory->destroy(array_tally);
    memory->create(array_tally,MAX(ntally,1),nvalue,
                   "gas/reaction/tally/kk:array_tally_host");
    auto h_array = k_array_tally.view_host();
    for (int i = 0; i < ntally; i++)
      for (int m = 0; m < nvalue; m++)
        array_tally[i][m] = h_array(i,m);
  }
}

/* ----------------------------------------------------------------------
   grow the device row buffer to hold at least N rows
   called by CollideVSSKokkos when an attempt overflowed, before repeating it
------------------------------------------------------------------------- */

void ComputeGasReactionTallyKokkos::grow_tally_kokkos(int n)
{
  if (n <= (int) d_array_tally.extent(0)) return;

  // grow past what the failed attempt asked for, so a step whose collision
  //   count is still climbing does not repeat the move again and again

  maxtally = MAX(n + DELTA, (int)(1.5*n));
  MemKK::realloc_kokkos(k_array_tally,"gas/reaction/tally/kk:array_tally",
                        maxtally,nvalue);
  d_array_tally = k_array_tally.view_device();
}
