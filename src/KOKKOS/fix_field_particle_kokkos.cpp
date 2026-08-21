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

#include "fix_field_particle_kokkos.h"
#include "particle.h"
#include "memory_kokkos.h"
#include "sparta_masks.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

FixFieldParticleKokkos::FixFieldParticleKokkos(SPARTA *sparta, int narg, char **arg) :
  FixFieldParticle(sparta, narg, arg)
{
  // the variable evaluation in compute_field() runs on the host, so this fix
  //   is a host fix.  it is not invoked through Modify -- UpdateKokkos calls
  //   compute_field() directly -- so the datamasks only describe the fact
  //   that no particle data is written here; VariableKokkos does its own
  //   sync(Host,PARTICLE_MASK) around the particle-style evaluation

  kokkos_flag = 0;
  execution_space = Host;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
}

/* ---------------------------------------------------------------------- */

FixFieldParticleKokkos::~FixFieldParticleKokkos()
{
}

/* ----------------------------------------------------------------------
   evaluate the per-particle field on the host, then publish it to the
     device view UpdateKokkos::field_per_particle() reads
   indexed by particle index, matching the ordering the move kernel uses
------------------------------------------------------------------------- */

void FixFieldParticleKokkos::compute_field()
{
  FixFieldParticle::compute_field();

  const int nlocal = particle->nlocal;
  if (!nlocal) return;
  const int ncols = size_per_particle_cols;

  if ((int) k_array_particle.extent(0) < nlocal ||
      (int) k_array_particle.extent(1) != ncols)
    MemKK::realloc_kokkos(k_array_particle,"field/particle/kk:array_particle",
                          nlocal,ncols);

  auto h_array_particle = k_array_particle.view_host();
  for (int i = 0; i < nlocal; i++)
    for (int j = 0; j < ncols; j++)
      h_array_particle(i,j) = array_particle[i][j];

  k_array_particle.modify_host();
  k_array_particle.sync_device();

  d_array_particle = k_array_particle.view_device();
}
