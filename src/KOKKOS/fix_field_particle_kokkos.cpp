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
#include <type_traits>
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

  // array_particle is one contiguous row-major block: Memory::create(TYPE**&,n1,n2)
  //   (memory.h:114-127) does a single allocation of n1*n2 and points each
  //   row pointer into it.  The Kokkos view is LayoutRight, so the two have
  //   the same element order: wrap the existing buffer in an unmanaged View
  //   and copy it to the device with deep_copy_convert(), which is a plain
  //   deep_copy when the value types match (KK_ACC_FLOAT is double) and
  //   otherwise converts on the host.

  Kokkos::View<double**,Kokkos::LayoutRight,Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged> >
    h_array_particle(array_particle[0],nlocal,ncols);

  // both sides can be longer than nlocal: the host array is sized to
  //   maxparticle/maxgrid, and the DualView guard above is grow-only
  //   (extent(0) < nlocal), so it keeps a high-water-mark row count after the
  //   count drops.  Kokkos::deep_copy throws on any extent mismatch
  //   (Kokkos_CopyViews.hpp:1163), so copy the leading nlocal rows rather
  //   than the whole allocation.  A leading row range of a LayoutRight view
  //   is still contiguous, so the subview costs nothing at runtime

  auto d_rows = Kokkos::subview(k_array_particle.view_device(),
                                Kokkos::make_pair(0,nlocal),Kokkos::ALL());
  // in a reduced precision build deep_copy_convert() converts on the host
  deep_copy_convert(d_rows,h_array_particle);
  k_array_particle.modify_device();

  d_array_particle = k_array_particle.view_device();
}
