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

#include "math.h"
#include "fix_vibmode_kokkos.h"
#include "update.h"
#include "particle.h"
#include "collide.h"
#include "comm.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "math_const.h"
#include "error.h"
#include "sparta_masks.h"

using namespace SPARTA_NS;
using namespace MathConst;

enum{INT,DOUBLE};                      // several files
enum{NONE,DISCRETE,SMOOTH};            // several files

/* ---------------------------------------------------------------------- */

FixVibmodeKokkos::FixVibmodeKokkos(SPARTA *sparta, int narg, char **arg) :
  FixVibmode(sparta, narg, arg),
  rand_pool(12345 + comm->me
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            )
{
  kokkos_flag = 1;
  execution_space = Device;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.init(random);
#endif
}

/* ---------------------------------------------------------------------- */

FixVibmodeKokkos::FixVibmodeKokkos(SPARTA *sparta) :
  FixVibmode(sparta),
  rand_pool(12345 // seed doesn't matter since it will just be copied over
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            )
{
  copy = 1;
}

/* ---------------------------------------------------------------------- */

FixVibmodeKokkos::~FixVibmodeKokkos()
{
  if (copy) return;

#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.destroy();
#endif
}

/* ---------------------------------------------------------------------- */

void FixVibmodeKokkos::pre_update_custom_kokkos()
{
  boltz = update->boltz;

  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK|CUSTOM_MASK);
  d_particles = particle_kk->k_particles.view_device();
  d_species = particle_kk->k_species.view_device();
  auto h_ewhich = particle_kk->k_ewhich.view_host();
  auto k_eiarray = particle_kk->k_eiarray;
  d_vibmode = k_eiarray.view_host()[h_ewhich[vibmodeindex]].k_view.view_device();
}

/* ----------------------------------------------------------------------
   snapshot/restore the vibmode custom array for the react-retry rollback:
   the move kernel mutates it for existing particles via
   update_custom_kokkos(), so restoring only the particle list would leave
   the mode levels inconsistent with the rolled-back evib.
   pre_update_custom_kokkos() must be called first so d_vibmode references
   the current allocation
------------------------------------------------------------------------- */

void FixVibmodeKokkos::backup_custom_kokkos()
{
  d_vibmode_backup = DAT::t_int_2d_lr(Kokkos::view_alloc("vibmode:vibmode_backup",Kokkos::WithoutInitializing),d_vibmode.extent(0),d_vibmode.extent(1));
  Kokkos::deep_copy(d_vibmode_backup,d_vibmode);
}

/* ---------------------------------------------------------------------- */

void FixVibmodeKokkos::restore_custom_kokkos()
{
  if (!d_vibmode_backup.data()) return;   // already restored this attempt

  Kokkos::deep_copy(d_vibmode,d_vibmode_backup);

  // deallocate reference to reduce memory use

  d_vibmode_backup = {};
}

/* ----------------------------------------------------------------------
   called when a particle with index is created
    or when temperature dependent properties need to be updated
   populate all vibrational modes and set evib = sum of mode energies
------------------------------------------------------------------------- */

void FixVibmodeKokkos::update_custom(int index, double temp_thermal,
                                     double temp_rot, double temp_vib,
                                     double *vstream)
{
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Host,PARTICLE_MASK|SPECIES_MASK|CUSTOM_MASK);
  FixVibmode::update_custom(index, temp_thermal, temp_rot, temp_vib, vstream);
  particle_kk->modify(Host,PARTICLE_MASK|CUSTOM_MASK);
}

