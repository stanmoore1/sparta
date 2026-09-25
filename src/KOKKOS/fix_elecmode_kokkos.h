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

#ifdef FIX_CLASS

FixStyle(elecmode/kk,FixElecmodeKokkos)

#else

#ifndef SPARTA_FIX_ELECMODE_KOKKOS_H
#define SPARTA_FIX_ELECMODE_KOKKOS_H

#include "fix_elecmode.h"
#include "kokkos_type.h"
#include "particle_kokkos.h"
#include "Kokkos_Random.hpp"
#include "rand_pool_wrap.h"

namespace SPARTA_NS {

class FixElecmodeKokkos : public FixElecmode {
 public:
  FixElecmodeKokkos(class SPARTA *, int, char **);
  FixElecmodeKokkos(class SPARTA *sparta);
  ~FixElecmodeKokkos();
  void pre_update_custom_kokkos();
  void update_custom(int, const TempsInfo &, double *);

  KOKKOS_INLINE_FUNCTION
  void update_custom_kokkos(int, double, double, double, double, const double *) const;

  KOKKOS_INLINE_FUNCTION
  void surf_react_kokkos(Particle::OnePart *, int) const;

 private:
  double boltz;
  int elecstyle;

#ifndef SPARTA_KOKKOS_EXACT
  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;

  //Kokkos::Random_XorShift1024_Pool<DeviceType> rand_pool;
  //typedef typename Kokkos::Random_XorShift1024_Pool<DeviceType>::generator_type rand_type;
#else
  RandPoolWrap rand_pool;
  typedef RandWrap rand_type;
#endif

  t_particle_1d d_particles;
  t_species_1d d_species;

  DAT::t_float_1d d_eelec;
  DAT::t_int_1d d_elecstate;
  DAT::t_int_1d d_nelecstates;
  t_elecstate_2d d_elecstates;

  KOKKOS_INLINE_FUNCTION
  int ielec(int, double, rand_type &) const;
};

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
int FixElecmodeKokkos::ielec(int isp, double temp_elec, rand_type &erandom) const
{
  enum{NONE,DISCRETE,SMOOTH};            // several files

  int ielec = 0;

  if (elecstyle == DISCRETE) {
    const int nelecstate = d_nelecstates[isp];
    if (!nelecstate) return 0;

    // sample the Boltzmann distribution over the species' states, with the
    // cumulative probabilities evaluated on the fly (no per-particle
    // scratch array); same arithmetic as the CPU FixElecmode, so the
    // selected state is identical for the same random number

    double partition_function = 0.0;
    for (int i = 0; i < nelecstate; ++i)
      partition_function +=
        d_elecstates(isp,i).degen*exp(-d_elecstates(isp,i).temp/temp_elec);

    double ran = erandom.drand();
    double cumulative = 0.0;
    ielec = 0;
    // bound the search: floating-point roundoff can leave ran above the
    // final cumulative entry, which would index past the last state
    while (ielec < nelecstate-1) {
      cumulative += d_elecstates(isp,ielec).degen *
        exp(-d_elecstates(isp,ielec).temp/temp_elec) / partition_function;
      if (!(ran > cumulative)) break;
      ++ielec;
    }
  }
  return ielec;
}

/* ----------------------------------------------------------------------
   called when a particle with index is created
    or when temperature dependent properties need to be updated
   populate an electronic state and set eelec
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixElecmodeKokkos::update_custom_kokkos(int index, double temp_thermal,
                                            double temp_rot, double temp_vib,
                                            double temp_elec, const double *) const
{
  int isp = d_particles[index].ispecies;
  int nstate = d_nelecstates[isp];

  // if no states, just return

  if (nstate == 0) return;

  rand_type rand_gen = rand_pool.get_state();

  d_elecstate[index] = ielec(isp,temp_elec,rand_gen);
  d_eelec[index] = boltz*d_elecstates(isp,d_elecstate[index]).temp;

  rand_pool.free_state(rand_gen);
}

/* ----------------------------------------------------------------------
   device version of FixElecmode::surf_react(), for surface collision
     models that do not resample internal energy (specular, piston):
   after a surface reaction changed particle I's species, keep its state
     index only if the new species has that state, else reset it to the
     ground state, and set eelec to the energy of that state
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixElecmodeKokkos::surf_react_kokkos(Particle::OnePart *iorig, int i) const
{
  if (i < 0) return;

  const int isp = d_particles[i].ispecies;
  if (iorig->ispecies == isp) return;

  const int nstate = d_nelecstates[isp];
  if (nstate == 0) {
    d_elecstate[i] = 0;
    d_eelec[i] = 0.0;
    return;
  }

  if (d_elecstate[i] >= nstate) d_elecstate[i] = 0;
  d_eelec[i] = boltz*d_elecstates(isp,d_elecstate[i]).temp;
}

}

#endif
#endif
