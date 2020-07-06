/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.sandia.gov
   Steve Plimpton, sjplimp@sandia.gov, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(vibmode/kk,FixVibmodeKokkos)

#else

#ifndef SPARTA_FIX_VIBMODE_KOKKOS_H
#define SPARTA_FIX_VIBMODE_KOKKOS_H

#include "stdio.h"
#include "fix_vibmode.h"

namespace SPARTA_NS {

class FixVibmodeKokkos : public FixVibmode {
 public:
  FixVibmodeKokkos(class SPARTA *, int, char **);
  ~FixVibmodeKokkos();
  void add_particle(int, double, double, double, double *);

 private:
#ifndef SPARTA_KOKKOS_EXACT
  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;

  //Kokkos::Random_XorShift1024_Pool<DeviceType> rand_pool;
  //typedef typename Kokkos::Random_XorShift1024_Pool<DeviceType>::generator_type rand_type;
#else
  RandPoolWrap rand_pool;
  typedef RandWrap rand_type;
#endif
};

}

#endif
#endif
