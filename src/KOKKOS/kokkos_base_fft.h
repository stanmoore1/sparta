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

#ifndef SPARTA_KOKKOS_BASE_FFT_H
#define SPARTA_KOKKOS_BASE_FFT_H

#include "fftdata_kokkos.h"

namespace SPARTA_NS {

class KokkosBaseFFT {
 public:
  KokkosBaseFFT() {}

  //Kspace
  virtual void pack_forward_kspace_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, DAT::tdual_int_2d &, int) {};
  virtual void unpack_forward_kspace_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, DAT::tdual_int_2d &, int) {};
  virtual void pack_reverse_kspace_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, DAT::tdual_int_2d &, int) {};
  virtual void unpack_reverse_kspace_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, DAT::tdual_int_2d &, int) {};
};

}

#endif

/* ERROR/WARNING messages:

*/
