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

#ifdef COMPUTE_CLASS

ComputeStyle(fft/grid/kk,ComputeFFTGridKokkos)

#else

#ifndef SPARTA_COMPUTE_FFT_GRID_KOKKOS_H
#define SPARTA_COMPUTE_FFT_GRID_KOKKOS_H

#include "compute_fft_grid.h"
#include "fftdata_kokkos.h"

namespace SPARTA_NS {

struct TagComputeFFTGrid_zero_array_grid{};
struct TagComputeFFTGrid_zero_vector_grid{};
struct TagComputeFFTGrid_fill_fft{};
struct TagComputeFFTGrid_copy_from_tmp{};
struct TagComputeFFTGrid_compute_norm_sq{};
struct TagComputeFFTGrid_update_conjugate{};
struct TagComputeFFTGrid_update_complex{};
struct TagComputeFFTGrid_scale_grid{};

class ComputeFFTGridKokkos : public ComputeFFTGrid {

  typedef Kokkos::RangePolicy< SPADeviceType, TagComputeFFTGrid_copy_from_tmp > copy_from_tmp;

  typedef Kokkos::RangePolicy< SPAHostType, TagComputeFFTGrid_zero_array_grid > zero_array_grid;
  typedef Kokkos::RangePolicy< SPAHostType, TagComputeFFTGrid_zero_vector_grid > zero_vector_grid;

  typedef Kokkos::RangePolicy< SPADeviceType, TagComputeFFTGrid_fill_fft > fill_fft;

  typedef Kokkos::RangePolicy< SPADeviceType, TagComputeFFTGrid_update_complex > update_complex;
  typedef Kokkos::RangePolicy< SPADeviceType, TagComputeFFTGrid_update_conjugate > update_conjugate;
  typedef Kokkos::RangePolicy< SPADeviceType, TagComputeFFTGrid_compute_norm_sq > compute_norm_sq;

  typedef Kokkos::RangePolicy< SPADeviceType, TagComputeFFTGrid_scale_grid > scale_grid;

  public:

    ComputeFFTGridKokkos(class SPARTA *, int, char **);
    ~ComputeFFTGridKokkos();
    void compute_per_grid();
    void reallocate();

    KOKKOS_INLINE_FUNCTION
    void operator()(TagComputeFFTGrid_zero_array_grid, const int &) const;

    KOKKOS_INLINE_FUNCTION
    void operator()(TagComputeFFTGrid_zero_vector_grid, const int &) const;

    KOKKOS_INLINE_FUNCTION
    void operator()(TagComputeFFTGrid_copy_from_tmp, const int &) const;

    KOKKOS_INLINE_FUNCTION
    void operator()(TagComputeFFTGrid_fill_fft, const int &) const;

    KOKKOS_INLINE_FUNCTION
    void operator()(TagComputeFFTGrid_update_complex, const int &) const;

    KOKKOS_INLINE_FUNCTION
    void operator()(TagComputeFFTGrid_update_conjugate, const int &) const;

    KOKKOS_INLINE_FUNCTION
    void operator()(TagComputeFFTGrid_compute_norm_sq, const int &) const;

    KOKKOS_INLINE_FUNCTION
    void operator()(TagComputeFFTGrid_scale_grid, const int &) const;

  private:

    int offset;

    // complex buf for performing FFT, length = nfft
    FFT_DAT::tdual_FFT_SCALAR_1d k_fft;


    // work buf in grid decomp, length = nglocal
    FFT_DAT::tdual_FFT_SCALAR_1d k_gridworkcomplex;


    // mapping of received SPARTA grid values to FFT grid
    // map1[i] = index into ordered FFT grid of 
    //           Ith value in buffer received
    //           from SPARTA decomp via irregular comm
    DAT::tdual_int_1d k_map1;


    // mapping of received FFT grid values to SPARTA grid
    // map2[i] = index into SPARTA grid of Ith value
    //           in buffer received from FFT decomp via
    //           irregular comm
    DAT::tdual_int_1d k_map2;


    // input grid values from compute,fix,variable
    // may be NULL if ingridptr just points to c/f/v
    DAT::tdual_float_1d k_ingrid;


    // work buf in FFT decomp, length = nfft
    DAT::tdual_float_1d k_fftwork;


    // work buf in grid decomp, length = nglocal
    DAT::tdual_float_1d k_gridwork;


    DAT::tdual_float_2d k_array_grid;
    DAT::tdual_float_1d k_vector_grid;

    DAT::t_float_2d k_tmp;

    void irregular_create();

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: Compute grid mixture ID does not exist

Self-explanatory.

E: Number of groups in compute grid mixture has changed

This mixture property cannot be changed after this compute command is
issued.

E: Invalid call to ComputeGrid::post_process_grid()

This indicates a coding error.  Please report the issue to the SPARTA
developers.

*/
