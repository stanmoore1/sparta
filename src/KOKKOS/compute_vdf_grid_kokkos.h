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

#ifdef COMPUTE_CLASS

ComputeStyle(vdf/grid/kk,ComputeVDFGridKokkos)

#else

#ifndef SPARTA_COMPUTE_VDF_GRID_KOKKOS_H
#define SPARTA_COMPUTE_VDF_GRID_KOKKOS_H

#include "compute_vdf_grid.h"
#include "kokkos_type.h"
#include "kokkos_base.h"

namespace SPARTA_NS {

template<int NEED_ATOMICS>
struct TagComputeVDFGrid_compute_per_grid_atomic{};

struct TagComputeVDFGrid_compute_per_grid{};

class ComputeVDFGridKokkos : public ComputeVDFGrid, public KokkosBase {
 public:
  typedef DeviceType::execution_space device_type;

  ComputeVDFGridKokkos(class SPARTA *, int, char **);
  ~ComputeVDFGridKokkos();
  void init();
  void compute_per_grid();
  void compute_per_grid_kokkos();
  void reallocate();

  template<int NEED_ATOMICS>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeVDFGrid_compute_per_grid_atomic<NEED_ATOMICS>,
                  const int &) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeVDFGrid_compute_per_grid, const int &) const;

 private:
  double mvv2e;
  int useweight;

  // per-value binning parameters, mirrored to the device

  DAT::t_int_1d d_value;
  DAT::t_int_1d d_nbin;
  DAT::t_int_1d d_binoffset;
  DAT::t_float_1d d_lo;
  DAT::t_float_1d d_hi;
  DAT::t_float_1d d_invdelta;

  DAT::tdual_float_2d_lr k_array_grid;

  int need_dup;
  Kokkos::Experimental::ScatterView<F_FLOAT**, typename DAT::t_float_2d_lr::array_layout,DeviceType,typename Kokkos::Experimental::ScatterSum,typename Kokkos::Experimental::ScatterDuplicated> dup_array_grid;
  Kokkos::Experimental::ScatterView<F_FLOAT**, typename DAT::t_float_2d_lr::array_layout,DeviceType,typename Kokkos::Experimental::ScatterSum,typename Kokkos::Experimental::ScatterNonDuplicated> ndup_array_grid;

  t_particle_1d d_particles;
  t_species_1d d_species;
  DAT::t_int_2d d_s2g;

  DAT::t_int_1d d_cellcount;
  DAT::t_int_2d d_plist;
  t_cinfo_1d d_cinfo;

  // shared by both kernels: value of one sample for value m of particle i

  KOKKOS_INLINE_FUNCTION
  double sample_of(const int i, const int m, const double mass) const;

  // bin index for a sample, or -1 if it is out of range and discarded

  KOKKOS_INLINE_FUNCTION
  int bin_of(const double sample, const int m) const;
};

}

#endif
#endif

/* ERROR/WARNING messages:

*/
