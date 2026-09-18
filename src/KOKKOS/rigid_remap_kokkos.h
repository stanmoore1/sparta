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

#ifndef SPARTA_RIGID_REMAP_KOKKOS_H
#define SPARTA_RIGID_REMAP_KOKKOS_H

#include "rigid_remap.h"
#include "kokkos_type.h"

namespace SPARTA_NS {

class RigidRemapKokkos : public RigidRemap {
 public:
  RigidRemapKokkos(class SPARTA *, class FixRigidKokkos *);
  ~RigidRemapKokkos() {}
  void collision_lists() override;
  void reset_collision_lists() override;

 private:
  class FixRigidKokkos *fix_kk;

  // (body, bin) work pairs over the bins of the cell index each body's
  //   swept box overlaps, and each body's bin range for the first-bin
  //   rule which lists a cell once per body

  DAT::tdual_int_1d k_pairbody,k_pairbin;
  DAT::tdual_int_2d k_qlo,k_qhi;
  int maxpair;

  // per cell: # of swept elements, their offsets, an atomic cursor, the
  //   entries (element indices, then the surviving local surf indices)
  //   and the # which survive the merge with the cut list

  DAT::t_int_1d d_swcount,d_swoff,d_swcursor,d_swext,d_swelem,d_subparent;
  int maxswcell_kk,maxswent_kk;

  // the split cell of every sub cell, from the host split info

  DAT::tdual_int_1d k_subcell,k_subpar;

  // the mover's graph, built here each step

  Kokkos::View<crs_size_type*,DeviceType> d_rowmap_move;
  DAT::t_int_1d d_entries_move;
  Kokkos::View<crs_size_type*,DeviceType> d_rowcount;
};

}

#endif
