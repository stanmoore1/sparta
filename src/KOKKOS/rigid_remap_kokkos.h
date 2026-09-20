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
#include "rigid_body_kokkos.h"
#include "cut2d_kokkos.h"
#include "cut3d_kokkos.h"

namespace SPARTA_NS {

class RigidRemapKokkos : public RigidRemap {
 public:
  RigidRemapKokkos(class SPARTA *, class FixRigidKokkos *);
  ~RigidRemapKokkos() {}
  void collision_lists() override;
  void reset_collision_lists() override;
  int recut() override;

 private:
  class FixRigidKokkos *fix_kk;

  // (body, bin) work pairs over the bins of the cell index each body's
  //   swept box overlaps, and each body's bin range for the first-bin
  //   rule which lists a cell once per body

  DAT::tdual_int_1d k_pairbody,k_pairbin;
  DAT::tdual_int_2d k_qlo,k_qhi;
  int maxpair;

  // the swept lists: per cell its count of swept elements and its row
  //   (-1 = none), reset each step for the cells the previous step
  //   touched; per touched cell its cell, row offset, atomic cursor and
  //   # of entries which survive the merge with the cut list; per hit
  //   of the count pass its cell and element, and the row entries
  //   (element indices, then the surviving local surf indices)

  DAT::t_int_1d d_swcount,d_swrow;
  DAT::t_int_1d d_swtouched,d_swoff,d_swcursor,d_swext;
  DAT::t_int_1d d_hitcell,d_hitelem,d_swelem;
  DAT::t_int_scalar d_ntouched,d_nhit;
  int maxswcell_kk,maxtouched_kk,maxhit_kk;
  int ntouched_prev;            // # of cells the previous step touched

  // the re-cut: per body its region, COM, radii and interior flag; the
  //   candidate cells; per candidate its new list, whether it changed
  //   and its new type; the changed lists packed for the host

  typedef RigidBodyKK::tdual_dbl_2d tdual_dbl_2d;
  tdual_dbl_2d k_bodyparam;
  DAT::tdual_int_1d k_cominside,k_staticinside;
  int staticgen_kk;
  DAT::t_int_1d d_candflag,d_candoff;
  int maxrcand_kk;
  DAT::tdual_int_1d k_rcand,k_newlist,k_newtype;
  DAT::t_int_1d d_newn,d_chflag,d_choff,d_newtype;
  int maxrcandlist_kk;
  // the changed lists as rows of one entries array, and the results of
  //   their cuts: per cell # of pieces, corner marks, split point and
  //   piece, per entry the piece map and piece volumes

  typedef Kokkos::DualView<double*,DeviceType::array_layout,DeviceType> tdual_dbl_1d;
  typedef Kokkos::View<Cut2dKokkos::Cline*,DeviceType> t_cline_1d;
  typedef Kokkos::View<Cut2dKokkos::Point*,DeviceType> t_point_1d;
  typedef Kokkos::View<Cut2dKokkos::Loop*,DeviceType> t_loop_1d;
  typedef Kokkos::View<Cut2dKokkos::PG*,DeviceType> t_pg_1d;

  typedef Kokkos::View<Cut3dKokkos::Vertex*,DeviceType> t_vertex_1d;
  typedef Kokkos::View<Cut3dKokkos::Edge*,DeviceType> t_edge_1d;
  typedef Kokkos::View<Cut3dKokkos::Loop*,DeviceType> t_loop3_1d;
  typedef Kokkos::View<Cut3dKokkos::PH*,DeviceType> t_ph_1d;

  DAT::tdual_int_1d k_chcand,k_chn,k_chloff,k_chlist;
  DAT::tdual_int_1d k_chnsplit,k_chcorner,k_chxsub,k_cherr,k_chmap;
  tdual_dbl_1d k_chxsplit,k_chvols;
  int maxch_kk,maxchent_kk;

  t_cline_1d d_clines;       // scratch rows of the device cut
  t_point_1d d_points;
  t_loop_1d d_loops;
  t_pg_1d d_pgs;
  DAT::t_int_1d d_used;
  t_vertex_1d d_verts;
  t_edge_1d d_edges;
  t_loop3_1d d_loops3;
  t_ph_1d d_phs;
  DAT::t_int_1d d_facelist,d_efaces,d_used3,d_stack;
  int maxvert_kk,maxedge_kk,maxcline_kk,maxpt_kk;
  DAT::t_int_1d d_cutstats;  // tiny edge and shrink counts of the 3d cut

  void grow_cut_scratch(int, int, int, int);
};

}

#endif
