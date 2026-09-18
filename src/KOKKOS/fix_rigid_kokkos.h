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

#ifdef FIX_CLASS

FixStyle(rigid/kk,FixRigidKokkos)

#else

#ifndef SPARTA_FIX_RIGID_KOKKOS_H
#define SPARTA_FIX_RIGID_KOKKOS_H

#include "fix_rigid.h"
#include "kokkos_type.h"
#include "particle_kokkos.h"
#include "grid_kokkos.h"

namespace SPARTA_NS {

struct TagFixRigidRemoveInside{};
struct TagFixRigidCombineSplit{};
struct TagFixRigidAssignSplit{};
struct TagFixRigidSumTallies{};
struct TagFixRigidCellMapInit{};
struct TagFixRigidCellMapSet{};
struct TagFixRigidRelabel{};

class FixRigidKokkos : public FixRigid {
 public:
  typedef DeviceType::execution_space device_type;
  typedef int value_type;

  FixRigidKokkos(class SPARTA *, int, char **);
  ~FixRigidKokkos();
  void init();
  void setup();
  void start_of_step();
  void end_of_step();
  void grid_changed();
  void surf_maps();
  void remove_inside_all(int);
  void particles_to_host();
  void combine_split_all();
  void sort_for_split_rebuild();
  void relabel_moved_cells();
  void sum_tallies();

  // flag particles inside a body, one thread per particle
  // the reduction value is the # of particles this body claimed, which
  //   FixRigid counts in ndeleted/ndelrun

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixRigidRemoveInside, const int&, int&) const;

  // relabel the particles of every sub cell of a changed split cell to the
  //   split cell itself, one thread per (split cell, sub cell) pair

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixRigidCombineSplit, const int&) const;

  // re-decide the sub cell of every particle of a changed split cell, one
  //   thread per particle of the split cell

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixRigidAssignSplit, const int&) const;

  // per-body sums of the mover's per-surf tallies, one thread per body

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixRigidSumTallies, const int&) const;

  // old -> new cell index map of the cells a restructure moved, and
  //   its application to every particle's cell label

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixRigidCellMapInit, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixRigidCellMapSet, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixRigidRelabel, const int&) const;

  // device split2d/split3d, the same tests as Update::split2d/split3d

  KOKKOS_INLINE_FUNCTION
  int split2d_kk(int, double*) const;
  KOKKOS_INLINE_FUNCTION
  int split3d_kk(int, double*) const;

 private:
  void host_begin();
  void host_end();

  // device port of FixRigid::remove_inside_all(), which is otherwise a
  //   host pass over every particle and so forces the particle array to
  //   the host and back on every step
  // returns 1 if it handled the deletion, 0 to fall back to the host

  int remove_inside_all_kokkos(int);
  void pack_body_device();

  // device replacement for the host
  //   sort + combine_split_cell_particles() pass in end_of_step(), which is
  //   the last thing forcing the particle array to the host every step
  // returns 1 if it handled it, 0 to leave it to the host

  int combine_split_kokkos();

  // device replacement for the host assign_split_cell_particles() pass in
  //   remove_inside_all_kokkos(), the last per-step host particle consumer

  int assign_split_kokkos();

  // the split cells to re-assign, and a flat (cell,slot) work list so one
  //   thread handles one particle

  DAT::tdual_int_1d k_asgcell,k_asgpart;
  DAT::t_int_1d d_asgcell,d_asgpart;
  int nasg_kk;

  // grid/surf device views the split tests read

  t_cell_1d d_cells_kk;
  t_sinfo_1d d_sinfo_kk;
  t_line_1d d_lines_kk;
  t_tri_1d d_tris_kk;
  Kokkos::Crs<int,DeviceType,void,crs_size_type> d_csurfs_kk,d_csplits_kk,d_csubs_kk;

  // one entry per sub cell of a changed split cell: which sub cell to scan
  //   and which split cell to relabel its particles to

  DAT::tdual_int_1d k_subcell,k_subparent;
  DAT::t_int_1d d_subcell,d_subparent;
  int nsub_kk;

  DAT::t_int_1d d_plist_kk;     // per-cell particle lists, from sort_kokkos
  DAT::t_int_2d d_plist2_kk;
  DAT::t_int_1d d_cellcount_kk;

  int dim_kk;                   // dim, captured for the kernel
  int nplocal_kk;

  // per-element body geometry, flattened for the device
  //   bodypt[i][j][k] -> d_bodypt(i,j,k), bodynorm[i][k] -> d_bodynorm(i,k)
  // explicitly double, not SPARTA_FLOAT: the inside test is a ray cast
  //   whose parity must match the host result exactly, and a single
  //   precision build (SPA_PRECISION 1) would change which side of an
  //   element a nearly tangent ray falls on

  typedef Kokkos::DualView<double***,DeviceType::array_layout,DeviceType> tdual_dbl_3d;
  typedef Kokkos::DualView<double**,DeviceType::array_layout,DeviceType> tdual_dbl_2d;

 public:

  // per local+ghost surf: force/torque (fx,fy,fz,tx,ty,tz) tallied by
  //   the KOKKOS move kernel for this step's collisions, zeroed in
  //   start_of_step(), read by UpdateKokkos::rigid_upload()

  tdual_dbl_2d k_ftally;
  tdual_dbl_2d::t_dev d_ftally;

 private:

  // per body: its local+ghost surfs as a CSR list in surf index order,
  //   built with the per-surf maps, and the per-body sums of the tallies

  DAT::tdual_int_1d k_bodysurfstart,k_bodysurflist;
  DAT::t_int_1d d_bodysurfstart,d_bodysurflist;
  tdual_dbl_2d k_ft;
  tdual_dbl_2d::t_dev d_ft;

  DAT::t_int_1d d_cellmap;
  DAT::tdual_int_1d k_movedfrom,k_movedto;
  DAT::t_int_1d d_movedfrom,d_movedto;

  tdual_dbl_3d k_bodypt;
  tdual_dbl_2d k_bodynorm;
  tdual_dbl_2d k_bbodylo,k_bbodyhi;
  DAT::tdual_int_1d k_bodystart;
  DAT::tdual_int_1d k_bodybinstart,k_bodybinlist;

  tdual_dbl_3d::t_dev d_bodypt;
  tdual_dbl_2d::t_dev d_bodynorm;
  tdual_dbl_2d::t_dev d_bbodylo,d_bbodyhi;
  DAT::t_int_1d d_bodystart;
  DAT::t_int_1d d_bodybinstart,d_bodybinlist;

  int nelem_kk;                 // # of body elements packed
  int nbin_kk;                  // # of body bins packed
  int bodynbin_kk[3];
  double bodybinlo_kk[3],bodybininv_kk[3];
  double rmaxall_kk;

  // deletion list, built on device exactly as collide/kk builds its own

  DAT::tdual_int_1d k_dellist_kk;
  DAT::t_int_1d d_dellist_kk;
  int maxdelete_kk;
  DAT::t_int_scalar d_ndelete_kk;
  HAT::t_int_scalar h_ndelete_kk;

  t_particle_1d d_particles_kk;
  t_cinfo_1d d_cinfo_kk;        // the device grid, current after apply_changes()
  int nlocal_kk;                // grid->nlocal, cinfo has no ghost rows
};

}

#endif
#endif

/* ERROR/WARNING messages:

*/
