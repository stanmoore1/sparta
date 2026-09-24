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
#include "rigid_body_kokkos.h"
#include "rigid_remap_kokkos.h"

namespace SPARTA_NS {

struct TagFixRigidRemoveInside{};
struct TagFixRigidCombineSplit{};
struct TagFixRigidAssignSplit{};
struct TagFixRigidSumTallies{};
struct TagFixRigidCellMapInit{};
struct TagFixRigidCellMapSet{};
struct TagFixRigidRelabel{};
struct TagFixRigidGeometry{};
struct TagFixRigidBodyGroup{};
struct TagFixRigidZeroTally{};
struct TagFixRigidScatterSurfs{};

// the box of a body's element boxes, current and end-of-step, as one
//   team reduction: lo[3], hi[3], then the end-of-step lo[3], hi[3]
// min and max are exact in any order, so the result is the one the
//   serial loop of FixRigid::body_bbox() gets

struct RigidBoxVal {
  double v[12];
};

struct RigidBoxReducer {
  typedef RigidBoxReducer reducer;
  typedef RigidBoxVal value_type;
  typedef Kokkos::View<value_type,Kokkos::HostSpace,
                       Kokkos::MemoryUnmanaged> result_view_type;
  value_type &value;

  KOKKOS_INLINE_FUNCTION
  RigidBoxReducer(value_type &v) : value(v) {}

  KOKKOS_INLINE_FUNCTION
  void join(value_type &dst, const value_type &src) const {
    for (int k = 0; k < 3; k++) {
      dst.v[k] = MIN(dst.v[k],src.v[k]);
      dst.v[3+k] = MAX(dst.v[3+k],src.v[3+k]);
      dst.v[6+k] = MIN(dst.v[6+k],src.v[6+k]);
      dst.v[9+k] = MAX(dst.v[9+k],src.v[9+k]);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type &val) const {
    for (int k = 0; k < 3; k++) {
      val.v[k] = val.v[6+k] = 1.0e20;
      val.v[3+k] = val.v[9+k] = -1.0e20;
    }
  }

  KOKKOS_INLINE_FUNCTION
  value_type &reference() const { return value; }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return result_view_type(&value,1); }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return true; }
};

// the running (sum, max) of the split-assign scan: the offsets of the
//   work list and the largest per-cell particle count, in one pass

struct RigidSumMax {
  int sum,mx;
  KOKKOS_INLINE_FUNCTION RigidSumMax() : sum(0), mx(0) {}
  KOKKOS_INLINE_FUNCTION RigidSumMax &operator+=(const RigidSumMax &o) {
    sum += o.sum;
    if (o.mx > mx) mx = o.mx;
    return *this;
  }
};

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
  void grid_rebuild();
  void grid_changed();
  void post_run();
  void surf_maps();
  void set_xv();
  void swept_boxes();
  void host_geometry(int);
  void refresh_host_surfs();
  int host_cells_needed();
  void refresh_all();
  void newghost_geometry();
  void stage_fence() override { Kokkos::fence(); }
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
  //   this proc holds

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

  // the body geometry from the pose, over the bodies this proc holds:
  //   one thread per local element (points, normal, box, and on a sweep
  //   the zero of its tally row), then a team per held body (its bbox
  //   and inflation, then a thread per element group inflating the
  //   group's element boxes and boxing them), and one thread per local
  //   surf copy (the surf itself)

  typedef Kokkos::TeamPolicy<DeviceType,TagFixRigidBodyGroup> t_bodygroup_policy;
  typedef t_bodygroup_policy::member_type t_bodygroup_member;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixRigidGeometry, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixRigidBodyGroup, const t_bodygroup_member&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixRigidZeroTally, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixRigidScatterSurfs, const int&) const;

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

  // device replacement for the host
  //   sort + combine_split_cell_particles() pass in end_of_step(), which is
  //   the last thing forcing the particle array to the host every step
  // returns 1 if it handled it, 0 to leave it to the host

  int combine_split_kokkos();

 public:

  // the replicated body table on the device, read by the kernels here
  //   and by RigidRemapKokkos: the element tables from pack_body_static(),
  //   the geometry and boxes from device_geometry() (current element
  //   boxes after set_xv(), the swept ones of the step after
  //   swept_boxes()), the bins of bodies by COM from pack_body_device()

  void pack_body_device();
  RigidBodyKK body;

  typedef RigidBodyKK::tdual_dbl_3d tdual_dbl_3d;
  typedef RigidBodyKK::tdual_dbl_2d tdual_dbl_2d;
  typedef Kokkos::DualView<double*,DeviceType::array_layout,DeviceType> tdual_dbl_1d;

 private:
  void pack_body_static();      // element tables to the device
  void pack_body_geometry();    // host geometry to the device, at setup
  void pack_body_lists();       // blist/its elements/bodystatus, on change
  void geometry_views();        // the device views the kernels below read
  void commit_geometry();       // the swept pass's end-of-step geometry
  void bbox_to_host(tdual_dbl_2d &);   // one read-back of the body boxes
  void device_geometry(int);    // geometry and boxes from the pose

  // per element: displace in the body frame and the body; per local
  //   surf copy: its surf index and element, and with distributed surfs
  //   per owned copy its owned index and element; per body: the pose of
  //   this step (xcm, ex, ey, ez, omega, rmax) and the box inflation
  // hostgeom = per body, 1 if the host copy of its geometry is current

  tdual_dbl_3d k_displace;
  DAT::tdual_int_1d k_body,k_copy_index,k_copy_elem,k_olist_own,k_olist_elem;
  tdual_dbl_2d k_pose;
  tdual_dbl_1d k_bboxeps;
  int ncopy_kk,nolist_kk;
  int *hostgeom;
  int maxhostgeom;
  int devicegeom;               // 1 once setup() put the geometry on the device

  tdual_dbl_3d::t_dev d_displace_kk,d_bodypt_kk;
  tdual_dbl_2d::t_dev d_bodynorm_kk,d_elemlo_kk,d_elemhi_kk,d_pose_kk;
  tdual_dbl_2d::t_dev d_elemglo_kk,d_elemghi_kk,d_glonew_kk,d_ghinew_kk;
  tdual_dbl_2d::t_dev d_bbodylo_kk,d_bbodyhi_kk;
  tdual_dbl_1d::t_dev d_bboxeps_kk;
  tdual_dbl_3d::t_dev d_ptnew_kk;
  tdual_dbl_2d::t_dev d_normnew_kk,d_ellonew_kk,d_elhinew_kk;
  tdual_dbl_2d::t_dev d_bblonew_kk,d_bbhinew_kk;
  tdual_dbl_1d::t_dev d_bbepsnew_kk;
  tdual_dbl_2d::t_dev d_bbox_kk,d_bboxnew_kk;
  DAT::t_int_1d d_lcopy_kk;
  DAT::t_int_1d d_body_kk,d_bodystart_kk,d_copy_index_kk,d_copy_elem_kk;
  DAT::t_int_1d d_olist_own_kk,d_olist_elem_kk;
  DAT::t_int_1d d_blist_kk,d_lelem_kk,d_bodystat_kk;
  DAT::t_int_1d d_lgroup_kk,d_groupelem_kk,d_groupstart_kk;
  t_line_1d d_mylines_kk;
  t_tri_1d d_mytris_kk;
  int nscatter_kk;              // the local copies come first in the scatter
  int sweep_kk,axiflag_kk;
  double dt_kk;

  // the zero of the held elements' tally rows rides on the sweep's
  //   geometry kernel: zerotally_kk tells the kernel, zeropending_kk is
  //   set at the start of a step until a kernel zeroed them, and
  //   zerogen_kk is the blistgen of the element list it zeroed

  int zerotally_kk,zeropending_kk,zerogen_kk;
  void body_group_boxes(int);

  KOKKOS_INLINE_FUNCTION
  void scatter_line(Surf::Line &, int) const;
  KOKKOS_INLINE_FUNCTION
  void scatter_tri(Surf::Tri &, int) const;


  // device replacement for the host assign_split_cell_particles() pass in
  //   remove_inside_all_kokkos(), the last per-step host particle consumer
  // public, and not private, only because nvcc refuses an extended
  //   __host__ __device__ lambda inside a private member function

 public:
  int assign_split_kokkos();
 private:

  // the split cells to re-assign, and a flat (cell,slot) work list so one
  //   thread handles one particle

  DAT::t_int_1d d_asgcell,d_asgpart;
  int nasg_kk;
  DAT::tdual_int_1d k_splitcells;    // the owned split cells
  DAT::t_int_1d d_splitcells,d_splitoff;
  int nsplit_kk;

  // grid/surf device views the split tests read

  t_cell_1d d_cells_kk;
  t_sinfo_1d d_sinfo_kk;
  t_line_1d d_lines_kk;
  t_tri_1d d_tris_kk;
  Kokkos::Crs<int,DeviceType,void,crs_size_type> d_csurfs_kk,d_csplits_kk,d_csubs_kk;

  // one entry per sub cell of a changed split cell: which sub cell to scan
  //   and which split cell to relabel its particles to

  DAT::tdual_int_1d k_subcell,k_subparent;
  int nsub_used;                // # of pairs of this step's combine
  int combined_kk;              // 1 while those pairs are current

  // the rows the assign pass redistributes after a device combine: the
  //   combine's pairs of the cells split now, plus the own row of a cell
  //   split now but not then; asgstamp marks the cells the pairs cover

  DAT::tdual_int_1d k_asgrowcell,k_asgrowparent;
  int maxasgrow;
  int *asgstamp;
  int maxasgstamp,asgcur;
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

 public:

  // per body element: force/torque (fx,fy,fz,tx,ty,tz) tallied by the
  //   KOKKOS move kernel for this step's collisions on any local copy of
  //   the element, zeroed in start_of_step(), summed by sum_tallies()

  tdual_dbl_2d k_ftally;
  tdual_dbl_2d::t_dev d_ftally;

 private:

  // per body: its element range, and the per-body sums of the tallies

  DAT::t_int_1d d_bodystart;
  tdual_dbl_2d k_ft;
  tdual_dbl_2d::t_dev d_ft;

  DAT::t_int_1d d_cellmap;
  DAT::tdual_int_1d k_movedfrom,k_movedto;
  DAT::t_int_1d d_movedfrom,d_movedto;

  tdual_dbl_3d k_bodypt;
  tdual_dbl_2d k_bodynorm;
  tdual_dbl_2d k_bbodylo,k_bbodyhi;
  tdual_dbl_2d k_elemlo,k_elemhi;

  // a box per group of elements, the device twin of FixRigid's: the
  //   per-cell scans of the remap skip a group whose box misses the
  //   cell.  groupelem is the element range of each group, a CSR over
  //   all bodies since the groups of a body partition its elements

  tdual_dbl_2d k_elemglo,k_elemghi;
  DAT::tdual_int_1d k_groupstart,k_groupelem;

  // the end-of-step geometry the sweep pass computes on its way to the
  //   swept boxes, committed by a handle swap in set_xv(): the geometry
  //   kernels run once per step, not twice
  // per body (lo,hi,eps) packed, so a stage reads one array back

  tdual_dbl_3d k_bodypt_new;
  tdual_dbl_2d k_bodynorm_new;
  tdual_dbl_2d k_bbodylo_new,k_bbodyhi_new;
  tdual_dbl_2d k_elemlo_new,k_elemhi_new;
  tdual_dbl_2d k_elemglo_new,k_elemghi_new;
  tdual_dbl_1d k_bboxeps_new;
  tdual_dbl_2d k_bbox,k_bbox_new;
  int newgeom;            // 1 if a sweep left an end-of-step geometry
  DAT::tdual_int_1d k_bodystart,k_lblist;
  DAT::tdual_int_1d k_bodybinstart,k_bodybinlist;

  // the bodies this proc holds and their elements, in blist order, plus
  //   the per-body status the surf scatter tests; re-packed when
  //   blistgen moved.  the new-ghost pre-pass runs the same kernels
  //   over its own short lists

  DAT::tdual_int_1d k_blist,k_lelem,k_bodystat,k_lgroup;

  // the surf copies of the bodies this proc holds, in the scatter's own
  //   numbering: the copies come first, the owned surfs of distributed
  //   surfs after them.  the surf arrays span every body, so without
  //   this the scatter is a pass over all of them on every proc

  DAT::tdual_int_1d k_lcopy;
  int nlcopy_kk;
  DAT::tdual_int_1d k_newblist,k_newelem,k_newgroup;
  int blistgen_kk;              // blistgen of the last upload, -1 = none
  int nlelem_kk;                // # of elements of the blist bodies
  int nlgroup_kk;               // # of their element groups

  int nelem_kk;                 // # of body elements packed
  int nbin_kk;                  // # of body bins packed
  int bodybingen_kk;            // bodybingen of the bins on the device

  // deletion list, built on device exactly as collide/kk builds its own

  DAT::tdual_int_1d k_dellist_kk;
  DAT::t_int_1d d_dellist_kk;
  int maxdelete_kk;
  DAT::t_int_scalar d_ndelete_kk;
  HAT::t_int_scalar h_ndelete_kk;

  t_particle_1d d_particles_kk;
  t_cinfo_1d d_cinfo_kk;        // the device grid, current after apply_changes()
  DAT::t_char_1d d_delflag_kk;  // per cell: 0 skip, 1 test, 2 inside a body
  int nlocal_kk;                // grid->nlocal, cinfo has no ghost rows
};

}

#endif
#endif

/* ERROR/WARNING messages:

*/
