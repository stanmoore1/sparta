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
  void remove_inside_all(int);
  void particles_to_host();

  // flag particles inside a body, one thread per particle
  // the reduction value is the # of particles this body claimed, which
  //   FixRigid counts in ndeleted/ndelrun

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixRigidRemoveInside, const int&, int&) const;

 private:
  void host_begin();
  void host_end();

  // device port of FixRigid::remove_inside_all(), which is otherwise a
  //   host pass over every particle and so forces the particle array to
  //   the host and back on every step
  // returns 1 if it handled the deletion, 0 to fall back to the host

  int remove_inside_all_kokkos(int);
  void pack_body_device();

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
  DAT::t_int_1d d_celltype_kk;  // cinfo[icell].type per owned+ghost cell
  DAT::t_int_1d d_cellnsurf_kk; // cells[icell].nsurf
  DAT::tdual_int_1d k_celltype_kk,k_cellnsurf_kk;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Fix rigid/kk requires compute surf/kk

The compute surf used by fix rigid must be the KOKKOS version, so that
the force/torque tallies are computed by the KOKKOS particle mover.

*/
