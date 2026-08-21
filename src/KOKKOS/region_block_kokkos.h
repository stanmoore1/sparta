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

#ifdef REGION_CLASS

RegionStyle(block/kk,RegBlockKokkos)

#else

#ifndef SPARTA_REGION_BLOCK_KOKKOS_H
#define SPARTA_REGION_BLOCK_KOKKOS_H

#include "region_block.h"

#include "kokkos_base.h"
#include "kokkos_type.h"

namespace SPARTA_NS {

struct TagRegBlockMatchAll{};

class RegBlockKokkos : public RegBlock, public KokkosBase {

 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  RegBlockKokkos(class SPARTA *, int, char **);
  RegBlockKokkos(class SPARTA *sparta);
  ~RegBlockKokkos() override;

  void match_all_kokkos(DAT::tdual_int_1d) override;

  // flatten to a single device-resident descriptor; see region_prim_kokkos.h

  int flatten_region_kokkos(tdual_region_prim_1d &k_prims, int &op) override
  {
    if ((int) k_prims.extent(0) < 1)
      k_prims = tdual_region_prim_1d("region:prims",1);
    RegionPrimKK &p = k_prims.view_host()[0];
    p.style = RKK_BLOCK;
    p.interior = interior;
    p.axis = 0;
    p.a = p.b = p.c = p.d = p.e = p.f = 0.0;
    p.n0 = p.n1 = p.n2 = 0.0;
    p.a = xlo; p.b = xhi; p.c = ylo; p.d = yhi; p.e = zlo; p.f = zhi;
    k_prims.modify_host();
    k_prims.sync_device();
    op = RKK_OP_NONE;
    return 1;
  }


  KOKKOS_INLINE_FUNCTION
  void operator()(TagRegBlockMatchAll, const int&) const;

  KOKKOS_INLINE_FUNCTION
  int match_kokkos(double x, double y, double z) const
  {
    return !(k_inside(x,y,z) ^ interior);
  }

 private:
  int groupbit;
  typename AT::t_int_1d d_match;
  t_particle_1d d_particles;

  KOKKOS_INLINE_FUNCTION
  int k_inside(double x, double y, double z) const
  {
    if (x >= xlo && x <= xhi && y >= ylo && y <= yhi && z >= zlo && z <= zhi)
      return 1;
    return 0;
  }
};

}

#endif
#endif

