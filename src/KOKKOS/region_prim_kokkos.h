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

#ifndef SPARTA_REGION_PRIM_KOKKOS_H
#define SPARTA_REGION_PRIM_KOKKOS_H

#include "kokkos_type.h"

namespace SPARTA_NS {

// Region::inside() is a host virtual, and virtual dispatch is not available
//   inside a device kernel.  Rather than have every consumer carry a KKCopy
//   of each concrete region type and switch on a style string -- which is
//   what update/emit used to do, and what capped the number of regions a
//   run could use -- each Kokkos region flattens itself into a small array
//   of these PODs, which a kernel can walk with no dispatch at all.
// a primitive flattens to one entry; region union and region intersect
//   flatten to one entry per sub-region plus the combining op below.

enum{RKK_BLOCK,RKK_CYLINDER,RKK_PLANE,RKK_SPHERE};
enum{RKK_OP_NONE,RKK_OP_UNION,RKK_OP_INTERSECT};

struct RegionPrimKK {
  int style;              // one of RKK_*
  int interior;           // this sub-region's own interior/exterior sense
  int axis;               // cylinder only: 0/1/2 for x/y/z

  // style-specific parameters, packed:
  //   BLOCK     a..f = xlo,xhi,ylo,yhi,zlo,zhi
  //   CYLINDER  a,b = c1,c2   c = radius   d,e = lo,hi
  //   PLANE     a,b,c = point   n0,n1,n2 = normal
  //   SPHERE    a,b,c = center  d = radius

  double a,b,c,d,e,f;
  double n0,n1,n2;
};

typedef Kokkos::DualView<RegionPrimKK*, DeviceType::array_layout, DeviceType>
  tdual_region_prim_1d;
typedef tdual_region_prim_1d::t_dev t_region_prim_1d;

/* ----------------------------------------------------------------------
   does x,y,z match a single flattened sub-region
   mirrors Region::match(): !(inside ^ interior)
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
int region_prim_match_kk(const RegionPrimKK &p,
                         const double x, const double y, const double z)
{
  int inside = 0;

  if (p.style == RKK_BLOCK) {
    if (x >= p.a && x <= p.b && y >= p.c && y <= p.d && z >= p.e && z <= p.f)
      inside = 1;

  } else if (p.style == RKK_CYLINDER) {
    double del1,del2;
    if (p.axis == 0) { del1 = y - p.a; del2 = z - p.b; }
    else if (p.axis == 1) { del1 = x - p.a; del2 = z - p.b; }
    else { del1 = x - p.a; del2 = y - p.b; }
    const double dist = sqrt(del1*del1 + del2*del2);
    const double along = (p.axis == 0) ? x : ((p.axis == 1) ? y : z);
    if (dist <= p.c && along >= p.d && along <= p.e) inside = 1;

  } else if (p.style == RKK_PLANE) {
    const double dot = (x-p.a)*p.n0 + (y-p.b)*p.n1 + (z-p.c)*p.n2;
    if (dot >= 0.0) inside = 1;

  } else {   // RKK_SPHERE
    const double delx = x - p.a;
    const double dely = y - p.b;
    const double delz = z - p.c;
    if (sqrt(delx*delx + dely*dely + delz*delz) <= p.d) inside = 1;
  }

  return !(inside ^ p.interior);
}

/* ----------------------------------------------------------------------
   does x,y,z match a flattened region: N sub-regions combined by OP,
     then the composite's own interior/exterior sense applied
   OP == RKK_OP_NONE means a single primitive, whose sense is already in it
------------------------------------------------------------------------- */

template<class ViewType>
KOKKOS_INLINE_FUNCTION
int region_match_kk(const ViewType &d_prims, const int nprim, const int op,
                    const int interior,
                    const double x, const double y, const double z)
{
  if (op == RKK_OP_NONE) return region_prim_match_kk(d_prims[0],x,y,z);

  int hit;
  if (op == RKK_OP_UNION) {
    hit = 0;
    for (int i = 0; i < nprim; i++)
      if (region_prim_match_kk(d_prims[i],x,y,z)) { hit = 1; break; }
  } else {
    hit = 1;
    for (int i = 0; i < nprim; i++)
      if (!region_prim_match_kk(d_prims[i],x,y,z)) { hit = 0; break; }
  }

  return !(hit ^ interior);
}

}

#endif
