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

#ifndef SPARTA_CUT_KOKKOS_H
#define SPARTA_CUT_KOKKOS_H

#include "kokkos_type.h"

namespace SPARTA_NS {

// device twins of the surf/cell overlap tests of the cut classes, the
//   same arithmetic in the same order as Cut2d::cliptest() and
//   Cut3d::clip(), so a device decision agrees with the host's

namespace CutKokkos {

/* ----------------------------------------------------------------------
   1 if line segment PQ overlaps the cell LO/HI, else 0 (Cut2d::cliptest)
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
int cliptest2d(const double *p, const double *q,
               const double *lo, const double *hi)
{
  double x,y;

  if (p[0] >= lo[0] && p[0] <= hi[0] &&
      p[1] >= lo[1] && p[1] <= hi[1]) return 1;
  if (q[0] >= lo[0] && q[0] <= hi[0] &&
      q[1] >= lo[1] && q[1] <= hi[1]) return 1;

  double a[2],b[2];
  a[0] = p[0]; a[1] = p[1];
  b[0] = q[0]; b[1] = q[1];

  if (a[0] < lo[0] && b[0] < lo[0]) return 0;
  if (a[0] < lo[0] || b[0] < lo[0]) {
    y = a[1] + (lo[0]-a[0])/(b[0]-a[0])*(b[1]-a[1]);
    if (a[0] < lo[0]) {
      a[0] = lo[0]; a[1] = y;
    } else {
      b[0] = lo[0]; b[1] = y;
    }
  }
  if (a[0] > hi[0] && b[0] > hi[0]) return 0;
  if (a[0] > hi[0] || b[0] > hi[0]) {
    y = a[1] + (hi[0]-a[0])/(b[0]-a[0])*(b[1]-a[1]);
    if (a[0] > hi[0]) {
      a[0] = hi[0]; a[1] = y;
    } else {
      b[0] = hi[0]; b[1] = y;
    }
  }

  if (a[1] < lo[1] && b[1] < lo[1]) return 0;
  if (a[1] < lo[1] || b[1] < lo[1]) {
    x = a[0] + (lo[1]-a[1])/(b[1]-a[1])*(b[0]-a[0]);
    if (a[1] < lo[1]) {
      a[0] = x; a[1] = lo[1];
    } else {
      b[0] = x; b[1] = lo[1];
    }
  }
  if (a[1] > hi[1] && b[1] > hi[1]) return 0;
  if (a[1] > hi[1] || b[1] > hi[1]) {
    x = a[0] + (hi[1]-a[1])/(b[1]-a[1])*(b[0]-a[0]);
    if (a[1] > hi[1]) {
      a[0] = x; a[1] = hi[1];
    } else {
      b[0] = x; b[1] = hi[1];
    }
  }

  return 1;
}

/* ----------------------------------------------------------------------
   point C on segment AB where coordinate DIM equals VALUE (Cut3d::between)
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void between3d(const double *a, const double *b, int dim, double value,
               double *c)
{
  if (dim == 0) {
    c[1] = a[1] + (value-a[dim])/(b[dim]-a[dim]) * (b[1]-a[1]);
    c[2] = a[2] + (value-a[dim])/(b[dim]-a[dim]) * (b[2]-a[2]);
    c[0] = value;
  } else if (dim == 1) {
    c[0] = a[0] + (value-a[dim])/(b[dim]-a[dim]) * (b[0]-a[0]);
    c[2] = a[2] + (value-a[dim])/(b[dim]-a[dim]) * (b[2]-a[2]);
    c[1] = value;
  } else {
    c[0] = a[0] + (value-a[dim])/(b[dim]-a[dim]) * (b[0]-a[0]);
    c[1] = a[1] + (value-a[dim])/(b[dim]-a[dim]) * (b[1]-a[1]);
    c[2] = value;
  }
}

/* ----------------------------------------------------------------------
   # of vertices of triangle P0P1P2 clipped to the cell LO/HI, 0 if none
     (Cut3d::clip, Sutherland-Hodgman against the 6 face planes)
   a triangle clipped by 6 planes has at most 9 vertices
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
int clip3d(const double *p0, const double *p1, const double *p2,
           const double *lo, const double *hi)
{
  int i,npath,nnew;
  double value;
  double path1[12][3],path2[12][3];
  double (*path)[3];
  double (*newpath)[3];
  double *s,*e;

  nnew = 3;
  for (i = 0; i < 3; i++) {
    path1[0][i] = p0[i];
    path1[1][i] = p1[i];
    path1[2][i] = p2[i];
  }

  if (p0[0] >= lo[0] && p0[0] <= hi[0] &&
      p0[1] >= lo[1] && p0[1] <= hi[1] &&
      p0[2] >= lo[2] && p0[2] <= hi[2] &&
      p1[0] >= lo[0] && p1[0] <= hi[0] &&
      p1[1] >= lo[1] && p1[1] <= hi[1] &&
      p1[2] >= lo[2] && p1[2] <= hi[2] &&
      p2[0] >= lo[0] && p2[0] <= hi[0] &&
      p2[1] >= lo[1] && p2[1] <= hi[1] &&
      p2[2] >= lo[2] && p2[2] <= hi[2]) return 1;

  for (int dim = 0; dim < 3; dim++) {
    path = path1;
    newpath = path2;
    npath = nnew;
    nnew = 0;

    value = lo[dim];
    s = path[npath-1];
    for (i = 0; i < npath; i++) {
      e = path[i];
      if (e[dim] >= value) {
        if (s[dim] < value) between3d(s,e,dim,value,newpath[nnew++]);
        for (int k = 0; k < 3; k++) newpath[nnew][k] = e[k];
        nnew++;
      } else if (s[dim] >= value) between3d(e,s,dim,value,newpath[nnew++]);
      s = e;
    }
    if (!nnew) return 0;

    path = path2;
    newpath = path1;
    npath = nnew;
    nnew = 0;

    value = hi[dim];
    s = path[npath-1];
    for (i = 0; i < npath; i++) {
      e = path[i];
      if (e[dim] <= value) {
        if (s[dim] > value) between3d(s,e,dim,value,newpath[nnew++]);
        for (int k = 0; k < 3; k++) newpath[nnew][k] = e[k];
        nnew++;
      } else if (s[dim] <= value) between3d(e,s,dim,value,newpath[nnew++]);
      s = e;
    }
    if (!nnew) return 0;
  }

  return nnew;
}

}

}

#endif
