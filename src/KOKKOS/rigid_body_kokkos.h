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

#ifndef SPARTA_RIGID_BODY_KOKKOS_H
#define SPARTA_RIGID_BODY_KOKKOS_H

#include "kokkos_type.h"
#include "geometry_kokkos.h"

namespace SPARTA_NS {

/* ----------------------------------------------------------------------
   the replicated rigid body table on the device: the element geometry
     and boxes, the per-body bboxes and the bins of bodies by COM, with
     the queries the host makes of FixRigid's table
   filled by FixRigidKokkos::pack_body_device(), copied by value into the
     kernels which read it
   explicitly double, not SPARTA_FLOAT: the inside test is a ray cast
     whose parity must match the host result exactly, and a single
     precision build (SPA_PRECISION 1) would change which side of an
     element a nearly tangent ray falls on
------------------------------------------------------------------------- */

struct RigidBodyKK {
  typedef Kokkos::DualView<double***,DeviceType::array_layout,DeviceType> tdual_dbl_3d;
  typedef Kokkos::DualView<double**,DeviceType::array_layout,DeviceType> tdual_dbl_2d;

  int nbody,dim;
  tdual_dbl_3d::t_dev d_bodypt;       // corner pts of each element
  tdual_dbl_2d::t_dev d_bodynorm;     // outward normal of each element
  tdual_dbl_2d::t_dev d_elemlo,d_elemhi;    // box of each element
  tdual_dbl_2d::t_dev d_bbodylo,d_bbodyhi;  // bbox of each body
  DAT::t_int_1d d_bodystart;          // elements of body I = start[I]..start[I+1]-1
  DAT::t_int_1d d_lblist;             // local surf index of each element
  DAT::t_int_1d d_binstart,d_binlist; // bins of bodies by COM
  int nbin[3];
  double binlo[3],bininv[3];
  double rmaxall;                     // max over bodies of rmaxbody + bbox inflation

  // bins overlapping a box inflated by rmaxall, FixRigid::body_box()

  KOKKOS_INLINE_FUNCTION
  void box_bins(const double *lo, const double *hi, int *blo, int *bhi) const
  {
    for (int k = 0; k < 3; k++) {
      blo[k] = (int) ((lo[k]-rmaxall-binlo[k]) * bininv[k]);
      bhi[k] = (int) ((hi[k]+rmaxall-binlo[k]) * bininv[k]);
      blo[k] = MAX(0,MIN(blo[k],nbin[k]-1));
      bhi[k] = MAX(0,MIN(bhi[k],nbin[k]-1));
    }
  }

  // 1 if the bbox of body ibody overlaps the box, touching counts

  KOKKOS_INLINE_FUNCTION
  int box_overlap(int ibody, const double *lo, const double *hi) const
  {
    if (d_bbodyhi(ibody,0) < lo[0] || d_bbodylo(ibody,0) > hi[0]) return 0;
    if (d_bbodyhi(ibody,1) < lo[1] || d_bbodylo(ibody,1) > hi[1]) return 0;
    if (d_bbodyhi(ibody,2) < lo[2] || d_bbodylo(ibody,2) > hi[2]) return 0;
    return 1;
  }

  // 1 if the box of element e overlaps the box

  KOKKOS_INLINE_FUNCTION
  int elem_overlap(int e, const double *lo, const double *hi) const
  {
    if (d_elemhi(e,0) < lo[0] || d_elemlo(e,0) > hi[0]) return 0;
    if (d_elemhi(e,1) < lo[1] || d_elemlo(e,1) > hi[1]) return 0;
    if (d_elemhi(e,2) < lo[2] || d_elemlo(e,2) > hi[2]) return 0;
    return 1;
  }

  // FixRigid::inside_body(): parity of the crossings of a ray from x
  //   to a point outside the body's bbox; the ray and the loop order
  //   are those of the host routine, so the count matches

  KOKKOS_INLINE_FUNCTION
  int inside_body(int ibody, const double *x) const
  {
    double blox = d_bbodylo(ibody,0), bhix = d_bbodyhi(ibody,0);
    double dmax = MAX(bhix-blox,d_bbodyhi(ibody,1)-d_bbodylo(ibody,1));
    dmax = MAX(dmax,d_bbodyhi(ibody,2)-d_bbodylo(ibody,2));

    double xin[3],xout[3],xc[3];
    xin[0] = x[0]; xin[1] = x[1]; xin[2] = x[2];
    xout[0] = bhix + 0.414159*dmax;
    xout[1] = x[1] + 0.271828*dmax;
    if (dim == 3) xout[2] = x[2] + 0.161803*dmax;
    else xout[2] = 0.0;

    int count = 0;
    for (int e = d_bodystart(ibody); e < d_bodystart(ibody+1); e++) {
      // only a 3d element has a third corner point; in 2d the
      //   third slot of d_bodypt is never written

      double p1[3],p2[3],p3[3],nrm[3];
      for (int k = 0; k < 3; k++) {
        p1[k] = d_bodypt(e,0,k);
        p2[k] = d_bodypt(e,1,k);
        p3[k] = (dim == 3) ? d_bodypt(e,2,k) : 0.0;
        nrm[k] = d_bodynorm(e,k);
      }
      double param;
      int side;
      bool hitflag;
      if (dim == 2)
        hitflag = GeometryKokkos::
          line_line_intersect(xin,xout,p1,p2,nrm,xc,param,side);
      else
        hitflag = GeometryKokkos::
          line_tri_intersect(xin,xout,p1,p2,p3,nrm,xc,param,side);
      if (hitflag) count++;
    }
    return count % 2;
  }

  // FixRigid::inside_any_body(): the bins overlapping x inflated by
  //   rmaxall give the candidate bodies, then each body's bbox is
  //   tested exactly, x as a degenerate box

  KOKKOS_INLINE_FUNCTION
  int inside_any_body(const double *x) const
  {
    int blo[3],bhi[3];
    box_bins(x,x,blo,bhi);

    for (int ibz = blo[2]; ibz <= bhi[2]; ibz++)
      for (int iby = blo[1]; iby <= bhi[1]; iby++)
        for (int ibx = blo[0]; ibx <= bhi[0]; ibx++) {
          const int ibin = (ibz*nbin[1] + iby)*nbin[0] + ibx;
          for (int m = d_binstart[ibin]; m < d_binstart[ibin+1]; m++) {
            const int ibody = d_binlist[m];
            if (!box_overlap(ibody,x,x)) continue;
            if (inside_body(ibody,x)) return 1;
          }
        }
    return 0;
  }
};

}

#endif
