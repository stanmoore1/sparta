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

/* ----------------------------------------------------------------------
   Contributing author: Mike Brown (SNL)
------------------------------------------------------------------------- */

#ifndef SPARTA_MATH_EXTRA_KOKKOS_H
#define SPARTA_MATH_EXTRA_KOKKOS_H

#include "spatype.h"
#include <Kokkos_Core.hpp>
#include <type_traits>
#include "math.h"
#include "stdio.h"
#include "string.h"

namespace MathExtraKokkos {

  // 3 vector operations, templated on the element types so they work for
  //   any mix of KK_FLOAT, KK_POS_FLOAT and double arguments

  template<class T>
  KOKKOS_INLINE_FUNCTION void norm3(T *v)
  {
    const T scale = static_cast<T>(1.0)/Kokkos::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    v[0] *= scale;
    v[1] *= scale;
    v[2] *= scale;
  }

  template<class T1, class T2>
  KOKKOS_INLINE_FUNCTION void normalize3(const T1 *v, T2 *ans)
  {
    const T1 scale = static_cast<T1>(1.0)/Kokkos::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    ans[0] = v[0]*scale;
    ans[1] = v[1]*scale;
    ans[2] = v[2]*scale;
  }

  template<class S, class T>
  KOKKOS_INLINE_FUNCTION void snorm3(const S length, T *v)
  {
    const T scale = length/Kokkos::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    v[0] *= scale;
    v[1] *= scale;
    v[2] *= scale;
  }

  template<class S, class T1, class T2>
  KOKKOS_INLINE_FUNCTION void snormalize3(const S length, const T1 *v, T2 *ans)
  {
    const T1 scale = length/Kokkos::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    ans[0] = v[0]*scale;
    ans[1] = v[1]*scale;
    ans[2] = v[2]*scale;
  }

  template<class T>
  KOKKOS_INLINE_FUNCTION void negate3(T *v)
  {
    v[0] = -v[0];
    v[1] = -v[1];
    v[2] = -v[2];
  }

  // scale vector v by s in place

  template<class S, class T>
  KOKKOS_INLINE_FUNCTION void scale3(S s, T *v)
  {
    v[0] *= s;
    v[1] *= s;
    v[2] *= s;
  }

  // scale vector v by s, return in ans

  template<class S, class T1, class T2>
  KOKKOS_INLINE_FUNCTION void scale3(S s, const T1 *v, T2 *ans)
  {
    ans[0] = s*v[0];
    ans[1] = s*v[1];
    ans[2] = s*v[2];
  }

  // axpy: y = alpha*x + y

  template<class S, class T1, class T2>
  KOKKOS_INLINE_FUNCTION void axpy3(S alpha, const T1 *x, T2 *y)
  {
    y[0] += alpha*x[0];
    y[1] += alpha*x[1];
    y[2] += alpha*x[2];
  }

  // axpy: ynew = alpha*x + y

  template<class S, class T1, class T2, class T3>
  KOKKOS_INLINE_FUNCTION void axpy3(S alpha, const T1 *x, const T2 *y, T3 *ynew)
  {
    ynew[0] += alpha*x[0] + y[0];
    ynew[1] += alpha*x[1] + y[1];
    ynew[2] += alpha*x[2] + y[2];
  }

  // ans = v1 + v2

  template<class T1, class T2, class T3>
  KOKKOS_INLINE_FUNCTION void add3(const T1 *v1, const T2 *v2, T3 *ans)
  {
    ans[0] = v1[0] + v2[0];
    ans[1] = v1[1] + v2[1];
    ans[2] = v1[2] + v2[2];
  }

  // ans = v1 - v2

  template<class T1, class T2, class T3>
  KOKKOS_INLINE_FUNCTION void sub3(const T1 *v1, const T2 *v2, T3 *ans)
  {
    ans[0] = v1[0] - v2[0];
    ans[1] = v1[1] - v2[1];
    ans[2] = v1[2] - v2[2];
  }

  // length of vector v

  template<class T>
  KOKKOS_INLINE_FUNCTION T len3(const T *v)
  {
    return Kokkos::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  }

  // squared length of vector v, or dot product of v with itself

  template<class T>
  KOKKOS_INLINE_FUNCTION T lensq3(const T *v)
  {
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
  }

  // dot product of 2 vectors, in the wider of the two precisions

  template<class T1, class T2>
  KOKKOS_INLINE_FUNCTION std::common_type_t<T1,T2> dot3(const T1 *v1, const T2 *v2)
  {
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
  }

  // cross product of 2 vectors

  template<class T1, class T2, class T3>
  KOKKOS_INLINE_FUNCTION void cross3(const T1 *v1, const T2 *v2, T3 *ans)
  {
    ans[0] = v1[1]*v2[2] - v1[2]*v2[1];
    ans[1] = v1[2]*v2[0] - v1[0]*v2[2];
    ans[2] = v1[0]*v2[1] - v1[1]*v2[0];
  }

  // reflect vector v around unit normal n
  // return updated v of same length = v - 2(v dot n)n

  template<class T1, class T2>
  KOKKOS_INLINE_FUNCTION void reflect3(T1 *v, const T2 *n)
  {
    const std::common_type_t<T1,T2> dot = dot3(v,n);
    v[0] -= static_cast<T1>(2.0)*dot*n[0];
    v[1] -= static_cast<T1>(2.0)*dot*n[1];
    v[2] -= static_cast<T1>(2.0)*dot*n[2];
  }

  // 3 vector operations


  // 3x3 matrix operations

  KOKKOS_INLINE_FUNCTION double det3(const double mat[3][3]);
  KOKKOS_INLINE_FUNCTION void diag_times3(const double *diagonal, const double mat[3][3],
                          double ans[3][3]);
  KOKKOS_INLINE_FUNCTION void plus3(const double m[3][3], const double m2[3][3],
                    double ans[3][3]);
  KOKKOS_INLINE_FUNCTION void times3(const double m[3][3], const double m2[3][3],
                     double ans[3][3]);
  KOKKOS_INLINE_FUNCTION void transpose_times3(const double mat1[3][3],
                               const double mat2[3][3],
                               double ans[3][3]);
  KOKKOS_INLINE_FUNCTION void times3_transpose(const double mat1[3][3],
                               const double mat2[3][3],
                               double ans[3][3]);
  KOKKOS_INLINE_FUNCTION void invert3(const double mat[3][3], double ans[3][3]);
  KOKKOS_INLINE_FUNCTION void matvec(const double mat[3][3], const double*vec, double *ans);
  KOKKOS_INLINE_FUNCTION void matvec(const double *ex, const double *ey, const double *ez,
                     const double *vec, double *ans);
  KOKKOS_INLINE_FUNCTION void transpose_matvec(const double mat[3][3], const double*vec,
                               double *ans);
  KOKKOS_INLINE_FUNCTION void transpose_matvec(const double *ex, const double *ey,
                               const double *ez, const double *v,
                               double *ans);
  KOKKOS_INLINE_FUNCTION void transpose_diag3(const double mat[3][3], const double*vec,
                              double ans[3][3]);
  KOKKOS_INLINE_FUNCTION void vecmat(const double *v, const double m[3][3], double *ans);
  KOKKOS_INLINE_FUNCTION void scalar_times3(const double f, double m[3][3]);

  // quaternion operations

  KOKKOS_INLINE_FUNCTION void axisangle_to_quat(const double *v, const double angle,
                                double *quat);
  KOKKOS_INLINE_FUNCTION void quat_to_mat(const double *quat, double mat[3][3]);
}

/* ----------------------------------------------------------------------
   normalize a vector in place
------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------
   determinant of a matrix
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double MathExtraKokkos::det3(const double m[3][3])
{
  double ans = m[0][0]*m[1][1]*m[2][2] - m[0][0]*m[1][2]*m[2][1] -
    m[1][0]*m[0][1]*m[2][2] + m[1][0]*m[0][2]*m[2][1] +
    m[2][0]*m[0][1]*m[1][2] - m[2][0]*m[0][2]*m[1][1];
  return ans;
}

/* ----------------------------------------------------------------------
   diagonal matrix times a full matrix
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::diag_times3(const double *d, const double m[3][3],
                            double ans[3][3])
{
  ans[0][0] = d[0]*m[0][0];
  ans[0][1] = d[0]*m[0][1];
  ans[0][2] = d[0]*m[0][2];
  ans[1][0] = d[1]*m[1][0];
  ans[1][1] = d[1]*m[1][1];
  ans[1][2] = d[1]*m[1][2];
  ans[2][0] = d[2]*m[2][0];
  ans[2][1] = d[2]*m[2][1];
  ans[2][2] = d[2]*m[2][2];
}

/* ----------------------------------------------------------------------
   add two matrices
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::plus3(const double m[3][3], const double m2[3][3],
                      double ans[3][3])
{
  ans[0][0] = m[0][0]+m2[0][0];
  ans[0][1] = m[0][1]+m2[0][1];
  ans[0][2] = m[0][2]+m2[0][2];
  ans[1][0] = m[1][0]+m2[1][0];
  ans[1][1] = m[1][1]+m2[1][1];
  ans[1][2] = m[1][2]+m2[1][2];
  ans[2][0] = m[2][0]+m2[2][0];
  ans[2][1] = m[2][1]+m2[2][1];
  ans[2][2] = m[2][2]+m2[2][2];
}

/* ----------------------------------------------------------------------
   multiply mat1 times mat2
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::times3(const double m[3][3], const double m2[3][3],
                       double ans[3][3])
{
  ans[0][0] = m[0][0]*m2[0][0] + m[0][1]*m2[1][0] + m[0][2]*m2[2][0];
  ans[0][1] = m[0][0]*m2[0][1] + m[0][1]*m2[1][1] + m[0][2]*m2[2][1];
  ans[0][2] = m[0][0]*m2[0][2] + m[0][1]*m2[1][2] + m[0][2]*m2[2][2];
  ans[1][0] = m[1][0]*m2[0][0] + m[1][1]*m2[1][0] + m[1][2]*m2[2][0];
  ans[1][1] = m[1][0]*m2[0][1] + m[1][1]*m2[1][1] + m[1][2]*m2[2][1];
  ans[1][2] = m[1][0]*m2[0][2] + m[1][1]*m2[1][2] + m[1][2]*m2[2][2];
  ans[2][0] = m[2][0]*m2[0][0] + m[2][1]*m2[1][0] + m[2][2]*m2[2][0];
  ans[2][1] = m[2][0]*m2[0][1] + m[2][1]*m2[1][1] + m[2][2]*m2[2][1];
  ans[2][2] = m[2][0]*m2[0][2] + m[2][1]*m2[1][2] + m[2][2]*m2[2][2];
}

/* ----------------------------------------------------------------------
   multiply the transpose of mat1 times mat2
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::transpose_times3(const double m[3][3], const double m2[3][3],
                                 double ans[3][3])
{
  ans[0][0] = m[0][0]*m2[0][0] + m[1][0]*m2[1][0] + m[2][0]*m2[2][0];
  ans[0][1] = m[0][0]*m2[0][1] + m[1][0]*m2[1][1] + m[2][0]*m2[2][1];
  ans[0][2] = m[0][0]*m2[0][2] + m[1][0]*m2[1][2] + m[2][0]*m2[2][2];
  ans[1][0] = m[0][1]*m2[0][0] + m[1][1]*m2[1][0] + m[2][1]*m2[2][0];
  ans[1][1] = m[0][1]*m2[0][1] + m[1][1]*m2[1][1] + m[2][1]*m2[2][1];
  ans[1][2] = m[0][1]*m2[0][2] + m[1][1]*m2[1][2] + m[2][1]*m2[2][2];
  ans[2][0] = m[0][2]*m2[0][0] + m[1][2]*m2[1][0] + m[2][2]*m2[2][0];
  ans[2][1] = m[0][2]*m2[0][1] + m[1][2]*m2[1][1] + m[2][2]*m2[2][1];
  ans[2][2] = m[0][2]*m2[0][2] + m[1][2]*m2[1][2] + m[2][2]*m2[2][2];
}

/* ----------------------------------------------------------------------
   multiply mat1 times transpose of mat2
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::times3_transpose(const double m[3][3], const double m2[3][3],
                                 double ans[3][3])
{
  ans[0][0] = m[0][0]*m2[0][0] + m[0][1]*m2[0][1] + m[0][2]*m2[0][2];
  ans[0][1] = m[0][0]*m2[1][0] + m[0][1]*m2[1][1] + m[0][2]*m2[1][2];
  ans[0][2] = m[0][0]*m2[2][0] + m[0][1]*m2[2][1] + m[0][2]*m2[2][2];
  ans[1][0] = m[1][0]*m2[0][0] + m[1][1]*m2[0][1] + m[1][2]*m2[0][2];
  ans[1][1] = m[1][0]*m2[1][0] + m[1][1]*m2[1][1] + m[1][2]*m2[1][2];
  ans[1][2] = m[1][0]*m2[2][0] + m[1][1]*m2[2][1] + m[1][2]*m2[2][2];
  ans[2][0] = m[2][0]*m2[0][0] + m[2][1]*m2[0][1] + m[2][2]*m2[0][2];
  ans[2][1] = m[2][0]*m2[1][0] + m[2][1]*m2[1][1] + m[2][2]*m2[1][2];
  ans[2][2] = m[2][0]*m2[2][0] + m[2][1]*m2[2][1] + m[2][2]*m2[2][2];
}

/* ----------------------------------------------------------------------
   invert a matrix
   does NOT checks for singular or badly scaled matrix
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::invert3(const double m[3][3], double ans[3][3])
{
  double den = m[0][0]*m[1][1]*m[2][2]-m[0][0]*m[1][2]*m[2][1];
  den += -m[1][0]*m[0][1]*m[2][2]+m[1][0]*m[0][2]*m[2][1];
  den += m[2][0]*m[0][1]*m[1][2]-m[2][0]*m[0][2]*m[1][1];

  ans[0][0] = (m[1][1]*m[2][2]-m[1][2]*m[2][1]) / den;
  ans[0][1] = -(m[0][1]*m[2][2]-m[0][2]*m[2][1]) / den;
  ans[0][2] = (m[0][1]*m[1][2]-m[0][2]*m[1][1]) / den;
  ans[1][0] = -(m[1][0]*m[2][2]-m[1][2]*m[2][0]) / den;
  ans[1][1] = (m[0][0]*m[2][2]-m[0][2]*m[2][0]) / den;
  ans[1][2] = -(m[0][0]*m[1][2]-m[0][2]*m[1][0]) / den;
  ans[2][0] = (m[1][0]*m[2][1]-m[1][1]*m[2][0]) / den;
  ans[2][1] = -(m[0][0]*m[2][1]-m[0][1]*m[2][0]) / den;
  ans[2][2] = (m[0][0]*m[1][1]-m[0][1]*m[1][0]) / den;
}

/* ----------------------------------------------------------------------
   matrix times vector
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::matvec(const double m[3][3], const double *v, double *ans)
{
  ans[0] = m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2];
  ans[1] = m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2];
  ans[2] = m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2];
}

/* ----------------------------------------------------------------------
   matrix times vector
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::matvec(const double *ex, const double *ey, const double *ez,
                       const double *v, double *ans)
{
  ans[0] = ex[0]*v[0] + ey[0]*v[1] + ez[0]*v[2];
  ans[1] = ex[1]*v[0] + ey[1]*v[1] + ez[1]*v[2];
  ans[2] = ex[2]*v[0] + ey[2]*v[1] + ez[2]*v[2];
}

/* ----------------------------------------------------------------------
   transposed matrix times vector
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::transpose_matvec(const double m[3][3], const double *v,
                                 double *ans)
{
  ans[0] = m[0][0]*v[0] + m[1][0]*v[1] + m[2][0]*v[2];
  ans[1] = m[0][1]*v[0] + m[1][1]*v[1] + m[2][1]*v[2];
  ans[2] = m[0][2]*v[0] + m[1][2]*v[1] + m[2][2]*v[2];
}

/* ----------------------------------------------------------------------
   transposed matrix times vector
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::transpose_matvec(const double *ex, const double *ey,
                                 const double *ez, const double *v,
                                 double *ans)
{
  ans[0] = ex[0]*v[0] + ex[1]*v[1] + ex[2]*v[2];
  ans[1] = ey[0]*v[0] + ey[1]*v[1] + ey[2]*v[2];
  ans[2] = ez[0]*v[0] + ez[1]*v[1] + ez[2]*v[2];
}

/* ----------------------------------------------------------------------
   transposed matrix times diagonal matrix
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::transpose_diag3(const double m[3][3], const double *d,
                                double ans[3][3])
{
  ans[0][0] = m[0][0]*d[0];
  ans[0][1] = m[1][0]*d[1];
  ans[0][2] = m[2][0]*d[2];
  ans[1][0] = m[0][1]*d[0];
  ans[1][1] = m[1][1]*d[1];
  ans[1][2] = m[2][1]*d[2];
  ans[2][0] = m[0][2]*d[0];
  ans[2][1] = m[1][2]*d[1];
  ans[2][2] = m[2][2]*d[2];
}

/* ----------------------------------------------------------------------
   row vector times matrix
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::vecmat(const double *v, const double m[3][3], double *ans)
{
  ans[0] = v[0]*m[0][0] + v[1]*m[1][0] + v[2]*m[2][0];
  ans[1] = v[0]*m[0][1] + v[1]*m[1][1] + v[2]*m[2][1];
  ans[2] = v[0]*m[0][2] + v[1]*m[1][2] + v[2]*m[2][2];
}

/* ----------------------------------------------------------------------
   matrix times scalar, in place
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::scalar_times3(const double f, double m[3][3])
{
  m[0][0] *= f; m[0][1] *= f; m[0][2] *= f;
  m[1][0] *= f; m[1][1] *= f; m[1][2] *= f;
  m[2][0] *= f; m[2][1] *= f; m[2][2] *= f;
}

/* ----------------------------------------------------------------------
   compute quaternion from axis-angle rotation
   v MUST be a unit vector
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::axisangle_to_quat(const double *v, const double angle,
                                  double *quat)
{
  double halfa = 0.5*angle;
  double sina = sin(halfa);
  quat[0] = cos(halfa);
  quat[1] = v[0]*sina;
  quat[2] = v[1]*sina;
  quat[3] = v[2]*sina;
}

/* ----------------------------------------------------------------------
   compute rotation matrix from quaternion
   quat = [w i j k]
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MathExtraKokkos::quat_to_mat(const double *quat, double mat[3][3])
{
  double w2 = quat[0]*quat[0];
  double i2 = quat[1]*quat[1];
  double j2 = quat[2]*quat[2];
  double k2 = quat[3]*quat[3];
  double twoij = 2.0*quat[1]*quat[2];
  double twoik = 2.0*quat[1]*quat[3];
  double twojk = 2.0*quat[2]*quat[3];
  double twoiw = 2.0*quat[1]*quat[0];
  double twojw = 2.0*quat[2]*quat[0];
  double twokw = 2.0*quat[3]*quat[0];

  mat[0][0] = w2+i2-j2-k2;
  mat[0][1] = twoij-twokw;
  mat[0][2] = twojw+twoik;

  mat[1][0] = twoij+twokw;
  mat[1][1] = w2-i2+j2-k2;
  mat[1][2] = twojk-twoiw;

  mat[2][0] = twoik-twojw;
  mat[2][1] = twojk+twoiw;
  mat[2][2] = w2-i2-j2+k2;
}

#endif
