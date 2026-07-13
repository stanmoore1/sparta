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

// Standalone unit tests for deterministic, self-contained SPARTA kernels:
//   - MathExtra   3-vector and 3x3 matrix helpers (src/math_extra.h/.cpp)
//   - RanKnuth    Knuth subtractive random number generator (src/random_knuth.*)
//
// These utilities have no dependence on the SPARTA object, MPI, or the parallel
// runtime, so they can be exercised directly against known-good values.  The
// harness uses a tiny assert layer (no external test framework) and returns a
// nonzero exit code if any check fails, which CTest interprets as failure.

#include "math_extra.h"
#include "random_knuth.h"

#include <cmath>
#include <cstdio>

using namespace SPARTA_NS;

static int nfail = 0;
static int ncheck = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    ++ncheck;                                                             \
    if (!(cond)) {                                                        \
      ++nfail;                                                            \
      std::printf("FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond);  \
    }                                                                     \
  } while (0)

#define CHECK_CLOSE(a, b, tol)                                            \
  do {                                                                    \
    ++ncheck;                                                             \
    double _va = (a), _vb = (b);                                          \
    if (std::fabs(_va - _vb) > (tol)) {                                   \
      ++nfail;                                                            \
      std::printf("FAIL %s:%d  CHECK_CLOSE(%s=%.12g, %s=%.12g, tol=%g)\n", \
                  __FILE__, __LINE__, #a, _va, #b, _vb, (double)(tol));   \
    }                                                                     \
  } while (0)

static const double TOL = 1.0e-12;

// ------------------------------------------------------------------
// MathExtra: 3-vector operations
// ------------------------------------------------------------------

static void test_vector_ops()
{
  double a[3] = {1.0, 2.0, 3.0};
  double b[3] = {4.0, 5.0, 6.0};
  double ans[3];

  CHECK_CLOSE(MathExtra::dot3(a, b), 32.0, TOL);
  CHECK_CLOSE(MathExtra::lensq3(a), 14.0, TOL);

  double c[3] = {3.0, 4.0, 0.0};
  CHECK_CLOSE(MathExtra::len3(c), 5.0, TOL);

  MathExtra::add3(a, b, ans);
  CHECK_CLOSE(ans[0], 5.0, TOL);
  CHECK_CLOSE(ans[1], 7.0, TOL);
  CHECK_CLOSE(ans[2], 9.0, TOL);

  MathExtra::sub3(b, a, ans);
  CHECK_CLOSE(ans[0], 3.0, TOL);
  CHECK_CLOSE(ans[1], 3.0, TOL);
  CHECK_CLOSE(ans[2], 3.0, TOL);

  MathExtra::scale3(2.0, a, ans);
  CHECK_CLOSE(ans[0], 2.0, TOL);
  CHECK_CLOSE(ans[1], 4.0, TOL);
  CHECK_CLOSE(ans[2], 6.0, TOL);

  // cross product: x cross y == z (right-handed)
  double ex[3] = {1.0, 0.0, 0.0};
  double ey[3] = {0.0, 1.0, 0.0};
  MathExtra::cross3(ex, ey, ans);
  CHECK_CLOSE(ans[0], 0.0, TOL);
  CHECK_CLOSE(ans[1], 0.0, TOL);
  CHECK_CLOSE(ans[2], 1.0, TOL);

  // cross product is anti-commutative: b x a == -(a x b)
  double axb[3], bxa[3];
  MathExtra::cross3(a, b, axb);
  MathExtra::cross3(b, a, bxa);
  CHECK_CLOSE(axb[0], -bxa[0], TOL);
  CHECK_CLOSE(axb[1], -bxa[1], TOL);
  CHECK_CLOSE(axb[2], -bxa[2], TOL);

  // a x b is orthogonal to both a and b
  CHECK_CLOSE(MathExtra::dot3(axb, a), 0.0, TOL);
  CHECK_CLOSE(MathExtra::dot3(axb, b), 0.0, TOL);

  // normalize3 yields a unit vector
  double n[3];
  MathExtra::normalize3(a, n);
  CHECK_CLOSE(MathExtra::len3(n), 1.0, TOL);

  // norm3 normalizes in place
  double d[3] = {0.0, 0.0, 7.0};
  MathExtra::norm3(d);
  CHECK_CLOSE(d[0], 0.0, TOL);
  CHECK_CLOSE(d[1], 0.0, TOL);
  CHECK_CLOSE(d[2], 1.0, TOL);
}

// ------------------------------------------------------------------
// MathExtra: 3x3 matrix operations
// ------------------------------------------------------------------

static void test_matrix_ops()
{
  double ident[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  CHECK_CLOSE(MathExtra::det3(ident), 1.0, TOL);

  double m[3][3] = {{2, 0, 1}, {1, 3, 2}, {1, 0, 3}};
  // det = 2*(9-0) - 0 + 1*(0-3) = 18 - 3 = 15
  CHECK_CLOSE(MathExtra::det3(m), 15.0, TOL);

  // matvec with identity returns the vector unchanged
  double v[3] = {5.0, -2.0, 7.0};
  double ans[3];
  MathExtra::matvec(ident, v, ans);
  CHECK_CLOSE(ans[0], 5.0, TOL);
  CHECK_CLOSE(ans[1], -2.0, TOL);
  CHECK_CLOSE(ans[2], 7.0, TOL);

  // M * inv(M) == identity
  double inv[3][3], prod[3][3];
  MathExtra::invert3(m, inv);
  MathExtra::times3(m, inv, prod);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      CHECK_CLOSE(prod[i][j], (i == j) ? 1.0 : 0.0, 1.0e-10);

  // times3 with identity returns the matrix unchanged
  double prod2[3][3];
  MathExtra::times3(ident, m, prod2);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      CHECK_CLOSE(prod2[i][j], m[i][j], TOL);
}

// ------------------------------------------------------------------
// MathExtra: quaternion / rotation
// ------------------------------------------------------------------

static void test_quaternion()
{
  // 90-degree rotation about z maps x-axis to y-axis
  double axis[3] = {0.0, 0.0, 1.0};
  double quat[4];
  MathExtra::axisangle_to_quat(axis, M_PI / 2.0, quat);

  double rot[3][3];
  MathExtra::quat_to_mat(quat, rot);

  double ex[3] = {1.0, 0.0, 0.0};
  double ans[3];
  MathExtra::matvec(rot, ex, ans);
  CHECK_CLOSE(ans[0], 0.0, 1.0e-12);
  CHECK_CLOSE(ans[1], 1.0, 1.0e-12);
  CHECK_CLOSE(ans[2], 0.0, 1.0e-12);

  // rotation matrix is orthogonal: det == 1
  CHECK_CLOSE(MathExtra::det3(rot), 1.0, 1.0e-12);
}

// ------------------------------------------------------------------
// RanKnuth: determinism, range, and distribution
// ------------------------------------------------------------------

static void test_rng_determinism()
{
  // identical seeds must produce identical sequences
  RanKnuth r1(12345);
  RanKnuth r2(12345);
  for (int i = 0; i < 1000; i++)
    CHECK(r1.uniform() == r2.uniform());

  // different seeds should diverge (extremely unlikely to match on all)
  RanKnuth r3(12345);
  RanKnuth r4(67890);
  int matches = 0;
  for (int i = 0; i < 100; i++)
    if (r3.uniform() == r4.uniform()) matches++;
  CHECK(matches < 100);
}

static void test_rng_range()
{
  RanKnuth r(987654);
  for (int i = 0; i < 100000; i++) {
    double u = r.uniform();
    CHECK(u >= 0.0 && u < 1.0);
  }
}

static void test_rng_distribution()
{
  // uniform() mean should approach 0.5 over many samples
  RanKnuth r(24680);
  const int N = 500000;
  double sum = 0.0;
  for (int i = 0; i < N; i++) sum += r.uniform();
  double mean = sum / N;
  CHECK_CLOSE(mean, 0.5, 0.01);

  // gaussian() should have ~zero mean and ~unit variance
  RanKnuth rg(13579);
  double gsum = 0.0, gsumsq = 0.0;
  for (int i = 0; i < N; i++) {
    double g = rg.gaussian();
    gsum += g;
    gsumsq += g * g;
  }
  double gmean = gsum / N;
  double gvar = gsumsq / N - gmean * gmean;
  CHECK_CLOSE(gmean, 0.0, 0.02);
  CHECK_CLOSE(gvar, 1.0, 0.02);
}

int main()
{
  test_vector_ops();
  test_matrix_ops();
  test_quaternion();
  test_rng_determinism();
  test_rng_range();
  test_rng_distribution();

  std::printf("\n%d checks run, %d failures\n", ncheck, nfail);
  if (nfail == 0) std::printf("All unit tests passed\n");
  return nfail == 0 ? 0 : 1;
}
