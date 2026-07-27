/* Is a hand-rolled pow() actually faster than glibc's in the collide kernel?
 *
 * The roofline says the collide kernel is latency/dependency bound, not
 * throughput bound: pow() sits on the chain vr2 -> pow -> vre -> compare.
 * So the number that matters is pow's *latency*, not its throughput, and
 * glibc's pow is table-driven while a polynomial replacement is a long
 * serial dependency chain. Measuring both settles which one wins.
 *
 * Reports, for each implementation:
 *   throughput - independent inputs, pipeline free to overlap
 *   latency    - each call's input depends on the previous call's output
 *   max relative error against glibc pow over the real input range
 *
 * build: g++ -O3 -std=c++11 -o micro_pow micro_pow.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <vector>
#include <algorithm>

static double wtime()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1.0e-9 * ts.tv_nsec;
}

/* ---- exp2(y*log2(x)) using glibc's exp2 and log2 ---- */
static inline double pow_exp2log2(double x, double y)
{
  return exp2(y * log2(x));
}

/* ---- fully polynomial pow, x > 0 ---- */
static inline double pow_poly(double x, double y)
{
  union { double d; uint64_t u; } v;

  /* split x = m * 2^e with m in [1,2) */
  v.d = x;
  int e = (int)((v.u >> 52) & 0x7ff) - 1023;
  v.u = (v.u & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
  double m = v.d;

  /* fold m into [1/sqrt2, sqrt2) so the series argument stays small */
  if (m > 1.4142135623730951) { m *= 0.5; e += 1; }

  /* log2(m) = (2/ln2) * atanh(t), t = (m-1)/(m+1), |t| <= 0.1716 */
  double t = (m - 1.0) / (m + 1.0);
  double t2 = t * t;
  double s = t * (1.0 + t2*(1.0/3.0 + t2*(1.0/5.0 + t2*(1.0/7.0 + t2*(1.0/9.0)))));
  double log2x = (double) e + 2.8853900817779268 * s;

  /* 2^(y*log2x): split into integer n and fraction f in [-0.5, 0.5] */
  double p = y * log2x;
  double n = floor(p + 0.5);
  double f = p - n;

  /* 2^f, Taylor in f*ln2 to degree 7 */
  double r = 1.0 + f*(6.9314718055994531e-01 + f*(2.4022650695910071e-01 +
             f*(5.5504108664821580e-02 + f*(9.6181291076284772e-03 +
             f*(1.3333558146428443e-03 + f*(1.5403530393381610e-04 +
             f*1.5252733804059840e-05))))));

  v.d = r;
  v.u += ((uint64_t)(int64_t) n) << 52;
  return v.d;
}

int main()
{
  const size_t N = 1 << 20;
  std::vector<double> vr2(N), rn(N);

  /* vr2 as it actually arrives in test_collision: relative speeds of Ar at
     273 K squared, so roughly 1e3 .. 2e6 */
  unsigned s = 7777u;
  for (size_t i = 0; i < N; i++) {
    s = s*1664525u + 1013904223u; double u1 = ((s >> 8) + 1) * (1.0/16777217.0);
    s = s*1664525u + 1013904223u; double u2 = (s >> 8) * (1.0/16777216.0);
    double g = sqrt(-2.0*log(u1))*cos(6.283185307179586*u2);
    double v = 337.0 * g;
    vr2[i] = 3.0*v*v + 1.0;
    s = s*1664525u + 1013904223u;
    rn[i] = ((s >> 8) + 1) * (1.0/16777217.0);      /* uniform(0,1) */
  }

  const double P1 = 0.19;         /* 1 - omega, omega = 0.81 */
  const double P2 = 1.0/1.4;      /* 1/alpha */

  struct Impl { const char *name; double (*f)(double,double); };
  Impl impls[3] = {
    {"glibc pow",           [](double x,double y){ return pow(x,y); }},
    {"exp2(y*log2(x))",     pow_exp2log2},
    {"polynomial pow",      pow_poly},
  };

  printf("%-20s %14s %14s %16s\n",
         "implementation", "throughput", "latency", "max rel err");
  printf("%-20s %14s %14s %16s\n", "", "ns/call", "ns/call", "vs glibc");

  for (int k = 0; k < 3; k++) {
    double (*f)(double,double) = impls[k].f;

    /* throughput: independent inputs */
    double best_t = 1e30;
    for (int r = 0; r < 5; r++) {
      double acc = 0.0;
      double t0 = wtime();
      for (size_t i = 0; i < N; i++) acc += f(vr2[i], P1);
      double t1 = wtime();
      if (acc == 1.2345e-300) printf("x");
      best_t = std::min(best_t, 1e9*(t1-t0)/N);
    }

    /* latency: feed each result into the next call, as the collide kernel's
       dependency chain effectively does */
    double best_l = 1e30;
    for (int r = 0; r < 5; r++) {
      double z = 2.0;
      double t0 = wtime();
      for (size_t i = 0; i < N; i++) z = f(vr2[i] + z*1e-9, P1);
      double t1 = wtime();
      if (z == 1.2345e-300) printf("x");
      best_l = std::min(best_l, 1e9*(t1-t0)/N);
    }

    /* accuracy over both call sites */
    double maxerr = 0.0;
    for (size_t i = 0; i < N; i++) {
      double a = f(vr2[i], P1), b = pow(vr2[i], P1);
      maxerr = std::max(maxerr, fabs(a-b)/fabs(b));
      a = f(rn[i], P2); b = pow(rn[i], P2);
      maxerr = std::max(maxerr, fabs(a-b)/fabs(b));
    }

    printf("%-20s %14.2f %14.2f %16.3e\n",
           impls[k].name, best_t, best_l, maxerr);
  }
  return 0;
}
