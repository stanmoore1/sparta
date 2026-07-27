/* Measure this machine's roofline ceilings.
 *
 * The host is a KVM guest; lscpu reports the *host's* cache sizes, which are
 * not what these vCPUs actually get. So we measure rather than assume:
 *
 *   1. sustained bandwidth vs working-set size, for the two access mixes the
 *      SPARTA kernels actually use:
 *        - pure read (collide: reads particle data, writes only velocities)
 *        - read+write 1:1 (move/sort: streams particles in and out)
 *      A knee in the curve locates each cache level's effective capacity, and
 *      the plateau on each side is that level's bandwidth roof.
 *   2. peak FLOP/s for scalar, AVX2, and AVX-512 FMA chains (8 independent
 *      accumulators to saturate the FMA pipeline).
 *   3. throughput of the transcendentals the collide kernel leans on
 *      (pow, sqrt, sin, cos, exp2/log2) so the roofline's "FLOP" axis can be
 *      annotated with what a pow() actually costs in FMA-equivalents.
 *
 * Output is machine-readable "key value" lines for roofline.py to consume.
 *
 * build: g++ -O3 -march=native -o machine_peak machine_peak.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

static double wtime()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1.0e-9 * ts.tv_nsec;
}

/* ---------- bandwidth ---------- */

/* Pure-read: sum a buffer. Uses 8 accumulators so the adds don't serialize
   and we measure memory, not FP latency. */
static double bw_read(double *a, size_t n, int reps)
{
  double best = 0.0;
  for (int r = 0; r < reps; r++) {
    double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
    double t0 = wtime();
    for (size_t i = 0; i < n; i += 8) {
      s0 += a[i+0]; s1 += a[i+1]; s2 += a[i+2]; s3 += a[i+3];
      s4 += a[i+4]; s5 += a[i+5]; s6 += a[i+6]; s7 += a[i+7];
    }
    double t1 = wtime();
    /* keep the sums alive */
    if (s0+s1+s2+s3+s4+s5+s6+s7 == 1.2345e-300) printf("x");
    double bw = n * sizeof(double) / (t1 - t0);
    if (bw > best) best = bw;
  }
  return best;
}

/* Read+write 1:1, the mix move/sort see when streaming particles. */
static double bw_copy(double *a, double *b, size_t n, int reps)
{
  double best = 0.0;
  for (int r = 0; r < reps; r++) {
    double t0 = wtime();
    memcpy(b, a, n * sizeof(double));
    double t1 = wtime();
    double bw = 2.0 * n * sizeof(double) / (t1 - t0);
    if (bw > best) best = bw;
  }
  return best;
}

/* Random-stride gather, the access pattern of an unsorted particle list
   walking a per-cell linked list. Measures latency-bound throughput. */
static double bw_gather(double *a, const int *idx, size_t n, int reps)
{
  double best = 0.0;
  for (int r = 0; r < reps; r++) {
    double s = 0.0;
    double t0 = wtime();
    for (size_t i = 0; i < n; i++) s += a[idx[i]];
    double t1 = wtime();
    if (s == 1.2345e-300) printf("x");
    double bw = n * 64.0 / (t1 - t0);   /* one cache line touched per access */
    if (bw > best) best = bw;
  }
  return best;
}

/* ---------- compute peaks ---------- */

static double peak_scalar()
{
  double best = 0.0;
  const long iters = 200000000L;
  for (int r = 0; r < 3; r++) {
    double a0=1.0,a1=1.0,a2=1.0,a3=1.0,a4=1.0,a5=1.0,a6=1.0,a7=1.0;
    const double b = 1.0000001, c = 0.0000001;
    double t0 = wtime();
    for (long i = 0; i < iters; i += 8) {
      a0 = a0*b + c; a1 = a1*b + c; a2 = a2*b + c; a3 = a3*b + c;
      a4 = a4*b + c; a5 = a5*b + c; a6 = a6*b + c; a7 = a7*b + c;
    }
    double t1 = wtime();
    if (a0+a1+a2+a3+a4+a5+a6+a7 == 1.2345e-300) printf("x");
    double f = 2.0 * iters / (t1 - t0);   /* FMA = 2 flops */
    if (f > best) best = f;
  }
  return best;
}

#if defined(__AVX2__)
static double peak_avx2()
{
  double best = 0.0;
  const long iters = 50000000L;
  for (int r = 0; r < 3; r++) {
    __m256d a0=_mm256_set1_pd(1.0), a1=a0, a2=a0, a3=a0, a4=a0, a5=a0, a6=a0, a7=a0;
    __m256d b = _mm256_set1_pd(1.0000001), c = _mm256_set1_pd(1e-7);
    double t0 = wtime();
    for (long i = 0; i < iters; i++) {
      a0=_mm256_fmadd_pd(a0,b,c); a1=_mm256_fmadd_pd(a1,b,c);
      a2=_mm256_fmadd_pd(a2,b,c); a3=_mm256_fmadd_pd(a3,b,c);
      a4=_mm256_fmadd_pd(a4,b,c); a5=_mm256_fmadd_pd(a5,b,c);
      a6=_mm256_fmadd_pd(a6,b,c); a7=_mm256_fmadd_pd(a7,b,c);
    }
    double t1 = wtime();
    double out[4]; _mm256_storeu_pd(out, _mm256_add_pd(_mm256_add_pd(a0,a1),
                                    _mm256_add_pd(a2,a3)));
    if (out[0] == 1.2345e-300) printf("x");
    double f = 2.0 * 4.0 * 8.0 * iters / (t1 - t0);
    if (f > best) best = f;
  }
  return best;
}
#endif

#if defined(__AVX512F__)
static double peak_avx512()
{
  double best = 0.0;
  const long iters = 25000000L;
  for (int r = 0; r < 3; r++) {
    __m512d a0=_mm512_set1_pd(1.0), a1=a0, a2=a0, a3=a0, a4=a0, a5=a0, a6=a0, a7=a0;
    __m512d b = _mm512_set1_pd(1.0000001), c = _mm512_set1_pd(1e-7);
    double t0 = wtime();
    for (long i = 0; i < iters; i++) {
      a0=_mm512_fmadd_pd(a0,b,c); a1=_mm512_fmadd_pd(a1,b,c);
      a2=_mm512_fmadd_pd(a2,b,c); a3=_mm512_fmadd_pd(a3,b,c);
      a4=_mm512_fmadd_pd(a4,b,c); a5=_mm512_fmadd_pd(a5,b,c);
      a6=_mm512_fmadd_pd(a6,b,c); a7=_mm512_fmadd_pd(a7,b,c);
    }
    double t1 = wtime();
    double out[8]; _mm512_storeu_pd(out, _mm512_add_pd(_mm512_add_pd(a0,a1),
                                    _mm512_add_pd(a2,a3)));
    if (out[0] == 1.2345e-300) printf("x");
    double f = 2.0 * 8.0 * 8.0 * iters / (t1 - t0);
    if (f > best) best = f;
  }
  return best;
}
#endif

/* ---------- transcendental cost ---------- */

/* Throughput (not latency) of each function, in ns/call, measured over
   independent inputs so the pipeline can overlap them. */
template <typename F>
static double trans_ns(F f, const double *in, size_t n, int reps)
{
  double best = 1e30;
  for (int r = 0; r < reps; r++) {
    double s = 0.0;
    double t0 = wtime();
    for (size_t i = 0; i < n; i++) s += f(in[i]);
    double t1 = wtime();
    if (s == 1.2345e-300) printf("x");
    double ns = 1e9 * (t1 - t0) / n;
    if (ns < best) best = ns;
  }
  return best;
}

int main()
{
  printf("# SPARTA roofline ceilings, measured on this machine\n");

  /* ---- bandwidth vs working set ---- */
  const size_t maxn = 1u << 27;             /* 1 GiB of doubles */
  double *a = (double*) aligned_alloc(64, maxn * sizeof(double));
  double *b = (double*) aligned_alloc(64, maxn * sizeof(double));
  if (!a || !b) { fprintf(stderr, "alloc failed\n"); return 1; }
  for (size_t i = 0; i < maxn; i++) { a[i] = 1.0 + 1e-9*i; b[i] = 0.0; }

  printf("# bytes_working_set  read_GBs  copy_GBs\n");
  for (size_t kb = 8; kb <= (maxn*sizeof(double))/1024; kb *= 2) {
    size_t n = (kb * 1024) / sizeof(double);
    if (n < 64) continue;
    /* more reps for small sizes so the timer resolution holds up */
    int reps = (kb < 1024) ? 200 : (kb < 65536 ? 20 : 4);
    double rd = bw_read(a, n, reps);
    double cp = bw_copy(a, b, n/2, reps);   /* n/2 each way = same footprint */
    printf("BW %zu %.3f %.3f\n", kb*1024, rd/1e9, cp/1e9);
    fflush(stdout);
  }

  /* ---- random gather ---- */
  {
    size_t n = 4u << 20;
    std::vector<int> idx(n);
    for (size_t i = 0; i < n; i++) idx[i] = (int)((i * 2654435761u) % n);
    printf("GATHER_GBs %.3f\n", bw_gather(a, idx.data(), n, 3)/1e9);
  }

  free(a); free(b);

  /* ---- compute peaks ---- */
  printf("PEAK_SCALAR_GFLOPs %.3f\n", peak_scalar()/1e9);
#if defined(__AVX2__)
  printf("PEAK_AVX2_GFLOPs %.3f\n", peak_avx2()/1e9);
#endif
#if defined(__AVX512F__)
  printf("PEAK_AVX512_GFLOPs %.3f\n", peak_avx512()/1e9);
#endif

  /* ---- transcendentals ---- */
  {
    const size_t n = 1u << 20;
    std::vector<double> in(n), in01(n);
    for (size_t i = 0; i < n; i++) {
      in[i]   = 1.0 + 1e5 * ((double)i / n);       /* vr2-like magnitudes */
      in01[i] = 1e-6 + 0.999998 * ((double)i / n); /* uniform(0,1)-like */
    }
    double ns_fma;
    {   /* calibrate: ns per scalar FMA, for the FMA-equivalent conversion */
      double t0 = wtime(); double s = 1.0;
      double q0=1,q1=1,q2=1,q3=1;
      for (size_t i = 0; i < n; i++) {
        q0 = q0*1.0000001 + 1e-7; q1 = q1*1.0000001 + 1e-7;
        q2 = q2*1.0000001 + 1e-7; q3 = q3*1.0000001 + 1e-7;
      }
      double t1 = wtime(); s = q0+q1+q2+q3;
      if (s == 1.2345e-300) printf("x");
      ns_fma = 1e9 * (t1 - t0) / (4.0 * n);
    }
    printf("NS_PER_FMA %.4f\n", ns_fma);

    double p_pow  = trans_ns([](double x){ return pow(x, 0.19); },  in.data(),   n, 5);
    double p_powa = trans_ns([](double x){ return pow(x, 1.0/1.4); }, in01.data(), n, 5);
    double p_sqrt = trans_ns([](double x){ return sqrt(x); },        in.data(),   n, 5);
    double p_sin  = trans_ns([](double x){ return sin(x); },         in01.data(), n, 5);
    double p_cos  = trans_ns([](double x){ return cos(x); },         in01.data(), n, 5);
    double p_e2l2 = trans_ns([](double x){ return exp2(0.19*log2(x)); }, in.data(), n, 5);
    double p_log2 = trans_ns([](double x){ return log2(x); },        in.data(),   n, 5);
    double p_exp2 = trans_ns([](double x){ return exp2(x*0.001); },  in01.data(), n, 5);

    printf("NS_POW_019 %.4f\n",   p_pow);
    printf("NS_POW_ALPHA %.4f\n", p_powa);
    printf("NS_SQRT %.4f\n",      p_sqrt);
    printf("NS_SIN %.4f\n",       p_sin);
    printf("NS_COS %.4f\n",       p_cos);
    printf("NS_EXP2LOG2 %.4f\n",  p_e2l2);
    printf("NS_LOG2 %.4f\n",      p_log2);
    printf("NS_EXP2 %.4f\n",      p_exp2);

    printf("# FMA-equivalents: pow=%.1f powalpha=%.1f sqrt=%.1f sin=%.1f cos=%.1f exp2log2=%.1f\n",
           p_pow/ns_fma, p_powa/ns_fma, p_sqrt/ns_fma, p_sin/ns_fma, p_cos/ns_fma, p_e2l2/ns_fma);
  }

  return 0;
}
