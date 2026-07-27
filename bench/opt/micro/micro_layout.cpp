/* Particle memory layout study for the SPARTA DSMC timestep.
 *
 * Round 2 established that the step is limited by how many bytes of particle
 * record it drags past the cache, not by any kernel's arithmetic. This asks the
 * two questions that follow from that:
 *
 *   1. What layout?  AoS is what SPARTA uses. SoA vectorises the mover and lets
 *      each field stream independently, but scatters a single particle's data
 *      across six arrays -- and the collide kernel touches two *random*
 *      particles per cell, which is the access pattern SoA is worst at. AoSoA
 *      (the Cabana layout) is the usual compromise: vector-length blocks of
 *      struct-of-arrays, so a block is SIMD-friendly while a single particle
 *      stays within one block.
 *
 *   2. Do we need to move the particles at all?  SPARTA reorders the whole
 *      particle array so each cell's particles are contiguous, then collides.
 *      But collide only touches about two particles per cell out of ten -- so
 *      the reorder moves ten records to make two accesses cheap. The
 *      alternative is to bin *indices* only: a CSR list of 4-byte particle
 *      indices per cell, with the particles left where they are. That trades
 *      ~192 B/particle of permutation traffic for two random gathers per cell.
 *      Round 1's flat cell lookup already made the mover largely insensitive to
 *      particle order, which is what makes this worth testing.
 *
 * Both questions are crossed, because they interact: index-only binning makes
 * collide gather-bound, which is precisely where SoA hurts and AoS helps.
 *
 * All variants run the same physics (uniform grid, reflective box, Ar, VSS,
 * NTC). They consume random numbers in the same order, so ncoll should agree
 * closely; equilibrium temperature must hold at 273.15 K.
 *
 * build: g++ -O3 -march=native -std=c++11 -o micro_layout micro_layout.cpp
 * usage: ./micro_layout [nx ny nz npercell nsteps reorder]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <vector>
#include <algorithm>

#define MY_PI 3.14159265358979323846
#define EPSZERO 1.0e-14

static double wtime()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1.0e-9 * ts.tv_nsec;
}

static const double KB    = 1.380658e-23;
static const double MASS  = 6.63e-26;
static const double DIAM  = 4.11e-10;
static const double OMEGA = 0.81;
static const double TREF  = 273.15;
static const double ALPHA = 1.4;
static const double TEMP0 = 273.15;
static const double CELLL = 1.0e-5;
static const double DT    = 7.0e-9;
static const double FNUM  = 7.07043e6;

/* ---------------- RNG ---------------- */

#define MBIG 1000000000
#define MSEED 161803398
#define RFAC (1.0/MBIG)

struct Rng {
  int seed, inext, inextp, ma[56];
  void init(int s) {
    seed = s;
    int i, ii, k, mj, mk;
    mj = labs(MSEED - labs(seed)); mj %= MBIG;
    ma[55] = mj; mk = 1;
    for (i = 1; i <= 54; i++) {
      ii = (21*i) % 55; ma[ii] = mk; mk = mj - mk;
      if (mk < 0) mk += MBIG;
      mj = ma[ii];
    }
    for (k = 0; k < 4; k++)
      for (i = 1; i <= 55; i++) {
        ma[i] -= ma[1 + (i+30) % 55];
        if (ma[i] < 0) ma[i] += MBIG;
      }
    inext = 0; inextp = 31;
  }
  inline double uniform() {
    int mj; double rn;
    while (1) {
      if (++inext == 56) inext = 1;
      if (++inextp == 56) inextp = 1;
      mj = ma[inext] - ma[inextp];
      if (mj < 0) mj += MBIG;
      ma[inext] = mj;
      rn = mj*RFAC;
      if (rn > 0.0 && rn < 1.0) break;
    }
    return rn;
  }
};

/* ---------------- VSS ---------------- */

struct VssConst {
  double mr, prefactor, omega1, alpha_r, vrm0;
  VssConst() {
    mr = MASS*MASS/(MASS+MASS);
    double cxs = MY_PI*DIAM*DIAM;
    prefactor = cxs * pow(2.0*KB*TREF/mr, OMEGA-0.5) / tgamma(2.5-OMEGA);
    omega1 = 1.0 - OMEGA;
    alpha_r = 1.0/ALPHA;
    vrm0 = 2.0*cxs*sqrt(2.0*KB*TEMP0/MASS);
  }
};

/* pair kernel on plain doubles; each layout copies velocities in and out */
static inline int vss_pair(const VssConst &C, Rng &rng, double &vremax,
                           double *vi, double *vj)
{
  double du = vi[0]-vj[0], dv = vi[1]-vj[1], dw = vi[2]-vj[2];
  double vr2 = du*du + dv*dv + dw*dw;
  if (vr2 < EPSZERO) return 0;
  double vre = pow(vr2, C.omega1) * C.prefactor;
  vremax = std::max(vre, vremax);
  if (vre/vremax < rng.uniform()) return 0;

  double vr = sqrt(vr2);
  double etrans = 0.5 * C.mr * vr2;
  double ucmf = 0.5*(vi[0]+vj[0]);
  double vcmf = 0.5*(vi[1]+vj[1]);
  double wcmf = 0.5*(vi[2]+vj[2]);

  double eps = rng.uniform() * 2*MY_PI;
  double scale = sqrt((2.0*etrans)/(C.mr*vr2));
  double cosX = 2.0*pow(rng.uniform(), C.alpha_r) - 1.0;
  double sinX = sqrt(1.0 - cosX*cosX);
  double sineps = sin(eps), coseps = cos(eps);

  double ua, vb, wc;
  double d = sqrt(dv*dv + dw*dw);
  if (d > 1.0e-6) {
    ua = scale * ( cosX*du + sinX*d*sineps );
    vb = scale * ( cosX*dv + sinX*(vr*dw*coseps - du*dv*sineps)/d );
    wc = scale * ( cosX*dw - sinX*(vr*dv*coseps + du*dw*sineps)/d );
  } else {
    ua = scale * ( cosX*du );
    vb = scale * ( sinX*du*coseps );
    wc = scale * ( sinX*du*sineps );
  }
  vi[0] = ucmf + 0.5*ua; vi[1] = vcmf + 0.5*vb; vi[2] = wcmf + 0.5*wc;
  vj[0] = ucmf - 0.5*ua; vj[1] = vcmf - 0.5*vb; vj[2] = wcmf - 0.5*wc;
  return 1;
}

/* ---------------- geometry ---------------- */

struct Geom {
  int nx, ny, nz, ncell;
  double lo, hix, hiy, hiz, invd, volume;
  void init(int a, int b, int c) {
    nx = a; ny = b; nz = c; ncell = a*b*c;
    lo = 0.0; hix = a*CELLL; hiy = b*CELLL; hiz = c*CELLL;
    invd = 1.0/CELLL; volume = CELLL*CELLL*CELLL;
  }
  inline int cell_of(double x, double y, double z) const {
    int i = (int)(x*invd), j = (int)(y*invd), k = (int)(z*invd);
    if (i < 0) i = 0; else if (i >= nx) i = nx-1;
    if (j < 0) j = 0; else if (j >= ny) j = ny-1;
    if (k < 0) k = 0; else if (k >= nz) k = nz-1;
    return (k*ny + j)*nx + i;
  }
};

/* =================== layouts ===================
 *
 * Each provides:
 *   resize(n), setp(i,x,v), getv(i,v), setv(i,v), cellof(i), setcell(i,c)
 *   move(g)          advance every particle, store its new cell, return nothing
 *   permute(order)   apply order[] so particle order[m] ends up at slot m
 */

/* ---- L0: array of structs, 96 bytes, exactly SPARTA's OnePart ---- */
struct AoS96 {
  struct alignas(16) P {
    int id, isp, icell, flag;
    double x[3], v[3];
    double erot, evib, dtremain, weight;
  };
  std::vector<P> a, b;
  long n;
  double bytes_per() const { return sizeof(P); }

  void resize(long n_) { n = n_; a.resize(n); b.resize(n); memset(a.data(),0,n*sizeof(P)); }
  void setp(long i, const double *x, const double *v) {
    for (int c = 0; c < 3; c++) { a[i].x[c] = x[c]; a[i].v[c] = v[c]; }
  }
  inline void getv(long i, double *v) const { v[0]=a[i].v[0]; v[1]=a[i].v[1]; v[2]=a[i].v[2]; }
  inline void setv(long i, const double *v) { a[i].v[0]=v[0]; a[i].v[1]=v[1]; a[i].v[2]=v[2]; }
  inline int cellof(long i) const { return a[i].icell; }

  void move(const Geom &g) {
    P *p = a.data();
    for (long i = 0; i < n; i++) {
      double x0 = p[i].x[0] + DT*p[i].v[0];
      double x1 = p[i].x[1] + DT*p[i].v[1];
      double x2 = p[i].x[2] + DT*p[i].v[2];
      if (x0 < g.lo) { x0 = -x0; p[i].v[0] = -p[i].v[0]; }
      else if (x0 > g.hix) { x0 = 2*g.hix - x0; p[i].v[0] = -p[i].v[0]; }
      if (x1 < g.lo) { x1 = -x1; p[i].v[1] = -p[i].v[1]; }
      else if (x1 > g.hiy) { x1 = 2*g.hiy - x1; p[i].v[1] = -p[i].v[1]; }
      if (x2 < g.lo) { x2 = -x2; p[i].v[2] = -p[i].v[2]; }
      else if (x2 > g.hiz) { x2 = 2*g.hiz - x2; p[i].v[2] = -p[i].v[2]; }
      p[i].x[0] = x0; p[i].x[1] = x1; p[i].x[2] = x2;
      p[i].icell = g.cell_of(x0, x1, x2);
    }
  }
  void permute(const int *order) {
    for (long m = 0; m < n; m++) b[m] = a[order[m]];
    a.swap(b);
  }
};

/* ---- L1: array of structs, 64 bytes, dead fields removed ---- */
struct AoS64 {
  struct alignas(16) P {
    int id, isp, icell, flag;
    double x[3], v[3];
  };
  std::vector<P> a, b;
  long n;
  double bytes_per() const { return sizeof(P); }

  void resize(long n_) { n = n_; a.resize(n); b.resize(n); memset(a.data(),0,n*sizeof(P)); }
  void setp(long i, const double *x, const double *v) {
    for (int c = 0; c < 3; c++) { a[i].x[c] = x[c]; a[i].v[c] = v[c]; }
  }
  inline void getv(long i, double *v) const { v[0]=a[i].v[0]; v[1]=a[i].v[1]; v[2]=a[i].v[2]; }
  inline void setv(long i, const double *v) { a[i].v[0]=v[0]; a[i].v[1]=v[1]; a[i].v[2]=v[2]; }
  inline int cellof(long i) const { return a[i].icell; }

  void move(const Geom &g) {
    P *p = a.data();
    for (long i = 0; i < n; i++) {
      double x0 = p[i].x[0] + DT*p[i].v[0];
      double x1 = p[i].x[1] + DT*p[i].v[1];
      double x2 = p[i].x[2] + DT*p[i].v[2];
      if (x0 < g.lo) { x0 = -x0; p[i].v[0] = -p[i].v[0]; }
      else if (x0 > g.hix) { x0 = 2*g.hix - x0; p[i].v[0] = -p[i].v[0]; }
      if (x1 < g.lo) { x1 = -x1; p[i].v[1] = -p[i].v[1]; }
      else if (x1 > g.hiy) { x1 = 2*g.hiy - x1; p[i].v[1] = -p[i].v[1]; }
      if (x2 < g.lo) { x2 = -x2; p[i].v[2] = -p[i].v[2]; }
      else if (x2 > g.hiz) { x2 = 2*g.hiz - x2; p[i].v[2] = -p[i].v[2]; }
      p[i].x[0] = x0; p[i].x[1] = x1; p[i].x[2] = x2;
      p[i].icell = g.cell_of(x0, x1, x2);
    }
  }
  void permute(const int *order) {
    for (long m = 0; m < n; m++) b[m] = a[order[m]];
    a.swap(b);
  }
};

/* ---- L2: struct of arrays, doubles ---- */
struct SoAd {
  std::vector<double> x0,x1,x2,v0,v1,v2;
  std::vector<double> y0,y1,y2,w0,w1,w2;    /* permutation targets */
  std::vector<int> ic, ic2;
  long n;
  double bytes_per() const { return 6*sizeof(double) + sizeof(int); }

  void resize(long n_) {
    n = n_;
    x0.assign(n,0); x1.assign(n,0); x2.assign(n,0);
    v0.assign(n,0); v1.assign(n,0); v2.assign(n,0);
    y0.resize(n); y1.resize(n); y2.resize(n);
    w0.resize(n); w1.resize(n); w2.resize(n);
    ic.assign(n,0); ic2.resize(n);
  }
  void setp(long i, const double *x, const double *v) {
    x0[i]=x[0]; x1[i]=x[1]; x2[i]=x[2];
    v0[i]=v[0]; v1[i]=v[1]; v2[i]=v[2];
  }
  inline void getv(long i, double *v) const { v[0]=v0[i]; v[1]=v1[i]; v[2]=v2[i]; }
  inline void setv(long i, const double *v) { v0[i]=v[0]; v1[i]=v[1]; v2[i]=v[2]; }
  inline int cellof(long i) const { return ic[i]; }

  void move(const Geom &g) {
    double *X0=x0.data(), *X1=x1.data(), *X2=x2.data();
    double *V0=v0.data(), *V1=v1.data(), *V2=v2.data();
    int *IC = ic.data();
    for (long i = 0; i < n; i++) {
      double a0 = X0[i] + DT*V0[i];
      double a1 = X1[i] + DT*V1[i];
      double a2 = X2[i] + DT*V2[i];
      if (a0 < g.lo) { a0 = -a0; V0[i] = -V0[i]; }
      else if (a0 > g.hix) { a0 = 2*g.hix - a0; V0[i] = -V0[i]; }
      if (a1 < g.lo) { a1 = -a1; V1[i] = -V1[i]; }
      else if (a1 > g.hiy) { a1 = 2*g.hiy - a1; V1[i] = -V1[i]; }
      if (a2 < g.lo) { a2 = -a2; V2[i] = -V2[i]; }
      else if (a2 > g.hiz) { a2 = 2*g.hiz - a2; V2[i] = -V2[i]; }
      X0[i]=a0; X1[i]=a1; X2[i]=a2;
      IC[i] = g.cell_of(a0,a1,a2);
    }
  }
  void permute(const int *order) {
    for (long m = 0; m < n; m++) {
      int s = order[m];
      y0[m]=x0[s]; y1[m]=x1[s]; y2[m]=x2[s];
      w0[m]=v0[s]; w1[m]=v1[s]; w2[m]=v2[s];
      ic2[m]=ic[s];
    }
    x0.swap(y0); x1.swap(y1); x2.swap(y2);
    v0.swap(w0); v1.swap(w1); v2.swap(w2);
    ic.swap(ic2);
  }
};

/* ---- L3/L4: AoSoA, the Cabana layout: blocks of V struct-of-arrays ---- */
template <int V>
struct AoSoA {
  struct alignas(64) Blk {
    double x[3][V];
    double v[3][V];
    int icell[V];
    int id[V];
  };
  /* std::vector does not honour over-aligned types before C++17, and with
     -march=native the compiler emits aligned vector loads against the declared
     alignment, so the blocks are allocated by hand */
  Blk *a, *b;
  long n, nblk;
  double bytes_per() const { return (double)sizeof(Blk)/V; }

  AoSoA() : a(NULL), b(NULL), n(0), nblk(0) {}
  ~AoSoA() { free(a); free(b); }

  void resize(long n_) {
    n = n_; nblk = (n + V - 1)/V;
    size_t bytes = (size_t)nblk*sizeof(Blk);
    if (posix_memalign((void**)&a, 64, bytes) != 0) { fprintf(stderr,"alloc failed\n"); exit(1); }
    if (posix_memalign((void**)&b, 64, bytes) != 0) { fprintf(stderr,"alloc failed\n"); exit(1); }
    memset(a, 0, bytes);
    memset(b, 0, bytes);
  }
  void setp(long i, const double *x, const double *v) {
    Blk &B = a[i/V]; int l = i%V;
    for (int c = 0; c < 3; c++) { B.x[c][l] = x[c]; B.v[c][l] = v[c]; }
  }
  inline void getv(long i, double *v) const {
    const Blk &B = a[i/V]; int l = i%V;
    v[0]=B.v[0][l]; v[1]=B.v[1][l]; v[2]=B.v[2][l];
  }
  inline void setv(long i, const double *v) {
    Blk &B = a[i/V]; int l = i%V;
    B.v[0][l]=v[0]; B.v[1][l]=v[1]; B.v[2][l]=v[2];
  }
  inline int cellof(long i) const { return a[i/V].icell[i%V]; }

  void move(const Geom &g) {
    Blk *p = a;
    double lo = g.lo, hix = g.hix, hiy = g.hiy, hiz = g.hiz;
    for (long bidx = 0; bidx < nblk; bidx++) {
      Blk &B = p[bidx];
      /* straight-line over V lanes: the compiler can vectorise this */
      for (int l = 0; l < V; l++) {
        double a0 = B.x[0][l] + DT*B.v[0][l];
        double a1 = B.x[1][l] + DT*B.v[1][l];
        double a2 = B.x[2][l] + DT*B.v[2][l];
        if (a0 < lo) { a0 = -a0; B.v[0][l] = -B.v[0][l]; }
        else if (a0 > hix) { a0 = 2*hix - a0; B.v[0][l] = -B.v[0][l]; }
        if (a1 < lo) { a1 = -a1; B.v[1][l] = -B.v[1][l]; }
        else if (a1 > hiy) { a1 = 2*hiy - a1; B.v[1][l] = -B.v[1][l]; }
        if (a2 < lo) { a2 = -a2; B.v[2][l] = -B.v[2][l]; }
        else if (a2 > hiz) { a2 = 2*hiz - a2; B.v[2][l] = -B.v[2][l]; }
        B.x[0][l]=a0; B.x[1][l]=a1; B.x[2][l]=a2;
      }
      for (int l = 0; l < V; l++)
        B.icell[l] = g.cell_of(B.x[0][l], B.x[1][l], B.x[2][l]);
    }
  }
  void permute(const int *order) {
    for (long m = 0; m < n; m++) {
      int s = order[m];
      const Blk &S = a[s/V]; int sl = s%V;
      Blk &D = b[m/V]; int dl = m%V;
      for (int c = 0; c < 3; c++) { D.x[c][dl] = S.x[c][sl]; D.v[c][dl] = S.v[c][sl]; }
      D.icell[dl] = S.icell[sl];
    }
    std::swap(a, b);
  }
};

/* ---------------- the timestep, generic over layout and binning ----------
 *
 * PERMUTE = 1: reorder the particles so each cell's are contiguous (SPARTA)
 * PERMUTE = 0: leave particles where they are and bin *indices* only
 */

struct Result {
  double t_move, t_bin, t_collide, t_total;
  double temp; long ncoll; double mem_mb;
};

template <int PERMUTE, class S>
static Result run(const Geom &g, int npercell, int nsteps, int reorder)
{
  long nlocal = (long)g.ncell*npercell;
  S s; s.resize(nlocal);
  VssConst C;

  std::vector<int> count(g.ncell), first(g.ncell), cursor(g.ncell);
  std::vector<int> order(nlocal);         /* permutation, or CSR index list */
  std::vector<double> vremax(g.ncell, C.vrm0), remain(g.ncell, 0.0);

  Rng rng; rng.init(12345);
  {
    double vth = sqrt(KB*TEMP0/MASS);
    long m = 0;
    for (int k = 0; k < g.nz; k++)
      for (int j = 0; j < g.ny; j++)
        for (int i = 0; i < g.nx; i++)
          for (int p = 0; p < npercell; p++) {
            double x[3], v[3];
            x[0] = (i + rng.uniform())*CELLL;
            x[1] = (j + rng.uniform())*CELLL;
            x[2] = (k + rng.uniform())*CELLL;
            for (int c = 0; c < 3; c++) {
              double u1 = rng.uniform(), u2 = rng.uniform();
              v[c] = vth*sqrt(-2.0*log(u1))*cos(2*MY_PI*u2);
            }
            s.setp(m++, x, v);
          }
  }

  Result r; memset(&r, 0, sizeof(Result));
  r.mem_mb = (PERMUTE ? 2.0 : 1.0)*nlocal*s.bytes_per()/1048576.0
             + nlocal*sizeof(int)/1048576.0;

  for (int step = 0; step < nsteps; step++) {
    double t0 = wtime();
    s.move(g);
    r.t_move += wtime() - t0;

    t0 = wtime();
    memset(count.data(), 0, g.ncell*sizeof(int));
    for (long i = 0; i < nlocal; i++) count[s.cellof(i)]++;
    long m = 0;
    for (int c = 0; c < g.ncell; c++) { first[c] = m; cursor[c] = m; m += count[c]; }
    for (long i = 0; i < nlocal; i++) order[cursor[s.cellof(i)]++] = (int)i;

    if (PERMUTE && (reorder == 1 || step % reorder == 0)) {
      s.permute(order.data());
      for (long i = 0; i < nlocal; i++) order[i] = (int)i;
    }
    r.t_bin += wtime() - t0;

    t0 = wtime();
    for (int c = 0; c < g.ncell; c++) {
      int np = count[c];
      if (np <= 1) continue;
      int f = first[c];
      double vrm = vremax[c];
      double att = 0.5*np*(np-1)*vrm*DT*FNUM/g.volume + remain[c];
      int natt = (int) att;
      remain[c] = att - natt;
      for (int t = 0; t < natt; t++) {
        int i = (int)(np*rng.uniform());
        int j = (int)(np*rng.uniform());
        while (i == j) j = (int)(np*rng.uniform());
        long pi = order[f+i], pj = order[f+j];
        double vi[3], vj[3];
        s.getv(pi, vi); s.getv(pj, vj);
        if (vss_pair(C, rng, vrm, vi, vj)) {
          s.setv(pi, vi); s.setv(pj, vj);
          r.ncoll++;
        }
      }
      vremax[c] = vrm;
    }
    r.t_collide += wtime() - t0;
  }

  r.t_total = r.t_move + r.t_bin + r.t_collide;
  double sum = 0.0;
  for (long i = 0; i < nlocal; i++) {
    double v[3]; s.getv(i, v);
    sum += v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
  }
  r.temp = MASS*sum/(3.0*KB*nlocal);
  return r;
}

static void report(const char *name, const Result &r, long nlocal, int nsteps,
                   double base)
{
  double per = 1e9*r.t_total/((double)nlocal*nsteps);
  printf("%-40s %7.2f %7.2fx | %6.2f %6.2f %6.2f | %7.2f %9ld %6.0f\n",
         name, per, base > 0 ? base/per : 1.0,
         1e9*r.t_move/((double)nlocal*nsteps),
         1e9*r.t_bin/((double)nlocal*nsteps),
         1e9*r.t_collide/((double)nlocal*nsteps),
         r.temp, r.ncoll, r.mem_mb);
}

int main(int argc, char **argv)
{
  int nx = (argc > 1) ? atoi(argv[1]) : 40;
  int ny = (argc > 2) ? atoi(argv[2]) : 50;
  int nz = (argc > 3) ? atoi(argv[3]) : 50;
  int npercell = (argc > 4) ? atoi(argv[4]) : 10;
  int nsteps = (argc > 5) ? atoi(argv[5]) : 20;
  int reorder = (argc > 6) ? atoi(argv[6]) : 2;

  Geom g; g.init(nx, ny, nz);
  long nlocal = (long)g.ncell*npercell;

  printf("# micro_layout: %dx%dx%d = %d cells, %ld particles, %d steps, "
         "reorder every %d\n", nx, ny, nz, g.ncell, nlocal, nsteps, reorder);
  printf("%-40s %7s %8s | %6s %6s %6s | %7s %9s %6s\n",
         "layout / binning", "ns/p/s", "speedup", "move", "bin", "coll",
         "T (K)", "ncoll", "MB");

  printf("-- particles permuted so cells are contiguous (what SPARTA does) --\n");
  Result p96 = run<1,AoS96>(g, npercell, nsteps, reorder);
  double base = 1e9*p96.t_total/((double)nlocal*nsteps);
  report("AoS 96 B", p96, nlocal, nsteps, 0);
  report("AoS 64 B", run<1,AoS64>(g, npercell, nsteps, reorder), nlocal, nsteps, base);
  report("SoA (doubles)", run<1,SoAd>(g, npercell, nsteps, reorder), nlocal, nsteps, base);
  report("AoSoA V=8", run<1,AoSoA<8> >(g, npercell, nsteps, reorder), nlocal, nsteps, base);
  report("AoSoA V=16", run<1,AoSoA<16> >(g, npercell, nsteps, reorder), nlocal, nsteps, base);

  printf("-- indices binned, particles never moved --\n");
  report("AoS 96 B, index-only", run<0,AoS96>(g, npercell, nsteps, reorder), nlocal, nsteps, base);
  report("AoS 64 B, index-only", run<0,AoS64>(g, npercell, nsteps, reorder), nlocal, nsteps, base);
  report("SoA, index-only", run<0,SoAd>(g, npercell, nsteps, reorder), nlocal, nsteps, base);
  report("AoSoA V=8, index-only", run<0,AoSoA<8> >(g, npercell, nsteps, reorder), nlocal, nsteps, base);

  printf("\nreference equilibrium temperature %.2f K\n", TEMP0);
  return 0;
}
