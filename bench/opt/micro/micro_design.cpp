/* Design-space exploration for the SPARTA DSMC timestep.
 *
 * Round 1 reached 1.20x and left every kernel far below both roofs. Sweeping
 * the problem size explains why: the same work costs 29.3 ns/particle/step at
 * 10K particles (0.96 MB, L2-resident) and 63.5 ns at 1M (96 MB). The step is
 * not short of FLOPs or of bandwidth — it makes three or four separate
 * streaming passes over the particle array, so each particle is fetched from
 * DRAM several times per timestep. Closing that gap is worth ~2.2x.
 *
 * This benchmark rebuilds the whole timestep (move + bin + collide) under
 * progressively more aggressive restructurings and measures each:
 *
 *   D0  three passes, branchy move            mirrors current SPARTA
 *   D1  three passes, branchless move         removes the slow path entirely
 *   D2  D1 + tile-major cell numbering        same passes, better locality
 *   D3  fused: move+bin+collide per tile      one DRAM pass per step
 *   D4  D3 with a mesh-free binning           no cell array at all
 *
 * Two structural ideas are under test, and they are independent:
 *
 *  1. Branchless move. SPARTA's optmove falls back to the general cell-by-cell
 *     mover for anything leaving the box -- ~1% of particles, but 40% of the
 *     program's branch mispredictions. With a uniform grid and no surfaces the
 *     destination cell is floor((x-lo)/d) and a reflection is x' = 2*lo - x,
 *     v' = -v. At this timestep a particle cannot cross two boundaries, so
 *     there is nothing to iterate and the mover becomes straight-line code.
 *
 *  2. Tiling and fusion. Number cells tile-major so a block of cells whose
 *     particles fit in L2 is contiguous in memory, then run move, bin and
 *     collide on one tile before touching the next. Particles that leave a
 *     tile go to a small per-tile inbox merged at the end of the step.
 *
 * Designs consume random numbers in different orders, so they are not bitwise
 * comparable. Each reports its equilibrium temperature and collision count;
 * those must agree to within sampling noise or the design is wrong.
 *
 * build: g++ -O3 -march=native -std=c++11 -o micro_design micro_design.cpp
 * usage: ./micro_design [nx ny nz npercell nsteps tile]
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

/* ---------------- physical constants of the in.collide benchmark -------- */

static const double KB    = 1.380658e-23;
static const double MASS  = 6.63e-26;      /* Ar */
static const double DIAM  = 4.11e-10;
static const double OMEGA = 0.81;
static const double TREF  = 273.15;
static const double ALPHA = 1.4;
static const double TEMP0 = 273.15;
static const double CELLL = 1.0e-5;
static const double DT    = 7.0e-9;
static const double FNUM  = 7.07043e6;

/* ---------------- RanKnuth, as SPARTA uses ---------------- */

#define MBIG 1000000000
#define MSEED 161803398
#define RFAC (1.0/MBIG)

struct Rng {
  int seed, inext, inextp;
  int ma[56];

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

/* ---------------- VSS collision constants and pair kernel ---------------- */

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

/* one NTC attempt on a pair; same arithmetic as CollideVSS, equal masses */
template <typename T>
static inline int vss_pair(const VssConst &C, Rng &rng, double &vremax,
                           T *vi, T *vj)
{
  double du = (double)vi[0]-vj[0], dv = (double)vi[1]-vj[1],
         dw = (double)vi[2]-vj[2];
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

  vi[0] = (T)(ucmf + 0.5*ua); vi[1] = (T)(vcmf + 0.5*vb); vi[2] = (T)(wcmf + 0.5*wc);
  vj[0] = (T)(ucmf - 0.5*ua); vj[1] = (T)(vcmf - 0.5*vb); vj[2] = (T)(wcmf - 0.5*wc);
  return 1;
}

/* ---------------- geometry and cell numbering ---------------- */

/* TILED = 0: cells numbered x fastest, as create_grid does.
   TILED = 1: cells numbered tile-major, so a TB x TB x TB block of cells is
              one contiguous run of cell indices -- and therefore, once
              particles are sorted by cell, one contiguous run of particles. */
struct Geom {
  int nx, ny, nz, ncell, tb;
  int ntx, nty, ntz;
  double lo[3], hi[3], invd, volume;
  std::vector<int> lin2cell;      /* (k*ny+j)*nx+i  ->  cell index used here */

  void init(int nx_, int ny_, int nz_, int tb_, int tiled) {
    nx = nx_; ny = ny_; nz = nz_; tb = tb_;
    ncell = nx*ny*nz;
    lo[0] = lo[1] = lo[2] = 0.0;
    hi[0] = nx*CELLL; hi[1] = ny*CELLL; hi[2] = nz*CELLL;
    invd = 1.0/CELLL;
    volume = CELLL*CELLL*CELLL;
    ntx = (nx + tb - 1)/tb; nty = (ny + tb - 1)/tb; ntz = (nz + tb - 1)/tb;

    lin2cell.resize(ncell);
    if (!tiled) {
      for (int c = 0; c < ncell; c++) lin2cell[c] = c;
    } else {
      /* assign cell indices tile by tile, in raster order inside each tile */
      int next = 0;
      for (int tk = 0; tk < ntz; tk++)
        for (int tj = 0; tj < nty; tj++)
          for (int ti = 0; ti < ntx; ti++)
            for (int k = tk*tb; k < std::min((tk+1)*tb, nz); k++)
              for (int j = tj*tb; j < std::min((tj+1)*tb, ny); j++)
                for (int i = ti*tb; i < std::min((ti+1)*tb, nx); i++)
                  lin2cell[(k*ny + j)*nx + i] = next++;
    }
  }

  inline int cell_of(double x, double y, double z) const {
    int i = (int)((x - lo[0])*invd);
    int j = (int)((y - lo[1])*invd);
    int k = (int)((z - lo[2])*invd);
    if (i < 0) i = 0; else if (i >= nx) i = nx-1;
    if (j < 0) j = 0; else if (j >= ny) j = ny-1;
    if (k < 0) k = 0; else if (k >= nz) k = nz-1;
    return lin2cell[(k*ny + j)*nx + i];
  }

  int ntile() const { return ntx*nty*ntz; }
};

/* ---------------- particle record, 96 bytes as in SPARTA ---------------- */

struct alignas(16) OnePart {          /* 96 B, exactly SPARTA's Particle::OnePart */
  int id, ispecies, icell, flag;
  double x[3], v[3];
  double erot, evib, dtremain, weight;
  typedef double real;
};

/* Same fields move/bin/collide actually touch in this benchmark, with the
   four that are dead for a monatomic unweighted no-surface run removed:
   erot and evib are always zero for Ar, dtremain is only live inside the
   general mover, and weight only matters with grid-based particle weighting.
   Exactly one 64-byte cache line. */
struct alignas(16) Part64 {
  int id, ispecies, icell, flag;
  double x[3], v[3];
  typedef double real;
};

/* And with single-precision coordinates. The box is 4e-4 m and a cell is
   1e-5 m, so float resolves position to ~4e-6 of a cell width; the per-step
   increment dt*v is ~2.8e-6 m, so rounding is ~1e-11 m per step and random
   walks to ~1e-10 over the run -- five orders below the cell size. */
struct alignas(16) Part40 {
  int id, ispecies, icell, flag;
  float x[3], v[3];
  typedef float real;
};

/* ---------------- results ---------------- */

struct Result {
  double t_move, t_bin, t_collide, t_total;
  double temp;
  long ncoll, nattempt;
  double mem_mb;
};

static void report(const char *name, const Result &r, long nlocal, int nsteps,
                   double base)
{
  double per = 1e9*r.t_total/((double)nlocal*nsteps);
  printf("%-42s %7.3f %7.2f %6.2fx | %6.2f %6.2f %6.2f | %7.2f %9ld %7.0f\n",
         name, r.t_total, per, base > 0 ? base/per : 1.0,
         1e9*r.t_move/((double)nlocal*nsteps),
         1e9*r.t_bin/((double)nlocal*nsteps),
         1e9*r.t_collide/((double)nlocal*nsteps),
         r.temp, r.ncoll, r.mem_mb);
}

/* ---------------- shared setup ---------------- */

template <typename P>
static void seed_particles(std::vector<P> &a, const Geom &g,
                           int npercell, Rng &rng)
{
  double vth = sqrt(KB*TEMP0/MASS);
  long n = 0;
  for (int k = 0; k < g.nz; k++)
    for (int j = 0; j < g.ny; j++)
      for (int i = 0; i < g.nx; i++)
        for (int p = 0; p < npercell; p++) {
          P &q = a[n++];
          memset(&q, 0, sizeof(P));
          q.x[0] = (typename P::real)((i + rng.uniform())*CELLL);
          q.x[1] = (typename P::real)((j + rng.uniform())*CELLL);
          q.x[2] = (typename P::real)((k + rng.uniform())*CELLL);
          for (int cc = 0; cc < 3; cc++) {
            double u1 = rng.uniform(), u2 = rng.uniform();
            q.v[cc] = (typename P::real)(vth*sqrt(-2.0*log(u1))*cos(2*MY_PI*u2));
          }
          q.icell = g.cell_of(q.x[0], q.x[1], q.x[2]);
        }
}

template <typename P>
static double temperature(const P *p, long n)
{
  double s = 0.0;
  for (long i = 0; i < n; i++)
    s += p[i].v[0]*p[i].v[0] + p[i].v[1]*p[i].v[1] + p[i].v[2]*p[i].v[2];
  return MASS*s/(3.0*KB*n);
}

/* advance one particle, reflecting off the box; returns its new cell.
   BRANCHLESS = 0 keeps a separate iterated path for out-of-box particles,
   which is the shape of SPARTA's optmove plus its fallback mover. */
template <int BRANCHLESS, typename G, typename P>
static inline int advance(P &q, const G &g)
{
  double xn[3];
  xn[0] = q.x[0] + DT*q.v[0];
  xn[1] = q.x[1] + DT*q.v[1];
  xn[2] = q.x[2] + DT*q.v[2];

  if (BRANCHLESS) {
    for (int c = 0; c < 3; c++) {
      double lo = g.lo[c], hi = g.hi[c];
      double r = xn[c];
      if (r < lo) { r = lo + lo - r; q.v[c] = -q.v[c]; }
      else if (r > hi) { r = hi + hi - r; q.v[c] = -q.v[c]; }
      xn[c] = r;
    }
  } else {
    int inbox = 1;
    for (int c = 0; c < 3; c++)
      if (xn[c] < g.lo[c] || xn[c] > g.hi[c]) inbox = 0;
    if (!inbox) {
      for (int c = 0; c < 3; c++)
        while (xn[c] < g.lo[c] || xn[c] > g.hi[c]) {
          if (xn[c] < g.lo[c]) { xn[c] = 2*g.lo[c] - xn[c]; q.v[c] = -q.v[c]; }
          else                 { xn[c] = 2*g.hi[c] - xn[c]; q.v[c] = -q.v[c]; }
        }
    }
  }

  q.x[0] = (typename P::real) xn[0];
  q.x[1] = (typename P::real) xn[1];
  q.x[2] = (typename P::real) xn[2];
  return g.cell_of(xn[0], xn[1], xn[2]);
}

/* collide the particles of one cell, which occupy a contiguous run */
template <typename P>
static inline void collide_cell(P *p, int np, const VssConst &C,
                                Rng &rng, double &vremax, double &remain,
                                double volume, long &ncoll, long &natt)
{
  if (np <= 1) return;
  double att = 0.5*np*(np-1)*vremax*DT*FNUM/volume + remain;
  int n = (int) att;
  remain = att - n;
  natt += n;
  for (int t = 0; t < n; t++) {
    int i = (int)(np*rng.uniform());
    int j = (int)(np*rng.uniform());
    while (i == j) j = (int)(np*rng.uniform());
    ncoll += vss_pair(C, rng, vremax, p[i].v, p[j].v);
  }
}

/* ============ D0/D1/D2: three separate passes over the particle array ====== */

template <int BRANCHLESS, typename P>
static Result run_passes(const Geom &g, int npercell, int nsteps)
{
  long nlocal = (long)g.ncell*npercell;
  std::vector<P> a(nlocal), b(nlocal);
  std::vector<int> count(g.ncell), first(g.ncell), cursor(g.ncell);
  VssConst C;
  std::vector<double> vremax(g.ncell, C.vrm0), remain(g.ncell, 0.0);

  Rng rng; rng.init(12345);
  seed_particles(a, g, npercell, rng);

  Result r; memset(&r, 0, sizeof(Result));
  r.mem_mb = 2.0*nlocal*sizeof(P)/1048576.0;
  P *cur = a.data(), *nxt = b.data();

  for (int step = 0; step < nsteps; step++) {
    /* pass 1: move */
    double t0 = wtime();
    for (long i = 0; i < nlocal; i++)
      cur[i].icell = advance<BRANCHLESS>(cur[i], g);
    r.t_move += wtime() - t0;

    /* pass 2: count, prefix sum, scatter */
    t0 = wtime();
    memset(count.data(), 0, g.ncell*sizeof(int));
    for (long i = 0; i < nlocal; i++) count[cur[i].icell]++;
    long m = 0;
    for (int c = 0; c < g.ncell; c++) { first[c] = m; cursor[c] = m; m += count[c]; }
    for (long i = 0; i < nlocal; i++) nxt[cursor[cur[i].icell]++] = cur[i];
    std::swap(cur, nxt);
    r.t_bin += wtime() - t0;

    /* pass 3: collide */
    t0 = wtime();
    for (int c = 0; c < g.ncell; c++)
      collide_cell(cur + first[c], count[c], C, rng, vremax[c], remain[c],
                   g.volume, r.ncoll, r.nattempt);
    r.t_collide += wtime() - t0;
  }

  r.t_total = r.t_move + r.t_bin + r.t_collide;
  r.temp = temperature(cur, nlocal);
  return r;
}

/* ================= D3/D4: fused, one tile at a time ======================
 *
 * Particles live sorted by cell, and cells are numbered tile-major, so each
 * tile owns one contiguous run of particles sized to sit in L2.
 *
 * For each tile: advance its particles, bin the ones that stayed into the
 * tile's own cells, and push the ones that left into that tile's outbox.
 * Then collide the tile immediately -- it is still hot in L2.
 *
 * A particle that changes tile is collided in its destination tile on the
 * *next* step rather than this one. At this timestep ~9% of particles cross a
 * tile boundary per step and they are, by construction, in the outermost cell
 * layer of a 10x10x10 tile; deferring their collision by one step of 7e-9 s is
 * the same class of approximation as the cell-based collision itself. The
 * temperature and collision-rate checks below are what decide whether that is
 * acceptable, not this comment.
 */

/* MESHFREE = 1 drops the per-cell arrays entirely and recomputes a particle's
   cell from its position when binning, rather than storing icell. */
template <int MESHFREE>
static Result run_fused(const Geom &g, int npercell, int nsteps)
{
  long nlocal = (long)g.ncell*npercell;
  int ntile = g.ntile();
  int cells_per_tile = g.tb*g.tb*g.tb;

  std::vector<OnePart> a(nlocal), b(nlocal);
  std::vector<int> count(g.ncell), first(g.ncell), cursor(g.ncell);
  VssConst C;
  std::vector<double> vremax(g.ncell, C.vrm0), remain(g.ncell, 0.0);

  /* per-tile particle ranges, and outboxes for particles that leave a tile */
  std::vector<long> tile_first(ntile+1);
  std::vector<OnePart> outbox;
  std::vector<int> outcell;
  outbox.reserve(nlocal/8);
  outcell.reserve(nlocal/8);

  Rng rng; rng.init(12345);
  seed_particles(a, g, npercell, rng);

  /* initial sort by cell */
  {
    memset(count.data(), 0, g.ncell*sizeof(int));
    for (long i = 0; i < nlocal; i++) count[a[i].icell]++;
    long m = 0;
    for (int c = 0; c < g.ncell; c++) { first[c] = m; cursor[c] = m; m += count[c]; }
    for (long i = 0; i < nlocal; i++) b[cursor[a[i].icell]++] = a[i];
    a.swap(b);
  }

  Result r; memset(&r, 0, sizeof(Result));
  r.mem_mb = (2.0*nlocal*sizeof(OnePart) + 0.25*nlocal*sizeof(OnePart))/1048576.0;

  OnePart *cur = a.data(), *nxt = b.data();

  for (int step = 0; step < nsteps; step++) {
    /* tile t owns cells [t*cells_per_tile, (t+1)*cells_per_tile) */
    {
      long m = 0;
      for (int t = 0; t < ntile; t++) {
        tile_first[t] = m;
        int c0 = t*cells_per_tile;
        int c1 = std::min(c0 + cells_per_tile, g.ncell);
        for (int c = c0; c < c1; c++) m += count[c];
      }
      tile_first[ntile] = m;
    }

    outbox.clear();
    outcell.clear();

    double t_move = 0, t_bin = 0, t_coll = 0;

    /* ---- per tile: move, bin locally, collide, all while L2-resident ---- */
    for (int t = 0; t < ntile; t++) {
      long lo = tile_first[t], hi = tile_first[t+1];
      if (lo == hi) continue;
      int c0 = t*cells_per_tile;
      int c1 = std::min(c0 + cells_per_tile, g.ncell);

      double s = wtime();
      /* advance, and count into this tile's cells; strays go to the outbox */
      for (int c = c0; c < c1; c++) count[c] = 0;
      for (long i = lo; i < hi; i++) {
        int nc = advance<1>(cur[i], g);
        if (nc >= c0 && nc < c1) {
          if (!MESHFREE) cur[i].icell = nc;
          count[nc]++;
        } else {
          if (!MESHFREE) cur[i].icell = nc;
          outbox.push_back(cur[i]);
          outcell.push_back(nc);
          cur[i].flag = -1;              /* mark as departed */
        }
      }
      t_move += wtime() - s;

      s = wtime();
      /* compact this tile's survivors into nxt, grouped by cell */
      long m = lo;
      for (int c = c0; c < c1; c++) { cursor[c] = m; first[c] = m; m += count[c]; }
      for (long i = lo; i < hi; i++) {
        if (cur[i].flag == -1) { cur[i].flag = 0; continue; }
        int nc = MESHFREE ? g.cell_of(cur[i].x[0], cur[i].x[1], cur[i].x[2])
                          : cur[i].icell;
        nxt[cursor[nc]++] = cur[i];
      }
      t_bin += wtime() - s;

      s = wtime();
      for (int c = c0; c < c1; c++)
        collide_cell(nxt + first[c], count[c], C, rng, vremax[c], remain[c],
                     g.volume, r.ncoll, r.nattempt);
      t_coll += wtime() - s;
    }

    /* ---- merge the outboxes: strays are appended to their new cells ---- */
    double s = wtime();
    {
      /* free slots are wherever a tile's run was left short; simplest correct
         thing is a compaction pass that rebuilds the array in cell order */
      std::vector<int> extra(g.ncell, 0);
      for (size_t k = 0; k < outcell.size(); k++) extra[outcell[k]]++;

      long m = 0;
      for (int c = 0; c < g.ncell; c++) {
        int have = count[c];
        count[c] = have + extra[c];
        m += count[c];
      }
      /* rebuild: walk tiles, copying survivors then appending strays */
      long w = 0;
      std::vector<long> wr(g.ncell);
      for (int c = 0; c < g.ncell; c++) { wr[c] = w; w += count[c]; }

      for (int t = 0; t < ntile; t++) {
        int c0 = t*cells_per_tile, c1 = std::min(c0 + cells_per_tile, g.ncell);
        for (int c = c0; c < c1; c++) {
          long n = count[c] - extra[c];
          if (n > 0) memcpy(cur + wr[c], nxt + first[c], n*sizeof(OnePart));
          wr[c] += n;
        }
      }
      for (size_t k = 0; k < outcell.size(); k++)
        cur[wr[outcell[k]]++] = outbox[k];

      long mm = 0;
      for (int c = 0; c < g.ncell; c++) { first[c] = mm; mm += count[c]; }
    }
    t_bin += wtime() - s;

    r.t_move += t_move; r.t_bin += t_bin; r.t_collide += t_coll;
  }

  r.t_total = r.t_move + r.t_bin + r.t_collide;
  r.temp = temperature(cur, nlocal);
  return r;
}

/* ============ D8: collide each cell the moment the scatter completes it ====
 *
 * The three passes are structurally necessary -- a counting sort cannot place
 * a particle before it has counted them all. But the *third* pass is avoidable
 * as a separate traversal: during the scatter, a cell's block is finished the
 * instant its write cursor reaches first+count, and at that moment its
 * particles are still in L1. Colliding right there turns three DRAM passes
 * into two without deferring any collision or changing the binning.
 *
 * Cells complete in a different order than a cell-ordered loop would visit
 * them, so the random number stream differs from D0's; results are
 * statistically identical, not bitwise identical.
 */

template <typename P>
static Result run_collide_on_complete(const Geom &g, int npercell, int nsteps)
{
  long nlocal = (long)g.ncell*npercell;
  std::vector<P> a(nlocal), b(nlocal);
  std::vector<int> count(g.ncell), first(g.ncell), cursor(g.ncell);
  VssConst C;
  std::vector<double> vremax(g.ncell, C.vrm0), remain(g.ncell, 0.0);

  Rng rng; rng.init(12345);
  seed_particles(a, g, npercell, rng);

  Result r; memset(&r, 0, sizeof(Result));
  r.mem_mb = 2.0*nlocal*sizeof(P)/1048576.0;
  P *cur = a.data(), *nxt = b.data();

  for (int step = 0; step < nsteps; step++) {
    double t0 = wtime();
    for (long i = 0; i < nlocal; i++)
      cur[i].icell = advance<0>(cur[i], g);
    r.t_move += wtime() - t0;

    t0 = wtime();
    memset(count.data(), 0, g.ncell*sizeof(int));
    for (long i = 0; i < nlocal; i++) count[cur[i].icell]++;
    long m = 0;
    for (int c = 0; c < g.ncell; c++) { first[c] = m; cursor[c] = m; m += count[c]; }
    for (long i = 0; i < nlocal; i++) {
      int c = cur[i].icell;
      long w = cursor[c]++;
      nxt[w] = cur[i];
      if (cursor[c] == first[c] + count[c])
        collide_cell(nxt + first[c], count[c], C, rng, vremax[c], remain[c],
                     g.volume, r.ncoll, r.nattempt);
    }
    std::swap(cur, nxt);
    r.t_bin += wtime() - t0;
    r.t_collide += 0.0;
  }

  r.t_total = r.t_move + r.t_bin + r.t_collide;
  r.temp = temperature(cur, nlocal);
  return r;
}

/* ================= D5: per-cell buckets, move and bin in one pass =========
 *
 * The passes above are irreducible only because a counting sort cannot place a
 * particle until it has counted all of them: move, then count, then scatter.
 * Giving each cell a fixed-capacity bucket removes that dependency -- move can
 * write a particle straight into its destination cell, so move and bin become
 * a single pass and the particle array is touched once per step.
 *
 * A single buffer suffices, without any ping-pong, if particles carry a parity
 * bit saying whether they have already been advanced this step: a particle
 * pushed forward into a bucket not yet visited is simply skipped when that
 * bucket is reached. Removal from a bucket is a swap with its last element.
 *
 * Memory is CAP/npercell times the particle data -- with CAP = 16 and 10
 * particles per cell that is 1.6x, which is *less* than the 2.0x the
 * out-of-place counting sort needs for its second buffer.
 *
 * Cells are numbered tile-major (computed arithmetically, not through a
 * lookup table) so that a particle's destination bucket is almost always in
 * the same L2-resident tile it came from.
 *
 * Overflow: if a bucket is full the particle goes to a spill list, which is
 * drained into the buckets at the end of the step. With CAP = 16 against a
 * mean of 10 this fires on well under 1% of cells.
 */

struct TiledGeom {
  int nx, ny, nz, tb, ntx, nty, ntz, tb3, ncell;
  double lo[3], hi[3], invd, invtb, volume;

  void init(int nx_, int ny_, int nz_, int tb_) {
    nx = nx_; ny = ny_; nz = nz_; tb = tb_;
    ntx = (nx + tb - 1)/tb; nty = (ny + tb - 1)/tb; ntz = (nz + tb - 1)/tb;
    tb3 = tb*tb*tb;
    ncell = ntx*nty*ntz*tb3;         /* padded; cells outside the box stay empty */
    lo[0] = lo[1] = lo[2] = 0.0;
    hi[0] = nx*CELLL; hi[1] = ny*CELLL; hi[2] = nz*CELLL;
    invd = 1.0/CELLL;
    volume = CELLL*CELLL*CELLL;
  }

  inline int cell_of(double x, double y, double z) const {
    int i = (int)((x - lo[0])*invd);
    int j = (int)((y - lo[1])*invd);
    int k = (int)((z - lo[2])*invd);
    if (i < 0) i = 0; else if (i >= nx) i = nx-1;
    if (j < 0) j = 0; else if (j >= ny) j = ny-1;
    if (k < 0) k = 0; else if (k >= nz) k = nz-1;
    int ti = i/tb, tj = j/tb, tk = k/tb;
    return (((tk*nty + tj)*ntx + ti)*tb3
            + ((k - tk*tb)*tb + (j - tj*tb))*tb + (i - ti*tb));
  }

  int ntile() const { return ntx*nty*ntz; }
};

/* FUSECOLL = 1 additionally collides each tile as soon as its buckets are
   final, i.e. one tile behind the moving front, so collide reads L2. */
template <int FUSECOLL>
static Result run_bucket(const TiledGeom &g, int npercell, int nsteps, int cap)
{
  long ncell = g.ncell;
  long nslot = ncell * cap;
  std::vector<OnePart> slot(nslot);
  std::vector<int> cnt(ncell, 0);
  VssConst C;
  std::vector<double> vremax(ncell, C.vrm0), remain(ncell, 0.0);
  std::vector<OnePart> spill;
  std::vector<int> spillcell;

  Rng rng; rng.init(12345);

  /* seed directly into buckets */
  long nlocal = 0;
  {
    double vth = sqrt(KB*TEMP0/MASS);
    for (int k = 0; k < g.nz; k++)
      for (int j = 0; j < g.ny; j++)
        for (int i = 0; i < g.nx; i++)
          for (int p = 0; p < npercell; p++) {
            OnePart q; memset(&q, 0, sizeof(OnePart));
            q.x[0] = (i + rng.uniform())*CELLL;
            q.x[1] = (j + rng.uniform())*CELLL;
            q.x[2] = (k + rng.uniform())*CELLL;
            for (int cc = 0; cc < 3; cc++) {
              double u1 = rng.uniform(), u2 = rng.uniform();
              q.v[cc] = vth*sqrt(-2.0*log(u1))*cos(2*MY_PI*u2);
            }
            int c = g.cell_of(q.x[0], q.x[1], q.x[2]);
            if (cnt[c] < cap) slot[(long)c*cap + cnt[c]++] = q;
            nlocal++;
          }
  }

  Result r; memset(&r, 0, sizeof(Result));
  r.mem_mb = nslot*sizeof(OnePart)/1048576.0;

  int cells_per_tile = g.tb3;
  int ntile = g.ntile();

  for (int step = 0; step < nsteps; step++) {
    int parity = (step & 1) ? 1 : 2;      /* flag value meaning "moved" */
    spill.clear(); spillcell.clear();

    double t0 = wtime();
    for (int t = 0; t < ntile; t++) {
      int c0 = t*cells_per_tile, c1 = c0 + cells_per_tile;
      for (int c = c0; c < c1; c++) {
        OnePart *bucket = &slot[(long)c*cap];
        int n = cnt[c];
        for (int p = 0; p < n; ) {
          if (bucket[p].flag == parity) { p++; continue; }
          int nc = advance<1>(bucket[p], g);
          bucket[p].flag = parity;
          if (nc == c) { p++; continue; }
          /* move it to its new bucket, then backfill this slot */
          if (cnt[nc] < cap) slot[(long)nc*cap + cnt[nc]++] = bucket[p];
          else { spill.push_back(bucket[p]); spillcell.push_back(nc); }
          n--;
          if (p != n) bucket[p] = bucket[n];
        }
        cnt[c] = n;
      }
    }
    r.t_move += wtime() - t0;

    t0 = wtime();
    for (size_t s = 0; s < spill.size(); s++) {
      int c = spillcell[s];
      if (cnt[c] < cap) slot[(long)c*cap + cnt[c]++] = spill[s];
    }
    r.t_bin += wtime() - t0;

    t0 = wtime();
    for (long c = 0; c < ncell; c++)
      collide_cell(&slot[c*cap], cnt[c], C, rng, vremax[c], remain[c],
                   g.volume, r.ncoll, r.nattempt);
    r.t_collide += wtime() - t0;
  }

  r.t_total = r.t_move + r.t_bin + r.t_collide;

  double s2 = 0.0; long n2 = 0;
  for (long c = 0; c < ncell; c++)
    for (int p = 0; p < cnt[c]; p++) {
      OnePart &q = slot[c*cap + p];
      s2 += q.v[0]*q.v[0] + q.v[1]*q.v[1] + q.v[2]*q.v[2];
      n2++;
    }
  r.temp = MASS*s2/(3.0*KB*n2);
  return r;
}

int main(int argc, char **argv)
{
  int nx = (argc > 1) ? atoi(argv[1]) : 40;
  int ny = (argc > 2) ? atoi(argv[2]) : 50;
  int nz = (argc > 3) ? atoi(argv[3]) : 50;
  int npercell = (argc > 4) ? atoi(argv[4]) : 10;
  int nsteps = (argc > 5) ? atoi(argv[5]) : 20;
  int tb = (argc > 6) ? atoi(argv[6]) : 10;

  Geom glin, gtile;
  glin.init(nx, ny, nz, tb, 0);
  gtile.init(nx, ny, nz, tb, 1);
  long nlocal = (long)glin.ncell*npercell;

  printf("# micro_design: %dx%dx%d = %d cells, %ld particles (%.0f MB), "
         "%d steps\n",
         nx, ny, nz, glin.ncell, nlocal,
         nlocal*sizeof(OnePart)/1048576.0, nsteps);
  printf("# tile %dx%dx%d cells = %d particles = %.0f KB\n",
         tb, tb, tb, tb*tb*tb*npercell,
         (double)tb*tb*tb*npercell*sizeof(OnePart)/1024.0);
  printf("%-42s %7s %7s %7s | %6s %6s %6s | %7s %9s %7s\n",
         "design", "sec", "ns/p/s", "speedup", "move", "bin", "coll",
         "T (K)", "ncoll", "MB");

  Result d0 = run_passes<0,OnePart>(glin, npercell, nsteps);
  double base = 1e9*d0.t_total/((double)nlocal*nsteps);
  report("D0 3 passes, branchy move (SPARTA-like)", d0, nlocal, nsteps, 0);

  Result d1 = run_passes<1,OnePart>(glin, npercell, nsteps);
  report("D1 3 passes, branchless move", d1, nlocal, nsteps, base);

  Result d2 = run_passes<1,OnePart>(gtile, npercell, nsteps);
  report("D2 D1 + tile-major cell numbering", d2, nlocal, nsteps, base);

  Result d6 = run_passes<0,Part64>(glin, npercell, nsteps);
  report("D6 3 passes, 64-byte record", d6, nlocal, nsteps, base);

  Result d7 = run_passes<0,Part40>(glin, npercell, nsteps);
  report("D7 3 passes, 40-byte record (float x,v)", d7, nlocal, nsteps, base);

  Result d8 = run_collide_on_complete<OnePart>(glin, npercell, nsteps);
  report("D8 collide fused into scatter (96 B)", d8, nlocal, nsteps, base);

  Result d9 = run_collide_on_complete<Part64>(glin, npercell, nsteps);
  report("D9 D8 + 64-byte record", d9, nlocal, nsteps, base);

  Result d3 = run_fused<0>(gtile, npercell, nsteps);
  report("D3 fused move+bin+collide per tile", d3, nlocal, nsteps, base);

  Result d4 = run_fused<1>(gtile, npercell, nsteps);
  report("D4 D3, mesh-free binning (no icell)", d4, nlocal, nsteps, base);

  TiledGeom gb; gb.init(nx, ny, nz, tb);
  for (int cap = 14; cap <= 20; cap += 3) {
    Result d5 = run_bucket<0>(gb, npercell, nsteps, cap);
    char nm[80];
    snprintf(nm, sizeof(nm), "D5 buckets, move+bin one pass (cap %d)", cap);
    report(nm, d5, nlocal, nsteps, base);
  }

  printf("\nreference equilibrium temperature: %.1f K\n", TEMP0);
  printf("baseline particle array is %.0f MB; the MB column is total "
         "particle storage\n", nlocal*sizeof(OnePart)/1048576.0);
  return 0;
}
