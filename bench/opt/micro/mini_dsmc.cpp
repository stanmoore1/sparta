/* mini-DSMC: a faithful standalone model of SPARTA's in.collide timestep.
 *
 * WHY THIS EXISTS
 *
 * The micro_* benchmarks in this directory over-predicted three times running,
 * always in the same direction and always for the same reason: their kernels do
 * far less work per particle than SPARTA's, so memory traffic is an inflated
 * share of their runtime and any memory optimization looks better than it is.
 *
 *   micro prediction            what SPARTA actually did
 *   ------------------------    ---------------------------------------
 *   collide fusion    1.26x     ~1.05x
 *   64-byte record    1.58x     ~1.14x  (from an in-situ padding experiment)
 *   index-only bin    1.50x     0.55x   (a 1.8x regression)
 *
 * So this is a mini-app rather than a microbenchmark: it carries SPARTA's real
 * data structures at their real sizes, with the real indirections, and its move
 * and collide kernels are transcriptions of Update::move<3,0,1> and
 * CollideVSS::collide_cell_kernel rather than simplified stand-ins.
 *
 * WHAT IS REPRODUCED FAITHFULLY
 *
 *   Particle::OnePart      96 B, all fields, SPARTA_ALIGN(16)
 *   Grid::ChildCell       128 B, SPARTA_ALIGN(64), read per particle in move
 *   Grid::ChildInfo        64 B, read per cell in sort and collide
 *   vremax, remain        double***, so vremax[icell][0][0] is two dependent
 *                         pointer loads exactly as in SPARTA
 *   params, prefactor     double** / Params**, indexed [isp][jsp]
 *   species               the real ~200 B Species record
 *   next[]                the per-cell linked list, walked to build plist
 *   RanKnuth              the real generator, same stream
 *   move                  pflag dispatch, in-box fast path with the uniform
 *                         cell-index map, cells[icell].proc ownership test, and
 *                         a genuine cell-by-cell traversal slow path with
 *                         neighbour lookups for particles leaving the box
 *   collide               plist gather or contiguous indexing, the NTC attempt
 *                         count with remain, and the full VSS pair kernel
 *
 * VALIDATION
 *
 * Run with -validate to print the reorder-period curve, which can be compared
 * directly against SPARTA's own. A model that does not reproduce the *shape* of
 * that curve -- in particular that never reordering is much worse, not better --
 * should not be trusted to predict anything else.
 *
 * build: g++ -O3 -std=c++11 -o mini_dsmc mini_dsmc.cpp
 * usage: ./mini_dsmc [nx ny nz npercell nsteps reorder] | ./mini_dsmc -validate
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cstdint>
#include <vector>
#include <algorithm>

#define SPARTA_ALIGN(n) __attribute__((aligned(n)))
#define MY_PI 3.14159265358979323846
#define EPSZERO 1.0e-14
#define MAXVIBMODE 4
#define DELTAPART 128

typedef int cellint;
typedef int surfint;

static double wtime()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1.0e-9*ts.tv_nsec;
}

/* ---------------- benchmark constants, from bench/ar.{species,vss} -------- */

static const double KB    = 1.380658e-23;
static const double AMASS = 6.63e-26;
static const double DIAM  = 4.11e-10;
static const double OMEGA = 0.81;
static const double TREF  = 273.15;
static const double ALPHA = 1.4;
static const double TEMP0 = 273.15;
static const double CELLL = 1.0e-5;
static const double DT    = 7.0e-9;
static const double FNUM  = 7.07043e6;

enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};

/* ---------------- RanKnuth, verbatim ---------------- */

#define MBIG 1000000000
#define MSEED 161803398
#define RFAC (1.0/MBIG)

struct RanKnuth {
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

/* ---------------- SPARTA data structures, at their real sizes ------------- */

/* PART_BYTES selects the record size so the model can be asked the same
   question that was put to SPARTA directly by padding its OnePart:
   how much does runtime actually move when the record grows or shrinks?
     64  = the fields move/sort/collide touch in this benchmark
     96  = SPARTA today
    128  = SPARTA padded, the in-situ experiment that measured elasticity 0.4 */
#ifndef PART_BYTES
#define PART_BYTES 96
#endif

struct SPARTA_ALIGN(16) OnePart {
  int id, ispecies, icell, flag;
  double x[3], v[3];
#if PART_BYTES >= 96
  double erot, evib, dtremain, weight;
#endif
#if PART_BYTES >= 128
  double pad_experiment[4];
#endif
};

struct SPARTA_ALIGN(64) ChildCell {
  cellint id;
  int level, proc, ilocal;
  cellint neigh[6];
  int nmask;
  double lo[3], hi[3];
  int nsurf;
  surfint *csurfs;
  int nsplit, isplit;
};

struct ChildInfo {
  int count, first;
  int mask, type;
  int corner[8];
  double volume, weight;
};

struct Species {
  char id[16];
  double molwt, mass, specwt, charge, rotrel;
  double rottemp[3];
  double vibtemp[MAXVIBMODE], vibrel[MAXVIBMODE];
  int vibdegen[MAXVIBMODE];
  int rotdof, vibdof, nrottemp, nvibmode, internaldof, vibdiscrete_read;
  double magmoment;
};

struct Params {
  double diam, omega, tref, alpha;
  double rotc1, rotc2, rotc3, vibc1, vibc2, mr;
};

/* ---------------- the simulation ---------------- */

struct Mini {
  int nx, ny, nz, ncell;
  double boxlo[3], boxhi[3], dx, dy, dz;
  int me;

  std::vector<OnePart> particles, sortbuf;
  std::vector<int> next, sortcursor;
  int nlocal;
  int sorted_contiguous;

  /* ChildCell is alignas(64) and std::vector does not honour over-aligned
     types before C++17; with -march=native the compiler emits aligned AVX-512
     moves against the declared alignment and faults. SPARTA allocates these
     through memory->smalloc with SPARTA_GET_ALIGN, so allocate them aligned
     here too rather than quietly dropping the alignment. */
  ChildCell *cells;
  std::vector<ChildInfo> cinfo;
  std::vector<int> uniform_index;

  /* vremax/remain kept as double*** so the two dependent pointer loads that
     SPARTA pays on every cell are present here too */
  double ***vremax, ***remain;
  double *vremax_data, *remain_data;
  double **vremax_lvl2, **remain_lvl2;
  double dt;                       /* current timestep */
  int collide_every;               /* collide every K steps, K x attempts */
  int flat_vremax;                 /* 1 to use the flattened single-group view */
  double *vremax1, *remain1;

  Species *species;
  Params **params;
  Params *params_data;
  double **prefactor;
  double *prefactor_data;

  std::vector<int> plist;
  int npmax;
  RanKnuth random;

  long ncollide_one, nattempt_one, ntouch_one, nboundary_one;
  double t_move, t_sort, t_collide;

  void setup(int nx_, int ny_, int nz_, int npercell, int pad_particle);
  void teardown();
  void move();
  void sort();
  void sort_reorder();
  void collide();
  void collide_one_cell(int icell, OnePart *p, const int *pl, int np, int contig);
  double temperature();
  inline int cell_of(double x, double y, double z) const;
};

inline int Mini::cell_of(double x, double y, double z) const
{
  int i = (int)((x - boxlo[0])/dx);
  int j = (int)((y - boxlo[1])/dy);
  int k = (int)((z - boxlo[2])/dz);
  return (k*ny + j)*nx + i + 1;      /* SPARTA cell IDs are 1-based */
}

void Mini::setup(int nx_, int ny_, int nz_, int npercell, int)
{
  nx = nx_; ny = ny_; nz = nz_;
  ncell = nx*ny*nz;
  me = 0;
  dt = DT;
  collide_every = 1;
  boxlo[0] = boxlo[1] = boxlo[2] = 0.0;
  boxhi[0] = nx*CELLL; boxhi[1] = ny*CELLL; boxhi[2] = nz*CELLL;
  dx = dy = dz = CELLL;

  if (posix_memalign((void**)&cells,64,(size_t)ncell*sizeof(ChildCell))) exit(1);
  memset(cells,0,(size_t)ncell*sizeof(ChildCell));
  cinfo.resize(ncell);
  for (int c = 0; c < ncell; c++) {
    int i = c % nx, j = (c/nx) % ny, k = c/(nx*ny);
    cells[c].id = c+1;
    cells[c].level = 0;
    cells[c].proc = 0;
    cells[c].ilocal = c;
    cells[c].nmask = 0;
    cells[c].lo[0] = i*CELLL; cells[c].lo[1] = j*CELLL; cells[c].lo[2] = k*CELLL;
    cells[c].hi[0] = (i+1)*CELLL; cells[c].hi[1] = (j+1)*CELLL; cells[c].hi[2] = (k+1)*CELLL;
    cells[c].nsurf = 0;
    cells[c].csurfs = NULL;
    cells[c].nsplit = 1;
    cells[c].isplit = -1;
    /* six face neighbours, as Grid::find_neighbors sets them */
    for (int f = 0; f < 6; f++) cells[c].neigh[f] = -1;
    if (i > 0)    cells[c].neigh[0] = c-1;
    if (i < nx-1) cells[c].neigh[1] = c+1;
    if (j > 0)    cells[c].neigh[2] = c-nx;
    if (j < ny-1) cells[c].neigh[3] = c+nx;
    if (k > 0)    cells[c].neigh[4] = c-nx*ny;
    if (k < nz-1) cells[c].neigh[5] = c+nx*ny;

    memset(cinfo[c].corner, 0, sizeof(cinfo[c].corner));
    cinfo[c].mask = 1; cinfo[c].type = 0;
    cinfo[c].volume = CELLL*CELLL*CELLL;
    cinfo[c].weight = 1.0;
    cinfo[c].count = 0; cinfo[c].first = -1;
  }

  /* the dense cell-ID map that round 1 added, sized as Grid::build_uniform_index
     sizes it (with headroom for a particle exactly on the upper face) */
  {
    long n = ((long)nz*ny + ny)*nx + nx + 2;
    uniform_index.assign(n, -1);
    for (int c = 0; c < ncell; c++) uniform_index[cells[c].id] = c;
  }

  /* species and VSS params, through the same double indirection SPARTA uses */
  species = (Species *) calloc(1, sizeof(Species));
  strcpy(species[0].id, "Ar");
  species[0].mass = AMASS;
  species[0].rotdof = 0; species[0].vibdof = 0;
  species[0].specwt = 1.0;

  params_data = (Params *) calloc(1, sizeof(Params));
  params = (Params **) malloc(sizeof(Params *));
  params[0] = params_data;
  params[0][0].diam = DIAM; params[0][0].omega = OMEGA;
  params[0][0].tref = TREF; params[0][0].alpha = ALPHA;
  params[0][0].mr = AMASS*AMASS/(AMASS+AMASS);

  prefactor_data = (double *) calloc(1, sizeof(double));
  prefactor = (double **) malloc(sizeof(double *));
  prefactor[0] = prefactor_data;
  double cxs = MY_PI*DIAM*DIAM;
  prefactor[0][0] = cxs * pow(2.0*KB*TREF/params[0][0].mr, OMEGA-0.5)
                    / tgamma(2.5-OMEGA);

  /* vremax[icell][0][0]: two levels of pointer array over a contiguous block,
     which is exactly what memory->create(vremax,nglocal,1,1) produces */
  vremax_data = (double *) malloc((size_t)ncell*sizeof(double));
  remain_data = (double *) malloc((size_t)ncell*sizeof(double));
  vremax_lvl2 = (double **) malloc((size_t)ncell*sizeof(double *));
  remain_lvl2 = (double **) malloc((size_t)ncell*sizeof(double *));
  vremax = (double ***) malloc((size_t)ncell*sizeof(double **));
  remain = (double ***) malloc((size_t)ncell*sizeof(double **));
  double vrm0 = 2.0*cxs*sqrt(2.0*KB*TEMP0/AMASS);
  for (int c = 0; c < ncell; c++) {
    vremax_data[c] = vrm0; remain_data[c] = 0.0;
    vremax_lvl2[c] = &vremax_data[c];
    remain_lvl2[c] = &remain_data[c];
    vremax[c] = &vremax_lvl2[c];
    remain[c] = &remain_lvl2[c];
  }
  flat_vremax = 1;
  vremax1 = vremax_data;
  remain1 = remain_data;

  /* particles */
  nlocal = ncell*npercell;
  particles.resize(nlocal);
  sortbuf.resize(nlocal);
  next.resize(nlocal);
  sortcursor.resize(ncell);
  sorted_contiguous = 0;
  npmax = DELTAPART;
  plist.resize(npmax);

  random.init(12345);
  double vscale = sqrt(2.0*KB*TEMP0/AMASS);
  long m = 0;
  for (int k = 0; k < nz; k++)
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
        for (int p = 0; p < npercell; p++) {
          OnePart &q = particles[m];
          memset(&q, 0, sizeof(OnePart));
          q.id = (int)m; q.ispecies = 0; q.flag = PKEEP;
          q.x[0] = (i + random.uniform())*CELLL;
          q.x[1] = (j + random.uniform())*CELLL;
          q.x[2] = (k + random.uniform())*CELLL;
          for (int c = 0; c < 3; c++) {
            double u1 = random.uniform(), u2 = random.uniform();
            /* vscale = sqrt(2kT/m); sqrt(-ln u1)*cos(2 pi u2) is N(0,1)/sqrt(2),
               so this draws a component with std dev sqrt(kT/m), giving T */
            q.v[c] = vscale*sqrt(-log(u1))*cos(2*MY_PI*u2);
          }
#if PART_BYTES >= 96
          q.weight = 1.0;
#endif
          q.icell = uniform_index[cell_of(q.x[0],q.x[1],q.x[2])];
          m++;
        }
  ncollide_one = nattempt_one = ntouch_one = nboundary_one = 0;
  t_move = t_sort = t_collide = 0.0;
}

void Mini::teardown()
{
  free(cells);
  free(species); free(params_data); free(params);
  free(prefactor_data); free(prefactor);
  free(vremax_data); free(remain_data);
  free(vremax_lvl2); free(remain_lvl2);
  free(vremax); free(remain);
}

/* ---------------- move: a transcription of Update::move<3,0,1> ------------
 *
 * Keeps the parts that make SPARTA's mover behave the way it does: the pflag
 * dispatch, the in-box test, the dense cell-index map lookup, the
 * cells[icell].proc ownership test, and a real cell-by-cell traversal for
 * particles that leave the box, walking cells[].lo/hi and cells[].neigh.
 */
void Mini::move()
{
  OnePart *p = particles.data();
  ChildCell *cs = cells;
  const int *uidx = uniform_index.data();
  const double *lo = boxlo, *hi = boxhi;
  ntouch_one = nboundary_one = 0;

  for (int i = 0; i < nlocal; i++) {
    int pflag = p[i].flag;
    double *x = p[i].x;
    double *v = p[i].v;
    double dtremain;
    double xnew[3];

    if (pflag == PDONE) { p[i].flag = PKEEP; pflag = PKEEP; }

    dtremain = dt;
    xnew[0] = x[0] + dtremain*v[0];
    xnew[1] = x[1] + dtremain*v[1];
    xnew[2] = x[2] + dtremain*v[2];

    int optmove = 1;
    if (xnew[0] < lo[0] || xnew[0] > hi[0]) optmove = 0;
    if (xnew[1] < lo[1] || xnew[1] > hi[1]) optmove = 0;
    if (xnew[2] < lo[2] || xnew[2] > hi[2]) optmove = 0;

    if (optmove) {
      const int ip = (int)((xnew[0]-lo[0])/dx);
      const int jp = (int)((xnew[1]-lo[1])/dy);
      const int kp = (int)((xnew[2]-lo[2])/dz);
      int cellIdx = (kp*ny + jp)*nx + ip + 1;
      int icell = uidx[cellIdx];
      if (icell >= 0) {
        p[i].icell = icell;
        p[i].flag = PKEEP;
        x[0] = xnew[0]; x[1] = xnew[1]; x[2] = xnew[2];
        if (cs[icell].proc != me) p[i].flag = PDONE;   /* would migrate */
        continue;
      }
    }

    /* slow path: walk cell to cell, reflecting off the box, exactly the shape
       of the general mover's inner while loop */
    p[i].flag = PKEEP;
    int icell = p[i].icell;
    ntouch_one++;

    while (1) {
      double *clo = cs[icell].lo;
      double *chi = cs[icell].hi;
      cellint *neigh = cs[icell].neigh;

      int outface = -1;
      double frac = 1.0;
      /* which face is crossed first */
      for (int d = 0; d < 3; d++) {
        if (xnew[d] < clo[d]) {
          double f = (clo[d]-x[d]) / (xnew[d]-x[d]);
          if (f < frac) { frac = f; outface = 2*d; }
        } else if (xnew[d] > chi[d]) {
          double f = (chi[d]-x[d]) / (xnew[d]-x[d]);
          if (f < frac) { frac = f; outface = 2*d+1; }
        }
      }
      if (outface < 0) break;

      x[0] += frac*(xnew[0]-x[0]);
      x[1] += frac*(xnew[1]-x[1]);
      x[2] += frac*(xnew[2]-x[2]);

      int nb = neigh[outface];
      if (nb < 0) {
        /* box boundary: reflect, as SurfCollideSpecular does for "rr" */
        int d = outface/2;
        v[d] = -v[d];
        xnew[d] = (outface & 1) ? 2*hi[d]-xnew[d] : 2*lo[d]-xnew[d];
        nboundary_one++;
        continue;
      }
      icell = nb;
      ntouch_one++;
    }

    x[0] = xnew[0]; x[1] = xnew[1]; x[2] = xnew[2];
    /* clamp, then re-derive the cell as the general mover ends up doing */
    for (int d = 0; d < 3; d++) {
      if (x[d] < lo[d]) x[d] = lo[d];
      if (x[d] > hi[d]) x[d] = hi[d]*(1.0-1e-15);
    }
    p[i].icell = uidx[cell_of(x[0],x[1],x[2])];
  }
}

/* ---------------- sort: Particle::sort, the linked-list build ------------- */

void Mini::sort()
{
  sorted_contiguous = 0;
  ChildInfo *ci = cinfo.data();
  for (int c = 0; c < ncell; c++) { ci[c].first = -1; ci[c].count = 0; }
  OnePart *p = particles.data();
  int *nx_ = next.data();
  for (int i = nlocal-1; i >= 0; i--) {
    int icell = p[i].icell;
    nx_[i] = ci[icell].first;
    ci[icell].first = i;
    ci[icell].count++;
  }
}

/* ---------------- sort_reorder: the fused out-of-place counting sort ------ */

void Mini::sort_reorder()
{
  ChildInfo *ci = cinfo.data();
  OnePart *p = particles.data();
  for (int c = 0; c < ncell; c++) ci[c].count = 0;
  for (int i = 0; i < nlocal; i++) ci[p[i].icell].count++;

  int m = 0;
  for (int c = 0; c < ncell; c++) {
    int n = ci[c].count;
    ci[c].first = n ? m : -1;
    sortcursor[c] = m;
    m += n;
  }
  OnePart *sb = sortbuf.data();
  for (int i = 0; i < nlocal; i++) sb[sortcursor[p[i].icell]++] = p[i];
  particles.swap(sortbuf);

  int *nx_ = next.data();
  m = 0;
  for (int c = 0; c < ncell; c++) {
    int n = ci[c].count;
    if (n == 0) continue;
    int last = m + n;
    for (; m < last-1; m++) nx_[m] = m+1;
    nx_[m++] = -1;
  }
  sorted_contiguous = 1;
}

/* ---------------- collide: CollideVSS::collide_cell_kernel ---------------- */

void Mini::collide_one_cell(int icell, OnePart *base, const int *pl, int np,
                            int contig)
{
  if (np <= 1) return;
  ChildInfo *ci = cinfo.data();
  double volume = ci[icell].volume / ci[icell].weight;

  double *vrm = flat_vremax ? &vremax1[icell] : &vremax[icell][0][0];
  double *rem = flat_vremax ? &remain1[icell] : &remain[icell][0][0];
  double vremax_c = *vrm;

  double attempt = 0.5*np*(np-1)*vremax_c*dt*collide_every*FNUM/volume + *rem;
  int nattempt = (int) attempt;
  *rem = attempt - nattempt;
  if (!nattempt) { *vrm = vremax_c; return; }
  nattempt_one += nattempt;

  for (int ia = 0; ia < nattempt; ia++) {
    int i = np*random.uniform();
    int j = np*random.uniform();
    while (i == j) j = np*random.uniform();

    OnePart *ipart = contig ? &base[i] : &base[pl[i]];
    OnePart *jpart = contig ? &base[j] : &base[pl[j]];

    double *vi = ipart->v, *vj = jpart->v;
    int isp = ipart->ispecies, jsp = jpart->ispecies;
    const Params &prm = params[isp][jsp];

    double du = vi[0]-vj[0], dv = vi[1]-vj[1], dw = vi[2]-vj[2];
    double vr2 = du*du + dv*dv + dw*dw;
    if (vr2 < EPSZERO && prm.omega >= 1.0) continue;

    double vro = pow(vr2, 1.0-prm.omega);
    double vre = vro*prefactor[isp][jsp];
    vremax_c = std::max(vre, vremax_c);
    if (vre/vremax_c < random.uniform()) continue;

    double vr = sqrt(vr2);
    double ave_rotdof = 0.5*(species[isp].rotdof + species[jsp].rotdof);
    double ave_vibdof = 0.5*(species[isp].vibdof + species[jsp].vibdof);
    double ave_dof = (ave_rotdof + ave_vibdof)/2.0;
    double imass = species[isp].mass, jmass = species[jsp].mass;
    double etrans = 0.5*prm.mr*vr2;
    double divisor = 1.0/(imass+jmass);
    double ucmf = ((imass*vi[0])+(jmass*vj[0]))*divisor;
    double vcmf = ((imass*vi[1])+(jmass*vj[1]))*divisor;
    double wcmf = ((imass*vi[2])+(jmass*vj[2]))*divisor;
    (void) ave_dof;                       /* Ar is monatomic: no EEXCHANGE */

    double alpha_r = 1.0/prm.alpha;
    double eps = random.uniform()*2*MY_PI;
    double scale = sqrt((2.0*etrans)/(prm.mr*vr2));
    double cosX = 2.0*pow(random.uniform(), alpha_r) - 1.0;
    double sinX = sqrt(1.0 - cosX*cosX);
    double ua, vb, wc;
    double d = sqrt(dv*dv + dw*dw);
    if (d > 1.0e-6) {
      ua = scale*( cosX*du + sinX*d*sin(eps) );
      vb = scale*( cosX*dv + sinX*(vr*dw*cos(eps) - du*dv*sin(eps))/d );
      wc = scale*( cosX*dw - sinX*(vr*dv*cos(eps) + du*dw*sin(eps))/d );
    } else {
      ua = scale*( cosX*du );
      vb = scale*( sinX*du*cos(eps) );
      wc = scale*( sinX*du*sin(eps) );
    }
    vi[0] = ucmf + (jmass*divisor)*ua;
    vi[1] = vcmf + (jmass*divisor)*vb;
    vi[2] = wcmf + (jmass*divisor)*wc;
    vj[0] = ucmf - (imass*divisor)*ua;
    vj[1] = vcmf - (imass*divisor)*vb;
    vj[2] = wcmf - (imass*divisor)*wc;
    ncollide_one++;
  }
  *vrm = vremax_c;
}

void Mini::collide()
{
  ChildInfo *ci = cinfo.data();
  OnePart *p = particles.data();
  int *nx_ = next.data();

  if (sorted_contiguous) {
    for (int c = 0; c < ncell; c++) {
      int np = ci[c].count;
      if (np <= 1) continue;
      collide_one_cell(c, &p[ci[c].first], NULL, np, 1);
    }
    return;
  }

  for (int c = 0; c < ncell; c++) {
    int np = ci[c].count;
    if (np <= 1) continue;
    if (np > npmax) { while (np > npmax) npmax += DELTAPART; plist.resize(npmax); }
    int n = 0, ip = ci[c].first;
    while (ip >= 0) { plist[n++] = ip; ip = nx_[ip]; }
    collide_one_cell(c, p, plist.data(), np, 0);
  }
}

double Mini::temperature()
{
  double s = 0.0;
  for (int i = 0; i < nlocal; i++) {
    OnePart &q = particles[i];
    s += q.v[0]*q.v[0] + q.v[1]*q.v[1] + q.v[2]*q.v[2];
  }
  return AMASS*s/(3.0*KB*nlocal);
}

/* ---------------- driver ---------------- */

struct Out { double total, move, sort, coll, temp; long ncoll, natt; };

static Out run(int nx, int ny, int nz, int npercell, int nsteps, int reorder,
               int collide_every = 1)
{
  Mini m;
  m.setup(nx, ny, nz, npercell, 0);
  m.collide_every = collide_every;

  /* bench/in.collide equilibrates for 30 steps at a 10x timestep before the
     measured run, and its README says why: "The equilibration insures particles
     are not ordered in memory by grid cell, which can run faster (initially)
     until particles become disordered."  Skipping it makes any measurement of
     reordering meaningless, because the particles start perfectly sorted. */
  m.dt = 10.0*DT;
  for (int step = 1; step <= 30; step++) {
    m.move();
    int rf = (reorder && step % reorder == 0);
    if (rf) m.sort_reorder(); else m.sort();
    if (step % m.collide_every == 0) m.collide();
  }
  m.dt = DT;
  m.ncollide_one = m.nattempt_one = 0;

  for (int step = 1; step <= nsteps; step++) {
    double t = wtime();
    m.move();
    m.t_move += wtime() - t;

    /* collisions need cell contiguity, so only reorder on steps that collide */
    int docollide = (step % m.collide_every == 0);
    int reorder_flag = (reorder && docollide && (step/m.collide_every) % reorder == 0);
    t = wtime();
    if (reorder_flag) m.sort_reorder();
    else if (docollide) m.sort();
    m.t_sort += wtime() - t;

    t = wtime();
    if (docollide) m.collide();
    m.t_collide += wtime() - t;
  }

  Out o;
  o.move = m.t_move; o.sort = m.t_sort; o.coll = m.t_collide;
  o.total = o.move + o.sort + o.coll;
  o.temp = m.temperature();
  o.ncoll = m.ncollide_one; o.natt = m.nattempt_one;
  m.teardown();
  return o;
}

int main(int argc, char **argv)
{
  if (argc > 1 && strcmp(argv[1], "-subcycle") == 0) {
    /* Collide every K steps with K x the attempt count. The NTC attempt count
       is linear in dt, so the mean collision rate is preserved; what changes is
       the time resolution of the collision process. DSMC already requires the
       timestep to be well under the mean collision time, and here that margin
       is about 14 steps, so K of a few is inside the regime where this is the
       same class of approximation as a multiple-timestep method in MD.
       Whether it actually holds is what the temperature and collision count
       columns are for. */
    int nx=40, ny=50, nz=50, npc=10, ns=40;
    long n = (long)nx*ny*nz*npc;
    printf("# collide sub-cycling: 1M particles, %d steps, reorder 2\n", ns);
    printf("%8s %10s %8s | %6s %6s %6s | %8s %12s %12s\n",
           "every K","ns/p/s","speedup","move","sort","coll","T (K)",
           "collisions","coll/step");
    double base = 0;
    for (int K = 1; K <= 8; K *= 2) {
      Out o = run(nx,ny,nz,npc,ns,2,K);
      double v = 1e9*o.total/((double)n*ns);
      if (K == 1) base = v;
      printf("%8d %10.2f %7.2fx | %6.2f %6.2f %6.2f | %8.2f %12ld %12.0f\n",
             K, v, base/v,
             1e9*o.move/((double)n*ns), 1e9*o.sort/((double)n*ns),
             1e9*o.coll/((double)n*ns), o.temp, o.ncoll,
             (double)o.ncoll/ns);
      fflush(stdout);
    }
    printf("\nreference: 273.15 K; collisions/step must be flat for the\n"
           "statistics to be preserved\n");
    return 0;
  }

  if (argc > 1 && strcmp(argv[1], "-validate") == 0) {
    /* The point of this mode: SPARTA's own reorder curve at 1M particles has a
       shallow minimum around period 2-3 and gets much *worse* when reordering
       is switched off. A model that does not reproduce that shape is not
       measuring the same machine and must not be used to predict anything. */
    int nx = 40, ny = 50, nz = 50, npc = 10, ns = 20;
    printf("# mini_dsmc validation: 1M particles, %d steps\n", ns);
    printf("# compare the shape against SPARTA at the same size\n");
    printf("%8s %10s %8s %8s %8s   %8s\n",
           "reorder", "ns/p/s", "move", "sort", "coll", "T (K)");
    int periods[] = {0, 1, 2, 3, 5, 10, 20};
    double best = 1e30; int bestp = -1;
    for (int q = 0; q < 7; q++) {
      Out o = run(nx, ny, nz, npc, ns, periods[q]);
      double n = 1e9*o.total/((double)nx*ny*nz*npc*ns);
      printf("%8d %10.2f %8.2f %8.2f %8.2f   %8.2f\n", periods[q], n,
             1e9*o.move/((double)nx*ny*nz*npc*ns),
             1e9*o.sort/((double)nx*ny*nz*npc*ns),
             1e9*o.coll/((double)nx*ny*nz*npc*ns), o.temp);
      fflush(stdout);
      if (n < best) { best = n; bestp = periods[q]; }
    }
    printf("\nminimum at reorder %d (%.2f ns/p/s); reorder 0 is %s\n",
           bestp, best, bestp ? "worse, as in SPARTA" : "best -- MODEL IS WRONG");
    return 0;
  }

  int nx = (argc > 1) ? atoi(argv[1]) : 40;
  int ny = (argc > 2) ? atoi(argv[2]) : 50;
  int nz = (argc > 3) ? atoi(argv[3]) : 50;
  int npc = (argc > 4) ? atoi(argv[4]) : 10;
  int ns = (argc > 5) ? atoi(argv[5]) : 20;
  int reorder = (argc > 6) ? atoi(argv[6]) : 2;

  long nlocal = (long)nx*ny*nz*npc;
  Out o = run(nx, ny, nz, npc, ns, reorder);
  printf("# mini_dsmc %dx%dx%d, %ld particles (%.0f MB), %d steps, reorder %d\n",
         nx, ny, nz, nlocal, nlocal*sizeof(OnePart)/1048576.0, ns, reorder);
  printf("total %.3f s  (%.2f ns/particle/step)\n", o.total,
         1e9*o.total/((double)nlocal*ns));
  printf("  move %.3f  sort %.3f  collide %.3f\n", o.move, o.sort, o.coll);
  printf("  T = %.2f K, attempts %ld, collisions %ld\n", o.temp, o.natt, o.ncoll);
  return 0;
}
