/* mini_store: storage-layout study built on the validated mini-DSMC of SPARTA's in.collide timestep.
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

#include <type_traits>
/* ================= storage policies =================
 *
 * Each policy owns the particle data and provides the same small interface.
 * Everything else in the timestep -- the cell arrays, vremax's indirection,
 * the VSS math, the control flow of move and collide -- is identical across
 * policies, so a difference between them is a difference of layout and nothing
 * else.
 *
 *   alloc(n) / swap()
 *   xg(i,c) xs(i,c,val)   position component get/set
 *   vg(i,c) vs(i,c,val)   velocity component get/set
 *   cg(i) cs(i,c)         cell index get/set
 *   fg(i) fs(i,f)         migration flag get/set
 *   copy(dst,src)         copy particle src of the live array into slot dst
 *                         of the spare array (the counting sort's scatter)
 *   bytes_per()           for reporting
 */

/* ---- AoS, SPARTA's layout. NB selects 64 or 96 bytes. ---- */
template <int NB>
struct StoreAoS {
  struct SPARTA_ALIGN(16) P64 { int id, ispecies, icell, flag; double x[3], v[3]; };
  struct SPARTA_ALIGN(16) P96 { int id, ispecies, icell, flag; double x[3], v[3];
                                double erot, evib, dtremain, weight; };
  typedef typename std::conditional<NB==64,P64,P96>::type P;

  P *a, *b;
  long n;
  StoreAoS() : a(NULL), b(NULL), n(0) {}
  ~StoreAoS() { free(a); free(b); }
  double bytes_per() const { return sizeof(P); }

  void alloc(long n_) {
    n = n_;
    size_t sz = (size_t)n*sizeof(P);
    if (posix_memalign((void**)&a,64,sz) || posix_memalign((void**)&b,64,sz)) exit(1);
    memset(a,0,sz); memset(b,0,sz);
  }
  inline void swap() { P *t = a; a = b; b = t; }
  inline double xg(long i,int c) const { return a[i].x[c]; }
  inline void   xs(long i,int c,double v) { a[i].x[c] = v; }
  inline double vg(long i,int c) const { return a[i].v[c]; }
  inline void   vs(long i,int c,double v) { a[i].v[c] = v; }
  inline int    cg(long i) const { return a[i].icell; }
  inline void   cs(long i,int c) { a[i].icell = c; }
  inline int    fg(long i) const { return a[i].flag; }
  inline void   fs(long i,int f) { a[i].flag = f; }
  inline void   copy(long dst, long src) { b[dst] = a[src]; }
};

/* ---- SoA: one contiguous array per field. T is double or float. ---- */
template <typename T>
struct StoreSoA {
  T *x0,*x1,*x2,*v0,*v1,*v2;
  T *y0,*y1,*y2,*w0,*w1,*w2;
  int *ic,*ic2,*fl,*fl2;
  long n;
  StoreSoA() : x0(NULL), n(0) {}
  ~StoreSoA() {
    free(x0); free(x1); free(x2); free(v0); free(v1); free(v2);
    free(y0); free(y1); free(y2); free(w0); free(w1); free(w2);
    free(ic); free(ic2); free(fl); free(fl2);
  }
  double bytes_per() const { return 6*sizeof(T) + 2*sizeof(int); }

  static T *al(long n) { void *p; if (posix_memalign(&p,64,(size_t)n*sizeof(T))) exit(1);
                         memset(p,0,(size_t)n*sizeof(T)); return (T*)p; }
  static int *ali(long n) { void *p; if (posix_memalign(&p,64,(size_t)n*sizeof(int))) exit(1);
                            memset(p,0,(size_t)n*sizeof(int)); return (int*)p; }
  void alloc(long n_) {
    n = n_;
    x0=al(n); x1=al(n); x2=al(n); v0=al(n); v1=al(n); v2=al(n);
    y0=al(n); y1=al(n); y2=al(n); w0=al(n); w1=al(n); w2=al(n);
    ic=ali(n); ic2=ali(n); fl=ali(n); fl2=ali(n);
  }
  inline void swap() {
    std::swap(x0,y0); std::swap(x1,y1); std::swap(x2,y2);
    std::swap(v0,w0); std::swap(v1,w1); std::swap(v2,w2);
    std::swap(ic,ic2); std::swap(fl,fl2);
  }
  inline double xg(long i,int c) const { return c==0?x0[i]:(c==1?x1[i]:x2[i]); }
  inline void   xs(long i,int c,double v) { if(c==0)x0[i]=(T)v; else if(c==1)x1[i]=(T)v; else x2[i]=(T)v; }
  inline double vg(long i,int c) const { return c==0?v0[i]:(c==1?v1[i]:v2[i]); }
  inline void   vs(long i,int c,double v) { if(c==0)v0[i]=(T)v; else if(c==1)v1[i]=(T)v; else v2[i]=(T)v; }
  inline int    cg(long i) const { return ic[i]; }
  inline void   cs(long i,int c) { ic[i] = c; }
  inline int    fg(long i) const { return fl[i]; }
  inline void   fs(long i,int f) { fl[i] = f; }
  inline void   copy(long d, long s) {
    y0[d]=x0[s]; y1[d]=x1[s]; y2[d]=x2[s];
    w0[d]=v0[s]; w1[d]=v1[s]; w2[d]=v2[s];
    ic2[d]=ic[s]; fl2[d]=fl[s];
  }
};

/* ---- AoSoA, the Cabana layout: blocks of V lanes, struct-of-arrays inside ---- */
template <int V>
struct StoreAoSoA {
  struct SPARTA_ALIGN(64) Blk {
    double x[3][V], v[3][V];
    int icell[V], flag[V];
  };
  Blk *a, *b;
  long n, nblk;
  StoreAoSoA() : a(NULL), b(NULL), n(0), nblk(0) {}
  ~StoreAoSoA() { free(a); free(b); }
  double bytes_per() const { return (double)sizeof(Blk)/V; }

  void alloc(long n_) {
    n = n_; nblk = (n+V-1)/V;
    size_t sz = (size_t)nblk*sizeof(Blk);
    if (posix_memalign((void**)&a,64,sz) || posix_memalign((void**)&b,64,sz)) exit(1);
    memset(a,0,sz); memset(b,0,sz);
  }
  inline void swap() { Blk *t = a; a = b; b = t; }
  inline double xg(long i,int c) const { return a[i/V].x[c][i%V]; }
  inline void   xs(long i,int c,double v) { a[i/V].x[c][i%V] = v; }
  inline double vg(long i,int c) const { return a[i/V].v[c][i%V]; }
  inline void   vs(long i,int c,double v) { a[i/V].v[c][i%V] = v; }
  inline int    cg(long i) const { return a[i/V].icell[i%V]; }
  inline void   cs(long i,int c) { a[i/V].icell[i%V] = c; }
  inline int    fg(long i) const { return a[i/V].flag[i%V]; }
  inline void   fs(long i,int f) { a[i/V].flag[i%V] = f; }
  inline void   copy(long d, long s) {
    const Blk &S = a[s/V]; int sl = s%V;
    Blk &D = b[d/V]; int dl = d%V;
    for (int c = 0; c < 3; c++) { D.x[c][dl]=S.x[c][sl]; D.v[c][dl]=S.v[c][sl]; }
    D.icell[dl]=S.icell[sl]; D.flag[dl]=S.flag[sl];
  }
};

/* ================= the materialization boundary =================
 *
 * If particle storage becomes SoA, the hot kernels are rewritten natively --
 * but the rest of SPARTA passes Particle::OnePart* by pointer into every
 * surface-collision model, every compute and fix, the Kokkos package and the
 * restart format. Those callers cannot all be converted, so an SoA storage
 * needs a boundary where a particle is gathered into a real OnePart, handed
 * over, and scattered back.
 *
 * The question is what that costs. It is paid only where the boundary is
 * actually crossed: the mover's slow path (~1% of particles per step at this
 * timestep, which is where Domain::collide and the surf models are called),
 * and whatever computes and dumps touch on output steps.
 */

struct SPARTA_ALIGN(16) OnePartView {
  int id, ispecies, icell, flag;
  double x[3], v[3];
  double erot, evib, dtremain, weight;
};

/* stands in for Domain::collide / SurfCollide::collide: an out-of-line callee
   that takes a OnePart* and mutates it, so it cannot be inlined away */
__attribute__((noinline))
static void boundary_collide(OnePartView *p, int dim, double wall)
{
  p->v[dim] = -p->v[dim];
  p->x[dim] = 2.0*wall - p->x[dim];
}

template <class S>
static inline void materialize(const S &st, long i, OnePartView &p)
{
  for (int c = 0; c < 3; c++) { p.x[c] = st.xg(i,c); p.v[c] = st.vg(i,c); }
  p.icell = st.cg(i); p.flag = st.fg(i);
  p.id = 0; p.ispecies = 0;
  p.erot = p.evib = p.dtremain = 0.0; p.weight = 1.0;
}

template <class S>
static inline void writeback(S &st, long i, const OnePartView &p)
{
  for (int c = 0; c < 3; c++) { st.xs(i,c,p.x[c]); st.vs(i,c,p.v[c]); }
  st.cs(i,p.icell); st.fs(i,p.flag);
}

/* ================= the simulation, generic over storage ================= */

template <class S>
struct Sim {
  int nx, ny, nz, ncell, me;
  double boxlo[3], boxhi[3], dx, dy, dz, dt;
  long nlocal;
  int sorted_contiguous;

  S st;
  /* ChildCell is alignas(64) and std::vector does not honour over-aligned
     types before C++17; with -march=native the compiler emits aligned AVX-512
     moves against the declared alignment and faults. SPARTA allocates these
     through memory->smalloc with SPARTA_GET_ALIGN, so allocate them aligned
     here too rather than quietly dropping the alignment. */
  ChildCell *cells;
  std::vector<ChildInfo> cinfo;
  std::vector<int> uniform_index, next, sortcursor, plist;
  int npmax;

  double *vremax1, *remain1;
  double ***vremax, ***remain;
  double *vremax_data, *remain_data, **vremax_l2, **remain_l2;

  Species *species;
  Params **params, *params_data;
  double **prefactor, *prefactor_data;

  RanKnuth random;
  long ncollide_one, nattempt_one;
  double t_move, t_sort, t_collide;

  inline int cell_of(double x, double y, double z) const {
    int i = (int)((x-boxlo[0])/dx), j = (int)((y-boxlo[1])/dy), k = (int)((z-boxlo[2])/dz);
    return (k*ny + j)*nx + i + 1;
  }

  void setup(int nx_, int ny_, int nz_, int npercell);
  void teardown();
  template <int MATB> inline void move_one(long i);
  template <int VEC, int MATB> void move();
  void materialize_all();      /* worst case: a compute touching every particle */
  long nboundary;
  void sort();
  void sort_reorder(int fuse);
  void collide();
  void collide_one_cell(int icell, long base, const int *pl, int np, int contig);
  double temperature();
};

template <class S>
void Sim<S>::setup(int nx_, int ny_, int nz_, int npercell)
{
  nx=nx_; ny=ny_; nz=nz_; ncell=nx*ny*nz; me=0; dt=DT;
  boxlo[0]=boxlo[1]=boxlo[2]=0.0;
  boxhi[0]=nx*CELLL; boxhi[1]=ny*CELLL; boxhi[2]=nz*CELLL;
  dx=dy=dz=CELLL;

  if (posix_memalign((void**)&cells,64,(size_t)ncell*sizeof(ChildCell))) exit(1);
  memset(cells,0,(size_t)ncell*sizeof(ChildCell));
  cinfo.resize(ncell);
  for (int c = 0; c < ncell; c++) {
    int i=c%nx, j=(c/nx)%ny, k=c/(nx*ny);
    cells[c].id=c+1; cells[c].level=0; cells[c].proc=0; cells[c].ilocal=c;
    cells[c].nmask=0; cells[c].nsurf=0; cells[c].csurfs=NULL;
    cells[c].nsplit=1; cells[c].isplit=-1;
    cells[c].lo[0]=i*CELLL; cells[c].lo[1]=j*CELLL; cells[c].lo[2]=k*CELLL;
    cells[c].hi[0]=(i+1)*CELLL; cells[c].hi[1]=(j+1)*CELLL; cells[c].hi[2]=(k+1)*CELLL;
    for (int f=0;f<6;f++) cells[c].neigh[f]=-1;
    if (i>0) cells[c].neigh[0]=c-1;
    if (i<nx-1) cells[c].neigh[1]=c+1;
    if (j>0) cells[c].neigh[2]=c-nx;
    if (j<ny-1) cells[c].neigh[3]=c+nx;
    if (k>0) cells[c].neigh[4]=c-nx*ny;
    if (k<nz-1) cells[c].neigh[5]=c+nx*ny;
    memset(cinfo[c].corner,0,sizeof(cinfo[c].corner));
    cinfo[c].mask=1; cinfo[c].type=0;
    cinfo[c].volume=CELLL*CELLL*CELLL; cinfo[c].weight=1.0;
    cinfo[c].count=0; cinfo[c].first=-1;
  }
  { long m = ((long)nz*ny+ny)*nx+nx+2;
    uniform_index.assign(m,-1);
    for (int c=0;c<ncell;c++) uniform_index[cells[c].id]=c; }

  species = (Species*)calloc(1,sizeof(Species));
  species[0].mass=AMASS; species[0].rotdof=0; species[0].vibdof=0; species[0].specwt=1.0;
  params_data=(Params*)calloc(1,sizeof(Params));
  params=(Params**)malloc(sizeof(Params*)); params[0]=params_data;
  params[0][0].diam=DIAM; params[0][0].omega=OMEGA; params[0][0].tref=TREF;
  params[0][0].alpha=ALPHA; params[0][0].mr=AMASS*AMASS/(AMASS+AMASS);
  prefactor_data=(double*)calloc(1,sizeof(double));
  prefactor=(double**)malloc(sizeof(double*)); prefactor[0]=prefactor_data;
  double cxs=MY_PI*DIAM*DIAM;
  prefactor[0][0]=cxs*pow(2.0*KB*TREF/params[0][0].mr,OMEGA-0.5)/tgamma(2.5-OMEGA);

  vremax_data=(double*)malloc((size_t)ncell*sizeof(double));
  remain_data=(double*)malloc((size_t)ncell*sizeof(double));
  vremax_l2=(double**)malloc((size_t)ncell*sizeof(double*));
  remain_l2=(double**)malloc((size_t)ncell*sizeof(double*));
  vremax=(double***)malloc((size_t)ncell*sizeof(double**));
  remain=(double***)malloc((size_t)ncell*sizeof(double**));
  double vrm0=2.0*cxs*sqrt(2.0*KB*TEMP0/AMASS);
  for (int c=0;c<ncell;c++) {
    vremax_data[c]=vrm0; remain_data[c]=0.0;
    vremax_l2[c]=&vremax_data[c]; remain_l2[c]=&remain_data[c];
    vremax[c]=&vremax_l2[c]; remain[c]=&remain_l2[c];
  }
  vremax1=vremax_data; remain1=remain_data;

  nlocal=(long)ncell*npercell;
  st.alloc(nlocal);
  next.resize(nlocal); sortcursor.resize(ncell);
  npmax=DELTAPART; plist.resize(npmax);
  sorted_contiguous=0;

  random.init(12345);
  double vscale=sqrt(2.0*KB*TEMP0/AMASS);
  long m=0;
  for (int k=0;k<nz;k++) for (int j=0;j<ny;j++) for (int i=0;i<nx;i++)
    for (int p=0;p<npercell;p++) {
      double x[3], v[3];
      x[0]=(i+random.uniform())*CELLL; x[1]=(j+random.uniform())*CELLL;
      x[2]=(k+random.uniform())*CELLL;
      for (int c=0;c<3;c++) {
        double u1=random.uniform(), u2=random.uniform();
        v[c]=vscale*sqrt(-log(u1))*cos(2*MY_PI*u2);
      }
      for (int c=0;c<3;c++) { st.xs(m,c,x[c]); st.vs(m,c,v[c]); }
      st.cs(m, uniform_index[cell_of(x[0],x[1],x[2])]);
      st.fs(m, PKEEP);
      m++;
    }
  ncollide_one=nattempt_one=0; nboundary=0; t_move=t_sort=t_collide=0.0;
}

template <class S>
void Sim<S>::teardown()
{
  free(cells);
  free(species); free(params_data); free(params);
  free(prefactor_data); free(prefactor);
  free(vremax_data); free(remain_data);
  free(vremax_l2); free(remain_l2); free(vremax); free(remain);
}

/* ---- move: SPARTA's control flow. VEC=1 additionally runs the in-box fast
        path over blocks of 8 with no per-particle early exit, which is the
        restructuring SoA makes possible and AoS does not. ---- */
/* one particle, SPARTA's control flow.
   MATB = 1 routes every boundary interaction through a materialized OnePart,
   which is what an SoA storage would have to do to keep calling the existing
   Domain::collide and SurfCollide models. */
template <class S> template <int MATB>
inline void Sim<S>::move_one(long i)
{
  ChildCell *cs_ = cells;
  const int *uidx = uniform_index.data();
  const double *lo = boxlo, *hi = boxhi;

  int pflag = st.fg(i);
  if (pflag == PDONE) st.fs(i,PKEEP);
  double xnew[3], v[3], x[3];
  for (int c=0;c<3;c++) { x[c]=st.xg(i,c); v[c]=st.vg(i,c); }
  for (int c=0;c<3;c++) xnew[c] = x[c] + dt*v[c];

  int optmove = 1;
  for (int c=0;c<3;c++) if (xnew[c] < lo[c] || xnew[c] > hi[c]) optmove = 0;

  if (optmove) {
    int ip=(int)((xnew[0]-lo[0])/dx), jp=(int)((xnew[1]-lo[1])/dy),
        kp=(int)((xnew[2]-lo[2])/dz);
    int icell = uidx[(kp*ny+jp)*nx+ip+1];
    if (icell >= 0) {
      for (int c=0;c<3;c++) st.xs(i,c,xnew[c]);
      st.cs(i,icell); st.fs(i,PKEEP);
      if (cs_[icell].proc != me) st.fs(i,PDONE);
      return;
    }
  }

  st.fs(i,PKEEP);
  nboundary++;
  int icell = st.cg(i);
  while (1) {
    double *clo = cs_[icell].lo, *chi = cs_[icell].hi;
    cellint *neigh = cs_[icell].neigh;
    int outface = -1; double frac = 1.0;
    for (int d=0;d<3;d++) {
      if (xnew[d] < clo[d]) { double f=(clo[d]-x[d])/(xnew[d]-x[d]); if (f<frac){frac=f;outface=2*d;} }
      else if (xnew[d] > chi[d]) { double f=(chi[d]-x[d])/(xnew[d]-x[d]); if (f<frac){frac=f;outface=2*d+1;} }
    }
    if (outface < 0) break;
    for (int c=0;c<3;c++) x[c] += frac*(xnew[c]-x[c]);
    int nb = neigh[outface];
    if (nb < 0) {
      int d = outface/2;
      double wall = (outface & 1) ? hi[d] : lo[d];
      if (MATB) {
        /* the boundary: gather into a real OnePart, hand it to the existing
           collision model, scatter the result back */
        OnePartView pv;
        materialize(st, i, pv);
        for (int c=0;c<3;c++) pv.x[c] = xnew[c];
        pv.v[d] = v[d];
        boundary_collide(&pv, d, wall);
        v[d] = pv.v[d];
        xnew[d] = pv.x[d];
        writeback(st, i, pv);
        st.vs(i,d,v[d]);
      } else {
        v[d] = -v[d]; st.vs(i,d,v[d]);
        xnew[d] = 2.0*wall - xnew[d];
      }
      continue;
    }
    icell = nb;
  }
  for (int c=0;c<3;c++) {
    double q = xnew[c];
    if (q < lo[c]) q = lo[c];
    if (q > hi[c]) q = hi[c]*(1.0-1e-15);
    st.xs(i,c,q);
  }
  st.cs(i, uidx[cell_of(st.xg(i,0),st.xg(i,1),st.xg(i,2))]);
}

/* VEC = 1 runs the in-box fast path over blocks of 8 with no per-particle
   early exit. A block containing any exception falls back to the scalar body
   *for that block only* -- the previous version broke out of the blocked loop
   entirely on the first exception, which at a 1% exception rate sent almost the
   whole array down the scalar path and made the experiment meaningless. */
template <class S> template <int VEC, int MATB>
void Sim<S>::move()
{
  const int *uidx = uniform_index.data();
  const double *lo = boxlo, *hi = boxhi;
  long i = 0;

  if (VEC) {
    const int B = 8;
    for (; i + B <= nlocal; i += B) {
      double xn[3][B];
      for (int c = 0; c < 3; c++)
        for (int l = 0; l < B; l++)
          xn[c][l] = st.xg(i+l,c) + dt*st.vg(i+l,c);

      int ok = 1;
      for (int c = 0; c < 3; c++)
        for (int l = 0; l < B; l++)
          if (xn[c][l] < lo[c] || xn[c][l] > hi[c]) ok = 0;

      int cell[B];
      if (ok) {
        for (int l = 0; l < B; l++) {
          int ip=(int)((xn[0][l]-lo[0])/dx), jp=(int)((xn[1][l]-lo[1])/dy),
              kp=(int)((xn[2][l]-lo[2])/dz);
          cell[l] = uidx[(kp*ny+jp)*nx+ip+1];
        }
        for (int l = 0; l < B; l++) if (cell[l] < 0) ok = 0;
      }

      if (ok) {
        for (int c = 0; c < 3; c++)
          for (int l = 0; l < B; l++) st.xs(i+l,c,xn[c][l]);
        for (int l = 0; l < B; l++) { st.cs(i+l,cell[l]); st.fs(i+l,PKEEP); }
      } else {
        for (int l = 0; l < B; l++) move_one<MATB>(i+l);
      }
    }
  }

  for (; i < nlocal; i++) move_one<MATB>(i);
}

/* worst case for the boundary: a compute or dump that wants every particle */
template <class S>
void Sim<S>::materialize_all()
{
  OnePartView pv;
  double acc = 0.0;
  for (long i = 0; i < nlocal; i++) {
    materialize(st, i, pv);
    acc += pv.v[0] + pv.x[1];
  }
  if (acc == 1.2345e-300) printf("x");
}

template <class S>
void Sim<S>::sort()
{
  sorted_contiguous = 0;
  ChildInfo *ci = cinfo.data();
  for (int c=0;c<ncell;c++) { ci[c].first=-1; ci[c].count=0; }
  int *nx_ = next.data();
  for (long i = nlocal-1; i >= 0; i--) {
    int icell = st.cg(i);
    nx_[i] = ci[icell].first;
    ci[icell].first = (int)i;
    ci[icell].count++;
  }
}

/* fuse = 1 collides each cell the instant the scatter completes it */
template <class S>
void Sim<S>::sort_reorder(int fuse)
{
  ChildInfo *ci = cinfo.data();
  for (int c=0;c<ncell;c++) ci[c].count=0;
  for (long i=0;i<nlocal;i++) ci[st.cg(i)].count++;
  long m=0;
  for (int c=0;c<ncell;c++) { int n=ci[c].count; ci[c].first = n?(int)m:-1; sortcursor[c]=(int)m; m+=n; }

  sorted_contiguous = 1;
  if (!fuse) {
    for (long i=0;i<nlocal;i++) st.copy(sortcursor[st.cg(i)]++, i);
    st.swap();
  } else {
    for (long i=0;i<nlocal;i++) {
      int c = st.cg(i);
      long w = sortcursor[c]++;
      st.copy(w, i);
      if (sortcursor[c] == ci[c].first + ci[c].count) {
        st.swap();                       /* collide reads the destination */
        collide_one_cell(c, ci[c].first, NULL, ci[c].count, 1);
        st.swap();
      }
    }
    st.swap();
  }

  int *nx_ = next.data();
  m=0;
  for (int c=0;c<ncell;c++) {
    int n=ci[c].count; if (!n) continue;
    long last=m+n;
    for (; m<last-1; m++) nx_[m]=(int)(m+1);
    nx_[m++]=-1;
  }
}

template <class S>
void Sim<S>::collide_one_cell(int icell, long base, const int *pl, int np, int contig)
{
  if (np <= 1) return;
  ChildInfo *ci = cinfo.data();
  double volume = ci[icell].volume / ci[icell].weight;
  double *vrm = &vremax1[icell], *rem = &remain1[icell];
  double vremax_c = *vrm;

  double attempt = 0.5*np*(np-1)*vremax_c*dt*FNUM/volume + *rem;
  int nattempt = (int)attempt;
  *rem = attempt - nattempt;
  if (!nattempt) { *vrm = vremax_c; return; }
  nattempt_one += nattempt;

  for (int ia=0; ia<nattempt; ia++) {
    int i = np*random.uniform();
    int j = np*random.uniform();
    while (i==j) j = np*random.uniform();
    long pi = contig ? base+i : pl[i];
    long pj = contig ? base+j : pl[j];

    double vi[3], vj[3];
    for (int c=0;c<3;c++) { vi[c]=st.vg(pi,c); vj[c]=st.vg(pj,c); }
    const Params &prm = params[0][0];

    double du=vi[0]-vj[0], dv=vi[1]-vj[1], dw=vi[2]-vj[2];
    double vr2 = du*du+dv*dv+dw*dw;
    if (vr2 < EPSZERO && prm.omega >= 1.0) continue;
    double vre = pow(vr2,1.0-prm.omega)*prefactor[0][0];
    vremax_c = std::max(vre,vremax_c);
    if (vre/vremax_c < random.uniform()) continue;

    double vr = sqrt(vr2);
    double imass = species[0].mass, jmass = species[0].mass;
    double etrans = 0.5*prm.mr*vr2;
    double divisor = 1.0/(imass+jmass);
    double ucmf=((imass*vi[0])+(jmass*vj[0]))*divisor;
    double vcmf=((imass*vi[1])+(jmass*vj[1]))*divisor;
    double wcmf=((imass*vi[2])+(jmass*vj[2]))*divisor;

    double alpha_r = 1.0/prm.alpha;
    double eps = random.uniform()*2*MY_PI;
    double scale = sqrt((2.0*etrans)/(prm.mr*vr2));
    double cosX = 2.0*pow(random.uniform(),alpha_r)-1.0;
    double sinX = sqrt(1.0-cosX*cosX);
    double ua,vb,wc, d = sqrt(dv*dv+dw*dw);
    if (d > 1.0e-6) {
      ua = scale*( cosX*du + sinX*d*sin(eps) );
      vb = scale*( cosX*dv + sinX*(vr*dw*cos(eps)-du*dv*sin(eps))/d );
      wc = scale*( cosX*dw - sinX*(vr*dv*cos(eps)+du*dw*sin(eps))/d );
    } else {
      ua = scale*(cosX*du); vb = scale*(sinX*du*cos(eps)); wc = scale*(sinX*du*sin(eps));
    }
    st.vs(pi,0,ucmf+(jmass*divisor)*ua); st.vs(pi,1,vcmf+(jmass*divisor)*vb);
    st.vs(pi,2,wcmf+(jmass*divisor)*wc);
    st.vs(pj,0,ucmf-(imass*divisor)*ua); st.vs(pj,1,vcmf-(imass*divisor)*vb);
    st.vs(pj,2,wcmf-(imass*divisor)*wc);
    ncollide_one++;
  }
  *vrm = vremax_c;
}

template <class S>
void Sim<S>::collide()
{
  ChildInfo *ci = cinfo.data();
  int *nx_ = next.data();
  if (sorted_contiguous) {
    for (int c=0;c<ncell;c++) {
      int np=ci[c].count; if (np<=1) continue;
      collide_one_cell(c, ci[c].first, NULL, np, 1);
    }
    return;
  }
  for (int c=0;c<ncell;c++) {
    int np=ci[c].count; if (np<=1) continue;
    if (np>npmax) { while (np>npmax) npmax+=DELTAPART; plist.resize(npmax); }
    int n=0, ip=ci[c].first;
    while (ip>=0) { plist[n++]=ip; ip=nx_[ip]; }
    collide_one_cell(c, 0, plist.data(), np, 0);
  }
}

template <class S>
double Sim<S>::temperature()
{
  double s=0.0;
  for (long i=0;i<nlocal;i++)
    for (int c=0;c<3;c++) { double v=st.vg(i,c); s+=v*v; }
  return AMASS*s/(3.0*KB*nlocal);
}

/* ================= driver ================= */

struct Out { double total, move, sort, coll, temp, bytes; long ncoll, natt, nbound; };

template <class S, int VEC, int FUSE, int MATB>
static Out run(int nx,int ny,int nz,int npc,int nsteps,int reorder)
{
  Sim<S> *m = new Sim<S>();
  m->setup(nx,ny,nz,npc);

  m->dt = 10.0*DT;
  for (int s=1;s<=30;s++) {
    m->template move<VEC,MATB>();
    if (reorder && s%reorder==0) m->sort_reorder(0); else m->sort();
    m->collide();
  }
  m->dt = DT;
  m->ncollide_one = m->nattempt_one = 0; m->nboundary = 0;

  for (int s=1;s<=nsteps;s++) {
    double t=wtime(); m->template move<VEC,MATB>(); m->t_move += wtime()-t;
    int rf = (reorder && s%reorder==0);
    t=wtime();
    if (rf) m->sort_reorder(FUSE); else m->sort();
    m->t_sort += wtime()-t;
    t=wtime();
    if (!(rf && FUSE)) m->collide();
    m->t_collide += wtime()-t;
  }

  Out o;
  o.move=m->t_move; o.sort=m->t_sort; o.coll=m->t_collide;
  o.total=o.move+o.sort+o.coll;
  o.temp=m->temperature(); o.ncoll=m->ncollide_one; o.natt=m->nattempt_one;
  o.bytes=m->st.bytes_per();
  o.nbound=m->nboundary;
  m->teardown();
  delete m;
  return o;
}

static long g_n;
static int g_steps;
static double g_base;

static void row(const char *name, const Out &o)
{
  double ns = 1e9*o.total/((double)g_n*g_steps);
  printf("%-34s %6.0f %8.2f %7.2fx | %6.2f %6.2f %6.2f | %7.2f %9ld\n",
         name, o.bytes, ns, g_base>0?g_base/ns:1.0,
         1e9*o.move/((double)g_n*g_steps),
         1e9*o.sort/((double)g_n*g_steps),
         1e9*o.coll/((double)g_n*g_steps),
         o.temp, o.ncoll);
  fflush(stdout);
}

int main(int argc, char **argv)
{
  int nx = (argc>1)?atoi(argv[1]):40;
  int ny = (argc>2)?atoi(argv[2]):50;
  int nz = (argc>3)?atoi(argv[3]):50;
  int npc = (argc>4)?atoi(argv[4]):10;
  int ns  = (argc>5)?atoi(argv[5]):20;
  int ro  = (argc>6)?atoi(argv[6]):3;

  g_n = (long)nx*ny*nz*npc; g_steps = ns; g_base = 0;

  printf("# mini_store: %dx%dx%d, %ld particles, %d steps, reorder %d\n",
         nx,ny,nz,g_n,ns,ro);
  printf("# every row runs SPARTA's control flow; only the storage differs\n");
  printf("%-34s %6s %8s %8s | %6s %6s %6s | %7s %9s\n",
         "configuration","B/p","ns/p/s","speedup","move","sort","coll","T (K)","ncoll");

  Out base = run<StoreAoS<96>,0,0,0>(nx,ny,nz,npc,ns,ro);
  g_base = 1e9*base.total/((double)g_n*ns);
  row("AoS 96 B  (SPARTA today)", base);
  row("AoS 64 B", run<StoreAoS<64>,0,0,0>(nx,ny,nz,npc,ns,ro));
  row("SoA doubles", run<StoreSoA<double>,0,0,0>(nx,ny,nz,npc,ns,ro));
  row("SoA floats", run<StoreSoA<float>,0,0,0>(nx,ny,nz,npc,ns,ro));
  row("AoSoA V=8", run<StoreAoSoA<8>,0,0,0>(nx,ny,nz,npc,ns,ro));
  row("AoSoA V=16", run<StoreAoSoA<16>,0,0,0>(nx,ny,nz,npc,ns,ro));

  printf("-- blocked fast-path mover (per-block fallback, not per-array) --\n");
  row("AoS 96 B    + blocked move", run<StoreAoS<96>,1,0,0>(nx,ny,nz,npc,ns,ro));
  row("SoA doubles + blocked move", run<StoreSoA<double>,1,0,0>(nx,ny,nz,npc,ns,ro));
  row("SoA floats  + blocked move", run<StoreSoA<float>,1,0,0>(nx,ny,nz,npc,ns,ro));
  row("AoSoA V=8   + blocked move", run<StoreAoSoA<8>,1,0,0>(nx,ny,nz,npc,ns,ro));

  printf("-- materialization boundary: every boundary interaction goes\n"
         "   through a real OnePart, as it must if the rest of SPARTA is\n"
         "   to keep taking OnePart* --\n");
  row("SoA doubles + mat boundary", run<StoreSoA<double>,0,0,1>(nx,ny,nz,npc,ns,ro));
  row("SoA floats  + mat boundary", run<StoreSoA<float>,0,0,1>(nx,ny,nz,npc,ns,ro));
  row("SoA dbl + blocked + mat bnd", run<StoreSoA<double>,1,0,1>(nx,ny,nz,npc,ns,ro));
  row("AoS 96 B    + mat boundary", run<StoreAoS<96>,0,0,1>(nx,ny,nz,npc,ns,ro));

  printf("-- collide fused into the scatter (landed in SPARTA) --\n");
  row("AoS 96 B  + fused collide", run<StoreAoS<96>,0,1,0>(nx,ny,nz,npc,ns,ro));

  printf("\nreference equilibrium temperature %.2f K\n", TEMP0);
  return 0;
}
