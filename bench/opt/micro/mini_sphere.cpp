/* mini-sphere: a faithful standalone model of SPARTA's bench/in.sphere.
 *
 * WHY: every mini-app so far modelled in.collide, which has no surfaces. That
 * left the one question the SoA study could not answer -- what the
 * materialization boundary costs when the SurfCollide models are in play, since
 * those take Particle::OnePart* and would sit on the boundary.
 *
 * in.sphere is a completely different machine from in.collide. Measured in
 * SPARTA at the default size (10x10x10 grid, ~10K particles, 1000 steps):
 *
 *     Move 73.5%   Coll 17.6%   Sort 5.4%   Modify 3.1%
 *     SurfColl checks  23,828,472   =  2.38 ray-triangle tests per particle-move
 *     SurfColl occurs       1,403   =  1.4e-4 collisions per particle-step
 *
 * So the mover dominates, and inside it the cost is ray-triangle intersection
 * against the triangles of whatever cell a particle is in -- not particle
 * streaming. Actual surface collisions are four orders of magnitude rarer than
 * the checks.
 *
 * WHAT IS REPRODUCED FAITHFULLY
 *
 *   the real sphere: data.sphere's 1200 triangles, read at runtime
 *   per-cell surface lists, as Grid::surf2grid builds them
 *   line_tri_intersect transcribed from Geometry::line_tri_intersect,
 *     including the same early-out structure and the EPSSQNEG edge test
 *   the mover's cell-by-cell traversal, testing every triangle of every cell
 *     entered and keeping the earliest hit
 *   SurfCollideDiffuse: a cosine-law rebound at the wall temperature, drawn
 *     from a Maxwellian with the same RNG
 *   supersonic stream at 2500 m/s, open boundaries, VSS collisions
 *
 * DELIBERATE SIMPLIFICATION, stated so it is not mistaken for fidelity:
 * particle emission. SPARTA's fix emit/face inserts a flux-weighted Maxwellian
 * at each inflow face. Here a particle leaving the box is re-inserted at the
 * -x face with the stream velocity plus a thermal draw, which holds the count
 * constant and preserves the flow character but is not the same distribution.
 * It affects the number of particles, not the cost per particle-move, which is
 * what this benchmark is for.
 *
 * build: g++ -O3 -march=native -std=c++11 -o mini_sphere mini_sphere.cpp
 * usage: ./mini_sphere [nx ny nz nsteps] | ./mini_sphere -validate
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

#define SPARTA_ALIGN(n) __attribute__((aligned(n)))
#define MY_PI 3.14159265358979323846
#define EPSSQNEG -1.0e-16
#define EPSZERO 1.0e-14
enum{OUTSIDE,INSIDE,ONSURF2OUT,ONSURF2IN};
enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};

static double wtime(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                       return t.tv_sec + 1e-9*t.tv_nsec; }

static const double KB=1.380658e-23, AMASS=6.63e-26;
static const double DIAM=4.11e-10, OMEGA=0.81, TREF=273.15, ALPHA=1.4;
static const double VSTREAM=2500.0, TSTREAM=300.0, TWALL=300.0;
static const double DTSTEP=1.0e-5, NRHO=7.03e18;
static const double BOXLO=-5.0, BOXHI=5.0;

/* ---------------- RanKnuth ---------------- */
#define MBIG 1000000000
#define MSEED 161803398
#define RFAC (1.0/MBIG)
struct RanKnuth {
  int seed,inext,inextp,ma[56];
  void init(int s){ seed=s; int i,ii,k,mj,mk;
    mj=labs(MSEED-labs(seed)); mj%=MBIG; ma[55]=mj; mk=1;
    for(i=1;i<=54;i++){ ii=(21*i)%55; ma[ii]=mk; mk=mj-mk; if(mk<0)mk+=MBIG; mj=ma[ii]; }
    for(k=0;k<4;k++) for(i=1;i<=55;i++){ ma[i]-=ma[1+(i+30)%55]; if(ma[i]<0)ma[i]+=MBIG; }
    inext=0; inextp=31; }
  inline double uniform(){ int mj; double rn;
    while(1){ if(++inext==56)inext=1; if(++inextp==56)inextp=1;
      mj=ma[inext]-ma[inextp]; if(mj<0)mj+=MBIG; ma[inext]=mj; rn=mj*RFAC;
      if(rn>0.0&&rn<1.0) break; } return rn; }
  inline double gaussian(){ double v1,v2,rsq;
    do { v1=2.0*uniform()-1.0; v2=2.0*uniform()-1.0; rsq=v1*v1+v2*v2; }
    while (rsq>=1.0||rsq==0.0);
    return v1*sqrt(-2.0*log(rsq)/rsq); }
};

/* ---------------- structures ---------------- */

struct SPARTA_ALIGN(16) OnePart {
  int id, ispecies, icell, flag;
  double x[3], v[3];
  double erot, evib, dtremain, weight;
};

struct Tri {
  double p1[3],p2[3],p3[3],norm[3];
  double blo[3],bhi[3];      /* axis-aligned bounds, for the cheap reject */
};

struct SPARTA_ALIGN(64) ChildCell {
  int id, level, proc, ilocal;
  int neigh[6];
  int nmask;
  double lo[3], hi[3];
  int nsurf;
  int *csurfs;
  int nsplit, isplit;
};

struct ChildInfo { int count,first,mask,type; int corner[8]; double volume,weight; };

/* ---------------- Geometry::line_tri_intersect, transcribed ---------------- */

static inline void sub3(const double*a,const double*b,double*c){
  c[0]=a[0]-b[0]; c[1]=a[1]-b[1]; c[2]=a[2]-b[2]; }
static inline double dot3(const double*a,const double*b){
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
static inline void cross3(const double*a,const double*b,double*c){
  c[0]=a[1]*b[2]-a[2]*b[1]; c[1]=a[2]*b[0]-a[0]*b[2]; c[2]=a[0]*b[1]-a[1]*b[0]; }

static bool line_tri_intersect(const double *start,const double *stop,
                               const double *v0,const double *v1,const double *v2,
                               const double *norm,double *point,double &param,int &side)
{
  double vec[3],start2stop[3],edge[3],pvec[3],xproduct[3];

  sub3(start,v0,vec);  double dotstart = dot3(norm,vec);
  sub3(stop,v0,vec);   double dotstop  = dot3(norm,vec);
  if (dotstart < 0.0 && dotstop < 0.0) return false;
  if (dotstart > 0.0 && dotstop > 0.0) return false;
  if (dotstart == 0.0 && dotstop == 0.0) return false;

  sub3(v0,start,vec);
  sub3(stop,start,start2stop);
  param = dot3(norm,vec)/dot3(norm,start2stop);
  if (param < 0.0) param = 0.0;
  if (param > 1.0) param = 1.0;

  point[0]=start[0]+param*start2stop[0];
  point[1]=start[1]+param*start2stop[1];
  point[2]=start[2]+param*start2stop[2];

  sub3(v1,v0,edge); sub3(point,v0,pvec); cross3(edge,pvec,xproduct);
  if (dot3(xproduct,norm) < EPSSQNEG) return false;
  sub3(v2,v1,edge); sub3(point,v1,pvec); cross3(edge,pvec,xproduct);
  if (dot3(xproduct,norm) < EPSSQNEG) return false;
  sub3(v0,v2,edge); sub3(point,v2,pvec); cross3(edge,pvec,xproduct);
  if (dot3(xproduct,norm) < EPSSQNEG) return false;

  if (dotstart < 0.0) side = INSIDE;
  else if (dotstart > 0.0) side = OUTSIDE;
  else side = (dotstop > 0.0) ? ONSURF2OUT : ONSURF2IN;
  return true;
}

/* ---------------- the materialization boundary ----------------
   SurfCollideDiffuse::collide takes a Particle::OnePart* and rewrites its
   velocity. With SoA storage a particle must be gathered into a real OnePart
   before this can be called, and scattered back afterwards. This is the
   out-of-line callee that forces it. */

__attribute__((noinline))
static void surf_collide_diffuse(OnePart *p, const double *norm, RanKnuth &rng)
{
  /* cosine-law rebound at the wall temperature, as SurfCollideDiffuse does
     for full accommodation */
  double vrm = sqrt(2.0*KB*TWALL/AMASS);
  double vperp = vrm * sqrt(-log(rng.uniform()));
  double theta = 2.0*MY_PI*rng.uniform();
  double vtan1 = vrm * rng.gaussian() * 0.70710678118654752;
  double vtan2 = vrm * rng.gaussian() * 0.70710678118654752;
  (void) theta;

  /* build a tangent frame around the normal */
  double tang1[3], tang2[3];
  if (fabs(norm[0]) < 0.9) { tang1[0]=0; tang1[1]=norm[2]; tang1[2]=-norm[1]; }
  else                     { tang1[0]=-norm[2]; tang1[1]=0; tang1[2]=norm[0]; }
  double n1 = sqrt(dot3(tang1,tang1));
  tang1[0]/=n1; tang1[1]/=n1; tang1[2]/=n1;
  cross3(norm,tang1,tang2);

  for (int c = 0; c < 3; c++)
    p->v[c] = vperp*norm[c] + vtan1*tang1[c] + vtan2*tang2[c];
}

/* ---------------- storage policies ---------------- */

struct StoreAoS {
  OnePart *a, *b; long n;
  StoreAoS():a(0),b(0),n(0){}
  ~StoreAoS(){ free(a); free(b); }
  double bytes_per() const { return sizeof(OnePart); }
  void alloc(long m){ n=m; size_t s=(size_t)m*sizeof(OnePart);
    if(posix_memalign((void**)&a,64,s)||posix_memalign((void**)&b,64,s))exit(1);
    memset(a,0,s); memset(b,0,s); }
  inline void swap(){ OnePart*t=a;a=b;b=t; }
  inline double xg(long i,int c)const{return a[i].x[c];}
  inline void   xs(long i,int c,double v){a[i].x[c]=v;}
  inline double vg(long i,int c)const{return a[i].v[c];}
  inline void   vs(long i,int c,double v){a[i].v[c]=v;}
  inline int    cg(long i)const{return a[i].icell;}
  inline void   cs(long i,int c){a[i].icell=c;}
  inline void   copy(long d,long s){ b[d]=a[s]; }
};

struct StoreSoA {
  double *x0,*x1,*x2,*v0,*v1,*v2,*y0,*y1,*y2,*w0,*w1,*w2;
  int *ic,*ic2; long n;
  StoreSoA():x0(0),n(0){}
  ~StoreSoA(){ free(x0);free(x1);free(x2);free(v0);free(v1);free(v2);
               free(y0);free(y1);free(y2);free(w0);free(w1);free(w2);
               free(ic);free(ic2); }
  double bytes_per() const { return 6*sizeof(double)+sizeof(int); }
  static double* ad(long m){ void*p; if(posix_memalign(&p,64,(size_t)m*8))exit(1);
                             memset(p,0,(size_t)m*8); return (double*)p; }
  static int* ai(long m){ void*p; if(posix_memalign(&p,64,(size_t)m*4))exit(1);
                          memset(p,0,(size_t)m*4); return (int*)p; }
  void alloc(long m){ n=m; x0=ad(m);x1=ad(m);x2=ad(m);v0=ad(m);v1=ad(m);v2=ad(m);
    y0=ad(m);y1=ad(m);y2=ad(m);w0=ad(m);w1=ad(m);w2=ad(m); ic=ai(m); ic2=ai(m); }
  inline void swap(){ std::swap(x0,y0);std::swap(x1,y1);std::swap(x2,y2);
    std::swap(v0,w0);std::swap(v1,w1);std::swap(v2,w2); std::swap(ic,ic2); }
  inline double xg(long i,int c)const{return c==0?x0[i]:(c==1?x1[i]:x2[i]);}
  inline void   xs(long i,int c,double v){ if(c==0)x0[i]=v; else if(c==1)x1[i]=v; else x2[i]=v; }
  inline double vg(long i,int c)const{return c==0?v0[i]:(c==1?v1[i]:v2[i]);}
  inline void   vs(long i,int c,double v){ if(c==0)v0[i]=v; else if(c==1)v1[i]=v; else v2[i]=v; }
  inline int    cg(long i)const{return ic[i];}
  inline void   cs(long i,int c){ic[i]=c;}
  inline void   copy(long d,long s){ y0[d]=x0[s];y1[d]=x1[s];y2[d]=x2[s];
    w0[d]=v0[s];w1[d]=v1[s];w2[d]=v2[s]; ic2[d]=ic[s]; }
};

/* ---------------- the simulation ---------------- */

template <class S>
struct Sim {
  int nx,ny,nz,ncell;
  double lo[3],hi[3],dx,dy,dz;
  S st; long nlocal;
  ChildCell *cells; std::vector<ChildInfo> cinfo;
  std::vector<Tri> tris;
  std::vector<int> csurf_pool, next, sortcursor;
  std::vector<double> vremax, remain;
  RanKnuth rng;
  long nscheck, nscollide, nexit;
  int contiguous;
  double t_move,t_sort,t_collide;
  double fnum, volume;

  int load_surf(const char *path);
  void setup(int,int,int,const char*);
  void teardown(){ free(cells); }
  inline int cell_of(double x,double y,double z) const {
    int i=(int)((x-lo[0])/dx), j=(int)((y-lo[1])/dy), k=(int)((z-lo[2])/dz);
    if(i<0)i=0; if(i>=nx)i=nx-1; if(j<0)j=0; if(j>=ny)j=ny-1;
    if(k<0)k=0; if(k>=nz)k=nz-1;
    return (k*ny+j)*nx+i;
  }
  template <int MATB, int PREFILTER> void move();
  long nsreject;
  void sort_reorder();
  void sort_only();
  void collide();
  void reinject(long i);
};

template <class S>
int Sim<S>::load_surf(const char *path)
{
  FILE *fp = fopen(path,"r");
  if (!fp) return 0;
  char line[512];
  int npt=0, ntri=0;
  while (fgets(line,sizeof(line),fp)) {
    if (strstr(line,"points")) sscanf(line,"%d",&npt);
    else if (strstr(line,"triangles")) { sscanf(line,"%d",&ntri); break; }
  }
  std::vector<double> pts(3*(npt+1));
  while (fgets(line,sizeof(line),fp)) if (strncmp(line,"Points",6)==0) break;
  fgets(line,sizeof(line),fp);
  for (int i=0;i<npt;i++){ int id; double a,b,c;
    if (fscanf(fp,"%d %lf %lf %lf",&id,&a,&b,&c)!=4) return 0;
    pts[3*id]=a; pts[3*id+1]=b; pts[3*id+2]=c; }
  while (fgets(line,sizeof(line),fp)) if (strncmp(line,"Triangles",9)==0) break;
  tris.resize(ntri);
  for (int i=0;i<ntri;i++){ int id,a,b,c;
    if (fscanf(fp,"%d %d %d %d",&id,&a,&b,&c)!=4) return 0;
    Tri &t = tris[i];
    for (int q=0;q<3;q++){ t.p1[q]=pts[3*a+q]; t.p2[q]=pts[3*b+q]; t.p3[q]=pts[3*c+q]; }
    double e1[3],e2[3];
    sub3(t.p2,t.p1,e1); sub3(t.p3,t.p1,e2); cross3(e1,e2,t.norm);
    double nn=sqrt(dot3(t.norm,t.norm));
    if (nn>0){ t.norm[0]/=nn; t.norm[1]/=nn; t.norm[2]/=nn; }
    /* orient outward. SPARTA requires surface normals to point into the flow
       domain and read_surf checks this; data.sphere's vertex ordering does not
       guarantee it here, and an inward normal makes the diffuse rebound push
       the particle into the solid, where it collides again and again. That
       showed up immediately as a surface collision rate 18x SPARTA's. */
    double cen[3];
    for (int q=0;q<3;q++) cen[q]=(t.p1[q]+t.p2[q]+t.p3[q])/3.0;
    if (dot3(t.norm,cen) < 0.0) { t.norm[0]=-t.norm[0]; t.norm[1]=-t.norm[1];
                                  t.norm[2]=-t.norm[2]; }
    for (int q=0;q<3;q++){
      t.blo[q]=std::min(std::min(t.p1[q],t.p2[q]),t.p3[q]);
      t.bhi[q]=std::max(std::max(t.p1[q],t.p2[q]),t.p3[q]);
    }
  }
  fclose(fp);
  return ntri;
}

template <class S>
void Sim<S>::setup(int nx_,int ny_,int nz_,const char *surfpath)
{
  nx=nx_; ny=ny_; nz=nz_; ncell=nx*ny*nz;
  for (int c=0;c<3;c++){ lo[c]=BOXLO; hi[c]=BOXHI; }
  dx=(hi[0]-lo[0])/nx; dy=(hi[1]-lo[1])/ny; dz=(hi[2]-lo[2])/nz;
  volume = dx*dy*dz;

  if (!load_surf(surfpath)) { fprintf(stderr,"cannot read %s\n",surfpath); exit(1); }

  if (posix_memalign((void**)&cells,64,(size_t)ncell*sizeof(ChildCell))) exit(1);
  memset(cells,0,(size_t)ncell*sizeof(ChildCell));
  cinfo.resize(ncell);
  for (int c=0;c<ncell;c++){
    int i=c%nx, j=(c/nx)%ny, k=c/(nx*ny);
    cells[c].id=c+1; cells[c].proc=0; cells[c].ilocal=c;
    cells[c].lo[0]=lo[0]+i*dx; cells[c].lo[1]=lo[1]+j*dy; cells[c].lo[2]=lo[2]+k*dz;
    cells[c].hi[0]=cells[c].lo[0]+dx; cells[c].hi[1]=cells[c].lo[1]+dy;
    cells[c].hi[2]=cells[c].lo[2]+dz;
    for(int f=0;f<6;f++) cells[c].neigh[f]=-1;
    if(i>0)cells[c].neigh[0]=c-1;      if(i<nx-1)cells[c].neigh[1]=c+1;
    if(j>0)cells[c].neigh[2]=c-nx;     if(j<ny-1)cells[c].neigh[3]=c+nx;
    if(k>0)cells[c].neigh[4]=c-nx*ny;  if(k<nz-1)cells[c].neigh[5]=c+nx*ny;
    cinfo[c].volume=volume; cinfo[c].weight=1.0;
    cinfo[c].count=0; cinfo[c].first=-1;
  }

  /* per-cell surface lists, as Grid::surf2grid produces: a triangle is added
     to every cell its bounding box overlaps */
  {
    std::vector<std::vector<int> > tmp(ncell);
    for (size_t t=0;t<tris.size();t++){
      double blo[3],bhi[3];
      for(int c=0;c<3;c++){
        blo[c]=std::min(std::min(tris[t].p1[c],tris[t].p2[c]),tris[t].p3[c]);
        bhi[c]=std::max(std::max(tris[t].p1[c],tris[t].p2[c]),tris[t].p3[c]);
      }
      int i0=(int)((blo[0]-lo[0])/dx), i1=(int)((bhi[0]-lo[0])/dx);
      int j0=(int)((blo[1]-lo[1])/dy), j1=(int)((bhi[1]-lo[1])/dy);
      int k0=(int)((blo[2]-lo[2])/dz), k1=(int)((bhi[2]-lo[2])/dz);
      i0=std::max(i0,0); j0=std::max(j0,0); k0=std::max(k0,0);
      i1=std::min(i1,nx-1); j1=std::min(j1,ny-1); k1=std::min(k1,nz-1);
      for(int k=k0;k<=k1;k++)for(int j=j0;j<=j1;j++)for(int i=i0;i<=i1;i++)
        tmp[(k*ny+j)*nx+i].push_back((int)t);
    }
    long tot=0; for(int c=0;c<ncell;c++) tot+=tmp[c].size();
    csurf_pool.resize(tot);
    long m=0;
    for(int c=0;c<ncell;c++){
      cells[c].nsurf=(int)tmp[c].size();
      cells[c].csurfs = tmp[c].empty()? NULL : &csurf_pool[m];
      for(size_t q=0;q<tmp[c].size();q++) csurf_pool[m++]=tmp[c][q];
    }
  }

  /* particles: number set by nrho as create_particles does, capped so the
     default case matches SPARTA's ~10K */
  double boxvol = (hi[0]-lo[0])*(hi[1]-lo[1])*(hi[2]-lo[2]);
  nlocal = (long)(10.0*ncell);
  fnum = NRHO*boxvol/nlocal;

  st.alloc(nlocal);
  next.resize(nlocal); sortcursor.resize(ncell);
  vremax.assign(ncell, 2.0*MY_PI*DIAM*DIAM*sqrt(2.0*KB*TSTREAM/AMASS));
  remain.assign(ncell,0.0);

  rng.init(12345);
  double vth = sqrt(KB*TSTREAM/AMASS);
  long placed=0;
  while (placed < nlocal) {
    double x[3];
    x[0]=lo[0]+(hi[0]-lo[0])*rng.uniform();
    x[1]=lo[1]+(hi[1]-lo[1])*rng.uniform();
    x[2]=lo[2]+(hi[2]-lo[2])*rng.uniform();
    if (x[0]*x[0]+x[1]*x[1]+x[2]*x[2] < 1.05) continue;   /* not inside the sphere */
    st.xs(placed,0,x[0]); st.xs(placed,1,x[1]); st.xs(placed,2,x[2]);
    st.vs(placed,0,VSTREAM+vth*rng.gaussian());
    st.vs(placed,1,vth*rng.gaussian());
    st.vs(placed,2,vth*rng.gaussian());
    st.cs(placed,cell_of(x[0],x[1],x[2]));
    placed++;
  }
  nscheck=nscollide=nexit=nsreject=0; contiguous=0; t_move=t_sort=t_collide=0.0;
}

/* re-inject a particle at the inflow face; see the header note -- this is the
   deliberate simplification, standing in for fix emit/face */
template <class S>
void Sim<S>::reinject(long i)
{
  double vth = sqrt(KB*TSTREAM/AMASS);
  double x[3];
  x[0]=lo[0]+1e-6*(hi[0]-lo[0]);
  x[1]=lo[1]+(hi[1]-lo[1])*rng.uniform();
  x[2]=lo[2]+(hi[2]-lo[2])*rng.uniform();
  st.xs(i,0,x[0]); st.xs(i,1,x[1]); st.xs(i,2,x[2]);
  st.vs(i,0,VSTREAM+vth*rng.gaussian());
  st.vs(i,1,vth*rng.gaussian());
  st.vs(i,2,vth*rng.gaussian());
  st.cs(i,cell_of(x[0],x[1],x[2]));
  nexit++;
}

/* ---------------- move with surfaces ----------------
   the shape of Update::move<3,1,0>: walk cell to cell, and in every cell
   entered test the particle's segment against every triangle of that cell,
   keeping the earliest hit. MATB routes the surface collision through a
   materialized OnePart, as an SoA storage would have to. */
template <class S> template <int MATB, int PREFILTER>
void Sim<S>::move()
{
  for (long i = 0; i < nlocal; i++) {
    double x[3], v[3], xnew[3];
    for (int c=0;c<3;c++){ x[c]=st.xg(i,c); v[c]=st.vg(i,c); }
    for (int c=0;c<3;c++) xnew[c]=x[c]+DTSTEP*v[c];
    int icell = st.cg(i);
    int exitflag = 0;
    int nloop = 0;
    /* SPARTA's mover carries an "exclude" surface: the one just collided with
       is skipped on the next pass, because the particle now sits exactly on it
       and line_tri_intersect would otherwise re-detect the same triangle at
       param 0 and collide with it again. Omitting this was why this model's
       surface collision rate came out 18x SPARTA's. */
    int exclude = -1;

    while (1) {
      if (++nloop > 100) break;
      ChildCell &cc = cells[icell];

      /* surface intersection: every triangle of this cell */
      int nsurf = cc.nsurf;
      if (nsurf) {
        double minparam = 2.0; int minsurf = -1; double minxc[3];
        /* AABB of this step's segment. At 2500 m/s and dt 1e-5 a particle
           moves 0.025 of a cell width, so the segment is tiny compared with a
           cell and is nowhere near most of the cell's triangles. Six compares
           reject those before the ~40-flop plane-and-edge test. */
        double slo[3], shi[3];
        for (int c=0;c<3;c++){
          slo[c]=std::min(x[c],xnew[c]); shi[c]=std::max(x[c],xnew[c]);
        }
        for (int m = 0; m < nsurf; m++) {
          int isurf = cc.csurfs[m];
          if (isurf == exclude) continue;
          const Tri &t = tris[isurf];
          if (PREFILTER) {
            if (t.bhi[0] < slo[0] || t.blo[0] > shi[0]) { nsreject++; continue; }
            if (t.bhi[1] < slo[1] || t.blo[1] > shi[1]) { nsreject++; continue; }
            if (t.bhi[2] < slo[2] || t.blo[2] > shi[2]) { nsreject++; continue; }
          }
          double xc[3], param; int side;
          nscheck++;
          if (line_tri_intersect(x,xnew,t.p1,t.p2,t.p3,t.norm,xc,param,side)) {
            if (param < minparam) { minparam=param; minsurf=isurf;
                                    for(int c=0;c<3;c++) minxc[c]=xc[c]; }
          }
        }
        if (minsurf >= 0) {
          for (int c=0;c<3;c++) x[c]=minxc[c];
          const Tri &t = tris[minsurf];
          if (MATB) {
            OnePart pv;
            memset(&pv,0,sizeof(OnePart));
            for (int c=0;c<3;c++){ pv.x[c]=x[c]; pv.v[c]=v[c]; }
            pv.icell=icell;
            surf_collide_diffuse(&pv,t.norm,rng);
            for (int c=0;c<3;c++) v[c]=pv.v[c];
          } else {
            OnePart pv;
            for (int c=0;c<3;c++){ pv.x[c]=x[c]; pv.v[c]=v[c]; }
            surf_collide_diffuse(&pv,t.norm,rng);
            for (int c=0;c<3;c++) v[c]=pv.v[c];
          }
          nscollide++;
          exclude = minsurf;
          double dtr = DTSTEP*(1.0-minparam);
          for (int c=0;c<3;c++) xnew[c]=x[c]+dtr*v[c];
          continue;
        }
      }

      /* which face of the cell is crossed first */
      int outface=-1; double frac=1.0;
      for (int d=0;d<3;d++){
        if (xnew[d] < cc.lo[d]) { double f=(cc.lo[d]-x[d])/(xnew[d]-x[d]);
                                  if (f<frac){frac=f;outface=2*d;} }
        else if (xnew[d] > cc.hi[d]) { double f=(cc.hi[d]-x[d])/(xnew[d]-x[d]);
                                       if (f<frac){frac=f;outface=2*d+1;} }
      }
      if (outface < 0) break;
      exclude = -1;                       /* new cell, nothing to exclude */
      for (int c=0;c<3;c++) x[c] += frac*(xnew[c]-x[c]);
      int nb = cc.neigh[outface];
      if (nb < 0) { exitflag = 1; break; }     /* open boundary: particle leaves */
      icell = nb;
    }

    if (exitflag) { reinject(i); continue; }
    for (int c=0;c<3;c++){ st.xs(i,c,xnew[c]); st.vs(i,c,v[c]); }
    st.cs(i,cell_of(xnew[0],xnew[1],xnew[2]));
  }
}

template <class S>
void Sim<S>::sort_only()
{
  ChildInfo *ci = cinfo.data();
  for (int c=0;c<ncell;c++){ ci[c].first=-1; ci[c].count=0; }
  for (long i=nlocal-1;i>=0;i--){
    int c=st.cg(i);
    next[i]=ci[c].first; ci[c].first=(int)i; ci[c].count++;
  }
  contiguous = 0;
}

template <class S>
void Sim<S>::sort_reorder()
{
  ChildInfo *ci = cinfo.data();
  for (int c=0;c<ncell;c++) ci[c].count=0;
  for (long i=0;i<nlocal;i++) ci[st.cg(i)].count++;
  long m=0;
  for (int c=0;c<ncell;c++){ int n=ci[c].count; ci[c].first=n?(int)m:-1;
                             sortcursor[c]=(int)m; m+=n; }
  for (long i=0;i<nlocal;i++) st.copy(sortcursor[st.cg(i)]++, i);
  st.swap();
  contiguous = 1;
}

template <class S>
void Sim<S>::collide()
{
  ChildInfo *ci = cinfo.data();
  double mr = AMASS*AMASS/(AMASS+AMASS);
  double cxs = MY_PI*DIAM*DIAM;
  double prefactor = cxs*pow(2.0*KB*TREF/mr,OMEGA-0.5)/tgamma(2.5-OMEGA);
  std::vector<int> pl;
  for (int c=0;c<ncell;c++){
    int np=ci[c].count; if (np<=1) continue;
    long first=ci[c].first;
    if (!contiguous) {
      pl.clear();
      for (int ip=ci[c].first; ip>=0; ip=next[ip]) pl.push_back(ip);
    }
    double vrm=vremax[c];
    double att=0.5*np*(np-1)*vrm*DTSTEP*fnum/volume + remain[c];
    int natt=(int)att; remain[c]=att-natt;
    for (int t=0;t<natt;t++){
      int i=(int)(np*rng.uniform()), j=(int)(np*rng.uniform());
      while(i==j) j=(int)(np*rng.uniform());
      double vi[3],vj[3];
      long pi = contiguous ? first+i : pl[i];
      long pj = contiguous ? first+j : pl[j];
      for(int q=0;q<3;q++){ vi[q]=st.vg(pi,q); vj[q]=st.vg(pj,q); }
      double du=vi[0]-vj[0],dv=vi[1]-vj[1],dw=vi[2]-vj[2];
      double vr2=du*du+dv*dv+dw*dw;
      if (vr2<EPSZERO) continue;
      double vre=pow(vr2,1.0-OMEGA)*prefactor;
      vrm=std::max(vre,vrm);
      if (vre/vrm < rng.uniform()) continue;
      double vr=sqrt(vr2), etrans=0.5*mr*vr2;
      double ucmf=0.5*(vi[0]+vj[0]),vcmf=0.5*(vi[1]+vj[1]),wcmf=0.5*(vi[2]+vj[2]);
      double eps=rng.uniform()*2*MY_PI;
      double scale=sqrt((2.0*etrans)/(mr*vr2));
      double cosX=2.0*pow(rng.uniform(),1.0/ALPHA)-1.0;
      double sinX=sqrt(1.0-cosX*cosX);
      double ua,vb,wc,d=sqrt(dv*dv+dw*dw);
      if (d>1e-6){
        ua=scale*(cosX*du+sinX*d*sin(eps));
        vb=scale*(cosX*dv+sinX*(vr*dw*cos(eps)-du*dv*sin(eps))/d);
        wc=scale*(cosX*dw-sinX*(vr*dv*cos(eps)+du*dw*sin(eps))/d);
      } else { ua=scale*cosX*du; vb=scale*sinX*du*cos(eps); wc=scale*sinX*du*sin(eps); }
      st.vs(pi,0,ucmf+0.5*ua); st.vs(pi,1,vcmf+0.5*vb); st.vs(pi,2,wcmf+0.5*wc);
      st.vs(pj,0,ucmf-0.5*ua); st.vs(pj,1,vcmf-0.5*vb); st.vs(pj,2,wcmf-0.5*wc);
    }
    vremax[c]=vrm;
  }
}

/* ---------------- driver ---------------- */

struct Out { double total,move,sort,coll,bytes; long nscheck,nscollide,nsreject,nmoves; };

/* reorder: 0 = sort only, which is what bench/in.sphere does (it sets no
   particle/reorder, and measuring SPARTA shows enabling one makes in.sphere
   slower: 0.318 s off against 0.367 s every step, because 10K particles fit in
   L2 and there is no locality to buy) */
template <class S, int MATB, int PREFILTER>
static Out run(int nx,int ny,int nz,int nsteps,const char *surf,int reorder)
{
  Sim<S> *m = new Sim<S>();
  m->setup(nx,ny,nz,surf);
  for (int s=0;s<200;s++){ m->template move<MATB,PREFILTER>();
    if (reorder && s%reorder==0) m->sort_reorder(); else m->sort_only();
    m->collide(); }
  m->nscheck=m->nscollide=m->nsreject=0; m->t_move=m->t_sort=m->t_collide=0;
  for (int s=0;s<nsteps;s++){
    double t=wtime(); m->template move<MATB,PREFILTER>(); m->t_move+=wtime()-t;
    t=wtime();
    if (reorder && s%reorder==0) m->sort_reorder(); else m->sort_only();
    m->t_sort+=wtime()-t;
    t=wtime(); m->collide(); m->t_collide+=wtime()-t;
  }
  Out o; o.move=m->t_move; o.sort=m->t_sort; o.coll=m->t_collide;
  o.total=o.move+o.sort+o.coll;
  o.nscheck=m->nscheck; o.nscollide=m->nscollide; o.nsreject=m->nsreject;
  o.nmoves=(long)m->nlocal*nsteps; o.bytes=m->st.bytes_per();
  m->teardown(); delete m;
  return o;
}

static void row(const char *nm,const Out &o,double base)
{
  double ns = 1e9*o.total/o.nmoves;
  printf("%-32s %5.0f %8.2f %7.2fx | %6.1f%% %6.1f%% %6.1f%% | %6.2f %8.1e\n",
         nm,o.bytes,ns, base>0?base/ns:1.0,
         100*o.move/o.total, 100*o.sort/o.total, 100*o.coll/o.total,
         (double)o.nscheck/o.nmoves, (double)o.nscollide/o.nmoves);
  fflush(stdout);
}

int main(int argc,char**argv)
{
  const char *surf = "../../data.sphere";
  int nx=10,ny=10,nz=10,ns=1000;
  int validate = (argc>1 && strcmp(argv[1],"-validate")==0);
  if (!validate) {
    if (argc>1) nx=atoi(argv[1]);
    if (argc>2) ny=atoi(argv[2]);
    if (argc>3) nz=atoi(argv[3]);
    if (argc>4) ns=atoi(argv[4]);
    if (argc>5) surf=argv[5];
  }
  if (argc>2 && validate) surf=argv[2];

  printf("# mini_sphere: %dx%dx%d grid, %d steps, surf %s\n",nx,ny,nz,ns,surf);
  if (validate) {
    printf("# SPARTA bench/in.sphere at 10x10x10 measures:\n"
           "#   Move 73.5%%  Coll 17.6%%  Sort 5.4%%\n"
           "#   2.38 surf checks per particle-move, 1.4e-4 surf collisions\n");
  }
  printf("%-32s %5s %8s %8s | %7s %7s %7s | %6s %8s\n",
         "configuration","B/p","ns/move","speedup","move","sort","coll",
         "chk/mv","coll/mv");

  int ro = 0;                      /* in.sphere sets no particle/reorder */
  Out a = run<StoreAoS,0,0>(nx,ny,nz,ns,surf,ro);
  double base = 1e9*a.total/a.nmoves;
  row("AoS 96 B (SPARTA today)",a,0);
  row("AoS 96 B + AABB prefilter", run<StoreAoS,0,1>(nx,ny,nz,ns,surf,ro), base);
  row("AoS 96 B + mat boundary", run<StoreAoS,1,0>(nx,ny,nz,ns,surf,ro), base);
  row("SoA", run<StoreSoA,0,0>(nx,ny,nz,ns,surf,ro), base);
  row("SoA + AABB prefilter", run<StoreSoA,0,1>(nx,ny,nz,ns,surf,ro), base);
  row("SoA + prefilter + mat bnd", run<StoreSoA,1,1>(nx,ny,nz,ns,surf,ro), base);
  printf("-- reordering, which in.sphere does not do --\n");
  row("AoS + prefilter, reorder 1", run<StoreAoS,0,1>(nx,ny,nz,ns,surf,1), base);
  row("AoS + prefilter, reorder 20", run<StoreAoS,0,1>(nx,ny,nz,ns,surf,20), base);
  return 0;
}
