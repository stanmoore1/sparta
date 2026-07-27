/* Microbenchmark of the SPARTA optimized-move kernel (Update::move<3,0,1>).
 *
 * The shipped fast path, for a particle that stays inside the box, is:
 *   xnew = x + dt*v; bounds check; ip/jp/kp = floor((xnew-boxlo)/d);
 *   cellIdx = (kp*uny + jp)*unx + ip + 1;
 *   icell   = grid->hash->find(cellIdx)     <-- std::unordered_map lookup
 *   store x, icell, flag
 *
 * optmove is only legal on a uniform grid with no surfaces (Update::init),
 * which means cellIdx -> icell is a dense, total map over [0, unx*uny*unz)
 * and the hash can be replaced by one indexed load. This benchmark measures
 * that, plus how much of the cost is the particle streaming itself.
 *
 * Variants:
 *   M0 hash          - std::unordered_map, as shipped
 *   M1 flat table    - int* lookup table indexed by cellIdx
 *   M2 no lookup     - cell index used directly (upper bound: what the
 *                      lookup costs at all)
 *   M3 flat + SoA-ish- flat table, but only the first 64 B of OnePart touched
 *
 * Particle order can be contiguous-by-cell (post-reorder) or shuffled, to
 * show how much the lookup's cache behavior depends on reordering.
 *
 * build: g++ -O3 -std=c++11 -o micro_move micro_move.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <unordered_map>

static double wtime()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1.0e-9 * ts.tv_nsec;
}

struct alignas(16) OnePart {
  int id;
  int ispecies;
  int icell;
  int flag;
  double x[3];
  double v[3];
  double erot;
  double evib;
  double dtremain;
  double weight;
};

typedef std::unordered_map<int,int> MyHash;

/* the fields of Grid::ChildCell that matter here; the real struct is 128 B
   and 64-byte aligned, so reading .proc pulls a full line out of a
   cells[] array that is 12.8 MB at this problem size */
struct alignas(64) ChildCell {
  long long id;
  int level, proc, ilocal;
  long long neigh[6];
  int nmask;
  double lo[3], hi[3];
  int nsurf;
  void *csurfs;
  int nsplit, isplit;
};

struct Grid {
  int unx, uny, unz;
  double boxlo[3], boxhi[3];
  double dx, dy, dz;
  MyHash hash;
  std::vector<int> table;      /* cellIdx -> icell, -1 if absent */
  std::vector<int> table_own;  /* same, but ownership encoded in the sign:
                                  >=0 owned, <=-2 ghost (icell = -2-v),
                                  -1 absent */
  std::vector<ChildCell> cells;
};

/* ---- the four move variants; LOOKUP selects which mapping is used ---- */

template <int LOOKUP>
static long move_kernel(OnePart *particles, int nlocal, const Grid &g, double dt)
{
  long ntouch = 0;                 /* particles taking the slow path */
  long nmigrate = 0;
  const double *boxlo = g.boxlo, *boxhi = g.boxhi;
  const double dx = g.dx, dy = g.dy, dz = g.dz;
  const int unx = g.unx, uny = g.uny;
  const int *table = g.table.data();
  const int *table_own = g.table_own.data();
  const ChildCell *cells = g.cells.data();

  for (int i = 0; i < nlocal; i++) {
    double *x = particles[i].x;
    double *v = particles[i].v;
    double xnew[3];
    xnew[0] = x[0] + dt*v[0];
    xnew[1] = x[1] + dt*v[1];
    xnew[2] = x[2] + dt*v[2];

    int optmove = 1;
    if (xnew[0] < boxlo[0] || xnew[0] > boxhi[0]) optmove = 0;
    if (xnew[1] < boxlo[1] || xnew[1] > boxhi[1]) optmove = 0;
    if (xnew[2] < boxlo[2] || xnew[2] > boxhi[2]) optmove = 0;

    if (optmove) {
      const int ip = (int)((xnew[0] - boxlo[0])/dx);
      const int jp = (int)((xnew[1] - boxlo[1])/dy);
      const int kp = (int)((xnew[2] - boxlo[2])/dz);
      int cellIdx = (kp*uny + jp)*unx + ip + 1;

      int icell = -1;
      int mine = 1;
      if (LOOKUP == 0) {
        MyHash::const_iterator it = g.hash.find(cellIdx);
        if (it != g.hash.end()) icell = it->second;
      } else if (LOOKUP == 1 || LOOKUP == 3) {
        icell = table[cellIdx];
      } else if (LOOKUP == 4) {
        /* flat table, then the shipped ownership test against cells[] */
        icell = table[cellIdx];
        if (icell >= 0) mine = (cells[icell].proc == 0);
      } else if (LOOKUP == 5) {
        /* ownership folded into the table, so cells[] is never touched */
        int v = table_own[cellIdx];
        if (v >= 0) icell = v;
        else if (v <= -2) { icell = -2 - v; mine = 0; }
      } else {
        icell = cellIdx - 1;
      }

      if (icell >= 0) {
        particles[i].icell = icell;
        particles[i].flag = 0;
        x[0] = xnew[0]; x[1] = xnew[1]; x[2] = xnew[2];
        if (!mine) nmigrate++;
        continue;
      }
    }

    /* slow path stand-in: reflect off the box, as boundary "rr" does */
    for (int c = 0; c < 3; c++) {
      if (xnew[c] < boxlo[c]) { xnew[c] = 2.0*boxlo[c] - xnew[c]; v[c] = -v[c]; }
      else if (xnew[c] > boxhi[c]) { xnew[c] = 2.0*boxhi[c] - xnew[c]; v[c] = -v[c]; }
      x[c] = xnew[c];
    }
    ntouch++;
  }
  return ntouch + (nmigrate & 0);   /* keep nmigrate live without changing the count */
}

int main(int argc, char **argv)
{
  int nx       = (argc > 1) ? atoi(argv[1]) : 40;
  int ny       = (argc > 2) ? atoi(argv[2]) : 50;
  int nz       = (argc > 3) ? atoi(argv[3]) : 50;
  int npercell = (argc > 4) ? atoi(argv[4]) : 10;
  int nsteps   = (argc > 5) ? atoi(argv[5]) : 20;
  int scramble = (argc > 6) ? atoi(argv[6]) : 0;

  Grid g;
  g.unx = nx; g.uny = ny; g.unz = nz;
  double L = 1.0e-5;
  for (int c = 0; c < 3; c++) g.boxlo[c] = 0.0;
  g.boxhi[0] = nx*L; g.boxhi[1] = ny*L; g.boxhi[2] = nz*L;
  g.dx = L; g.dy = L; g.dz = L;

  int ncell = nx*ny*nz;
  int nlocal = ncell * npercell;

  /* cellIdx runs 1..ncell; icell is the local index. In a serial run after
     balance_grid the two differ by a permutation, which is what makes the
     lookup a real (non-identity) map. Model that with a fixed permutation. */
  g.table.assign(ncell + 2, -1);
  g.table_own.assign(ncell + 2, -1);
  g.cells.assign(ncell, ChildCell());
  std::vector<int> cell_of_idx(ncell);
  for (int i = 0; i < ncell; i++) cell_of_idx[i] = i;
  {   /* deterministic permutation, standing in for the RCB cell ordering */
    unsigned s = 987654321u;
    for (int i = ncell-1; i > 0; i--) {
      s = s*1664525u + 1013904223u;
      int j = (int)(s % (unsigned)(i+1));
      std::swap(cell_of_idx[i], cell_of_idx[j]);
    }
  }
  for (int i = 0; i < ncell; i++) {
    g.table[i+1] = cell_of_idx[i];
    g.table_own[i+1] = cell_of_idx[i];      /* serial run: every cell is owned */
    g.hash[i+1]  = cell_of_idx[i];
    g.cells[cell_of_idx[i]].proc = 0;
  }

  printf("# micro_move: grid %dx%dx%d = %d cells, %d particles (%.1f MB), "
         "%d steps, scramble=%d\n",
         nx, ny, nz, ncell, nlocal, nlocal*sizeof(OnePart)/1048576.0,
         nsteps, scramble);

  /* build particles: uniform in the box, Maxwellian velocities */
  std::vector<OnePart> proto(nlocal);
  {
    unsigned s = 12345u;
    double kb = 1.380658e-23, mass = 6.63e-26;
    double vth = sqrt(2.0*kb*273.15/mass);
    std::vector<int> order(nlocal);
    for (int i = 0; i < nlocal; i++) order[i] = i;
    if (scramble) {
      for (int i = nlocal-1; i > 0; i--) {
        s = s*1664525u + 1013904223u;
        std::swap(order[i], order[(int)(s % (unsigned)(i+1))]);
      }
    }
    for (int c = 0; c < ncell; c++) {
      int ix = c % nx, iy = (c/nx) % ny, iz = c/(nx*ny);
      for (int k = 0; k < npercell; k++) {
        int slot = order[c*npercell + k];
        OnePart &p = proto[slot];
        p.id = slot; p.ispecies = 0; p.icell = cell_of_idx[c]; p.flag = 0;
        s = s*1664525u + 1013904223u; double u1 = (s>>8)*(1.0/16777216.0);
        s = s*1664525u + 1013904223u; double u2 = (s>>8)*(1.0/16777216.0);
        s = s*1664525u + 1013904223u; double u3 = (s>>8)*(1.0/16777216.0);
        p.x[0] = (ix + u1)*L; p.x[1] = (iy + u2)*L; p.x[2] = (iz + u3)*L;
        for (int cc = 0; cc < 3; cc++) {
          s = s*1664525u + 1013904223u; double a = ((s>>8)+1)*(1.0/16777217.0);
          s = s*1664525u + 1013904223u; double b = (s>>8)*(1.0/16777216.0);
          p.v[cc] = vth*sqrt(-log(a))*cos(2*M_PI*b)*0.70710678118654752;
        }
        p.erot = p.evib = 0.0; p.dtremain = 0.0; p.weight = 1.0;
      }
    }
  }

  const double dt = 7.0e-9;
  const char *names[6] = {"M0 unordered_map (shipped lookup)",
                          "M1 flat lookup table",
                          "M2 no lookup at all (bound)",
                          "M3 flat table, hot fields only",
                          "M4 flat table + cells[].proc test (real move)",
                          "M5 ownership folded into the table"};
  double base = 0.0;

  for (int v = 0; v < 6; v++) {
    std::vector<OnePart> particles = proto;
    long touched = 0;
    double t0 = wtime();
    for (int s = 0; s < nsteps; s++) {
      switch (v) {
        case 0: touched = move_kernel<0>(particles.data(), nlocal, g, dt); break;
        case 1: touched = move_kernel<1>(particles.data(), nlocal, g, dt); break;
        case 2: touched = move_kernel<2>(particles.data(), nlocal, g, dt); break;
        case 3: touched = move_kernel<3>(particles.data(), nlocal, g, dt); break;
        case 4: touched = move_kernel<4>(particles.data(), nlocal, g, dt); break;
        case 5: touched = move_kernel<5>(particles.data(), nlocal, g, dt); break;
      }
    }
    double t1 = wtime();
    if (v == 0) base = t1 - t0;
    printf("%-34s  %8.4f s  %7.3fx  %6.2f ns/particle/step  slowpath=%.2f%%\n",
           names[v], t1-t0, base/(t1-t0),
           1e9*(t1-t0)/((double)nlocal*nsteps),
           100.0*touched/nlocal);
  }
  return 0;
}
