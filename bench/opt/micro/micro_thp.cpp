/* Does the particle working set want huge pages?
 *
 * At 1M particles the particle array is 96 MB = ~24000 4 KB pages, well past
 * any dTLB. The counting sort's scatter pass writes into ~100000 per-cell
 * destinations spread across all of it, which is close to a worst case for
 * TLB reach. This measures the same scatter with and without
 * madvise(MADV_HUGEPAGE), which would cut the mapping to ~48 2 MB pages.
 *
 * Also measures the plain streaming copy, to separate "TLB helps the random
 * scatter" from "TLB helps everything".
 *
 * build: g++ -O3 -std=c++11 -o micro_thp micro_thp.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>
#include <sys/mman.h>

static double wtime()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1.0e-9 * ts.tv_nsec;
}

struct alignas(16) OnePart {
  int id, ispecies, icell, flag;
  double x[3], v[3];
  double erot, evib, dtremain, weight;
};

static void *alloc(size_t bytes, int huge)
{
  void *p = NULL;
  if (posix_memalign(&p, 2*1024*1024, bytes) != 0) return NULL;
  if (huge) madvise(p, bytes, MADV_HUGEPAGE);
  else      madvise(p, bytes, MADV_NOHUGEPAGE);
  memset(p, 0, bytes);   /* fault it in so the madvise takes effect */
  return p;
}

int main(int argc, char **argv)
{
  int ncell    = (argc > 1) ? atoi(argv[1]) : 100000;
  int npercell = (argc > 2) ? atoi(argv[2]) : 10;
  int nsteps   = (argc > 3) ? atoi(argv[3]) : 20;

  int nlocal = ncell * npercell;
  size_t bytes = (size_t) nlocal * sizeof(OnePart);
  printf("# micro_thp: %d cells, %d particles, %.1f MB, %d steps\n",
         ncell, nlocal, bytes/1048576.0, nsteps);

  std::vector<int> cursor(ncell), count(ncell);

  for (int huge = 0; huge < 2; huge++) {
    OnePart *a = (OnePart *) alloc(bytes, huge);
    OnePart *b = (OnePart *) alloc(bytes, huge);
    if (!a || !b) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* particles laid out cell-contiguous, then given a realistic churn:
       at this timestep a particle moves ~0.3 of a cell width per axis, so a
       large fraction change cell each step, always to a neighbouring cell */
    unsigned s = 11111u;
    for (int c = 0; c < ncell; c++)
      for (int k = 0; k < npercell; k++) {
        OnePart &p = a[c*npercell + k];
        memset(&p, 0, sizeof(OnePart));
        p.id = c*npercell + k;
        p.icell = c;
      }

    double t_scatter = 0.0, t_copy = 0.0;
    for (int step = 0; step < nsteps; step++) {
      /* churn: ~60% of particles hop to a neighbouring cell */
      for (int i = 0; i < nlocal; i++) {
        s = s*1664525u + 1013904223u;
        if ((s >> 24) < 154) {                 /* ~60% */
          int d = ((s >> 8) % 7) - 3;
          int c = a[i].icell + d;
          if (c < 0) c = 0;
          if (c >= ncell) c = ncell-1;
          a[i].icell = c;
        }
      }

      double t0 = wtime();
      memcpy(b, a, bytes);
      t_copy += wtime() - t0;

      for (int c = 0; c < ncell; c++) count[c] = 0;
      for (int i = 0; i < nlocal; i++) count[a[i].icell]++;
      int m = 0;
      for (int c = 0; c < ncell; c++) { cursor[c] = m; m += count[c]; }

      t0 = wtime();
      for (int i = 0; i < nlocal; i++) b[cursor[a[i].icell]++] = a[i];
      t_scatter += wtime() - t0;

      std::swap(a, b);
    }

    printf("%-22s  scatter %7.4f s (%5.2f ns/particle)   memcpy %7.4f s (%5.2f ns/particle)\n",
           huge ? "MADV_HUGEPAGE" : "MADV_NOHUGEPAGE",
           t_scatter, 1e9*t_scatter/((double)nlocal*nsteps),
           t_copy, 1e9*t_copy/((double)nlocal*nsteps));

    free(a); free(b);
  }
  return 0;
}
