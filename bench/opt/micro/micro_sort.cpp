/* Microbenchmark of SPARTA's particle sort/reorder.
 *
 * Shipped path, per timestep (Update::run):
 *   Particle::sort()     - clear cinfo, then a reverse loop over particles
 *                          pushing each onto its cell's linked list. Random
 *                          writes into cinfo[], one per particle.
 *   Particle::reorder()  - every reorder_period steps: walk the linked lists
 *                          to build order[], then apply the permutation
 *                          *in place*, one cycle at a time, memcpy'ing
 *                          96-byte structs. Then rebuild next[].
 *
 * Proposed replacement (S2): a single stable counting sort straight into a
 * second particle buffer (ping-pong). Counts come from one pass over icell,
 * a prefix sum gives cinfo.first, and one scatter pass places every particle.
 * This produces exactly the ordering sort()+reorder() produces, but as two
 * streaming passes instead of a random-write linked-list build plus an
 * in-place cycle permutation.
 *
 * S3 additionally assumes the counts were accumulated during move (which
 * already computes each particle's destination cell), removing the count pass.
 *
 * Variants:
 *   S0  sort() only                      (what runs on non-reorder steps)
 *   S1  sort() + reorder()               (what runs on reorder steps)
 *   S2  fused out-of-place counting sort (count + prefix + scatter)
 *   S3  fused, counts supplied by move   (prefix + scatter)
 *
 * The harness checks S1/S2/S3 all produce the identical particle ordering.
 *
 * build: g++ -O3 -std=c++11 -o micro_sort micro_sort.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

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

struct ChildInfo {          /* matching Grid::ChildInfo's hot fields */
  int count;
  int first;
  int mask, type;
  int corner[8];
  double volume, weight;
};

/* ---------------- S0: Particle::sort() ---------------- */

static void sparta_sort(OnePart *particles, int nlocal,
                        ChildInfo *cinfo, int nglocal, int *next)
{
  for (int icell = 0; icell < nglocal; icell++) {
    cinfo[icell].first = -1;
    cinfo[icell].count = 0;
  }
  for (int i = nlocal-1; i >= 0; i--) {
    int icell = particles[i].icell;
    next[i] = cinfo[icell].first;
    cinfo[icell].first = i;
    cinfo[icell].count++;
  }
}

/* ---------------- S1: Particle::reorder() ---------------- */

static void sparta_reorder(OnePart *particles, int nlocal,
                           ChildInfo *cinfo, int nglocal, int *next,
                           int *order)
{
  const int nbytes = sizeof(OnePart);
  if (nlocal == 0) return;

  int m = 0;
  for (int icell = 0; icell < nglocal; icell++) {
    int ip = cinfo[icell].first;
    while (ip >= 0) { order[m++] = ip; ip = next[ip]; }
  }

  OnePart copy;
  for (int i = 0; i < nlocal; i++) {
    if (order[i] == i) continue;
    memcpy(&copy, &particles[i], nbytes);
    int dst = i;
    int src = order[i];
    while (src != i) {
      memcpy(&particles[dst], &particles[src], nbytes);
      order[dst] = dst;
      dst = src;
      src = order[src];
    }
    memcpy(&particles[dst], &copy, nbytes);
    order[dst] = dst;
  }

  m = 0;
  for (int icell = 0; icell < nglocal; icell++) {
    if (cinfo[icell].count == 0) { cinfo[icell].first = -1; continue; }
    cinfo[icell].first = m;
    int last = m + cinfo[icell].count;
    for (; m < last-1; m++) next[m] = m+1;
    next[m++] = -1;
  }
}

/* ---------------- S2/S3: fused counting sort, out of place ---------------- */

/* COUNTED = 1 means the per-cell counts were already accumulated elsewhere
   (during move), so the counting pass is skipped. */
template <int COUNTED>
static void fused_counting_sort(const OnePart *in, OnePart *out, int nlocal,
                                ChildInfo *cinfo, int nglocal, int *cursor)
{
  if (!COUNTED) {
    for (int icell = 0; icell < nglocal; icell++) cinfo[icell].count = 0;
    for (int i = 0; i < nlocal; i++) cinfo[in[i].icell].count++;
  }

  /* prefix sum -> cinfo.first, and a write cursor per cell */
  int sum = 0;
  for (int icell = 0; icell < nglocal; icell++) {
    int c = cinfo[icell].count;
    cinfo[icell].first = c ? sum : -1;
    cursor[icell] = sum;
    sum += c;
  }

  /* stable scatter: particles keep their relative order within a cell,
     exactly as sort()+reorder() leaves them */
  for (int i = 0; i < nlocal; i++) {
    int icell = in[i].icell;
    out[cursor[icell]++] = in[i];
  }
}

int main(int argc, char **argv)
{
  int ncell    = (argc > 1) ? atoi(argv[1]) : 100000;
  int npercell = (argc > 2) ? atoi(argv[2]) : 10;
  int nsteps   = (argc > 3) ? atoi(argv[3]) : 20;
  /* churn = fraction of particles that changed cell since the last sort;
     at the benchmark's timestep this is small, which is why "nearly sorted"
     matters for the in-place permutation's cost */
  double churn = (argc > 4) ? atof(argv[4]) : 0.05;
  int scramble = (argc > 5) ? atoi(argv[5]) : 0;

  int nlocal = ncell * npercell;
  printf("# micro_sort: %d cells, %d particles (%.1f MB), %d steps, "
         "churn=%.2f scramble=%d\n",
         ncell, nlocal, nlocal*sizeof(OnePart)/1048576.0, nsteps,
         churn, scramble);

  std::vector<OnePart> proto(nlocal);
  {
    unsigned s = 24680u;
    std::vector<int> order(nlocal);
    for (int i = 0; i < nlocal; i++) order[i] = i;
    if (scramble)
      for (int i = nlocal-1; i > 0; i--) {
        s = s*1664525u + 1013904223u;
        std::swap(order[i], order[(int)(s % (unsigned)(i+1))]);
      }
    for (int c = 0; c < ncell; c++)
      for (int k = 0; k < npercell; k++) {
        OnePart &p = proto[order[c*npercell + k]];
        memset(&p, 0, sizeof(OnePart));
        p.id = c*npercell + k;
        p.icell = c;
        p.weight = 1.0;
      }
  }

  /* per-step cell churn: the same deterministic reassignment for all variants */
  std::vector<int> churn_idx, churn_cell;
  {
    unsigned s = 13579u;
    int nch = (int)(churn * nlocal);
    churn_idx.resize(nch); churn_cell.resize(nch);
    for (int i = 0; i < nch; i++) {
      s = s*1664525u + 1013904223u; churn_idx[i]  = (int)(s % (unsigned)nlocal);
      s = s*1664525u + 1013904223u; churn_cell[i] = (int)(s % (unsigned)ncell);
    }
  }

  std::vector<ChildInfo> cinfo(ncell);
  std::vector<int> next(nlocal), order(nlocal), cursor(ncell);
  std::vector<OnePart> a, b;

  double t[4]; std::vector<int> final_order[4];

  for (int v = 0; v < 4; v++) {
    a = proto;
    b.assign(nlocal, OnePart());
    memset(cinfo.data(), 0, cinfo.size()*sizeof(ChildInfo));

    /* only the sort work itself is timed; the churn that stands in for move
       is applied outside the accumulator and is identical for all variants */
    double acc = 0.0;
    for (int s = 0; s < nsteps; s++) {
      for (size_t k = 0; k < churn_idx.size(); k++)
        a[churn_idx[k]].icell = churn_cell[k];

      if (v == 3) {
        /* counts accumulated during move, so they are not the sort's cost */
        for (int i = 0; i < ncell; i++) cinfo[i].count = 0;
        for (int i = 0; i < nlocal; i++) cinfo[a[i].icell].count++;
      }

      double t0 = wtime();
      if (v == 0) {
        sparta_sort(a.data(), nlocal, cinfo.data(), ncell, next.data());
      } else if (v == 1) {
        sparta_sort(a.data(), nlocal, cinfo.data(), ncell, next.data());
        sparta_reorder(a.data(), nlocal, cinfo.data(), ncell,
                       next.data(), order.data());
      } else if (v == 2) {
        fused_counting_sort<0>(a.data(), b.data(), nlocal,
                               cinfo.data(), ncell, cursor.data());
        a.swap(b);
      } else {
        fused_counting_sort<1>(a.data(), b.data(), nlocal,
                               cinfo.data(), ncell, cursor.data());
        a.swap(b);
      }
      acc += wtime() - t0;
    }
    t[v] = acc;

    final_order[v].resize(nlocal);
    for (int i = 0; i < nlocal; i++) final_order[v][i] = a[i].id;
  }

  const char *names[4] = {"S0 sort() only",
                          "S1 sort() + reorder()  (shipped)",
                          "S2 fused counting sort (count+scatter)",
                          "S3 fused, counts from move (scatter only)"};
  for (int v = 0; v < 4; v++) {
    bool same = (v == 0) ? true : (final_order[v] == final_order[1]);
    printf("%-44s  %8.4f s  %7.3fx vs S1  %6.2f ns/particle/step  order=%s\n",
           names[v], t[v], t[1]/t[v],
           1e9*t[v]/((double)nlocal*nsteps),
           (v == 0) ? "-" : (same ? "identical" : "DIFFERENT"));
  }
  return 0;
}
