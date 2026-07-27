# Profiling `bench/in.collide`

Two independent profilers, because neither alone is enough here:

- **gprof** (`-O3 -g -pg`) gives wall-clock attribution but inflates the cost of
  small, very frequently called functions (`mcount` runs on every call, and
  `RanKnuth::uniform` alone is called ~320M times).
- **callgrind** (`-O3 -g`, `--cache-sim=yes --branch-sim=yes`) gives exact
  instruction counts, cache misses and branch mispredictions. There is no PMU in
  this KVM guest, so this simulation is the only source of miss counts available.

Reproduce with `bench/opt/profile.sh gprof|callgrind BINARY TAG SIZE REORDER`.

---

## 1. gprof — where the time goes

Baseline build, 1M particles, `optmove yes`, `particle/reorder 5`, using
`bench/opt/in.collide.prof` (30 equilibration steps + **400** benchmark steps, so
the steady state dominates rather than the short high-collision-rate
equilibration).

| % time | self (s) | calls | function |
|-------:|---------:|------:|----------|
| 33.3 | 12.33 | 430 | `Update::move<3,0,1>` |
| 22.9 |  8.50 | 68.4M | `CollideVSS::test_collision` |
| 15.5 |  5.75 | 86 | `Particle::reorder` |
| 12.3 |  4.57 | 432 | `Particle::sort` |
|  7.6 |  2.83 | 430 | `Collide::collisions_one<0,0>` |
|  2.2 |  0.83 | 43.0M | `CollideVSS::attempt_collision` |
|  2.2 |  0.80 | 49.3M | `CollideVSS::SCATTER_TwoBodyScattering` |
|  1.6 |  0.58 | 319.7M | `RanKnuth::uniform` |
|  0.6 |  0.21 | 49.3M | `CollideVSS::setup_collision` |
|  0.3 |  0.10 | 49.3M | `CollideVSS::perform_collision` |

The striking entry is `test_collision` at **124 ns/call**, which is far more than
its ~10 floating point operations and one `pow` can account for. That is the
first hint that the collide kernel's problem is memory and dependency latency
rather than arithmetic — confirmed below.

Note also how cheap the virtual-call machinery turns out to be:
`perform_collision`, `setup_collision` and `attempt_collision` together are under
5% of runtime. This is why devirtualizing them (optimization A2) produced no
measurable gain — see `RESULTS.md`.

---

## 2. callgrind — instruction mix, cache and branches

100K particles, 130 steps (callgrind runs ~50x slower, so the 1M size is
impractical). Whole-program totals:

```
Ir  5,434,281,385    Dr 1,644,195,352    Dw   633,281,041
D1mr    80,737,471   D1mw     2,928,868
DLmr        49,034   DLmw       251,878
Bc     479,455,523   Bcm     18,156,255   Bi 30,755,974   Bim 1,865,088
```

**Last-level misses are negligible at this size** (49K reads out of 1.6G) because
100K particles is only 9.6 MB and fits in cache. The pressure is entirely at L1:
80.7M D1 read misses. At the 1M size the 96 MB particle array does not fit, and
these same accesses become L3/DRAM traffic — so treat the miss *counts* below as
correct and the *cost per miss* as a lower bound for the 1M benchmark.

### Per function

| function | Ir | D1 read miss | Bc mispred | Bi mispred |
|---|---:|---:|---:|---:|
| `Update::move<3,0,1>` | 1.479G (27.2%) | 28.7M (35.5%) | 7.31M (40.3%) | 1.86M (**99.8%**) |
| `__ieee754_pow_fma` | 0.672G (12.4%) | 0.23M | — | — |
| `RanKnuth::uniform` | 0.664G (12.2%) | 861 | 94 | — |
| `SCATTER_TwoBodyScattering` | 0.475G (8.7%) | 130 | 2 | — |
| `Collide::collisions_one<0,0>` | 0.324G (6.0%) | 4.83M (6.0%) | 3.56M (19.6%) | 10.8M (35.0%) |
| `setup_collision` | 0.273G (5.0%) | 260 | — | — |
| `test_collision` | 0.248G (4.6%) | **8.41M (10.4%)** | 1.24M (6.8%) | — |
| `__sincos_fma` | 0.234G (4.3%) | 0.42M | 1.49M (8.2%) | — |
| `pow@@GLIBC` wrapper | 0.146G (2.7%) | 262 | 25 | — |
| `Particle::sort` | 0.146G (2.7%) | **18.1M (22.4%)** | 495 | — |
| `Particle::reorder` | 0.133G (2.5%) | 10.4M (12.9%) | 0.69M (3.8%) | — |

### What this says

1. **`pow` is 15% of all instructions** (`__ieee754_pow_fma` 12.4% plus its
   wrapper 2.7%), from 26.6M calls. Add `__sincos_fma` and the transcendentals
   are ~22% of the instruction stream. `micro/machine_peak` measures `pow` at
   **12.0 ns, i.e. 36 FMA-equivalents**.

2. **`test_collision` is instruction-light but miss-heavy**: 4.6% of instructions
   but 10.4% of D1 read misses. It is the first toucher of two randomly chosen
   particles per attempt, plus `params[i][j]` (two levels of pointer chasing) and
   `vremax[icell][0][0]` (three). This explains gprof's 124 ns/call.

3. **`Particle::sort` is almost pure cache miss**: 2.7% of instructions but
   **22.4% of D1 read misses and 72.8% of D1 write misses**. The write misses are
   the random `cinfo[icell].first/count` updates of the linked-list build.

4. **`move` owns the branch mispredictions**: 40% of all conditional mispredicts
   and 99.8% of all indirect-branch mispredicts. Line-level annotation puts most
   of the conditional ones on the slow path's cell-crossing tests
   (`if (xnew[0] < lo[0])` and friends, ~32% misprediction each). Only 1.09% of
   particles take the slow path in the benchmark run, yet it dominates the
   branch behaviour.

5. **Virtual dispatch is not a problem.** `collisions_one` executes 10.8M indirect
   branches with only 4 mispredictions — the call sites are monomorphic and the
   BTB handles them perfectly. Removing them (A2) saves instruction count, not
   stalls.

### Line-level: the move fast path

```
   Ir          D1mr      source
52,000,000   6,500,000   pflag = particles[i].flag;
52,000,000   6,500,000   xnew[0] = x[0] + dtremain*v[0];
52,000,000   6,500,000   xnew[1] = x[1] + dtremain*v[1];
86,021,320   4,936,743   if (cells[icell].proc != me) {
12,288,760         130   Grid::MyHash::iterator hashptr = grid->hash->find(cellIdx);
```

Three streaming reads of the 96-byte particle record account for 19.5M misses —
exactly the 1.5 cache lines per particle that a 96-byte record costs. This is the
irreducible cost of the layout, and it is why the kernel is bandwidth bound.

The hash lookup shows **almost no misses here**, because this callgrind run has
only 10,000 grid cells and the whole hash fits in cache. At the 1M size there are
100,000 cells and the hash is 10x larger; that is where replacing it with a flat
array (A1) pays off, and the effect is visible in the wall-clock numbers but not
in this trace. A profile taken at one problem size does not transfer to another.
