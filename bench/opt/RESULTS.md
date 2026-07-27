# SPARTA `bench/in.collide` — CPU optimization results

All numbers are single-core, `make serial` (g++ 13.3, `-O3 -std=c++11`, the shipped
`src/MAKE/Makefile.serial` flags). **Compiler flags are frozen across every code
measurement in this file** so that speedups are attributable to code changes rather
than build options; flag experiments are reported separately in section 6.

Machine: 4-vCPU Intel Xeon (Sapphire Rapids class, AVX-512 + AMX), 15 GB RAM, KVM guest.
Runs are pinned with `taskset -c 2`. Each configuration is run N times and reported as
median (min–max). Run-to-run spread on this shared VM is 2–13%, so **differences under
about 5% are not meaningful** and are called out as such rather than claimed as wins.
The `Loop time` used is the **second** one in the log — the 100-step benchmark run, not
the 30-step equilibration.

Benchmark input: `bench/in.collide.opt`, identical physics to the shipped
`bench/in.collide` plus `global optmove yes` and `global particle/reorder ${reorder}`.
Primary size is **1M particles** (`-var x 40 -var y 50 -var z 50`, 100K cells, 10
particles per cell), per the sizes documented in `bench/README`. That is the largest
size that keeps a median-of-5 comparison to a couple of minutes on one core, which is
what makes a profile → optimize → remeasure loop practical; 10M would cost ~25 minutes
per comparison.

Reproduce with:

```
bench/opt/run_bench.sh   -b BINARY -s 1M -r 3 -n 5      # time a configuration
bench/opt/sweep_reorder.sh BINARY 1M 3                  # sweep the reorder period
bench/opt/verify.sh      REF NEW 100K 3                 # bitwise output check
bench/opt/regress.sh     REF NEW                        # in-tree example regressions
bench/opt/profile.sh     gprof|callgrind BINARY TAG     # profiles
```

---

## 1. Baseline: choosing the reorder period

`global optmove yes` is on throughout. `particle/reorder 0` disables reordering;
`sort()` still runs every step because collisions need the per-cell lists.

Reordering is worth a great deal — it roughly halves both Move and Collide by
restoring locality — but the reorder itself costs ~43 ms/step at 1M particles, so
there is an optimum.

| reorder | loop time (s) median | min | Move | Coll | Sort |
|--------:|---------------------:|----:|-----:|-----:|-----:|
| 0   | 14.05 | 12.72 | 5.81 | 6.80 | 1.34 |
| 1   | 10.32 |  9.88 | 2.26 | 2.32 | 5.66 |
| 2   |  8.61 |  8.17 | 2.20 | 2.51 | 3.82 |
| 3   |  8.44 |  7.76 | 2.39 | 2.84 | 3.12 |
| 4   |  7.36 |  7.09 | 2.27 | 2.67 | 2.33 |
| **5** | **7.34** | **7.01** | 2.33 | 2.75 | 2.17 |
| 6   |  7.77 |  6.82 | 2.51 | 3.04 | 2.13 |
| 8   |  7.68 |  7.34 | 2.64 | 3.07 | 1.87 |
| 10  |  7.84 |  7.37 | 2.88 | 3.12 | 1.75 |
| 20  |  9.41 |  8.90 | 3.88 | 3.86 | 1.58 |
| 50  | 11.07 | 10.76 | 4.67 | 4.85 | 1.45 |
| 100 | 12.90 | 12.73 | 5.30 | 6.07 | 1.43 |

Rows 0, 1, 2, 10, 20, 50 and 100 come from a 3-repetition sweep; rows 3, 4, 5, 6 and
8 from a 5-repetition sweep run afterwards to resolve the minimum. The two sweeps
agree to within their spread where they overlap (period 5: 7.23 s over 3 reps,
7.34 s over 5).

Periods 4, 5 and 6 are statistically tied. **The baseline is `particle/reorder 5`**,
which has the lowest median and sits in the middle of the flat basin.

> **Baseline: 7.34 s** at 1M particles / 100 steps, split Move 32% / Collide 37% / Sort 30%.

The shape of this curve is itself a finding: unreordered, the same physics takes
14.05 s. Nearly half the runtime of the shipped default configuration is locality
loss rather than useful work.

---

## 2. What the profile said

Full detail in `PROFILE.md`; the four facts that drove everything below:

1. `pow` is **15% of all instructions** and measures **12.0 ns — 36 FMA-equivalents**.
2. `test_collision` is instruction-light (4.6%) but miss-heavy (10.4% of D1 read
   misses) — gprof shows it at 124 ns/call, which arithmetic cannot explain.
3. `Particle::sort` is 2.7% of instructions but **22% of D1 read misses and 73% of
   D1 write misses** — the random `cinfo` updates of the linked-list build.
4. Virtual dispatch is **not** a problem: `collisions_one` issues 10.8M indirect
   branches with 4 mispredictions.

And from `ROOFLINE.md`: move and sort sit at ~30–40% of the DRAM bandwidth roof,
while **collide sits at 7% of the scalar compute roof and 5% of its bandwidth
roof — it is latency bound**, stalled on dependent loads, not short of FLOPs.

---

## 3. Tier A — optimizations that are bitwise identical

Each of these preserves the random number stream and the order of floating point
operations exactly, so the optimized binary must reproduce baseline output character
for character. It does: see section 5.

| id | change | files |
|----|--------|-------|
| A1 | Replace the `std::unordered_map` cell-ID lookup in the `optmove` fast path with a dense array. `optmove` already requires a uniform grid, so the map is total over `1..unx*uny*unz`; `Grid::rehash()` maintains it so it can never go stale. Declines (and falls back to the hash) if the dense array would be much larger than the cells a proc owns. | `grid.{h,cpp}`, `update.cpp` |
| A2 | A VSS-specific fused collision kernel (`CollideVSS::collisions_one_opt`) that inlines the attempt/test/setup/scatter math instead of reaching it through four virtual calls per attempt. Declines and falls back for chemistry, ambipolar, near-neighbour, gas-tally, multi-group and Poisson-attempt cases. | `collide.{h,cpp}`, `collide_vss.{h,cpp}` |
| A3 | Inline `RanKnuth::uniform()` into the header; the seeding stays out of line. Same arithmetic, same stream. | `random_knuth.{h,cpp}` |
| A4 | Hoist per-cell invariants in the collision loop: `params`/`prefactor` rows as references, and `vremax[icell][0][0]` (three levels of pointer chasing) held in a register across the cell and written back once. | `collide_vss.cpp` |
| A5 | Skip building `plist` when the particles of a cell are already contiguous in memory — a new `Particle::sorted_contiguous` flag. This removes a ten-deep dependent load chain per cell. | `particle.h`, `collide_vss.cpp` |
| A6 | Replace `sort()` + `reorder()` with a fused, out-of-place, stable counting sort (`Particle::sort_reorder`). Produces exactly the same ordering, `cinfo.first/count` and `next[]`, but as three streaming passes instead of a random-write linked-list build followed by an in-place cycle permutation of 96-byte structs. | `particle.{h,cpp}`, `update.cpp` |

**A6 costs memory**: being out of place, it needs a second particle buffer, so peak
particle memory doubles (a 10M-particle run would go from 955 MB to 1.9 GB). The buffer is
allocated lazily inside `sort_reorder()`, so a run with `particle/reorder 0` — the
default — pays nothing. `Particle::memory_usage()` reports it, so it shows up in the
run's memory summary rather than silently. Runs that are memory-limited rather than
time-limited should leave reordering off, or use a larger period.

### Measured, at the baseline's own reorder period (5), so the comparison is like for like

| configuration | loop (s) | Move | Coll | Sort |
|---|---:|---:|---:|---:|
| baseline, reorder 5 | 7.34 | 2.33 | 2.75 | 2.17 |
| Tier A, reorder 5 | **6.47** | 2.02 | 2.76 | 1.60 |

Move −13%, Sort −26%, Collide **unchanged**.

That Collide did not move is the most useful negative result here. A1/A3/A4 helped
Move and Sort; A2 — the devirtualization, which was the most obvious-looking
optimization in the whole codebase — did nothing at all, exactly as the profile's
"4 mispredictions out of 10.8M indirect branches" predicted. Removing instructions
from a kernel that is stalled on dependent loads does not make it faster.

### A6 changes the economics of reordering, so the period had to be re-tuned

The fused counting sort makes reordering ~2.6x cheaper, which moves the optimum:

| reorder | loop (s) median | min | Move | Coll | Sort |
|--------:|----------------:|----:|-----:|-----:|-----:|
| 1 | 6.71 | 6.70 | 1.81 | **1.98** | 2.84 |
| 2 | 6.30 | 6.13 | 1.80 | 2.29 | 2.12 |
| **3** | **6.13** | **5.95** | 1.85 | 2.45 | 1.75 |
| 5 | 6.37 | 6.16 | 1.99 | 2.70 | 1.59 |
| 8 | 6.68 | 6.61 | 2.11 | 2.96 | 1.51 |

Note what happens to Collide within this build as the period shortens: **2.70 s at
period 5 → 1.98 s at period 1**. That is A5 finally engaging, because contiguity now
holds on every step and the `plist` walk disappears. The collide win came from
deleting a pointer chase, not from touching the collision math at all.

Period 1 is not the overall optimum — the sort it requires costs more than the
collide it saves — but it is what makes the mechanism visible.

> **Tier A result: 7.34 s → 6.13 s, a 1.20x speedup**, at 1M particles.
> The optimum reorder period moves from 5 to 3.

---

## 4. Tier B — changes that alter numerics, reported separately

These are **off by default**. The shipped build remains bitwise reproducible.

### B1 — faster `pow` (`-DSPARTA_VSS_FASTPOW`)

`pow` is 15% of the instruction stream, so this looked like the obvious next win.
The roofline said otherwise, and `micro/micro_pow` confirmed it — what matters for a
dependency-bound kernel is `pow`'s **latency**, not its throughput:

| implementation | throughput ns | latency ns | max rel. error |
|---|---:|---:|---:|
| glibc `pow` | 12.38 | 30.21 | — |
| `exp2(y*log2(x))` | 8.81 | **25.84** | 1.2e-15 |
| hand-rolled polynomial | 14.46 | **49.66** | 7.1e-09 |

The hand-rolled polynomial issues far fewer instructions and is **64% slower**,
because it replaces glibc's table lookup with a long serial dependency chain on
exactly the critical path (`vr2 -> pow -> vre -> compare`). It was discarded.

`exp2(y*log2(x))` is the only version that helps. End to end at 1M, reorder 3:

| build | loop (s) | Coll |
|---|---:|---:|
| Tier A | 6.33 | 2.54 |
| Tier A + `SPARTA_VSS_FASTPOW` | 6.16 | 2.36 |

Collide −7%, total −2.7%, for results that agree with `pow` to ~1e-15 but are not
bit-identical. Left off by default: a 2.7% gain does not justify giving up exact
reproducibility, but the switch is there for anyone who wants it.

### B2 — compiler flags

| flags | loop (s) median | min |
|---|---:|---:|
| `-O3` (shipped) | 6.33 | 6.32 |
| `-O3 -march=native` | 6.21 | 5.96 |
| `-O3 -march=native -ffast-math -funroll-loops` | 6.55 | 5.90 |

**At most 2%, within run-to-run noise.** This is what the roofline predicts: nothing
here is compute bound, so wider vectors and relaxed FP have nothing to bite on.

### B3 — splitting the 96-byte `OnePart` — evaluated and declined

The idea was to move `erot/evib/dtremain/weight` into a parallel array so the hot
record is exactly one 64-byte cache line, cutting Move's traffic by a third. It was
**not implemented**, on the following evidence:

- `Sort` is the most tightly bandwidth-bound kernel, and it must permute *every*
  field regardless of which array they live in, so its traffic would not change.
- `Collide` reads `erot`/`evib`, so it would go from touching one array to two,
  plausibly making it worse.
- The change touches ~280 sites across the codebase (`grep -c` on `erot`,
  `dtremain`, `weight`).

Cost is high, and the measurement says the benefit lands on the one kernel of the
three with the *least* headroom. Recorded here so the reasoning is not re-derived.

### Also tried and refuted

- **Huge pages** for the counting sort's ~100,000 open write destinations:
  `micro/micro_thp` measures `MADV_HUGEPAGE` at 15.31 vs 12.69 ns/particle — slightly
  *worse*. Dropped.
- **Folding cell ownership into the lookup table** to avoid the `cells[icell].proc`
  load in the move fast path: `micro/micro_move` shows that load costs ~0.1 ns
  (M4 8.34 vs M1 8.41 ns/particle). There was nothing to win. Dropped.

---

## 5. Verification

**Bitwise**, baseline vs Tier A, comparing every per-step stats column
(`Step Np Natt Ncoll c_temp`; the `CPU` column is wall time and is excluded):

| case | result |
|---|---|
| 100K, reorder 0 / 1 / 3 / 5 / 10 | IDENTICAL |
| 10K, reorder 5 | IDENTICAL |
| 100K with the stock `bench/in.collide` (no optmove, no reorder) | IDENTICAL |

**Regression** over in-tree examples, chosen to cover the paths the benchmark does
*not* exercise (`bench/opt/regress.sh`):

| case | what it covers | result |
|---|---|---|
| `collide/in.collide` | generic move, no optmove | IDENTICAL |
| `collide/in.collideInterspecies` | multiple groups → `collisions_group` | IDENTICAL |
| `free/in.free` | no collisions at all | IDENTICAL |
| `sphere/in.sphere` | surfaces → `move<3,1,0>`, surf collide | IDENTICAL |
| `ambi/in.ambi` | ambipolar path | IDENTICAL |
| `chem/in.chem` | chemistry, so the fused kernel must decline and fall back | IDENTICAL |

**Scaling across problem size.** The speedup holds as the working set grows from
comfortably cached to well past cache:

| size | particles | working set | baseline (reorder 5) | tuned (reorder 3) | speedup |
|---|---:|---:|---:|---:|---:|
| 100K | 0.1M | 9.6 MB | 0.440 s | 0.369 s | 1.19x |
| 1M | 1M | 96 MB | 7.34 s | 6.13 s | 1.20x |

10M was deliberately left out: a single 10M run takes ~4 minutes on one core, so a
median-of-3 comparison costs ~25 minutes, which is not a sensible turnaround for this
study. One 10M baseline run was taken before that call was made and is recorded here
only as an observation, not as a result: 103.2 s, split Move 26.5 / Coll 34.3 /
**Sort 41.6**. That `Sort` becomes the largest section once the particle array is
~1 GB is consistent with the roofline's reading of it as the most bandwidth-bound
kernel, and suggests the fused counting sort matters *more* at scale — but with no
tuned counterpart measured, that is a prediction and is flagged as one. To check it:

```
bench/opt/run_bench.sh -b bench/opt/bin/spa_tierA -s 10M -r 3 -n 3 -t tierA_scale
```

**Kokkos**: `src/KOKKOS/` is not built by `make serial` here. `CollideVSSKokkos`
derives from `CollideVSS` but overrides none of the methods touched — it uses its own
`*_kokkos` variants — and the new `collisions_one_opt` is a fresh virtual with a base
implementation returning 0, so Kokkos inherits the fall-back. `RanKnuth` is used
through its public interface only. Verified by inspection, not by build.

---

## 6. Summary

| | loop time at 1M (s) | vs baseline |
|---|---:|---:|
| shipped default (`optmove yes`, no reorder) | 14.05 | — |
| **baseline** (`optmove yes`, `reorder 5`) | **7.34** | 1.00x |
| Tier A, `reorder 5` (like for like) | 6.47 | 1.13x |
| **Tier A, `reorder 3`** (re-tuned) | **6.13** | **1.20x** |
| Tier A + `SPARTA_VSS_FASTPOW`, `reorder 3` | 6.16 * | 1.19x * |
| Tier A + `-march=native`, `reorder 3` | 6.21 * | 1.18x * |

\* measured in a later batch whose Tier A reference read 6.33 s rather than 6.13 s;
compare within a batch, not across. Relative to their own batch reference,
`SPARTA_VSS_FASTPOW` is −2.7% and `-march=native` is −1.9%.

The headline: **1.20x on the standard benchmark, entirely from bitwise-identical
changes**, with the numerics-changing options adding at most another 3%.

The three biggest contributors, in order: the fused counting sort (A6), which made
frequent reordering affordable; skipping the `plist` pointer chase (A5), which
frequent reordering then enabled; and the dense cell lookup (A1). The optimizations
that *looked* most promising from reading the code — devirtualizing the collision
kernel, and replacing `pow` — were worth 0% and 2.7% respectively.
