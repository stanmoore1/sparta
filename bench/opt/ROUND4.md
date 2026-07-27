# Round 4 — a benchmark that tells the truth

Two follow-up questions from round 3, and then the fix for why the benchmarks
kept lying.

---

## 1. "AoSoA loses on the reorder — what if you turn reordering off?"

Fair challenge, and the answer is: it helps AoSoA more than anything else, and it
still is not enough. Sweeping the reorder period per layout (1M particles):

| layout | every 1 | every 2 | every 4 | every 8 | every 16 | never | best |
|---|---:|---:|---:|---:|---:|---:|---:|
| AoS 96 B | 81.3 | 70.2 | 61.0 | 58.5 | 60.4 | **47.9** | 1.59x |
| AoS 64 B | 54.6 | 53.3 | 49.5 | 43.0 | 42.6 | **31.2** | 2.45x |
| **SoA** | 29.3 | 27.0 | 24.3 | 23.4 | 23.7 | **21.1** | **3.62x** |
| AoSoA V=8 | 37.6 | 39.7 | 35.7 | 33.2 | 32.6 | **32.0** | 2.38x |

AoSoA improves from 40.1 to 32.0 once reordering stops — the largest relative
gain of any layout, exactly as the diagnosis predicted. But SoA with reordering
*off* is 21.1, so SoA still wins by 1.5x, and SoA with reordering *on* (23.4)
also still beats the best AoSoA. Removing AoSoA's handicap does not make it
competitive; it just stops it being penalised twice.

**And the premise does not survive contact with SPARTA.** Turning reordering off
is a **1.8x regression** in the real code (11.70 s against 6.43 s), because
SPARTA's mover and collide degrade far more with disordered particles than any of
these models do. So "turn off reordering to help AoSoA" is not available as a
strategy — the reordering is load-bearing.

## 2. "Make the grid cells SoA too"

Also tested directly in SPARTA, using the padding method from round 3 rather than
a model. Pad the cell structures, re-time, read off the sensitivity:

| variant | loop time at 1M, reorder 2 |
|---|---:|
| baseline (contemporaneous) | 6.02 s |
| `ChildInfo` 64 -> 128 B | 6.28 s |
| `ChildCell` 128 -> 192 B | 5.94 s |

**Zero measurable elasticity** — the padded builds land on both sides of the
baseline, inside a 6-8% run-to-run spread. Doubling the per-cell data costs
nothing, so packing it more tightly would gain nothing.

The reason is arithmetic rather than subtle: there are ten particles per cell, so
cell data is an order of magnitude smaller than particle data. `cinfo` is 6.4 MB
against the particle array's 96 MB, and it is streamed once per step; `cells[]` is
read per particle in the mover, but consecutive particles in a sorted list hit the
same cell, so it is served from cache. Neither is a bottleneck. **SoA grid cells
are not worth doing.**

## 3. Why the benchmarks kept lying, and the fix

The `micro_*` benchmarks over-predicted three times running, always in the same
direction:

| prediction | what SPARTA did |
|---|---|
| collide fusion 1.26x | ~1.05x |
| 64-byte record 1.58x | ~1.14x |
| index-only binning 1.50x gain | 1.8x **regression** |

`micro/mini_dsmc.cpp` replaces them with a mini-app rather than a
microbenchmark. It carries SPARTA's real structures at their real sizes —
`OnePart` at 96 B, `ChildCell` at 128 B, `ChildInfo` at 64 B, `vremax` as a
`double***` so the two dependent pointer loads per cell are present, `params` and
`prefactor` through their real double indirection, the `next[]` linked list, the
real `RanKnuth` — and its `move` and `collide` are transcriptions of
`Update::move<3,0,1>` and `CollideVSS::collide_cell_kernel`, including the pflag
dispatch, the `cells[icell].proc` ownership test, and a genuine cell-by-cell
traversal slow path that walks `cells[].lo/hi/neigh`.

**It also validates itself.** `./mini_dsmc -validate` prints the reorder-period
curve, whose shape can be checked against SPARTA's:

| | SPARTA at 1M | mini_dsmc |
|---|---|---|
| equilibrium temperature | 272.86 K | 273.01 K |
| best reorder period | 2–3 | 3 |
| cost at best | ~60 ns/particle/step | 63.6 |
| **penalty for never reordering** | **1.82x** | **1.73x** |

Building it caught two fidelity bugs immediately, and both are instructive:

- **A factor of sqrt(2) in the Maxwellian** put the gas at 136 K instead of 273 K,
  which would have depressed relative velocities and the collision rate. The
  temperature check caught it on the first run. None of the earlier
  microbenchmarks checked temperature against a reference value.
- **No equilibration phase.** `bench/in.collide` runs 30 steps at a 10x timestep
  before measuring, and `bench/README` says exactly why: *"The equilibration
  insures particles are not ordered in memory by grid cell, which can run faster
  (initially) until particles become disordered."* Starting from a perfectly
  sorted array made the penalty for never reordering 1.07x instead of 1.73x —
  which is precisely the error that produced round 3's wrong index-only-binning
  prediction. The benchmark was measuring a state the real run is never in.

### Does the mini-app actually predict better?

The one prediction with an independent in-situ measurement to check against is the
record size, so that is the test:

| source | 96 -> 64 B | elasticity |
|---|---:|---:|
| old `micro_design` | 1.58x | — |
| **new `mini_dsmc`** | **1.18x** | 0.46 |
| **SPARTA, measured by padding** | **1.15x** | 0.38 |

The model went from 38% wrong to 3% wrong on the question that matters. It is
still slightly optimistic, and in the upward direction (96 -> 128 B) it gives
elasticity 0.72 against SPARTA's 0.38, so it should not be treated as exact — but
it is now close enough to use for ranking options, which the `micro_*` benchmarks
demonstrably were not.

## 4. Standing position

Performance is unchanged from round 2: **7.34 s -> 6.26 s at 1M particles**.
Rounds 3 and 4 landed no optimizations, because everything cheap enough to try was
measured first and the measurements said no.

What is now known, with the evidence for each:

| question | answer | evidence |
|---|---|---|
| AoS, SoA or AoSoA? | **SoA**, by 3.62x over AoS-96 and 1.5x over AoSoA | full-timestep rebuild, all reorder periods |
| Why does AoSoA lose? | lane-granular relocation on reorder; V=16 worse than V=8 | reorder sweep |
| Does turning reordering off rescue AoSoA? | no, and reordering cannot be turned off anyway | sweep + 1.8x regression in SPARTA |
| SoA grid cells? | **no** — zero measured elasticity | padding `ChildInfo` and `ChildCell` in SPARTA |
| Shrink `OnePart` to 64 B? | **~1.15x**, not the 1.58x first claimed | padding in SPARTA, confirmed by the mini-app at 1.18x |
| Mesh-free DSMC? | slower; NTC needs volume binning and the grid is already arithmetic | round 2 |

The remaining large opportunity is still SoA particle storage, and it is still
unquantified in SPARTA — but it can now be estimated properly, because
`mini_dsmc` is faithful enough to be extended with an SoA storage backend and
believed. That is the obvious next step, and it is a far smaller job than
converting SPARTA itself.
