# Round 10 — five algorithmic restructurings; four fail

Round 9 exhausted the implementation-level ideas. What is left in the timestep
is two streaming passes over the particle array — the move (read and write) and
the counting sort's scatter (read and write) — plus the collision arithmetic
fused into the second one. Roughly 384 MB per step at 1M particles.

So this round asked a different question: **what would have to change about the
algorithm** for that to be less than two passes. Five candidates, four of them
prototyped head to head in `micro/mini_algo.cpp`, which reuses `mini_dsmc`'s
validated physics and structures verbatim and changes only how the timestep is
organised.

## The five

| # | idea | mechanism | verdict |
|---|---|---|---|
| 1 | **defer** the displacement | move records only the destination cell; the scatter applies `x += v*dt` as it copies, so the move never writes the array the sort is about to rewrite | model 1.19x; **SPARTA: a wash at reorder 1, +8% at `collide/every 8`** |
| 2 | **slot** — abolish the sort | every cell owns a fixed-capacity region; the move writes each particle straight into its destination cell. One read, one write, no sort at all | **0.91x–0.63x — fails** |
| 3 | **batch** the collisions | split pair selection from VSS scattering so the `pow`/`sqrt`/`sin`/`cos` run over flat arrays instead of behind two pointer chases | **0.87x — fails** |
| 4 | **coarse** grid + nearest-neighbour partners | 8x fewer, larger cells; Bird's NN selection is what makes them admissible | **0.41x — fails badly** |
| 5 | **deviational** (variance-reduced) DSMC | simulate only the deviation from equilibrium | quantified, not built — see below |

Model results, 1M particles, 20 steps, reorder 1. The baseline row is **fused**,
not **base**, because SPARTA has fused the collide into the scatter since round 2
and reorder 1 always takes that path — scoring against the unfused row would
have credited every alternative with a win SPARTA already has:

| mode | cap | ns/p/s | vs fused | move | sort | coll | T (K) | coll/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base (unfused) | | 72.15 | — | 19.24 | 30.66 | 22.26 | 273.01 | 70689 |
| **fused** (SPARTA today) | | 63.64 | 1.00x | 18.97 | 44.68 | 0 | 273.01 | 70681 |
| **defer** | | **53.50** | **1.19x** | 17.46 | 36.05 | 0 | 273.01 | 70681 |
| batch | | 72.79 | 0.87x | 18.67 | 30.23 | 23.90 | 273.87 | 70673 |
| slot | 16 | 70.27 | 0.91x | 70.27 | 0 | 0 | 273.01 | **68640** |
| slot | 24 | 88.44 | 0.72x | 88.44 | 0 | 0 | 273.01 | 70681 |
| slot | 32 | 100.27 | 0.63x | 100.27 | 0 | 0 | 273.01 | 70689 |

## 2. Why abolishing the sort fails

This was the idea with the best arithmetic behind it. If each cell owns a
fixed-capacity slot region, the move can write each particle directly into its
destination cell in a second slot array, and the sort disappears entirely: one
read, one write, 192 MB instead of 384.

It loses anyway, and the table shows why in one column: **capacity is the whole
story**. At capacity 32 the array is 3.2x larger than the data in it, and the
gaps are read straight through — 0.63x. At 24 it is 0.72x. At 16 it is 0.91x,
still a loss, *and* the collision rate falls 2.9% (70689 to 68640) because
overflowing particles land in a spill list and stop being collision candidates.

There is no capacity that works, and the reason is structural rather than
tunable: mean occupancy is 10 with Poisson spread, so a capacity that overflows
rarely enough to preserve the statistics is one that wastes more bandwidth in
gaps than the sort it replaced ever cost. The counting sort is not overhead —
it is what buys the perfectly packed array that every other pass then streams.

This confirms round 2's rejection of fixed-capacity buckets, which had been made
on `micro_design`, the benchmark that later turned out to over-predict three
times. It is worth having the same answer from a model that is trusted.

## 3. Why batching the collisions fails

The VSS scattering kernel is where the transcendentals live, and it sits behind
two pointer dereferences and an acceptance branch — the classic case for
gathering accepted pairs into flat arrays and running the arithmetic over them.
Selection has to stay sequential (it consumes the random stream and updates the
cell's running `vremax`), so the split is: select, gather, apply.

**0.87x.** The gather and the write-back scatter cost more than the vectorisation
returns. The acceptance rate is high enough (about 75% here) that the branch was
never the problem, and the two velocity vectors the kernel needs are already in
L1 when the collide is fused into the scatter — which it is. Flattening data
that is already in L1 buys nothing and costs two extra passes over the batch.

## 4. Why the coarse grid fails, decisively

Bird's nearest-neighbour partner selection permits cells several mean free paths
across. Fewer cells means less grid metadata, longer contiguous runs, and fewer
cell crossings per step. The bounding experiment — same 1M particles,
progressively fewer and larger cells, baseline organisation, NN *not*
implemented so only the cost is meaningful:

| cells | particles/cell | ns/p/s | vs 100K cells | move | sort | coll |
|---:|---:|---:|---:|---:|---:|---:|
| 100000 | 10 | 66.80 | 1.00x | 17.41 | 28.40 | 20.99 |
| 12500 | 80 | 162.22 | **0.41x** | 19.92 | 30.72 | **111.57** |
| 1690 | 592 | 643.17 | **0.10x** | 19.45 | 29.44 | **594.27** |

Move and sort are flat, as predicted. **Collide explodes — 5x, then 28x.** NTC's
`vremax` is a running maximum of `vr*sigma` over the pairs it samples, so a
larger cell population drives it toward the true maximum of the distribution,
which raises the attempt count and lowers the acceptance rate in step. The
attempt count is superlinear in particles per cell in a way the volume scaling
does not cancel.

This is the quantitative version of the standard advice to keep about 20
particles per cell, and it closes the direction: NN selection would fix the
*accuracy* of a coarse grid but would add cost on top of a collide term that is
already 5x worse. **Coarsening the grid is the wrong direction for NTC**, and
`in.collide`'s 10 particles per cell is already near the right operating point.

## 1. Deferring the displacement — the one that survived, partly

The move reads `x` and `v`, computes the new position, and writes it back. The
scatter then reads that record and copies it somewhere else. The write is
redundant: the scatter is going to write a whole new record anyway, and it can
apply `x += v*dt` for free while doing so. So the move records only the
destination cell into a compact `int` array and leaves the 96-byte record
untouched.

The saving is larger than the write itself, because a store to a line that is
not resident costs a read-for-ownership *and* a later writeback. Skipping it
turns a read-modify-writeback of the particle array into a plain read.

**In SPARTA it is bitwise identical** — verified on `in.collide` at 1M, on
`bench/in.sphere`, and on all six `regress.sh` cases against the unmodified
baseline binary. It is gated to the cases where nothing between the move and the
sort can observe a stale position: one proc, the `optmove` fast path, no custom
attributes, no cell weighting, and only on steps that reorder.

The performance is configuration-dependent and **I cannot fully account for it**:

| configuration | runs | result |
|---|---|---|
| reorder 1, `collide/every` 1 | 1.046x, 0.972x, 1.022x | **a wash** |
| reorder 4, `collide/every` 8 | 1.071x, 1.121x, 1.083x | **+8%** |

The Move timer moves the right way in both (1.538 -> 1.417 s and 1.828 -> 1.628 s),
which is the mechanism doing what it should. At reorder 1 the scatter gives it
all back (3.128 -> 3.376 s), because there the scatter runs every step and the
extra branch and arithmetic are paid 100 times instead of 13. What I cannot
explain is why the scatter at `collide/every 8` measures *faster* (1.840 -> 1.757 s)
when it is strictly doing more work — that should not happen, and a control run
with reordering disabled entirely (where the deferral never activates) showed
1.010x, so it is not simply code layout.

**It is committed** because it is bitwise-verified, tightly gated, and helps the
configuration that is actually fastest. But the +8% should be treated as
provisional until the scatter anomaly is understood, and that is stated here
rather than papered over.

One implementation note worth recording, because it was worth 17% on its own:
writing the scatter as "copy the record, then overwrite the three fields" was a
**10% regression**; writing it as "load into a local, modify, store once" was a
4.6% gain. Identical semantics, identical instruction count to within a few, and
a 17-point swing — the first form creates a store-then-reload dependency on
lines that were just written.

## 5. Deviational DSMC — quantified, not built

The idea that would matter most is not a faster timestep but fewer particles.
Deviational (low-variance) DSMC represents only the departure from a reference
Maxwellian, so the statistical noise scales with the *deviation* rather than with
the absolute distribution. For a flow with signal `Ma`, standard DSMC needs
`N ~ 1/Ma^2` particles for a given signal-to-noise, deviational methods need
`N ~ 1/Ma`.

`in.collide` is the limiting case: it is an equilibrium box, the deviation is
zero, and a deviational code would need essentially no particles to reproduce
it. `in.sphere` at Mach ~7 is the opposite — the deviation is the whole flow and
there is nothing to gain.

That is the honest summary of its scope: **it is transformative for low-signal
problems and useless for strong ones**, it changes the particle representation
rather than the timestep, and it is a different code rather than a change to this
one. It is listed for completeness, not as a recommendation for SPARTA.

## Standing position after ten rounds

`in.collide`, 1M particles, FP64 throughout:

| | loop (s) | speedup |
|---|---:|---:|
| original SPARTA, `optmove` + reorder 2 | 7.944 | 1.00x |
| rounds 1-9, reorder 1 | 4.755 | 1.67x |
| + `collide/every 8`, reorder 4 | 3.494 | 2.27x |
| + round 10 deferred displacement | ~3.23 | **~2.46x** |

`in.sphere`: 0.3128 s -> 0.2767 s, **1.13x**, bitwise identical.

What this round settled, and it is mostly negative, which is the point of
running it: the counting sort is not overhead to be eliminated but the thing
that makes every other pass cheap; NTC's cost is superlinear in particles per
cell, so coarser grids are the wrong direction; and the collision kernel's data
is already in L1, so flattening it for SIMD buys nothing.

The two large items are unchanged and neither is a timestep restructuring:
**SoA particle storage** (~1.9x, rounds 5-7) and **an 80-byte FP64 record**
(~1.17x, round 9).
