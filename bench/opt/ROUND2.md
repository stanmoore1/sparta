# Round 2 — what is the actual bottleneck, and can it be removed?

Round 1 got 1.20x and left every kernel far below both roofs. This round set out
to find out why, by rebuilding the whole DSMC timestep several different ways and
measuring, rather than by optimizing the existing structure further.

**Short answer: the bottleneck is the size of the particle record, not the
algorithm.** Every algorithmic restructuring tried here — cache tiling, fusing the
three passes, fixed-capacity cell buckets, mesh-free binning, a branchless mover —
is worth at most 1.26x in a microbenchmark and about 1.05x in the real code.
Simply shrinking `Particle::OnePart` from 96 bytes to 64, changing nothing else,
is worth **1.58x**.

---

## 1. The diagnosis: the same work costs 2.17x more once it leaves cache

Identical work per particle, swept across problem size (`spa_tierA`, reorder 3):

| grid | particles | working set | ns/particle/step | move | coll | sort |
|---|---:|---:|---:|---:|---:|---:|
| 10x10x10 | 10K | 0.96 MB | **29.3** | 10.1 | 13.6 | 5.1 |
| 15x15x15 | 34K | 3.2 MB | 37.0 | 11.3 | 16.6 | 8.5 |
| 20x20x25 | 100K | 9.6 MB | 40.0 | 10.9 | 18.0 | 10.6 |
| 30x30x30 | 270K | 26 MB | 41.8 | 11.1 | 18.2 | 11.9 |
| 40x40x40 | 640K | 61 MB | 53.0 | 15.4 | 21.1 | 15.9 |
| 40x50x50 | 1M | 96 MB | **63.5** | 19.1 | 25.0 | 18.5 |

At 10K everything is L2-resident and the step costs 29.3 ns/particle. At 1M it
costs 63.5. Nothing about the physics changed. **2.17x of the runtime is the
memory hierarchy**, and `sort` degrades worst (3.6x) because it is the most purely
bandwidth-bound of the three.

That is the number to beat, and it reframes round 1's roofline: the kernels sit
below both roofs not because they are badly written but because the step drags a
96 MB array past the cache three or four times per timestep.

## 2. The design space

`micro/micro_design.cpp` rebuilds move + bin + collide under nine designs. All
hold the equilibrium temperature; `ncoll` is the check that the physics is intact.
1M particles, 20 steps, tiles of 10x10x10 cells (938 KB, sized for L2).

| design | ns/p/s | speedup | move | bin | coll | ncoll | MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0 three passes, branchy move (SPARTA-like) | 61.2 | 1.00x | 14.5 | 28.8 | 17.9 | 1363827 | 183 |
| D1 + branchless move | 62.6 | 0.98x | 14.4 | 29.3 | 18.9 | 1363827 | 183 |
| D2 + tile-major cell numbering | 70.8 | 0.86x | 16.6 | 32.0 | 22.1 | 1364525 | 183 |
| D3 fused move+bin+collide per tile | 61.3 | 1.00x | 18.6 | 31.3 | 11.4 | **1277641** | 206 |
| D4 D3 + mesh-free binning | 59.2 | 1.03x | 17.8 | 31.3 | 10.2 | **1277641** | 206 |
| D5 fixed-capacity buckets (cap 14/17/20) | 61–76 | 0.81–1.00x | **39–49** | 0.1–0.8 | 22–27 | **wrong** | 128–183 |
| D8 collide fused into the scatter | 48.7 | 1.26x | 13.4 | 35.3 | 0.0 | 1363948 | 183 |
| **D6 64-byte record** | **42.5** | **1.44x** | 11.0 | 18.4 | 13.1 | 1363827 | 122 |
| D9 D8 + 64-byte record | 36.6 | 1.67x | 9.8 | 26.9 | 0.0 | 1363948 | 122 |
| **D7 40-byte record (float x, v)** | **31.0** | **1.97x** | 8.9 | 10.8 | 11.3 | 1363744 | 92 |

### What each result says

**D6/D7 — shrink the record. This is the answer.** D6 keeps the identical
algorithm and identical `ncoll`; it only drops the four fields that are dead in a
monatomic, unweighted, surface-free run (`erot`, `evib`, `dtremain`, `weight`).
Its per-kernel gains are 1.32x / 1.57x / 1.37x — all three kernels improve
together, in proportion to 96/64, because this is pure bandwidth. D7 additionally
puts positions and velocities in single precision (40 B) for 1.97x, with `ncoll`
differing by 0.006%.

**D5 — fixed-capacity buckets are actively harmful.** Giving each cell a bucket
with slack lets move write particles straight into their destination, collapsing
move and bin into one pass — `bin` really does drop to ~0.1 ns. But `move` triples,
because iterating buckets streams `cap` slots per cell to use `npercell` of them.
When bandwidth is the binding constraint, inflating the footprint to save a pass
is a bad trade. (The temperature also drifts at low capacity: overflow is common
enough that particles were being dropped, which is its own warning.)

**D3/D4 — tiled fusion changes the physics.** Fusing move, bin and collide per
tile makes collide 1.8x faster by keeping the tile in L2, but `ncoll` falls from
1363827 to 1277641 — **6.3% low**. Particles that cross a tile boundary miss a
step of collisions. Temperature is unaffected (elastic collisions conserve energy
whatever the pairing) which is exactly why temperature alone is not a sufficient
check. Rejected on the collision rate.

**D4 — mesh-free binning is slower, not faster.** Dropping the stored cell index
and recomputing it from position costs more than the 4 bytes it saves. There is no
win in eliminating the mesh here: the uniform grid is already just arithmetic, and
DSMC's NTC method fundamentally needs particles grouped by volume, so a genuinely
mesh-free collision partner search (neighbor lists, k-d trees) would replace an
O(1) bin with something more expensive, not less.

**D1 — a branchless mover does not help.** Round 1's profile showed the ~1% of
particles taking the slow path own 40% of branch mispredictions, so removing the
slow path looked attractive. But testing three reflections unconditionally for
every particle costs more than one almost-always-true in-box test, and the branch
predictor handles a 99%-biased branch essentially for free.

**D8 — collide the cell the instant the scatter finishes writing it.** A counting
sort cannot place a particle before counting them all, so move-then-count-then-
scatter is irreducible. But the *third* pass is avoidable as a separate traversal:
a cell is complete the moment its write cursor reaches `first+count`, and at that
instant its particles are in L1. This is physics-preserving (`ncoll` 1363948 vs
1363827, 0.009%) and measured 1.26x in the microbenchmark.

## 3. What was implemented, and what it was actually worth

Two changes were landed:

- **Collide fused into the sort scatter** (D8). `Particle::sort_reorder()` takes an
  optional `Collide *` and calls `collide_one_cell()` as each cell completes;
  `CollideVSS` exposes a per-cell entry point sharing one templated kernel with the
  whole-step path. Falls back whenever the style cannot be driven per cell
  (chemistry, ambipolar, near-neighbour, gas tallies, multiple groups, Poisson
  attempts) or custom per-particle data exists.
- **Flat `vremax`/`remain` views.** These are `double***`, so `vremax[icell][0][0]`
  is two dependent pointer loads before the value. Walking cells in order those
  prefetch; the fused path visits cells in *completion* order, which turned them
  into two cache misses per cell. With one group the data block is contiguous, so
  `vremax1[icell]` is a single load. This helps both paths.

Head to head at 1M, 6 repetitions each, best setting for each:

| build | reorder | loop time median | min |
|---|---:|---:|---:|
| round 1 (`spa_tierA`) | 3 | 6.76 s | 6.06 s |
| round 2 (`spa_r2b`) | 2 | **6.26 s** | 5.99 s |

**About 5% on the median and nothing on the minimum — at the edge of this
machine's noise.** The microbenchmark predicted 1.26x and delivered ~1.05x.

### Why the microbenchmark over-predicted, which is worth recording

`micro_design`'s collide kernel is lighter than SPARTA's: no per-species table
lookups, no `EEXCHANGE` branch, no group machinery, flat per-cell arrays. Memory
was therefore a larger fraction of *its* collide time than of SPARTA's, where
collide is dominated by `pow`/`sqrt`/`sincos` and dependent loads — exactly what
round 1's roofline said when it put collide at 7% of the scalar compute peak.
Making already-compute-bound work L1-resident does not speed it up.

**The lesson: a microbenchmark that simplifies the kernel under test will overstate
the benefit of memory optimizations to that kernel.** The D6/D7 results do not
suffer from this, because a uniform 96/64 traffic reduction across all three
kernels is a mechanism that does not depend on the collide kernel's internals —
and indeed D6's three kernels all improved by the predicted ratio.

## 4. Verification

- `examples/{collide,collideInterspecies,free,sphere,ambi,chem}`: **IDENTICAL** to
  the pre-round-1 baseline. None use reordering, so none take the fused path.
- 100K, `reorder 0` (fusion inactive): **bitwise identical**.
- With fusion active (`reorder 3`), results are deliberately **not** bitwise
  identical, because cells complete in scatter order rather than cell order and so
  consume random numbers in a different sequence. Physics checks:
  equilibrium temperature **272.86255 K in both**, identical to every printed
  digit; over the sampled steps mean `nattempt` differs by 0.12% and mean `ncoll`
  by 0.33%.

## 5. The recommendation for the next round

**Shrink `Particle::OnePart`.** It is 96 bytes, of which 32 (`erot`, `evib`,
`dtremain`, `weight`) are dead for a monatomic, unweighted, surface-free run — and
every one of the three passes pays to stream them. Measured worth: **1.58x**, more
than everything in rounds 1 and 2 combined.

Note this is *not* the hot/cold split declined in round 1, and round 1's reasoning
for declining that was right: splitting into two arrays that both still have to be
permuted leaves sort's traffic unchanged. The win here comes from those fields
**not existing** for runs that do not need them, so there is nothing to permute.
The natural mechanism is the one SPARTA already has for optional per-particle data:
`ncustom` side arrays, allocated only when some species has internal degrees of
freedom or grid weighting is on.

It was not attempted here because it touches roughly 300 sites across the codebase
(`erot` 176, `dtremain` 105, `weight` 43) and is a change to the central data
structure — not something to land at the end of a session on the strength of a
microbenchmark, however clear. The measurement is recorded so the next round can
start from it rather than re-deriving it.

Second, smaller: single-precision positions and velocities (D7, 1.97x total) are
worth evaluating on physics grounds. Float resolves position to ~4e-6 of a cell
width and `ncoll` moved by 0.006%, but that is a fidelity decision for the SPARTA
developers, not a performance one.
