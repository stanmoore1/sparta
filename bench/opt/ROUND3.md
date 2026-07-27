# Round 3 — AoS vs SoA vs AoSoA, and how much of it is real

Round 2 concluded that the timestep is limited by how many bytes of particle
record it drags past the cache, and recommended shrinking `Particle::OnePart`
on the strength of a microbenchmark that measured 1.58x. This round answers the
layout question properly — and then measures, in SPARTA itself rather than in a
proxy, how much of any of it actually transfers.

**Two headline results, and they point in opposite directions:**

1. In a full rebuild of the timestep, **SoA beats AoS by 3.31x** and beats AoSoA
   by 1.75x. That is a real and large effect in the model.
2. In SPARTA itself, a direct in-situ experiment says the **elasticity of runtime
   to record size is only ~0.4** — a third more bytes costs an eighth more time.
   So the record-shrink payoff is about **1.14x, not the 1.58x round 2 predicted**.

---

## 1. The layout study

`micro/micro_layout.cpp` runs the same physics (uniform grid, reflective box, Ar,
VSS, NTC) over five layouts crossed with two binning strategies. 1M particles,
20 steps, reorder period 2. Every variant reproduces the same `ncoll` and holds
273.01 K, so the comparison is like for like.

| layout | B/particle | ns/p/s | speedup | move | bin | coll | MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| **particles permuted so cells are contiguous (what SPARTA does)** |
| AoS 96 B (SPARTA today) | 96 | 75.8 | 1.00x | 17.1 | 32.6 | 26.1 | 187 |
| AoS 64 B | 64 | 51.9 | 1.46x | 12.1 | 20.5 | 19.2 | 126 |
| **SoA (doubles)** | 52 | **22.9** | **3.31x** | 4.7 | 8.7 | 9.5 | 103 |
| AoSoA V=8 (Cabana) | 56 | 40.1 | 1.89x | 8.2 | 15.3 | 16.6 | 111 |
| AoSoA V=16 | 56 | 43.9 | 1.73x | 8.5 | 17.2 | 18.2 | 111 |
| **indices binned, particles never moved** |
| AoS 96 B | 96 | 50.4 | 1.50x | 14.1 | 14.9 | 21.4 | 95 |
| AoS 64 B | 64 | 37.0 | 2.05x | 9.5 | 10.5 | 17.0 | 65 |
| SoA | 52 | 26.4 | 2.87x | 5.5 | 3.6 | 17.3 | 53 |
| AoSoA V=8 | 56 | 30.3 | 2.50x | 5.2 | 8.8 | 16.4 | 57 |

### SoA wins, and not merely by being smaller

SoA is 52 B/particle against AoS-64's 64 B — 1.23x fewer bytes — yet it is 2.26x
faster. The extra factor is structural:

- **The mover vectorises.** Each field is its own contiguous stream, so gcc
  turns the position update, the bound tests and the cell-index arithmetic into
  AVX-512 over 8 particles at a time. Move drops 17.1 -> 4.7 ns. AoS cannot do
  this: a 96-byte stride means every lane is a separate gather.
- **Collide touches only velocity.** The pair kernel needs `v` and nothing else.
  In SoA that is three cache lines of pure velocity; in AoS the same two
  particles drag `x`, `erot`, `evib`, `dtremain` and `weight` along with them.
  Collide drops 26.1 -> 9.5 ns.
- **The permutation streams.** Reordering SoA is six independent
  sequential-read / scattered-write passes over narrow arrays rather than one
  pass moving 96-byte blobs. Bin drops 32.6 -> 8.7 ns.

### AoSoA lands in between, and V=8 beats V=16

AoSoA gets the vectorised mover (8.2 ns, about half way between SoA and AoS) but
loses on the permutation: moving one particle means writing seven individual
lanes into a destination block rather than copying one contiguous record, so bin
is 15.3 ns against SoA's 8.7. Widening to V=16 makes this worse, not better —
larger blocks mean more of each block is touched to relocate a single particle.
For a workload that reorders its particles every few steps, the Cabana layout's
strength (SIMD within a block) is partly cancelled by its weakness (lane-granular
relocation). SoA has no such penalty because "lanes" and "arrays" are the same
thing.

### Index-only binning helps AoS and hurts SoA

Skipping the permutation entirely and binning 4-byte indices instead is worth
1.50x for AoS-96, because the permutation it avoids is expensive. For SoA it is a
*loss* (22.9 -> 26.4), because the SoA permutation is cheap while the random
gathers it forces on collide are not. Which layout you choose changes which
algorithm is right.

## 2. Does any of this transfer? A direct test in SPARTA

The microbenchmark has now over-predicted twice — round 2's collide fusion
(predicted 1.26x, delivered ~1.05x) and this round's index-only binning. So
before recommending a large refactor on the strength of a third prediction, the
transfer factor was measured in SPARTA itself.

**Index-only binning, tested in SPARTA.** Running with `particle/reorder 0`
(no permutation) against `reorder 2`:

| | loop time | Move | Coll | Sort |
|---|---:|---:|---:|---:|
| reorder 2 (permuted) | **6.43 s** | 1.97 | 1.33 | 3.04 |
| reorder 0 (not permuted) | 11.70 s | 3.55 | 6.73 | 1.32 |

Not permuting is **1.8x worse**, the opposite of the microbenchmark's 1.50x gain.
SPARTA's move and collide are far more order-sensitive than the model's: the real
mover touches `cells[]` and the real collide walks `next[]` and reads per-species
tables, all of which degrade when particles are in arbitrary order. The
permutation earns its keep. **Rejected on measurement.**

**Record size, tested in SPARTA.** Rather than guess, `OnePart` was padded from
96 to 128 bytes — one line, no other change — and re-measured at 1M, reorder 2:

| record | loop time | Move | Coll | Sort |
|---|---:|---:|---:|---:|
| 96 B | 6.43 s | 1.97 | 1.33 | 3.04 |
| 128 B | 7.24 s | 2.24 | 1.38 | 3.52 |

A **1.33x increase in bytes costs 1.13x in time**. The elasticity of runtime to
record size, in the real code, is:

| kernel | elasticity |
|---|---:|
| sort | 0.48 |
| move | 0.40 |
| collide | **0.11** |
| total | **0.38** |

Collide barely notices, which is exactly what round 1's roofline said when it put
collide at 7% of the scalar compute peak: it is latency-bound on transcendentals
and dependent loads, so its bytes are nearly free.

Extrapolating the other way, to a 64-byte record:

| kernel | now | predicted at 64 B |
|---|---:|---:|
| move | 1.97 s | 1.71 s |
| sort | 3.04 s | 2.55 s |
| collide | 1.33 s | 1.28 s |
| **total** | **6.43 s** | **~5.6 s (1.14x)** |

**This corrects round 2's headline recommendation.** The 64-byte record is worth
about **1.14x in SPARTA, not 1.58x.** A ~300-site refactor of the central data
structure for 14% is a much weaker proposition than the same refactor for 58%,
and that is a judgement the SPARTA developers should make with the real number
rather than the model's.

The measurement is cheap and direct, and it generalises: **multiply any
byte-count saving in these kernels by ~0.4 to get its effect on SPARTA's
runtime.**

## 3. What this says about SoA

SoA's 3.31x cannot be deflated by the same factor, because only part of it comes
from bytes. Decomposing the model's result:

- ~1.2x of it is the smaller record (52 B vs 64 B for AoS-64), which the
  elasticity says is worth ~1.08x in SPARTA;
- the rest is vectorisation of the mover and velocity-only cache lines in
  collide, which are structural and do not scale with the byte count.

A defensible estimate for SoA in SPARTA is therefore well above 1.14x but well
below 3.31x — and the honest answer is that it cannot be pinned down without
building it, because both prior transfer estimates from this microbenchmark were
optimistic by roughly a factor of two.

What can be said firmly is the *ordering*, which is what was asked:

> **SoA > AoSoA > AoS**, decisively, for this workload — and the reason AoSoA
> does not win is specific and worth knowing: DSMC reorders its particles every
> few steps, and relocating one particle in an AoSoA block is a lane-granular
> scatter rather than a contiguous copy.

Converting SPARTA to SoA is not a bounded change: `Particle::OnePart` is passed
by pointer through the mover, every surface-collision model, every compute, every
fix, the Kokkos package and the restart format. It is a rewrite of the particle
subsystem, not a refactor of it. This round did not attempt it, and on the
evidence here nobody should attempt it without first prototyping the mover and
collide kernels against a real SPARTA problem to see how much of the 3.31x
survives contact.

## 4. Where that leaves the study

Nothing from round 3 was landed: the two ideas that were cheap enough to
implement (index-only binning, record shrink) were both measured in SPARTA first,
and the measurements said one is a 1.8x regression and the other is worth 14%
rather than 58%. Measuring first is the result.

Standing performance after rounds 1 and 2 is **7.34 s -> 6.26 s at 1M particles**
(median of 6, best reorder period for each build).

Ranked by measured value in SPARTA, what remains:

| opportunity | estimated value | cost |
|---|---|---|
| SoA particle storage | large but unquantified; 3.31x in model, prior transfers ran ~2x optimistic | rewrite of the particle subsystem |
| 64-byte record | **1.14x** (measured elasticity, not modelled) | ~300 sites |
| single-precision x, v | a further ~1.1x by the same elasticity | fidelity decision, ~300 sites |
| threading the mover and collide over cores | untested here; the box is 4 cores | orthogonal to all of the above |

The most useful thing this round produced is not an optimization but a
calibration: **a ~0.4 elasticity between bytes and runtime in the real code, and
a demonstrated ~2x optimism factor in the microbenchmark.** Both were expensive
to learn and are cheap to reuse.
