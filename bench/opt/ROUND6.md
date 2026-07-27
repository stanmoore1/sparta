# Round 6 — the materialization boundary, and whether vectorization is available

Two questions, plus a correction to round 5.

---

## 0. Correction: round 5's SIMD result was wrong

Round 5 reported that a blocked SIMD mover makes SoA slower and concluded that
SPARTA's exception handling defeats vectorization. **That measurement was
invalid.** The blocked loop used `break` on the first block containing an
exception, which exits the blocked loop *permanently* rather than falling back
for that block alone. At a ~1% exception rate the first such block appears within
the first hundred particles, so essentially the entire 1M-particle array went
down the scalar path. Round 5 measured scalar code with extra overhead.

With the fallback corrected to be per-block, the conclusion reverses.

## 1. Can vectorization be improved?

**Not with AoS. Yes with SoA.** Same blocked mover, both layouts:

| configuration | ns/p/s | speedup | move |
|---|---:|---:|---:|
| AoS 96 B (SPARTA today) | 65.3 | 1.00x | 20.3 |
| AoS 96 B + blocked move | 84.8 | **0.77x** | 20.5 |
| SoA doubles | 38.4 | 1.70x | 14.4 |
| SoA doubles + blocked move | **34.6** | **1.89x** | **9.9** |
| SoA floats + blocked move | 29.9 | 2.18x | 9.4 |
| AoSoA V=8 + blocked move | 49.3 | 1.32x | 12.1 |

Blocking the mover cuts SoA's move time by **1.46x** (14.4 -> 9.9) and is worth
1.70x -> 1.89x overall. On AoS it is a **23% regression**: gathering eight
particles whose fields are 96 bytes apart costs more than the vectorization
returns, and the block's speculative work is wasted.

So the answer to "can vectorization be improved in SPARTA" is that it is not
really available today. The mover's arithmetic is vectorizable in principle — the
position update, the bound tests and the cell-index computation are all
straight-line — but with a 96-byte stride there is nothing to vectorize *over*.
The layout is the gate, not the control flow. That is the opposite of what round
5 concluded, and it matters, because it means the SoA case is stronger than
stated: SoA buys narrower streams *and* unlocks a mover that AoS cannot have.

The exception rate does bound how much is available. About 1% of particles need
the slow path, so roughly 8% of eight-wide blocks contain one and fall back
scalar. That caps the blocked mover's benefit but does not eliminate it, because
the fallback is now per-block rather than per-array.

## 2. The materialization boundary: affordable

If particle storage goes SoA, the hot kernels get rewritten natively — but the
rest of SPARTA passes `Particle::OnePart*` by pointer into every
surface-collision model, every compute and fix, the Kokkos package and the
restart format. Those callers cannot all be converted, so SoA needs a boundary
that gathers a particle into a real `OnePart`, hands it over, and scatters the
result back.

`mini_store` now prototypes exactly that. `boundary_collide()` is an out-of-line
callee taking a `OnePartView*` — the shape of `Domain::collide` and the
`SurfCollide` models — and the mover's slow path materializes, calls, and writes
back on every boundary interaction.

| configuration | ns/p/s | vs same layout without boundary |
|---|---:|---:|
| SoA doubles | 38.4 | — |
| SoA doubles + materialization boundary | 41.6 | +8.4% |
| SoA doubles + blocked move | 34.6 | — |
| SoA doubles + blocked + boundary | **33.1** | **-4%** (i.e. within noise) |
| SoA floats + boundary | 38.5 | +19% |
| AoS 96 B + boundary | 65.9 | +1.0% |

The boundary costs SoA **somewhere between nothing and 8%**. The blocked+boundary
row measuring *faster* than blocked alone is not physical — it bounds the
run-to-run noise on this machine at about 5% — so the honest reading is that at a
1% crossing rate the boundary is at or below the noise floor, and the best SoA
configuration lands at **1.97x over AoS-96**.

It is nearly free for AoS (+1%), as expected: materializing from AoS is a struct
copy.

**Why it is cheap: the boundary is crossed rarely.** It is paid on the mover's
slow path — about 1% of particles per step, which is exactly where
`Domain::collide` and the surf models are called — and on whatever computes and
dumps touch at output steps. The hot paths (the in-box mover, the NTC pair
kernel, the counting sort) never cross it, because those get native SoA versions.

The worst case is a compute or dump that wants every particle; `materialize_all()`
is in the prototype to measure it, and at 1M particles that is one extra gather
pass over the array, comparable to a single kernel pass and paid only on output
steps.

## 3. What this means for SPARTA

Best estimate for SoA particle storage, now including the boundary it would
actually need:

| | speedup over AoS-96 |
|---|---:|
| SoA, kernels rewritten natively | 1.70x |
| SoA + blocked mover | 1.89x |
| SoA + blocked mover + materialization boundary | **1.97x** |
| the same with single-precision x and v | 2.2x |

For comparison, rounds 1 and 2 landed 1.17x in total.

On implementation shape, given that templating and `if constexpr` are acceptable:
`Update::move` is already templated on `<DIM,SURF,OPT>` and takes a storage tag
naturally; `CollideVSS::collide_cell_kernel` is already templated on `<CONTIG>`;
`Particle::sort_reorder` is a single function. Those three are the hot paths and
the whole of the 1.97x. Everything else keeps taking `OnePart*` across the
materialization boundary, which this prototype says is affordable.

The remaining unknown is not performance but surface area: how many call sites
sit on the boundary in practice, and whether any of them are hot enough to matter
(the surf-collision models would be, for a problem with surfaces — this benchmark
has none, so this prototype does not exercise that case and should not be read as
covering it).

## Standing position

SPARTA performance is unchanged at **7.34 s -> 6.26 s** (1.17x) from rounds 1-2.
No code landed in rounds 3-6; the work was establishing what is worth doing, and
correcting two of my own wrong answers along the way (round 2's 1.58x record
claim, corrected to 1.15x by in-situ measurement in round 3; and round 5's
vectorization claim, corrected here).
