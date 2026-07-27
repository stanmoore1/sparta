# Round 5 — the kitchen sink, on a benchmark that no longer lies

`micro/mini_store.cpp` takes the validated mini-app from round 4 and templates it
on the particle storage, so every configuration runs **SPARTA's control flow** —
the pflag dispatch, the `cells[icell].proc` test, the cell-by-cell traversal slow
path, the `double***` vremax indirection, the `next[]` linked list, the real
RanKnuth and VSS math — and only the layout differs.

1M particles, 20 steps after a 30-step equilibration, reorder period 3.
`ncoll` is 1413959 for every double-precision row, so the physics is fixed and
the differences are layout alone.

| configuration | B/p | ns/p/s | speedup | move | sort | coll |
|---|---:|---:|---:|---:|---:|---:|
| AoS 96 B (SPARTA today) | 96 | 67.8 | 1.00x | 21.0 | 19.4 | 27.4 |
| AoS 64 B | 64 | 66.3 | 1.02x | 24.8 | 15.1 | 26.4 |
| **SoA doubles** | 56 | **34.9** | **1.94x** | 11.8 | 9.3 | 13.9 |
| **SoA floats** | 32 | **28.7** | **2.36x** | 9.7 | 6.5 | 12.5 |
| AoSoA V=8 | 56 | 41.3 | 1.64x | 13.3 | 11.7 | 16.4 |
| AoSoA V=16 | 56 | 47.1 | 1.44x | 15.8 | 13.0 | 18.3 |
| SoA + blocked SIMD move | 56 | 36.8 | 1.84x | 12.5 | 10.0 | 14.3 |
| SoA floats + blocked move | 32 | 31.4 | 2.16x | 11.1 | 7.4 | 12.9 |
| AoS + fused collide | 96 | 60.2 | 1.13x | 19.3 | 22.9 | 18.1 |

## What changed once the benchmark became faithful

**SoA fell from 3.6x to ~1.9x, and that is the number to plan with.** The earlier
`micro_layout` figure came from a model whose mover had no pflag dispatch, no
ownership test and no slow path, so it vectorised freely and made SoA look twice
as good as it is. With SPARTA's control flow the mover cannot vectorise, and SoA's
remaining advantage is what it should be: narrower streams and velocity-only
cache lines in collide.

**AoSoA closed most of the gap but still loses**, now by 1.2x rather than 1.5x.
V=16 remains worse than V=8, for the reason identified in round 3: relocating one
particle during the reorder is a lane-granular scatter, and wider blocks make that
worse.

**The 64-byte record collapsed to 1.02x here.** That disagrees with the same
model's direct-field-access version (`mini_dsmc`, 1.18x) and with SPARTA measured
in situ (1.15x) — and the direct-access numbers are the trustworthy ones. See the
caveat below.

## Two negative results worth having

**Restructuring the mover for SIMD does not pay.** The blocked fast path processes
eight particles with no per-particle early exit, which is the restructuring SoA
is supposed to enable. It made SoA *slower* (34.9 -> 36.8). At this timestep about
1% of particles need the slow path, so roughly 8% of eight-wide blocks contain an
exception and must be abandoned and redone scalar — and the block does redundant
work on the way to finding out. SPARTA's exception handling defeats blocking, and
that is a property of the physics (particles do leave cells and boxes), not of the
code.

**Collide fusion helps AoS and hurts SoA.** AoS gains 1.13x, consistent with the
~1.05x it delivered in SPARTA. The SoA rows lose, but that is an artifact of this
implementation rather than a finding: fusing requires the collide kernel to read
the *destination* buffer, which for SoA means swapping sixteen pointers twice per
completed cell, 100000 cells per step. A real implementation would pass the
destination base pointers instead. The SoA + fused rows should be ignored.

## Caveat on the AoS rows, stated plainly

To share one timestep across five layouts, this benchmark reaches particle data
through accessors (`st.xg(i,c)`) rather than direct struct members. That is
natural for SoA and costs AoS something: `mini_store`'s AoS-96 runs at 67.8
ns/p/s against `mini_dsmc`'s hand-written AoS-96 at 63.4, so the abstraction costs
AoS about 7%, and it evidently costs AoS-64 considerably more (66.3 here against
53.7 in `mini_dsmc`).

Correcting for that, the honest SoA advantage over a natively written AoS is
**63.4 / 34.9 = 1.82x**, and the AoS-64 question should be answered from
`mini_dsmc` and the in-situ SPARTA measurement (1.18x and 1.15x), not from this
table. Writing each layout's mover natively would remove the caveat; it was not
done here.

## What this means for SPARTA

Best estimate for converting particle storage to SoA: **~1.8x**, with another
~1.2x available from single precision on top. Both are far larger than anything
rounds 1 to 4 landed, and both are much smaller than the 3.6x the unfaithful
benchmark promised.

On implementation: templating the hot kernels on a storage policy is the right
shape, and `if constexpr` makes it tractable — `Update::move`, `CollideVSS`'s
per-cell kernel and `Particle::sort_reorder` are already templated or easily
templated, and each would take a storage tag with the AoS path preserved for
everything else in the codebase. The obstacle is not those three kernels; it is
that `Particle::OnePart *` is passed by pointer into every surface-collision
model, every compute and fix, the Kokkos package and the restart format. A
storage policy confined to the hot loops would need a materialisation boundary
where the rest of SPARTA still sees an `OnePart`, and whether that boundary can
be drawn cheaply is the question worth prototyping next — in `mini_store`, where
it costs an afternoon rather than a release cycle.

## Standing position after five rounds

Performance in SPARTA: **7.34 s -> 6.26 s at 1M particles** (1.17x), from rounds
1 and 2. Rounds 3, 4 and 5 landed no code, because everything cheap enough to try
was measured first and the measurements said no.

| question | answer | source |
|---|---|---|
| AoS, SoA or AoSoA? | **SoA**, ~1.8x over AoS; AoSoA ~1.6x | faithful model |
| Turn reordering off to help AoSoA? | helps it most, still loses; and impossible in SPARTA (1.8x regression) | model + SPARTA |
| SoA grid cells? | no — zero measured elasticity | SPARTA, by padding |
| 64-byte particle record? | ~1.15x | SPARTA, by padding |
| Single precision x, v? | ~1.2x on top of SoA | faithful model |
| SIMD mover? | no — the slow path defeats blocking | faithful model |
| Mesh-free DSMC? | no — NTC needs volume binning | round 2 |
| Cache tiling, fused passes, cell buckets? | no | round 2 |
