# Round 7 — mini-app for `bench/in.sphere`, and the surface question settled

Every mini-app so far modelled `in.collide`, which has no surfaces. That left the
one question the SoA study could not answer: what the materialization boundary
costs when the `SurfCollide` models are in play, since those take
`Particle::OnePart*` and would sit on the boundary.

`micro/mini_sphere.cpp` closes that gap.

## `in.sphere` is a different machine from `in.collide`

Measured in SPARTA at the default size (10x10x10 grid, ~10K particles, 1000 steps):

| | in.collide (1M) | in.sphere (10K) |
|---|---:|---:|
| Move | 30% | **73.5%** |
| Collide | 40% | 17.6% |
| Sort | 30% | 5.4% |
| surface checks per particle-move | — | **2.38** |
| surface collisions per particle-move | — | 1.4e-4 |

The mover dominates, and inside it the cost is **ray-triangle intersection**
against the triangles of whatever cell a particle occupies — not particle
streaming. Actual surface collisions are four orders of magnitude rarer than the
checks that look for them.

That distinction is what settles the boundary question, because only the
collisions cross the boundary. The 23.8 million checks are geometry on `x` and
`xnew`; only the 1403 hits call `SurfCollide::collide(OnePart*)`.

## Results

`mini_sphere` reads the real `data.sphere` (1200 triangles), builds per-cell
surface lists as `Grid::surf2grid` does, and transcribes
`Geometry::line_tri_intersect` including its early-outs and the `EPSSQNEG` edge
test. 10x10x10, 1000 steps after 200 of equilibration:

| configuration | B/p | ns/move | speedup | move | sort | coll | checks/move |
|---|---:|---:|---:|---:|---:|---:|---:|
| AoS 96 B (SPARTA today) | 96 | 28.7 | 1.00x | 65.3% | 25.9% | 8.8% | 2.45 |
| AoS 96 B + mat boundary | 96 | 29.7 | 0.97x | 64.9% | 26.1% | 9.0% | 2.45 |
| SoA | 52 | 23.7 | 1.21x | 70.0% | 20.7% | 9.3% | 2.45 |
| SoA + mat boundary | 52 | 22.7 | 1.27x | 70.6% | 20.4% | 9.1% | 2.45 |

**The materialization boundary is free when surfaces are present.** AoS moves
28.7 -> 29.7 and SoA 23.7 -> 22.7 — one up, one down, both inside the noise. This
is the answer to the question left open at the end of round 6, and it is the
expected one once the check/collision ratio is known: the boundary is crossed
1.4e-4 times per particle-move, so it cannot cost anything measurable no matter
what a crossing costs.

**But SoA is worth much less here — 1.21x against 1.9x on `in.collide`.** The
reason is visible in the same table: move is 65-70% of the time and is spent on
ray-triangle arithmetic, which is compute, not streaming. SoA only helps the
streaming part. **The SoA payoff is problem-dependent**, and a surface-dominated
problem gets roughly a quarter of what a collision-dominated one gets. Any
decision to convert SPARTA's particle storage should be made against the mix of
problems that matter, not against `in.collide` alone.

## Validation, after two fidelity fixes

| metric | SPARTA | mini_sphere before | after |
|---|---:|---:|---:|
| move fraction | 73.5% | 65.3% | **71.0%** |
| sort fraction | 5.4% | 25.9% | **6.1%** |
| surface checks per particle-move | 2.38 | 2.45 | 2.05 |
| surface collisions per particle-move | 1.4e-4 | 2.5e-3 | **1.4e-4** |
| ns per particle-move | 31.5 | 27.8 | 26.1 |

Two bugs, both found by the validation table rather than by inspection:

- **No surface exclusion.** SPARTA's mover carries an `exclude` surface: the
  triangle just collided with is skipped on the next pass, because the particle
  now sits exactly on it and `line_tri_intersect` would re-detect it at param 0
  and collide again. Omitting that produced a runaway re-collision loop, capped
  only by an iteration guard, and was the entire cause of the 18x collision
  rate. With it, the rate matches SPARTA exactly at 1.4e-4.
- **Reordering every step.** `bench/in.sphere` sets no `particle/reorder`, so
  SPARTA sorts and never reorders. Reordering every step inflated the sort share
  to 26% against SPARTA's 5.4%. Corrected, sort is 6.1%.

On which: **reordering does not help in.sphere, and enabling it hurts.**
Measured in SPARTA by adding `global particle/reorder N` to the input:

| reorder | 0 | 1 | 5 | 20 |
|---|---:|---:|---:|---:|
| loop time | **0.318 s** | 0.367 | 0.348 | 0.322 |

10K particles is 0.96 MB and sits in L2, so there is no locality to buy and the
reorder is pure overhead. The default is right, and this is the opposite of
in.collide, where reordering is worth 1.9x.

## Optimizing the move: a bounding-box prefilter

The mover spends its time on 2.05 ray-triangle intersections per particle-move.
At 2500 m/s and a 1e-5 timestep a particle moves 0.025 of a cell width, so the
segment is tiny compared with the cell and is nowhere near most of the cell's
triangles. Six comparisons against a precomputed triangle bounding box reject
those before the ~40-flop plane-and-edge test:

| configuration | ns/move | speedup | full intersection tests per move |
|---|---:|---:|---:|
| AoS 96 B (SPARTA today) | 26.1 | 1.00x | 2.05 |
| AoS 96 B + AABB prefilter | 22.4 | **1.16x** | **0.01** |
| SoA | 24.2 | 1.08x | 2.05 |
| SoA + AABB prefilter | 21.2 | **1.23x** | 0.01 |
| SoA + prefilter + mat boundary | 21.2 | 1.23x | 0.01 |

**The prefilter eliminates 99.5% of the full intersection tests** — 2.05 per
move down to 0.01 — for 1.16x overall, with the surface collision rate
unchanged at 1.4e-4, so it rejects only triangles that would have missed.

**SPARTA has no such prefilter.** `Update::move` calls
`Geometry::line_tri_intersect` directly for every surface in the cell, after
only the `exclude` test. This is a contained and worthwhile optimization, and
the reason it is not implemented here is that doing it properly needs a
precomputed bounding box per surface element — computing one inline from
`tri->p1/p2/p3` costs about as much as the plane test it would save — which
means a `Surf`-side array maintained anywhere tris are created or moved
(`read_surf`, `move_surf`, `fix ablate`, implicit-surf grid adaptation,
distributed surfs). That is a bounded change but not one to land unverified at
the end of a session.

## Where the whole study stands

SPARTA performance: **7.34 s -> 6.26 s at 1M particles** (1.17x), from rounds 1-2.

| question | answer | evidence |
|---|---|---|
| AoS, SoA or AoSoA? | SoA; ~1.9x on in.collide, **~1.2x on in.sphere** | faithful mini-apps |
| Materialization boundary cost? | **free** — at or below noise, with and without surfaces | rounds 6, 7 |
| Reordering for in.sphere? | **no** — 0.318 s off against 0.367 s on; 10K particles fit in L2 | SPARTA |
| Surface bbox prefilter? | **1.16x on in.sphere**, 99.5% fewer intersection tests; SPARTA has none | round 7 |
| Vectorized mover? | 1.46x on move, but only with SoA; 23% regression on AoS | round 6 |
| SoA grid cells? | no — zero measured elasticity | SPARTA, by padding |
| 64-byte particle record? | ~1.15x | SPARTA, by padding |
| Mesh-free DSMC, tiling, buckets, fused passes? | no | round 2 |

The case for SoA is now quantified on both benchmarks and its main obstacle —
the `OnePart*` boundary — is measured and cheap. What is no longer in doubt is
the mechanism; what remains is that the payoff ranges from 1.2x to 1.9x
depending on whether a problem is surface-dominated or collision-dominated, and
that the conversion touches the particle subsystem rather than three kernels.
