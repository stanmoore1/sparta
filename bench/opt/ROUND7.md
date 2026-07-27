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

## Validation, and one gap left open

| metric | SPARTA | mini_sphere |
|---|---:|---:|
| surface checks per particle-move | 2.38 | **2.45** |
| move fraction of runtime | 73.5% | 65.3% |
| surface collisions per particle-move | 1.4e-4 | **2.5e-3** |

The **check rate agrees to 3%**, which is the metric that matters here: it is the
2.45 ray-triangle tests per move that consume the runtime, and reproducing that
is what makes the timing comparison meaningful. The move fraction is lower than
SPARTA's mostly because this model reorders every step where `in.sphere` sorts
only, inflating the sort share.

**The surface collision rate is 18x too high and I have not tracked it down.**
Checked and eliminated: triangle normal orientation (forcing outward normals
changed nothing, so the vertex ordering in `data.sphere` was already consistent).
The most likely cause is the emission model, flagged as a deliberate
simplification at the top of the file: every particle leaving the box is
re-injected at the -x face, so the whole population continuously streams through
the sphere's path, whereas SPARTA's `fix emit/face` inserts a flux-weighted
Maxwellian on all faces and reaches a steady state with a bow shock and a
depleted wake. That would put more of this model's particles on trajectories that
intersect the sphere.

This matters for anything that depends on the collision rate and does not matter
for the conclusions above, both of which rest on the check rate and on a boundary
crossing that is rare under any of these numbers. Stating it rather than quietly
reporting the favourable metric.

## Where the whole study stands

SPARTA performance: **7.34 s -> 6.26 s at 1M particles** (1.17x), from rounds 1-2.

| question | answer | evidence |
|---|---|---|
| AoS, SoA or AoSoA? | SoA; ~1.9x on in.collide, **~1.2x on in.sphere** | faithful mini-apps |
| Materialization boundary cost? | **free** — at or below noise, with and without surfaces | rounds 6, 7 |
| Vectorized mover? | 1.46x on move, but only with SoA; 23% regression on AoS | round 6 |
| SoA grid cells? | no — zero measured elasticity | SPARTA, by padding |
| 64-byte particle record? | ~1.15x | SPARTA, by padding |
| Mesh-free DSMC, tiling, buckets, fused passes? | no | round 2 |

The case for SoA is now quantified on both benchmarks and its main obstacle —
the `OnePart*` boundary — is measured and cheap. What is no longer in doubt is
the mechanism; what remains is that the payoff ranges from 1.2x to 1.9x
depending on whether a problem is surface-dominated or collision-dominated, and
that the conversion touches the particle subsystem rather than three kernels.
