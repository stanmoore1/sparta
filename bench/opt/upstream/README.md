# Standalone patches for upstreaming

Each patch here applies to **pristine upstream** (`09391fe`, "Merge pull request #649"),
independently of the rest of this branch, and each was built and measured **on its own** so the
number attached to it is that change's own contribution rather than a share of a bundle.

Both are **bitwise identical** to the unpatched binary: same `np`, `nattempt`, `ncoll`, `nscoll`
and `c_temp` on every stats line.

Timings are the minimum of 8 **interleaved** A/B pairs on one pinned core. Interleaving matters
here — this machine drifts by tens of percent over minutes, and measuring A as a block then B as a
block produced a 17-point phantom swing earlier in this work.

| patch | benchmark | pristine | patched | speedup | lines |
|---|---|---:|---:|---:|---:|
| `0001-hoist-tri-plane-rejection.patch` | `bench/in.sphere` | 0.3629 s | 0.3163 s | **1.147x** | ~20 |
| `0002-dense-uniform-cell-index.patch` | `bench/in.collide` 1M, `optmove yes`, `reorder 5` | 7.250 s | 6.943 s | **1.044x** | ~180 |

## 0001 — hoist the triangle plane rejection out of `line_tri_intersect`

`Update::move` calls `Geometry::line_tri_intersect()` for every surface element in a cell, and the
first thing that out-of-line function does is reject segments lying wholly on one side of the
triangle's plane. At `in.sphere`'s timestep a particle covers a few percent of a cell per step, so
nearly every candidate is rejected there — after paying for a cross-translation-unit call, at 2.4
calls per particle-move.

The patch performs that same test inline before the call, with identical arithmetic in identical
order, so the set of triangles that reach the full test is unchanged.

Only the `DIM == 3` branch is patched. The 2d (`line_line_intersect`) and axisymmetric
(`axi_line_intersect`) branches have the same shape and would likely benefit similarly, but they
were not measured here and so were left alone.

## 0002 — dense cell-ID map for the optimized move

The `optmove` fast path does `grid->hash->find(cellIdx)` — an `std::unordered_map` lookup — once
per particle per timestep. But `optmove` already requires a uniform grid (`Update::init` errors out
otherwise), and on a uniform grid that map is *dense* over `1 .. unx*uny*unz`, so it can be a plain
indexed array instead.

`Grid::request_uniform_index()` asks for the array; it is built inside `Grid::rehash()`, so it is
rebuilt on exactly the events that rebuild the hash and can never go stale. It declines — returning
NULL, and the mover falls back to the hash unchanged — for non-uniform grids and when the dense
array would be much larger than the cells the proc actually owns. It is sized to include particles
sitting exactly on the upper box face.

This replaces two dependent cache misses per particle per step with one indexed load. The gain is
modest because, as the profiling in `../PROFILE.md` and `../ROOFLINE.md` shows, the mover is
bandwidth- and latency-bound rather than short of instructions.

## Measured, and deliberately not included

- **Inlining `RanKnuth::uniform()` into the header.** Measured on its own it is inside the noise
  (0.97x and 1.03x on two separate 8-pair runs). It was part of a bundle that helped, but it does
  not stand up as an individual change and is not worth the churn.
- **Devirtualizing the VSS collision kernel.** No effect at all — see `../RESULTS.md`. The profile
  had already said why: 4 mispredictions out of 10.8M indirect branches. Removing instructions from
  a kernel stalled on dependent loads does not make it faster. This was the most obvious-looking
  optimization in the codebase and it is worth recording that it does nothing.
- `-march=native` is worth about 3% and is a build-flag choice, not a code change.

Everything else on this branch (the fused counting sort, collision sub-cycling, counting cells
during the move, deferring the position write-back) is either a larger change to `Particle`'s
internals, a new user-facing option, or — in the last case — a measured effect I could not fully
explain. Those are written up in `../ROUND*.md` and are not proposed here.
