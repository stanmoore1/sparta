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
| `0001b-inline-line-tri-intersect.patch` (alternative to 0001) | `bench/in.sphere` | 0.3303 s | 0.2964 s | **1.114x** | ~110 moved |
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

### 0001b — the same win by inlining the function instead

`0001b` is an **alternative to `0001`, not an addition** — apply one or the other. Rather than
duplicating the plane test at the call site, it moves `Geometry::line_tri_intersect` out of
`geometry.cpp` and into `geometry.h` as an `inline` function, so the compiler can see the early-out
and drop the call itself.

Measured head to head in the same run, the two are indistinguishable:

| | `in.sphere` | speedup |
|---|---:|---:|
| pristine | 0.3303 s | 1.000x |
| 0001, hand-hoisted plane test | 0.2958 s | **1.117x** |
| 0001b, `line_tri_intersect` inlined | 0.2964 s | **1.114x** |

**0001b is probably the better change to upstream**, for a reason that only became clear from
looking at the Kokkos package: `GeometryKokkos::line_tri_intersect`
(`src/KOKKOS/geometry_kokkos.h:421`) is already a `KOKKOS_INLINE_FUNCTION` defined in a header, so
the device path has always had this. The host version is out-of-line in a separate translation
unit, and that difference is the entire effect being measured here. 0001b brings the host into line
with what Kokkos already does, keeps one copy of the test rather than two that must stay in sync,
and would extend naturally to `line_line_intersect` and `axi_line_intersect` (not done here).

Its cost is that `geometry.h` grows by ~110 lines and more translation units recompile when it
changes. 0001 is the smaller, more surgical option if that matters.

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

## Does any of this need porting to Kokkos?

Checked directly, and the short answer is **no, and none of it can break the Kokkos path either**.

- `UpdateKokkos::run()` (`src/KOKKOS/update_kokkos.cpp:355-400`) is a **separate timestep loop**. It
  never calls `Particle::sort_reorder()`, `Particle::sort()`, `cellcount_start/stop()`, or sets
  `defer_flag` — all of those are non-virtual host methods reached only from `Update::run()`. That
  matters most for `sort_reorder`, whose ping-pong swap of the `particles` and `sortbuf` pointers
  would desynchronize the Kokkos `DualView` (on that path `particles` *is*
  `k_particles.view_host().data()`, `particle_kokkos.cpp:648`). It cannot fire.
- **Kokkos already has the structural equivalent of the reorder work.** `ParticleKokkos::sort_kokkos()`
  keeps per-cell counts in `d_cellcount`, bins into a CSR `d_plist` rather than a linked list, and
  reorders out of place via `COPYPARTICLELIST` into `d_sorted`. The host changes on this branch
  brought the CPU path up to roughly what the device path already did; they did not overtake it.
- The remaining host-side reorder tricks are CPU cache behaviour that does not map to a GPU: fusing
  the collide into the scatter (locality), counting cells during the move (avoiding a serial
  streaming read; Kokkos counts with atomics inside an already-parallel pass), and deferring the
  position write-back (avoiding read-for-ownership).
- **0001/0001b are not needed on Kokkos** for the reason given above — that path is already inlined.

Two things that *are* worth a fix, neither of which is a live bug today:

1. **`global collide/every N` is silently ignored under Kokkos.** It is parsed by the inherited
   `Update::global()`, so the input deck is accepted, but `UpdateKokkos::run()` calls
   `collide->collisions()` unconditionally every step and never sets `Collide::nstep_collide`. The
   physics stays correct (it just does not sub-cycle), but the user gets no indication their request
   was dropped. `UpdateKokkos::init()` should error out — or the option should be ported. This one is
   specific to this branch, since `collide/every` does not exist upstream.
2. **`sorted_contiguous` is not reset on the Kokkos path.** `UpdateKokkos::move` resets
   `particle->sorted` (`update_kokkos.cpp:783`) but not `sorted_contiguous`, and
   `ParticleKokkos::compress_reactions` (`particle_kokkos.cpp:162`) does the same; the host
   equivalents reset both (`update.cpp:1431`, `particle.cpp:255,292`). It is harmless today because
   nothing on the Kokkos path ever sets the flag to 1 and `CollideVSS::collisions()` — its only
   reader, at `collide_vss.cpp:398` — is virtual and overridden by `CollideVSSKokkos`. It becomes a
   silent wrong-answer bug the moment anyone sets that flag on the device path, so it is worth the
   one-line addition now.

Everything else on this branch (the fused counting sort, collision sub-cycling, counting cells
during the move, deferring the position write-back) is either a larger change to `Particle`'s
internals, a new user-facing option, or — in the last case — a measured effect I could not fully
explain. Those are written up in `../ROUND*.md` and are not proposed here.
