# SWS (Species Weighting Scheme): implementation review and refactor plan

Status: written after the nagoya branch was merged onto master, reviewed,
and bug-fixed (see `git log` for the fix series). The behavior of the
current implementation is pinned by the `examples/sws` regression suite
(fixed-seed gold logs) and `examples/sws/verify_sws.py` (seed-independent
physics checks). Any refactor must keep those green.

## 1. What the feature does

Each species carries a weight `w_i` (`Particle::Species::specwt`, column 9
of the species file). With the `SWS`/`SWSmax` keywords on the `species`
command (`particle->sws` = 1/2), the physical-to-numerical ratio of species
`i` becomes `w_i * fnum`, so trace species can be oversampled. The touched
subsystems:

| Subsystem | Change |
|---|---|
| particle.cpp | parse keywords, reset weights when off, `count_wi` in `sort()` |
| mixture.cpp | `cummulative` becomes weighted by `f_i/w_i` (normalized) |
| create_particles.cpp | total `np` scaled by `sum(f_i/w_i)` |
| fix_emit_face/face_file/surf | per-species targets scaled by `1/w_i`; subsonic estimators use `count_wi` |
| collide.cpp | four cloned collision loops (`*_SWS`), per-cell `count_wi_group`/`maxwigr`, `Ewilost` pool |
| collide_vss.cpp | cloned kernels: `attempt/test/setup/perform_collision_SWS`, `SCATTER_TwoBodyScattering_SWS`, `EEXCHANGE_NonReactingEDisposal_SWS` |
| compute_*.cpp | tallies multiplied by `specwt`; `sumwi` grid value |
| grid.h | `ChildInfo::count_wi` field |

Key invariant that makes the design workable: **when SWS is off, every
`specwt` is forced to 1.0**, so all weighted formulas algebraically reduce
to the baseline. This was verified bit-for-bit against master (same seed,
identical stats) for the collide and emission paths.

## 2. Why the current shape is expensive

The dominant cost is the four cloned collision loops in `collide.cpp`
(~1,700 lines) and six cloned kernels in `collide_vss.cpp` (~800 lines).
They were copied from a 2024-era master and immediately went stale: master
meanwhile templated the loops on `GASTALLY` and restructured
`perform_collision()`, which made the merge conflict-heavy and left the
clones without the new gas-tally feature. Most of the bugs found in review
(index clobbering, stale `plist[k]` deletions, duplicated copy blocks,
NULL/uninitialized dereferences) lived in the cloned bookkeeping code, not
in the physics — exactly the kind of code that copy-paste multiplies.

## 3. Least-invasive reimplementation

### 3.1 Collision loops: one more template parameter, one helper

Master already dispatches `collisions_one<NEARCP,GASTALLY>()`. Add `SWS`:

```
template < int NEARCP, int GASTALLY, int SWS > void collisions_one();
```

Inside the existing loop the SWS deltas are small and local:

* while building `plist` (the loop already walks the cell's particles),
  accumulate `count_wi` and `maxwi` when `SWS` — this **removes the need
  for `Grid::ChildInfo::count_wi`** entirely (see 3.3);
* `attempt = attempt_collision(icell, np_eff, volume)` with
  `np_eff = count_wi` (SWS), `np*maxwi` (SWSmax), `np` (off) — one kernel,
  no `_SWS` clone (the baseline formula is the `w_i==1` special case);
* `test_collision(..., wscale)` with `wscale = max(w_i,w_j)/maxwi` for
  SWSmax, 1.0 otherwise;
* after `perform_collision(...)` returns the product multiplicities
  (`n_i,n_j,n_k,n_pre`, all 1/1/1/0 when SWS is off), call **one shared
  helper** for the genuinely new logic:

```
// clone/delete reaction products according to SWS multiplicities
void sws_products(int &np, int &i, int &j, Particle::OnePart *&ipart, ...);
```

That helper is the ~100-line replacement for the four divergent inline
copies of the product bookkeeping (which is where most review bugs were).
The ambipolar variants need a second, elist-aware overload — still one
copy instead of two.

With `SWS` as a template parameter the non-SWS instantiation compiles to
exactly today's code (no runtime cost), and the SWS instantiation
automatically inherits every future change to the baseline loop
(gas tallies, nearcp, etc.) instead of forking from it.

### 3.2 VSS kernels: parameterize instead of clone

* `attempt_collision(icell,np,volume)` → add an `np_eff` (double) argument;
  delete both `attempt_collision_SWS` overloads.
* `test_collision(...)` → add trailing `double wscale = 1.0`; delete
  `test_collision_SWS`.
* `setup_collision(...)`: the only SWS delta is re-injecting the pending
  split-merge energy; guard with `if (ewilost) etrans += ...` — the branch
  is free when the pool is empty. Delete `setup_collision_SWS`.
* `SCATTER_TwoBodyScattering(...)`: the SWS variant differs only by a
  post-scatter merge of the heavier particle's velocity when `phi < 1`
  (plus the `Ewilost` bookkeeping). Fold in as a guarded epilogue; delete
  the clone. Same for `EEXCHANGE_NonReactingEDisposal` (the `phi`-weighted
  internal-energy merge).
* `perform_collision(...)`: give it the four multiplicity out-params
  (or a small struct). When `sws == 0`, set 1/1/1/0 and take today's
  paths untouched; the probabilistic keep/copy draws happen only under
  `if (sws)` so the RNG stream of non-SWS runs is unchanged. Delete
  `perform_collision_SWS` (which is a stale fork of the pre-restructure
  master code).

Net: `collide_vss.cpp` shrinks by roughly 600 lines relative to the
current branch and stops depending on `particle->sws` inside kernels
(pass it or the derived quantities down explicitly).

### 3.3 Drop `ChildInfo::count_wi`

The per-cell weight sum is consumed in exactly two places:

1. the SWS collision loops — which already walk the cell's particle list
   and can accumulate it on the fly (3.1);
2. the subsonic emission estimators (`subsonic_grid()` in the three emit
   fixes) — which already loop the cell's particles to accumulate
   `mv[]`/`masstot` and can sum `w_i` in the same loop (they already
   compute `masstot_wi` there).

Removing the field reverts the `grid.h` struct change, the
`Particle::sort()` changes, the `subsonic_sort()` maintenance (which was
missing on the branch and had to be bug-fixed), and any future
pack/comm concerns about keeping it coherent. The recombination density
(`count_wi * fnum / volume`) also falls out of the loop-local sum.

### 3.4 Keep (they are already minimal)

* **Weights multiply through the computes.** Because `specwt == 1.0` when
  SWS is off, the `* specwt` factors in the tally computes are exact
  no-ops in baseline runs. Keep them; the review already normalized the
  few inconsistent spots (NFLUX/MFLUX guards, echem/etot, multi-mode
  tvib). The `sumwi`/`COUNT_WI` machinery in compute_grid is small.
* **Weighted `Mixture::cummulative`.** Folding `f_i/w_i` into the existing
  array (rather than a parallel `cummulative_weighted` array) keeps every
  consumer (create_particles, all three emit fixes) untouched. Document
  the changed semantics in mixture.h. The abandoned `cummulative_weighted`
  declarations have been deleted.
* **Emission `1/w_i` scaling** (6 sites) and the `create_particles` `np`
  scaling: inherently per-call-site, small, and now indexed correctly.
* **`sws` restart persistence** (added in the fix series).

### 3.5 The `Ewilost` pool: candidate for elimination

The branch conserves energy in split-merge collisions by accumulating the
merge loss and re-injecting it into a later max-weight pair collision.
The review had to make the pool per-cell persistent (`ewilost_cell`,
managed like `remain`) because the original per-step reset discarded the
energy and cooled an adiabatic box by 50% in 500 steps.

A refactor could remove the pool entirely by using Boyd's original
probabilistic update instead of velocity averaging: the heavier particle
takes the full post-collision velocity **with probability `phi`** (else
keeps its pre-collision velocity). Momentum and energy are then conserved
in expectation per collision with no pool, no per-cell array, no
pack/unpack — at the cost of slightly higher statistical noise and a
changed RNG stream (gold logs would need regeneration and a documented
re-bless). This is the single largest simplification available beyond
de-duplication, but it is a *physics* change and should be validated
against the relaxation benchmarks, not just the regression suite.

## 4. Explicitly unsupported combinations (now enforced with errors)

* gas-phase chemistry with a multi-group mixture (`Collide::init`)
* gas-collision tally computes (`Collide::collisions`)
* surface reactions (`SurfReact::init`)
* `create_particles` `species`/custom-fractions options
* `fix emit/surf` custom per-surf fractions
* the KOKKOS package (`Particle::init`)

Each error site is a TODO marker for finishing the feature; the refactor
in 3.1/3.2 makes the first two nearly free (the multi-group loop gets the
shared `sws_products()` helper; the gas-tally template dimension composes
with the SWS one).

## 5. Protection in place for the refactor

* `examples/sws` gold-log suite (7 fixed-seed tests, ctest suite "sws"):
  SWS box, SWSmax box, SWS-off equivalence, weighted emission, TCE
  chemistry, ambipolar, restart round-trip.
* `examples/sws/verify_sws.py`: seed-independent physics assertions
  (species split `~ f_i/w_i`, `sumwi` tallies, steady temperature,
  weighted emission ratio, physical mass conservation through reactions).
* SWS-off bit-equivalence versus master was verified manually for the
  collide and emit paths; `in.sws0.box` pins it going forward.

A refactor per 3.1–3.4 should reproduce the gold logs **only if** it
preserves the RNG call sequence (same draws in the same order). Where
that is impractical, rely on `verify_sws.py` plus a documented re-bless
of the gold logs.
