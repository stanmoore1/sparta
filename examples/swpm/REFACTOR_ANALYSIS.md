# SWPM: least-invasive re-implementation analysis

This note analyzes how the stochastic weighted particle method (SWPM) can be
re-implemented with a smaller, better-isolated footprint, without changing its
behavior.  The accompanying suite (`examples/swpm/` gold-log ctests,
`verify_swpm.py`) exists so the refactor can be done under correctness
protection.  The concrete step-by-step plan derived from this analysis is in
`REFACTOR_PLAN.md`.

Revision note: an earlier version of this analysis recommended moving the SWPM
algorithm off the base `Collide` class into `CollideVSS`.  That recommendation
was withdrawn after checking the codebase precedents; see section 3.4.

## 1. Current footprint (what a refactor is starting from)

Measured against `master` (src/ only): 23 files, ~2050 insertions.  The change
falls into four groups, very different in how intrinsic they are to the feature:

| Group | Files | Intrinsic? |
|-------|-------|-----------|
| Per-particle weight storage (`fix_stochastic_weight`, custom attribute) | 2 new + `particle_custom.cpp` | Yes — idiomatic, keep |
| SWPM collision loop + split + grouping + reduction | `collide_swpm.cpp`, `collide_reduce.cpp` (new) | Yes — the algorithm |
| Hooks in the base `Collide` class + `CollideVSS` | `collide.h` (+14 members), `collide.cpp` (+126), `collide_vss.cpp` (2 hot-path branches) | Yes — matches the ambipolar precedent (see 3.4) |
| Weighting every diagnostic | 12 `compute_*.cpp` + `compute_vmom_grid.*` (new) | Partly — reducible |

The new files and the weight-storage design are essentially irreducible and
already idiomatic.  The reducible invasiveness is concentrated in the
diagnostics and in incidental churn.

## 2. Problems worth fixing in a refactor

1. **The diagnostic weighting is copy-pasted across 12 computes.**  Each one
   repeats the same shape:
   ```cpp
   double *sweights = particle->stochastic_weights();   // string lookup
   double swfrac = 1.0;
   ...
   if (sweights) swfrac = sweights[i];
   else if (particle->weightflag) swfrac = particles[i].weight;
   mass *= swfrac;                    // or: tally += value*swfrac
   ```
   `Particle::stochastic_weights()` does a `find_custom("stochastic_wt")`
   string scan on every compute invocation, and the
   `else if (particle->weightflag)` fallback is dead code:
   `Particle::weightflag` is initialized to 0 and never set anywhere.

2. **`compute_reduce.cpp` is ~242 changed lines of which only ~37 are real.**
   The rest is tab/space indentation churn that bloats the diff against
   upstream and will cause merge pain forever.  Separately, the weighting
   semantics added there are questionable: SUM multiplies every per-particle
   value (including positions `x`!) by the weight, while AVE divides the
   weighted sum by the **raw particle count** (`count_included()`), which is
   neither an unweighted average nor a weighted average.  This needs a
   deliberate semantics decision, not just a cleanup.

3. **Dead / half-implemented code.**
   - `group_octree()` is a stub whose body is `error->all(...)`, and the
     `OCTREE` enum value is unreachable (`collide_modify reduce` only parses
     `binary`/`weight`).
   - The `NEARCP` and `GASTALLY` template parameters of
     `collisions_one_stochastic_weighting<>` are never used in its body:
     nearcp+SWPM is rejected at init, and gas-collision tallies are silently
     skipped under SWPM (a documented limitation at best, a surprise at worst).
   - `enum{ENERGY,HEAT,STRESS}`, `enum{BINARY,WEIGHT,OCTREE}`,
     `enum{INT,DOUBLE}` are re-declared in several translation units.
   - `Particle::weightflag` (see item 1).

4. **Small hot-path inefficiencies.**  `CollideVSS::test_collision` re-fetches
   the custom weight array via `particle->edvec[particle->ewhich[...]]` and
   recomputes `MAX(isw,jsw)/max_stochastic_weight` per candidate pair.  Minor,
   but easy to cache per cell.

## 3. Design decisions

### 3.1 Weight storage: keep the custom attribute (do not change)

`fix stochastic_weight` + the `stochastic_wt` custom per-particle attribute is
the idiomatic SPARTA mechanism: it already handles particle-array growth,
migration, restart and `compress_reactions` correctly, and it keeps
`Particle::OnePart` (and the restart file format) identical to upstream.
Storing weights relative to `fnum`, and the mutual exclusion with grid-based
cell weighting, are also right.  Keep all of it.

### 3.2 Diagnostics: one cached accessor instead of 12 copies

Give `Particle` a cached effective-weight accessor:

```cpp
// particle.h
int index_sweight;                  // custom index of "stochastic_wt", -1 if absent
                                    // set/cleared by fix stochastic_weight
inline double *sweight_vector() {   // NULL when SWPM inactive
  if (index_sweight < 0) return NULL;
  return edvec[ewhich[index_sweight]];
}
```

Each compute then does, once per invocation:

```cpp
double *swgt = particle->sweight_vector();
...
if (swgt) mass *= swgt[i];          // expression shape kept identical
```

This removes the per-invocation string lookup, deletes the dead
`weightflag` fallback (and the `weightflag` member itself, restoring
`particle.h` to upstream plus one index + one inline function), and shrinks
each compute's diff to the minimum it can be — the tally multiplication
itself, which is irreducible because each compute genuinely must weight its
tally.  Kept bit-exact by preserving the multiplication order in each tally
expression.

### 3.3 `compute_reduce`: revert churn, then decide semantics

Two independent actions:
- Revert the pure-whitespace churn (mechanical, zero risk, shrinks the
  upstream diff by ~200 lines).
- Decide the weighting semantics.  Recommendation: weight only quantities that
  are physically extensive (KE, EROT, EVIB, custom vectors), do not weight
  coordinates/velocities under SUM, and normalize AVE by the weight sum rather
  than the particle count — or, more conservatively, drop weighting from
  `compute reduce` entirely and document that per-grid computes are the
  weighted path.  Either way the chosen semantics must be documented in
  `compute_reduce.txt`.  (No regression deck currently pins this behavior, so
  the decision is still cheap to make.)

### 3.4 Class placement: keep SWPM on the base `Collide` class

The earlier recommendation to move the SWPM methods/members into `CollideVSS`
is withdrawn, for three reasons checked against the codebase:

1. **The interface the SWPM loop uses is the base-class interface.**
   `attempt_collision`, `test_collision`, `setup_collision`,
   `perform_collision` are pure virtuals on `Collide`; the SWPM loop is
   style-agnostic in exactly the way `collisions_one` and
   `collisions_one_ambipolar` are.
2. **Ambipolar is the precedent, and it lives on the base class.**  An
   optional collision-time algorithm, enabled by `collide_modify`, backed by a
   companion fix and custom particle data — that is ambipolar's exact shape,
   and its loop, flag and helpers are all on `Collide`.  SWPM's current
   placement follows the house convention; relocating it would make the two
   features inconsistent with each other.
3. **A future Kokkos port wants the members on the base class.**
   `CollideVSSKokkos : public CollideVSS`; flags and thresholds on `Collide`
   are inherited by the whole hierarchy, which is how `ambiflag` reaches the
   Kokkos style today.

What remains VSS-specific — the ~20 lines in `CollideVSS::attempt_collision`
and `test_collision` that scale the attempt count and acceptance probability
by the weights — is already in `collide_vss.cpp`, i.e. in the right place.
A `collide vss/sw` subclass (registered as its own style) is only worth
revisiting if SWPM grows large enough to strain the base class (Kokkos
port, multi-species reduction); it changes the user-facing input syntax and
is strictly optional.

### 3.5 Behavior-preservation classes for refactor steps

Refactor steps must be classified before starting, because the gold logs pin
bit-exact trajectories:

- **Bit-exact** steps (no RNG-call-order change, no FP reassociation): file
  moves, enum consolidation, dead-code deletion, the accessor conversion of
  section 3.2 (expression shape preserved), index caching.  The gold logs must
  keep passing unchanged — any diff is a bug in the refactor.
- **Statistically-equivalent** steps (RNG order or FP order changes, e.g.
  restructuring the acceptance test into separate thinning draws): the gold
  logs will legitimately change and must be re-blessed, and `verify_swpm.py`
  plus the BKW benchmark become the correctness evidence.  Avoid mixing these
  with bit-exact steps in one commit.

The plan in `REFACTOR_PLAN.md` keeps every step bit-exact except where
explicitly marked.

## 4. Known limitations to track (not part of this refactor)

Documented here so they are not silently lost; each is guarded by an init
error or is invisible today:

- SWPM is not Kokkos-enabled (init error added).
- Particle reduction requires a single species (init error added); the
  reduction schemes use a single mass for the group.
- Gas-collision tallies (`compute .../gas/tally`) are not tallied by the SWPM
  collision loop (unused `GASTALLY` template parameter).
- `react` chemistry alongside SWPM is untested; reduction and chemistry share
  the deletion list.
- Octree grouping is unimplemented (stub slated for deletion).
- Particles created by splitting within a timestep join grouping/reduction on
  the next timestep (the cell linked lists are from the pre-collision sort).
