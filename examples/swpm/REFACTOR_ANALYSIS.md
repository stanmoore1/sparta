# SWPM: least-invasive re-implementation analysis

This note analyzes how the stochastic weighted particle method (SWPM) can be
re-implemented with a smaller, better-isolated footprint, without changing its
behavior.  It is written to guide the planned refactor; the accompanying
example/regression suite (`examples/swpm/`, `verify_swpm.py`) exists so the
refactor can be done under correctness protection — the invariants in
`verify_swpm.py` and the gold logs must keep passing.

## 1. Current footprint (what a refactor is starting from)

Measured against `master` (src/ only): 23 files, ~2050 insertions.  The change
falls into four groups, very different in how intrinsic they are to the feature:

| Group | Files | Intrinsic? |
|-------|-------|-----------|
| Per-particle weight storage (`fix_stochastic_weight`, custom attribute) | 2 new + `particle_custom.cpp` | Yes — idiomatic, keep |
| SWPM collision loop + split + grouping + reduction | `collide_swpm.cpp`, `collide_reduce.cpp` (new) | Yes — the algorithm |
| Hooks in the base `Collide` class + `CollideVSS` | `collide.h` (+14 members), `collide.cpp` (+126), `collide_vss.cpp` (2 hot-path branches) | Partly — misplaced |
| Weighting every diagnostic | 12 `compute_*.cpp` + `compute_vmom_grid.*` (new) | Partly — reducible |

The last two groups are where the invasiveness lives and where a refactor has
leverage.  The first two are essentially irreducible and are already close to
idiomatic SPARTA.

## 2. Problems worth fixing in a refactor

1. **SWPM logic lives on the base `Collide` class but is VSS-only.**
   `collisions_one_stochastic_weighting()`, `split()`, `group()`, `group_bt()`,
   `group_octree()`, `reduce_energy/heat/stress()`, and 14 data members were
   added to the base class.  Yet the SWPM loop calls `attempt_collision()`,
   `test_collision()`, `setup_collision()`, `perform_collision()` — all of
   which are **VSS** concepts (the base class declares them but only VSS
   implements the physics).  Any future non-VSS collision style inherits a
   large block of dead interface.

2. **Two `if (stochastic_weight_flag)` branches in VSS hot paths**
   (`attempt_collision`, `test_collision`).  Correct but they sit in the
   inner collision loop and re-fetch the custom array / recompute
   `MAX(isw,jsw)/max_stochastic_weight` per candidate pair.

3. **The diagnostic weighting is copy-pasted across 12 computes.**  Each one
   repeats the same shape:
   ```cpp
   double *sweights = particle->stochastic_weights();   // string lookup
   double swfrac = 1.0;
   ...
   if (sweights) swfrac = sweights[i];
   else if (particle->weightflag) swfrac = particles[i].weight;
   mass *= swfrac;                    // or: tally += value*swfrac
   ```
   `stochastic_weights()` does a `find_custom("stochastic_wt")` (an
   O(ncustom) `strcmp` scan) on every compute invocation.

4. **Dead / half-implemented code.**  `group_octree()` is a stub that calls
   `error->all`; `OCTREE` is parseable-ish but unreachable.  Duplicate
   `enum{ENERGY,HEAT,STRESS}` / `enum{BINARY,WEIGHT,OCTREE}` / `enum{INT,DOUBLE}`
   are redefined in several translation units.

## 3. Proposed re-implementation, by leverage

### 3.1 Move SWPM off the base class into the VSS layer (highest leverage, lowest risk)

SWPM is a VSS variant.  Relocate everything SWPM-specific from `Collide` into
`CollideVSS` (the code **moves**, it does not change):

- the 14 data members and the method declarations in `collide.h`;
- `collisions_one_stochastic_weighting`, `split`, `group`, `group_bt`,
  `reduce_*` (i.e. `collide_swpm.cpp` + `collide_reduce.cpp` compile into the
  VSS layer);
- the `collide_modify stochastic_weight|split|reduce` parsing (override
  `CollideVSS::modify_params`, falling back to `Collide::modify_params`).

The base `Collide::collisions()` dispatch block (the `else if
(stochastic_weight_flag)` ladder) collapses back to its original form.  This
keeps the **user interface unchanged** (`collide vss` + `collide_modify
stochastic_weight yes`) while removing SWPM from the base class entirely.

Two viable structures:
- **(a) Keep it inside `CollideVSS`.**  Smallest diff, no new style.  The SWPM
  members are unused when the option is off (a few ints/doubles).
- **(b) A `CollideVSSSW : public CollideVSS` subclass** registered as a distinct
  style.  Cleaner separation and zero overhead for plain VSS, but changes the
  user interface to `collide vss/sw` and duplicates some plumbing.  Only worth
  it if SWPM grows further (e.g. multi-species, Kokkos).

Recommendation: **(a)** now (minimal, preserves inputs and the gold logs);
revisit (b) only if SWPM is extended.

### 3.2 Collapse the diagnostic weighting behind one accessor (widest reduction)

Replace the repeated block in the 12 computes with a single effective-weight
accessor on `Particle`, resolved once and read per particle:

```cpp
// particle.h  (inline, no per-call string lookup)
inline double Particle::weight_one(int i) const {
  if (sweight)          return sweight[i];      // cached SWPM array ptr, or NULL
  if (weightflag)       return particles[i].weight;
  return 1.0;
}
```

`sweight` is refreshed (once) wherever the particle array can move — the same
places `particles`/`nlocal` are already refreshed — so the string
`find_custom` lookup disappears from the compute hot loops.  Each compute then
reads `double w = particle->weight_one(i);` and multiplies, turning the
repeated 4-line branch into one line and unifying SWPM with the pre-existing
grid weighting (`weightflag`) that several computes already handled.

This does not reduce the *number* of computes touched (each genuinely needs to
weight its tally), but it minimizes the change in each and removes the
per-invocation lookup.  A follow-up could push the multiply into shared tally
helpers, but that is a larger change than the feature itself warrants.

### 3.3 Cache the custom-attribute index

`Particle::stochastic_weights()` (or the `sweight` pointer above) should be
resolved from a cached index set when `fix stochastic_weight` registers the
attribute, not by `find_custom("stochastic_wt")` on every call.  `Collide`
already caches `index_stochastic_weight` in `init()`; the computes should use
the same cached index rather than re-scanning by name.

### 3.4 Remove dead code and de-duplicate enums

- Delete `group_octree()` and the `OCTREE` enum until octree grouping is
  actually implemented (a stub that only `error->all`s is a maintenance trap).
- Put the shared `enum{ENERGY,HEAT,STRESS}`, `enum{BINARY,WEIGHT,OCTREE}`,
  `enum{INT,DOUBLE}` in one header instead of redefining them per file.

## 4. What NOT to change

- **`fix stochastic_weight` + custom per-particle attribute.**  This is the
  idiomatic SPARTA way to attach per-particle data and it already handles
  restart and particle growth correctly.  Keep it.
- **Storing weights relative to `fnum`.**  Mutually exclusive with grid-based
  `particle->weight`; the exclusivity is already enforced.
- **The split/reduce math.**  The moment-preserving reductions are the feature;
  the refactor is about *placement*, not *algorithm*.  The invariants in
  `verify_swpm.py` pin this behavior.

## 5. Suggested order (each step keeps the suite green)

1. De-dup enums + delete the octree stub (mechanical, isolated).
2. Add `Particle::weight_one()` (or a cached `sweight` ptr) and convert the 12
   computes one at a time, re-running the gold logs after each.
3. Move SWPM from `Collide` into `CollideVSS` (structure 3.1a); rebuild and run
   the `swpm` ctest suite + `verify_swpm.py` at mpi_1 and mpi_4.

Steps 1–2 are independent of step 3 and can land separately.  After each step,
`ctest -R swpm` and `python3 verify_swpm.py <binary>` must still pass — that is
the correctness protection this suite was built to provide.
