# Unify SWS and SWPM particle weighting in SPARTA

## Context

SPARTA now has three particle-weighting schemes answering the same question —
"how many physical particles does this numerical particle represent, relative
to fnum" — implemented three different ways:

1. **Cell weighting** (master): per-cell `cinfo[icell].weight`, snapshot in
   `OnePart::weight`, clone/delete via `Particle::pre_weight()/post_weight()`,
   enabled by `global weight` (`grid->cellweightflag`). Applies its weight at
   the **per-cell normalization** stage of computes (`wt = fnum * cinfo.weight / norm`).
2. **SWS** (branch `claude/nagoya-code-review-l1e63d`, reviewed + fixed +
   tested): static per-species `Species::specwt`, `particle->sws` = 0/1/2,
   four cloned `_SWS` collision loops on base `Collide`, six `_SWS` VSS
   kernels, `Ewilost`/`ewilost_cell` energy pool, `* specwt` **inside** the
   tally loops of ~9 computes. Gold-log suite `examples/sws` + `verify_sws.py`.
3. **SWPM** (branch `origin/claude/swpm-review-tests-b8jf0s`, reviewed +
   fixed + tested): dynamic per-particle weight in custom attribute
   `"stochastic_wt"` owned by `fix stochastic_weight`, split/reduce algorithms
   in `collide_swpm.cpp`/`collide_reduce.cpp` on base `Collide`,
   `collisions_one_stochastic_weighting()` dispatched by
   `stochastic_weight_flag` (`collide_modify` keywords), `* sweights[i]`
   **inside** the same tally loops (+ `compute_temp`, `compute_reduce`,
   new `compute vmom/grid`). Gold-log suite `examples/swpm` + `verify_swpm.py`.

SWS and SWPM textually conflict in `collide.h`/`collide.cpp` (same member
block, same dispatch tree, both re-template the loops), `collide_vss.cpp`
(same `test_collision` acceptance line), and ~10 compute tally loops. Both
branches carry refactor docs (`doc/sws_refactor_analysis.md`,
`examples/swpm/REFACTOR_{ANALYSIS,PLAN}.md`) that independently converge on
the same seam: *one multiplicative effective weight per particle* applied to
(a) collision acceptance and (b) every diagnostic tally.

**Goal**: both features coexisting on one branch behind a single, modular
weighting abstraction, with both gold-log suites green, ~2,000 duplicated
lines deleted, and a clear extension path for future schemes and a Kokkos
port.

**User decisions**: full unification (Steps 1–5); include the compute_temp
SWS fix (re-bless); SWS/SWPM/cell weighting stay **mutually exclusive**.

## Architecture

### A. One weight accessor, two application sites (the core abstraction)

Weight enters diagnostics at two distinct sites; only site 2 is unified:

- Site 1 (per-cell normalization, `wt = fnum * cinfo[icell].weight / norm`):
  cell weighting only. **Unchanged** — keeps master bit-exact and avoids
  overloading `OnePart::weight`.
- Site 2 (per-particle in-loop scaling): SWS + SWPM. Unified via a new
  inline accessor on `Particle` (in `src/particle.h`):

```cpp
enum WeightMode { WEIGHT_NONE, WEIGHT_CELL, WEIGHT_SPECIES, WEIGHT_PARTICLE };
int weight_mode;      // resolved once per run by setup_weighting()
int index_sweight;    // custom index of "stochastic_wt", -1 if absent

inline double pweight(int i) {          // in-loop weight, relative to fnum
  switch (weight_mode) {
    case WEIGHT_SPECIES:  return species[particles[i].ispecies].specwt;
    case WEIGHT_PARTICLE: return edvec[ewhich[index_sweight]][i];
    default:              return 1.0;   // NONE and CELL
  }
}
inline double *pweight_vector() {       // hot-path hoist; NULL unless SWPM
  return weight_mode == WEIGHT_PARTICLE ? edvec[ewhich[index_sweight]] : NULL;
}
```

Every compute's copied weighting block collapses to `double w =
particle->pweight(i);` with the multiplication order of each tally expression
preserved exactly (bit-exactness). Collision hot paths do NOT call
`pweight(i)` per pair: SWS keeps passing derived per-species quantities down;
SWPM hoists via `pweight_vector()` (replaces its per-pair
`edvec[ewhich[...]]` re-fetch and the `stochastic_weights()` per-call string
lookup). The dead `Particle::weightflag` / `else if (weightflag)` fallback in
every SWPM compute is deleted.

Rejected alternatives (recorded for the record): a NULL-able weight vector
alone doesn't cover SWS's per-species weights; materializing all schemes into
`OnePart::weight` conflicts with `post_weight()`'s cell-weight semantics and
adds persisted per-particle state for zero benefit.

### B. One weighting-mode resolution + mutual exclusion

New `Particle::setup_weighting()` called from `Update::setup()` (after
commands parse, before any run, works with or without a collide style):

- reads `grid->cellweightflag`, `particle->sws`,
  `collide ? collide->stochastic_weight_flag : 0`
- errors if more than one is active (single message naming the three
  commands); sets `weight_mode`
- folds in the two existing KOKKOS guards (particle.cpp:156, SWPM's) so
  neither scheme can silently run under Kokkos
- resolves `index_sweight` when SWPM is active

The three user-facing commands (`species ... SWS|SWSmax`,
`collide_modify stochastic_weight ...`, `global weight`) are untouched.

### C. Collide dispatch: two loops, runtime flags — NO new template dimensions

The algorithms stay separate (different physics: SWS static + Ewilost pool;
SWPM split/reduce). Since they are mutually exclusive:

```cpp
if (weight_mode == WEIGHT_PARTICLE) collisions_one_stochastic_weighting(); // SWPM
else if (ambiflag) { ... collisions_*_ambipolar<GASTALLY>() ... }          // SWS inside
else { ... collisions_one/group<NEARCP,GASTALLY>() ... }                   // SWS inside
```

The existing `<NEARCP,GASTALLY>` template params stay exactly as upstream
has them; **SWS does NOT become a third template dimension** (that path
leads to 12 instantiations per loop and multiplies with every future
option). Instead SWS folds into the baseline loops as **runtime flags +
kernel parameters + shared helpers** — the established house pattern:
`recombflag`/`remainflag` are already checked per collision attempt at
runtime inside these same hot loops (collide.cpp:552 etc.), and SWS's
branches are equally run-constant and perfectly predicted.

How the SWS deltas distribute (this is the modularity structure):

1. **Values, not branches, wherever possible.** Per cell, while building
   `plist` (a walk the loop already does), the loop calls one SWS helper to
   get `count_wi`/`maxwi`/`np_eff` (= `count_wi` for SWS, `np*maxwi` for
   SWSmax, `np` when off). `attempt_collision(icell, np, np_eff, volume)`
   takes it as an argument — the kernel has no weighting branch at all;
   baseline passes `np_eff == np`. Same for
   `test_collision(..., double wscale = 1.0)`: SWSmax passes
   `MAX(w_i,w_j)/maxwi`, everyone else 1.0. (This `wscale` seam is also
   where SWPM's acceptance scaling plugs in, ending the two-way contention
   on that line.)
2. **One-line guarded calls at the kernel seams; bodies in the SWS file.**
   `SCATTER_TwoBodyScattering` and `EEXCHANGE_NonReactingEDisposal` end
   with `if (phi < 1.0) sws_scatter_merge(...)` /
   `sws_energy_merge(...)` — the ~30-line split-merge blend + Ewilost
   bookkeeping bodies live in `collide_sws.cpp`, not in the base kernel.
   With weights off `phi == 1.0` and the guard is dead-predictable. Delete
   the `_SWS` kernel clones. `perform_collision` gains
   product-multiplicity out-params (a small struct, e.g.
   `Collide::ProductCount {n_i,n_j,n_k,n_pre}`), filled 1/1/1/0 on the
   non-SWS path; the probabilistic multiplicity draws + p_pre creation are
   an SWS helper call.
3. **File-level separation via the multi-file-class idiom** (the pattern
   SWPM already uses — `collide_swpm.cpp`/`collide_reduce.cpp` define
   `Collide::` methods with no own header; same as
   `grid_adapt/grid_comm/grid_surf.cpp`, `particle_custom.cpp`). New
   **`src/collide_sws.cpp`** holds ALL SWS method bodies (methods of both
   `Collide` and `CollideVSS`; declarations stay in the existing headers,
   no new class):
   - `sws_cell_weights()` — per-cell count_wi/maxwi/np_eff aggregates
   - `sws_attempt()` — the weighted attempt-count formulas
   - `sws_product_counts()` — phi draws + p_pre creation
     (from `perform_collision_SWS`)
   - `sws_products()` + elist-aware ambipolar overload — product
     creation/deletion/plist bookkeeping (the code where most review bugs
     lived)
   - `sws_scatter_merge()`, `sws_energy_merge()`, `sws_ewilost_inject()` —
     the split-merge physics and Ewilost pool interplay
   After this, `collide.cpp` contains only ~5 one-line guarded calls per
   loop, and `collide_vss.cpp` only the neutral parameters + 2–3 guarded
   one-liners. The four `_SWS` loop clones (~1,700 lines) are deleted; the
   baseline loops grow by ~10 lines each.
4. **SWPM stays its own loop** on base `Collide` in its own files
   (ambipolar precedent, its plan §3.4) — its structure (split before
   collision, group/reduce per cell) is genuinely different, and mutual
   exclusivity means it never multiplies with the other options. Drop its
   unused `<NEARCP,GASTALLY>` template params (its plan step 1.4) → a
   plain method. Where it needs the same idioms (plist growth,
   deletion-swap), reuse the shared helpers.

File layout after unification (symmetric per scheme):

| Scheme | Own files | Base-file footprint |
|---|---|---|
| SWS  | `collide_sws.cpp` (new) | guarded one-line calls + neutral kernel params |
| SWPM | `collide_swpm.cpp`, `collide_reduce.cpp`, `fix_stochastic_weight.*` | dispatch branch + `modify_params` keywords + `wscale` seam |
| cell | (master, unchanged) | normalization stage only |

Rejected alternative (recorded): a policy/strategy object with virtual
hooks at the loop's variation points (setup/attempt/test/transform/
products/epilogue). It would give one skeleton for DSMC/SWS/SWPM, but adds
per-collision virtual dispatch, departs from SPARTA's explicit-loop style,
and maps poorly onto the mirrored Kokkos loop implementations. The
flag+parameter+helper design achieves the same de-duplication with zero
dispatch machinery.

- `collide.h`: SWS and SWPM members kept in separate commented sub-blocks;
  the duplicated file-scope enums (`{ENERGY,HEAT,STRESS}`,
  `{BINARY,WEIGHT,OCTREE}`) consolidated into `collide.h`. The six `_SWS`
  pure-virtuals on `Collide` are deleted along with the kernel clones
  (the base interface returns to upstream's shape plus the two new kernel
  parameters).

## Steps

**Verification gate after every step** (also see per-step notes):
build serial; `ctest -R sws` and `ctest -R swpm` equivalents via
`tools/testing/regression.py` (mpi_1 gold logs; mpi_4 where available);
`python3 examples/sws/verify_sws.py <spa>`;
`python3 examples/swpm/verify_swpm.py <spa>`.
On a **bit-exact** step, any gold-log diff = bug in the step; stop and fix,
never re-bless.

### Step 0 — baseline (no code)
Record both suites' passing output on their own branches. Save the SWPM BKW
benchmark (`examples/swpm/bkw_init.py` + `in.swpm.bkw`) and the SWS thermal
box as physics baselines for later re-bless steps.

### Step 1 — merge SWPM into the SWS branch [bit-exact]
Merge `origin/claude/swpm-review-tests-b8jf0s` into
`claude/nagoya-code-review-l1e63d` (work on a new branch or the designated
branch per instructions at execution time). Resolve conflicts to a
side-by-side state: SWS loops still cloned, SWPM loop separate, both
features' members present, dispatch guarded so at most one runs.
- `src/collide.h` / `src/collide.cpp`: reconcile member block, dispatch tree,
  `modify_params`; consolidate duplicated enums here.
- `src/collide_vss.cpp`: keep SWS `_SWS` overloads AND SWPM's in-place
  `vremax_init`/`test_collision` weighting (guarded by
  `stochastic_weight_flag`); note the review branch already removed SWPM's
  bad `omega==1` override — keep the reviewed version.
- `src/particle.h/.cpp`: both additions are adjacent; keep both. Drop
  SWPM's write-only `OnePartRestart::weight` (already dropped on review
  branch).
- Temporary ad-hoc exclusion guard (SWS + SWPM together → error) until
  Step 3.
Gate: both suites green, `in.sws0.box` and `in.swpm.periodic` (each feature
off/neutral) byte-identical to their gold logs.

### Step 2 — unified `pweight` accessor; convert all computes [bit-exact]
Add `weight_mode` (derived inline from existing flags for now),
`index_sweight`, `pweight()`, `pweight_vector()` to `src/particle.h/.cpp`.
Convert one compute per commit, preserving multiplication order:
`compute_grid.cpp`, `compute_thermal_grid.cpp`, `compute_eflux_grid.cpp`,
`compute_pflux_grid.cpp`, `compute_sonine_grid.cpp`, `compute_tvib_grid.cpp`,
`compute_boundary.cpp`, `compute_surf.cpp`, `compute_isurf_grid.cpp`,
`compute_vmom_grid.cpp`, `compute_reduce.cpp`.
Then delete `Particle::weightflag` and the `stochastic_weights()`
string-lookup accessor (SWPM plan steps 2.1–2.3). Reconcile SWS's
`COUNT_WI`/`NUMWI`/`sumwi` and SWPM's `SIMCOUNT` enums in compute_grid into
one coherent set (weighted count + raw simulator count outputs available
under either scheme).

### Step 3 — single mode/exclusion model [bit-exact]
Add `Particle::setup_weighting()`; call from `Update::setup()`
(`src/update.cpp`). Replace scattered pairwise guards (particle.cpp KOKKOS
check, collide.cpp SWPM checks, SWPM/cell exclusivity, Step-1 temp guard).
Dispatch and accessor now read `weight_mode` only.

### Step 4 — de-duplicate the collision machinery [bit-exact target; re-bless fallback]
Fold SWS into the baseline loops as runtime flags + kernel parameters +
helpers whose bodies live in a new `src/collide_sws.cpp` (Architecture §C —
NOT a new template dimension, no new class):
- 4a. Create `src/collide_sws.cpp`; move/derive the SWS helper bodies there
  (`sws_cell_weights`, `sws_attempt`, `sws_product_counts`,
  `sws_products` + ambipolar overload, `sws_scatter_merge`,
  `sws_energy_merge`, `sws_ewilost_inject`). Kernel signatures:
  `attempt_collision` gains `np_eff`; `test_collision` gains
  `wscale = 1.0`; `perform_collision` gains the `ProductCount` out-param;
  one-line guarded calls at the `SCATTER_*`/`EEXCHANGE_*` seams. Baseline
  callers pass neutral values — bit-exact, both suites green, `_SWS`
  kernels still present but now thin wrappers.
- 4b. One loop family per commit (one, group, ambipolar-one,
  ambipolar-group): add the per-cell `sws_cell_weights()` call and the
  `if (sws)` product-bookkeeping call to the baseline loop, switch
  dispatch to it, delete the `_SWS` clone.
- 4c. Delete the six `_SWS` VSS kernels + their `Collide` pure-virtuals;
  de-template the SWPM loop; delete octree stub if not already gone.
  After 4c, `git grep -c SWS src/collide.cpp src/collide_vss.cpp` should
  show only the guarded call sites; all SWS bodies are in
  `collide_sws.cpp`.
- Highest-risk step: preserve the RNG draw order so gold logs stay
  byte-identical. Where a change is unavoidable, isolate it in its own
  commit, justify, re-bless only the affected decks with `verify_sws.py` +
  physics baselines as evidence.
- The `sws && ngas_tally` error becomes removable (GASTALLY now composes
  with SWS at zero extra instantiations); keep or lift per test results.

### Step 5 — close the compute_temp SWS gap [behavior change; re-bless]
`compute_temp.cpp` gets the uniform `pweight` path with weighted
effective-count normalization (already SWPM's behavior). Under SWS this
changes output (previously unweighted) — a correctness fix. Isolated commit;
re-bless affected SWS gold logs; add a `verify_sws.py` assertion pinning the
weighted temperature.

### Step 6 — recorded follow-ups (NOT in this effort)
`Ewilost` pool elimination via Boyd's probabilistic update (physics change);
`ChildInfo::count_wi` removal (SWS doc §3.3); `compute reduce` weighting
semantics decision (SWPM plan 2.5); SWS+SWPM composition (per-species
initial weights for SWPM); Kokkos ports. Update
`doc/sws_refactor_analysis.md` + `examples/swpm/REFACTOR_PLAN.md` to point
at the unified architecture, and document the mode model in the doc pages
(`doc/species.txt`, `doc/collide_modify.txt`, `doc/global.txt` cross-refs).

## Critical files

- `src/particle.h`, `src/particle.cpp` — WeightMode enum, `pweight`,
  `pweight_vector`, `index_sweight`, `setup_weighting()`; delete
  `weightflag`/`stochastic_weights()`
- `src/collide.h`, `src/collide.cpp` — member-block merge, enum
  consolidation, unified dispatch, `_SWS` clone deletion, guarded call
  sites only
- `src/collide_sws.cpp` (NEW) — all SWS method bodies (multi-file-class
  idiom, mirrors collide_swpm.cpp; no own header, no new class)
- `src/collide_vss.cpp` / `.h` — neutral kernel params (`np_eff`,
  `wscale`, multiplicity out-params) + guarded one-line seam calls;
  SWPM's acceptance scaling
- `src/collide_swpm.cpp`, `src/collide_reduce.cpp`,
  `src/fix_stochastic_weight.*` — arrive via merge; de-templating only
- `src/update.cpp` — `setup_weighting()` call site
- Computes (pattern repeated): `src/compute_grid.cpp`,
  `src/compute_surf.cpp`, `src/compute_temp.cpp` are representative; also
  thermal/eflux/pflux/sonine/tvib grid, boundary, isurf_grid, vmom_grid,
  reduce
- Tests: `examples/sws/*`, `examples/swpm/*`,
  `cmake/common/set/sparta_cmake_defaults.cmake` (both suites registered)

## Verification (end-to-end)

1. Per-step gate: both regression suites via
   `python3 tools/testing/regression.py mpi_1 <spa> examples/sws` (and
   `examples/swpm`), plus `verify_sws.py` and `verify_swpm.py` (serial; MPI
   where available).
2. Cross-checks that must hold at the end:
   - `in.sws0.box` (weights present, SWS off) byte-identical to pre-merge
     gold — proves the accessor is a no-op for unweighted runs.
   - `in.swpm.periodic` (SWPM neutral thresholds) byte-identical — proves
     the merge didn't perturb SWPM.
   - New negative test: input enabling two schemes at once errors with the
     unified message (add to both verify scripts).
   - A/B vs master binary on untouched examples (`examples/collide`,
     `examples/chem`, `examples/emit`, `examples/circle`) — stats
     byte-identical (site-1 cell weighting untouched).
3. Final acceptance: the four `_SWS` loops + six `_SWS` kernels are
   deleted (~2,000 lines) with all SWS bodies isolated in
   `src/collide_sws.cpp`; base `collide.cpp`/`collide_vss.cpp` carry only
   guarded one-line call sites and neutral kernel parameters; each
   compute's weighting is the one-line `pweight` accessor; one exclusion
   guard; both suites green (with only Step-5's documented re-bless).
