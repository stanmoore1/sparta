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
SWS fix (re-bless); SWS/SWPM/cell weighting stay **mutually exclusive**;
separate implementation files per scheme; minimal changes to base collide.

**Review**: this plan was independently reviewed
(`doc/sws_swpm_unification_plan_review.md`, commit `e44408a6` on
`claude/swpm-review-tests-b8jf0s`). The review verified the kernel seams
against both implementations and endorsed the architecture; its required
amendments are folded in below and marked **[review §N]**: the
`tally_weights()` triple accessor for surf/boundary tallies (§2), the
pinned count-keyword semantics (§3.1), the two-touch scatter seam (§3.2),
the explicit out-of-scope list for emission/creation files (§3.3), and the
`ewilost_cell` lifecycle notes (§3.4). It also endorsed merge-first
ordering over de-duplicate-first (§4) and the guarded-one-liners trade
against the zero-base-edit alternative (keeping the clones relocated would
permanently duplicate ~1,700 drift-prone lines).

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

The grid/global computes' copied weighting blocks collapse to `double w =
particle->pweight(i);` with the multiplication order of each tally expression
preserved exactly (bit-exactness). Collision hot paths do NOT call
`pweight(i)` per pair: SWS keeps passing derived per-species quantities down;
SWPM hoists via `pweight_vector()` (replaces its per-pair
`edvec[ewhich[...]]` re-fetch and the `stochastic_weights()` per-call string
lookup). The dead `Particle::weightflag` / `else if (weightflag)` fallback in
every SWPM compute is deleted.

**[review §2] Second accessor for the surf/boundary tally triple.** Three
files — `compute_boundary.cpp`, `compute_surf.cpp`,
`compute_isurf_grid.cpp` — do NOT reduce to `pweight(i)`: their tallies
weight three participants `(worig, wi, wj)` per surface/boundary event, and
`iorig` is a stack copy of the pre-interaction particle that an index-based
accessor cannot address. The two schemes source the triple differently but
use it in structurally identical expressions
(`weight * (ierot*wi + jerot*wj - iorig->erot*worig)`), so the unifying
abstraction there is a triple accessor:

```cpp
// weight triple for surface/boundary tally callbacks
void Particle::tally_weights(const OnePart *iorig, const OnePart *ip,
                             const OnePart *jp,
                             double &worig, double &wi, double &wj);
// WEIGHT_SPECIES : specwt by each particle's species (incl. the stack
//                  copy; exact even when a surface reaction changes species)
// WEIGHT_PARTICLE: index-based weights for ip/jp; worig = average of the
//                  outgoing particles' weights (valid under SWPM's
//                  no-boundary-splitting assumption)
// default        : 1.0 / 1.0 / 1.0
```

These three files are also where the hardest Step-1 merge conflicts live
(both branches rewrote the same expressions); `tally_weights()` is their
prescribed target state in Steps 1–2.

**[review §3.1] Count-keyword semantics, pinned now.** Both branches
already agree `n` = raw simulator-particle count (SWS kept `COUNT` raw and
added `COUNT_WI`; SWPM moved raw to `SIMCOUNT` and made `COUNT` weighted).
Decision: internal accumulators are one raw (`COUNT`) + one weighted
(`COUNT_WI`, = Σ pweight); SWPM's `SIMCOUNT` duplicate is dropped in favor
of this pair. User-facing: `n` = raw under all schemes; the single
weighted-count keyword is **`sumwi`** (already shipped in the SWS decks and
`doc/compute_grid.txt`; under SWPM it returns the summed per-particle
weights — one name, both schemes); `nrho` and the mass/energy tallies
normalize by the weighted count. Upstream `nwt` in the surf computes keeps
its existing cell-weight meaning, unchanged. Any SWPM deck pinning a
weighted-count column must be updated/re-blessed once at Step 2 — flagged,
not silent.

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
  neither scheme can silently run under Kokkos — this also replaces the
  misplaced exclusivity check currently inside
  `Particle::stochastic_weights()` on the SWPM branch
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
   baseline passes `np_eff == np`. (Verified by the review against both
   implementations: the three attempt variants differ ONLY in this
   pair-count factor, and SWPM's `fnum` scaling by
   `max_stochastic_weight*(1+pre_wtf*wtf)` folds into the same parameter
   since nattempt is linear in it.) Same for
   `test_collision(..., double wscale = 1.0)`: SWSmax passes
   `MAX(w_i,w_j)/maxwi`, SWPM passes `MAX(isw,jsw)/max_stochastic_weight`
   (shape-identical, review-verified), everyone else 1.0 — one seam, no
   more two-branch contention on that line. The per-cell helper returns
   `count_wi` separately so `recomb_density = count_wi*fnum/volume` keeps
   working (baseline recovered since `count_wi == np` when weights off).
2. **Guarded calls at the kernel seams; bodies in the SWS file.**
   `SCATTER_TwoBodyScattering` and `EEXCHANGE_NonReactingEDisposal` get the
   split-merge treatment as guarded seam calls — **[review §3.2] honestly
   two touches per kernel, not one**: a 2–3-line guarded capture of the
   pre-collision velocities/energies at the top (needed before the
   scattering draw) plus `if (phi < 1.0) sws_scatter_merge(...)` /
   `sws_energy_merge(...)` at the bottom. The ~30-line blend + Ewilost
   bookkeeping bodies live in `collide_sws.cpp`, not in the base kernel.
   With weights off `phi == 1.0` and the guards are dead-predictable.
   Delete the `_SWS` kernel clones. `perform_collision` gains
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
   loop, and `collide_vss.cpp` only the neutral parameters + the guarded
   seam touches. The four `_SWS` loop clones (~1,700 lines) are deleted;
   the baseline loops grow by ~10 lines each. (The review endorsed this
   trade explicitly against the zero-base-edit alternative of relocating
   the clones, which would permanently duplicate the loop code.)
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
  parameters). **[review §3.5, optional]** group each scheme's members into
  a small named struct (`SWSState`, `SWPMState`) to keep the base header
  near-upstream-shaped — cosmetic, do only if it doesn't disturb the
  bit-exact gates.

### D. Explicitly NOT unified (scheme-specific by nature) [review §3.3]

The emission/creation surface has no cross-scheme counterpart and is
deliberately left per-scheme — this is a decision, not an oversight:

- **SWS**: `create_particles.cpp` (np scaled by Σ f_i/w_i), `mixture.cpp/.h`
  (weighted `cummulative`), `fix_emit_face.cpp/.h`,
  `fix_emit_face_file.cpp/.h`, `fix_emit_surf.cpp/.h` (per-species targets
  scaled by `1/specwt`; subsonic `count_wi`). These stay as reviewed on the
  SWS branch.
- **SWPM**: `fix_stochastic_weight.*` initializes new particles' weights to
  1.0 via `update_custom()`. Stays as reviewed on the SWPM branch.
- **Cell weighting**: `pre_weight()`/`post_weight()` clone/delete —
  untouched master code.

Site-1 normalization (cell weighting) in the computes is likewise untouched.

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
(Status at execution: SWS suite verified green 7/7 on the current binary;
SWPM review-branch binary built in the scratch worktree; SWPM suite baseline
run still to do.)

### Step 1 — merge SWPM into the SWS branch [bit-exact]
Merge `origin/claude/swpm-review-tests-b8jf0s` into
`claude/nagoya-code-review-l1e63d`. Resolve conflicts to a
side-by-side state: SWS loops still cloned, SWPM loop separate, both
features' members present, dispatch guarded so at most one runs.
Known conflict surface (from a dry-run merge): `collide.h`, `collide.cpp`,
`compute_boundary/grid/isurf_grid/sonine_grid/surf/thermal_grid/tvib_grid.cpp`,
`particle.h`, `cmake/common/set/sparta_cmake_defaults.cmake`.
- `src/collide.h` / `src/collide.cpp`: reconcile member block, dispatch tree,
  `modify_params`; consolidate duplicated enums here.
- `src/collide_vss.cpp`: keep SWS `_SWS` overloads AND SWPM's in-place
  `vremax_init`/`test_collision` weighting (guarded by
  `stochastic_weight_flag`); note the review branch already removed SWPM's
  bad `omega==1` override — keep the reviewed version.
- `src/particle.h/.cpp`: both additions are adjacent; keep both. Drop
  SWPM's write-only `OnePartRestart::weight` (already dropped on review
  branch).
- `src/compute_boundary.cpp`, `src/compute_surf.cpp`,
  `src/compute_isurf_grid.cpp` [review §2]: the hardest conflicts — both
  branches rewrote the same `(worig,wi,wj)` tally expressions. Resolve
  toward the `tally_weights()` target state: keep ONE copy of each
  expression with the triple sourced per active scheme (an if/else on the
  scheme flags at Step 1; replaced by the accessor in Step 2). Do not
  improvise two parallel expression sets.
- `cmake/common/set/sparta_cmake_defaults.cmake`: register BOTH suites
  ("sws" and "swpm") and both KOKKOS-exact exclusion lists.
- Temporary ad-hoc exclusion guard (SWS + SWPM together → error) until
  Step 3.
Gate: both suites green, `in.sws0.box` and `in.swpm.periodic` (each feature
off/neutral) byte-identical to their gold logs.

### Step 2 — unified accessors; convert all computes [bit-exact]
Add `weight_mode` (derived inline from existing flags for now),
`index_sweight`, `pweight()`, `pweight_vector()`, and `tally_weights()`
[review §2] to `src/particle.h/.cpp`.
Convert one compute per commit, preserving multiplication order:
- via `pweight(i)`: `compute_grid.cpp`, `compute_thermal_grid.cpp`,
  `compute_eflux_grid.cpp`, `compute_pflux_grid.cpp`,
  `compute_sonine_grid.cpp`, `compute_tvib_grid.cpp`,
  `compute_vmom_grid.cpp`, `compute_reduce.cpp`.
- via `tally_weights(iorig,ip,jp,...)` [review §2]: `compute_boundary.cpp`,
  `compute_surf.cpp`, `compute_isurf_grid.cpp`.
Then delete `Particle::weightflag` and the `stochastic_weights()`
string-lookup accessor (SWPM plan steps 2.1–2.3). Apply the pinned
count-keyword semantics from Architecture §A [review §3.1]: COUNT raw +
COUNT_WI weighted, drop SIMCOUNT, `n` raw everywhere, `sumwi` = the
weighted count under both schemes; update/re-bless any SWPM deck pinning a
weighted-count column (one flagged exception to this step's bit-exactness).

### Step 3 — single mode/exclusion model [bit-exact]
Add `Particle::setup_weighting()`; call from `Update::setup()`
(`src/update.cpp`). Replace scattered pairwise guards (particle.cpp KOKKOS
check, collide.cpp SWPM checks, SWPM/cell exclusivity — including the one
misplaced inside `stochastic_weights()` — and the Step-1 temp guard).
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
  guarded capture + guarded merge at the `SCATTER_*`/`EEXCHANGE_*` seams
  (two touches per kernel [review §3.2]). Baseline callers pass neutral
  values — bit-exact, both suites green, `_SWS` kernels still present but
  now thin wrappers.
- 4b. One loop family per commit (one, group, ambipolar-one,
  ambipolar-group): add the per-cell `sws_cell_weights()` call and the
  `if (sws)` product-bookkeeping call to the baseline loop, switch
  dispatch to it, delete the `_SWS` clone.
- 4c. Delete the six `_SWS` VSS kernels + their `Collide` pure-virtuals;
  de-template the SWPM loop; delete octree stub if not already gone.
  After 4c, `git grep SWS src/collide.cpp src/collide_vss.cpp` should
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
- `Ewilost` pool elimination via Boyd's probabilistic update (physics
  change; validate against relaxation benchmarks).
- **[review §3.4] `ewilost_cell` lifecycle**: the pooled split-merge energy
  IS migrated on load balance (packed/unpacked in
  `pack_grid_one`/`unpack_grid_one` alongside vremax/remain) but is
  **dropped/zeroed on grid adaptation** (`Collide::adapt_grid()` re-inits
  new cells) — a small documented energy leak on refine/coarsen that the
  Boyd-update elimination would remove entirely. Related: Ewilost is
  re-injected only into collisions where BOTH partners carry the max
  weight (`setup_collision_SWS`), so in strongly mixed cells pooled energy
  can sit unreturned for many steps. Record both in the doc pages until
  eliminated.
- `ChildInfo::count_wi` removal (SWS doc §3.3).
- `compute reduce` weighting semantics decision (SWPM plan 2.5) — current
  SWPM behavior weights positions under SUM and divides AVE by raw count;
  needs a deliberate extensive-only/weight-normalized decision.
- SWS+SWPM composition (per-species initial weights for SWPM); Kokkos
  ports of both schemes.
- Update `doc/sws_refactor_analysis.md` + `examples/swpm/REFACTOR_PLAN.md`
  to point at the unified architecture, and document the mode model in the
  doc pages (`doc/species.txt`, `doc/collide_modify.txt`, `doc/global.txt`
  cross-refs).

## Critical files

- `src/particle.h`, `src/particle.cpp` — WeightMode enum, `pweight`,
  `pweight_vector`, `tally_weights` [review §2], `index_sweight`,
  `setup_weighting()`; delete `weightflag`/`stochastic_weights()`
- `src/collide.h`, `src/collide.cpp` — member-block merge, enum
  consolidation, unified dispatch, `_SWS` clone deletion, guarded call
  sites only
- `src/collide_sws.cpp` (NEW) — all SWS method bodies (multi-file-class
  idiom, mirrors collide_swpm.cpp; no own header, no new class)
- `src/collide_vss.cpp` / `.h` — neutral kernel params (`np_eff`,
  `wscale`, multiplicity out-params) + guarded seam touches;
  SWPM's acceptance scaling
- `src/collide_swpm.cpp`, `src/collide_reduce.cpp`,
  `src/fix_stochastic_weight.*` — arrive via merge; de-templating only
- `src/update.cpp` — `setup_weighting()` call site
- Computes (pattern repeated): `pweight` path — `src/compute_grid.cpp`,
  `src/compute_temp.cpp` representative, plus thermal/eflux/pflux/sonine/
  tvib/vmom grid + reduce; `tally_weights` path — `src/compute_surf.cpp`,
  `src/compute_boundary.cpp`, `src/compute_isurf_grid.cpp`
- Emission/creation files: explicitly out of scope (Architecture §D)
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
   - SWPM conservation invariants (total weight, weighted momentum/energy
     through split + all three reduction schemes, binary and weight
     grouping) and its guardrail error conditions — all pinned by
     `verify_swpm.py` [review §5].
   - New negative test: input enabling two schemes at once errors with the
     unified message (add to both verify scripts).
   - A/B vs master binary on untouched examples (`examples/collide`,
     `examples/chem`, `examples/emit`, `examples/circle`) — stats
     byte-identical (site-1 cell weighting untouched).
3. Final acceptance: the four `_SWS` loops + six `_SWS` kernels are
   deleted (~2,000 lines) with all SWS bodies isolated in
   `src/collide_sws.cpp`; base `collide.cpp`/`collide_vss.cpp` carry only
   guarded call sites and neutral kernel parameters; each compute's
   weighting is the `pweight`/`tally_weights` accessor; one exclusion
   guard; both suites green (with only Step-5's and the count-keyword
   deck's documented re-blesses).
