# Review of `kk_claude_porting`: Kokkos ports vs CPU implementations

Scope: full diff of `origin/kk_claude_porting` vs `origin/master` (~9,300 added lines,
120 files). Every new Kokkos port was compared line-by-line against its CPU
counterpart, with emphasis on physics faithfulness, RNG-sequence fidelity
(SPARTA_KOKKOS_EXACT), sync/modify discipline, host pointers reaching device code,
tally atomicity, and hot loops left on the host.

Overall assessment: the ports are careful and largely faithful — the surface-collision
physics (CLL/TD/impulsive/adiabatic), the adsorb GS reaction math, the QK/TCE-QK
reaction models, the multigroup ambipolar collision logic, and the subsonic emission
kernels all reproduce the CPU formulas and RNG draw ordering, and the ScatterView /
KKCopy / DualView patterns follow the established master conventions. The issues below
are concentrated in cross-cutting glue: dynamic-temperature indexing, the
retry/restore path, post-processing sync for direct consumers, and dispatch plumbing.

---

## Critical

### C1. CPU ambipolar + gas tally dispatch bug: no collisions at all
`src/collide.cpp:398-405` — in the `ambiflag` branch both arms test `!ngas_tally`:

```cpp
} else if (ambiflag) {
    if (!ngas_tally) {
      ...collisions_one_ambipolar<0>();
    } else if (!ngas_tally) {        // should be (ngas_tally)
      ...collisions_one_ambipolar<1>();
```

With an ambipolar mixture and an active `compute gas/collision/grid` or
`gas/reaction/grid`, the CPU silently performs **zero collisions** on every tally
step. The Kokkos port fixed its own copy of this dispatch
(`src/KOKKOS/collide_vss_kokkos.cpp:480-492` correctly uses `else if (ngas_tally)`),
so CPU and GPU now give different physics for the same input. One-word fix on the
CPU side.

### C2. CUSTOM per-surf temperature indexed by global custom index, not `ewhich` slot (issue #641 class)
`src/KOKKOS/surf_collide_cll_kokkos.cpp:213-219`,
`surf_collide_td_kokkos.cpp:213-216`, `surf_collide_impulsive_kokkos.cpp:213-216` —
the `tmode == CUSTOM` branch of `dynamic()` does:

```cpp
h_edvec_local[tindex_custom].k_view.sync_device();
d_t_persurf = h_edvec_local[tindex_custom].k_view.view_device();
```

but `edvec_local`/`k_edvec_local` are indexed by the per-type slot
`surf->ewhich[tindex_custom]` (see `Surf::add_custom`). This branch fixes exactly
this bug in the base class (`src/surf_collide.cpp:230-237`) and in
`surf_collide_diffuse_kokkos.cpp:214-222` (which even carries an explanatory
comment), but the fix was not propagated to the three new ports. With any other
custom surf attribute defined, they sync/read the wrong vector or index past the end
of `k_edvec_local` → wrong surface temperatures or OOB on device. Works only when
the temperature vector happens to be custom index 0, which is why simple tests pass.

### C3. `compute isurf/grid/kk` returns all zeros to direct consumers
`src/KOKKOS/compute_isurf_grid_kokkos.{h,cpp}` — device tallies reach the host only
via `tallyinfo()`. The sibling ports override the virtual post-process to force that
sync (`compute_react_isurf_grid_kokkos.cpp:181-187`,
`compute_react_surf_kokkos.cpp:173-179`: `if (combined) return; tallyinfo(dummy);
Base::post_process...();`), but `ComputeISurfGridKokkos` has **no
`post_process_isurf_grid()` override**. Consumers that call it directly — `fix
ablate` (`fix_ablate.cpp:941`), `dump grid`, `dump image`, grid-style variables,
`compute reduce`, `fix ave/histo`, the library interface — collate with host
`ntally == 0` (reset by `clear()` every step) and silently produce all-zero
per-grid values. Only the `fix ave/grid` path works (it calls `tallyinfo()` itself).
Fix: copy the react variant's 5-line override.

### C4. `SurfReactAdsorbKokkos::restore()` corrupts state on the react/retry path
`src/KOKKOS/surf_react_adsorb_kokkos.cpp:465-484` — when the move kernel retries
after a mid-kernel particle realloc, `restore()` only restores the RanKnuth:

```cpp
void SurfReactAdsorbKokkos::restore() {
#ifdef SPARTA_KOKKOS_EXACT
  memcpy(random,random_backup,sizeof(RanKnuth));
#endif
}
```

The established pattern zeroes tallies (`SurfReactGlobalKokkos::restore()` does
`Kokkos::deep_copy(d_scalars,0)`), and adsorb has additional persistent state:
`d_species_delta` and `d_mark` accumulate **across steps** until the next `nsync`
sync, and `react_kokkos` increments them before the overflow/`d_retry` check. Every
retry therefore double-counts `nsingle`/`tally_single` and double-applies
adsorption/desorption deltas at the next sync — wrong coverage → wrong subsequent
reaction probabilities. `backup()` must snapshot `d_species_delta`/`d_mark`;
`restore()` must copy them back and zero `d_scalars`.

### C5. Gas tallies double-counted on the collide retry path
`src/KOKKOS/collide_vss_kokkos.cpp` — retry loops in `collisions_one` (~677-749) and
`collisions_one_ambipolar` (~1525-1600). `restore()` rolls back particles, plist,
`d_vremax`, `d_remain`, and `ReactBirdKokkos::d_tally_reactions`, but **not** the
gas-tally computes' `d_vector_grid`/`d_array_grid`. Tally events from the aborted
pass survive and are added again on the re-run → inflated `gas/collision/grid` /
`gas/reaction/grid` values whenever GASTALLY=1 and a retry occurs. Fix: re-invoke
`clear()` on the active gas-tally computes (or snapshot/restore their arrays) in the
`h_retry()` branch, symmetric with `d_tally_reactions`.

---

## Bugs

### B1. Missing `isr >= 0` guard in TD / impulsive / adiabatic device reaction block
`surf_collide_td_kokkos.h:129`, `surf_collide_impulsive_kokkos.h:129`,
`surf_collide_adiabatic_kokkos.h:128` use `if (REACT)` where CLL correctly has
`if (REACT && isr >= 0)` (`surf_collide_cll_kokkos.h:131`) and the CPU always guards
with `if (isr >= 0)`. With reactions defined globally but a surf/face without a
reaction model (`isr == -1`, a legal mixed assignment), the device reads
`sr_type_list[-1]`/`sr_map[-1]` and may invoke a never-`copy()`d KKCopy object — UB.
(The same hazard pre-exists in master's diffuse/specular/piston ports; the branch
demonstrably knows the fix and applied it only to CLL.)

### B2. `ComputeISurfGridKokkos::clear()` lacks the resize guard its react twin has
`compute_isurf_grid_kokkos.cpp:99-109` vs `compute_react_isurf_grid_kokkos.cpp:97-109`.
The react variants re-check `surf->nlocal + surf->nghost > nsurf_tally_alloc` every
tally step; the isurf variant sizes its device views only in `init_normflux()`.
Mid-run ablation can increase the implicit-surf count while
`ComputeISurfGrid::reallocate()` early-returns (grid size unchanged), so
`surf_tally_kk` then writes `d_surf2tally`/`d_array_surf_tally` (and reads
`d_normflux`) out of bounds on device. Copy the react variants' guard and
`nsurf_tally_alloc` bookkeeping. (Related: initialize `nsurf_tally_alloc = 0` in the
react variants' constructors — currently read uninitialized on the first `clear()`.)

### B3. Style-string dispatch rejects explicitly suffixed compute styles
`src/KOKKOS/update_kokkos.cpp` `tally_set()` (~2100-2160) and the post-tally loop
(~863-882); same pattern in `collide_vss_kokkos.cpp:548-594` (`setup_gas_tally`).
Dispatch is `strcmp(style,"isurf/grid")` etc., but `Compute::style` stores the name
as typed. `-sf kk` works, but a user explicitly writing `compute 1 isurf/grid/kk ...`
(the styles are registered under those names) falls to the generic branch and aborts
with the misleading "Kokkos does not (yet) support compute surf/collision/tally...".
Dispatch by `dynamic_cast` (most-derived first) or also match the `/kk` names.

### B4. `surf_react adsorb/kk` dispatchable only from `cll/kk`; its presence breaks every other collide style
Only `surf_collide_cll_kokkos.cpp:258-263` recognizes style "adsorb" (sr_type 2).
Every other Kokkos collide model's `pre_collide()` loops over **all** `surf->sr` and
hits `error->all(FLERR,"Unknown Kokkos surface reaction method")` — so defining one
adsorb model anywhere forbids diffuse/specular/td/impulsive/adiabatic in the same
run, with a misleading message. Either wire adsorb into the remaining styles or emit
an explicit "surf_react adsorb not yet supported with this Kokkos surf_collide
style". Related dead code: the new `wrapper_kokkos()` in
`surf_collide_diffuse_kokkos.h:204-224` and `surf_collide_specular_kokkos.h:205-217`
is never called (adsorb uses its own device scatter replicas), and the specular one
drops the CPU wrapper's persistent `noslip_flag = flags[0]` assignment.

### B5. Adsorb: no `grid_changed()` override — stale device state after load balance / adaptation
`surf_react_adsorb_kokkos.cpp:317` fixes `nstate_` at `init()`, but the CPU base
(`surf_react_adsorb.cpp:1196-1243`) reallocates `species_delta`/`mark` to the new
`surf->nlocal + surf->nghost` mid-run. After a grid change, `pre_react()`/
`tally_update()`/`react_kokkos` use the stale count → host OOB reads or device OOB
indexing. Needs a `grid_changed()` override that calls the base and rebuilds device
state (safe: the base already restricts grid changes to sync steps).

### B6. Adsorb: CLL cmodel eccentricity silently dropped on device
`surf_react_adsorb_kokkos.cpp:41` copies only 5 coeffs for CLL cmodels, but
`cll_scatter` reads `cf[5]` as eccen (`surf_react_adsorb_kokkos.h:437`) — always 0,
so the `pflag` branch runs wrong physics with a different RNG draw count. Note the
CPU has a companion pre-existing bug: `readfile_gs` allocates `ncoeffs = 5` for cll
while `SurfCollideCLL::flags_and_coeffs` can write `coeffs[5]` — a heap overflow.
Carry 6 coeffs on both sides.

### B7. PS (on-surface) reaction tallies lost/garbled with `compute react/surf/kk`
`PS_react()` tallies on host via the base `ComputeReactSurf::surf_tally()`, but
(a) `ComputeReactSurfKokkos::clear()` (`compute_react_surf_kokkos.cpp:93-103`) never
clears the base `hash`, so stale surfID→itally entries persist across steps;
(b) the host path uses a dense-tally layout while the device path is isurf-indexed —
incompatible in the same `k_array_surf_tally`; (c) `tallyinfo()`'s `sync_host()`
(device side marked modified) overwrites host-written PS tallies. Net: PS reactions
silently vanish from per-surf reaction tallies. FACE-mode boundary tallies have the
analogous concern.

### B8. Subsonic EXACT mode: sampling order inverts when particles arrive sorted
`fix_emit_face_kokkos.cpp:729-738` (and the same block in
`fix_emit_surf_kokkos.cpp`) iterate `for (n = np-1; n >= 0; n--)` on the premise
that CPU linked-list traversal is decreasing-index. That holds only when the CPU
rebuilds via its private `subsonic_sort()` (`if (!particle->sorted)`). With a
collide style active (the common subsonic case), `particle->sorted == 1` at emit
time and the CPU traverses `Particle::sort()` lists, which are deliberately built
for **increasing**-index traversal (`particle.cpp:437-446`); Kokkos meanwhile skips
`sort_kokkos()` (`sorted_kk==1`) and walks `d_plist` in decreasing order — the
opposite. For cells with ≥3 particles the moment sums are order-dependent, so
`nrho`/`temp_thermal`/`vstream` are not bit-identical, defeating commit c510487's
goal exactly in the collide-without-chemistry configuration. Fix: choose loop
direction based on whether particles were already sorted at emit entry.

### B9. Subsonic emit sets the global `sorted_kk` flag, unlike CPU's private sort
`fix_emit_face_kokkos.cpp:639-643`, `fix_emit_surf_kokkos.cpp:888-892` — CPU
`subsonic_sort()` builds private lists and does **not** set `particle->sorted`; the
Kokkos version calls `ParticleKokkos::sort_kokkos()`, which sets `sorted_kk = 1`,
then inserts new particles without clearing it. With two subsonic emit fixes and no
collide, CPU fix #2 resamples including fix #1's freshly emitted particles; Kokkos
fix #2 sees `sorted_kk==1` and samples the stale pre-emission `d_plist` — a real
statistical divergence. Either sort into local views or clear `sorted_kk` after
insertion.

### B10. `fix ave/grid` PERGRIDSURF keeps `kokkos_flag = 1` with empty device views
`fix_ave_grid_kokkos.cpp:49-61` — the PERGRIDSURF branch early-returns from the
ctor, leaving `d_vector_grid`/`d_array_grid` zero-length, yet `kokkos_flag = 1`
stands. Device consumers gate only on that flag (e.g.
`compute_lambda_grid_kokkos.cpp:197-215`) and would dereference a 0-length view in a
kernel — silent OOB instead of a clean error. Set `kokkos_flag = 0` in that branch
(it runs entirely on host anyway).

### B11. Inherited `tallyinfo()` compression drops tallies in the dense regime
`compute_isurf_grid_kokkos.cpp:168-181`, `compute_react_isurf_grid_kokkos.cpp:158-170`,
`compute_react_surf_kokkos.cpp:151-163` — copied verbatim from master
`ComputeSurfKokkos::tallyinfo()`. The guard
`while (h_surf2tally[istart] != -1 && istart < nsurf-2) istart++;` wrongly caps
`istart` at `nsurf-2`: when all (or all-but-one) local+ghost surfs are tallied, one
or two tallies are destroyed/dropped. For implicit surfs, every-surf-hit-per-step is
a realistic regime. Not a regression (master has it), but now replicated 3×; fix in
all four places: `while (istart < nsurf && h_surf2tally[istart] != -1) istart++;`
with an `istart >= iend` termination check.

### B12. Adsorb device path drops CPU's mode/isurf sanity checks
CPU `react()` errors on "adsorb surf used with box faces" and vice-versa
(`surf_react_adsorb.cpp:641-644`); on device a SURF-mode model attached to a box
face yields `idx < 0` and indexes `d_area[idx]` OOB. Validate on host at init /
pre_collide dispatch.

---

## Performance

### P1. Adsorb: full per-surf state H2D copy + mirror allocations every step
`surf_react_adsorb_kokkos.cpp:373-388` — `pre_react()` allocates four fresh mirrors
and copies `total_state`/`area`/`weight`/`species_state` H2D every step (and once per
CLL instance per step). State changes only on sync steps; `area`/`weight` are static
between grid changes. Cache the mirrors and gate the copies.

### P2. Adsorb: unconditional full-particle D2H+H2D round trip every step with `ps`/`gsps`
`surf_react_adsorb_kokkos.cpp:410-443` — `tally_update()` does
`sync(Host,PARTICLE_MASK)` + `modify(Host,PARTICLE_MASK)` every step, but PS
chemistry only runs on `ntimestep % nsync == 0`. Likewise `k_species_delta`
(nstate × nspecies) and `k_mark` are synced D2H every step though consumed only on
sync steps. Gate all of it on the sync step — cheap, high-value fix.

### P3. VARSURF temperature: fresh device allocation + blocking H2D copy per recompute
`surf_collide_cll_kokkos.cpp:199-200` (same in td/impulsive):
`create_mirror_view_and_copy` every `temp/freq` interval; with `temp/freq 1` that is
per-step. Reuse a persistent buffer. (Matches the master diffuse pattern, so
consistent — but worth fixing across the family.)

### P4. Smaller items
- `d_nattempt_pair` (`collide_vss_kokkos.cpp:993-996, 1202-1205`): a persistent
  `nglocal × ngroups²` global-memory scratch used only within one kernel; a
  per-thread stack array (≤256 ints at MAXGROUP=16) would avoid the round trip.
- Gas-tally computes `post_gas_tally()` does `modify_device(); sync_host();` every
  tally step even when the only consumer is device-side `fix ave/grid/kk`.
- `ComputePropertySurfKokkos::compute_per_surf()` does a redundant device→device
  `deep_copy` plus an element-wise host copy each invocation; write kernel output
  into `k_*.view_device()` and use dual-view sync.
- `memory_usage()` in the new tally ports reports via a never-updated `maxtally` —
  underreports; cosmetic.

---

## Minor / consistency

- **KOKKOS_MAX_TOT_SURF_COLL not raised** (`update_kokkos.h:42`, check at
  `update_kokkos.cpp:558`): still 10 while 9 model types × 2 instances = 18 legal;
  safe (check precedes writes) but the cap and the "two instances of each" message
  should be updated.
- **`sr_map[n] = nprob` copy-paste bug in the *old* diffuse/specular ports**
  (`surf_collide_specular_kokkos.cpp:129`, `surf_collide_diffuse_kokkos.cpp:250`,
  "global" branch should assign `nglob`): with models defined in order prob-then-
  global, the kernel dispatches a stale device copy. Pre-exists on master; the four
  new ports get it right — fix the old files while here.
- **`src/KOKKOS/Install.sh` incomplete**: adds the four surf-collide pairs +
  `fix_emit_kokkos.h` but omits ~20 other new files (gas tally computes, react
  QK/TCEQK, adsorb, isurf/react computes, property/surf, temp/global/rescale).
  Builds are unaffected (CMake globs), but be consistent.
- **`compute property/surf/kk` intentionally diverges from CPU — CPU is the buggy
  one**: CPU `pack_v3y/z` return `p1[1]/p1[2]` instead of `p3`, `pack_area` (3D)
  crosses with an uninitialized vector, and pack loops iterate `nsown` over a
  `nchoose`-sized `cglobal` (OOB with a surf subset group). The KK port is correct;
  fix the CPU compute rather than making the port bug-compatible, and note /kk vs
  non-/kk results won't match until then.
- **React style cast safety** (`collide_vss_kokkos.cpp:620-624` etc.): the
  `(ReactTCEKokkos*) react; if (!react_kk)` NULL check is dead (C-cast), and a
  host-only react style can survive its own init when the collide style string is
  suffix-created ("vss") → object memcpy'd as the wrong type, UB. Add
  `if (react && !react->kokkos_flag) error->all(...)` in `init()`.
- **emit/surf/kk uses per-step `dt` for subsonic ntarget; CPU uses init-frozen
  member `dt`** — divergence under `fix dt/reset`. The KK behavior is arguably
  correct (face agrees on both sides); flag the CPU staleness upstream rather than
  porting the bug.
- **Adsorb EXACT-on-retry**: per-cmodel RanKnuth streams (`d_cmodel_rand`) are not
  in `backup()`/`restore()`, so post-retry scatter draws diverge from a no-retry run
  under SPARTA_KOKKOS_EXACT. Statistically harmless.
- **Adsorb `translate`/`rotate` cmodel options silently ignored on device** — the
  CPU wrapper applies tflag/rflag inside `diffuse()`/`cll()`; the device replicas
  never do and init doesn't reject them. Error at init.
- **Stale comments**: `surf_react_adsorb_kokkos.h:291-293` claims diffuse/cll/td
  scatter is deferred (it's implemented); `update_kokkos.h` partition comment omits
  `nslist_react_surf`; face `grow_task()` allocation labeled `"emit/surf:tasks"`
  (pre-existing).
- **Retry rollback of surf-tally scatter views** is not done (aborted pass
  contributions re-added on retry) — exactly master's behavior for
  `ComputeSurfKokkos`, shared by the four new ports; noted for completeness, and
  worth fixing together with C4/C5 if the retry path is touched.
- `subsonic_inflow()` re-acquires `d_species_all`/`d_cinfo` after `subsonic_grid()`
  nulls them and never releases — harmless, inconsistent with the stated cleanup.
- Distributed `property/surf/kk` relies on `k_mylines`/`k_mytris` being synced only
  at wrap time (`SurfKokkos::sync()` covers only `k_lines`/`k_tris`); fine for
  static geometry, worth a comment or explicit sync.

---

## Verified faithful / correct (spot list)

- **CLL physics & RNG** (`surf_collide_cll_kokkos.h` vs `surf_collide_cll.cpp`):
  tangent construction incl. random-tangent branch, vrm with per-surf twall,
  normal/tangential CLL sampling, pflag rejection loop, trflag translate/rotate,
  rotational (dof 2 / >2) and vibrational (DISCRETE dof==2 incl. `evib_star`,
  SMOOTH, dof>2) modes — RNG draw ordering matches CPU line-for-line, with
  `gaussian()` correctly mapped through `normal()`/RandPoolWrap under EXACT.
- **TD & impulsive**: barrier/initenergy/bond energy handling, twall_rot/twall_vib,
  device `erot()/evib()` replicas identical to `Particle::erot/evib` (DISCRETE
  quantization, 10 kT rejection loop); impulsive theta/phi rejection loops incl.
  `step`/`double`, softsphere vs tempvar with dynamic per-surf T, intenergy split
  replicated down to CPU's quirky branch structure. Making `v_f_avg` a local is a
  thread-safety improvement with identical results.
- **Adsorb GS chemistry**: reaction probability computation (coverage/Kisliuk
  terms, ER/CI/DA/LH branches, stoichiometry multipliers, even the CPU's sticky
  `coeff_val` quirk), single-draw cumulative selection, per-branch RNG sequence,
  outcome branches (DISSOCIATION/EXCHANGE/RECOMBINATION/DA/LH1/ER/CI, velreset
  semantics), atomic `species_delta`/tally updates, host PS split with two-way sync
  on sync steps, particle growth via `d_nlocal` fetch-add + `d_retry` — all match
  CPU (modulo C4/B5/B6 above).
- **Multigroup ambipolar & tallies** (`collide_vss_kokkos`): group attempt formula
  (0.5·n(n−1) same-group, un-halved cross-group, remain carry), two-phase RNG
  ordering, per-group vremax with the flipped-index electron asymmetry, electron
  build/write-back in plist order, e/e exclusion; `gas_tally_kk` matches CPU
  `gas_tally` exactly (filters, group masks, EVERY/SELECT columns); reaction indices
  preserved via `d_list[i]+1`; one-thread-per-cell kernels make non-atomic
  `(icell,·)` tally writes race-free; per-step `setup_gas_tally()` re-partition
  can't go stale.
- **React QK/TCEQK**: line-by-line match incl. deliberate bug-compatibility
  (`react_prob` carry-over between list entries in QK, per-reaction reset in TCEQK),
  maxlev/limlev, exothermic exchange `mspec` + `coeff[6]` omega, MAXCOEFF=7 covers
  QK's precomputed coeffs, RNG-per-evaluated-reaction order preserved.
- **Subsonic emission kernels**: `mol_inflow_kokkos` expression-identical;
  `subsonic_inflow`/`subsonic_grid` moment accumulation, PTBOTH/PONLY, np≤1 and
  np==0 fallbacks, pressure relaxation, vstream correction, per-species vscale from
  recomputed T — all term-for-term vs CPU; no RNG consumed in sampling so EXACT
  stream alignment holds (modulo B8); `d_tempmax` via `atomic_max`; the
  `grow_task()` sync-before-modify reorder is a genuine fix.
- **Wiring**: sc_type 5-8 registered identically at all three collision sites +
  backup/restore; tally KKCopy lists bounded and pre_surf_tally called before copy;
  post_* routed to original host objects; `fix temp/global/rescale/kk` is a faithful
  device port with correct sync/modify and uniform MPI collectives.
- **CPU-side misc**: dump ubuf changes fix a real INT/cellID printing bug;
  `create_isurf` tolerance+clamp sane; `region_*` atof→numeric adds validation;
  python loader fallback correct; `SPARTA_KOKKOS_EXACT` cmake option and CI ctest
  wiring sane; base-class `copy`-guarded destructors and `kokkos_random()` hook
  correct.

---

## Suggested fix order

1. **C1** — one-word CPU fix (`else if (ngas_tally)`), physics-breaking.
2. **C2** — one-line `ewhich` fix ×3, same change already on the branch for diffuse.
3. **C3** — 5-line `post_process_isurf_grid()` override, silent-zeros in supported usage.
4. **C4 + C5** — retry-path restore for adsorb state and gas-tally arrays.
5. **B1, B2** — small guards preventing device OOB/UB.
6. **B3, B4, B10** — dispatch/UX correctness.
7. **P1, P2** — cheap, high-value adsorb performance gating.
8. Remaining B/P/minor items as cleanup, incl. back-porting fixes to the older
   diffuse/specular ports (`sr_map`, `isr >= 0`, tallyinfo compression).
