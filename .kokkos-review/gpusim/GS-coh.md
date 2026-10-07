# GS-coh: host/device coherence fixes G1, R8/G2, R2/G3, R1, U-1, R5/R5b (A' vs C')

Setup: W=$S/gpusim/GS-coh. src_A = worktree of e48b9a88 + tool 5a2c2a52 merged (conflicts collide_vss_kokkos.h,
grid_kokkos.cpp resolved as gpusim-B: HEAD logic, SPARTA_NS::DualView / SPARTA_NS::resize).
A' = $W/spa_A (DEBUG_SYNC, Serial, MPI, Release, Ninja); C' = same tree + fixes ($W/spa_C).

## Progress
- [ ] A' build
- [x] A' build ($W/spa_A), C' build ($W/spa_C, = src_A + fix.diff, incremental), stock S ($W/spa_S: C source, DEBUG_SYNC off,
  Serial, MPI) building. Fix diff: $W/fix.diff (= `git diff -- src` of /home/user/sparta).
- compile_one.sh (OpenMP, non-tool build flags) OK on all 8 changed files.

## Changes (in /home/user/sparta/src/KOKKOS)
- G1 particle_custom_kokkos.cpp ParticleKokkos::grow_custom: per grown vector `k.sync_host(); k.modify_host();` before
  grow_kokkos (resize now happens on the host and keeps its contents), then `k.sync_device()` when not prewrap (both
  halves valid again, as the device kernels that follow an in-run grow() expect). Only the vector being grown is touched.
- G1 grid_custom_kokkos.cpp GridKokkos::reallocate_custom: `sync(Host,CUSTOM_MASK); modify(Host,CUSTOM_MASK);` for all
  cases (was: prewrap only; else sync(Device)), then `sync(Device,CUSTOM_MASK)` at the end when not prewrap.
  (allocate_custom in grid/surf left alone: it only ever creates fresh views, no resize of existing data.)
- R8/G2 update_kokkos.cpp UpdateKokkos::setup prewrap branch: `surf_kk->modify(Host,CUSTOM_MASK)` after wrap_kokkos
  (SurfKokkos::modify(Host) is allowed under prewrap). Between runs the non-prewrap branch already claims
  `surf_kk->modify(Host,ALL_MASK)` (covers custom surf set / read_surf between runs) -> no change needed there.
- R1 update_kokkos.cpp UpdateKokkos::move: particle/grid sync(Device) moved to the top of the while(1) body, ahead of the
  view_device() grabs (the in-loop particle->grow() re-grabs d_particles after its own sync/resize/modify).
- R2/G3 compute_property_grid_kokkos.cpp: grid_kk->sync(Device,CELL|CINFO) before the view_device() grabs.
- U-1 fix_emit_face_kokkos.cpp / fix_emit_face_file_kokkos.cpp destructors: dead loop removed (tasks=NULL, and for
  face/file ntaskmax=0, already keep the base destructors away from the Kokkos-owned memory). fix_emit_surf_kokkos.cpp:
  already fixed in HEAD (k_tasks.clear_sync_state() before its loop, which must stay: it deletes the host-only
  path/fracarea arrays) -> unchanged.
- R5 compute_react_isurf_grid_kokkos.cpp tallyinfo: k_tally2surf/k_array_surf_tally.clear_sync_state() after the
  in-place host compaction. R5b compute_boundary_kokkos.cpp compute_array: k_array.clear_sync_state() after the host
  normalization. clear_sync_state is right here: the host copy is a per-step product, the device copy is regenerated
  and re-claimed (modify_device) next time; modify_host would make that next modify_device a concurrent claim.
  On real Kokkos both are no-ops in effect (sync_host already zeroed the flags); documentation + detector silence.

## G1 -- ParticleKokkos::grow_custom / GridKokkos::reallocate_custom device-side resize
decks: $W/decks/in.G1g (grid custom set 1.0, run, adapt_grid refine, set 3.0 via equal-style, run), in.G1g2 (2 grid
vectors, no set after adapt), in.G1p (2 particle custom vectors, run, create_particles 30000, equal-style set, run).
positive control (A', WATCH/STALE/STRICT, np1):
  G1g: `[stale] grid:dvector <- GridKokkos::reallocate_custom: host side read while device side is newer`
  G1g2: same for grid:dvector and grid:ivector; G1p: `particle:{d,i}vector <- ParticleKokkos::grow_custom` (host read
  while device newer). Trace (TRACE=grid:dvector): `resize flags=(0,0)` -> `sync_host flags=(0,1)`: the resize goes to
  the device and leaves the host a fresh mirror behind a device claim; the next GridKokkos sync(Host) copies it back.
negative control: A' stats == cpu (same binary, no -k) on all three: G1g c_gr 2500 -> 30000, G1g2 0/30000, G1p
  224000 = 7*32000. So on these decks the host is always refreshed (copy_custom/zero_custom/next reallocate sync_host)
  before host code reads it: the fault is the stale-side read of the pointer grab plus a window in which the host half
  is a zeroed mirror; on a GPU it also costs an extra D2H copy per resize. No output fault reproduced.
complete (C'): 0 watch/stale reports on G1g, G1g2, G1p; stats identical to A' and cpu.
verdict: NECESSARY (detector-shown stale-side access, latent window) + COMPLETE.
  (A' also reports R1 `grid:cells <- UpdateKokkos::move<2,0,0,0>` on G1g/G1g2; C' none.)

Targeted decks below: GS-sweep templates ($S/gpusim/GS-sweep/work/_tmpl/<ex>, shortened runs), run by $W/bin/run_ex.sh,
outputs $W/out/{A,C}.np1/<ex>.<deck>.{log,err}; variant A/C = WATCH= STALE= STALE_STRICT=1 on A'/C'.

## R8/G2 -- owned surf custom written before the first run never claimed
positive control (A'): examples/custom in.custom.circle.{set,file,file.distributed}: `[watch] surf:dvector: the host
  side was written without a claim and this sync_device has nothing to copy -- the device keeps stale data, element 0 of
  50` (4 lines incl. "suppressed"); read_restart decks in.custom.{cube.read,step.set}.restart: same report for
  `surf:iarray` and `surf:darray` (8 lines) -- this is GS-autosync follow-up 2 (surf custom arrays from read_restart).
negative control: in.custom.cube.read / in.custom.step.set (no surf custom before run 1): no surf watch report on A' or C'.
complete (C'): 0 watch reports on circle.set, circle.file, circle.file.distributed, cube.read.restart,
  step.set.restart; stats A' == C' on all of them (no device reader of the owned copies today, so no output change).
  TRACE on C' shows the new claim (`surf:dvector modify_host` in setup) ahead of the first sync_device.
remaining (not this fix, follow-up R8b): in.custom.circle.{set,file}.fix still report surf:dvector on C', but only from
  step 1000 on: `fix custom 1000 surf set ...` (non-Kokkos FixCustom::end_of_step -> Custom::set_surf host write) inside
  the run; ModifyKokkos::end_of_step claims particles only. Backtrace: SurfKokkos::sync <- UpdateKokkos::move<2,1,0,0>.
  Before the fix the report started at step 1 (pre-run set); after it, only the in-run set remains. Proposed: claim
  `((SurfKokkos*)surf)->modify(Host,CUSTOM_MASK)` at the top of ModifyKokkos::custom_surf_changed() (called by
  FixCustom::end_of_step for surf mode and by FixSurfTemp, both after host writes; FixSurfTempKokkos already claims
  the same itself). Not applied: modify_kokkos.cpp is outside this task's files. Latent (owned surf custom has no
  device reader; consumers use the _local copies, which spread_custom claims).
verdict: NECESSARY (detector-shown, latent) + COMPLETE for pre-first-run set / read_surf / read_restart; in-run fix
  custom surf set = follow-up R8b.

## R1 -- UpdateKokkos::move takes view_device() handles before the syncs
positive control (A'): adapt/in.adapt.static `[stale] grid:cells, grid:pcells <- UpdateKokkos::move<2,1,0,0>: device side
  read while host side is newer`; in.adapt.rotate also grid:sinfo; G1g/G1g2 grid:cells <- move<2,0,0,0>.
complete (C'): none of these on adapt.static, adapt.rotate, G1g, G1g2; stats A' == C'.
verdict: cosmetic/accessor-order (the handle is the same allocation after the sync) -> fixed, detector-clean.

## R2/G3 -- ComputePropertyGridKokkos reads device grid before syncing
positive control (A'): ablation/in.ablation.multi.inner.3d `[stale] grid:cinfo <- ComputePropertyGridKokkos::
  compute_per_grid_kokkos(): device side read while host side is newer`.
complete (C'): report gone; stats A' == C' (51 rows).
verdict: accessor-order -> fixed, detector-clean.

## R5 -- compute react/isurf/grid/kk tallyinfo host compaction
positive control (A'): ablation_s/in.ablation.3d.reactions `[watch] react/isurf/grid:{tally2surf,array_surf_tally}:
  the host side was written, never claimed, and is now lost`.
complete (C'): both watch reports gone; stats A' == C' (6 rows). Remaining common (A' and C') reports on this deck are
  R3/R4: `react/isurf/grid:* <- ~ComputeReactISurfGridKokkos` (destroy_kokkos view_device().data() test),
  `surf_react:models <- SurfCollideDiffuseKokkos::pre_collide`, `update:tally_models <- setup_surf_tally_copies`.
## R5b -- compute boundary/kk compute_array host normalization
positive control (A'): free/in.free `[watch] boundary:array: the host side was written, never claimed, and is now lost`.
complete (C'): gone; stats A' == C' (11 rows). Common R3/R4 left: `boundary:array <- ~ComputeBoundaryKokkos`,
  `update:tally_models <- UpdateKokkos::tally_set`.
verdict R5/R5b: benign by design; clear_sync_state() is the right annotation (deliberate divergence, see Changes) ->
  documented, detector-clean.

## U-1 -- FixEmitFace{,File}Kokkos destructor writes the stale host side of k_tasks
positive control (A side; poison is the only detector that sees it -- a plain-pointer write at teardown, no later
  coherence call for watch to check): B_poison ($S/bpoison_B, 39e1c1f7 + tool; the destructor is byte-identical in
  HEAD, `git diff 39e1c1f7 HEAD -- src/KOKKOS/fix_emit_face_kokkos.cpp` empty), GS-fixes F-G12-2 in.face2 np1, rerun
  here: `use-after-poison ... in FixEmitFaceKokkos::~FixEmitFaceKokkos() fix_emit_face_kokkos.cpp:90 <-
  Modify::delete_fix modify.cpp:389 <- ~Modify` (log $W/out/u1/face2.Bp.*), rc=0.
  fix_emit_face_file_kokkos uses a plain Kokkos::DualView for k_tasks (not instrumented), so no detector can see it
  there; same dead loop, removed on the same reasoning. fix_emit_surf_kokkos already does k_tasks.clear_sync_state()
  first in HEAD (and its loop is needed: it deletes host-only path/fracarea).
negative/complete (C'p = $W/spa_Cp, poison build of C' source, RelWithDebInfo+ASAN, SPARTA_KOKKOS_POISON=1):
  in.face2 np1: 0 ASan reports, rc 0; np4: 0 reports (B_poison np4 on the same deck: 4 reports, :90/:91 x 2 ranks).
  in.file2, in.surf2 np1: 0 reports. face2 stats C'p == stock S (md5 of thermo rows identical).
verdict: NECESSARY (poison-shown, benign teardown write) + COMPLETE.

## G1 / R1 multi-rank and stock (np1 and np4; S = stock non-tool build of C' source)
decks G1g, G1g2 (np4: -var NX 100), G1p, GS-autosync p3 (np4: -var NADD 80000), g1a (np4: NX 100):
- stats (thermo rows) S == A' == C' on every deck, np1 and np4 (md5 of rows).
- A' np4: the same G1 reports (grid:{d,i}vector <- reallocate_custom, particle:{d,i}vector <- grow_custom) and R1
  (grid:cells <- move<2,0,0,0>) on every rank that resizes.
- C' np4: no G1/R1 report. Only `irregular:index_send <- IrregularKokkos::{augment_data_uniform,~IrregularKokkos}`
  (R4 class, irregular_kokkos.cpp handle grabs; also on A', pre-existing, out of scope).
G1 verdict (final): NECESSARY (detector-shown; latent) + COMPLETE, np1/np4, stats == stock.

## Regression np1: every example, C' (WATCH= STALE= STALE_STRICT=1) vs stock S
Harness $W/bin/sweep.sh (GS-sweep shortened templates). Watch-infeasible full decks replaced by GS-sweep's reduced ones
(ablation_s 3d.reactions, adjust_temp_s sphere.*, ambi_t, ambi_3body_t, surf_react_heatflux_t); skipped as in GS-sweep:
ablation.3d (stock 131 s), implicit.3d.big, jagged.3d* (no valid data). Table: $W/compare_np1.txt.
Result: 134 decks, all rc=0 on S and C', thermo rows (CPU column dropped) identical on all 134.
C' report classes ($W/labels_C_np1.txt), all known benign classes from GS-sweep:
  R3/R3b metadata reads (update:tally_models <- setup_surf_tally_copies/tally_set/run, surf_react:models/index <-
  SurfCollide{CLL,Diffuse}Kokkos::pre_collide / ComputeSurfKokkos::pre_surf_tally, collide:gas_tally_models,
  fix_emit_surf:slist_surf <- FixEmitSurfKokkos::perform_task);
  R4 handle/pointer grabs after resize/destroy (~GridKokkos, ~SurfKokkos, ~Compute{Surf,ISurfGrid,ReactISurfGrid,
  Boundary}Kokkos, ~FixAveGridKokkos, FixAveGridKokkos::grow_percell, ParticleKokkos::grow, SurfKokkos::remove_custom,
  ComputeSurfKokkos::init_normflux);
  R8b `surf:dvector` watch on the 4 custom/in.custom.circle.{set,file}.fix[.distributed] decks (in-run fix custom, above).
  No G1/R1/R2/R5/R5b/R8(pre-run) report anywhere.

## G1 in-run grow check (custom arrays grown inside the timestep loop, device-claimed at the time)
GS-fixes F-G00-10 decks (ambipolar ionambi/velambi custom + surf_react, `-pk kokkos react/retry yes`, collide grows
particles mid-step -> grow_custom with the device side current), $W/g10:
- cll, td: C' (plain split-memory) == stock S, whole stats table, np1 and np4; invariant c_ia == N2+ + N+ exact
  (np1 cll 11327 = 5599+5728, td 16714; np4 cll 11410 = 5525+5885, td 16715 = 5525+11190).
  (np4 used to abort with the auto_sync double claim -- fixed earlier in HEAD; now runs.)
- C'p poison np1 cll and impulsive: 0 ASan reports, rc 0, invariant exact (11327, 15255 = 5599+9656).
So the host-side resize + sync_device keeps the device data valid for the device kernels after an in-run grow.

## A' vs C' over the same np1 sweep (A' = HEAD + tool, no fixes; same harness, same 134 decks)
- A' stats == S on all 134 too (none of these items changes results on the split-memory model).
- C'-only report labels (deck, kind, array, routine): NONE. The fixes introduce no report anywhere.
- A'-only labels (removed by the fixes): R1 `grid:cells/pcells/sinfo <- UpdateKokkos::move<2,1,0,0>` (adapt.*,
  custom.spiky.set, ...) and R1b `particle:particles <- UpdateKokkos::move<{2,3},*,1,0>` (5 surf_react_adsorb decks:
  the particle handle taken before the particle sync, same remedy); R8 `surf:{dvector,iarray,darray}` empty-sync
  watch (custom circle/restart decks); R5 react/isurf/grid; R5b boundary:array; R2 grid:cinfo <-
  ComputePropertyGridKokkos; plus `surf:dvector <- ~SurfKokkos` on circle.file.fix.distributed (gone with the claim).

## Regression np4 subset (S vs C' vs A', WATCH/STALE/STRICT, mpirun -np 4)
circle, sphere, adapt.static/rotate, surf.move/remove/add, emit.face/surf.flow/surf.normal, ablation.2d,
ablation.multi.inner.3d, collide, collide_3D, custom circle.set / circle.file.distributed / cube.read(.restart) /
step.set(.restart), free: 21 decks, all rc=0, C' stats == S on all 21 (and A' == S). ($W/compare_np4.txt)
C'-only labels: NONE. C' reports: R3 (update:tally_models, fix_emit_surf:slist_surf), R4 (irregular:index_send
handle grabs, ~ComputeBoundaryKokkos). A'-only: R1/R1b (grid cells/pcells/sinfo, particle:particles <- move<2|3,1,0,0>),
R2 (grid:cinfo <- ComputePropertyGridKokkos), R8 (surf dvector/iarray/darray incl. restart decks), R5b (boundary:array).
