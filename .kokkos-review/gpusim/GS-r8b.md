# GS-r8b -- in-run `fix custom N surf set` leaves owned surf custom unclaimed (R8b)
Setup: S=scratchpad, W=$S/gpusim/GS-coh, R=$S/gpusim/GS-r8b. C_pre = $W/spa_C_pre (copy of GS-coh spa_C = HEAD 3859618b
fixes + tool), C = $W/spa_C rebuilt incrementally in $W/bA from $W/src_A + this fix (modify_kokkos.cpp only). Stock S =
$W/spa_S (DEBUG_SYNC off, Serial). Variant C/C_pre = SPARTA_KOKKOS_WATCH= STALE= STALE_STRICT=1. Runner $R/run.sh,
outputs $R/out/<tag>/; the GS-coh sweeps' old C outputs were moved to $W/out/Cpre.np{1,4} (they are the C_pre baseline).

## Trace of the call order (before deciding the placement)
- Write: FixCustom::end_of_step -> Custom::process_actions -> Custom::set_surf (or the file action) writes
  surf->edvec/eivec/edarray/eiarray[...] (OWNED arrays, plain host pointers), sets estatus=0, then calls
  modify->custom_surf_changed() (FixCustom::end_of_step, mode SURF).  FixSurfTemp::end_of_step does the same (host write
  of the owned tsurf vector, estatus=0, custom_surf_changed()); FixSurfTempKokkos wraps it in sync(Host)/modify(Host).
- Bracket: ModifyKokkos::end_of_step syncs/claims PARTICLE data only (particle_kk->sync/modify with the fix datamask)
  and sets auto_sync=1 for non-Kokkos fixes; nothing touches surf custom there, and auto_sync only acts inside
  SurfKokkos::sync/modify calls, of which there are none on this path.  So the host write is never claimed.
- Can the device be claimed on surf custom at that point?  No: no code calls SurfKokkos::modify(Device,...) or
  modify_device() on k_e{i,d}{vec,array}[_local] (grep over src/KOKKOS: only SurfKokkos::modify itself); device readers
  only sync_device (move, surf_collide_*_kokkos dynamic via the _local copies, fix emit/surf/kk, computes).
  allocate_custom in-run uses sync(Device).  Even if a Kokkos style claimed the device inside the non-Kokkos fix,
  auto_sync=1 makes SurfKokkos::modify(Device) sync(Host) immediately.  TRACE (SPARTA_KOKKOS_TRACE=surf:dvector, C_pre,
  circle.set.fix np1): device flag never set in the whole run, every entry flags=(x,0); at the step-1000 set the pair is
  (0,0) (in sync).  So a sync(Host) before the write is not owed anything, and modify(Host) after the write cannot abort
  (no two-sided claim).  => placement: claim AFTER the write, at the top of ModifyKokkos::custom_surf_changed() (the
  common post-write hook of both writers), before the listed consumers run, so a consumer that syncs the device
  (fix emit/surf) sees the new values.

## Fix (src/KOKKOS/modify_kokkos.cpp)
`if (surf->exist) ((SurfKokkos*) surf)->modify(Host,CUSTOM_MASK);` at the top of ModifyKokkos::custom_surf_changed(),
plus includes surf_kokkos.h, sparta_masks.h.  SurfKokkos::modify(Host,CUSTOM_MASK) marks owned and _local halves;
the _local host copies are current too (only host spread_custom writes them), so the extra claim is harmless.
Diff: $R/r8b.diff.  compile_one.sh src/KOKKOS/modify_kokkos.cpp: OK.  Tool build: ninja -C $W/bA src/spa_ OK (4 steps).

## R8b -- the 4 custom/in.custom.circle.{set,file}.fix[.distributed] decks (GS-sweep templates)
positive control (C_pre, np1 and np4, $R/out/pre.np{1,4}): all 4 decks, both np: `[watch] surf:dvector: the host side
  was written without a claim and this sync_device has nothing to copy -- the device keeps stale data` (np1: 3 + "further
  empty-sync reports suppressed" per deck; np4: 9-12 + suppressed x4 ranks).  TRACE C_pre: after the step-1000 set the
  next ~500/1000 sync_device run with flags=(0,0): the write is never claimed.
complete (C, $R/out/post.np{1,4}): 0 [watch] reports on all 4 decks, np1 and np4.  TRACE C: `modify_host flags=(0,0)` at
  step 1000 (and 2000 where the deck runs that far) then `sync_host (1,0)` (no-op, by the consumer) and `sync_device
  (1,0)` = the copy now happens; never a device flag, no abort, rc=0 everywhere.
stats: C_pre == C == stock S on all 4 decks, np1 and np4 (17/19/22/22 thermo rows; owned surf custom has no device
  reader -- tsurf reaches the kernels through the spread _local copies -- so no output change; latent fault).
remaining labels (not R8b): `[stale] surf:dvector <- ~SurfKokkos` = R4 (memory_kokkos.h destroy_kokkos
  view_device().data() pointer test, no data access), present on C_pre too (np1: set.fix, file.fix, set.fix.distributed;
  np4: set.fix, file.fix, set.fix.distributed).  On C it shows when the last in-run set is the final step of the run
  (2000-step distributed decks: claim at step 2000, run ends without another sync_device, destructor tests the device
  pointer with the host newer): np1 {set,file}.fix.distributed, np4 same two; gone on {set,file}.fix (1500-step
  template, set at 1000 is followed by a sync_device).  So it moves between decks but is the same known R4 label
  (accepted in GS-coh), and correct: the host IS newer there.  `irregular:index_send <- ~IrregularKokkos` (R4, np4) on
  both C_pre and C.

## Regression np4 (GS-coh 21-deck subset, $W/bin/np4.sh C with the fixed spa_C; table $R/cmp_np4.txt)
all 21 rc=0; C stats == S on all 21.  Report labels C vs C_pre ($W/out/Cpre.np4): identical on every deck.  (The
cmp.sh "NEW/GONE" on sphere and step.set.restart are 4-rank stderr interleavings of the same R4 label
`irregular:index_send <- IrregularKokkos::augment_data_uniform`; un-garbled counts are equal: 4/4 and 2/2.)
No R8b-related or new label.

## Regression np1 (GS-coh 134-deck sweep, $W/bin/sweep.sh C, PAR=3; table $R/cmp_np1.txt)
all 134 rc=0; C stats == S on all 134.  Labels C vs C_pre ($W/out/Cpre.np1, the GS-coh C run): identical on 130 decks;
differences only on the 4 R8b decks: the surf:dvector [watch] empty-sync reports are GONE on all 4; the R4
`surf:dvector <- ~SurfKokkos` destroy-pointer label moves as described above (gone on set.fix and file.fix, new on
file.fix.distributed where the claimed set is the run's last step; set.fix.distributed has it before and after).
The other caller is covered too: adjust_temp/in.circle.{adjust,constant} and adjust_temp_s/in.sphere.{adjust,constant}
(fix surf/temp/kk -> FixSurfTemp::end_of_step -> custom_surf_changed, i.e. the new claim runs right after
FixSurfTempKokkos's own sync(Host)): rc=0, stats == S, labels unchanged (no abort from a two-sided claim).
Not exercised by any deck: a listed consumer (fix emit/surf with per-surf custom) running behind the new claim.

verdict: NECESSARY (detector-shown unclaimed host write on 4 decks x np1/np4; latent -- the owned surf custom copy has no
device reader today, consumers use the _local copies which spread_custom claims, so no output change) + COMPLETE
(0 watch reports on the 4 decks np1/np4, stats == stock, no new label on 134 np1 + 21 np4 decks).  Placement: after the
write is correct because the device never holds a claim on surf custom data (no modify(Device) on it anywhere; TRACE shows
device flag 0 for the whole run); a sync(Host) before the write would be a no-op and is not added.

## Changes
src/KOKKOS/modify_kokkos.cpp: ModifyKokkos::custom_surf_changed() claims `((SurfKokkos*) surf)->modify(Host,CUSTOM_MASK)`
(guarded by surf->exist) before invoking the listed fixes; includes surf_kokkos.h and sparta_masks.h.  (Same patch
applied to $W/src_A for the tool build; $W/spa_C is now the fixed binary, $W/spa_C_pre the previous one.)
compile_one.sh src/KOKKOS/modify_kokkos.cpp: OK.  Not committed.

## STATUS: COMPLETE
