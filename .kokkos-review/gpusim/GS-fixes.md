# GS-fixes: GPU-only fixes re-tested with split-memory detector builds
Binaries: A_sync=$S/bsync_A/src/spa_ (e071055f+tool), B_sync=$S/bsync_B/src/spa_ (39e1c1f7+tool; all listed fix commits are ancestors).
Poison builds ($S/bpoison_{A,B}/src/spa_): not present at start (checked again per item).
Detector env for "watch/stale" runs: SPARTA_KOKKOS_WATCH= SPARTA_KOKKOS_STALE= SPARTA_KOKKOS_STALE_STRICT=1.
Work dir: $S/gpusim/GS-fixes/<ID>. Stock CPU reference = A_sync binary without -k on.
Known noise (A and B identical, unrelated to these fixes): `[stale] irregular:index_send: device side read while host side is newer,
from IrregularKokkos::augment_data_uniform / ~IrregularKokkos` on every multi-rank run.

### F-G04-1 — ReactBirdKokkos::init did not zero device reaction tallies
deck: AB1 in.tw (1 cell, N2 dissociation tce, run 100 + run 100); in.tw4 (same on 2x2x2 grid, balance rcb) for 1/2/4 ranks.
positive control: A_sync np1: run1 147/178, run2 **149/220** (= 147+2, 178+42: cumulative) vs B_sync run2 **2/42** (per-run); thermo A==B identical (bug only in reported tallies). in.tw4: A np1 run2 177/223 vs B 7/48; np2 A 157/211 vs B 6/37; np4 (react/retry yes, A needs it: pre-existing react/extra overflow) A 161/224 vs B 7/39. CPU ref run2 4/50 (per-run). REPRODUCED (wrong output) under split memory — the earlier CPU A/B could not.
detector: default gpu/aware yes path (extract_tally allreduces d_tally_reactions directly) -> watch/stale silent in A and B (host array never synced, nothing to compare). With `-pk kokkos gpu/aware no`: A and B both report once `[watch] react_bird:tally_reactions: the host side was written, never claimed, and is now lost ... between sync_host and modify_device, element 8 changed 147 -> 0` (from ReactBirdKokkos::extract_tally <- Finish::end on run 2). In B this is benign (ReactBird::init zeroes host unclaimed, deep_copy zeroes device, so both sides are 0 and output is right), but it is a residual detector report: cleaner form would be `ReactBird::init(); k_tally_reactions.modify_host(); k_tally_reactions.sync_device();` (or zero device and clear_sync_state).
negative control: run 1 of every variant A==B identical (147/178 etc.); single-run decks unaffected.
necessary: YES (shown: A reports cumulative tallies on run 2 under split memory, 1/2/4 ranks, gpu/aware yes and no)
complete: YES for correctness (B per-run on all variants); minor: B still triggers one benign [watch] report under gpu/aware no (unclaimed host zero in ReactBird::init).
verdict: NECESSARY+COMPLETE (cosmetic watch residual in B, suggested modify_host/sync_device instead of deep_copy)
artifacts: $S/gpusim/GS-fixes/F-G04-1

### F-G13-4 — ParticleKokkos::remove_custom did not sync ewhich/eicol/edcol to device
deck: AB2 in.c (2d circle + emit/face, custom particle a int / b float 2 / c int, run 100; `custom particle remove a`; run 100; `custom particle remove b`; run 100), kk np1 (np4 below).
positive control: A_sync np1 watch/stale/strict: `[stale] particle:ewhich: device side read while host side is newer, from CollideVSSKokkos::collisions_one<0,0>` (200 times = every step after the removal). B_sync: this label absent. REPRODUCED (detector: device reads the pre-removal ewhich layout).
negative control: labels common to A and B (noise, same counts): particle:darray/ivector <- ParticleKokkos::grow_custom (host read during resize), particle:particles <- ParticleKokkos::grow (3 times). Stats/observables: A and B differ from step 50 on (before any removal; A and B differ in other fixed code paths), both statistically equal to CPU (step 300 np A 78050 / B 78067 / CPU see log.cpu, sum c 2115/2135). The stale ewhich did not change the observables in this deck (the collide kernel only consults ewhich for vibmode/custom copy, absent here), so wrongness is shown by the detector, not by output.
necessary: YES (detector-shown under split memory: device collide kernel reads stale ewhich)
complete: YES for the particle side (no remaining ewhich/eicol/edcol report in B); see FU-9 for the grid side.
verdict: NECESSARY (detector) + COMPLETE
artifacts: $S/gpusim/GS-fixes/F-G13-4

### FU-9 — GridKokkos::remove_custom did not sync ewhich/eicol/edcol to device
deck: AB10 FU-9/in.gc (2d, custom grid a/b[2]/c/d, remove a, adapt_grid refine, remove b, balance rcb part, dump g_c g_d), np1 and np4, watch/stale/strict.
positive control: none possible: no Kokkos kernel reads the grid's device ewhich/eicol/edcol (grep: compute_reduce_kokkos reads grid->ewhich on host; d_ewhich users are all particle-side), so A_sync gives no ewhich/eicol/edcol report and identical output.
negative control: A vs B np1: step 400 np 47243, c_cd 31925/382.25 both; np4 47423/28525/344.25 both; dump files byte-identical A==B at np1 and np4. Report sets identical in A and B: grid:darray <- GridKokkos::remove_custom (device read while host newer, 1/rank; common, not caused by this fix), grid:cells and grid:pcells <- UpdateKokkos::move<2,0,0,0> (1-2/rank after adapt/balance; see side finding SF-1), particle:particles <- ParticleKokkos::grow.
necessary: NOT shown (latent: no device consumer)
complete: YES by source (mirrors F-G13-4 and GridKokkos::add_custom); B adds no reports.
verdict: NOT-SHOWN-NECESSARY (latent, gpu-only hygiene); negative control A==B identical
artifacts: $S/gpusim/GS-fixes/FU-9

### F-G00-14 — compute reduce/kk read fix ave/grid device output without sync_pergrid_device_kokkos
deck: $S/gpusim/GS-fixes/F-G00-14/in.bal: 3d 4^3, 20000 N2/O2, `fix av ave/grid all 1 2 2 c_g[*] c_pg`, compute reduce sum/max f_av[*], run 6; then MODE=bal `balance_grid random` or MODE=adapt `adapt_grid all refine particle 200 10`; run 2 (stats at setup step 6 read f_av via reduce/kk before ave/grid's end_of_step refreshes the device). np1 and np4; CPU ref = A_sync without -k.
positive control (setup line of run 2, c_r1 c_r2 c_r3 c_r4):
  bal np4: CPU 20000 3.18751526500508e-24 1e-12 1.5625e-14 | **A 18433 2.93922010052955e-24 9.21875e-13** | B = CPU. (np1 random balance moves nothing: A=B=CPU.)
  adapt np1: CPU 0 0 0 0 (refined cells' fix output zeroed on host) | **A 20000 3.186e-24 1e-12 1.5625e-14 (stale pre-adapt device values)** | B 0 0 0 0 = CPU.
  adapt np4: CPU 0 0 0 0 | **A 20000 3.1875e-24 ...** | B 0 0 0 0.
  REPRODUCED (wrong output) — CPU A/B (AB4) could not.
negative control: run-1 lines and the step-8 line after the fix's own end_of_step: A = B = CPU in all 4 variants.
detector: the fault itself is invisible to watch/stale (reduce/kk reads the cached d_array_grid handle, no accessor). Common A/B labels: ave/grid:array_grid,tally <- FixAveGridKokkos::grow_percell (host read during DualView resize), grid:cells <- UpdateKokkos::move<3,0,0,0> (SF-1). B-only extra: `ave/grid:array_grid: device side read while host side is newer, from grow_percell` called from Modify::init (B's new init-time grow_percell(0) after adapt/balance; results correct; worth a look but not a data error here).
necessary: YES (A reports migrated/stale f_av after balance (np4) and adapt (np1, np4))
complete: YES for reduce/kk (B == CPU in all variants)
verdict: NECESSARY+COMPLETE
artifacts: $S/gpusim/GS-fixes/F-G00-14

### F-G16-6 — compute ke/particle/kk result never synced to host (host consumers: dump particle, particle variable)
deck: $S/gpusim/GS-fixes/F-G16-6/in.ke: 3d 4^3, 2000 N2/O2, `compute ke ke/particle`, `variable kv particle c_ke*1e20`, compute reduce sum c_ke (device path) and sum v_kv (host path), `dump particle id c_ke` every 5 steps, run 10; np1 and np2; CPU ref A_sync without -k.
positive control: A_sync: c_rv (host particle variable) **0** at every step (CPU 1293.979.. np1 / 1239.004.. np2) and dump c_ke column **all 0** (sum 0 over 3 snapshots) while c_rk (reduce/kk, device path) is right. B_sync: c_rv and dump sums equal CPU exactly (np1 dump sum 3.88193747003721e-17, np2 3.71701333460328e-17; c_rv 1293.97915667907 / 1239.00444486776). REPRODUCED (wrong output) — the CPU A/B (AB4) could not.
detector: watch/stale/strict silent in both A and B (host consumers read vector_particle through the plain pointer, which only poison mode can see; poison builds not available).
negative control: c_rk (device consumer) A = B = CPU at all steps, np1/np2.
necessary: YES (host consumers get zeros under split memory; shown jointly with the modify_device half of F-G00-2 — both are needed: B has modify_device in compute_per_particle_kokkos and sync_host in compute_per_particle)
complete: YES for dump particle and particle-style variable (np1, np2).
verdict: NECESSARY+COMPLETE
artifacts: $S/gpusim/GS-fixes/F-G16-6

### F-G00-2 — compute ke/particle/kk kernel read host update->mvv2e on device (+ missing modify_device)
mvv2e part: NOT catchable by this tool. The Serial backend runs the kernel on the host, so dereferencing the host Update object is legal; poison mode only poisons the stale side of DualViews, not ordinary host objects; split memory only splits DualView allocations. Values A = B = CPU (F-G16-6 deck, reduce/kk c_rk). Needs a real GPU (illegal address) to show.
modify_device part: shown together with F-G16-6 (see above): without the device claim + host sync the host consumers read zeros.
necessary: mvv2e part NOT-SHOWN (tool cannot see host-object deref); modify_device part YES (via F-G16-6)
complete: B captures mvv2e into a member (source) and values unchanged.
verdict: NOT-CATCHABLE (mvv2e) / NECESSARY via F-G16-6 (modify_device)
artifacts: $S/gpusim/GS-fixes/F-G16-6

### F-G16-5 — compute distsurf/grid/kk did not sync grid cells/cinfo to device (2nd run after host grid change)
deck: AB4 F-G16-5/in.x (sphere 1200 tris, 8^3 grid, compute distsurf/grid, dump grid c_d, run 1; adapt_grid refine surf + balance_grid rcb cell; new dump, run 1); np4 uses in.x4 (gridcut -1, else "ghost cells do not exist" in all builds). Compare per-cell c_d to CPU (A_sync without -k), tol 1e-9 rel.
positive control: A_sync second-run dump: **293/1072 cells wrong at np1, 363/1072 at np4**; detector A-only labels: `grid:cells: device side read while host side is newer, from ComputeDistSurfGridKokkos::compute_per_grid_kokkos()` and `grid:cinfo: ... from ComputeDistSurfGridKokkos::compute_per_grid_kokkos()` (+ `grid:cells <- UpdateKokkos::move<3,1,0,0>`). B_sync: 0/1072 mismatches at np1 and np4; distsurf labels gone. REPRODUCED (wrong output + detector).
negative control: first-run dump (512 cells, before the grid change): A = B = CPU, 0 mismatches, np1 and np4.
residual in B (both np): `grid:pcells: device side read while host side is newer, from UpdateKokkos::move<3,1,0,0>()` (1/rank) — not distsurf; see SF-1.
necessary: YES
complete: YES for distsurf (B == CPU, no distsurf report)
verdict: NECESSARY+COMPLETE
artifacts: $S/gpusim/GS-fixes/F-G16-5

