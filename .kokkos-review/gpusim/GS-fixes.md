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
np4: A `particle:ewhich <- CollideVSSKokkos::collisions_one<0,0>` on all 4 ranks; B absent; common labels identical; step 300 A == B (np 77813, sum c 2101).
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

### F-G22-5 — fix grid/check/kk built error messages from stale host particles/cells
method: no input makes a particle sit in a wrong cell (AB9 tried), so fault injection. Private binaries $S/gpusim/GS-fixes/F-G22-5/build/spa_inj_{A,B}: only fix_grid_check_kokkos.cpp recompiled (A or B source + identical injection block), linked against the unmodified bsync_{A,B} libraries. Injection (env GS_INJECT=<step>): after the fix's sync(Device), a device kernel moves particle k=nlocal/2 half a cell past its cell's hi[0]; with GS_INJ_SWAP=1 it also swaps particles 0 and k on the device (a reorder, as the device sort does); then particleKK->modify(Device,PARTICLE_MASK) (as a device kernel would claim); it prints the device-side TRUE id/cell of the moved particle. Deck in.gc: 2d 20x20, O flow with emit/face + vss, `fix gc grid/check 1 error`, inject at step 10; np1/2/4.
positive control (GS_INJ_SWAP=1): A_sync message names the WRONG particle and cell: np1 `Particle 0,884975670 ... outside cell 3` vs TRUE id 1498251674 cell 334; np2 proc0 884975670/cell 3 vs 1519176102/101, proc1 1567536433/13 vs 117828811/399; np4 3/3 reporting ranks wrong the same way. B_sync: every message equals the TRUE device id/cell (np1, np2, np4). REPRODUCED (stale host data in the error message under split memory).
negative control: displacement-only injection (no reorder; id/icell unchanged, only x differs on the host): A and B print identical, correct messages (np1/2/4); without injection (GS_INJECT=-1) both run clean (no ERROR, A/B step 20 np 11622).
necessary: YES (with a device-side reorder, A reports a different particle id and cell than the one flagged)
complete: YES for the outside-cell branch exercised; the single sync(Host,PARTICLE_MASK)/sync(Host,CELL_MASK) precedes all five message branches (source). Note B syncs CELL_MASK only; the messages read cells[icell].id only (no cinfo), so that suffices.
verdict: NECESSARY+COMPLETE (fault-injected)
artifacts: $S/gpusim/GS-fixes/F-G22-5 (build/fgc_{A,B}.cpp, os.*, o.*)

### FU-10 — GridKokkos first allocation (realloc_kokkos, NoInit) memset only on the host side
question here: under split memory, is the device copy of the first cells/cinfo/sinfo allocation ever read before it is initialized?
deck: $S/gpusim/GS-fixes/FU-10/in.f (2d 20x20, circle surf, emit/face, vss, run 100), np1, GLIBC_TUNABLES=glibc.malloc.perturb=171 (non-zero garbage in fresh allocations), SPARTA_KOKKOS_VERIFY=grid: WATCH=grid: STALE=grid: STALE_STRICT=1, plus TRACE=grid:cells.
result: A and B both silent (no verify/watch/stale report for any grid array); step 100 np 47148 in both. Trace (A and B identical): the first operations on grid:cells are `modify_host` then `sync_device` with flags (1,0) — a whole-span host->device copy — before any device access; so in B the host memset reaches the device at the first sync and the device tail is never read uninitialized; in A the same copy carries the host garbage, which nothing reads (as AB10 found).
positive control: none (no reader of the uninitialized tail on either side; the device side is never read before the first full sync).
negative control: A vs B identical (np 47148), no reports.
necessary: NOT shown (hygiene / CPU parity, as in AB10)
complete: YES under split memory: the host-only memset is sufficient because GridKokkos::sync(Device) with auto_sync claims the host and copies the whole span on first use; no explicit device memset needed. (sinfo not exercised here: no split cells in this deck; same allocation/sync pattern by source.)
verdict: NOT-SHOWN-NECESSARY (hygiene); device-copy concern from AB10 resolved: COMPLETE
artifacts: $S/gpusim/GS-fixes/FU-10

### F-G00-4 — fix ave/histo/kk binned global scalars via host bin_one() atomics on device views
deck: AB7 in.pos (v_n equal np + c_t compute temp, ave running) and AB7 complete/in.c (fix scalar f_gc from grid/check/kk, equal-style v_s, ave window, beyond extra/end); np1, np4; watch/stale/strict.
result: A_sync and B_sync: stats tables identical and all four histogram files byte-identical (np1 and np4); no ave/histo report from any detector in A or B.
why not catchable: A's host code writes into d_bin/d_stats — the *device* side of k_bin/k_stats — and that is coherent in the DualView model (the device side is the one being declared modified). The split-memory build only gives the host side its own allocation; the device side is still host-addressable memory on the Serial backend, so a host thread writing it is legal and poison mode (which poisons only the stale side) would not fault either. The fault is a physical-address-space one (host dereferences GPU memory) and needs a real GPU without UVM.
positive control: n/a (tool cannot express the fault)
negative control: A == B byte-identical, np1/np4.
necessary: NOT catchable by this tool
complete: B's device bin_scalar path gives identical output to A's host path on every call site (compute scalar, fix scalar, equal variable), np1/np4.
verdict: NOT-CATCHABLE (gpu address-space fault); B output-equivalent, COMPLETE by source
artifacts: $S/gpusim/GS-fixes/F-G00-4

### F-G10-6 — adsorb/kk state_synced_to_device set only on the blitted image (H2D state copy every step; perf)
method: the copies are Kokkos::deep_copy into plain device Views (sra:total_state/species_state/area/weight) from create_mirror_view mirrors, not DualViews, so SPARTA_KOKKOS_TRACE (DualView-only) does not see them (it lists 399/200 other sra DualView events, identical A vs B). Counted instead with the AB3 Kokkos-tools hook ($S/ab/AB3/tools/kmem.so, KOKKOS_TOOLS_LIBS) on the split-memory builds. Decks AB3 F-G10-5 in.surf / in.face (adsorb gs, nsync 10, 500 steps, react/retry no).
positive control: A_sync: 500 deep_copies each into sra:total_state, species_state, area, weight (every step), surf and face. B_sync: 50 each (once per nsync window). REPRODUCED (perf).
negative control: step-500 stats A == B (surf: 3215 16785 20000; face: 797 0).
necessary: YES (perf only: removes 450 redundant H2D copies per array per 500 steps)
complete: YES (both SURF and FACE adsorb modes)
verdict: NECESSARY+COMPLETE (perf-only)
artifacts: $S/gpusim/GS-fixes/F-G10-6

### F-G18-1 — device species2group table built only at first run (re-check under split memory)
deck: AB2 F-G18-1 in.regroup (`mixture air O group two` between runs, compute grid + reduce) and in.regroup2 (beam + circle, regroup, compute surf + compute boundary per group); np1 and np4; watch/stale/strict; label diff A vs B keyed on array <- routine (script $S/gpusim/GS-fixes/bin/ab.sh).
positive control: regroup step 20: A groups (10000, 0) vs B (6928, 3072) np1 = CPU (6928, 3072); np4 A (10000,0) vs B (6932,3068). regroup2 step 300: A surf (755.24, 0), boundary (382.08, 0) vs B (530.94, 224.3)/(267.77, 114.31) np1 [CPU 529.87/225, 264.11/113.96]; np4 A (758.26,0)/(379.57,0) vs B (533.87,224.39)/(266.43,113.14). REPRODUCED (same as CPU-memory A/B: cpu-observable).
detector: no A-only and no B-only labels in any of the 4 runs (common labels only: the known particle grow/grow_custom and irregular noise). The species2group refresh in B introduces no stale access or unclaimed write under split memory.
negative control: label sets A == B; run-1 lines identical.
necessary: YES (wrong group tallies; independent of memory model)
complete: YES (B correct and detector-clean, np1/np4)
verdict: NECESSARY+COMPLETE (split-memory clean)
artifacts: $S/gpusim/GS-fixes/F-G18-1

### F-G17-3 — compute surf/kk & isurf/grid/kk reallocate() mid-step recreated surf2tally (re-check under split memory)
decks: AB5 F-G17-3/in.base (2d circle, compute surf n press, fix move/surf every 10 with zero displacement, NEV=10) np1; AB5 MPI in.exp_cs (explicit/distributed, fix balance 10 rcb part) np4; in.imp_igu (implicit ablate surfs, compute isurf/grid, fix balance 10, THR=180.5) np4. watch/stale/strict, label diff.
positive control: A_sync: compute tallies 0 on every reallocate step: base np1 c_r = 0 0 at steps 70..100 while nscoll 48/50/43/66; exp_cs np4 c_cs 0 vs nscoll 88/100/114/147; imp_igu np4 c_ig 0 vs nscoll 185/186/172/163. B_sync: tally == nscoll on every line (base press 4.57e-20..6.26e-20). REPRODUCED (cpu-observable, also under split memory).
detector: no A-only labels. B-only labels, all on the tally DualViews and all at allocation/teardown time:
  compute surf/kk (base np1, exp_cs np4): `surf:array_surf_tally` and `surf:tally2surf`: device side read while host side is newer, from ~ComputeSurfKokkos() only.
  compute isurf/grid/kk (imp_igu np4): `isurf/grid:array_surf_tally`/`tally2surf`: host-side and device-side reads of the stale side from ComputeISurfGridKokkos::init_normflux() (the grow_kokkos/DualView::resize B now does instead of re-creating the views, plus taking d_ handles right after it) and from the destructor.
  Trace of surf:array_surf_tally in B (base): each step goes modify_device (kernel) -> resize (device newer, so resized on the device) -> sync_host -> modify_host (host-side post-processing) -> clear_sync_state at the next step. The kernel writes only after clear_sync_state, so no stale data reaches a consumer, and the outputs are exact (tally == nscoll on every line). These are benign accessor-order reports from B's resize-in-place design (A re-created the views, so it never had a pair that had diverged). Optional cleanup: take the d_ handles after the sync, and skip the host-newer state in the destructor.
negative control: common label sets (grow/irregular noise) identical; non-reallocate output lines A == B.
necessary: YES
complete: YES (B correct on np1 and np4 for both computes); only benign B-only detector notes, listed above
verdict: NECESSARY+COMPLETE (benign resize/destructor detector notes in B)
artifacts: $S/gpusim/GS-fixes/F-G17-3

### F-G00-1 — emit/surf/kk custom fractions: k_cummulative_custom never allocated, d_cummulative_custom never assigned (re-check under split memory)
decks: AB12 F-G00-1 in.pos (2d circle, emit/surf normal yes perspecies no custom fractions s_fr 0.9/-1) and in.bal (+ fix balance 10 rcb part); np1 and np4; watch/stale/strict.
positive control: A_sync: SIGSEGV (rc 139) in all 4 runs. B_sync: rc 0, step 100 N fraction pos np1 6143/6760 = 0.909, np4 5994/6684 = 0.897, bal np1 0.909, bal np4 6006/6687 = 0.898 (CPU ref ~0.90, AB12). REPRODUCED.
detector (B, no A labels since A crashes):
  `fix/emit/surf:cummulative_custom: device side read while host side is newer, from FixEmitSurfKokkos::grid_changed()` — benign ordering: grid_changed fills the host side, calls modify_host(), then takes d_cummulative_custom = view_device() (the accessor counts as a read) and sync_device() is done before use (perform_task, line ~475). Same allocation, so the handle is valid after the sync.
  `[watch] surf:darray: the host side was written without a claim and this sync_device has nothing to copy -- the device keeps stale data` (element 0 of 100, from SurfKokkos::sync <- FixEmitSurfKokkos::perform_task / UpdateKokkos::move): the per-surf custom array `fr` written by the `custom surf set` command is never claimed (trace: claims stay (0,0) all run). This is not F-G00-1's code. emit/surf/kk reads the fractions from the host side when it builds cummulative_custom, so the result is correct, but the device copy of the owned surf custom array stays at its initial values. See side finding SF-2.
  others: surf:darray <- ~SurfKokkos, grid:cells <- UpdateKokkos::move (SF-1), irregular noise.
negative control: in.neg not re-run (AB12: A == C identical); no-crash B outputs equal CPU fractions.
necessary: YES (A crashes, 1 and 4 ranks)
complete: YES (B correct on np1/np4 with and without rebalance); F-G00-1's own arrays are detector-clean except the benign handle-order note.
verdict: NECESSARY+COMPLETE (side finding SF-2: `custom surf set` leaves the surf custom DualView unclaimed)
artifacts: $S/gpusim/GS-fixes/F-G00-1


### F-G12-2 — emit face/face-file/surf kk subsonic_sort left sorted_kk=1 (2nd subsonic emit fix reused a stale d_plist)
decks: AB6 F-G12-2 in.face2 (2 identical subsonic emit/face xlo, 200 steps), in.file2 (2 emit/face/file, press-only file, 60 steps), in.surf2 (2 emit/surf circle, 8 steps; np4 uses in.surf4 = fnum 4e18 to keep the run small); watch/stale/strict via ab.sh; CPU ref = A_sync without -k. Signature: Kokkos RNG pools share the seed, so a stale list gives in1 == in2 exactly; correct behaviour is in2 > in1 (fix 2 sees fix 1's insertions), as on the CPU.
positive control (step: np in1 in2):
  face2 np1: A 76305 36594 = 36594 | B 80341 38258 38857 | CPU (AB6) 83582 39961 40506. np4: A 80593 38674 = 38674 | B 90158 43143 43834 | CPU 92050 44092 44735.
  file2 np1: A 13073 2531 = 2531 | B 13096 2509 2576. np2: A 13031 2532 = 2532 | B 13331 2665 2699 | CPU 12996 2491 2538. np4: A 13279 2631 = 2631 | B 13512 2690 2805 | CPU 13345 2624 2705.
  surf2 np1: A 10694 1886 = 1886 | B 55850 15356 33572 | CPU (AB6) 59681 15670 37089. surf4 np1: A 4516 1394 = 1394 | B 51598 14379 35491 | CPU 57179 16331 39120; np4: A 2417 349 = 349 | B 18435 4488 12228 | CPU 7876 1983 4174 (this deck is a runaway positive feedback with "Excessive subsonic thermal temp" warnings in all three builds, so totals diverge fast; the in2/in1 ratio ~2.3-2.7 in B and CPU vs exactly 1 in A is the signature).
  REPRODUCED under split memory at np1/2/4 for all three styles (multi-proc was untested in AB6).
detector: sorted_kk is a host flag, not a DualView, so watch/stale cannot see this fault. Labels: no A-only, no B-only except surf2 np1 B `particle:particles <- ParticleKokkos::grow` (the common grow note; A never grew because it inserted 5x fewer particles). np4: only the irregular noise, identical A/B. B adds no stale access under split memory.
negative control: AB6 single-fix and disjoint mixed decks A == B; here the step-0 rows and the create_particles count A = B = CPU.
necessary: YES (all three styles, np1/np2/np4)
complete: YES (B shows the CPU in2 > in1 behaviour at np1/2/4 for face, face/file, surf; detector-clean)
verdict: NECESSARY+COMPLETE (closes AB6's multi-proc gap)
artifacts: $S/gpusim/GS-fixes/F-G12-2 (in.face2, in.file2, in.surf2, in.surf4, o.*, lab.*, batch*.out)

### F-G00-10 — cll/td/impulsive/adiabatic kk backup() did not refresh the ambipolar/vibmode fix copies after a react/retry grew the particle arrays (re-check under split memory)
deck: $S/gpusim/GS-fixes/F-G00-10/in.g.{cll,td,impulsive,adiabatic} = AB2 ion deck (2d circle, N2+ beam, fix ambipolar e N2+ N+, surf_react prob N2+ -> N + N+, emit/face) shrunk to fnum 0.01 / 100 steps (the AB2-size deck ran >10 min per 40 steps under watch/stale); run with `-pk kokkos react/retry yes`. Invariant: c_ia (sum ionambi) == N2+ + N+. CPU ref = A_sync without -k.
positive control (np1, plain split-memory runs): A_sync cll and td abort `free(): invalid next size` (rc 134), impulsive SIGSEGV (rc 139), all before step 75. B_sync step 100: cll 17081, ia 11327 = 5599+5728; td 27861, 16714 = 5599+11115; impulsive 24884, 15255 = 5599+9656 (invariant exact). CPU ref step 100: cll 17176 / td 27842 / impulsive 25076 np (within noise). adiabatic: A == B identical (13681, 9634 = 5599+4035), as AB2 found (adiabatic necessity not shown). REPRODUCED.
detector (cll np1, watch/stale/strict, 14 min): B rc 0 and its stats table identical to the plain B run; A hangs after the heap abort (timeout 124) with the same labels as B up to the crash. No A-only, no B-only labels; the 6 common ones are the particle grow/grow_custom notes and `surf_react:models/index <- SurfCollideCLLKokkos::pre_collide/backup` (GS-sweep R3: blit without modify_host, benign). The fault is a use-after-free of the fix's cached host/device pointer to the pre-grow particle custom array, not a DualView coherence error, so watch/stale cannot name it; poison (ASan) would, but the poison builds were still absent.
negative control: same deck with `-pk kokkos react/extra 4.0` (no retry): A and B stats tables byte-identical (step 100 17069, 11308).
np4: not testable on the sync builds: A and B (all four models) abort at step 0+ with `DualView::modify_host ERROR: concurrent modification ... "particle:ivector"` from ParticleKokkos::sync <- grow_custom <- Comm::migrate_particles, i.e. the known GS-autosync double-claim bug (ambipolar custom ivector + migration growth), not in 39e1c1f7. AB2's CPU-memory t4 runs (cll, td, impulsive) cover multi-thread.
necessary: YES for cll/td/impulsive (A crashes under split memory too); adiabatic NOT shown (A == B)
complete: YES at np1 (invariant exact for all 4 models, detector-clean on cll); np>1 blocked by the known auto_sync abort
verdict: NECESSARY+COMPLETE (np1; multi-rank blocked by the GS-autosync bug)
artifacts: $S/gpusim/GS-fixes/F-G00-10 (in.g.*, p.*, p4.*, n.cll.*, o.g.cll.*, lab.g.cll.*)

## Side findings (pre-existing, A and B identical, not regressions)
- SF-1: `[stale] grid:cells / grid:pcells (/sinfo) device side read while host side is newer, from UpdateKokkos::move<D,..>` after adapt_grid / balance_grid (FU-9, F-G00-14, F-G16-5, F-G00-1 decks). Same as GS-sweep R1: update_kokkos.cpp takes `d_cells/d_pcells/d_sinfo = grid_kk->k_*.view_device()` before `grid_kk->sync(Device, CELL_MASK|PCELL_MASK|SINFO_MASK|...)`; same allocation after the sync, so the kernel reads fresh data. Benign accessor order; cosmetic fix: sync before taking the handles.
- SF-2: `custom surf set` writes the per-surf custom DualView (surf:darray) on the host without modify_host(), so `SurfKokkos::sync(Device)` has nothing to copy and the device copy keeps its initial values ([watch] report in F-G00-1 decks). Current Kokkos consumers (emit/surf/kk) read the host side, so results are right; a future device reader would see stale data. Same class as GS-autosync follow-up 2 (read_restart surf custom arrays unclaimed). Suggested: claim the host in the `custom` command's surf path (surf_kk->modify(Host, CUSTOM_MASK)).
- Residual cosmetic detector notes in B listed per item: F-G04-1 (unclaimed host zero in ReactBird::init under gpu/aware no), F-G17-3 (resize/destructor accessor order in compute surf/kk, isurf/grid/kk), F-G00-14 (ave/grid grow_percell at init), F-G00-1 (cummulative_custom handle before sync).
- Poison builds ($S/bpoison_{A,B}) never appeared during this run, so no ASan/poison pass was done; F-G16-6 (host plain-pointer reads) and F-G00-10 (use-after-free of fix copies) are the items a poison run would add detector evidence to.

## Summary
| ID | verdict |
|---|---|
| F-G04-1 | NECESSARY+COMPLETE (split memory: A reports cumulative react tallies on run 2; cosmetic [watch] residual in B under gpu/aware no) |
| F-G13-4 | NECESSARY (detector: device collide reads stale particle ewhich) + COMPLETE, np1/np4 |
| FU-9 | NOT-SHOWN-NECESSARY (no device consumer of grid ewhich); A == B, B clean |
| F-G00-14 | NECESSARY+COMPLETE (A stale/migrated f_av via reduce/kk after balance np4 and adapt np1/np4; B == CPU) |
| F-G16-6 | NECESSARY+COMPLETE (A host consumers of ke/particle/kk read zeros; B == CPU np1/np2) |
| F-G00-2 | mvv2e part NOT-CATCHABLE by the tool; modify_device part NECESSARY via F-G16-6 |
| F-G16-5 | NECESSARY+COMPLETE (A 293/1072 and 363/1072 wrong distsurf cells np1/np4; B 0) |
| F-G22-5 | NECESSARY+COMPLETE (fault-injected device reorder: A names wrong particle/cell, B right, np1/2/4) |
| FU-10 | NOT-SHOWN-NECESSARY (hygiene); host-only memset suffices under split memory: COMPLETE |
| F-G00-4 | NOT-CATCHABLE (host writing device address space; needs real GPU); A == B, B COMPLETE by source |
| F-G10-6 | NECESSARY+COMPLETE (perf: 500 -> 50 H2D state copies per array, surf and face) |
| F-G18-1 | NECESSARY+COMPLETE (cpu-observable; B split-memory detector-clean np1/np4) |
| F-G17-3 | NECESSARY+COMPLETE (cpu-observable; benign resize/destructor detector notes in B) |
| F-G00-1 | NECESSARY+COMPLETE (A segfault np1/np4; B correct; side finding SF-2) |
| F-G12-2 | NECESSARY+COMPLETE (all 3 emit styles, np1/2/4; closes AB6 multi-proc gap) |
| F-G00-10 | NECESSARY+COMPLETE at np1 for cll/td/impulsive (adiabatic not shown); np4 blocked by the known GS-autosync particle:ivector abort in both A and B |

## STATUS: COMPLETE
