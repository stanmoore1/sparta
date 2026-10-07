# GS-poison: GS-fixes decks re-run in poison mode (ASan) on A_poison / B_poison
Binaries: A_poison=$S/bpoison_A/src/spa_ (e071055f+tool), B_poison=$S/bpoison_B/src/spa_ (39e1c1f7+tool), both -fsanitize=address -fsanitize-recover=address, DEBUG_SYNC + DEBUG_SYNC_ASAN.
Run: SPARTA_KOKKOS_POISON=1 ASAN_OPTIONS=detect_leaks=0:halt_on_error=0:log_path=<dir>/a <bin> -k on -sf kk -in <deck> (mpirun -x ... -np N for multi-rank).
Scripts: $S/gpusim/GS-poison/bin/pz.sh (runs A then B, summarises the ASan logs with sum.py into sum.<tag>.<A|B>.<np>, deletes the raw logs), sum.py (dedup by top 3 non-Kokkos frames; "TOP" lines).
Work dir: $S/gpusim/GS-poison/<ID> (decks copied from $S/gpusim/GS-fixes/<ID>).
Note: ASan in recover mode prints each faulting PC once per process, so counts are "distinct faulting sites x ranks", not access counts.
Note: poison only covers class 1 (stale access to a side the counters say is stale). A missing modify_*() (unclaimed write) leaves the counters clean, so nothing is poisoned; such items stay with the watch/stale/output evidence from GS-fixes.

### F-G16-5 — compute distsurf/grid/kk did not sync grid cells/cinfo to device
deck: GS-fixes F-G16-5/in.x (np1) and in.x4 (np4).
positive control: A_poison np1: 17 reports / 12 distinct sites, all use-after-poison READ in ComputeDistSurfGridKokkos::operator()(TagComputeDistSurfGrid_surf_distance) compute_distsurf_grid_kokkos.cpp:178,179,181,182,194 (d_cinfo[icell].mask, d_cells[icell].nsplit/nsurf/...), :209-211 (d_cells lo/hi) and GeometryKokkos::dist_tri_hex geometry_kokkos.h:1263-1267 (called from :227 with the cell lo/hi), <- compute_per_grid_kokkos :135 <- compute_per_grid :60 <- DumpGrid::count (second run, after adapt+balance). np4: same 12 sites on all 4 ranks (68 reports). rc 0.
negative control: B_poison np1 and np4: 0 reports (rc 0). The SF-1 UpdateKokkos::move grid:cells handle-before-sync note from watch/stale does NOT fault under poison in A or B (the handle is dereferenced only after the sync), confirming SF-1 is benign.
necessary: YES (stale device grid cells/cinfo read at the faulting instruction, np1/np4)
complete: YES (B 0 reports np1/np4)
verdict: NECESSARY+COMPLETE (poison-confirmed at the faulting instruction)

### F-G04-1 — ReactBirdKokkos::init did not zero device reaction tallies
deck: GS-fixes in.tw np1; in.tw4 np1/np2/np4 (np4 with -pk kokkos react/retry yes); gpu/aware no variants in.tw np1, in.tw4 np2.
positive control (output, reproduced under the poison builds): run-2 tallies A tw 149/220 vs B 2/42; tw4 np1 A 177/223 vs B 7/48; np2 A 157/211 vs B 6/37; np4 A 161/224 vs B 7/39; gpu/aware no identical pattern (A cumulative, B per-run).
poison: 0 reports in A and B in all 6 variants (rc 0). Expected: A's fault is an unclaimed host zero (ReactBird::init zeroes the host array, the device keeps its counts, counters never move), i.e. class 2; nothing is ever marked stale so nothing is poisoned. Poison cannot upgrade or add to this item.
negative control: run-1 tallies A == B in every variant.
necessary: YES (output, as GS-fixes; poison n/a for this fault class)
complete: YES (B per-run tallies; 0 poison reports in B, so B's deep_copy zeroing introduces no stale access)
verdict: NECESSARY+COMPLETE (poison silent in A by fault class; B poison-clean)

### F-G13-4 — ParticleKokkos::remove_custom did not sync ewhich/eicol/edcol to device
deck: GS-fixes F-G13-4/in.c, np1 and np4.
poison: A_poison and B_poison 0 reports, np1 and np4 (rc 0). Step 300: np1 A = B 78050 / sum c 2115; np4 A = B 77813 / 2101.
why A is silent: A's remove_custom compacts ewhich/eicol/edcol on the host through the plain pointers with no modify_host(), so the counters stay in sync and the device side is never marked stale (class 2, unclaimed write). Poison only fences off a side the counters call stale, so it cannot see this; GS-fixes' evidence (STALE_STRICT `particle:ewhich <- CollideVSSKokkos::collisions_one`) is the right detector. (Also, the stale device ewhich is dereferenced only for vibmode/ambipolar/custom-copy paths that this deck doesn't exercise.)
negative control: B 0 reports (B's added modify_host+sync_device leaves no stale side that anything reads).
necessary: YES (from GS-fixes STALE_STRICT; poison n/a for this class)
complete: YES (B poison-clean np1/np4)
verdict: NECESSARY+COMPLETE (unchanged; poison n/a in A, B clean)

### FU-9 — GridKokkos::remove_custom did not sync ewhich/eicol/edcol to device
deck: GS-fixes FU-9/in.gc (remove a, adapt refine, remove b, balance rcb part, dump), np1 and np4.
poison: A_poison and B_poison 0 reports np1 and np4 (rc 0); step 400 A = B (np1 47243/31925/382.25, np4 47423/28525/344.25).
why A is silent: same class as F-G13-4: A compacts the grid ewhich/eicol/edcol on the host with no modify_host(), so the counters never mark the device side stale (unclaimed write, class 2) — and in addition no Kokkos kernel dereferences grid d_ewhich/d_eicol/d_edcol. Poison cannot upgrade this item. The watch/stale notes from GS-fixes on this deck (grid:darray <- GridKokkos::remove_custom; grid:cells/pcells <- UpdateKokkos::move, SF-1) do not fault under poison in A or B: they are accessor-order notes, not stale dereferences.
necessary: NOT shown (no device consumer; poison n/a for unclaimed writes)
complete: YES (B poison-clean, A == B)
verdict: NOT-SHOWN-NECESSARY (unchanged; poison cannot upgrade: unclaimed-write class + no device reader)

### F-G00-14 — compute reduce/kk read fix ave/grid device output without sync_pergrid_device_kokkos
deck: GS-fixes F-G00-14/in.bal, MODE=bal (balance_grid random) and MODE=adapt, np1 and np4.
positive control: A_poison: use-after-poison READ8 at compute_reduce_kokkos.cpp:612 (gather_float lambda `l_values(i) = l_src(i)`, l_src = subview of fkk->d_array_grid) <- gather_float :610 <- setup_values :493 <- compute_one_kokkos :245 <- compute_scalar :92: adapt np1 (1 report), adapt np4 (all 4 ranks), bal np4 (all 4 ranks). The fix's d_array_grid was marked host-newer (ave/grid reallocated/repopulated it on the host after the grid change) and reduce/kk dereferenced the cached device handle. Output reproduced at the same time: setup line of run 2 A adapt np1/np4 c_r1 20000 (stale) vs B 0; bal np4 A 18433 vs B 20000.
negative control: bal np1 (random balance moves nothing): A and B 0 reports, A = B. B 0 reports in all 4 variants.
necessary: YES — upgraded from output-only to detector-shown: poison names the faulting read (the GS-fixes watch/stale pass could not, because reduce/kk read the cached handle with no accessor call)
complete: YES (B 0 reports; B == CPU). The B-only watch/stale note `ave/grid:array_grid <- grow_percell` (init) does not fault under poison: benign accessor order.
verdict: NECESSARY+COMPLETE (poison-confirmed at compute_reduce_kokkos.cpp:612, np1 adapt, np4 adapt/bal)

### F-G16-6 — compute ke/particle/kk result never synced to host (dump particle, particle variable)
deck: GS-fixes F-G16-6/in.ke, np1 and np2.
positive control (output, reproduced on the poison builds): A c_rv = 0 every step and dump c_ke sum 0 (np1, np2); B c_rv 1293.97915667907 (np1) and dump sums 3.88194e-17 / 3.71701e-17 = GS-fixes CPU values.
poison: A_poison and B_poison 0 reports np1/np2 (rc 0).
why A is silent: A's kernel writes the device side of k_vector_particle with no modify_device(), so the counters stay equal and the host side is never marked stale; the host consumers read zeros from an un-poisoned, never-updated host buffer (class 2, unclaimed write). Poison cannot flag it; the output evidence stands.
negative control: c_rk (device path) A = B.
necessary: YES (output; poison n/a)
complete: YES (B correct and poison-clean: B's modify_device + sync_host leave no stale-side read)
verdict: NECESSARY+COMPLETE (unchanged; poison silent in A by fault class)

### F-G00-2 — compute ke/particle/kk kernel read host update->mvv2e (+ missing modify_device)
deck: F-G16-6 deck above.
mvv2e part: poison silent in A (0 reports np1/np2). As expected: poison poisons only the stale side of DualView allocations; `update` is an ordinary host object and on the Serial backend the "device" kernel runs on the host, so the deref is legal and unpoisoned. Still NOT catchable by this tool (needs a real GPU).
modify_device part: missing device claim = unclaimed write (class 2), invisible to poison; shown by output in F-G16-6.
necessary: mvv2e NOT-CATCHABLE (poison confirms it cannot express it); modify_device YES via F-G16-6 output
complete: B poison-clean
verdict: NOT-CATCHABLE (mvv2e) / NECESSARY via F-G16-6 (modify_device) — unchanged

### F-G22-5 — fix grid/check/kk built error messages from stale host particles/cells
method: GS-fixes' fault injection (fgc_{A,B}.cpp: after the fix's sync(Device) a device kernel moves particle nlocal/2 out of its cell, GS_INJ_SWAP=1 also swaps particles 0 and k on the device, then modify(Device,PARTICLE_MASK)) recompiled with each poison build's flags and linked against the unmodified bpoison_{A,B} libraries ($S/gpusim/GS-poison/F-G22-5/build/spa_pinj_{A,B}; binaries deleted after the runs, 0.9G). Deck GS-fixes in.gc, inject at step 10.
positive control: A_poison, swap injection: use-after-poison READ4 at FixGridCheckKokkos::end_of_step fgc_A.cpp:235 (`particles[i].icell`, plain host particle pointer) and :244 (`particles[i].id` / `cells[icell].id` in the outside-cell message) <- ModifyKokkos::end_of_step modify_kokkos.cpp:92 <- UpdateKokkos::run update_kokkos.cpp:510; np1, np2 (both ranks), np4 (4 and 3 ranks). Message names the wrong particle, as in GS-fixes (np1 `Particle 0,884975670 ... outside cell 3` vs TRUE id 1498251674 cell 334). Displacement-only injection (GS-fixes' negative control, where A's message happened to be right): A_poison still reports the same two stale host reads (:235, :244) — poison shows the stale read even when the message text is correct.
negative control: B_poison 0 reports in swap np1/2/4 and displacement np1 (message = TRUE id/cell); no injection (GS_INJECT=-1) np4: A and B 0 reports, step 20 np 11619 both.
necessary: YES — upgraded: poison flags the stale plain-pointer host read at the faulting line for every injection variant, not just the reorder one
complete: YES (B 0 reports, np1/2/4)
verdict: NECESSARY+COMPLETE (fault-injected; poison-confirmed at fix_grid_check_kokkos end_of_step message reads)

### FU-10 — GridKokkos first allocation (realloc_kokkos, NoInit) memset only on the host side
deck: GS-fixes FU-10/in.f, np1 (plain and GLIBC_TUNABLES=glibc.malloc.perturb=171) and np4.
poison: A_poison and B_poison 0 reports in all three variants (rc 0); step 100 A = B (np1 47148, np4 47043).
reading: poison shows no device-side read of grid cells/cinfo/sinfo while the host side is newer, i.e. nothing touches the device copy before the first whole-span host->device sync (agrees with the GS-fixes trace). Poison cannot see a read of uninitialized-but-current memory (ASan here tracks the poisoned stale side, not initialization), so it cannot upgrade the "uninitialized tail" question beyond what the trace showed.
necessary: NOT shown (no stale or uninitialized read observed)
complete: YES (B poison-clean; host memset + first sync covers the device copy)
verdict: NOT-SHOWN-NECESSARY (hygiene) / COMPLETE — unchanged; poison silent in A and B

### F-G00-4 — fix ave/histo/kk binned global scalars via host bin_one() atomics on device views
deck: GS-fixes F-G00-4/in.pos and in.c, np1 and np4.
poison: A_poison and B_poison 0 reports in all 4 runs (rc 0); all histogram files A == B byte-identical (hgc, hs, hv, hc; np1 and np4).
why: A's host code writes d_bin/d_stats, the device side of k_bin/k_stats, while that side is the current one (it is the side later declared modified), so it is not poisoned; on the Serial backend the device allocation is host-addressable. The fault is a host thread touching GPU address space, not a stale-side access, so poison cannot express it — as GS-fixes predicted.
necessary: NOT catchable (confirmed under poison)
complete: B output-equivalent and poison-clean
verdict: NOT-CATCHABLE (unchanged)

### F-G10-6 — adsorb/kk state_synced_to_device set only on the blitted image (H2D copy every step; perf)
deck: GS-fixes F-G10-6/in.surf and in.face (-var n1 500), np1 and np4.
poison: A_poison and B_poison 0 reports in all 4 runs (rc 0). np1 step 500 A = B (surf 3215/16785/20000, face 797); np4 surf A 3331/16669 vs B 3321/16679 (conserved total 20000 both), face A 774 vs B 763 (stochastic divergence from rank-order RNG, not a coherence fault; no poison report on either side).
reading: the redundant copies are deep_copy into plain device Views (not DualViews), so poison has nothing to fence; it confirms B's once-per-nsync copy leaves no stale state read in between (np1/np4).
necessary: YES (perf, from GS-fixes' kmem count; poison n/a)
complete: YES (B poison-clean, surf and face, np1/np4)
verdict: NECESSARY+COMPLETE (perf-only; poison-clean in B)

### F-G18-1 — device species2group table built only at first run
deck: GS-fixes F-G18-1/in.regroup and in.regroup2, np1 and np4.
positive control (output, reproduced on the poison builds): regroup step 20 A groups (10000, 0) vs B (6928, 3072) np1 / (6932, 3068) np4; regroup2 step 300 A surf (755.24, 0) boundary (382.08, 0) vs B (530.94, 224.3)/(267.77, 114.31) np1; np4 A (758.26,0)/(379.57,0) vs B (533.87,224.39)/(266.43,113.14) — identical to GS-fixes.
poison: A_poison and B_poison 0 reports, all 4 runs (rc 0). A's fault is a device table never rebuilt after `mixture ... group` (a stale value in a current buffer, not a stale DualView side), so poison cannot name it — cpu-observable item.
necessary: YES (output)
complete: YES (B correct and poison-clean np1/np4)
verdict: NECESSARY+COMPLETE (unchanged)

### F-G17-3 — compute surf/kk & isurf/grid/kk reallocate() mid-step recreated surf2tally
deck: GS-fixes F-G17-3 in.base np1 (-var NEV 10), in.exp_cs np4, in.imp_igu np4 (-var THR 180.5).
positive control (output, reproduced on the poison builds): A tallies 0 on every reallocate step (base c_r 0 at 70..100 vs nscoll 48/50/43/66; exp_cs np4 0 vs 88/100/114/147; imp_igu np4 0 vs 185/186/172/163); B tally == nscoll on every line.
poison: A_poison and B_poison 0 reports in all 3 runs (rc 0). A's fault (views re-created mid-step, tallies lost) is a logic error, not a stale-side read. In B, poison also stays silent on the resize-in-place path that watch/stale flagged (B-only `surf:array_surf_tally/tally2surf <- ~ComputeSurfKokkos`, `isurf/grid:* <- init_normflux`): no stale byte is dereferenced, confirming those notes are benign accessor-order reports.
necessary: YES (output)
complete: YES (B correct and poison-clean np1/np4)
verdict: NECESSARY+COMPLETE (B's watch/stale residuals confirmed benign by poison)

### F-G00-1 — emit/surf/kk custom fractions: k_cummulative_custom never allocated / d_cummulative_custom never assigned
deck: GS-fixes F-G00-1 in.pos and in.bal, np1 and np4.
positive control: A_poison: ASan SEGV (not a poison report) in all 4 runs at FixEmitSurfKokkos::grid_changed fix_emit_surf_kokkos.cpp:241 (`k_cummulative_custom.view_host()(isurf,isp) = ...` on the never-allocated view: the `extent(0) > max` guard skips the realloc when extent is 0) <- grid_changed :228 <- FixEmitSurfKokkos::init :164 <- Modify::init; np4: every rank (2 ranks' stacks unsymbolized as the job aborted). rc 1.
negative control: B_poison 0 reports, rc 0, step 100 N fraction pos np1 6143/6760, np4 5994/6684, bal np1 6143/6760, bal np4 6006/6687 (= GS-fixes B). B's `cummulative_custom` handle-before-sync note (watch/stale) does not fault under poison: benign. SF-2 (`custom surf set` unclaimed host write) is class 2 and stays invisible to poison, as expected.
necessary: YES (A crashes at the unallocated-view write; ASan names the line)
complete: YES (B correct and poison-clean, np1/np4, with and without rebalance)
verdict: NECESSARY+COMPLETE (ASan-located SEGV in A; B poison-clean)

### F-G12-2 — emit face/face-file/surf kk subsonic_sort left sorted_kk=1 (2nd subsonic emit reused stale d_plist)
deck: GS-fixes F-G12-2 in.face2 (np1, np4), in.file2 (np1, np2, np4), in.surf2 (np1), in.surf4 (np1, np4).
positive control (output, reproduced on the poison builds, values = GS-fixes): face2 np1 A 36594 = 36594 vs B 38258 < 38857; np4 A 38674 = 38674 vs B 43143 < 43834; file2 np1 A 2531 = 2531 vs B 2509 < 2576, np2 A 2532 = vs B 2665 < 2699, np4 A 2631 = vs B 2690 < 2805; surf2 np1 A 1886 = vs B 15356 < 33572; surf4 np4 A 349 = vs B 4488 < 12228.
poison: no report on the fixed path in A or B (sorted_kk is a host flag; the stale d_plist is a plain device View, not a DualView side, so poison cannot name it). file2/surf2/surf4: 0 reports A and B.
common A/B report (not this item): face2 np1 (1 rank) and np4 (the 2 ranks owning xlo faces), A and B identical: use-after-poison WRITE8 at FixEmitFaceKokkos::~FixEmitFaceKokkos fix_emit_face_kokkos.cpp:90 and :91 (`tasks[i].ntargetsp = NULL; tasks[i].vscale = NULL;`) <- Modify::delete_fix <- ~Modify. `tasks` is the host pointer of k_tasks, which subsonic_inflow left device-newer (modify_device at :716), so the destructor writes the poisoned stale host side. Benign (teardown; the writes only stop the base destructor freeing Kokkos-owned arrays, and line 94 sets tasks = NULL anyway, so the base ~FixEmitFace skips the whole loop). See "Unexplained B reports" U-1.
necessary: YES (output; poison n/a)
complete: YES (B CPU-like in2 > in1 everywhere; no B report on the fixed path)
verdict: NECESSARY+COMPLETE (unchanged; side report U-1 pre-existing in A and B)

### F-G00-10 — cll/td/impulsive/adiabatic kk backup() did not refresh the ambipolar/vibmode fix copies after react/retry grew the particle arrays
deck: GS-fixes F-G00-10/in.g.{cll,td,impulsive,adiabatic} with -pk kokkos react/retry yes, np1; negative in.g.cll with react/extra 4.0; np4 cll.
positive control: A_poison reports **heap-buffer-overflow** (ASan, not poison: the fault is a stale cached view of the pre-grow custom array, which ASan sees as an out-of-bounds access past the old allocation) in FixAmbipolarKokkos::update_custom_kokkos fix_ambipolar_kokkos.h:79 (READ4, ionambi read), :82/:89 (WRITE4, ionambi), :101-103 (WRITE8, velambi xyz) <- SurfCollide{CLL,TD,Impulsive}Kokkos::collide_kokkos<1,0> surf_collide_{cll,td,impulsive}_kokkos.h:197 <- UpdateKokkos::surf_collide_dispatch update_kokkos.h:234-236 <- UpdateKokkos::operator()<2,1,1,0,0> update_kokkos.cpp:1856: cll 5 sites, td 5 sites, impulsive 6 sites. With ASan's redzones the run no longer aborts (rc 0), and the overflow shows up as a broken invariant in A: c_ia (sum ionambi) vs N2+ + N+ at step 100: cll A 11299 vs 5599+5728 = 11327; td A 16681 vs 16714; impulsive A 15250 vs 15255. B: exact (11327, 16714, 15255).
adiabatic: A and B 0 reports, A = B (13681, 9634 = 5599+4035) — adiabatic necessity still not shown (as AB2/GS-fixes).
negative control: react/extra 4.0 (no retry): A and B 0 reports, step 100 identical (17069, 11308).
np4: A and B both abort at startup with `DualView::modify_host ERROR: concurrent modification ... "particle:ivector"` (the known GS-autosync double-claim in ParticleKokkos::grow_custom, fixed in the repo after 39e1c1f7), so no multi-rank evidence — same as GS-fixes.
necessary: YES for cll/td/impulsive — upgraded: ASan names the out-of-bounds access through the fix's stale cached view at the exact lines, and shows the wrong ionambi count it produces; adiabatic NOT shown
complete: YES at np1 (B 0 reports, invariant exact for all four models); np>1 blocked by the known auto_sync abort
verdict: NECESSARY+COMPLETE (np1; ASan-located in A; multi-rank blocked by the known GS-autosync bug)

## Summary
| ID | A poison/ASan on fixed path | B | verdict (change vs GS-fixes) |
|---|---|---|---|
| F-G16-5 | YES: use-after-poison in ComputeDistSurfGridKokkos::operator() compute_distsurf_grid_kokkos.cpp:178-211 + dist_tri_hex geometry_kokkos.h:1263-1267 (stale device cells/cinfo), np1/np4 | 0 | NECESSARY+COMPLETE (now poison-confirmed) |
| F-G00-14 | YES: use-after-poison compute_reduce_kokkos.cpp:612 (gather_float on fix d_array_grid) <- setup_values :493; adapt np1/np4, bal np4 | 0 | NECESSARY+COMPLETE (upgraded: detector-shown, was output-only) |
| F-G22-5 | YES (fault-injected): use-after-poison fix_grid_check_kokkos end_of_step :235/:244 (stale host particles/cells), np1/2/4; also on displacement-only injection | 0 | NECESSARY+COMPLETE (poison-confirmed) |
| F-G00-10 | YES (ASan heap-buffer-overflow, not poison): fix_ambipolar_kokkos.h:79-103 via cll/td/impulsive collide_kokkos; A breaks the ionambi invariant | 0, invariant exact | NECESSARY+COMPLETE np1 (upgraded: ASan-located); adiabatic not shown; np4 blocked by known auto_sync abort |
| F-G00-1 | ASan SEGV fix_emit_surf_kokkos.cpp:241 (unallocated k_cummulative_custom) np1/np4 | 0 | NECESSARY+COMPLETE (ASan-located) |
| F-G04-1 | silent (unclaimed host write, class 2) | 0 | NECESSARY+COMPLETE (output; unchanged) |
| F-G13-4 | silent (unclaimed host write of ewhich/eicol/edcol) | 0 | NECESSARY+COMPLETE (STALE_STRICT; unchanged) |
| FU-9 | silent (unclaimed write + no device reader) | 0 | NOT-SHOWN-NECESSARY (unchanged; poison cannot upgrade) |
| F-G16-6 | silent (missing modify_device = unclaimed device write) | 0 | NECESSARY+COMPLETE (output; unchanged) |
| F-G00-2 | silent (mvv2e: host object, not a DualView; modify_device: class 2) | 0 | NOT-CATCHABLE (mvv2e) / NECESSARY via F-G16-6 (unchanged) |
| FU-10 | silent (no device read before first full sync; poison doesn't track init) | 0 | NOT-SHOWN-NECESSARY / COMPLETE (unchanged) |
| F-G00-4 | silent (host writes the current device side; address-space fault) | 0 | NOT-CATCHABLE (unchanged; confirmed) |
| F-G10-6 | silent (plain device Views, perf) | 0 | NECESSARY+COMPLETE perf (unchanged) |
| F-G18-1 | silent (stale table in plain View) | 0 | NECESSARY+COMPLETE (output; unchanged) |
| F-G17-3 | silent (logic error: views re-created mid-step) | 0 | NECESSARY+COMPLETE (B's watch/stale resize notes confirmed benign) |
| F-G12-2 | silent (host flag / plain View) | 0 on fixed path; U-1 in A and B | NECESSARY+COMPLETE (unchanged) |

Upgrades from this pass: F-G00-14 (output-only -> detector-shown at compute_reduce_kokkos.cpp:612), F-G00-10 (crash -> ASan-located overflow at fix_ambipolar_kokkos.h:79-103 plus a broken invariant), F-G22-5 (poison flags the stale host read even when the message happens to be right), F-G16-5 (faulting instruction named). Items poison could not upgrade (F-G00-4, FU-9, FU-10, F-G00-2 modify part): each A fault is either an unclaimed write (counters stay clean, so nothing is poisoned) or not a DualView-side access. Poison cannot see either kind by design.
Benign-confirmed by poison (watch/stale notes that do not dereference stale bytes in A or B): SF-1 UpdateKokkos::move grid:cells/pcells handle-before-sync; grid:darray <- GridKokkos::remove_custom; ave/grid grow_percell at init (F-G00-14 B-only); compute surf/kk and isurf/grid/kk resize/destructor (F-G17-3 B-only); emit/surf cummulative_custom handle (F-G00-1).
Known/unfixed-in-B items seen: the particle:ivector auto_sync double-claim abort (F-G00-10 np4, A and B). The others (custom-vector device resize, custom surf set unclaimed write SF-2, ComputePropertyGridKokkos sync order, UpdateKokkos::move handles) gave no poison report on these decks: SF-2 is class 2, and the move handles are only dereferenced after the sync.

## Unexplained B reports (not explained by a known item)
- U-1 (pre-existing, identical in A; benign teardown): use-after-poison WRITE8 at src/KOKKOS/fix_emit_face_kokkos.cpp:90 and :91 in FixEmitFaceKokkos::~FixEmitFaceKokkos() (`tasks[i].ntargetsp = NULL; tasks[i].vscale = NULL;`) <- Modify::delete_fix (modify.cpp:389) <- ~Modify. Deck F-G12-2 in.face2 (subsonic emit/face), np1 and np4 (the ranks owning xlo faces). Cause: `tasks` is k_tasks' host pointer; subsonic_inflow ends with k_tasks.modify_device() (:716), so at teardown the host side is the stale (poisoned) side and the destructor writes it. No data consequence (the host bytes are discarded; line 94 sets tasks = NULL, so the base ~FixEmitFace skips its free loop anyway). Proposed fix: drop the loop at :89-92 (tasks = NULL already prevents the base destructor from freeing the Kokkos-owned ntargetsp/vscale), or call k_tasks.sync_host() first (costs a D2H copy at teardown). The same destructor loop exists at fix_emit_face_file_kokkos.cpp:105-106 and fix_emit_surf_kokkos.cpp:142-143 (not hit on these decks: their subsonic paths left k_tasks host-current at exit); apply the same fix there.
- No other B report on any deck: B_poison was 0 reports on every run except U-1.

## STATUS: COMPLETE
