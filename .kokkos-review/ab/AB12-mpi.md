# AB12 — MPI / bounds-check addendum (closes multi-rank and DEBUG_BOUNDS_CHECK gaps of AB1-AB11)
Binaries: A_mpi = $S/bmpi_base/src/spa_kokkos_omp (e071055f), C_mpi = $S/spa_C3_mpi (== bmpi_fixed, HEAD source f9743740, all fixes incl. follow-ups); both real MPI (OpenMPI) + Kokkos OpenMP with Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON. CPU reference = same binary without -sf kk.
Runner: $S/ab/AB12/r.sh <A|C> <np> <kk1|kk2|kk4|cpu> <deck> <log>  (mpirun --allow-run-as-root --oversubscribe, timeout 90 s, ulimit -v 8 GB). Work dirs $S/ab/AB12/<ID>/.

### F-G00-1 (MPI/bounds addendum to AB6)
binary: C_mpi = spa_C3_mpi, every C run re-done with spa_C4_mpi (clean rebuild) -> identical results (see "C4 re-run" at the end)
class: mpi + bounds-check
positive control: AB6 decks, emit/surf normal yes perspecies no twopass custom fractions (0.9 N / rest O), 2d circle 100 steps: in.pos; in.bal (+ fix balance 10 rcb part -> grid_changed re-run with changed nslocal); in.dist (+ global surfs explicit/distributed); in.splitno (per-surf file 1-25 N, 26-50 O, perspecies no, dump at step 5); in.3d (sphere 1200 tris, run 10 + adapt_grid + run 10) | A_mpi kk: rc 134 bounds abort "out of bounds access label=(**UNMANAGED**) indices [0,0] extents [0,0]" (addr2line: host write into never-allocated k_cummulative_custom in grid_changed, called from FixEmitSurfKokkos::init) on EVERY variant: np1, np2, np4 (pos/bal/dist), np4 split, np4 3d, np2 t2 | C_mpi kk: rc 0 everywhere, no bounds errors | REPRODUCED
  N fraction at step 100 (CPU ref = same binary, identical A_mpi/C_mpi): pos np2 C-kk 0.904 / cpu 0.900; pos np4 0.897 / 0.900; bal np2 0.904 / 0.900; bal np4 0.898 / 0.901; dist np2 0.903 / 0.901; dist np4 0.900 / 0.898; pos np2 t2 6867 np (0.897); pos np1 t1/t4 0.909/0.895 (== AB6 values). split np4 step 5: C-kk N 169 O 153 wrong-half 0, CPU N 169 O 154 wrong-half 0. 3d np4: C-kk step 20 np 9855 N 0.896 vs CPU 9673 N 0.893.
negative control: in.neg (no custom fractions) np4 kk1 | A_mpi vs C_mpi: identical (step 100 np 6586, 3301/3285)
necessary: YES - A aborts (bounds check) / segfaults (AB6 release) on 1, 2 and 4 ranks, all variants.
complete: YES - C correct (N fraction 0.90 +-0.01 = CPU) on np1/2/4, t1/t2/t4, rcb balance (grid_changed with changed per-rank surf counts), distributed surfs, per-surf fractions, 3d + adapt; no bounds errors anywhere.
verdict: NECESSARY+COMPLETE (was NECESSARY, COMPLETENESS-PARTIAL)
artifacts: $S/ab/AB12/F-G00-1

### F-G11-3 (MPI/bounds addendum to AB6)
binary: C_mpi = spa_C3_mpi, C runs re-done with spa_C4_mpi -> identical (see "C4 re-run" at the end)
class: bounds-check
positive control: full-SPARTA DEBUG_BOUNDS_CHECK runs of the F-G00-1 decks (custom fractions + perspecies no = the path where the fixed kernel line used to evaluate &d_cummulative_mix[0] on the 0-extent mix view) | A_mpi: aborts earlier in init() on F-G00-1 (host k_cummulative_custom), so the kernel line cannot be reached in an unmodified A build (as in AB6) | C_mpi: 13 bounds-checked runs (np1 t1/t4, np2 t1/t2, np4 t1; pos/bal/dist/split/3d), all rc 0, no "out of bounds" message, correct species fractions (see F-G00-1) | n/a (necessity isolated at unit level in AB6)
negative control: in.neg (mix path, allocated mix view) np4 | A_mpi vs C_mpi identical
necessary: YES at unit level (AB6: A's line aborts under bounds check); full-run isolation impossible because F-G00-1 aborts first in A.
complete: YES - the bounds-checked full-SPARTA B run that AB6 lacked is now done on every custom-fraction variant, 1-4 ranks, t1/t2/t4: no bounds violations.
verdict: NECESSARY+COMPLETE (unit-level necessity; was NECESSARY, COMPLETENESS-PARTIAL)
artifacts: $S/ab/AB12/F-G00-1 (logs *.C.*)

### F-G12-2 (MPI/bounds addendum to AB6)
binary: C_mpi = spa_C3_mpi, C runs re-done with spa_C4_mpi -> identical (see "C4 re-run" at the end)
class: cpu-observable + mpi
positive control: AB6 decks with two identical subsonic PONLY emit fixes per step (stats np, f_in1[2], f_in2[2] cumulative; stale list signature = in1 == in2 exactly), np 2 and 4, kk t1, bounds-checked:
  emit/face (in.face2, 200 steps): A np2 82940 39874 = 39874, np4 80593 38674 = 38674 | C-kk np2 85211 40742/41293, np4 90158 43143/43834 | CPU np2 87188 41711/42320, np4 92050 44092/44735
  emit/face/file (in.file2, 60 steps): A np2 2532 = 2532, np4 2631 = 2631 | C-kk np2 2665/2699, np4 2690/2805 | CPU np2 2491/2538, np4 2624/2705
  emit/surf (in.surf2, 8 steps, exponential feedback -> large seed spread): A np2 10337 1716 = 1716, np4 9386 1232 = 1232 | C-kk np2 71399 18195/46299, np4 77286 19512/50852 | CPU np2 27111 6471/13735, np4 30708 7393/16393; seeds 1-3 np2: A 10777/10424/10739 (in1 == in2 always), C-kk 69788/46553/65214, CPU 59896/77811/27123 (same distribution; step-2/4 values C-kk 68-79/95-111 vs CPU 68-91/111-145) | REPRODUCED on 2 and 4 ranks
negative control: AB6 neg decks unaffected by MPI (single subsonic fix); here A vs C both rc 0, no bounds errors in any of the 24 MPI runs (A and C)
necessary: YES - on 2 and 4 ranks A's second fix never sees fix 1's insertions (in1 == in2 exactly, emit/surf growth collapses ~5x vs CPU) for all three styles.
complete: YES - C gives in2 > in1 with totals statistically consistent with the CPU reference for emit/face, emit/face/file, emit/surf on np2/np4 (plus AB6: np1 t1/t4); no bounds-check errors.
verdict: NECESSARY+COMPLETE (was NECESSARY, COMPLETENESS-PARTIAL)
artifacts: $S/ab/AB12/F-G12-2

### F-G00-14 (MPI/bounds addendum to AB4)
binary: C_mpi = spa_C3_mpi (compute reduce / fix ave/grid path)
class: mpi + bounds-check (gpu-only for necessity)
positive control (multi-rank balance test AB4 could not run): AB4 in.x reworked: 8^3 grid, 20000 N2/O2, balance_grid rcb cell, fix ave/grid all 1 2 2 c_g[*] c_pg, compute reduce/kk sum f_av[1] (== np invariant), sum f_av[2] (mass), sum f_av[3] (== box vol 1e-12), max f_av[3]; `fix bal balance N 1.0001|1.0 rcb part` so cells (and fix ave/grid per-cell arrays) migrate between ranks before the reduce reads them; dt 1e-8, 20 steps. Variants: in.bal (every 2 steps) np2/np4, in.bal2/in.bal3 (every 2 / every step, f_bal in stats: imbalance 1.0007-1.004, per-rank cells 154-180 at np3) np3/np4; in.balad (AB4 in.rc + fix balance: adapt refine/coarsen + balance) np2/np4 | A_mpi kk vs C_mpi kk vs CPU: identical to the last printed digit at every output step in every variant (e.g. bal3 np3 step 20: r1 20000, r2 2.54911333390847e-23, r3 9.99999999999998e-13, r4 1.953125e-15); no bounds errors in any of 32 runs | NOT REPRODUCED (as expected: on OpenMP device and host views alias, and grow_percell re-points the handles)
negative control: in.nobal (same deck without fix balance) np2/np4 | A vs C vs CPU identical
necessary: NOT SHOWN - staleness needs separate device memory (GPU); also not triggerable by MPI migration on a host backend.
complete: YES for host backends: C == CPU on 2, 3, 4 ranks with per-step migration and with adapt+balance; no bounds violations. GPU still untested.
verdict: NOT-SHOWN-NECESSARY (gpu-only); MPI/balance completeness gap closed (A == C == CPU, 2-4 ranks)
artifacts: $S/ab/AB12/F-G00-14

### F-G00-15 (MPI/bounds addendum to AB1)
binary: C_mpi = spa_C3_mpi (collide path; not affected by the stale-emit-object issue)
class: cpu-observable + mpi + bounds-check
invariant (AB1 chk.py): every CO2 particle (4 vib modes, collide_modify vibrate discrete, fix vibmode) must have evib == sum_i level_i*kB*theta_i after the run.
positive control 1 — the two ambipolar kernels AB1 did not run: new deck in.va = 1 cell, CO2 0.8 / CO+ 0.2 (mars.species CO2 CO O CO+ O+ e, vibfile co2.species.vib), fix ambipolar e CO+ O+, collide_modify ambipolar yes, tce CO2 dissociation, T 1e5 K, dt 3e-8, 2 steps, -pk kokkos react/retry yes; `collide vss gas` (one group -> one_ambipolar kernel, KK warns "Single-group ambipolar") and `collide vss species` (group_ambipolar kernel):
  one_ambipolar np1 t1 seeds 1-6: A 5/476 17/474 6/475 9/462 0/447 8/458 (45/2792 bad) | C 0/2500 | CPU 0/433
  group_ambipolar np1 t1 seeds 1-6: A 6 5 3 1 5 10 (30/3712 bad) | C 0/3516 | CPU 0/591
  t4: A one 16/470, group 4/620 | C 0/418, 0/594
  np2 / np4 (8 cells, 8000 particles, fnum/8 so per-cell rate == np1): A one 67/3522, 21/3462; group 24/4698, 17/4779 | C 0/3356, 0/3261; 0/4647, 0/4667
positive control 2 — gas-tally-forced backup path (no react/retry, the per-event tally overflow triggers backup/restore): in.vt = in.va + compute gas/collision/tally (id/cell id1 id2) dumped every step, kk without react/retry: A one 10/16/11 bad (3 seeds), group 2/5/1 | C 0 in all six. Pure tally path (no react at all, in.vtnr: 1 cell, 3000 CO2, T 2e4, only the tally overflow can force a restore): A np1 t1 401/3000 bad (maxrel 17), t4 477/3000; 4 cells np2 1764/12000, np4 1761/12000 | C 0/3000 (t1, t4), 0/12000 (np2, np4) | CPU 0 | REPRODUCED in every new variant
negative control: in.negnr (in.vtnr without the tally compute: no backup/restore ever) | A vs C dumps byte-identical (0/3000 bad)
bounds check: all 60+ runs above are DEBUG_BOUNDS_CHECK builds: no out-of-bounds message in A or C.
necessary: YES - A leaves aborted-pass vibmode levels in both ambipolar kernels and in the tally-forced restore path, on 1/2/4 ranks and t1/t4.
complete: YES - C has 0 inconsistent particles in every kernel (collisions_one/subcell/group from AB1 + one_ambipolar + group_ambipolar here), retry- and tally-forced restores, np1/2/4, t1/t4.
verdict: NECESSARY+COMPLETE (was NECESSARY, COMPLETENESS-PARTIAL)
artifacts: $S/ab/AB12/F-G00-15 (in.va, in.va8, in.vt, in.vtnr, in.vtnr4, in.negnr, rv.sh, dump.*, log.*)

### F-G04-2 (MPI/bounds addendum to AB1)
binary: C_mpi = spa_C3_mpi (collide/react path, not affected by the stale-emit-object issue)
class: cpu-observable + mpi
positive control: counts of "Negative TCE reaction probability" / "exceeded 1.0" warnings (summed over ranks' output) for the two kernels AB1 did not run, kk t1 with react/retry yes, vs CPU (same binary):
  subcell kernel (AB1 in.w + collide_modify partners subcell, 2x run 20, T 1e5): neg.tce np1: A 0 | C 2 | CPU 2; np4 (balance_grid rcb cell): A 0 | C 8 | CPU 8; pos.tce (stochastic >1): np1 A 0 | C 2 | CPU 1; np4 A 0 | C 0 | CPU 1
  group_ambipolar kernel (examples ambi_3body deck with `collide vss species`, no single-group warning; tce with dissociation A coeff made negative, ambi_neg.tce): np1 A 0 | C 1 | CPU 1; np4 A 0 | C 4 | CPU 4; original ambi_3body.tce (stochastic >1): np1 A 0 | C 1 | CPU 0; np4 all 0 | REPRODUCED
negative control: subcell T 8000 pos.tce (prob in [0,1]): A vs C thermo identical (step 40 ncoll 169 T 8184.903), 0 warnings; ambi_3body np4 group: A vs C stats identical (step 100 np 100740 ncoll 15024), 0 warnings in A, C, CPU. No bounds errors anywhere.
necessary: YES - A never warns, on any kernel, 1 or 4 ranks.
complete: YES - deterministic negative-probability warning count C == CPU on subcell and group_ambipolar, np1 and np4 (plus AB1: collisions_one, group, one_ambipolar); stochastic >1 warning present in C with CPU-like frequency.
verdict: NECESSARY+COMPLETE (was NECESSARY, COMPLETENESS-PARTIAL)
artifacts: $S/ab/AB12/F-G04-2 (rw.sh, in.wsub, in.wsubm, in.agrpv, ambi_neg.tce, out.*)

### F-G14-5 (MPI/bounds addendum to AB7)
binary: C_mpi = spa_C3_mpi (fix ave/grid path; deck uses emit/face only as a particle source, the emit fix itself is not under test and its results agree with CPU; see C4 re-run note at the end)
class: cpu-observable + mpi + bounds-check
positive control: AB7 spiky deck (20x20 -> 483 cells with split sub-cells), `fix ag ave/grid all 1 1 10 c_g[1]` defined BEFORE read_surf, compute reduce sum f_ag vs np (invariant), 200 steps; in.c array variant (c_g[1] c_th[1] + dump grid); in.bal (+ fix balance 50 rcb part); np2 / np4 kk t1 | A_mpi: rc 134 bounds abort "out of bounds access label=("ave/grid:tally") with indices [200,0] but extents [200,1]" (np2) / [100,0] extents [100,1] (np4), array variant extents [.,7], every variant | C_mpi kk: c_r == np at all 21 outputs in every variant (step 200: np2 33079, np4 33023), no bounds errors; CPU (same binary) invariant also holds (33133 / 33135); array variant max f_ag[2] 1173 / 692 vs CPU 660 / 556 (max over sparse split cells, noisy, as in AB7) | REPRODUCED
negative control: fix ave/grid after read_surf, np4 kk | A vs C identical (step 200 np 33023, c_r 33023)
necessary: YES - A aborts under bounds check on 2 and 4 ranks (release build: garbage/segfault per AB7).
complete: YES - C satisfies the invariant on 1 (AB7), 2 and 4 ranks, vector and array, with fix balance; t1/t4 covered in AB7. 3d not run (same init() code, dimension-independent).
verdict: NECESSARY+COMPLETE (multi-rank gap closed)
artifacts: $S/ab/AB12/F-G14-5

### F-G00-9 (MPI/bounds addendum to AB4)
binary: C_mpi = spa_C3_mpi (compute property/surf/kk; no emit)
class: cpu-observable + mpi
positive control: AB4 in.3d (sphere 1200 tris, group sub = tris 100:300, property/surf sub 11 columns + vector form xc + all area; dump surf, run 0) and in.3dd (global surfs explicit/distributed), gridcut -1 + balance_grid rcb cell, np2 and np4, kk t1; reference = analytic from data file (AB4 chk3d.py) | A_mpi kk: 198/1200 rows wrong on np2 and np4, both explicit and distributed (all 13 columns incl. the vector form) | C_mpi kk: 0/1200 rows wrong in all 4 multi-rank variants; CPU (same binary) 0/1200 | REPRODUCED
negative control: `property/surf all area` column (group all) | A == C == analytic in every run (area_all is the only column absent from A's wrong-column list)
necessary: YES - A fills the wrong rows on 2 and 4 ranks (per-rank nsown loop), explicit and distributed surfs.
complete: YES - C == analytic on np1 (AB4), np2, np4; explicit and distributed; array and vector forms; no bounds errors.
verdict: NECESSARY+COMPLETE (multi-rank gap closed)
artifacts: $S/ab/AB12/F-G00-9

### FU-13 (MPI/bounds addendum to AB11)
binary: C_mpi = spa_C3_mpi (compute react/isurf/grid; A_mpi = e071055f). Deck uses emit/face only as particle source.
class: cpu-observable + mpi
positive control: AB11 in.g2d (implicit circle, 150^2 grid, global 0.3 reactions, 5 react/isurf/grid computes on grid groups all / inner (superset) / left (subset) / outer (disjoint) / left+r:N r:O columns, dump grid every 10, 200 steps), np4, (a) as is, (b) + fix balance 20 1.0001 rcb part; chk.py per-cell checks, inv.sh sum over cells == nsreact:
  A_mpi cpu (b): rgi = rgl = 0 everywhere (sum 0 vs expected 364/360), 354/348 cell mismatches | A_mpi kk (b): aborts rc 134 at the first balance on the F-G17-6 bounds error (react/isurf/grid:surf2tally [3963] extents [1968]), so FU-13 cannot be isolated in A-kk on N ranks
  C_mpi cpu/kk (a, no balance): 0 cell mismatches for rgi/rgl/rgo/rgv; sum rgi == rga (364 cpu / 368 kk), rgl == expected (360 / 365); tally sum == nsreact 21/21 steps
  C_mpi cpu/kk (b, balance): rgl == rga*[left] in every cell (360/360 cpu, 392/392 kk), rgo 0, rgv == rgl; tally sum == nsreact 21/21; rgi != rga in 1 (cpu) / 2 (kk) cells - see NEW finding below (those cells are outside the inner group, so rgi = 0 is the correct group masking) | REPRODUCED (CPU)
negative control: group-all column rga: A-cpu vs C-cpu on (b) identical totals (366 both) incl. the same misplaced cell
NEW pre-existing issue (not FU-13; present in A_mpi and C_mpi, CPU and kk): with fix balance, on a step where balance migrates cells, the react/isurf/grid per-cell tally of that step is reported at the wrong cell: e.g. step 40 cell id 11554 (3.5,77.5) rga 2 (A-cpu and C-cpu), C-kk step 40 cell 1 (0.5,0.5) rga 4, step 160 cell 79 (78.5,0.5) rga 1 - cells with no surfs, far outside the surface region. Global sums stay right (sum == nsreact). Without fix balance no misplaced tallies (np4 C cpu/kk 0). Likely the tally is accumulated during move with pre-balance cell indices and post-processed after balance re-ordered the cells. Recommend follow-up (also check compute isurf/grid and compute surf/grid-style per-grid tallies with fix balance).
necessary: YES (CPU, 4 ranks): A zeroes every non-all grid-group tally.
complete: YES for FU-13 - group masking exact on 4 ranks, CPU and kk, with and without balance; remaining per-cell discrepancies come from the separate balance-step misplacement above.
verdict: NECESSARY+COMPLETE (MPI gap closed; NEW unrelated balance-step tally misplacement reported)
artifacts: $S/ab/AB12/FU-13 (r.<mode>.<A|C>.np4[.nobal]/)

### FU-1 (MPI/bounds addendum to AB10)
binary: C_mpi = spa_C3_mpi (compute sonine/grid/kk, dt/grid/kk)
class: cpu-observable (crash) + mpi
positive control: AB10 decks in.sib (compute sonine/grid a x 3 b xy 2 used before the first run by adapt_grid value c_X[4] + dump, run 0) and in.dt (dt/grid with 5 property/grid inputs, same prewrap path), N2+O2, np2 and np4, kk t1 | A_mpi: SIGSEGV rc 139 in all 4 runs | C_mpi: rc 0, dumped c_X == CPU (same binary) cell by cell: sonine np2 253/253 cells, np4 281/281, dt np2/np4 512/512 cells, 0 mismatches (rel 1e-9); no bounds errors | REPRODUCED
negative control: AB10 in.neg (sonine+dt through ave/grid, run/adapt/run) np4 kk | A vs C dump byte-identical
necessary: YES - A segfaults on 2 and 4 ranks.
complete: YES - both computes match CPU on np1 (AB10), np2, np4 (adapt with per-rank nglocal change).
verdict: NECESSARY+COMPLETE (MPI gap closed)
artifacts: $S/ab/AB12/FU-1

### FU-4 (MPI/bounds addendum to AB10)
binary: C_mpi = spa_C3_mpi (kk consumers of per-grid computes)
class: cpu-observable (crash) + mpi
positive control: AB10 implicit-surf decks with `compute is isurf/grid` (kokkos_flag, not KokkosBase) fed to dt/grid (tau slot dt1, usq slot dt3), fft/grid, lambda/grid (nrho slot, temp slot), fix dt/reset; np4 kk t1 | A_mpi: SIGSEGV rc 139 in all 6 | C_mpi: clean "Cannot (yet) use non-Kokkos computes with <style>/kk" in all 6, rc 1, all ranks exit (no hang, no bounds errors) | REPRODUCED
negative control: AB10 in.neg (KokkosBase inputs only: ave/grid, lambda/grid, fft/grid, dt/grid, fix dt/reset), np4 kk 30 steps | A vs C stats identical (step 30 np 94841)
necessary: YES - A segfaults on 4 ranks.
complete: YES - every exercised site errors cleanly and collectively on 4 ranks (guards sit in init/host code before any collective); the error-not-parity resolution is unchanged from AB10.
verdict: NECESSARY+COMPLETE (MPI gap closed)
artifacts: $S/ab/AB12/FU-4

### FU-8 (MPI/bounds addendum to AB10)
binary: C_mpi = spa_C3_mpi (compute surf/kk surf_react dispatch)
class: cpu-observable + mpi
positive control: AB10 decks (2d circle, diffuse/kk, explicit `surf_react r1 prob/kk` (sr_kk), `global/kk 0.1 0.1` (gsr_kk), plain diffuse + prob/kk (sr_kkreactonly); compute surf -> ave/surf -> reduce, 400 steps), np4 kk t1 | A_mpi: "Unknown Kokkos surface reaction method" (compute_surf_kokkos.cpp) in all 3 | C_mpi: all run; stats tables byte-identical to the plain-name decks on 4 ranks (sr: step 400 np 40733 nscoll 185 nsreact 79 c_sr 182.43; gsr: np 40309 nsreact 36 c_sr 182.51 5.1759406) | REPRODUCED
negative control: sr_plain np4 | A vs C identical (np 40733 ...). (gsr_plain A fails earlier on the unrelated F-G09-4b surf_collide name check in A; C runs.)
necessary: YES (4 ranks). complete: YES - explicit prob/kk and global/kk with compute surf, 1 (AB10) and 4 ranks, kk == plain-name bit-identical; no bounds errors.
verdict: NECESSARY+COMPLETE (MPI gap closed)
artifacts: $S/ab/AB12/FU-8

### FU-18 (MPI/bounds addendum to AB11)
binary: C_mpi = spa_C3_mpi (CPU compute dt/grid; coordinator: compute paths of C3 are fine)
class: cpu-observable (crash) + mpi
positive control: AB11 per-slot decks (each of tau/temp/usq/vsq/wsq taken from a post-processed compute, the others from fix ave/grid; 3d 4^3, 20000 N2/O2, run 3, dump c_X every step), CPU, np4 | A_mpi: SIGSEGV rc 139 in all 5 | C_mpi: all run; dumps of every slot BYTE-IDENTICAL to the fix-input reference deck at steps 0-3 (5 slots x 4 dumps) | REPRODUCED
negative control: the 5 reference (fix-input) decks np4 | A vs C dumps identical (checked tau: 4/4 identical; all A ref runs rc 0)
necessary: YES (4 ranks). complete: YES - all 5 sites == reference on 4 ranks (AB11: np1).
verdict: NECESSARY+COMPLETE (MPI gap closed)
artifacts: $S/ab/AB12/FU-18 (r.<slot>.<pp|ref>.<A|C>/)

### FU-6 (MPI/bounds addendum to AB10)
binary: C_mpi = spa_C3_mpi (collide retry growth)
class: performance (per-rank logic) + bounds-check
run: AB10 in.m (27 cells + balance_grid rcb cell, 5000 N2 at 1e5 K, full dissociation, -pk kokkos react/retry yes), np2 / np4, kk t1, DEBUG_BOUNDS_CHECK | C_mpi: rc 0, no bounds errors, final np 10000 and T 45609.27 (np2) / 45450.014 (np4) == CPU of the same binary; loop time C 1.64 s / 2.88 s vs A_mpi (e071055f, which pads by react/extra because of the inverted F-G21V-1 test, i.e. not the pre-FU-6 regression baseline) 1.91 s / 3.23 s, CPU 1.07 / 2.12 s. Also all retry runs of F-G00-15 above (np1/2/4, t1/t4, all 5 kernels incl. ambipolar maxelectron sites) are bounds-clean in C.
necessary: as AB10 (perf regression shown there vs spa_new_final; no pre-FU-6 MPI binary exists, so not re-shown on N ranks).
complete: YES on N ranks - retry growth correct and bounds-clean on 2 and 4 ranks; no slowdown vs the padded pre-review binary.
verdict: NECESSARY+COMPLETE (perf; MPI correctness/bounds gap closed, MPI perf necessity not re-measured)
artifacts: $S/ab/AB12/FU-6

### FU-3 / FU-3b (MPI/bounds addendum to AB10 / AB11)
binary: C = spa_C4_mpi (clean rebuild, HEAD f9743740; == bmpi_fixed/src/spa_kokkos_omp built 03:43). A = A_mpi (e071055f, no cold-cell guard at all).
class: cpu-observable (CPU and Kokkos) + mpi
positive control: AB10 FU-3 decks (2d 10x10, 4000 particles prefilled from a temp-0 "still" mixture with vstream VS -> roundoff-only thermal energy, PONLY emit air T=10 1.38e-22 NULL, 5 steps), np2 and np4, CPU and kk t1, ulimit 3 GB/rank, timeout 60 s:
  A_mpi np4: face VS 1/10 and face/file VS 1/10: CPU "subsonic insertion count exceeds 32-bit int", kk abort rc 134 (same check); surf normal no VS 10: CPU "Failed to reallocate 2.8 GB particle:particles", kk Kokkos BadAlloc; surf normal no VS 3.3: CPU and kk run but insert NOTHING (f_in 0)
  C4 np2 / np4, f_in cumulative CPU | kk: face VS1 116 112 | 110 112 ; face VS10 116 114 | 112 112 ; face/file VS1 116 112 | 110 112 ; face/file VS10 116 114 | 112 112 ; surf normal-no VS10 130 109 | 139 117 ; VS3.3 126 108 | 140 121 ; VS1 126 107 | 141 123 (np2 CPU, np2 kk | np4 CPU, np4 kk). All 28 runs rc 0, no bounds errors, values match AB11 np1 (face 115/113, surf VS10 134/155 CPU/kk) - CPU/kk consistent | REPRODUCED
negative control: surf in.neg (warm subsonic emit/surf, 300 steps) np4 | A vs C4 stats identical except the CPU-time column (kk: step 300 np 37755 ... 42810; CPU: 37523 ... 42756)
necessary: YES - A errors / OOMs / inserts nothing on 4 ranks in every style, CPU and kk.
complete: YES - all 3 styles x CPU/kk run with physical insertion counts on 2 and 4 ranks with the correctly built binary (per-cell local test; same arithmetic as np1). Build note: spa_C3_mpi CPU surf VS10 np4 also ran (rc 0) - not used for the verdict.
verdict: FU-3b NECESSARY+COMPLETE incl. MPI (FU-3's AB10 INCOMPLETE is closed by FU-3b on 1, 2, 4 ranks)
artifacts: $S/ab/AB12/FU-3b/{face,file,surf} (log.<deck>.<A|C4>.<mode>.np<N>)

### C4 re-run (coordinator: spa_C3_mpi may hold stale emit objects)
spa_C4_mpi (clean rebuild of every file changed since e071055f; differs from spa_C3_mpi as a binary) re-ran every emit-related C run of this file:
- F-G00-1 / F-G11-3: pos/bal/dist np2+np4 kk1, split np4, 3d np4, np1 t4, np2 t2, neg np4, CPU pos np4 -> final stats IDENTICAL to C3 run for run (e.g. pos np4 6684/5994/690, 3d np4 step 20 9855/8835/1020, split 169/153 wrong-half 0); no bounds errors.
- F-G12-2: face2/file2/surf2 x np2/np4 x kk1/cpu (12 runs) -> IDENTICAL to C3 (e.g. surf2 np4 kk 77286 19512/50852).
- F-G14-5 (emit/face as source): pos np4 kk -> IDENTICAL (33023, c_r == np).
- FU-13 balance-step tally misplacement (the only unexpected C result): reproduced IDENTICALLY with C4 (cpu cell 11554 at step 40; kk cells 1 and 79) -> genuine pre-existing behaviour, not a build artefact.
- FU-3b was run with C4 only.
Conclusion: the stale-object issue did not affect any MPI result above (the stale CPU fix_emit_surf.cpp object only changes the subsonic cold-cell path, which only FU-3b exercises).

## Not testable here
- F-G10-5 (AB3) EXACT RanKnuth backups, F-G21-7 (AB9) EXACT+MPI migrate: need a SPARTA_KOKKOS_EXACT build (none of A_opt/B_opt/A_mpi/C3_mpi/C4_mpi is one). Verdicts unchanged.
- F-G16-6 (AB4) host sync_host/modify_device part, F-G00-2, F-G16-5 (AB4), F-G04-1 (AB1), F-G13-4 (AB2), FU-9 (AB10), F-G00-4 (AB7), F-G22-5 (AB9): gpu-only (separate host/device memory). F-G00-14 (AB4): MPI part closed above, necessity stays gpu-only.
- F-G20-9 (AB8) 1d_only timing path: no caller in SPARTA (np 1/2/4 already verified in AB8). F-G20-5 (AB8) idle-rank/value-init half: unreachable (create crashes first, deferred F-G20-4).
- F-G22-1 (AB9): already run on np 2/4 in AB9; end-to-end masked by self-correction (function-level only) - an MPI rerun cannot change that.
- F-G02-2 (AB1) race, F-G00-17, F-G01-4/F-G02-1 (AB1), F-G17-2 (AB5): race/unreachable on host backends, not MPI-dependent.
- F-G19-4 (AB8) CPU FFT_SINGLE 2D INCOMPLETE, F-G09-4a (AB2) INCOMPLETE -> fixed by FU-8 (MPI-verified above), F-G00-20 (AB7), G12x-F-G14-1 / G12x-F-G16-2 sibling INCOMPLETEs: not MPI/bounds gaps (resolved by FU-4/FU-1, both MPI-verified above).

## Summary
| ID (source) | updated verdict |
|---|---|
| F-G00-1 (AB6) | NECESSARY+COMPLETE (A bounds-abort np1/2/4; C == CPU N-fraction on np1/2/4, balance, distributed, per-surf, 3d; C3 == C4) |
| F-G11-3 (AB6) | NECESSARY+COMPLETE (unit-level necessity; bounds-checked full runs of C clean on 13 variants, np1-4, t1/t2/t4) |
| F-G12-2 (AB6) | NECESSARY+COMPLETE (A in1 == in2 on np2/np4 for face, face/file, surf; C in2 > in1, == CPU distribution) |
| F-G00-14 (AB4) | NOT-SHOWN-NECESSARY (gpu-only); MPI completeness closed: A == C == CPU with per-step rcb balance on 2-4 ranks, no bounds errors |
| F-G00-9 (AB4) | NECESSARY+COMPLETE (np2/np4 explicit + distributed: A 198/1200 wrong, C 0) |
| F-G00-15 (AB1) | NECESSARY+COMPLETE (one_ambipolar + group_ambipolar kernels and tally-forced restore: A inconsistent on np1/2/4 t1/t4, C 0) |
| F-G04-2 (AB1) | NECESSARY+COMPLETE (subcell + group_ambipolar: A 0 warnings, C == CPU on np1/np4) |
| F-G14-5 (AB7) | NECESSARY+COMPLETE (A bounds abort np2/np4; C invariant c_r == np, vector/array/balance) |
| FU-1 (AB10) | NECESSARY+COMPLETE (A segfault np2/np4; C == CPU cell-exact) |
| FU-4 (AB10) | NECESSARY+COMPLETE (A segfault np4 x6 sites; C clean collective error) |
| FU-6 (AB10) | NECESSARY+COMPLETE (perf; MPI retry correct + bounds-clean np2/np4; MPI perf necessity not re-measured) |
| FU-8 (AB10) | NECESSARY+COMPLETE (A "Unknown method" np4; C == plain-name bit-identical) |
| FU-3 / FU-3b (AB10/AB11) | NECESSARY+COMPLETE incl. MPI (C4: all 3 styles CPU+kk np2/np4 run with physical insertion counts; A errors/OOM/no insertion) |
| FU-13 (AB11) | NECESSARY+COMPLETE (np4 CPU+kk group masking exact) + NEW pre-existing issue: react/isurf/grid per-cell tally misplaced on fix-balance steps (A and C, CPU and kk; sums still correct) |
| FU-18 (AB11) | NECESSARY+COMPLETE (A segfault np4 all 5 slots; C byte-identical to reference) |

Housekeeping note: during the F-G00-15 runs a `pkill -f spa_kokkos_omp` / `pkill -f spa_C3_mpi` (~03:12) was issued to stop a stuck background loop of this agent; it may also have killed concurrent runs of other agents using those binary names.

## STATUS: COMPLETE
