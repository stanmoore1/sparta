# AB12 — MPI / bounds-check addendum (closes multi-rank and DEBUG_BOUNDS_CHECK gaps of AB1-AB11)
Binaries: A_mpi = $S/bmpi_base/src/spa_kokkos_omp (e071055f), C_mpi = $S/spa_C3_mpi (== bmpi_fixed, HEAD source f9743740, all fixes incl. follow-ups); both real MPI (OpenMPI) + Kokkos OpenMP with Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON. CPU reference = same binary without -sf kk.
Runner: $S/ab/AB12/r.sh <A|C> <np> <kk1|kk2|kk4|cpu> <deck> <log>  (mpirun --allow-run-as-root --oversubscribe, timeout 90 s, ulimit -v 8 GB). Work dirs $S/ab/AB12/<ID>/.

### F-G00-1 (MPI/bounds addendum to AB6)
class: mpi + bounds-check
positive control: AB6 decks, emit/surf normal yes perspecies no twopass custom fractions (0.9 N / rest O), 2d circle 100 steps: in.pos; in.bal (+ fix balance 10 rcb part -> grid_changed re-run with changed nslocal); in.dist (+ global surfs explicit/distributed); in.splitno (per-surf file 1-25 N, 26-50 O, perspecies no, dump at step 5); in.3d (sphere 1200 tris, run 10 + adapt_grid + run 10) | A_mpi kk: rc 134 bounds abort "out of bounds access label=(**UNMANAGED**) indices [0,0] extents [0,0]" (addr2line: host write into never-allocated k_cummulative_custom in grid_changed, called from FixEmitSurfKokkos::init) on EVERY variant: np1, np2, np4 (pos/bal/dist), np4 split, np4 3d, np2 t2 | C_mpi kk: rc 0 everywhere, no bounds errors | REPRODUCED
  N fraction at step 100 (CPU ref = same binary, identical A_mpi/C_mpi): pos np2 C-kk 0.904 / cpu 0.900; pos np4 0.897 / 0.900; bal np2 0.904 / 0.900; bal np4 0.898 / 0.901; dist np2 0.903 / 0.901; dist np4 0.900 / 0.898; pos np2 t2 6867 np (0.897); pos np1 t1/t4 0.909/0.895 (== AB6 values). split np4 step 5: C-kk N 169 O 153 wrong-half 0, CPU N 169 O 154 wrong-half 0. 3d np4: C-kk step 20 np 9855 N 0.896 vs CPU 9673 N 0.893.
negative control: in.neg (no custom fractions) np4 kk1 | A_mpi vs C_mpi: identical (step 100 np 6586, 3301/3285)
necessary: YES - A aborts (bounds check) / segfaults (AB6 release) on 1, 2 and 4 ranks, all variants.
complete: YES - C correct (N fraction 0.90 +-0.01 = CPU) on np1/2/4, t1/t2/t4, rcb balance (grid_changed with changed per-rank surf counts), distributed surfs, per-surf fractions, 3d + adapt; no bounds errors anywhere.
verdict: NECESSARY+COMPLETE (was NECESSARY, COMPLETENESS-PARTIAL)
artifacts: $S/ab/AB12/F-G00-1

### F-G11-3 (MPI/bounds addendum to AB6)
class: bounds-check
positive control: full-SPARTA DEBUG_BOUNDS_CHECK runs of the F-G00-1 decks (custom fractions + perspecies no = the path where the fixed kernel line used to evaluate &d_cummulative_mix[0] on the 0-extent mix view) | A_mpi: aborts earlier in init() on F-G00-1 (host k_cummulative_custom), so the kernel line cannot be reached in an unmodified A build (as in AB6) | C_mpi: 13 bounds-checked runs (np1 t1/t4, np2 t1/t2, np4 t1; pos/bal/dist/split/3d), all rc 0, no "out of bounds" message, correct species fractions (see F-G00-1) | n/a (necessity isolated at unit level in AB6)
negative control: in.neg (mix path, allocated mix view) np4 | A_mpi vs C_mpi identical
necessary: YES at unit level (AB6: A's line aborts under bounds check); full-run isolation impossible because F-G00-1 aborts first in A.
complete: YES - the bounds-checked full-SPARTA B run that AB6 lacked is now done on every custom-fraction variant, 1-4 ranks, t1/t2/t4: no bounds violations.
verdict: NECESSARY+COMPLETE (unit-level necessity; was NECESSARY, COMPLETENESS-PARTIAL)
artifacts: $S/ab/AB12/F-G00-1 (logs *.C.*)

### F-G12-2 (MPI/bounds addendum to AB6)
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
class: mpi + bounds-check (gpu-only for necessity)
positive control (multi-rank balance test AB4 could not run): AB4 in.x reworked: 8^3 grid, 20000 N2/O2, balance_grid rcb cell, fix ave/grid all 1 2 2 c_g[*] c_pg, compute reduce/kk sum f_av[1] (== np invariant), sum f_av[2] (mass), sum f_av[3] (== box vol 1e-12), max f_av[3]; `fix bal balance N 1.0001|1.0 rcb part` so cells (and fix ave/grid per-cell arrays) migrate between ranks before the reduce reads them; dt 1e-8, 20 steps. Variants: in.bal (every 2 steps) np2/np4, in.bal2/in.bal3 (every 2 / every step, f_bal in stats: imbalance 1.0007-1.004, per-rank cells 154-180 at np3) np3/np4; in.balad (AB4 in.rc + fix balance: adapt refine/coarsen + balance) np2/np4 | A_mpi kk vs C_mpi kk vs CPU: identical to the last printed digit at every output step in every variant (e.g. bal3 np3 step 20: r1 20000, r2 2.54911333390847e-23, r3 9.99999999999998e-13, r4 1.953125e-15); no bounds errors in any of 32 runs | NOT REPRODUCED (as expected: on OpenMP device and host views alias, and grow_percell re-points the handles)
negative control: in.nobal (same deck without fix balance) np2/np4 | A vs C vs CPU identical
necessary: NOT SHOWN - staleness needs separate device memory (GPU); also not triggerable by MPI migration on a host backend.
complete: YES for host backends: C == CPU on 2, 3, 4 ranks with per-step migration and with adapt+balance; no bounds violations. GPU still untested.
verdict: NOT-SHOWN-NECESSARY (gpu-only); MPI/balance completeness gap closed (A == C == CPU, 2-4 ranks)
artifacts: $S/ab/AB12/F-G00-14

### F-G00-15 (MPI/bounds addendum to AB1)
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

