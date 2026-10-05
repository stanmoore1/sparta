# AB1 collide/react — A/B results
A = build_base/src/spa_kokkos_omp (e071055f), B = spa_new_final (HEAD). Work dir: $S/ab/AB1/<ID>/ (S = session scratchpad).
MPI builds (bmpi_*) not present at start of AB1.

### F-G21V-1 — react/extra padding applied only with react/retry yes (inverted)
class: cpu-observable
positive control: in.one2 = 1-cell 3d box, 1000 N2 at 1e5 K, dt 3e-8, tce dissociation (np 1000->2000), t 1 | A: "Ran out of space in Kokkos collisions, increase react/extra" at step 1 for react/extra 1.0, 2.0 AND 4.0 (knob has no effect) | B: react/extra 1.0 -> same error (expected), 2.0 and 4.0 -> run completes, step 30 np 2000 T 42909.171 = CPU ref (np 2000 T 42909.171) | REPRODUCED
negative control: (a) react/retry yes on in.one2: A and B both complete, np 2000 T 42909.171; (b) in.one (3e4 K, low rate, no overflow) react/extra 1.0 and 2.0: A == B identical thermo (step 200 np 1360 ncoll 323 T 11965.688) | A vs B: identical
verdict: VERIFIED
artifacts: $S/ab/AB1/F-G21V-1

### F-G01-1 — collide ambipolar lookup misses explicit "fix ambipolar/kk" (OOB fix[nfix]) / no kokkos_flag check
class: cpu-observable
positive control: examples/ambi/in.ambi (200 steps) with `fix ambi ambipolar/kk ...`, -k on t 1 -sf kk: (i) with react + react/retry yes, (ii) react removed | A: Segmentation fault (exit 139) in both | B: runs; (i) step 200 np 129956 ncoll 1019 T 171442.82 = identical to B with plain `fix ambipolar` (-sf kk); (ii) np 128615 T 173285.75 = A/B plain-fix values | REPRODUCED
positive control 2 (kokkos_flag check): in.cpufix2 = all styles explicit /kk (no -sf), plain CPU `fix ambipolar`, no surfs | A: runs silently with FixAmbipolar C-cast to FixAmbipolarKokkos (step 200 np 160048, no ions ever formed) | B: clean "ERROR: Must use fix ambipolar/kk when Kokkos is enabled" | REPRODUCED
negative control: in.ambi with plain `fix ambipolar` -sf kk: no-react A == B identical (np 128615 ncoll 942 T 173285.75); react+retry A/B both complete (np 129943 vs 129956, stat. equal; differ only through other fixes in this commit range) | A vs B: identical / agree within noise
note: CPU collide.cpp/react_bird.cpp half cannot be exercised: CPU `collide vss` is refused with -k on ("Must use Kokkos-supported collision style"), and ambipolar/kk needs -k on.
side observation (not this fix): stock examples/ambi/in.ambi with -k on -sf kk and default react/extra 1.1 errors "Ran out of space in Kokkos collisions" at step ~100-200 on both A and B (collisions_one_ambipolar); passes with react/retry yes.
verdict: VERIFIED
artifacts: $S/ab/AB1/F-G01-1

### F-G01-2 — vremax/remain reallocated on group-count change but not re-seeded when vre_start=no (kk + CPU)
class: cpu-observable
positive control: in.no = 10^3-cell box, 10000 N2/N 50:50 at 1e4 K, `collide_modify vremax 0 no`, run 60, `mixture air group SELF` (1->2 groups), run 60; kk (t 1) and CPU styles | A kk: run 2 nattempt=0 ncoll=0 every step, T frozen 10167.044; A cpu: same, 0/0, T frozen 10135.772 | B kk: run 2 steps 80/100/120 nattempt 631/608/595 ncoll 176/168/188; B cpu: 591/614/590, 149/194/171 | REPRODUCED (kk and CPU)
negative control: in.yes (same but vremax 0 yes, re-seed every run): A == B bit-identical for kk and CPU, and equal to B's in.no run 2 (631/176..., 591/149...) | A vs B: identical; run 1 of in.no also identical A vs B
verdict: VERIFIED
artifacts: $S/ab/AB1/F-G01-2

### F-G04-1 — ReactBirdKokkos::init did not zero device reaction tallies (2nd run reports cumulative)
class: gpu-only
positive control: in.tw = 1-cell N2 dissociation (F-G21V-1 in.one), run 100 + run 100, "Gas reaction tallies" after each run, t 1 | A: run1 147/178, run2 2/42 (per-run, correct: on OpenMP the DualView host and device arrays alias, so ReactBird::init's host zeroing also zeroes the device copy) | B: identical 147/178, 2/42 | NOT REPRODUCED (needs separate device memory)
negative control: same input | A vs B: identical thermo and tallies; CPU ref run2 4/50 (per-run, different RNG stream) - B matches the per-run semantics
necessary: NOT shown on this machine (aliasing host/device memory)
complete: only one init path (ReactBirdKokkos::init, shared by tce/qk/tce-qk kk styles); B correct here
verdict: NOT-SHOWN-NECESSARY (gpu-only); negative control A==B identical
artifacts: $S/ab/AB1/F-G04-1

### F-G00-17 — test_collision_kokkos lacks vremax==0 guard (0/0 NaN accepts collision)
class: unreachable
positive control attempted: in.t0 = 4^3 cells, 1000 N2/N at temp 0 (vstream 100, vremax_initial=0), vremax reset every 10 steps, group change 1->2 groups for run 2; kk t 1 | A: nattempt 0 every step (T 8.3694335 constant) | B: same | NOT REPRODUCED
analysis: attempt count for a (cell,group pair) is npairs*vremax*dt*fnum/V + (remain<1 or drand<1), and MCF uses poisson(0)=0, so vremax==0 always gives nattempt 0 and test_collision_kokkos is never called with vremax==0 (vremax only grows between resets). The guard is a CPU-parity safety net only.
negative control: in.t300 (normal gas, same script) | A vs B: bit-identical all 12 thermo lines (CPU ref statistically equal, T 313-316)
necessary: NOT shown (path unreachable through attempt count)
complete: single function, guard present in B, matches collide_vss.cpp:228
verdict: NOT-SHOWN-NECESSARY (unreachable); negative control identical
artifacts: $S/ab/AB1/F-G00-17

### F-G00-13 — one-group ambipolar kernel disabled recombination 3rd body for np==2 even when J is an electron
class: cpu-observable
positive control: in.r = 1 cell, exactly 2 heavy particles (2 O2+ + 2 ambipolar e), fix ambipolar, `collide vss plasma` single group (collisions_one_ambipolar), recomb.tce "O2+ + e --> O2" (R, large rate), fnum 1e15, 50 steps; seeds 1-5, t 1 and t 4 | A: O2,O2+ = 0,2 for all 5 seeds (never recombines, e.g. seed1 8166 attempts 1834 coll) | B: 2,0 for all 5 seeds | CPU ref: 2,0 for all 5 seeds | REPRODUCED
negative control: same with 20 heavy particles (np>=3 path, unchanged): A and B both 20,0 at step 20 (seeds 2,3, A==B identical thermo), CPU 19,1 | A vs B: identical
complete: the fix covers collisions_one_ambipolar only; the group ambipolar kernel (in.rg, groups heavy/electron) keeps `np<=2` and B gives 0,2 for seeds 1-3 = CPU ref 0,2 (CPU collisions_group_ambipolar also uses np<=2), so kk matches CPU on both ambipolar kernels; non-ambipolar kernels use np<=2 like CPU (no electron J possible). No remaining sibling site.
necessary: yes (A never recombines where CPU does)
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB1/F-G00-13

### F-G00-15 — backup()/restore() did not save fix vibmode levels (DISCRETE): react/retry roll-back leaves aborted-pass mode levels
class: cpu-observable
positive control: in.v = 1 cell, 1000 CO2 (mars.species + co2.species.vib, 4 modes), CO2 dissociation tce, `collide_modify vibrate discrete`, fix vibmode, T 6e4 K, react/retry yes (plist overflow -> restore), N steps then dump id type evib p_vibmode[1-4]; invariant per CO2 particle: evib == sum_i level_i*kB*theta_i (chk.py) | A: inconsistent CO2 particles N=1: 158/863 (max rel err 8.35), N=3: 90/590, N=10: 4/76; t 4: 80/591 | B: 0 inconsistent in every run (max rel 5e-6 = dump rounding); CPU ref (N=10) 0/77 | REPRODUCED
negative control: same input at T 5000 K (no overflow, no restore), retry yes, 20 steps | A vs B: dumps byte-identical, thermo identical, 0 inconsistent
complete: kernels sharing backup/restore tested: collisions_one (in.v), subcell (in.vsub: A 74/567 bad, B 0/602), group (in.vgrp: A 283/485 bad, B 0/533); t 1 and t 4. Not tested: the two ambipolar kernels (would need ionized polyatomic species with discrete vib) and the gas-tally-forced backup path; they call the same backup()/restore() so the fix applies structurally.
necessary: yes
verdict: NECESSARY, COMPLETENESS-PARTIAL (ambipolar kernels and gas-tally retry path not run; same backup()/restore() code)
artifacts: $S/ab/AB1/F-G00-15

### F-G04-2 (+R-A-1, R-A-2) — Kokkos TCE never warned about react_prob <0 or >1 (CPU warns once per run)
class: cpu-observable
positive control: in.w = 2^3 cells, 1000 N2/N, T 1e5 K, 2 x run 20, react/retry yes; pos.tce (real rates) / neg.tce (negative A coeff) / both.tce (N2+N2 negative, N2+N huge) | A: 0 TCE warnings in every case (all kernels, t1/t4) | B: pos 2 "exceeded 1.0" (one per run), neg 2 "Negative TCE reaction probability" (one per run), both 1 "Negative" (bit-or, negative preferred, R-A-2) | CPU: neg 2 "Negative", both 1 "Negative", pos 0-1 (stochastic) | REPRODUCED
negative control: T 8000 K (prob always in [0,1]) | A vs B: identical thermo (step 40 ncoll 187 T 7928.8758), 0 warnings in A, B and CPU; neg.tce run: A==B identical thermo (warning only, no behaviour change)
complete: kernels: collisions_one (in.w, t1 and t4: B warns 1), group (in.wsg mixture group SELF, seeds 1/3/7: A 0, B 1, CPU 1), one-group ambipolar with 3-body recombination (examples/ambi_3body 100 steps, seeds 1-4: B warns 1/1/1/1, CPU 1/1/1/0, A 0). Warning reset per run (B prints once in each of 2 runs for neg/pos, same as CPU ReactTCE::init reset). tce/qk and qk have no CPU warning, so no sibling needed. R-A-1: source check - check_prob_warn(int) has no deep_copy; flag rides in d_scalars slot 9 (existing per-pass copy). Not tested: subcell, group-ambipolar kernels (same check in collisions()).
observation (not this fix): over 30 seeds (in.ws, 40 steps, T 1e5) B kk warned ">1" in 25/30 runs vs CPU 16/30 (p~0.02); the probability formula and check placement are identical in react_tce_kokkos.h and react_tce.cpp, so this reflects a difference in the sampled collision-energy tail between kk and CPU collision paths, not the warning logic. Low priority follow-up.
necessary: yes
verdict: NECESSARY, COMPLETENESS-PARTIAL (subcell and group-ambipolar kernels not run; same host-side check)
artifacts: $S/ab/AB1/F-G04-2
