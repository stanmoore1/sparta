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
