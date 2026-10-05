# AB6 (emit) A/B results

### F-G00-1 — emit/surf/kk custom fractions: k_cummulative_custom never allocated (inverted realloc test), d_cummulative_custom never assigned
class: cpu-observable
positive control: 2d circle, emit/surf normal yes perspecies no twopass custom fractions s_fr (fr = 0.9/-1 -> 0.9 N, 0.1 O), 100 steps, compute count N O | A: kk t1 SEGFAULT (rc 139) before step 0; CPU ref step100 np 6686 N 6004 O 682 (N frac 0.898) | B: kk t1 np 6760 N 6143 O 617 (0.909), kk t4 np 6763 N 6052 O 711 (0.895); CPU A==B | REPRODUCED
negative control: same deck without custom fractions | A vs B: identical (kk: 6574/3282/3292 both; CPU 6552 both)
verdict: VERIFIED
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB6/F-G00-1

### F-G11-1 — emit/surf/kk: ncands==0 early return skipped post_surf_tally(), orphaning a surf-react particle-list reference per step
class: cpu-observable (memory leak)
positive control: 2d circle, surf_react prob + emit/surf from mixture with nrho 1e-30 (ncands==0 every step) + compute surf/fix ave/surf every step; 40x {create_particles 20000; run 2} so the particle array reallocates ~40 times; kk t1, max RSS via getrusage | A: maxRSS 820 MB (old particle arrays never freed) | B: 175 MB; final np 787357 in both, stats identical | REPRODUCED
negative control: same deck, emit mixture nrho 1.0 (ncands>0, normal post_surf_tally path) | A vs B: identical stats (np 791785), maxRSS 172 vs 173 MB
verdict: VERIFIED
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB6/F-G11-1

### F-G11-2 — emit/surf (CPU+kk): custom-fraction cumulative row not forced to 1.0; roundoff (>=2 unset entries) lets a draw near 1.0 walk isp past nspecies
class: unreachable (in a run: ~1e-16 per draw) -> unit-tested
positive control: standalone unit replicating init unset-fill + grid_changed row (A vs B code) and the selection loop, rn = Kokkos XorShift64 drand max (=1.0) and rn = nextafter(last) | A: 0.3/-1/-1 -> last=0.99999999999999989, rn=1.0 -> isp=6 (valid 0..2, walks through next surf's row and off the end); 0.07+7x-1 -> last=1-2.2e-16, isp=16 (valid 0..7) | B: last=1.0 exactly, isp=2 / 7 | REPRODUCED (unit)
negative control: 2d circle, 3 species N O NO, custom fractions 0.3/-1/-1, emit/surf twopass, 100 steps | CPU A vs B: identical (np 6035; 1786/2106/2143); kk B np 6039 1784/2168/2087 (fractions 0.30/0.36/0.35, matches); kk A segfaults (F-G00-1, same path)
verdict: VERIFIED (unit-level; run-level positive control infeasible, probability ~1e-16)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB6/F-G11-2 (unit.cpp, unit.out)
