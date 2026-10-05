# AB7 — fix ave (fix ave/histo(/weight)/kk, fix ave/grid/kk), commit 088431eb
Work dir: $S/ab/AB7 (S = scratchpad). Base deck: 2d 20x20 box, N/O air, emit/face xlo, `compute g grid all all n`.

### F-G00-3 — ave/histo(/weight)/kk group-masked grid binning derefs uninitialized GridKokkos* in kernel
class: cpu-observable
positive control: `group gl grid region left one` (220 of 400 cells) + `fix ave/histo 10 1 10 0 400 8 c_g[1] group gl mode vector` (and ave/histo/weight c_g2[1] c_g2[2] group gl) | A: segfault rc=139 at first end_of_step, t1 and t4, both styles | B: runs, f_h[1]=220 cells/step (weight: sum 41.9 @50) matching CPU (220; 41.99) | REPRODUCED
negative control: same fix without `group` (bins all 400 cells) | A vs B: identical (t1 and t4 stats + histogram files, diff clean)
verdict: VERIFIED
artifacts: $S/ab/AB7/F-G00-3 (neg/, w/)

### F-G00-4 — ave/histo/kk bins global scalars (compute/fix scalar, equal-style var) via host bin_one() atomics on device views
class: gpu-only
positive control: n/a on host backend (OpenMP device views are host memory, so host atomics are legal; bug = host write to device memory on CUDA/HIP)
negative control: `fix ave/histo 1 10 10 0 12000 6 v_n` (v_n equal np) + `fix ave/histo 1 10 10 0 400 8 c_t` (compute temp), ave running, 50 steps | A vs B: identical (stats + both histogram files, t1 and t4); B histogram of np is plausible vs CPU (50 values, min 0, max 10508 vs CPU 10529; same bin distribution within 1 count); new device bin_scalar path gives same min/max/counts as old host path
verdict: UNTESTABLE-HERE (gpu-only; negative control A==B identical)
artifacts: $S/ab/AB7/F-G00-4

### F-G00-12 — ave/histo/weight/kk per-particle (variable) input with `region` never binned (kernels commented out)
class: cpu-observable
positive control: `variable vx particle vx`, `variable one particle 1.0`; `fix ave/histo/weight 10 1 10 -2000 2000 8 v_vx v_one region left mode vector` (+ same with `mix sub`, sub = N only) | A: histogram always empty, count 0, min/max inf/-inf every step (t1,t4) | B: step 50 t1 count 10442 (region-only), 5331 (region+mix), min 9.16 max 1663.9; B histogram for v_vx+region is bin-for-bin identical to B/A explicit-attribute `vx`+region path; CPU region+mix reference 5337 of 10529 (statistically consistent) | REPRODUCED
negative control: same deck, `v_vx v_one mix sub` (no region, BinParticles3) and `vx v_one region left` (X/V path) | A vs B: stats identical t1/t4; histogram files identical except the step-0 empty-histogram min/max line (inf/-inf -> 1e+20/-1e+20, the intended F-G14-1 change)
side finding (CPU, not in scope): CPU FixAveHistoWeight::bin_particles(double*,int) (src/fix_ave_histo_weight.cpp:412) segfaults with `region` and no `mix` (reads particle->mixture[imix] with imix unset); CPU reference therefore only available for region+mix.
verdict: VERIFIED
artifacts: $S/ab/AB7/F-G00-12 (neg/)

### F-G14-1 (= G12x-F-G14-2) — ave/histo/kk empty histogram reports min/max as Kokkos identities instead of CPU +-1e20
class: cpu-observable
positive control: `region right block 9 10 ...` (no particles reach it in 30 steps); `fix ave/histo 10 1 10 -2000 2000 8 vx region right mode vector` | A: f_he[3]/[4] and file header = inf / -inf every output (t1,t4) | B: 1e+20 / -1e+20 | CPU: 1e+20 / -1e+20 | REPRODUCED
negative control: same fix with `region left` (non-empty, 2106..6303 values) | A vs B: stats identical (count/min/max, t1/t4); files identical except the step-0 header line where the histogram is still empty (inf -> 1e+20, intended)
verdict: VERIFIED
artifacts: $S/ab/AB7/F-G14-1
