# AB7 — fix ave (fix ave/histo(/weight)/kk, fix ave/grid/kk), commit 088431eb
Work dir: $S/ab/AB7 (S = scratchpad). Base deck: 2d 20x20 box, N/O air, emit/face xlo, `compute g grid all all n`.

### F-G00-3 — ave/histo(/weight)/kk group-masked grid binning derefs uninitialized GridKokkos* in kernel
class: cpu-observable
positive control: `group gl grid region left one` (220 of 400 cells) + `fix ave/histo 10 1 10 0 400 8 c_g[1] group gl mode vector` (and ave/histo/weight c_g2[1] c_g2[2] group gl) | A: segfault rc=139 at first end_of_step, t1 and t4, both styles | B: runs, f_h[1]=220 cells/step (weight: sum 41.9 @50) matching CPU (220; 41.99) | REPRODUCED
negative control: same fix without `group` (bins all 400 cells) | A vs B: identical (t1 and t4 stats + histogram files, diff clean)
necessary: yes - A segfaults (t1,t4) on every group-masked per-grid input tried: c_g[1], v_cx (grid var), f_ag (ave/grid vector), f_ag2[2] (ave/grid array col), and ave/histo/weight.
complete: yes - B on all 5 call sites of bin_grid_cells(groupflag) in both styles: deterministic grid-var test `v_cx` (cxlo) group gl, 20 bins 0..10 and ave/histo/weight `v_cx v_cy` (cylo+1): B t1 and t4 histogram files bit-identical to CPU (count 220 = 11 columns x 20, weight 1265 = 11 x sum(0.5j+1) analytic); f_ag / f_ag2[2] with group give 220 counts = CPU. Sibling grep: no other kernel dereferences a GridKokkos* member (grid_kk only in modify_kokkos.h host-side).
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB7/F-G00-3 (neg/, w/, complete/)

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
necessary: yes - A prints inf/-inf where CPU prints 1e+20/-1e+20.
complete: yes - empty histogram in ave one / ave running / ave window 2, ave/histo and ave/histo/weight (region+mix), Nevery/Nrepeat 10 1 and 5 2: B histogram files bit-identical to CPU for t1 and t4 (one: 1e+20/-1e+20 every output; running/window: 0 0 in A, B and CPU alike). Weight style shares the patched end_of_step (no separate site).
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB7/F-G14-1

### G12x-F-G14-1 — ave/histo(/weight)/kk with a kokkos_flag-but-not-KokkosBase per-grid compute (isurf/grid/kk) -> NULL computeKKBase call
class: cpu-observable
positive control: examples/implicit 2d deck (150x150, read_isurf binary.101x101, 30 steps) + `compute is isurf/grid all air n` + `fix ave/histo 10 1 10 0 10 10 c_is[1] mode vector`; weight variant `ave/histo/weight ... c_is[1] c_is[2]` (n nwt) | A: segfault rc=139 (t1,t4, both styles), gdb: fix_ave_histo_kokkos.cpp:244 computeKKBase->compute_per_grid_kokkos() | B: clean "ERROR: Fix ave/histo/kk does not (yet) support this compute" (t1,t4, both styles) | CPU: runs | REPRODUCED
negative control: same deck, `compute g grid all air n` (KokkosBase) with ave/histo/kk | A vs B: identical stats and histogram files (t1 and t4)
necessary: yes - A crashes on both ave/histo/kk and ave/histo/weight/kk.
complete: within fix_ave_histo(_weight)_kokkos yes (init() guard shared by both styles, both tested). B gives an error, not CPU parity (CPU bins the values) - acceptable per verify "minimal" option. SIBLING SITES WITH THE SAME BUG REMAIN in B: `compute fft/grid/kk c_is[1]` -> segfault in A and B (gdb B: compute_fft_grid_kokkos.cpp:184 cKKBase->compute_per_grid_kokkos(), cKKBase NULL); `compute lambda/grid/kk c_is[*] NULL lambda` -> segfault in A and B (gdb B: compute_lambda_grid_kokkos.cpp:142). CPU runs both. fix ave/grid/kk with c_is[1] is fine (has a host PERGRIDSURF path), A==B. react/isurf/grid/kk not exercised (needs implicit surf reactions).
verdict: INCOMPLETE (sibling sites compute fft/grid/kk and compute lambda/grid/kk still NULL-deref a non-KokkosBase isurf/grid/kk compute; the ave/histo fix itself is NECESSARY and works)
artifacts: $S/ab/AB7/G12x-F-G14-1 (neg/, w/, sib/)
