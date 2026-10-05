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
necessary: NOT SHOWN - on OpenMP the device views alias host memory, so A's host atomics are correct; failure needs CUDA/HIP without UVM.
complete: yes on host for all three call sites: compute scalar (c_t, A==B identical), fix scalar (`fix gc grid/check/kk 1 error` -> f_gc, beyond extra) and equal-style var (`v_s equal step`, ave window 2, beyond end): B (and A) histogram files bit-identical to CPU for t1 and t4; analytic check window steps 11..30 -> bins 0,0,4,5,11, min 11 max 30. Nrepeat>1 covered. Weight style: global scalar inputs are rejected by calculate_weights() (error), so no extra site; grep: no remaining host bin_one(minmax,...) call in fix_ave_histo*_kokkos.cpp.
verdict: NOT-SHOWN-NECESSARY (gpu-only; B correct on every host call site, A==B==CPU)
artifacts: $S/ab/AB7/F-G00-4 (complete/)

### F-G00-12 — ave/histo/weight/kk per-particle (variable) input with `region` never binned (kernels commented out)
class: cpu-observable
positive control: `variable vx particle vx`, `variable one particle 1.0`; `fix ave/histo/weight 10 1 10 -2000 2000 8 v_vx v_one region left mode vector` (+ same with `mix sub`, sub = N only) | A: histogram always empty, count 0, min/max inf/-inf every step (t1,t4) | B: step 50 t1 count 10442 (region-only), 5331 (region+mix), min 9.16 max 1663.9; B histogram for v_vx+region is bin-for-bin identical to B/A explicit-attribute `vx`+region path; CPU region+mix reference 5337 of 10529 (statistically consistent) | REPRODUCED
negative control: same deck, `v_vx v_one mix sub` (no region, BinParticles3) and `vx v_one region left` (X/V path) | A vs B: stats identical t1/t4; histogram files identical except the step-0 empty-histogram min/max line (inf/-inf -> 1e+20/-1e+20, the intended F-G14-1 change)
side finding (CPU, not in scope): CPU FixAveHistoWeight::bin_particles(double*,int) (src/fix_ave_histo_weight.cpp:412) segfaults with `region` and no `mix` (reads particle->mixture[imix] with imix unset); CPU reference therefore only available for region+mix.
necessary: yes - A bins nothing (count 0) for v_ particle-variable + region, with and without mix (t1,t4).
complete: yes - both re-enabled kernels (BinParticles1 region+mix, BinParticles2 region-only) with a non-trivial weight `v_w = 1+0.001*vy`: B t1 and t4 histogram files bit-identical to the independent explicit-attribute path (`vx v_w region left [mix sub]`, X/V kernels) in the same run (A differs: a1!=a2, b1!=b2); counts statistically = CPU (t1 3205.9/6250.9 wt, CPU 3224.3/6305.0). Sibling base ave/histo/kk v_vx region+mix already worked (A and B c1==c2). Per-particle compute input (c_ke) is rejected by ave/histo(/weight)/kk in A and B ("Compute kind not compatible"), so particle-style variable is the only reachable site.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB7/F-G00-12 (neg/, complete/)

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

### F-G14-5 — ave/grid/kk init() lacks CPU `nglocal = grid->nlocal; grow_percell(0)`; read_surf after the fix -> split sub-cells untallied / reads past allocation
class: cpu-observable
positive control: examples/spiky deck (20x20, read_surf data.spiky -> 400 -> 483 cells incl. split sub-cells), `fix ag ave/grid all 1 1 10 c_g[1]` defined BEFORE read_surf, `compute reduce sum f_ag` vs np, 200 steps | A: c_r = 2.7616126e+267 garbage at every output (t1; t4 same, 82 s under shared-machine load) | B: c_r == np exactly every output (t1 33079, t4 33180) = invariant; CPU same invariant | REPRODUCED
negative control: fix ave/grid defined AFTER read_surf | A vs B: identical stats t1 and t4 (c_r == np)
necessary: yes - A garbage (vector variant) and segfault rc=139 (array variant below).
complete: yes - B correct for per-grid vector (c_g[1]) and array with a post-processed compute (`c_g[1] c_th[1]`, compute thermal/grid temp): f_ag[1] sum == np at t1/t4, max temp plausible vs CPU, dump grid of f_ag[*] over all 483 cells works (A segfaults on this input); t1 and t4. Not tested: real-MPI N procs (MPI builds not present), 3d.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB7/F-G14-5 (neg/, complete/)

### F-G14-6 — ave/grid/kk grow_percell grows maxgrid by a single DELTAGRID (1024) even when nglocal+nnew needs more
class: cpu-observable
positive control: 50x50 box, 100x100 grid (10000 cells), `fix ag ave/grid all 1 1 10 c_g[1] c_g[1]` defined, then 25x `read_surf data.spiky trans ... scale 0.4 0.4 1` before the first run -> 12075 cells (2075 new split sub-cells > 1024), `compute reduce sum f_ag[1]/[2]` vs np, 50 steps. Since A also lacks F-G14-5, a scratch variant V = B with only this fix reverted (single `maxgrid += DELTAGRID;`, scratch-compiled fix_ave_grid_kokkos.cpp linked against the B build objects; $S/ab/AB7/F-G14-6/variant) isolates F-G14-6 | A: segfault rc=139 (t1,t4) | V: segfault rc=139 (t1,t4), gdb: compute_reduce.cpp:829 reading f_ag array sized 11024 < nglocal 12075 | B: rc=0, c_r == c_r2 == np at every output (t1 5218, t4 5292); CPU same invariant (12075 cells) | REPRODUCED
negative control: same 25-surf deck with fix ave/grid defined after read_surf | A vs B vs V: stats identical t1 and t4. F-G14-5 spiky deck (83 new cells < 1024): V == B identical.
necessary: yes - shown with the isolated variant V (B minus F-G14-6 crashes; B fine). Other route (unpack_grid_one with nsplit>1024 via `global splitmax` + migration) not exercised: needs one cell split into >1024 sub-cells.
complete: yes - single growth site in fix_ave_grid_kokkos.cpp, now identical to CPU; covers grow_percell(0) from init and grow_percell(nsplit) (same code). Vector/array outputs fine (array tested). Sibling sweep: CollideVSSKokkos::grow_percell already uses while; no other KOKKOS per-grid grower with a single bump (emit ntaskmax single bumps are guarded per-task, different pattern). Also checked: adapt_grid refine (2x2 / 4x4, 1600 / 6400 cells) before first run is NOT a trigger - A, B, V all correct there (adapt path notifies the fix).
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB7/F-G14-6 (variant/, neg/, in.c2/in.c4 adapt checks)

### F-G00-20 (dup F-G14-7) — ave/grid/kk init() has no CUSTOM case (value2index=-1 -> etype[-1]/ewhich[-1]); nvalues>1, j>0 INT custom read from edarray
class: unreachable
positive control: stock parser accepts only c_/f_/v_ (stock B: "No values in fix ave/grid command" for g_ inputs), so the CUSTOM branch is dead. Instrumented scratch builds A'/B' (copy of src/fix_ave_grid.cpp whose parser maps `g_name` -> which=CUSTOM, + flavor_pergrid=1; linked against the A resp. B build objects; $S/ab/AB7/F-G00-20/variant). Deck: 20x20 grid, custom grid ivec=floor(2*cxlo)+1, iarray=[3,7]*ivec, dvec=cylo+0.25, darray=[2,3,5]*dvec; `fix ave/grid all 1 5 10` with f1 g_ivec | f2 g_iarray[2] | f3 g_dvec g_iarray[2] g_darray[3] g_ivec | f4 g_darray[1]; reduce sums. Analytic: 4200 | 29400 | 2000 29400 10000 4200 | 4000 | A': 4200 | 29400 | 4200 6000 10000 4200 | 12600 (wrong attribute picked for f3[1] and f4; INT array col read as double -> 6000) t1 and t4 | B': exact analytic values t1 and t4, == CPU-style (B' -in) | REPRODUCED (instrumented only)
negative control: stock A vs B, `fix ave/grid all 1 5 10 v_gdvec v_giarray2 c_g[1]` | A vs B: identical t1/t4 (2000 / 29400 analytic; c_g[1] sum = np-average, CPU 3787.4 vs 3783.2)
necessary: not in the stock code (unreachable from input); the latent defect is real and demonstrated with the instrumented parser (A' wrong on 3 of 6 values).
complete: yes - B' correct on all CUSTOM branches: nvalues==1 INT vec / INT array j>0 / DOUBLE array j>0, and nvalues>1 DOUBLE vec, INT array j>0 (the eiarray fix), DOUBLE array j>0, INT vec; t1 and t4.
verdict: NOT-SHOWN-NECESSARY (unreachable: parser never produces CUSTOM; with an instrumented parser A' is wrong and B' == CPU == analytic on every branch)
artifacts: $S/ab/AB7/F-G00-20 (variant/, neg/)

## AB7 summary
| ID | verdict |
|---|---|
| F-G00-3 | NECESSARY+COMPLETE (A segfault on group-masked per-grid input; B == CPU bit-exact, both styles, all 5 input sites) |
| F-G00-4 | NOT-SHOWN-NECESSARY (gpu-only; B == A == CPU on compute/fix/variable scalar sites) |
| F-G00-12 | NECESSARY+COMPLETE (A empty histogram; B == independent X/V path bit-exact, region and region+mix) |
| F-G14-1 (= G12x-F-G14-2) | NECESSARY+COMPLETE (A inf/-inf; B == CPU 1e+20/-1e+20, all ave modes, both styles) |
| G12x-F-G14-1 | INCOMPLETE (ave/histo(/weight)/kk fixed: A segfault -> B clean error; siblings compute fft/grid/kk and lambda/grid/kk still segfault on isurf/grid/kk input in B) |
| F-G14-5 | NECESSARY+COMPLETE (A garbage 2.8e267 / segfault; B reduce == np invariant) |
| F-G14-6 | NECESSARY+COMPLETE (isolated variant B-minus-fix segfaults with 2075 new split cells; B correct) |
| F-G00-20 | NOT-SHOWN-NECESSARY (unreachable; instrumented parser: A' wrong, B' == CPU == analytic) |
Side finding (CPU, out of scope): src/fix_ave_histo_weight.cpp:412 reads particle->mixture[imix] with imix uninitialized when `region` is used without `mix` -> CPU segfault (observed).

## STATUS: COMPLETE
