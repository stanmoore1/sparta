# AB4 results (grid/misc computes)

Method note: inputs use `timestep 1e-12` + `create_particles ... twopass` so particles are identical in
CPU and Kokkos runs and essentially do not move; with `collide vss` the particles are sorted each step,
so `-k on t 4` (need_atomics=1, atomic_reduction=0 on CPU) selects the sorted per-cell kernels,
`t 1` selects atomic<0> (non-dup), and t 4 without collide selects atomic<1> with ScatterDuplicated (dup).
Kokkos and CPU outputs are compared value by value (dumps with %20.15g; "rel" = max relative diff).
dump grid uses the host post_process_grid; fix ave/grid/kk uses the Kokkos post_process_grid_kokkos.
$S = /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad

### F-G00-5 — sorted compute grid / pflux/grid kernels `return` instead of `continue` on species not in mixture
class: cpu-observable
positive control: N2+O2 box, 20k particles, 4^3 cells, collide vss, `compute grid all onlyO2 n u ke` and `pflux/grid all onlyO2 momxx momyy momxy` vs the O2 columns of the same computes with mixture `species`; t 4 (sorted) | A: sum of onlyO2 n = 42 vs compute count O2 = 10079; onlyO2 columns differ from species-O2 columns by rel 1.80 (grid), 1.23 (pflux), and from CPU by 1.80 | B: sum n = 10079, onlyO2 == species-O2 columns exactly, all columns bit-identical to CPU | REPRODUCED
negative control: (a) same sorted run, mixture `species` (no skipped species) columns: A vs B identical; (b) no collide, t 4 (dup atomic path): A vs B identical, both identical to CPU; (c) t 1: A, B, CPU identical
necessary: yes — A fails on both fixed call sites (compute grid and pflux/grid sorted kernels), t 4 sorted.
complete: B bit-identical to CPU on all 3 kernel paths (sorted t4, atomic t1, dup t4 no-collide) for both computes. Sibling sorted kernels (thermal/grid, eflux/grid, sonine, tvib) use `continue`; tested thermal/grid temp press + eflux/grid heatx heaty with onlyO2 on sorted t4: A and B both == species-O2 columns and == CPU (no sibling bug). grep: no other `return` inside a d_plist per-cell loop in compute_*_kokkos.cpp.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB4/F-G00-5 (in.x, in.sib)

### F-G15-1 — sonine/grid sorted kernel never tallies the mass column (moments shifted)
class: cpu-observable
positive control: N2+O2 box (20k, 4^3 cells, dt 1e-12), collide vss, `sonine/grid all air a x 3 b xy 2` + `sonine/grid all onlyO2 a y 2 b xx 2`, dump grid step 2, t 4 (sorted kernel) | A: max rel err vs CPU 1.99 (s1) / 1.92 (s2); signature exactly the column shift: A out[A1]=CPU_A2/CPU_A1 (988334 = -7.5206e12/-7.6094e6), out[A2]=CPU_A3/CPU_A1, out[A3]=CPU_B1/CPU_A1=169.5, last moment 0 | B: bit-identical to CPU | REPRODUCED
negative control: t 1 (atomic<0>, non-dup) with collide: A, B, CPU all identical
necessary: yes (A wrong on sorted path for both a- and b-moments, both mixtures incl. subset mixture).
complete: B == CPU on sorted t4, atomic t1, dup t4 for a and b keywords, full and subset mixture. Only one sorted tally kernel exists in sonine (vcom sorted kernel tallies mass correctly).
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB4/sonine (dump.grid.{A,B}t{1,4}c1, dump.grid.cpuc1)

### F-G00-7 — sonine/grid dup path normalizes vcom before contribute() (COM = raw mass-weighted sums)
class: cpu-observable
positive control: same input, no collide (unsorted) t 4 -> need_dup ScatterDuplicated path (also step 0 of the collide run) | A: max rel err vs CPU 1.96 (s1) / 2.00 (s2) at steps 0 and 2 | B: bit-identical to CPU | REPRODUCED
negative control: t 1 (non-dup) no collide: A, B, CPU identical
necessary: yes (A wrong on the dup path, both computes).
complete: B == CPU on dup t4, atomic t1 and sorted t4. The only other dup+normalize ordering in the KOKKOS computes would be sonine's own; grid/pflux/thermal/eflux have no two-stage vcom.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB4/sonine (dump.grid.*c0)

### F-G00-6 — tvib/grid kk post_process: shared d_tspecies scratch race + modeflag 2 uses d_groupspecies(index,..)
class: race (a) + cpu-observable/crash (b)
positive control: CO2(4 modes)+N2, mixture mix (one group, 2 species) and species, tvib 3000, dt 1e-14, collide vss; consumer `fix ave/grid all 1 1 2` (kk post_process) of tvib/grid all mix / all species [mode]. (a) race: 20^3 cells, 500k particles, t 4, 5 repeats: smooth (modeflag 0) | A rel vs CPU 0.25,0.22,0.21,0,0.24 | B 0 x5; discrete+fix vibmode (modeflag 1) | A 0.69,0.30,1.0,0,0.50 | B 0 x5. (b) `mode` keyword (modeflag 2), 6^3 cells, 100k particles | A: SIGSEGV at t1 and t4 in post_process (gdb: compute_tvib_grid_kokkos.cpp:433 d_species[d_groupspecies(index,isp)]) | B: == CPU bit-identical (t1, t4; also 20^3 t4 x3) | REPRODUCED
negative control: A t 1 (no concurrency) modeflag 0/1, 6^3 and 20^3: A == B == CPU; dump grid (host post_process) A == B == CPU for all modes.
necessary: yes for both parts (race shown on modeflag 0 and 1; crash on modeflag 2).
complete: B == CPU on modeflag 0, 1, 2 at t1 and t4 (repeats), mixtures mix and species. Siblings: other Kokkos post_process kernels (grid, thermal, eflux, pflux, sonine) write only d_vec[icell] / read d_etally(icell,..) — no shared scratch.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB4/F-G00-6 (in.x, in.race, chk.py)

### F-G00-9 — compute property/surf/kk with strict-subset surf group fills rows 0..nchoose-1 ignoring group
class: cpu-observable
positive control: (2d) data.circle (50 lines) `group sub surf id 10:30`, `property/surf sub id v1x v1y xc yc area normx normy` + vector form `property/surf sub xc`; (3d) data.sphere (1200 tris) `group sub surf id 100:300`, `property/surf sub id v1x v2y v3z v3x v3y xc yc zc area normx normz` + `property/surf sub xc`; dump surf all, run 0; reference = analytic values from the data file (3d) / CPU B (2d) | A kk (t1 and t4): 2d 18/50 rows wrong, 3d 198/1200 rows wrong (rows 1..99 non-zero for out-of-group tris, rows 202..300 zero for in-group tris), array and vector forms | B kk: 0 rows wrong (2d and 3d, t1 and t4, array and vector) | REPRODUCED
negative control: group all (`property/surf all area`): A kk == B kk == analytic (3d) / CPU (2d)
necessary: yes (A wrong in 2d and 3d, both nvalues==1 and nvalues>1 outputs).
complete: B == reference for 2d lines, 3d tris, explicit and `global surfs explicit/distributed` (3d, A 198 wrong / B 0), t1/t4, vector and array. Not tested: >1 MPI rank (MPI builds not available; nsown logic is per-rank and identical to CPU).
side finding (NOT this fix, CPU-only): src/compute_property_surf.cpp pack_v3y/pack_v3z return tris[m].p1[1]/p1[2] instead of p3 -> CPU B v3y/v3z wrong for all in-group tris (201/1200 rows, e.g. tri 150 v3z CPU 0.348 vs true 0.507 = Kokkos). Kokkos (h:71-72) is correct. Unfixed in HEAD; see F-G16-9.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB4/surf (in.2d, in.3d, in.3dd, chk3d.py)

### F-G16-9 — CPU compute property/surf 3d area: second sub3 overwrote p12, p23 uninitialized
class: cpu-observable
positive control: data.sphere (1200 tris), CPU run (no -k), `property/surf all area` and `property/surf sub ... area` | A cpu: area = 0 for all 1200 tris (garbage from uninitialized p23) | B cpu: area == analytic 0.5|p12 x p13| (rel < 1e-9) for all 1200, == Kokkos area (rel 1.2e-15) | REPRODUCED
negative control: 2d (lines) CPU area: A == B == kk; Kokkos area (not changed): A kk == B kk == analytic
necessary: yes (A CPU area 0 for every triangle).
complete: B correct for 3d explicit and explicit/distributed (mytris), in-group and group all. Sibling site in the same file NOT fixed: pack_v3y / pack_v3z (CPU) read p1 instead of p3 (same copy-paste class of bug) -> CPU B v3y/v3z still wrong (201/201 in-group rows). Out of the F-G16-9 claim (area only) but a remaining sibling bug in the CPU property/surf packers.
verdict: NECESSARY+COMPLETE (for area); NEW sibling CPU bug pack_v3y/pack_v3z unfixed (recommend fix: p1 -> p3 at compute_property_surf.cpp:427,444)
artifacts: $S/ab/AB4/surf (dump.surf.cpu{A,B}_3d*, chk3d.py)

### F-G16-1 — lambda/grid (CPU + kk) reads post-processed nrho from column nrhoindex-1 instead of m
class: cpu-observable (CPU and Kokkos)
note: nrho is one argument expanded by expand_args, so j-1 != m needs a ranged wildcard, e.g. `c_g[2*3]` (c_g[*] always gives j-1 == m).
positive control: N2+O2, `compute g grid all species n nrho` (post-process compute, 4 cols), `fix av ave/grid all 1 1 1 c_g[*]`; `lambda/grid c_g[2*3] c_th[1] lambda tau` vs reference `lambda/grid f_av[2*3] c_th[1] lambda tau` (the fix branch reads columns directly); dump step 1 | A: Lc vs Lf rel 1.0 on CPU and kk (e.g. lambda 2.2475e-5 vs 2.0478e-5; m=1 reads column 2 of a 2-column array_grid1, i.e. out of bounds) | B: Lc == Lf exactly on CPU and kk; kk B == CPU B bit-identical | REPRODUCED
negative control: `compute g grid all species nrho` + c_g[*] (j-1 == m): CPU A, kk A, kk B == CPU B, Lc == Lf
necessary: yes (both CPU and Kokkos A wrong with a ranged wildcard).
complete: B correct on CPU and Kokkos (t4), with temp compute. Sibling branches checked by reading: the non-post-process compute branch and the fix branch correctly read column j-1 of the source's own array (CPU and kk). nrho_values==1 branch unchanged.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB4/lambda (in.x, in.neg)

### G12x-F-G16-2 — lambda/grid/kk prewrap host compute_per_grid derefs NULL host arrays (lambdainv, array_grid1, ...)
class: cpu-observable (crash)
positive control: before first run: `compute L lambda/grid c_g[*] <temp> lambda tau` + `adapt_grid all refine value c_L[1] 0.0 1.0 thresh more more maxlevel 2 iterate 1`, then dump grid + run 0; 4 variants: {N2 O2 (nrho_values=2, array_grid1), N2 only (nrho_values=1)} x {temp c_th[1], temp NULL} | A kk t4: SIGSEGV in all 4 variants (gdb: compute_lambda_grid.cpp:473 from AdaptGrid::refine_value) | B kk: no crash, 64 cells refined, dumped c_L (512 cells after adapt) bit-identical to CPU in all 4 variants | REPRODUCED
negative control: CPU runs (A/B CPU unaffected); lambda tests above (after first run) A kk == B kk
necessary: yes (crash in all 4 variants).
complete: for lambda/grid itself yes (all allocation branches: nrho_values>1/==1, temp/NULL, and reallocate after adapt changes nglocal). SIBLINGS NOT FIXED — same pattern (Kokkos reallocate() overrides the base and skips host scratch the prewrap CPU path uses), same adapt_grid-before-run input:
  - compute sonine/grid/kk: A and B SIGSEGV at compute_sonine_grid.cpp:173 (host vcom NULL; kk reallocate creates only d_vcom); CPU OK (32 cells refined).
  - compute dt/grid/kk: A and B SIGSEGV at compute_dt_grid.cpp:460 (host tau/temp/usq/vsq/wsq NULL; kk reallocate creates only d_*_vector); CPU OK (64 refined). Input: dt/grid all 0.1 0.1 c_pp[1..5] with property/grid inputs.
  - checked OK in A and B: grid, thermal/grid, eflux/grid, pflux/grid, tvib/grid, property/grid (kk) via the same adapt_grid-before-run input.
verdict: INCOMPLETE (sibling prewrap NULL-deref remains in compute sonine/grid/kk (vcom) and compute dt/grid/kk (tau,temp,usq,vsq,wsq); lambda/grid part itself is NECESSARY+COMPLETE)
artifacts: $S/ab/AB4/G12x-F-G16-2 (in.x, in.sib, in.dt)

Library-driver note (F-G16-4/6/7/8): these bugs are only reachable through the C library API
(sparta_extract_compute, library.cpp). A small driver ($S/ab/AB4/lib/drv.cpp) was compiled with each build's
flags and linked against the static libs of A ($S/build_base/src) and B ($S/build/src, same binary as
spa_new_final). It loads in.base (N2+O2, 20k particles, 4^3 cells, `compute cnt count N2 O2`,
`compute pg property/grid all xc yc`, `compute ke ke/particle`), then calls sparta_extract_compute and
inspects the compute's invoked_* members.

### F-G16-4 — compute count (CPU + kk) compute_vector() set invoked_scalar instead of invoked_vector
class: cpu-observable (library API only)
positive control: driver `flags`: run 0, sparta_extract_compute("cnt",0,1) | A: invoked_vector -1, invoked_scalar 0 (CPU and kk) | B: invoked_vector 0 (== step), invoked_scalar -1 (CPU and kk) | REPRODUCED
negative control: vector values A == B == CPU (10039 9961)
necessary: yes (wrong flag on both CPU and Kokkos); impact is only redundant recomputation in sparta_extract_compute.
complete: B correct on CPU and kk. Sibling check: grep of compute_vector() bodies in src/ and src/KOKKOS for `invoked_scalar =` found no other compute_vector setting the scalar flag.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB4/lib

### F-G16-8 — compute property/grid/kk never set invoked_per_grid
class: cpu-observable (library API only)
positive control: driver `flags`: sparta_extract_compute("pg",2,2) after run 0 | A kk: invoked_per_grid -1 (CPU 0) | B kk: 0 | REPRODUCED
negative control: CPU A == B (0); values identical
necessary: yes (flag never set in A kk; consequence is re-invocation on every library call only).
complete: B sets it at the top of compute_per_grid_kokkos (covers both prewrap and kk paths reached via compute_per_grid). Siblings: other KOKKOS per-grid computes (grid, thermal, tvib, sonine, pflux, eflux, lambda, distsurf, dt, fft) set invoked_per_grid (grep).
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB4/lib

### F-G16-6 — compute ke/particle/kk: no invoked_per_particle, no modify_device/sync_host for host consumers
class: gpu-only (sync part) + cpu-observable (invoked flag, library API)
positive control: driver `flags`: sparta_extract_compute("ke",1,1) after run 0 | A kk: invoked_per_particle -1 | B kk: 0 | REPRODUCED (flag part). Host-staleness part cannot occur on OpenMP (host and device views alias).
negative control: sum of extracted ke A kk == B kk == CPU (1.29286550629737e-16)
necessary: flag part yes; host-sync part NOT-SHOWN (gpu-only).
complete: flag + values correct in B; sync part untestable here.
verdict: NECESSARY, COMPLETENESS-PARTIAL (host sync_host/modify_device part gpu-only, untested)
artifacts: $S/ab/AB4/lib

### F-G16-7 — compute ke/particle/kk: prewrap host call sets nmax, later kk call writes into empty device view
class: cpu-observable (crash; library API only)
positive control: driver `prewrap_ke`: sparta_extract_compute("ke",1,1) BEFORE the first run (prewrap -> host ke allocated, sum 1.29286550629737e-16), then `compute r reduce sum c_ke` + run 0 (reduce/kk -> compute_per_particle_kokkos) | A kk: SIGSEGV in ComputeKEParticleKokkos::operator() (compute_ke_particle_kokkos.cpp:101, write to 0-extent d_vector_particle) | B kk: no crash, reduce sum 1.29286550629736e-16, extract after run sum 1.29286550629737e-16, clean exit (no double free) | REPRODUCED
negative control: CPU A == CPU B (same sums); without the prewrap extract (driver `flags`) A kk works
necessary: yes (A crashes).
complete: B correct for the kk path after prewrap (reduce/kk and library extract) and clean shutdown (ke = NULL after destroy_kokkos). Sibling: other per-particle KOKKOS computes with a prewrap host fallback — only ke/particle has per-particle output in KOKKOS (grep compute_*_kokkos.cpp per_particle).
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB4/lib

### F-G00-2 — compute ke/particle/kk operator() reads host `update->mvv2e` on device; missing modify_device
class: gpu-only
positive control: n/a on this machine (host pointer deref is legal on OpenMP; host/device alias, so modify_device is a no-op)
negative control: ke/particle values: library extract sums A kk == B kk == CPU (1.29286550629737e-16), also the reduce/kk sum after run (B) matches CPU
necessary: NOT-SHOWN (gpu-only).
complete: n/a on CPU backend; B uses the member mvv2e (code read) and values unchanged.
verdict: NOT-SHOWN-NECESSARY (gpu-only); negative control A == B == CPU
artifacts: $S/ab/AB4/lib

### F-G00-14 — compute reduce/kk reads fix ave/grid d_vector_grid/d_array_grid without sync_pergrid_device_kokkos()
class: gpu-only (on host backends) / mpi
positive control attempted: fix ave/grid all 1 2 2 c_g[*] c_pg (+ reduce/kk sum/max of f_av columns) with fix adapt refine at step 2 (64->512 cells) and refine+coarsen every 2/4 steps, t1/t4 | A == B == CPU at every output step (e.g. step 4: r1 20000, r2 2.54884181362022e-23, r3 1e-12) | NOT REPRODUCED. Reason: on OpenMP host and device views alias, and FixAveGridKokkos::grow_percell re-points d_vector_grid/d_array_grid itself, so the cached handle is never stale; staleness needs separate device memory (GPU). MPI builds ($S/bmpi_*) not finished (make at 28%), so a multi-rank balance test was not possible.
negative control: no adapt, t1/t4: A == B == CPU (r1 20000, r2 3.18637472966332e-24 ...)
necessary: NOT-SHOWN (gpu-only on host backends).
complete: B == CPU in all tested adapt variants; MPI multi-rank and GPU untested.
verdict: NOT-SHOWN-NECESSARY (gpu-only: host/device alias on OpenMP); negative controls A == B == CPU
artifacts: $S/ab/AB4/F-G00-14 (in.x, in.rc)

### F-G16-5 — compute distsurf/grid/kk does not sync grid cells/cinfo to device before use (second run after grid change)
class: gpu-only
positive control attempted: sphere (1200 tris), 8^3 grid, run 1; adapt_grid refine surf (80 cells refined) + balance_grid; new dump grid c_distsurf, run 1 (output at second-run setup) | A == B == CPU at both steps (1072 cells, rel 0), t1 and t4 | NOT REPRODUCED (k_cells/k_cinfo host and device alias on OpenMP; sync is a no-op)
negative control: same input A vs B vs CPU identical (first run 512 cells, second run 1072 cells)
necessary: NOT-SHOWN (gpu-only).
complete: B == CPU on CPU backend; GPU untested.
verdict: NOT-SHOWN-NECESSARY (gpu-only); negative control A == B == CPU
artifacts: $S/ab/AB4/F-G16-5

## Summary AB4
| ID | verdict |
|---|---|
| F-G15-1 | NECESSARY+COMPLETE (sonine sorted kernel; A shifted moments, B == CPU) |
| F-G00-5 | NECESSARY+COMPLETE (grid+pflux sorted; A undercounts 42 vs 10079; siblings thermal/eflux OK) |
| F-G00-6 | NECESSARY+COMPLETE (tvib race A rel up to 1.0 at t4 modeflag 0/1; modeflag 2 A segfault; B == CPU) |
| F-G00-7 | NECESSARY+COMPLETE (sonine dup path; A rel ~2, B == CPU) |
| F-G00-2 | NOT-SHOWN-NECESSARY (gpu-only); A == B == CPU |
| F-G00-9 | NECESSARY+COMPLETE (property/surf/kk subset group; 2d/3d/distributed, vector/array) |
| F-G00-14 | NOT-SHOWN-NECESSARY (gpu-only on host backends; MPI builds unavailable); A == B == CPU |
| G12x-F-G16-2 | INCOMPLETE — lambda/grid fixed (A segfault, B == CPU, 4 variants) but same prewrap NULL-deref remains in compute sonine/grid/kk (vcom) and dt/grid/kk (tau/temp/usq/vsq/wsq): B segfaults with adapt_grid before first run, CPU OK |
| F-G16-1 | NECESSARY+COMPLETE (CPU + kk; needs ranged wildcard c_g[2*3]) |
| F-G16-4 | NECESSARY+COMPLETE (CPU + kk; library API) |
| F-G16-5 | NOT-SHOWN-NECESSARY (gpu-only); A == B == CPU |
| F-G16-6 | NECESSARY, COMPLETENESS-PARTIAL (invoked flag shown; host-sync part gpu-only) |
| F-G16-7 | NECESSARY+COMPLETE (library prewrap extract -> A segfault, B correct) |
| F-G16-8 | NECESSARY+COMPLETE (library API invoked_per_grid) |
| F-G16-9 | NECESSARY+COMPLETE for area (A CPU area 0 for all tris); NEW unfixed sibling: CPU pack_v3y/pack_v3z read p1 instead of p3 (compute_property_surf.cpp:427,444) |

## STATUS: COMPLETE
