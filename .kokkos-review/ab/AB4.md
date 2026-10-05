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
