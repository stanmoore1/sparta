# AB5 — surf tally computes / temp rescale (A/B results)
A_opt = build_base/src/spa_kokkos_omp (e071055f), B_opt = spa_new_final; Kokkos runs `-k on t N -sf kk`, CPU ref = same binary without -sf kk. $S = scratchpad.
Entries follow the GOAL CLARIFICATION (necessary / complete).

### F-G00-16 — fix temp/rescale/kk per-cell: vscale=sqrt(Tt/0)=inf for cold cell -> NaN velocities
class: cpu-observable
positive control: 3d box 4^3 cells, 1000 Ar, mixture vstream 1 0 0 temp 0 (all v identical), fix temp/rescale 1 300 300 ave no, run 3 | A: kk c_temp/c_ke = -nan from step 1 (t1 and t4); A cpu = 0.0016006965 / 3.315e-23 | B: kk = 0.0016006965 / 3.315e-23 == CPU ref (t1, t4) | REPRODUCED
negative control: temp 300 gas, temp/rescale 1 600 600 ave no | A vs B: identical (kk t1/t4 and cpu, T=634.68801 at step 3)
necessary: yes — A-kk NaN in every particle velocity on cold cells (t1, t4); CPU ref finite.
complete: fix covers both kk sites (per-cell kernel line 140 and ave path line 215, see F-G18-3); B==CPU ref bit-identical at t1 and t4. Sibling search: only other `sqrt(t_target/t_current)` is fix temp/global/rescale(/kk), which already returns when t_current==0 and has t_target>=0 -> not a sibling bug. Not run with N MPI ranks (per-cell kernel has no communication).
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB5/F-G00-16 (in.base, run.sh, run4.sh, log.{t4_,}pos_cell.*, log.{t4_,}neg_cell.*)

### F-G18-3 — fix temp/rescale/kk ave mode: t_current==0 -> vscale=sqrt(0/0)=NaN written to all particles
class: cpu-observable
positive control: cold gas (vstream 1 0 0 temp 0), fix temp/rescale 1 0 0 ave yes (t_one=t_target=0 everywhere) | A: kk -nan from step 1 (t1, t4); cpu 0.0016006965 | B: kk 0.0016006965 == CPU ref (t1, t4) | REPRODUCED
negative control: temp 300 gas, temp/rescale 1 600 600 ave yes (t1, t4), and 1 0 0 ave yes on warm gas | A vs B: identical (kk and cpu; T=635.29137 resp. 16.558581 at step 3)
necessary: yes — A-kk NaN at t1 and t4.
complete: ave path is the only site; B==CPU ref at t1/t4. Multi-rank (MPI_Allreduce of t_current) not run: MPI builds were not available; the guard is applied after the Allreduce so rank count cannot change it.
verdict: NECESSARY, COMPLETENESS-PARTIAL (N-rank MPI run not done; MPI builds still compiling)
artifacts: $S/ab/AB5/F-G00-16 (log.{t4_,}pos_ave.*, log.{t4_,}neg_ave.*, log.pos_ave_warm.*)

### F-G00-16-note — temp/rescale ave mode: t_current/=n_current with n_current==0 (CPU + kk)
class: unreachable
positive control: n/a — n_current counts every local cell with nsplit<=1 (unsplit + sub cells, empty ones included), summed over all ranks; at least one such cell always exists, so n_current==0 cannot occur with any valid grid | A: n/a | B: n/a | n/a
negative control: ave-mode runs above (cold T=0 and warm gas, kk t1/t4 and cpu) | A vs B: identical
necessary: not shown — no valid input reaches n_current==0.
complete: both sites (CPU fix_temp_rescale.cpp and kk) carry the guard (diff 826bf094); no other `/= n_current` sites.
verdict: NOT-SHOWN-NECESSARY (unreachable: there is always at least one counted cell); negative control passes
artifacts: $S/ab/AB5/F-G00-16

### F-G17-1 — compute surf(/kk) TX/TY/TZ read NULL vorig when fix emit/surf tallies (no FX before it)
class: cpu-observable
positive control: 2d circle (examples/emit data.circle), fix emit/surf normal yes, compute surf all all tx ty tz com 0 0 0, fix ave/surf every 1 | A: SIGSEGV (rc 139) in kk and CPU | B: runs; torques bit-identical to A's guarded-path reference (`fx fy fz tx ty tz`, where FX sets fflag first) | REPRODUCED
negative control: same deck with `fx fy fz tx ty tz` | A vs B: identical (kk and cpu, all 6 columns, every stats line)
necessary: yes — A segfaults on all 48 positive variants (below).
complete: matrix {CPU, kk t1, kk t4} x {perspecies yes,no} x {onepass, twopass} (= all 4 CPU call sites fix_emit_surf.cpp:840/958/1163/1268 and the kk site :882) x {tx alone, ty alone, tz alone, tx ty tz}: 48/48 B runs give torque columns bit-identical to A's fx-first reference; A rc=139 in 48/48 (the "n tN" single-value variants use `n` first so no fflag is set). Sibling search: unguarded `scale3(-origmass,vorig,...)` remains in compute_boundary(.cpp,_kokkos.h) and compute_isurf_grid_kokkos.h FX/FY/FZ, but those are only called from the move with a non-NULL iorig (boundary_tally update.cpp:1456; emit/surf errors on implicit surfs, fix_emit_surf.cpp:93), and surf_react_adsorb's NULL-iorig call returns early (`if (!iorig && reaction) return;`) -> unreachable, not a missed site. 3d (tris) not run; the TX code is dimension-independent.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB5/F-G17-1 (in.base, in.var, var.sh, var.out, var2.out, var/log.*)

### F-G17-2 — compute surf/kk and boundary/kk store mvv2e as int (latent truncation)
class: unreachable
positive control: n/a — update.cpp sets mvv2e = 1.0 for both cgs and si, so int truncation never changes a value | A: n/a | B: n/a | n/a
negative control: 2d circle, emit/face + create_particles, compute surf all all n ke etot (ave/surf) + compute boundary all n ke, run 100 | A vs B: identical stats (kk and cpu, 11 lines each); kk ke values agree with CPU ref within noise (xlo ke step100 4.39e-19 kk vs 3.74e-19 cpu)
necessary: not shown — mvv2e==1.0 in every unit system.
complete: both int declarations changed (compute_surf_kokkos.h, compute_boundary_kokkos.h); grep shows compute_isurf_grid_kokkos.h already double; no other int mvv2e in src/KOKKOS.
verdict: NOT-SHOWN-NECESSARY (unreachable: mvv2e==1.0); negative control passes
artifacts: $S/ab/AB5/F-G17-2

### F-G17-5 — compute react/surf & react/isurf/grid init(): `return` on first surf not in group skips warning, clear(), combined=0 (and collective Allreduce)
class: cpu-observable (CPU base class, inherited by kk)
positive control: (a) 2d circle, 50 lines, surf_react 2 on ids 1-25, surf_react 3 on 26-50, compute react/surf g 2 with group g = ids 2:50 (lines[0] not in group), run 50 + run 30 | A (cpu and kk): no "25 surfs are not assigned" warning; setup row of run 2 (step 50) shows stale c_r = 16 (cpu) / 22 (kk) | B: warning printed; setup row c_r = 0, identical to the reference deck where lines[0] is in the group (g = 1:50) | REPRODUCED
  (b) 3d sphere (1200 tris), same scheme, g = 2:1200 | A: no warning, stale setup c_r = 37 (cpu) / 32 (kk) | B: "600 surfs are not assigned" warning, setup c_r = 0 == reference g=1:1200 | REPRODUCED
  (c) implicit 2d (ablation binary.101x101, 150^2 grid) and 3d (binary.21x21x21), surf group sg = half-domain region, sg react 3, rest react 2, compute react/isurf/grid inner 2 | A (cpu, kk): no warning | B: "3986/4020 surfs (2d), 10842/10498 surfs (3d) are not assigned" warning, cpu and kk | REPRODUCED
negative control: group containing lines[0]/tris[0] (g = 1:50, 1:1200) | A vs B: identical stats and warnings (cpu and kk, 2d and 3d)
necessary: yes — A skips the warning and leaves stale tallies after re-init on all four CPU loops (react/surf lines+tris, react/isurf/grid lines+tris), visible in both CPU and kk runs.
complete: all 4 changed loops exercised (2d+3d x explicit+implicit), CPU and kk t1; B==reference in each. The multi-rank deadlock aspect (one rank returning before the collective MPI_Allreduce) not run: MPI builds ($S/bmpi_*) still compiling (26%) at test time.
verdict: NECESSARY, COMPLETENESS-PARTIAL (N-rank deadlock variant not run, MPI builds unavailable)
artifacts: $S/ab/AB5/F-G17-5 (2d explicit), F-G17-5/3d, F-G17-5/isurf, F-G17-5/isurf3d
