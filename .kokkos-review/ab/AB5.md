# AB5 — surf tally computes / temp rescale (A/B results)
A_opt = build_base/src/spa_kokkos_omp (e071055f), B_opt = spa_new_final; Kokkos runs `-k on t 1 -sf kk`, CPU ref = same binary without -sf kk.

### F-G00-16 — fix temp/rescale/kk per-cell: vscale=sqrt(Tt/0)=inf for cold cell -> NaN velocities
class: cpu-observable
positive control: 3d box 4^3 cells, 1000 Ar, mixture vstream 1 0 0 temp 0 (all v identical), fix temp/rescale 1 300 300 ave no, run 3 | A: kk c_temp/c_ke = -nan from step 1; A cpu = 0.0016007 / 3.315e-23 | B: kk = 0.0016006965 / 3.315e-23 == CPU ref | REPRODUCED
negative control: same box, temp 300 gas, temp/rescale 1 600 600 ave no | A vs B: identical (kk and cpu, T=634.68801 at step 3)
verdict: VERIFIED
artifacts: $S/ab/AB5/F-G00-16 (in.base, run.sh, log.pos_cell.*, log.neg_cell.*)

### F-G18-3 — fix temp/rescale/kk ave mode: t_current==0 -> vscale=sqrt(0/0)=NaN written to all particles
class: cpu-observable
positive control: cold gas (vstream 1 0 0 temp 0), fix temp/rescale 1 0 0 ave yes (t_one=t_target=0 everywhere) | A: kk -nan from step 1; cpu 0.0016007 | B: kk 0.0016006965 == CPU ref | REPRODUCED
negative control: temp 300 gas, temp/rescale 1 600 600 ave yes, and 1 0 0 ave yes on warm gas | A vs B: identical (kk and cpu; T=635.29137 resp. 16.558581 at step 3)
verdict: VERIFIED
artifacts: $S/ab/AB5/F-G00-16 (log.pos_ave.*, log.neg_ave.*, log.pos_ave_warm.*)

### F-G00-16-note — temp/rescale ave mode: t_current/=n_current with n_current==0 (CPU + kk)
class: unreachable
positive control: n/a — n_current counts every local cell with nsplit<=1 (unsplit + sub cells, empty ones included), summed over all ranks; at least one such cell always exists, so n_current==0 cannot occur with any valid grid | A: n/a | B: n/a | n/a
negative control: ave-mode runs above (cold T=0 and warm gas, kk and cpu) | A vs B: identical
verdict: UNTESTABLE-HERE (unreachable)
artifacts: $S/ab/AB5/F-G00-16

### F-G17-1 — compute surf(/kk) TX/TY/TZ read NULL vorig when fix emit/surf tallies (no FX before it)
class: cpu-observable
positive control: 2d circle (examples/emit data.circle), fix emit/surf normal yes, compute surf all all tx ty tz com 0 0 0, fix ave/surf every 1, run 200 | A: SIGSEGV (rc 139) in both kk and CPU at step 0-1 | B: runs; torques B-cpu step200 = -4.1334529e-22 3.1170531e-22 -9.3089316e-23, identical to A-cpu's torques when fx fy fz precede tx (guarded path); B-kk likewise identical to A-kk's fx-first torques (-1.6119225e-22 2.6878033e-22 2.3659846e-22) | REPRODUCED
negative control: same deck with `fx fy fz tx ty tz` (fflag set by guarded FX branch) | A vs B: identical (kk and cpu, all 6 columns, every stats line)
verdict: VERIFIED
artifacts: $S/ab/AB5/F-G17-1

### F-G17-2 — compute surf/kk and boundary/kk store mvv2e as int (latent truncation)
class: unreachable
positive control: n/a — update.cpp sets mvv2e = 1.0 for both cgs and si, so int truncation never changes a value; no input can produce a difference | A: n/a | B: n/a | n/a
negative control: 2d circle, emit/face + create_particles, compute surf all all n ke etot (ave/surf) + compute boundary all n ke, run 100 | A vs B: identical stats (kk 11 lines, cpu 11 lines); kk ke values agree with CPU ref within noise (xlo ke step100 4.39e-19 kk vs 3.74e-19 cpu, same magnitude)
verdict: UNTESTABLE-HERE (unreachable: mvv2e==1.0) + negative control passes
artifacts: $S/ab/AB5/F-G17-2
