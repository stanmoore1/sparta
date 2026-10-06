# AB5 — surf tally computes / temp rescale (A/B results)
A_opt = build_base/src/spa_kokkos_omp (e071055f), B_opt = spa_new_final; Kokkos runs `-k on t N -sf kk`, CPU ref = same binary without -sf kk. $S = scratchpad.
Entries follow the GOAL CLARIFICATION (necessary / complete).

### F-G00-16 — fix temp/rescale/kk per-cell: vscale=sqrt(Tt/0)=inf for cold cell -> NaN velocities
class: cpu-observable
positive control: 3d box 4^3 cells, 1000 Ar, mixture vstream 1 0 0 temp 0 (all v identical), fix temp/rescale 1 300 300 ave no, run 3 | A: kk c_temp/c_ke = -nan from step 1 (t1 and t4); A cpu = 0.0016006965 / 3.315e-23 | B: kk = 0.0016006965 / 3.315e-23 == CPU ref (t1, t4) | REPRODUCED
negative control: temp 300 gas, temp/rescale 1 600 600 ave no | A vs B: identical (kk t1/t4 and cpu, T=634.68801 at step 3)
necessary: yes — A-kk NaN in every particle velocity on cold cells (t1, t4); CPU ref finite.
complete: fix covers both kk sites (per-cell kernel line 140 and ave path line 215, see F-G18-3); B==CPU ref bit-identical at t1 and t4. Sibling search: only other `sqrt(t_target/t_current)` is fix temp/global/rescale(/kk), which already returns when t_current==0 and has t_target>=0 -> not a sibling bug. 4-rank MPI run (A_mpi/B_mpi, see F-G18-3 addendum): pos cell A-kk -nan, B-kk == cpu 0.0016006965; neg cell all 631.41428.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB5/F-G00-16 (in.base, run.sh, run4.sh, log.{t4_,}pos_cell.*, log.{t4_,}neg_cell.*)

### F-G18-3 — fix temp/rescale/kk ave mode: t_current==0 -> vscale=sqrt(0/0)=NaN written to all particles
class: cpu-observable
positive control: cold gas (vstream 1 0 0 temp 0), fix temp/rescale 1 0 0 ave yes (t_one=t_target=0 everywhere) | A: kk -nan from step 1 (t1, t4); cpu 0.0016006965 | B: kk 0.0016006965 == CPU ref (t1, t4) | REPRODUCED
negative control: temp 300 gas, temp/rescale 1 600 600 ave yes (t1, t4), and 1 0 0 ave yes on warm gas | A vs B: identical (kk and cpu; T=635.29137 resp. 16.558581 at step 3)
necessary: yes — A-kk NaN at t1 and t4.
complete: ave path is the only site; B==CPU ref at t1/t4. MPI addendum (A_mpi/B_mpi, 4 ranks, balance_grid rcb cell, t1): pos ave (cold, 0 0 yes): A-kk -nan, A-cpu/B-cpu/B-kk 0.0016006965 / 3.315e-23; pos cell (0 gas, 300 no): A-kk -nan, B-kk == cpu 0.0016006965; neg ave (300->600 yes): all four 639.65247; neg cell: all four 631.41428 -> guard after the Allreduce is correct on N ranks.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB5/F-G00-16 (log.{t4_,}pos_ave.*, log.{t4_,}neg_ave.*, log.pos_ave_warm.*), $S/ab/AB5/F-G00-16/mpi (np4: log.{pos,neg}_{ave,cell}.{A,B}.{kk,cpu})

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
complete: all 4 changed loops exercised (2d+3d x explicit+implicit), CPU and kk t1; B==reference in each.
MPI addendum (A_mpi/B_mpi, 4 ranks): (d) explicit/distributed 2d circle (global surfs explicit/distributed + balance_grid rcb cell), g = 2:50 | A cpu and kk: HANG (timeout 60 s, rc=124); A-cpu printed a garbage warning count "4627448617123184654 surfs are not assigned" before hanging (mismatched collectives) | B cpu and kk: run completes (both runs), stats identical to the g=1:50 reference. (e) implicit 2d (binary.101x101, 150^2, fnum 0.1), sg = y 0..75 | A cpu and kk: HANG (rc=124) | B cpu and kk: complete, warning "3986 surfs" (same count as 1 rank). Negative (g=1:50; implicit sg = whole domain): A vs B identical (cpu and kk, incl. warnings 28 / 8006). Also seen: g=26:50 on 4 ranks, A gives no warning and no hang (every rank returns early).
Sibling (NOT fixed, pre-existing, also in B): in explicit/distributed mode init() loops lines[0..nlocal) instead of the owned mylines, so the warning count depends on rank count: g=1:50 -> 25 (np1), 26 (np2), 28 (np4); and g=26:50 (25 surfs) -> 28 at np4. Warning text only, no tally impact.
verdict: NECESSARY+COMPLETE (deadlock and warning/clear fixed on 1 and 4 ranks, cpu+kk, explicit+implicit); sibling warning-count bug noted above
artifacts: $S/ab/AB5/F-G17-5 (2d explicit), F-G17-5/3d, F-G17-5/isurf, F-G17-5/isurf3d, F-G17-5/mpi (distributed: log.{pos,neg,np1,np2,np4g26}.*), F-G17-5/mpi_isurf (implicit np4)

### F-G18-2 — compute gas/reaction/grid(/kk): column map sized from react->nlist at definition; re-issued `react` (more reactions) or `react none` -> OOB / NULL deref (kk + CPU)
class: cpu-observable
positive control: 4^3 box, N2/N at 60000 K, 20000 particles, `react tce small.tce` (2 O2 reactions), compute g gas/reaction/grid all air <mode>, then `react tce air.tce` (45 reactions; N2+N2=#9, N2+N=#10 fire) or `react none`, compute reduce sum c_g[*], run 30
  every + grow: A-kk wrong counts (cols 60/68 vs nreact 132 — tallies of #9/#10 spill into neighbouring rows) then abort rc=134; A-cpu SIGSEGV | B (kk, cpu): ERROR "reactions changed since compute was defined" | REPRODUCED
  select 1 2 + grow: A-kk silently wrong (col1 = 131/138, i.e. reactions #9/#10 counted as selected reaction 1 via OOB reaction2col read); A-cpu SIGSEGV | B: same ERROR | REPRODUCED
  select 1 2 + react none: A-kk SIGSEGV in init (react->nlist on NULL); A-cpu runs (0 counts) | B: ERROR | REPRODUCED (kk)
negative control: (1) react unchanged (small.tce, every) and (2) compute defined after the full air.tce (every, cols 9/10 = 62/70 sum = nreact 132 kk; 51/72 = 123 cpu); (3) mode all with react grown or set to none | A vs B: identical (kk and cpu) in all three
necessary: yes — A gives wrong counts/abort/segfault on every grow variant (kk and CPU) and segfault on kk select + react none.
complete: both code paths covered (CPU init in compute_gas_reaction_grid.cpp, kk init in _kokkos.cpp), modes every and select, grow and none, cpu and kk — B errors cleanly in all. Sibling search: react->nlist is used outside react*/ only in compute_gas_reaction_grid(_kokkos) and finish.cpp (runtime use, no stale sizing) -> no missed site. Behaviour note: B also errors for formerly harmless inputs (every/select + `react none`, which A-cpu/A-kk-every ran with 0 counts; and a re-issued react with FEWER reactions). A re-issued react with the same count is still accepted. This is a conservative, documented error, not a wrong result. Thread count irrelevant (check is in init()).
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB5/F-G18-2 (in.base, run.sh, small.tce, log.{pos_every,pos_select,pos_none_sel,pos_none_every,neg_every,neg_full,neg_all,neg_all_none}.*)

### F-G17-3 — compute surf/kk & isurf/grid/kk: reallocate() mid-step (grid->notify_changed by fix move/surf, adapt, balance) recreated surf2tally -> step's tallies lost
class: cpu-observable (explicit, 1 rank) / mpi (isurf/grid via fix balance)
positive control: 2d circle, emit/face flow, compute surf all all n press, fix move/surf all 10 100000 trans 0 0 0 (notify_changed every 10 steps, zero displacement), stats 10 with compute reduce sum c_cs[1] c_cs[2] | A-kk: tallies 0 0 on every output step (t1 and t4) while nscoll = 39..66; A-cpu: n == nscoll | B-kk: n == nscoll exactly on every step (t1, t4), press ~4.3e-20 same magnitude as CPU ref | REPRODUCED
  variant consumer fix ave/surf (every 1, running ave over 10, defined after move/surf): A-kk running mean n = 9.76 at step 100 vs B-kk 12.86, CPU ref 13.28 (A drops each move-step tally, ~1/10 of samples) | REPRODUCED
negative control: same decks with move/surf Nevery=1000 (never fires in 100 steps) | A vs B: identical (kk and cpu; n==nscoll e.g. 148 at step 100 kk), incl. the ave/surf variant
necessary: yes for compute surf/kk (A-kk zero tallies on reallocate steps, t1 and t4).
complete: compute surf/kk verified with two consumers (compute reduce at output, fix ave/surf end_of_step), t1+t4, invariant n==nscoll holds in B.
MPI addendum (A_mpi/B_mpi, 4 ranks, kk t1, fix balance NEV rcb part, stats every 10 = balance steps):
  (f) explicit/distributed 2d circle, compute surf n only (in.exp_cs, NEV=10; surfs per rank change 14 -> 0..33, i.e. shrink AND grow branch) | A-kk: c_cs sum = 0 on every balance step (nscoll 5..147) | B-kk: c_cs == nscoll on all 10 lines | REPRODUCED
  (g) compute isurf/grid/kk, implicit circle, uniform fill (create_particles) so no rank drops to 0 surfs (in.imp_igu, NEV=10, surfs/rank 630..3404) | A-kk: c_ig sum = 0 on every balance step (nscoll 526..163) | B-kk: c_ig == nscoll on all 10 lines | REPRODUCED
  (with emit-only flow (in.imp_ig) A-kk aborts first on the F-G00-19 bounds error; B-kk: c_ig == nscoll, also on 2 ranks)
negative (MPI): NEV=1000 (balance never fires): in.exp (cs+rs), in.imp (ig+rg), in.imp_igu, kk and cpu | A vs B: identical, invariants hold in both. 1-rank in.exp NEV=10 (balance no-op): A==B.
complete: both reallocate sites (compute surf/kk explicit, isurf/grid/kk implicit), shrink + grow, 1 and 2/4 ranks, t1/t4 (t4 only 1 rank) -> B satisfies the invariant everywhere.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB5/F-G17-3 (in.base.orig, in.ave, run.sh, log.{pos,pos_t4,neg,ave_pos,ave_neg}.*), $S/ab/AB5/MPI (run.sh, in.exp*, in.imp*, log.{exp_pos,exp_neg,exp_neg1,cs_pos,ig_pos,ig2_pos,igu_pos,igu_neg,imp_neg}.*)

### F-G17-4 — compute react/surf/kk tallyinfo(): scans current nlocal+nghost instead of tally allocation (surf count changed by fix balance between tally and consumer)
class: mpi + bounds-check
positive control: 2d circle, global surfs explicit/distributed, 4 ranks, compute react/surf all 2 (+ compute surf), fix balance 10 1.00001 rcb part, compute reduce at stats every 10 (in.exp, NEV=10); A_mpi/B_mpi kk t1 | A-kk: abort rc=134, "Kokkos::View ERROR: out of bounds access label=("react/surf:surf2tally") with indices [52] but extents [14]" (grew) plus [-1] (rank left with 0 surfs, F-G00-19 pattern) | B-kk: runs; c_rs sum == nsreact on all 10 stats lines (5, 7, 15, 25, 32, 28, 28, 31, 36 ...) incl. ranks that shrank to 0 surfs | REPRODUCED
  react/surf only deck (in.exp_rs): A-kk abort (react/surf:surf2tally [-1], extents [14]); B-kk c_rs == nsreact on every line | REPRODUCED
negative control: same deck with NEV=1000 (no rebalance), kk and cpu, 4 ranks; and NEV=10 on 1 rank | A vs B: identical (c_rs == nsreact in both)
necessary: yes — A-kk host OOB read (bounds abort) on 4 ranks when the surf count changes between tally and tallyinfo().
complete: grow and shrink-to-zero branches both exercised; B satisfies the invariant on every balance step. CPU base uses a hash (no dependence on nsurf) -> CPU ref not affected (A-cpu/B-cpu identical, invariant holds). Only 4 ranks, t1 tested with MPI (thread count irrelevant: host-side scan).
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB5/MPI (in.exp, in.exp_rs, out./log.{exp_pos,rs_pos,exp_neg,exp_neg1}.*)

### F-G17-6 — compute react/isurf/grid/kk tallyinfo(): same scan-length bug (implicit surfs migrate with cells in fix balance)
class: mpi + bounds-check
positive control: implicit 2d circle (binary.101x101, threshold 180.5, 150^2 grid), compute react/isurf/grid all 2 feeding fix ablate and compute reduce, fix balance 10 rcb part, 4 ranks kk t1 | in.imp_rg (emit flow): A-kk abort, "react/isurf/grid:surf2tally indices [2961] but extents [1968]" and "[2906] extents [2009]"; in.imp_rgu (uniform fill, no rank at 0 surfs): A-kk abort, indices [2042]/[2061]/[2113] vs extents [1968]/[2052]/[2009]; 2 ranks: [5986] vs [3977] | B-kk: runs; c_rg sum == nsreact on all stats lines (e.g. 65,56,60,55,53 at steps 60-100 for rgu; 2-rank 4..28) | REPRODUCED
negative control: NEV=1000 (in.imp, in.imp_rgu), 4 ranks, kk and cpu | A vs B: identical; c_rg == nsreact in both
necessary: yes — A-kk OOB host read of surf2tally on every balance-before-consumer variant (2 and 4 ranks).
complete: grow branch (bounds abort in A) and shrink branch (rank losing surfs; B invariant holds on all ranks) both covered; consumer = compute reduce after balance on the same step (post_process_isurf_grid -> tallyinfo, the same path fix ablate uses). Ablation outcome not compared to CPU (kk RNG differs); invariant used instead.
Sibling (NOT fixed, pre-existing upstream, CPU + kk): compute react/isurf/grid uses a GRID group bitmask (grid->bitmask) but tests it against SURF masks (lines[i].mask/tris[i].mask in init() and surf_tally(), src/compute_react_isurf_grid.cpp:147,152,229,232 and KOKKOS/compute_react_isurf_grid_kokkos.h:62,66). With any grid group other than "all", tallies are silently zero (or follow an unrelated surf group with the same bit): 1 rank, same deck, group all -> c_rg = 13 (cpu) / 28 (kk) == nsreact; group inner (contains every surf cell) -> 0 / 0. Evidence: $S/ab/AB5/sibling_gridgroup/log.{all,inner}.{cpu,kk}.
verdict: NECESSARY+COMPLETE (fix); unfixed sibling grid-group/surf-mask bug reported above
artifacts: $S/ab/AB5/MPI (in.imp, in.imp_rg, in.imp_rgu, log/out.{imp_pos,rg_pos,rg2_pos,rgu_pos,rgu_neg,imp_neg}.*), $S/ab/AB5/sibling_gridgroup

### F-G00-19 — isurf/grid/kk and react/isurf/grid/kk (and react/surf/kk) tallyinfo(): `h_surf2tally[iend]==-1 && iend>0` reads index -1 on a rank with no surfs
class: mpi + bounds-check
positive control: implicit circle in a 400x150 domain, 4 ranks, balance_grid rcb cell -> 2 ranks own 0 surfs from the start, NO fix balance (in.imp0, NEV=1000) | A-kk: abort rc=134 "isurf/grid:surf2tally indices [-1] but extents [0]"; react/isurf/grid-only deck (in.imp0_rg): A-kk abort "react/isurf/grid:surf2tally indices [-1] but extents [1]" | B-kk: runs, c_ig == nscoll and c_rg == nsreact on all lines (e.g. step 100: 74/74, 28/28) | REPRODUCED
  react/surf/kk site (explicit distributed, rank emptied by fix balance, in.exp_rs): A-kk abort "react/surf:surf2tally [-1] extents [14]"; B-kk correct (see F-G17-4) | REPRODUCED
  isurf/grid/kk after balance to 0 surfs (in.imp_ig 4 and 2 ranks): A-kk "[-1] extents [0]"; B-kk c_ig == nscoll | REPRODUCED
negative control: in.imp 4 and 8 ranks without balance (every rank owns surfs, min 446/1968) | A vs B: identical (kk), cpu A==B; and in.imp0 cpu A==B (CPU uses hashes, unaffected)
necessary: yes — A-kk bounds abort at all three sites (isurf/grid, react/isurf/grid, react/surf) on ranks with 0 surfs.
complete: all three tallyinfo() loops exercised with a 0-surf rank, static (no balance) and after balance; compute surf/kk already had the safe operand order (grep: all four kk tallyinfo loops now `iend > 0 && ...`). No other `surf2tally[iend]` sites in src/KOKKOS.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB5/MPI (in.imp0, in.imp0_rg, log/out.{zero_neg,zero_rg,zero8,ig_pos,ig2_pos,rs_pos}.*)

### Notes (outside AB5 scope)
- A_mpi CPU run of in.exp on 4 ranks (exp_pos A cpu) finished all 100 steps with correct stats but exited rc=1 ("process rank 2 exited improperly", no error message); B_mpi rc=0. Not related to the AB5 fixes (different commit); not investigated further. $S/ab/AB5/MPI/out.exp_pos.A.cpu

## STATUS: COMPLETE
