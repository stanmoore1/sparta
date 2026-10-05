# AB3 (surf_react) A/B results

Tool: $S/ab/AB3/tools/kmem.so = Kokkos tools lib (KOKKOS_TOOLS_LIBS) counting allocate/deallocate per view label;
prints per-label alloc/dealloc counts and live bytes at Kokkos finalize (after SPARTA teardown), so "alloc N dealloc 0" = orphaned views.
A = build_base (e071055f), B = spa_new_final (780d12b2 included). Runs: -k on t 1 -sf kk.

### F-G10-1 — surf_react global/kk backup() allocates backup views on the blitted image -> leaked each backup
class: cpu-observable (allocation count via Kokkos tools hook)
positive control: 2d circle, O+N, diffuse + surf_react global 0.2 0.1, -pk kokkos react/retry yes, 1000/3000 steps | A: nsingle_backup alloc 1978/5978, dealloc 0 (tally_single_backup same; leak grows linearly with steps) | B: alloc 1 dealloc 1 (both labels) | REPRODUCED
negative control: same input, stats (np, nscoll, nscheck every 100 steps) with retry yes (1000,3000 steps) and with react/retry no | A vs B: identical (all rows); B retry vs no-retry also identical
verdict: VERIFIED
artifacts: $S/ab/AB3/F-G10-1 (in.leak, out.global.*, log.global.*)

### F-G10-2 — surf_react prob/kk backup() same leak as F-G10-1
class: cpu-observable (allocation count via Kokkos tools hook)
positive control: same input with surf_react prob on.surf (O->N, N->O, E S 0.3), react/retry yes, 1000/3000 steps | A: nsingle_backup alloc 1000/3000 dealloc 0, tally_single_backup same | B: alloc 1 dealloc 1 | REPRODUCED
negative control: same input, retry yes 1000/3000 steps and retry no 1000 steps | A vs B: identical stats
verdict: VERIFIED
artifacts: $S/ab/AB3/F-G10-1 (out.prob.*, log.prob.*)

### F-G10-4 — adsorb/kk init() re-zeroes device species_delta/mark every run -> pending GS deltas lost across runs
class: cpu-observable
positive control: closed box (boundary r r p), 20000 O particles, specular circle + adsorb gs sample-GS_1.surf (only O(g)->O(s), prob 1) nsync 10 surf mode; invariant np + sum(s_nstick_total) at sync steps; `run 105` + `run 100` | CPU: 20000 at steps 110..200 | A (t1,t4): 19714 from step 110 on (286 adsorptions of steps 101-105 permanently lost; nstick 7618 vs 7904 at 110, 11461 vs 11747 at 200) | B (t1,t4): 20000, full stats identical to CPU run | REPRODUCED
negative control: same input, `run 100` + `run 105` (first run ends on a sync step) | A vs B: identical (t1 and t4), both conserve 20000
verdict: VERIFIED
artifacts: $S/ab/AB3/F-G10-4 (in.persist, log.{A,B}.{split,aligned}.t{1,4}, log.cpu.split)
