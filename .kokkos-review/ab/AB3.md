# AB3 (surf_react) A/B results

S = /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad
Tool: $S/ab/AB3/tools/kmem.so (source kmem.cpp) = Kokkos tools lib (KOKKOS_TOOLS_LIBS) counting
allocate/deallocate per view label + begin_deep_copy per destination label ("sra:*"); report printed at Kokkos
finalize, i.e. after SPARTA teardown, so "alloc N dealloc 0" = orphaned views ("KLEAK" lines).
A = build_base (e071055f), B = spa_new_final (contains 780d12b2). Runs: -k on t 1|4 -sf kk, OMP_WAIT_POLICY=PASSIVE
(machine is shared/oversubscribed). MPI builds (bmpi_*) were still compiling (25%) when this was written.

Sweep ($S/ab/AB3/sweep, sweep_t1.txt): 2d circle, O+N inflow, 300 steps; surf_collide/kk in {diffuse, specular,
cll, td, impulsive, adiabatic} x surf_react in {global, prob(on.surf), adsorb gs SURF nsync 10} x backup trigger in
{-pk kokkos react/retry yes, compute surf/collision/tally + dump tally every step, compute surf/reaction/tally + dump}
= 54 A/B pairs, plus piston (bound yhi, face adsorb) x 3 react styles, plus cll x 3 x {retry,coll} at t 4.

### F-G10-1 — surf_react global/kk backup() allocates backup views on the blitted image -> leaked each backup
class: cpu-observable (allocation count via Kokkos tools hook)
positive control: 2d circle, O+N, diffuse + surf_react global 0.2 0.1, react/retry yes, 1000/3000 steps | A: nsingle_backup alloc 1978/5978 dealloc 0 (tally_single_backup same; grows linearly) | B: alloc 1 dealloc 1 | REPRODUCED
negative control: same input with react/retry no, and retry yes 1000/3000 steps | A vs B: identical stats; B retry vs no-retry identical
necessary: A leaks on every one of the 18 global sweep cases (6 surf_collide models x {retry, coll-tally, react-tally}): 578 nsingle_backup + 578 tally_single_backup orphaned per 300 steps; also piston (t1) not measurable in A (A segfaults with piston+global even with retry no -- separate piston bug fixed elsewhere, 7af97cde/08c0ccac)
complete: B: 0 leaked views in all 18 sweep cases + piston + cll t4 {retry,coll}; stats and tally dump files md5-identical A vs B (t1) in every case; t4 np within noise (35780 vs 35798). Sibling scan: no other lazy "_backup" allocation inside a backup() run on blitted images (surf_collide_*_kokkos random_backup lazies are EXACT-only and pre-allocated in their ctors; collide_vss/react_bird/update backups are on live objects and showed alloc==dealloc in all runs)
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB3/F-G10-1, $S/ab/AB3/sweep

### F-G10-2 — surf_react prob/kk backup() same leak as F-G10-1
class: cpu-observable (allocation count via Kokkos tools hook)
positive control: same input with surf_react prob on.surf (O->N, N->O, E S 0.3), react/retry yes, 1000/3000 steps | A: nsingle_backup alloc 1000/3000 dealloc 0, tally_single_backup same | B: alloc 1 dealloc 1 | REPRODUCED
negative control: retry no; and retry yes | A vs B: identical stats
necessary: A leaks 300+300 views per 300 steps in all 18 prob sweep cases and piston+prob (t1) and cll t4
complete: B 0 leaked views in all of them; stats/tally dumps md5-identical A vs B at t1; piston+prob stats identical
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB3/F-G10-1 (out.prob.*), $S/ab/AB3/sweep

### F-G10-4 — adsorb/kk init() re-zeroes device species_delta/mark every run -> pending GS deltas lost across runs
class: cpu-observable
positive control: closed box (boundary r r p), 20000 O, specular circle + adsorb gs sample-GS_1.surf (only O(g)->O(s), prob 1) nsync 10 SURF mode; invariant np + sum(s_nstick_total) at sync steps; `run 105` + `run 100` | CPU: 20000 at 110..200 | A (t1,t4): 19714 from step 110 on (286 adsorptions of steps 101-105 lost; nstick 7618 vs 7904 at 110) | B (t1,t4): 20000, full stats identical to CPU | REPRODUCED
negative control: `run 100` + `run 105` (first run ends on a sync step) | A vs B identical (t1,t4), both 20000
necessary: shown above (A violates conservation by exactly the pending deltas)
complete: SURF: B==CPU bit-for-bit, t1 and t4. FACE (in.face_split: closed box, all 4 faces adsorb, site density 0.1 => capacity 4000 => saturated np plateau = 16000): `run 5` + `run 300` | CPU 16001 | A t1/t4: 15374/15375 (the 625 adsorptions of steps 1-5 forgotten -> 625 extra sites refilled) | B t1/t4: 16000/15999; aligned (`run 10`+`run 295`) A==B 16000. Size-change path of alloc_state_kokkos(0) only reachable via grid change (F-G10-3)
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB3/F-G10-4

### F-G10-5 — adsorb/kk backup() allocates nsingle/tally_single/species_delta/mark backups on the blitted image -> leaked each backup (+ RanKnuth under EXACT)
class: cpu-observable (allocation count via Kokkos tools hook)
positive control: closed box, 20000 O, adsorb gs nsync 10, react/retry yes, 500 steps; SURF (circle) and FACE (4 box faces) | A: SURF: 4 labels each alloc 500 dealloc 0 (308 KB; species_delta_backup scales with nlocal+nghost); FACE: 3 labels alloc 500 dealloc 0 | B: alloc 1 dealloc 1 | REPRODUCED
negative control: react/retry no, and retry yes | A vs B identical stats (np+nstick = 20000 at step 500 in all 4 SURF runs)
necessary: A leaks in all 18 adsorb sweep cases (6 models x {retry, coll-tally, react-tally}), cll t4; piston+adsorb A segfaults (unrelated piston bug)
complete: B 0 leaked views in all 18 sweep cases, piston+adsorb(FACE), cll t4, SURF/FACE; stats/tally md5-identical A vs B at t1. EXACT RanKnuth (host new) part NOT tested: no SPARTA_KOKKOS_EXACT binary
verdict: NECESSARY, COMPLETENESS-PARTIAL (SPARTA_KOKKOS_EXACT RanKnuth backups untested; no EXACT build)
artifacts: $S/ab/AB3/F-G10-5, $S/ab/AB3/sweep

### F-G10-6 — adsorb/kk state_synced_to_device set only on blitted image -> H2D state copy on every pre_react (perf only)
class: cpu-observable (deep_copy count via Kokkos tools begin_deep_copy hook)
positive control: F-G10-5 runs (500 steps, nsync 10) | A: deep_copy into sra:total_state/species_state/area/weight 500x each (every step), SURF and FACE, retry yes/no | B: 50x each (once per nsync window) | REPRODUCED
negative control: same runs | A vs B identical stats
necessary: A copies every step in all 18 adsorb sweep cases (300 copies/300 steps) and cll t4 -- perf only, so "necessary" = removes redundant work, not a wrong result
complete: B 30 copies/300 steps in all 18 sweep cases (6 surf_collide models x 3 triggers), piston FACE, cll t4; results unchanged (stats/tally identical A vs B; B==CPU in F-G10-4 split run, which also exercises the init() flag reset). Fix sits in pre_react itself so all call sites (surf_collide_*, compute_surf_kokkos) are covered; flag reset on grid change only checkable with MPI (F-G10-3)
verdict: NECESSARY+COMPLETE (perf-only)
artifacts: $S/ab/AB3/F-G10-5, $S/ab/AB3/sweep (KDC lines in out.*)

### F-G00-18 — adsorb/kk scatter_cmodel SPECULAR ignores the noslip flag of a "specular noslip" product cmodel
class: cpu-observable
positive control: in.ns: 2d box, boundary "rs p p", gas vstream (100,50,0) temp 0.01, FACE adsorb gs on xhi (bound collide specular), 1000 steps; nneg = (np*50 - sum vy)/100 = # particles whose vy was flipped (noslip negates v, plain specular keeps vy=+50) | file ip.noslip (DA O2(g)->O(s)+O(g), cmodel_ip "specular noslip"): CPU nneg 9920.6 (= #O 9919 + thermal) | A 0 (-0.86) | B 9920.6555 (bit-identical to CPU) | REPRODUCED
negative control: same files with plain "specular" (ip.plain, ip2.plain, jpda.plain, ci.plain, lh.plain) | CPU == A == B (identical stats lines)
necessary: A gives nneg ~0 in every noslip variant below (all CPU > 0)
complete: B == CPU in every scatter_cmodel call path:
  - DA ip product, cmip flags (h:453): FACE t1/t4 and SURF mode (single line wall, in.ns_surf): CPU 10000.707, A -0.707, B 10000.707
  - DA stoich-2 jp product with cmip flags (h:461; ip2: CO2(g)->C(b)+2O(g)): CPU 9931.31 | A -1.77 | B 9931.31 (CPU/B both reverse only one of the two O, since jp is cloned from the already-scattered ip and negated again -- B reproduces CPU semantics)
  - DA distinct 2nd gas product, cmjp flags (h:470; jpda: CO2(g)->O(s)+O(g)+CO(g), ip plain / jp noslip): CPU 9931.31 | A -1.77 | B 9931.31, t1 and t4
  - CI ip plain + jp noslip with cmjp flags (h:510; ci: AA O->O(s), CI CO2(g)+O(s)->CO(g)+O2(g)): CPU 9.70 (10 O2) | A -0.29 | B 13.80 (14 O2 created in the Kokkos run) -> B = #jp products, as on CPU
  - LH1 ip noslip (h:484; lh: C(g)+O(s)->CO(g)): CPU 14.86 (14 CO) | A -0.28 | B 13.82 (14 CO)
  (CI/LH counts are small because O(s) coverage limits them; the per-event signature is exact.)
  Sibling scan: only other reflect3 uses are diffuse partial-specular (no noslip on CPU either) and surf_collide_specular_kokkos (already honors noslip). Not run: CI stoich-2 jp with cmip flags (h:502), ER (same h:484 line as LH1).
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB3/F-G00-18 (in.ns, in.ns_surf, *.surf, log.{CPU,A,B}.*)
