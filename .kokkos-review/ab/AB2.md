# AB2 results (surf_collide/update/particle)
S = /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad ; work dir $S/ab/AB2 ; runner $S/ab/AB2/run.sh
(note: on this loaded box t 4 runs need OMP_WAIT_POLICY=passive OMP_PROC_BIND=false or they stall; run.sh exports them)

### F-G09-4b — UpdateKokkos::surf_collide_style_tag rejects explicit "<style>/kk" surf_collide names
class: cpu-observable
positive control: 2d circle, N beam, `surf_collide 1 diffuse/kk 300 1.0` (-sf kk); 3d piston deck with `piston/kk`, `vanish/kk`, `specular/kk` | A: ERROR "Unknown Kokkos surface collide method" (update_kokkos.cpp:2768) for both, t1 and t4 | B: runs; stats byte-identical to plain-name decks (diffuse step 400 np 40086 nscoll 191; piston step 100 np 15127 T 273.39) | REPRODUCED
negative control: plain-name decks | A vs B: identical (all stats rows)
necessary: yes - A aborts on every explicit /kk surf_collide name tried (diffuse, piston, vanish, specular).
complete: B OK for diffuse/kk, piston/kk, vanish/kk, specular/kk, t1 and t4 (diffuse); cll/td/impulsive/adiabatic/transparent explicit names not run individually (same sc_style_is helper, all 9 names go through it). Siblings: see 4a residual.
verdict: NECESSARY+COMPLETE (for the update_kokkos.cpp site)
artifacts: $S/ab/AB2/F-G09-4 (in.sc_kk, pist/in.piston_kk)

### F-G09-4a — surf_collide_*_kokkos surf_react dispatch rejects explicit "prob/kk", "global/kk", "adsorb/kk"
class: cpu-observable
positive control: same 2d deck + `surf_react r1 prob/kk` (N->O, p=0.5) for diffuse/cll/td/impulsive/adiabatic/specular; `global/kk 0.1 0.1` (diffuse); examples/surf_react_adsorb in.circle.gs with `adsorb/kk`; 3d piston with `prob/kk` on xlo (piston dispatch) | A: every case aborts "Unknown Kokkos surface reaction method" (t1 and t4) | B: all run; each stats table byte-identical to the plain-name deck (diffuse+prob step 400: np 40466 nscoll 183 nsreact 80; global: np 40003; adsorb: identical; piston+prob: identical, t4 also runs) | REPRODUCED
negative control: plain-name decks (6 models + global + adsorb) | A vs B: identical; CPU ref (diffuse+prob) np 40755 nscoll 194 nsreact 92 (same stats, different RNG)
necessary: yes - A aborts for prob/kk on all 7 models, global/kk, adsorb/kk.
complete: NO - sibling site missed: compute_surf_kokkos.cpp:189/201 still exact-matches "global"/"prob". Deck in.sr_kk (prob/kk + `compute surf ... n nflux`): B aborts "Unknown Kokkos surface reaction method" (compute_surf_kokkos.cpp:214). Any explicit surf_react */kk with compute surf/kk still fails in B. (collide_vss_kokkos.cpp:391 already accepts both names.)
verdict: INCOMPLETE (compute_surf_kokkos.cpp surf_react dispatch still exact-name; explicit prob/kk/global/kk + compute surf aborts in B)
artifacts: $S/ab/AB2/F-G09-4 (in.sr4a_*, in.glob_kk, ads/in.gs_kk, pist/in.pr_kk, in.sr_kk = failing B case)

### F-G08-1 / F-G09-3 — surf_collide_*_kokkos ignore `fix ambipolar/kk` / `vibmode/kk` given by explicit name (flag stays 0)
class: cpu-observable
positive control (ambipolar): 2d circle, N+ beam, fix ambipolar e N+, surf_react prob N+->N p=1, `fix ambi ambipolar/kk`, models diffuse/cll/td/impulsive/adiabatic/specular + 3d piston (react on xlo); invariant sum(p_ionambi) == count(N+) | A: violated, ionambi sum == np (diffuse step 500: np 119932, N+ 51595, sum ionambi 119932; piston: 13468 vs N+ 13306) | B: sum == N+ for all 7 models (51595 / 13306), tables byte-identical to plain-name runs; CPU ref 51572 == 51572 | REPRODUCED
positive control (vibmode): CO2 (4 vib modes) beam, collide vibrate discrete, `fix vm vibmode/kk`, cll and td; invariant evib/k - sum(theta_m*vibmode_m) == 0 | A: -2.9e8 (cll), -1.8e9 (td) | B: ~1e-4 (roundoff), identical to plain-name run | REPRODUCED
negative control: plain-name `fix ambipolar` / `fix vibmode` decks | A vs B: identical (all rows, all models)
necessary: yes - A breaks the invariant for every model with ambipolar/kk and for cll/td with vibmode/kk.
complete: B correct for ambipolar/kk on all 7 surf_collide kk models, vibmode/kk on cll, td (vibmode on others: same 2-line change, not run). Sibling collide_vss_kokkos.cpp:391 already accepts both names.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB2/F-G08-1

### F-G00-10 / F-G09-1 — cll/td/impulsive/adiabatic kk backup() did not refresh ambipolar/vibmode fix copies after a retry grew the particle arrays
class: cpu-observable (react/retry + particle growth)
positive control: -pk kokkos react/retry yes, 2d circle with dissociating surf reaction so the move kernel overflows and retries:
  (a) N2+ beam, fix ambipolar, N2+ -> N + N+ (p=1); (b) N beam, fix ambipolar, N -> N+ + e (p=1); (c) CO2 beam, fix vibmode, CO2 -> CO2 + CO2 (p=0.5), invariant evib vs vibmode
  | A: cll, td, impulsive crash in all three decks ("double free or corruption" exit 134 / SIGSEGV 139); diffuse (already fixed upstream) OK; adiabatic (b): sum ionambi 42671 != N+ 42675
  | B: all 5 models run; invariants hold: (a) sum ionambi == N2+ + N+ (cll 117327 = 55570+61757); (b) e count 0, ionambi == N+; (c) mismatch ~1e-4 roundoff; stats agree with CPU ref (cll (a): B N+ 61757 vs CPU 61674) | REPRODUCED
negative control: deck (a) cll with react/extra 4.0 (no retry) | A vs B: identical stats tables
necessary: yes for cll/td/impulsive (crashes); adiabatic only via the ambipolar surf_react path, and A's adiabatic miss (4 ionambi lost in (b)) is confounded with R-A-4 in the same deck.
complete: B correct for all 5 models t1; t4 checked for cll (a), td (c), impulsive (b): all correct.
verdict: NECESSARY+COMPLETE (adiabatic necessity not isolated from R-A-4)
artifacts: $S/ab/AB2/F-G00-10 (in.{diffuse,cll,td,impulsive,adiabatic}, in.ion.*, in.vib2.*)

### F-G08-3 — vanish/transparent collide() never set `reaction` (callers read it)
class: cpu-observable
positive control: 2d, N beam, circle (diffuse + prob N->O p=0.5) + closed square group sq with surf_collide vanish or transparent; on sq: compute surf etot echem + compute surf/reaction/tally dumped every step
  | A CPU: SIGSEGV (exit 139) for vanish and transparent (surf->sr[isr=-1] with stale reaction)
  | A KK: no crash, but spurious reaction events on sq: 3840 rows (vanish), 3549 rows (transparent) in the surf/reaction/tally dump; etot/echem equal to B by luck
  | B CPU and KK: 0 tally rows, runs complete; KK echem 0, etot 9.04e-18 vs CPU 9.08e-18 | REPRODUCED
negative control: same decks without surf_react | A vs B stats: identical (CPU and KK, both styles). Note A KK still dumped 23 spurious rows there (uninitialized reaction), B 0.
necessary: yes (CPU crash; KK spurious tally events).
complete: B correct for vanish and transparent, CPU and KK.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB2/F-G08-3

### F-G08-2 — piston (CPU + kk) dereferenced ip after surface chemistry deleted it; orphaned reaction product on outside-box return
class: cpu-observable
positive control: examples/surf_collide/in.piston (100 steps) + `surf_react r1 global 0.3 0.0` (pdelete) on the piston face xlo | A: CPU SIGSEGV (exit 139), KK t1 SIGSEGV (exit 139) | B: CPU, KK t1, KK t4 all complete (step 100: np 15011 / 15046 / 15003, T 272.4 / 270.4 / 269.0) | REPRODUCED
second part (orphan product, deck in.diss: N2 -> N + N p=0.1 on piston face): A and B both run; A CPU np 15062, B CPU 14972, B KK 15009, B KK t4 15032 - no direct observable separates the orphan from RNG noise | NOT REPRODUCED
negative control: in.piston without surf_react | A vs B: identical stats tables (CPU and KK)
necessary: yes for the null-deref part (crash on CPU and KK); the orphan-discard part not shown necessary (no observable found).
complete: B correct for delete reaction on CPU, KK t1, KK t4; dissociation path runs on all three (KK discard relies on R-A-4, see below).
verdict: NECESSARY+COMPLETE (null-deref part); orphan-discard sub-part NOT-SHOWN-NECESSARY (no distinguishing observable)
artifacts: $S/ab/AB2/F-G08-2

### R-A-4 — KK move: reaction product flagged PDISCARD inside a surface collision was never put on the migrate list (advected with garbage, never deleted)
class: cpu-observable
positive control: 2d circle, N beam, fix ambipolar e N+, surf_react N -> N+ + e (p=1): fix ambipolar sets j=-1 so the electron product is flagged PDISCARD; react/extra 4.0 (no retry, isolates from F-G00-10); invariant count(e) == 0 (CPU ref: 0) | A: stray electrons survive at step 300: diffuse 2351, cll 2592, td 2325, impulsive 77994, adiabatic 31936, specular 39931 (np inflated accordingly, e.g. specular 134374 vs CPU 94668) | B: e = 0 for all 6 models at t1 and t4 (5 models), np close to CPU (specular 94443 vs 94668; diffuse 226228 vs CPU 227080) | REPRODUCED
piston outside-box discard path (new PDISCARD source from F-G08-2): B KK t1/t4 with piston + N2 -> N+N on xlo run clean (F-G08-2 in.diss); no A comparison possible (path did not exist in A)
negative control: N2+ -> N + N+ deck (no PDISCARD product), cll, react/extra 4.0 | A vs B: identical stats tables
necessary: yes (all 6 surf_collide kk models leak discarded electrons in A).
complete: B correct for diffuse/cll/td/impulsive/adiabatic/specular t1, 5 of them t4; piston path runs clean.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB2/R-A-4 (also $S/ab/AB2/F-G00-10/in.ion.* with retry)
