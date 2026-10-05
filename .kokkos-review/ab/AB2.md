# AB2 results (surf_collide/update/particle)
S = /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad ; work dir $S/ab/AB2

### F-G09-4b — UpdateKokkos::surf_collide_style_tag rejects explicit "diffuse/kk"
class: cpu-observable
positive control: 2d circle, N->beam, `surf_collide 1 diffuse/kk 300 1.0` with -k on t 1 -sf kk | A: ERROR "Unknown Kokkos surface collide method" (update_kokkos.cpp:2768) | B: runs, stats table byte-identical to the `diffuse` (-sf kk) run (step 400: np 40086 nscoll 191) | REPRODUCED
negative control: same deck with plain `diffuse` | A vs B: identical (all stats rows)
verdict: VERIFIED
artifacts: $S/ab/AB2/F-G09-4 (in.sc_kk, in.sc_plain)

### F-G09-4a — surf_collide_*_kokkos surf_react dispatch rejects explicit "prob/kk"
class: cpu-observable
positive control: same deck + `surf_react r1 prob/kk` (N->O exchange, p=0.5), surf_collide = diffuse, cll, td, impulsive, adiabatic, specular (plain names, -sf kk), no compute surf | A: all 6 abort "Unknown Kokkos surface reaction method" | B: all 6 run; each stats table byte-identical to the same deck with plain `prob` (e.g. diffuse step 400: np 40466 nscoll 183 nsreact 80) | REPRODUCED
negative control: plain `prob` decks, 6 models | A vs B: identical (all rows); CPU reference (diffuse+prob) step 400 np 40755 nscoll 194 nsreact 92 (same statistics, different RNG stream)
note (residual, not part of 4a/4b): with `compute surf` active, B still aborts "Unknown Kokkos surface reaction method" at compute_surf_kokkos.cpp:214 for `prob/kk` -- the compute_surf_kokkos.cpp:180/192 part of the F-G09-4 fix proposal was never applied (in.sr_kk). Explicit surf_react */kk + compute surf/kk remains broken.
verdict: VERIFIED (surf_collide + update parts); compute_surf_kokkos part NOT FIXED (residual)
artifacts: $S/ab/AB2/F-G09-4 (in.sr4a_*, in.sr_kk)

### F-G08-1 / F-G09-3 — surf_collide_*_kokkos ignore `fix ambipolar/kk` / `vibmode/kk` given by explicit name (ambi_flag stays 0)
class: cpu-observable
positive control: 2d circle, N+ beam (fix ambipolar e N+), surf_react prob N+ -> N (p=1), `fix ambi ambipolar/kk` explicit, -sf kk, models diffuse/cll/td/impulsive/adiabatic/specular; observable c_ia = reduce sum p_ionambi vs count N+ | A: ionambi sum == np (e.g. diffuse step 500: np 119932, N+ 51595, sum ionambi 119932 -> every neutral N still flagged as ion) | B: sum ionambi 51595 == count N+, all 6 models; stats tables byte-identical to the `fix ambipolar` (plain name) runs. CPU reference: sum ionambi 51572 == N+ 51572 | REPRODUCED
negative control: same decks with plain `fix ambi ambipolar` | A vs B: identical (all rows, all 6 models)
verdict: VERIFIED (vibmode/kk branch is the same one-line change; not separately run)
artifacts: $S/ab/AB2/F-G08-1
