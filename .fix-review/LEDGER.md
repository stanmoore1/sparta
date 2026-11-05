# Ledger (milestones; newest last). Earlier phases: see report/sparta-fix-verification.html and git log.
- Rounds 1-6 (2026-10-03..10-07): 303+16+53+5 fixes A/B-verified (report v3). P-9 backed out (6080fb3c).
- 10-29 Round 7 start: merged origin/master 5702a655 (fee92ef5; 11 conflicts; fix_emit_surf reset_fractions made MPI-safe),
  fem393 report (da3f121a). Merge reviewers: 32 files OK, 1 problem (kk TCE warning) -> 3338e708. fix_ablate realloc after
  cleanup f001f59f. F=f001f59f for the round-7 sweeps.
- v_mst DONE: 0 regressions, 0 lost; F better than M in 4 (emit/surf fraction hang, host compute surf tx/ty/tz segfault,
  custom surf create before isurf segfault, kk TCE warning); 1 pre-existing bug in M and F (implicit-surf custom spread
  heap overflow, Surf::spread_own2local_rendezvous) -> follow-up task queued; 3 doc gaps fixed (18dbbe81, 147fef69);
  builds: CMake mpi, make serial, c++11/c++17 OK; no new warnings except false positives.
- v_ab2 DONE (chunks 15-29, ~135 ids): 0 lost, 0 regressions. Note: react none after compute gas/reaction/grid rejected
  (KOKKOS-review F-G18-2) -> relaxed c8111607 (verified: react none == M; redefinition with growth still clean error, M ASan).
- v_ex DONE (147 examples serial / np4 / asan): 0 regressions; 8 differ vs M, each bisected to one fix (08-1, 19-8, 12-7,
  19-10, 06-9, 29-11; np4 also 15-2 SurfColl checks, P-1 memory); shipped mpi_4 gold logs differ on 6 -> CI impact (r3).
  ASan: nothing new vs M; dump fwrite(NULL,0) fixed a018e57c (verified).
- v_ab1 PARTIAL: 61/143 OK (21 harness + 10-4 + 39 deck ids), 0 lost; queues for the rest finished but comparison lost.
- 11-05 CONTAINER REPLACED: scratchpad (all decks, binaries, checkpoints) lost; agents gone. Branch intact.
  Merged fem393 again (d06625d6): FFT 2d remap single precision (877267ee), gas/reaction/grid check: conflict resolved
  as "reject only growth" (both relaxations combined). Durable state moved into .fix-review/ (this dir, committed).
- 11-05 new container set up: OpenMPI installed, d06625d6 builds; .fix-review committed (5103c45b) + checkpoint loop; agents r1 r2 r3 launched; ensure_builds running.
- 11-05 KOKKOS handoff (98a5f00d merged): R4-P1 migrate-list overflow "pre-existing, not fixed" was measured on the
  KOKKOS branch, which lacks this branch's 25-2 fixes. On F: in.ml np1 (40000 entries vs 32768) serial rc=0 and ASan
  clean (only OpenMPI libevent leaks, now suppressed); M serial SEGV (rc 139). in.ml2 np2 gridcut -1 and 0.0: F mpi and
  asan rc=0. Pending: M np2, kfin_kk (bounds/OOB check) vs kmst_kk. Decks: decks/orch/mlist.
- R4-P1 migrate-list overflow: FIXED ON THIS BRANCH (25-2 host + kk). in.ml2 np2 gridcut -1 and 0.0: M mpi "malloc():
  corrupted top size" then hang (timeout 300 s); F mpi + asan rc=0 (60000 particles); kfin_kk np2 with react/retry and
  react/extra 3 rc=0 (60000). kfin_kk in.ml np1: default stops with SPARTA's normal "Ran out of space for Kokkos
  reactions" message (expected), react/retry and react/extra 3 rc=0. The handoff's "not fixed" referred to its own branch.
- r1 DONE: 143/143 OK (64 carried, 56 diff, 23 runtime), 0 lost/regr; gas/reaction/grid + FFT single verified (M single silently wrong). Pre-existing kk react-none segfault (M too) -> fixed b63cf190; runtime test pending (kfin2 build).
- KOKKOS react none crash (r1 finding, also in M): first fix b63cf190 (reset react_defined) insufficient; root cause =
  stale KKCopy byte copies of the deleted react object in the collision functor. Fixed 99a27009 (KKCopy::uncopy(), drop
  react copies in CollideVSSKokkos::init). Verified on kfin2_kk: nocompute/every/select .none rc=0 2 runs (kmst rc=139);
  every.fewer/same stats == kmst; examples chem and collide_3D np2 kk stats identical to kfin. Probed sibling KKCopy
  users: uncompute of surf/boundary computes and unfix vibmode between runs do not crash (decks/orch/kkcopy).
- r2 DONE: 58 phase-3/merge-2 rows OK (24 runtime, 28 diff, 3 diff+neg, 2 rejections hold, 2 sweeps redone), 0 lost/regr.
  KOKKOS sweep 141 inputs x np1/np2: 242 SAME, 24 same-error, 16 DIFF all explained (08-1, 19-8, 12-7, 06-9, kk 19-10,
  ambi both fail). Notes: P-3 F leaks 56-62 B on fatal dump ctor error (M SEGV) - benign; 23-7 neg deck SEGV on kmst only.
  run.sh ASan tag now matches SEGV. F.txt -> current HEAD (adds only the KOKKOS react-none fix after d06625d6).
- r3 DONE (CI replica, VTK skipped): M 246/246 both jobs; F 234/246 both jobs, same 12 failures = 6 examples x np1/np4,
  each confirmed by reverting only the fix (12-7 x3, 06-9, 19-10, 19-8); no test fails on both. Reblessed the 12 gold
  logs (commit below); rerun of the 12 tests in both F jobs: 12/12 pass. Production blockers: none remaining.
- ROUND 7 COMPLETE: verdict production-ready; report v4 published. Checkpoint loop may be stopped (touch /tmp/fix-review-checkpoint.stop).
- 11-05 implicit-surf custom spread FIXED (be90280d + follow-up): spread_own2local_implicit (owned = first nlocal; ghosts
  requested by ghost cell + csurfs position; EMPTY ghosts skipped). Verified (decks/orch/isurf_spread): diffuse s_ts=500
  == diffuse 500.0 bit-identical at np1/2/3/4 on fix8_asan (ASan clean) and on kfin2_kk np1/2/4; old F ASan
  heap-buffer-overflow at np1/np4. Found: custom surf data is not migrated with implicit cells on rebalance -> fix balance /
  balance_grid now error for implicit surfs with custom attrs (documented). Neg controls F vs new, np2: ablation 2d +
  multi.inner.3d, 26 custom examples, explicit2implicit circle.2d/sphere.3d: identical stats; no shipped example affected.
