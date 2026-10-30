# V3 round: orchestrator state and resume procedure

User request (2026-10-29): "Do another exhaustive round of verification. Are these changes ready for
production, with no regressions?"  Plan and rules: V3_PROTOCOL.md.  Per-agent results: v3/<agent>.md.

## Resume after a session limit / container restart / new session
1. `cd /home/user/sparta && git pull origin ccr-3dcb1e77-fem393` (only if a new container), then
   `nohup .kokkos-review/checkpoint.sh >/dev/null 2>&1 &` (single-instance; commits+pushes .kokkos-review every
   5 min, writes wip-src.patch and v3/scratch-files.tgz = agents' scripts/decks).
2. If the scratchpad ($S below) is gone (new container): `mkdir -p $S/final && tar xzf
   .kokkos-review/v3/scratch-files.tgz -C $S/final` to restore the agents' scripts and decks, then
   `nohup .kokkos-review/ensure_builds.sh > $S/final/ensure.log 2>&1 &` (idempotent; recreates worktrees
   base/ sync_A/ final/src_sync and every binary; `grep rc= $S/final/ensure.log`).  If the session's scratchpad
   path changed, run with `S=<new scratchpad>` and update S/F in V3_PROTOCOL.md first.
   If $S survived (container restart only): just re-run ensure_builds.sh; it only rebuilds what is stale.
3. For each agent below not marked DONE: if its agent is still addressable (same session) SendMessage it
   "Resume after restart: background processes were killed; read your checkpoint and continue."; otherwise
   launch a new general-purpose agent with the prompt under "Prompts" (it resumes from its checkpoint file).
4. Then continue with "Orchestrator pending" below.

S=/tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad, F=$S/final

## Agent status (update on every report)
| agent | scope | status | result |
|---|---|---|---|
| V3-R1 | grid_collate, isurf/react isurf grid, surf, surf_comm/custom, compute surf/property surf | DONE | no regression; PERF note only |
| V3-R2 | FFT/remap/pack/kissfft, STUBS, compute fft/grid kk | DONE | R2-1 INCOMPLETE -> fixed 877267ee (verified); R2-2 pre-existing unreachable |
| V3-R3 | collide_vss/update/react kk, collide.cpp, gas/reaction/grid, kokkos.cpp, kokkos_type.h, comm kk | DONE | V3R3-1 REGRESSION (minor) -> fixed c99e6089 (verified, v3/V3-R3-fix.md); V3R3-2 SYCL needs Kokkos>=4.5 (note); timing within noise |
| V3-R4 | surf_collide_*, surf_react_* | DONE | no regression; R4-P1 migrate-list overflow PRE-EXISTING (A and B), not fixed |
| V3-R5 | grid/particle/surf/modify kk, custom kk, fixes, Install.sh/CMake | DONE | no regression; notes N1-N3 |
| V3-R6 | grid computes kk + CPU count/dt/lambda | DONE | no regression; R6-1 pre-existing (dt/grid step 0) |
| V3-T1 | all examples A vs B: kk t1/t4, CPU, MPI, cpu/stub builds | DONE | 141 inputs x 9 configs: 0 regressions; diffs = F-G21V-1 (ambi), F-G00-7 (relax_const t4), F-G00-19 (ablation MPI); t4 noise-equal (8v8 repeats) |
| V3-T2 | split-memory sweep B_sync/B_poison vs A_poison | DONE | 0 NEW reports; poison 0 ASan np1/np2; audit 0 undeclared; Bn==Bx==O bitwise on 138 decks; stale/watch labels all pre-existing; several FIXED-BY-BRANCH; tool bug T2-1 (audit, not SPARTA) |
| V3-T3 | ASAN/UBSAN (B_asan) + valgrind sweep | DONE | 0 NEW; ASAN CPU+kk 133/141 clean, 8 decks with 4 PRE-EXISTING signatures (U1 marching_cubes qsort NULL, U2 surf_comm memset NULL, L2 Python leak, L3 watertight error-exit leaks); valgrind 32 decks: only OpenMPI noise + V1 pre-existing (Grid::update_halo_index uninit) |
| V3-T4 | merged tree with ccr-014f4f5c-hhi3vb, restart compat, formats, warnings, perf | DONE | merged clean, 141 run, 9 stat diffs bisected to other-branch fixes; restart/format PASS; no new warnings; perf geo-mean 0.995, max instr +2.4% emit.surf.subsonic |

## Orchestrator pending
- DONE (round complete): report regenerated with round-3 section (305fdfcb), artifact republished (version 3). Verdict given to user.
- DONE: B_opt and B_mpi rebuilt with 877267ee + c99e6089 (opt2, mpi2 rc=0).
- DONE: V3-T3 launched (B_asan built).
- After T1/T2/T4 finish: rerun examples/fft and examples/chem (+ tally_computes) on the newest B_opt/B_mpi,
  since 877267ee/c99e6089 landed after the sweeps' B binaries (expected identical).
- Final: update make_report.py with a V3 section (findings, fixes, sweeps), regenerate report.html, commit, push,
  republish artifact https://claude.ai/artifact/DFzEfKs3xnM37M1ge3gpck (same file path
  $S/report/kokkos-fix-verification.html), answer the user: production-ready yes/no with caveats.

## Prompts (for relaunching an agent in a new session)
Common preamble: "You are <agent> in a production-readiness verification of the SPARTA (DSMC) branch
ccr-3dcb1e77-fem393. Read /home/user/sparta/.kokkos-review/V3_PROTOCOL.md fully and follow it exactly;
checkpoint /home/user/sparta/.kokkos-review/v3/<agent>.md (resume from it if it exists). Report back a
concise findings table (id, class, file:line, how verified, summary), only verified findings, plus REFUTED
suspicions in one line each."
- V3-T1: "+ Runtime item T1. Every examples/*/in.* (skip python/paraview/vtk), A vs B: kk t1, t4 (+A-vs-A
  spread), CPU styles of A_opt/B_opt, A_mpi/B_mpi kk np2/np4, cpu_A/cpu_B np1/2/4, stub_A/stub_B np1. Same caps for A
  and B. Explain every difference (FIXES.md id/commit) or call it REGRESSION. At most 2-3 simulations at once."
- V3-T2: "+ Runtime item T2. Read $F/src_sync/.github/dev-docs/kokkos-sync-debugging.md. Every example -k on -sf kk
  on B_sync and B_poison, np1/np2, STALE_STRICT/WATCH/AUDIT/POISON; classify B reports vs A_poison as NEW /
  PRE-EXISTING / FIXED-BY-BRANCH; root-cause NEW ones; cross-check B_sync thermo vs plain OpenMP B."
- V3-T3: "+ Runtime item T3. Every example on B_asan ($F/basan/src/spa_*), CPU styles and -k on t 2 -sf kk; set
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=0 and UBSAN_OPTIONS=print_stacktrace=1; for every report build or reuse
  an A_asan (same flags on $S/base, in $F/V3-T3/) and classify NEW vs PRE-EXISTING; valgrind memcheck a sample
  (>=20 decks) of cpu_B vs cpu_A. Root-cause NEW reports to file:line."
- V3-T4: "+ Runtime item T4: merged tree (worktree of local branch other-hhi3vb = origin/ccr-014f4f5c-hhi3vb merged
  with ccr-3dcb1e77-fem393, never push) built like B_opt and run over all examples kk t1 + CPU vs B_opt, attribute
  differences to the other branch's commits; restart A<->B; output formats; new warnings; perf A_opt vs B_opt t4
  3 repeats when load < 2."
- V3-R1..R6: file lists are in each v3/V3-R<n>.md header (all DONE).
