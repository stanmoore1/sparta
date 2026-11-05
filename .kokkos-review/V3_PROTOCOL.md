# Round V3: production-readiness verification (no regressions)

Question: is branch ccr-3dcb1e77-fem393 (HEAD) ready for production with no regressions vs the
pre-review code A = e071055f?  Everything changed in `git diff e071055f HEAD -- src cmake`
(111 files, ~2900 lines) is in scope.  This is an independent re-review: do not trust earlier
verdicts in ab/, gpusim/, r2/; re-derive them.

S=/tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad, F=$S/final
- A source: $S/base (worktree at e071055f).  B source: /home/user/sparta (HEAD, read-only for you).
- FIXES.md maps fix IDs to commits; `git log -L` / `git log -p e071055f..HEAD -- <file>` gives the why.
- Binaries (check existence; builds run in the background, log $F/build.log, $F/build2.log):
  - A_opt $S/build_base/src/spa_kokkos_omp, B_opt $S/build/src/spa_kokkos_omp (OpenMP, MPI stubs, -O3)
  - A_mpi $S/bmpi_base/src/spa_kokkos_omp, B_mpi $S/bmpi_fixed/src/spa_kokkos_omp (OpenMP, MPI, bounds check)
  - B_sync $F/bsync/src/spa_, B_poison $F/bpoison/src/spa_ (HEAD + split-memory tool, Serial, MPI);
    A_poison $S/bpoison_A/src/spa_ (e071055f + tool). Tool guide: $F/src_sync/.github/dev-docs/kokkos-sync-debugging.md
  - cpu_A/cpu_B $F/bcpu_{A,B}/src/spa_* (no KOKKOS, MPI, -Wall -Wextra); stub_A/stub_B $F/bstub_{A,B}/src/spa_* (MPI stubs)
  - B_asan $F/basan/src/spa_* (HEAD, OpenMP, stubs, ASAN+UBSAN, bounds check)
- Kokkos run: `<bin> -k on t N -sf kk -in in.x` (Serial builds: no `t N`). MPI: `mpirun --allow-run-as-root --oversubscribe -np N`.
- Machine: 4 cores, no GPU, ~15 GB disk free: keep runs small, delete your large outputs, use `timeout`.
- Never edit /home/user/sparta/src, never git commit/push, never build in the shared dirs above,
  never `pkill -f` broad patterns.  For a modified binary use your own dir under $F/<agent>/.
- Helpers: .kokkos-review/run_examples.sh, compare_runs.py, compile_one.sh (-fsyntax-only with B_opt flags).

## Reviewer agents (V3-R1..R6): line by line over your files
For EVERY changed hunk (both A and B side), check, and write down that you checked:
1. Is the new code correct?  Off-by-one, types/overflow, uninitialised, aliasing, lifetime, leaks, error paths.
2. Regression: can it change results/behaviour for an input that A handled correctly?  Input syntax,
   restart/dump file format, output text other tools parse, defaults, performance in hot loops
   (new fences, deep_copy, atomics, host<->device syncs per step), memory growth.
3. GPU (CUDA/HIP) correctness that this machine cannot run: host-only calls/pointers/virtuals/std:: in
   KOKKOS_INLINE_FUNCTION or lambdas, `this` capture, DualView modify/sync pairing on every path,
   static/constexpr members used on device, ScatterView/atomic duplication semantics.
4. MPI: collectives reached by all ranks on all branches (including error paths), counts/types, comm freeing.
5. Parity: Kokkos style vs its CPU parent, and vice versa (fix on one side but not the other).
6. Compiles in all configs: MPI stubs, no-FFT, no-KOKKOS, CUDA (by reading), Make build (src/Makefile, Install.sh).
Every suspected problem must be VERIFIED before it is reported: a run (A vs B, positive/negative
control) or, when unrunnable, a precise code-path argument with line numbers.  Classify:
REGRESSION (B worse than A for some input) | INCOMPLETE (fix misses a case) | GPU-RISK |
BUILD | PERF | PRE-EXISTING (not caused by the branch; note only) | REFUTED.

## Checkpointing (mandatory; agents can be killed at any time)
Your file: .kokkos-review/v3/<agent>.md.  Edit tool only (never shell redirection into shared files).
Header: list of your files with status [ ] / [x].  After each file (or each 50-line chunk of a large
file) append the hunks checked and findings, and tick it.  On restart, read your file and resume at
the first unticked item.  End with a "## Findings" table (id, class, file:line, verified-by, summary)
and "## STATUS: COMPLETE".

## Runtime agents (V3-T1..T4): whole-suite sweeps
Same checkpoint rules (.kokkos-review/v3/<agent>.md, one line per example/config as it completes, so a
restart skips finished ones).  Every difference A vs B must be explained: either an intended fix
(cite the FIXES.md id / commit) or a REGRESSION.  "Statistically equal" claims need a number (e.g.
relative difference of time-averaged stats vs the run-to-run spread of A with different seeds/threads).
- T1 regression: every examples/*/in.* : A vs B for CPU styles (stub_A/stub_B and cpu_A/cpu_B np 1,2,4),
  Kokkos t 1 and t 4 (A_opt/B_opt), Kokkos MPI np 2/4 (A_mpi/B_mpi, bounds-check).
- T2 split memory: every example on B_sync and B_poison (np 1 and 2) with SPARTA_KOKKOS_STALE_STRICT=1,
  POISON, AUDIT; compare with A_poison; any report on B is a finding.
- T3 memory/UB: every example on B_asan (CPU and kk t 2) and valgrind memcheck on a sample of cpu_B runs;
  compare against the same on A (build A_asan in $F/V3-T3/ if needed) so only new reports count.
- T4 compatibility/perf/build: restart files written by A read by B and vice versa; dump/stats format
  unchanged; timing of every kk example A_opt vs B_opt (t 4, 3 repeats, report ratio); compiler warnings
  new in B (cpu_B vs cpu_A logs); and the merged tree HEAD + branch other-hhi3vb (local branch) built
  (OpenMP, stubs) and run over all examples kk + CPU vs B to check the two branches together.

## Kill-safety (all agents)
- Assume you can be killed at any moment (session limit, container restart) and relaunched as a NEW agent that
  only has this file and your checkpoint.  So: record every result INLINE in your checkpoint (numbers, verdicts,
  file:line), not only as a pointer to a scratch log; scratch dirs may be gone in a new container.
- Keep run scripts idempotent: a re-run must skip runs whose output is complete and redo partial ones.
- Put scripts and hand-written decks under $F/<agent>/ with names *.sh, *.py, in.*, *.diff: the checkpoint loop
  snapshots those (<100 kB each) into .kokkos-review/v3/scratch-files.tgz every 5 min.
- Background processes you start die with the container: after a restart, re-launch them.
