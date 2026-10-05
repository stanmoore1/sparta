# A/B testing of every bugfix (phase 6)

Goal: for every fix in FIXES.md, demonstrate with a run that the bug existed (A = pre-fix
build) and is gone (B = fixed build), using positive and/or negative controls.

## Binaries  (S=/tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad)
- A_opt = $S/build_base/src/spa_kokkos_omp     (commit e071055f, MPI stubs, OpenMP, -O3)
- B_opt = $S/spa_new_final                     (fixed, MPI stubs, OpenMP)
- A_mpi = $S/bmpi_base/src/spa_kokkos_omp      (e071055f, real MPI, Kokkos DEBUG_BOUNDS_CHECK)
- B_mpi = $S/bmpi_fixed/src/spa_kokkos_omp     (fixed,   real MPI, Kokkos DEBUG_BOUNDS_CHECK)
  (the MPI builds may still be compiling; check the file exists. Run with `mpirun --allow-run-as-root --oversubscribe -np N`)
- Kokkos run:  <bin> -k on t <N> -sf kk -in in.x
- CPU reference run (same binary, non-Kokkos styles): <bin> -in in.x
- Source of A: $S/base (git worktree at e071055f); source of B: /home/user/sparta (HEAD).
- Per-fix diff: `git -C /home/user/sparta log --oneline e071055f..HEAD -- <file>`, FIXES.md maps IDs to commits.
- Example inputs/data: /home/user/sparta/examples/*. Docs: /home/user/sparta/doc/*.txt.
- Machine: 4 cores, no GPU. Keep runs small (seconds). Use `timeout`.

## Controls
- Positive control: an input that exercises the buggy path. Expect A to misbehave (wrong value vs
  CPU reference or analytic value, crash, bounds-check abort, NaN, hang->timeout, error) and B to be
  correct. Use the CPU-style run as the reference when it is not itself buggy.
- Negative control: an input on the same code path that does NOT trigger the bug; A and B must
  agree (identical or statistically equal), proving the fix didn't change unrelated behaviour.
- Races: repeat several times with t 4, compare spread/vs t 1.
- Bugs that only exist with separate host/device memory (GPU) cannot be reproduced on this machine.
  Mark them UNTESTABLE-HERE (gpu-only) but still run a negative control (A==B on CPU).
  If a source-level check is possible (e.g. instrument a copy), you may do it in a scratch copy, never
  in /home/user/sparta.
- Unreachable/latent paths: mark UNTESTABLE-HERE (unreachable) + negative control, or test the
  unit directly (e.g. standalone program including the header, cmake -P for CMake).

## Output / checkpoint
- Work dir: $S/ab/<CLUSTER>/<ID>/ (inputs, logs). Results: /home/user/sparta/.kokkos-review/ab/<CLUSTER>.md
- On start read your results file and skip IDs already present (resume).
- After EACH fix, immediately append:
  ```
  ### <ID> — <one-line bug>
  class: cpu-observable | bounds-check | race | mpi | gpu-only | unreachable | build-system
  positive control: <input summary> | A: <observed> | B: <observed> | REPRODUCED / NOT REPRODUCED / n/a
  negative control: <input summary> | A vs B: <identical / agree within noise / differ>
  verdict: VERIFIED | VERIFIED-NEG-ONLY | UNTESTABLE-HERE (<why>) | FIX-FAILS | INCONCLUSIVE
  artifacts: <dir>
  ```
  Keep each entry concise with the key numbers.
- Do NOT edit source under /home/user/sparta (only your results file). Do not run git except read-only.
- If a fix appears not to work (FIX-FAILS) record evidence clearly; do not fix it.
- When all IDs are done append `## STATUS: COMPLETE`.
