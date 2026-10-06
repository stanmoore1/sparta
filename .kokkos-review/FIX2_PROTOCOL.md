# Round 2 fixes (user request 2026-10-06): fix + prove NECESSARY and COMPLETE
Items:
- R2-1 (=FU-21): compute react/isurf/grid with fix balance: on a step where cells migrate, that step's
  per-cell tally lands on wrong cells (global sums right). Pre-existing, CPU + Kokkos. Check compute
  isurf/grid (CPU+kk) and any other per-grid tally compute that tallies by surf/cell during move and
  collates after balance, for the same bug. Evidence: $S/ab/AB12/FU-13 (and AB12-mpi.md).
- R2-2 (=FU-16/FU-17 + F-G20-4 rest): remap collective mode: (a) Kokkos remap3d (and check remap2d)
  crashes for a rank with no data in a collective plan (send_size[] never allocated, :689); CPU remap
  handles it; (b) MPI_Comm_group / MPI_Group_incl groups never freed (Kokkos remap3d_kokkos.cpp:789-793,
  CPU src/FFT/remap3d.cpp:609-613; check 2d both). Evidence/harness: $S/ab/AB8/fftharness, $S/ab/AB8/G20-5.
- R2-3 (=FU-14): compute react/surf init() warning count depends on #ranks with distributed surfs
  (loops lines[0..nlocal) instead of owned surfs). Also check compute react/isurf/grid warning
  (after FU-13 it counts all surfs, once per compute) and any similar init() warning counts.
S=/tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad

## Rules
- Own only the files listed in your prompt. Do NOT run git (orchestrator commits).
- NEVER build in the shared build dirs ($S/build, $S/build_base, $S/bmpi_base, $S/bmpi_fixed) and
  never `pkill -f` broad patterns. Build PRIVATE binaries in $S/r2/<item>/: compile only the .cpp files
  you changed with the exact command from $S/bmpi_fixed/compile_commands.json (MPI + bounds-check) or
  $S/build/compile_commands.json (opt), with `-o` pointing into your dir, then relink with that build's
  link line substituting your objects (see how $S/ab/AB11/c3b did it). Name binaries spa_<item>_{A,B}_{mpi,opt}.
  Note: if you change a header, every .cpp including it must be recompiled for your private binary.
- Baselines: A = current HEAD before your change (= $S/spa_C4_mpi / $S/spa_C4_opt; source /home/user/sparta at commit 39e1c1f7),
  A0 = pre-review $S/bmpi_base/src/spa_kokkos_omp. B = your private binary with the fix.
- Prove NECESSARY (positive control: A misbehaves) and COMPLETE (B correct on all variants: CPU and
  Kokkos styles, 1/2/4 ranks, 2d/3d, explicit/implicit where relevant, t1/t4; siblings checked) plus
  negative controls (A == B off the buggy path). Compile-check with .kokkos-review/compile_one.sh too.
- Checkpoint: /home/user/sparta/.kokkos-review/r2/<item>.md — append progress and results immediately
  (root cause, change summary, then per-test entries in the AB format: positive/negative control,
  necessary:, complete:, verdict:). Resume from it if it exists. End with "## STATUS: COMPLETE".
