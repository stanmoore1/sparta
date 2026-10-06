# GPU-memory (split host/device) testing with the KOKKOS sync-debugging tool
Tool: branch origin/claude/lammps-sparta-kokkos-port-i15vd4 (5a2c2a52), guide .github/dev-docs/kokkos-sync-debugging.md
(read it fully: poison / watch / stale / audit / trace; env vars SPARTA_KOKKOS_POISON, _WATCH, _STALE, _STALE_STRICT, _AUDIT, _TRACE).
S=/tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad
- A_sync  = $S/bsync_A/src/spa_   : pre-review code (e071055f) + tool, Serial, DEBUG_SYNC, MPI      (source $S/sync_A)
- B_sync  = $S/bsync_B/src/spa_   : reviewed fixes (39e1c1f7) + tool merged, same config            (source $S/sync_B, branch gpusim-B, local only)
- A_poison/B_poison = $S/bpoison_{A,B}/src/spa_ : same + SPARTA_KOKKOS_DEBUG_SYNC_ASAN (may still be building; check existence)
Run: <bin> -k on -sf kk -in in.x   (Serial backend: no `t N`; MPI: mpirun --allow-run-as-root --oversubscribe -np N)
Rules: never build in the shared dirs above; for a modified binary make your own worktree/build dir under $S/gpusim/<agent>/
(e.g. `git -C /home/user/sparta worktree add $S/gpusim/<agent>/src gpusim-B` then patch + cmake with the same options, or
recompile single objects using commands from `ninja -C <builddir> -t commands`). Do not run git commit/push in
/home/user/sparta (orchestrator commits). Do not `pkill -f` broad patterns.
Goal per fix: NECESSARY (A_sync/A_poison shows the GPU-only fault: stale access report, watch/stale divergence, audit diff,
or wrong results vs stock/CPU) and COMPLETE (B shows none on all variants: 1/2/4 ranks, 2d/3d, relevant options).
Record results in /home/user/sparta/.kokkos-review/gpusim/<agent>.md immediately per item in AB format
(positive control, negative control, necessary:, complete:, verdict:). End with "## STATUS: COMPLETE".
