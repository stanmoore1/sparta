# Orchestration state (for resuming after session loss)
- Phase 1: first wave died on API weekly limit at ~02:00 UTC; relaunched incomplete groups 15:22 UTC Oct 2. COMPLETE: G01 G07 G08; G03 G09 G15 complete; G03b re-review of 3601-5138 launched (collision)
  To resume: for each group in GROUPS.md without `## STATUS: COMPLETE` in groups/<G>.md, relaunch a reviewer with the same prompt (protocol handles resume).
- Phase 2 (verify): per-group verifiers launched as groups complete; output in verify/<G>.md
- Phase 3 (fix): not started
- Phase 3 (fix): fixer agents own disjoint file sets; checkpoints in fixes/<FX>.md; orchestrator commits source changes per fixer. compile_one.sh = syntax check vs OpenMP build.
  Launched: FX-surfreact, FX-collide, FX-surfcollide, FX-computegrid, FX-emitsurf, FX-update, FX-surftally. Pending: FX-fixave (G00-3,4,12,20,G14), FX-fft (G00-8,21,G19,G20), FX-computes (G00-2,9,14,G16), FX-emitface (G12), FX-infra (G21), G22 fixes.
- In-progress (uncommitted) source edits are snapshotted to wip-src.patch by checkpoint.sh every 5 min; on restart: git apply .kokkos-review/wip-src.patch
