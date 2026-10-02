# Orchestration state (for resuming after session loss)
- Phase 1: first wave died on API weekly limit at ~02:00 UTC; relaunched incomplete groups 15:22 UTC Oct 2. COMPLETE: G01 G07 G08; G03 G09 G15 complete; G03b re-review of 3601-5138 launched (collision)
  To resume: for each group in GROUPS.md without `## STATUS: COMPLETE` in groups/<G>.md, relaunch a reviewer with the same prompt (protocol handles resume).
- Phase 2 (verify): per-group verifiers launched as groups complete; output in verify/<G>.md
- Phase 3 (fix): not started
- Phase 3 (fix): fixer agents own disjoint file sets; checkpoints in fixes/<FX>.md; orchestrator commits source changes per fixer. compile_one.sh = syntax check vs OpenMP build.
  Launched: FX-surfreact, FX-collide, FX-surfcollide, FX-computegrid, FX-emitsurf, FX-update, FX-surftally. Pending: FX-fixave (G00-3,4,12,20,G14), FX-fft (G00-8,21,G19,G20), FX-computes (G00-2,9,14,G16), FX-emitface (G12), FX-infra (G21), G22 fixes.
- In-progress (uncommitted) source edits are snapshotted to wip-src.patch by checkpoint.sh every 5 min; on restart: git apply .kokkos-review/wip-src.patch
- ID COLLISION NOTE: verify/G12.md verdicts for F-G14-1, F-G14-2, F-G16-1..5, F-G05-1, F-G22-3 refer to entries
  written (by script collision) into groups/G12.md; they are NOT the same as same-named IDs in G14.md/G16.md/G05.md.
  Aliases: G12x-F-G14-1 (isurf/grid computes not KokkosBase -> fix ave/histo/kk null deref, HIGH),
  G12x-F-G14-2 (= G14's F-G14-1 minmax BIG), G12x-F-G16-1 (= G16 F-G16-1 lambda), G12x-F-G16-2 (lambda/grid/kk host arrays
  unallocated before first run), G12x-F-G16-4 (= F-G00-14), G12x-F-G05-1 (= F-G05-1), G12x-F-G22-3 (SurfKokkos::grow no memset).
