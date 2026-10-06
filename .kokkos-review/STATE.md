# Orchestration state (for resuming after session loss)
- Phase 1: first wave died on API weekly limit at ~02:00 UTC; relaunched incomplete groups 15:22 UTC Oct 2. COMPLETE: G01 G07 G08; G03 G09 G15 complete; G03b re-review of 3601-5138 launched (collision)
  To resume: for each group in GROUPS.md without `## STATUS: COMPLETE` in groups/<G>.md, relaunch a reviewer with the same prompt (protocol handles resume).
- Phase 2 (verify): per-group verifiers launched as groups complete; output in verify/<G>.md
- Phase 3 (fix): ALL FIXERS COMPLETE. Next: full build + example runs (phase 4), then adversarial review of the fix diff.
- Phase 3 (fix): ALL FIXERS COMPLETE. Next: full build + example runs (phase 4), then adversarial review of the fix diff.
  Launched: FX-surfreact(done), FX-collide, FX-surfcollide, FX-computegrid(done), FX-emitsurf(done), FX-update, FX-surftally, FX-computes, FX-fixave, FX-emitface. Pending: FX-fft (after verify G19/G20), FX-infra (after verify G21), FX-grid (after verify G22). Phase 1 ALL COMPLETE.
- In-progress (uncommitted) source edits are snapshotted to wip-src.patch by checkpoint.sh every 5 min; on restart: git apply .kokkos-review/wip-src.patch
- ID COLLISION NOTE: verify/G12.md verdicts for F-G14-1, F-G14-2, F-G16-1..5, F-G05-1, F-G22-3 refer to entries
  written (by script collision) into groups/G12.md; they are NOT the same as same-named IDs in G14.md/G16.md/G05.md.
  Aliases: G12x-F-G14-1 (isurf/grid computes not KokkosBase -> fix ave/histo/kk null deref, HIGH),
  G12x-F-G14-2 (= G14's F-G14-1 minmax BIG), G12x-F-G16-1 (= G16 F-G16-1 lambda), G12x-F-G16-2 (lambda/grid/kk host arrays
  unallocated before first run), G12x-F-G16-4 (= F-G00-14), G12x-F-G05-1 (= F-G05-1), G12x-F-G22-3 (SurfKokkos::grow no memset).
- Phase 4: full rebuild running (scratchpad/build make2.log); then run examples with -k on t 2 -sf kk. Phase 5: fix-diff reviewers A/B/C launched (fixreview/).
- Phase 4: run_examples.sh new build t1 -> scratchpad/run_new_t1; baseline (e071055f) build in scratchpad/build_base (git worktree scratchpad/base) for output comparison.
- Phase 4 results: examples t1 base(e071055f) vs final: 140/141 thermo IDENTICAL; ambi differs (now runs further before pre-existing "Ran out of space" with default react/extra).
  Failures identical in base: ambi (react/extra), cylinder (rc 137 killed), implicit*/jagged.3d* (missing generated data files). Running t4 crash/NaN check -> scratchpad/run_final_t4.
- t4 run: same outcome set as t1 (no new crashes/NaN). ALL PHASES COMPLETE.
- Phase 6 (A/B every fix, user request 2026-10-05): protocol AB_PROTOCOL.md, clusters AB_CLUSTERS.md, results ab/AB1..AB9.md.
  9 agents launched. MPI+boundscheck builds of base/fixed compiling in scratchpad/bmpi_{base,fixed}. Then: write report (artifact).
  To resume: relaunch any ABn without "## STATUS: COMPLETE" with the same prompt (protocol resumes).
- 2026-10-06 02:15 UTC: session limit killed AB2/3/5/8/9; relaunched (MPI builds now done). FX-followups fixer launched for FU-1..6.
- 2026-10-06: AB11 complete (FU-3b, FU-13, FU-18 NECESSARY+COMPLETE). Found stale fix_emit_surf.cpp.o in spa_C3_opt (built concurrently with edit); clean rebuild of all changed files -> spa_C4_opt / spa_C4_mpi. Report generator: .kokkos-review/make_report.py -> publish as artifact when AB12-mpi completes.
