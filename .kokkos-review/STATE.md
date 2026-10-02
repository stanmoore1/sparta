# Orchestration state (for resuming after session loss)
- Phase 1 (line-by-line review): G01-G20 launched; G21, G22 queued (agent concurrency limit 20).
  To resume: for each group in GROUPS.md without `## STATUS: COMPLETE` in groups/<G>.md, relaunch a reviewer with the same prompt (protocol handles resume).
- Phase 2 (verify): not started
- Phase 3 (fix): not started
