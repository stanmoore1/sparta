# Round 7 (resumed) verification protocol — SPARTA branch ccr-014f4f5c-hhi3vb
Goal: decide whether F (commit in /home/user/sparta/.fix-review/F.txt) is production-ready: no REGRESSION (F worse
than M = origin/master on any input) and no LOST FIX (a fix verified earlier no longer works on F).

## Environment
- `source /home/user/sparta/.fix-review/tools/env.sh` sets SP (scratch), BIN, MPI root vars, sanitizer options.
- Binaries in $BIN: fin_{serial,mpi,asan} (F), mst_{serial,mpi,asan} (M), kfin_kk, kmst_kk (KOKKOS OpenMP+MPI).
  They may still be building: wait until they exist (`tail $SP/ab/ensure.log`). Trees: $SP/ab/fin, $SP/ab/mst (read-only).
- Runner: `$FR/tools/run.sh <fin|mst> <serial|mpi|asan> <np> <timeout_s> <input> [args]` in the CWD; prints a RESULT line
  (rc, TIMEOUT, ASAN:kind, LEAK, UBSAN, SEGV, SPARTA_ERROR[msg]) and the log path.
- KOKKOS: `OMP_NUM_THREADS=1 timeout 120 mpirun --oversubscribe -np N $BIN/kfin_kk -k on t 1 -sf kk -in in.x`
  (OMPI root vars come from env.sh).
- Evidence of earlier rounds: $FR/report/rows.json (378 rows: id, title, v=verdict, pos/comp/neg = what was run and
  observed, res, round 1..4, ch=chunk). $FR/LEDGER.md. Fix commits: `git -C /home/user/sparta log --oneline e071055f..F`
  (round-1 fixes are grouped per subsystem, e.g. a2361d44 "Fix grid computes ..."; later commits name the finding).
  The original decks were lost with the old container: recreate small decks from the rows' pos/neg text.

## Durability rules (the session can be killed and the container replaced at any time)
- Checkpoint file: $FR/v/<name>.md. Read it on start; skip items already recorded. Append a record as soon as each item
  is decided. A "## RESTART" section must always say how to relaunch anything running in the background.
- Put every hand-written deck/script you need to keep under $FR/decks/<name>/<id>/ (small text files only, < 200 KB
  each; never logs, dumps or binaries). Run outputs go to $SP/v7/<name>/ (scratch, may vanish).
- Background queues must be idempotent: a unit writes its result file only when complete; reruns skip done units.
- Do NOT run git commands that change state, do not edit /home/user/sparta/src or anything outside $FR/v/<name>.md and
  $FR/decks/<name>/ (a loop commits $FR every 5 min). No subagents.
- CPU: 4 cores shared by 3 agents: at most 2 concurrent runs, np <= 4, always timeouts. Kill only your own PIDs (never
  pkill -f with broad patterns). Disk: keep > 3 GB free (`df -h /`), delete big outputs.

## Verdicts per item
OK (fixed behavior shown on F at runtime, or by diff: the fixed code is byte-identical to where it was verified AND the
file was not changed by the master merge), LOST FIX, REGRESSION, NOT-RERUN (reason). For F vs M differences: explain by
a specific fix (id/commit) or flag REGRESSION with a minimal repro, file:line.
Record format: `### <id> <title>` / `- Verdict:` / `- Evidence:` (commands, observed values, <= 6 lines).
End the file with `## V COMPLETE` and a counts line. Final answer: counts + every non-OK item with repro.
