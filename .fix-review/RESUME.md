# RESUME GUIDE — SPARTA fix review, branch ccr-014f4f5c-hhi3vb (read first after ANY interruption)
Everything durable lives here, in git (.fix-review/ is committed+pushed every 5 min by tools/checkpoint.sh).
The session scratchpad ($SP, see tools/env.sh) holds only rebuildable things (worktrees, binaries, run outputs);
it is LOST when the container is replaced (this happened on 2026-11-05, see LEDGER).

## After a new container / session / limit
1. cd /home/user/sparta && git fetch origin && git checkout -B ccr-014f4f5c-hhi3vb origin/ccr-014f4f5c-hhi3vb
2. nohup .fix-review/tools/checkpoint.sh >/dev/null 2>&1 &          (single instance; safe to rerun)
3. source .fix-review/tools/env.sh; mkdir -p $SP/ab; nohup .fix-review/tools/ensure_builds.sh > $SP/ab/ensure.log 2>&1 &
   (idempotent; installs OpenMPI if missing; builds fin/mst host serial+mpi+asan and kfin/kmst KOKKOS; ~1.5 h fresh)
4. For each agent in "Agents" not DONE: SendMessage it if addressable, else launch a new general-purpose agent with
   its prompt from PROMPTS.md (it resumes from its checkpoint file v/<name>.md).
5. Continue "Orchestrator pending".

## Current phase: ROUND 7 — production readiness (user: "Is this ready for production, with no new regressions?")
F (final candidate) = commit in F.txt. M = origin/master 5702a655.
Done (results in LEDGER.md and the report): master merge + reviews, v_ex (all examples), v_ab2 (chunks 15-29),
v_mst (master fixes, build hygiene), 61 of chunk 01-14 ids. Lost with the old container (must redo): rest of
chunk 01-14 lost-fix check, phase-3/KOKKOS lost-fix check, KOKKOS example sweep F vs M, CI replica.

## Agents (protocol PROTOCOL_V.md, prompts PROMPTS.md, checkpoint v/<name>.md, decks decks/<name>/)
| name | scope | status |
|---|---|---|
| r1 | lost-fix chunks 01-14 remainder + new-merge checks | RUNNING (launched 11-05) |
| r2 | phase-3 + KOKKOS lost-fix, KOKKOS example sweep F vs M | RUNNING (launched 11-05) |
| r3 | CI replica: ctest mpi + kokkos-exact on F and M, explain every failure | RUNNING (launched 11-05) |

## Orchestrator pending (ROUND 7 COMPLETE 11-05: verdict given, report v4 published)
- When r1-r3 are DONE: fix any REGRESSION/LOST FIX (commit, push, rerun affected checks), update F.txt.
- Update report report/sparta-fix-verification.html (ROWS JSON + "Round 7" section); republish with the Artifact tool
  from a copy at the SAME scratch path used before is not possible (scratch lost): publish with url=
  https://claude.ai/artifact/LhhQLVotWFzfVhQoVaWLtY (read it first; it was read on 2026-11-05).
- Answer the user: production-ready yes/no, evidence, residual risks, follow-up task (implicit-surf custom spread).
