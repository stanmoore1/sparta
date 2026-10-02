# KOKKOS package line-by-line review — protocol

Goal: every line of every file in src/KOKKOS is read and checked. Findings are
verified, then verified bugs are fixed. All state lives in this directory and is
committed+pushed periodically, so any agent can be restarted and resume.

## Reviewer agents (phase 1)
Each reviewer owns one group file: `.kokkos-review/groups/<GROUP>.md`.

1. On start, READ your group file. If it exists, it contains a `## Progress`
   section listing `DONE <file>:<start>-<end>` lines and a `## Findings` section.
   RESUME at the first range not marked DONE. Never re-review DONE ranges, never
   delete existing findings.
2. If it does not exist, create it with: header (group name, assigned
   files/ranges), `## Progress` (empty), `## Findings` (empty).
3. Review in chunks of at most ~150 lines, using Read with offset/limit so that
   EVERY line is actually read. For each chunk check every line for:
   - correctness / logic errors, off-by-one, wrong index, wrong variable, shadowing
   - Kokkos-specific: host pointer/member deref inside device kernels (this->,
     update->, grid->, etc. in KOKKOS_INLINE_FUNCTION / operator()), host code
     touching device views (d_*) without sync, missing modify()/sync() on DualViews,
     stale view handles after realloc/grow, race conditions (non-atomic writes to
     shared data in parallel_for), scatter-view contribute ordering, missing fence
     before host reads, wrong execution/memory space, uninitialized views,
     uninitialized members, resize without copy, reductions with wrong init/join
   - parity with the CPU (non-Kokkos) counterpart in src/ (e.g. src/collide_vss.cpp
     for src/KOKKOS/collide_vss_kokkos.cpp): diff the logic, flag any divergence,
     including bug fixes the CPU code got that the Kokkos version is missing
     (`git log -p --since=2023-01-01 -- src/<cpu_file>` is useful)
   - memory errors, integer overflow, divide by zero, NaN guards the CPU has
   - MPI/comm correctness, restart read/write symmetry
4. After EACH chunk, immediately append to `## Progress`:
   `DONE <file>:<start>-<end>` and append any findings to `## Findings`.
   Do this with a small Edit/append right away — do not batch. This is the
   checkpoint; work not written down is lost if the session dies.
5. Finding format (one per bullet, keep it tight):
   `- [F-<GROUP>-<n>] <file>:<line> | severity: high/med/low | <one-line defect>
     | scenario: <concrete inputs -> wrong output/crash> | cpu-ref: <file:line or n/a>`
   Only real defects (bugs, races, crashes, wrong results, missing CPU fixes).
   No style nits. If unsure, still record it with `confidence: low`.
6. Do NOT modify any source files. Only write your group file.
7. When all ranges are DONE, append `## STATUS: COMPLETE` and return a short
   summary listing the finding IDs.

## Verify phase (phase 2)
Verifier reads findings, adversarially tries to disprove each against code and
CPU reference, and writes `VERDICT: CONFIRMED|REFUTED|PLAUSIBLE <reason>` under
each finding in the group file (or in verify/<GROUP>.md).

## Fix phase (phase 3)
Only CONFIRMED findings are fixed, minimal change, matching CPU code style.
Fixed findings get `FIXED in <commit>`.

## Verifier agents (phase 2) — detailed
- Input: groups/<G>.md findings. Output/checkpoint: verify/<G>.md.
- On start, read verify/<G>.md if it exists; skip findings already having a VERDICT.
- For each finding: read the cited code (and surrounding context, callers, CPU reference),
  and try hard to DISPROVE it (is the path reachable? is there a sync/guard elsewhere?
  does the CPU code behave the same? is the "fix" already present?). Then write immediately:
  `### <ID>` / `VERDICT: CONFIRMED|REFUTED|PLAUSIBLE` / `reason: ...` / `fix: <precise minimal
  fix description, file:line, matching CPU code where applicable>`.
  If the same bug is already in the CPU code (not Kokkos-specific), note `cpu-also: yes`.
- Do not edit source files; do not run git.
- When all done append `## STATUS: COMPLETE`.

## IMPORTANT (added after collision incident)
Do NOT write helper scripts into the shared scratchpad (ck.py etc.) — several agents
overwrote each other's scripts and checkpoint entries. Update your checkpoint ONLY with
the Edit tool on your own group file, re-reading it first.

## Fixer agents (phase 3) — detailed
- Each fixer owns a disjoint set of SOURCE FILES (listed in its prompt). Edit only those files.
- Input: all findings in verify/*.md with VERDICT CONFIRMED (or PLAUSIBLE with a clear, safe fix)
  whose fix lands in your files. Skip REFUTED. Skip findings marked FIXED in FIXES.md.
- Checkpoint: fixes/<FIXER>.md. On start read it; skip findings already listed there.
- For each finding: make the minimal fix (match CPU code style and surrounding idiom; no
  refactors), then run `.kokkos-review/compile_one.sh <each .cpp you touched or that includes
  the header you touched>` from /home/user/sparta; it must print OK. Then IMMEDIATELY append to
  fixes/<FIXER>.md: `- <ID> | <files> | <one-line description of change> | compile OK`.
  If a fix is too large/risky (multi-file API change, design question), do NOT do it; append
  `- <ID> | DEFERRED | <reason + proposed patch>`.
- Do NOT run git (the orchestrator commits). Do not touch files outside your list.
- When done append `## STATUS: COMPLETE`.
