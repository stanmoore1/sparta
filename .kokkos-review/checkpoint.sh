#!/bin/bash
# Periodically commit and push the review state so nothing is lost.
# Safe to start more than once: a lock keeps a single instance running.
# Start (after any container restart):  nohup .kokkos-review/checkpoint.sh >/dev/null 2>&1 &
exec 9>/tmp/kokkos-review-checkpoint.lock
flock -n 9 || exit 0
cd /home/user/sparta
while true; do
  git diff -- src > .kokkos-review/wip-src.patch 2>/dev/null
  .kokkos-review/snapshot_scratch.sh
  if [ -n "$(git status --porcelain .kokkos-review)" ]; then
    git add .kokkos-review && git commit -q -m "kokkos review: checkpoint $(date -u +%H:%M)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F8VLGsCD9g87adeU934Kd8" -- .kokkos-review
  fi
  if [ -n "$(git log origin/ccr-3dcb1e77-fem393..HEAD 2>/dev/null)" ]; then
    for d in 2 4 8 16; do git push -q -u origin ccr-3dcb1e77-fem393 && break; sleep $d; done
  fi
  sleep 300
done
