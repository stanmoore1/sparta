#!/bin/bash
# Periodically commit and push the review state so nothing is lost.
cd /home/user/sparta
while true; do
  git diff -- src > .kokkos-review/wip-src.patch 2>/dev/null
  if [ -n "$(git status --porcelain .kokkos-review)" ]; then
    git add .kokkos-review && git commit -q -m "kokkos review: checkpoint $(date -u +%H:%M)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01F8VLGsCD9g87adeU934Kd8" -- .kokkos-review
    for d in 2 4 8 16; do git push -q -u origin ccr-3dcb1e77-fem393 && break; sleep $d; done
  fi
  sleep 300
done
