#!/bin/bash
# single-instance loop: every 5 min commit+push ONLY .fix-review/ on the review branch (durable across containers)
exec 9>/tmp/fix-review-checkpoint.lock; flock -n 9 || { echo "already running"; exit 0; }
cd /home/user/sparta
while true; do
  if [ ! -f .git/index.lock ] && [ -n "$(git status --porcelain -- .fix-review)" ]; then
    git add -A .fix-review && git commit -q -m "fix review: checkpoint $(date -u +%H:%M)" -- .fix-review && \
    for i in 1 2 3 4; do git push -q origin HEAD:ccr-014f4f5c-hhi3vb && break; sleep $((2**i)); done
  fi
  [ -f /tmp/fix-review-checkpoint.stop ] && exit 0
  sleep 300
done
