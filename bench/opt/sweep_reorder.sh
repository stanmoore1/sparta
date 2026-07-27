#!/bin/bash
# Sweep the "global particle/reorder N" period for a given binary/size.
# reorder 0 disables reordering entirely (sort() still runs, since collide needs it).
set -u
BIN="${1:?need binary}"
SIZE="${2:-1M}"
REPS="${3:-3}"
TAG="${4:-sweep}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "reorder sweep: bin=$BIN size=$SIZE reps=$REPS"
for r in 0 1 2 5 10 20 50 100; do
  "$HERE/run_bench.sh" -b "$BIN" -s "$SIZE" -r "$r" -n "$REPS" -t "$TAG"
done
