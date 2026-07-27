#!/bin/bash
# Bitwise verification: run two binaries on the same input and confirm they
# produce identical per-step physics output.
#
# The stats columns are "step cpu np nattempt ncoll c_temp". "cpu" is wall
# time and obviously differs, so it is stripped; everything else must match
# character for character. Any difference at all means the optimization
# perturbed the RNG stream or the arithmetic, which for a Tier A change is a
# bug, not an acceptable variation.
#
# usage: verify.sh REF_BINARY NEW_BINARY [SIZE] [REORDER] [INPUT]
set -u

REF="${1:?need reference binary}"
NEW="${2:?need new binary}"
SIZE="${3:-100K}"
REORDER="${4:-5}"
INPUT="${5:-in.collide.opt}"

BENCHDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$BENCHDIR/opt/logs"
mkdir -p "$OUT"

case "$SIZE" in
  10K)  GX=10;  GY=10;  GZ=10  ;;
  100K) GX=20;  GY=20;  GZ=25  ;;
  1M)   GX=40;  GY=50;  GZ=50  ;;
  10M)  GX=100; GY=100; GZ=100 ;;
  *,*,*) IFS=',' read -r GX GY GZ <<< "$SIZE" ;;
  *) echo "unknown size $SIZE" >&2; exit 1 ;;
esac

cd "$BENCHDIR"

extract () {
  # keep only the numeric stats table lines, drop the cpu column (field 2)
  awk '/^ *Step +CPU/{on=1; next}
       /^Loop time/{on=0}
       on && NF>=5 && $1 ~ /^[0-9]+$/ {printf "%s", $1; for(i=3;i<=NF;i++) printf " %s", $i; print ""}'
}

for pair in "ref:$REF" "new:$NEW"; do
  name="${pair%%:*}"; bin="${pair#*:}"
  "$bin" -var x $GX -var y $GY -var z $GZ -var reorder "$REORDER" \
      < "$INPUT" > "$OUT/verify.$name.raw" 2>&1 || {
        echo "run failed for $name ($bin)"; tail -5 "$OUT/verify.$name.raw"; exit 1; }
  extract < "$OUT/verify.$name.raw" > "$OUT/verify.$name.stats"
done

NLINES=$(wc -l < "$OUT/verify.ref.stats")
if diff -q "$OUT/verify.ref.stats" "$OUT/verify.new.stats" > /dev/null; then
  echo "IDENTICAL  ($NLINES stat lines, size=$SIZE reorder=$REORDER)"
  exit 0
else
  echo "DIFFERS    (size=$SIZE reorder=$REORDER)"
  echo "--- first differing lines ---"
  diff "$OUT/verify.ref.stats" "$OUT/verify.new.stats" | head -20
  exit 1
fi
