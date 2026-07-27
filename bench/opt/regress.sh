#!/bin/bash
# Regression: run a set of in-tree example inputs with two binaries and diff
# the physics output. These deliberately cover the code paths that the
# benchmark does NOT exercise, so that optimizations aimed at the optmove +
# single-group-VSS fast path can be shown not to disturb anything else:
#
#   collide/in.collide              generic move (no optmove), single species
#   collide/in.collideInterspecies  multiple groups -> collisions_group path
#   free/in.free                    no collisions at all
#   sphere/in.sphere                surfaces present -> move<3,1,0>, surf collide
#   ambi/in.ambi                    ambipolar path
#   chem/in.chem                    chemistry, so react != NULL -> fast path
#                                   must decline and fall back
#
# usage: regress.sh REF_BINARY NEW_BINARY
set -u

REF="$(cd "$(dirname "${1:?need ref}")" && pwd)/$(basename "$1")"
NEW="$(cd "$(dirname "${2:?need new}")" && pwd)/$(basename "$2")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/bench/opt/logs/regress"
mkdir -p "$OUT"

CASES="collide/in.collide collide/in.collideInterspecies free/in.free
       sphere/in.sphere ambi/in.ambi chem/in.chem"

extract () {
  awk '/^ *Step +CPU/{on=1; next}
       /^Loop time/{on=0}
       on && NF>=3 && $1 ~ /^[0-9]+$/ {printf "%s", $1; for(i=3;i<=NF;i++) printf " %s", $i; print ""}'
}

fail=0
for c in $CASES; do
  d="$ROOT/examples/$(dirname "$c")"
  f="$(basename "$c")"
  [ -f "$d/$f" ] || { printf '%-34s SKIP (no such input)\n' "$c"; continue; }
  name="$(echo "$c" | tr '/' '_')"

  ( cd "$d" && "$REF" < "$f" > "$OUT/$name.ref.raw" 2>&1 )
  rc1=$?
  ( cd "$d" && "$NEW" < "$f" > "$OUT/$name.new.raw" 2>&1 )
  rc2=$?

  if [ $rc1 -ne 0 ] || [ $rc2 -ne 0 ]; then
    printf '%-34s RUN FAILED (ref rc=%d new rc=%d)\n' "$c" $rc1 $rc2
    tail -3 "$OUT/$name.new.raw" | sed 's/^/    /'
    fail=1
    continue
  fi

  extract < "$OUT/$name.ref.raw" > "$OUT/$name.ref.stats"
  extract < "$OUT/$name.new.raw" > "$OUT/$name.new.stats"
  n=$(wc -l < "$OUT/$name.ref.stats")

  if [ "$n" -eq 0 ]; then
    printf '%-34s NO STATS PARSED\n' "$c"; fail=1; continue
  fi

  if diff -q "$OUT/$name.ref.stats" "$OUT/$name.new.stats" > /dev/null; then
    printf '%-34s IDENTICAL (%s stat lines)\n' "$c" "$n"
  else
    printf '%-34s DIFFERS\n' "$c"
    diff "$OUT/$name.ref.stats" "$OUT/$name.new.stats" | head -6 | sed 's/^/    /'
    fail=1
  fi
done

exit $fail
