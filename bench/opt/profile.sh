#!/bin/bash
# Profile a SPARTA binary on the in.collide benchmark.
#
#   profile.sh gprof     BINARY TAG [SIZE] [REORDER]
#   profile.sh callgrind BINARY TAG [SIZE] [REORDER]
#
# gprof needs a binary built with -pg (bench/opt/build.sh prof).
# callgrind wants a binary with -g but still -O3 (build.sh debugsym); it runs
# ~50x slower, so use a small size. It gives exact instruction counts and,
# with --cache-sim, D1/LL miss counts per function -- which is what the
# roofline's byte figures come from, since this KVM guest has no PMU.
set -eu

MODE="${1:?gprof|callgrind}"
BIN="${2:?need binary}"
TAG="${3:?need tag}"
SIZE="${4:-100K}"
REORDER="${5:-5}"

BENCHDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$BENCHDIR/opt/prof"
mkdir -p "$OUT"
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

case "$SIZE" in
  10K)  GX=10;  GY=10;  GZ=10  ;;
  100K) GX=20;  GY=20;  GZ=25  ;;
  1M)   GX=40;  GY=50;  GZ=50  ;;
  *,*,*) IFS=',' read -r GX GY GZ <<< "$SIZE" ;;
  *) echo "unknown size $SIZE" >&2; exit 1 ;;
esac

cd "$BENCHDIR"
ARGS="-var x $GX -var y $GY -var z $GZ -var reorder $REORDER"

if [ "$MODE" = "gprof" ]; then
  rm -f gmon.out
  taskset -c 2 "$BIN" $ARGS < in.collide.opt > "$OUT/$TAG.gprof.run.log" 2>&1
  [ -f gmon.out ] || { echo "no gmon.out; was the binary built with -pg?" >&2; exit 1; }
  gprof "$BIN" gmon.out > "$OUT/$TAG.gprof.txt" 2>/dev/null
  mv gmon.out "$OUT/$TAG.gmon.out"
  echo "=== $TAG: gprof flat profile, top 20 ==="
  sed -n '1,28p' "$OUT/$TAG.gprof.txt"
  echo "full profile: $OUT/$TAG.gprof.txt"

elif [ "$MODE" = "callgrind" ]; then
  taskset -c 2 valgrind --tool=callgrind \
      --callgrind-out-file="$OUT/$TAG.callgrind.out" \
      --cache-sim=yes --branch-sim=yes \
      --separate-callers=0 \
      "$BIN" $ARGS < in.collide.opt > "$OUT/$TAG.callgrind.run.log" 2>&1
  callgrind_annotate --threshold=95 "$OUT/$TAG.callgrind.out" \
      > "$OUT/$TAG.callgrind.txt" 2>/dev/null || true
  echo "=== $TAG: callgrind, top functions ==="
  sed -n '/Ir /,/^$/p' "$OUT/$TAG.callgrind.txt" | head -40
  echo "full annotation: $OUT/$TAG.callgrind.txt"

else
  echo "unknown mode $MODE" >&2; exit 1
fi
