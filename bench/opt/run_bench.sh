#!/bin/bash
# Timing harness for the SPARTA in.collide benchmark.
#
# Runs a binary N times and reports the median of the *second* "Loop time"
# (the 100-step benchmark run; the first run is the 30-step equilibration),
# plus SPARTA's own Move/Coll/Sort/Comm timer breakdown from the median run.
#
# usage: run_bench.sh -b BINARY [-i INPUT] [-s SIZE] [-r REORDER] [-n REPS] [-t TAG]
#   SIZE is one of: 10K 100K 1M 10M, or an explicit "x,y,z"
#
# All runs are pinned to a single core to reduce scheduler noise.

set -u

BENCHDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN=""
INPUT="in.collide.opt"
SIZE="1M"
REORDER="10"
REPS=5
TAG=""
CORE=2

while getopts "b:i:s:r:n:t:c:" opt; do
  case $opt in
    b) BIN="$OPTARG" ;;
    i) INPUT="$OPTARG" ;;
    s) SIZE="$OPTARG" ;;
    r) REORDER="$OPTARG" ;;
    n) REPS="$OPTARG" ;;
    t) TAG="$OPTARG" ;;
    c) CORE="$OPTARG" ;;
    *) echo "bad option" >&2; exit 1 ;;
  esac
done

[ -z "$BIN" ] && { echo "need -b BINARY" >&2; exit 1; }
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

# grid dimensions per the sizes documented in bench/README
case "$SIZE" in
  10K)  GX=10;  GY=10;  GZ=10  ;;
  100K) GX=20;  GY=20;  GZ=25  ;;
  1M)   GX=40;  GY=50;  GZ=50  ;;
  10M)  GX=100; GY=100; GZ=100 ;;
  *,*,*) IFS=',' read -r GX GY GZ <<< "$SIZE" ;;
  *) echo "unknown size $SIZE" >&2; exit 1 ;;
esac

[ -z "$TAG" ] && TAG="$(basename "$BIN")"
LOGDIR="$BENCHDIR/opt/logs"
mkdir -p "$LOGDIR"

cd "$BENCHDIR"

times=()
for ((i=0; i<REPS; i++)); do
  LOG="$LOGDIR/${TAG}.${SIZE}.r${REORDER}.${i}.log"
  taskset -c "$CORE" "$BIN" -var x $GX -var y $GY -var z $GZ -var reorder "$REORDER" \
      < "$INPUT" > "$LOG" 2>&1
  if [ $? -ne 0 ]; then echo "RUN FAILED, see $LOG" >&2; tail -5 "$LOG" >&2; exit 1; fi
  # second "Loop time" = the 100-step benchmark run
  t=$(grep "^Loop time" "$LOG" | sed -n '2p' | awk '{print $4}')
  [ -z "$t" ] && { echo "no loop time in $LOG" >&2; exit 1; }
  times+=("$t")
done

# median + spread, and the timer breakdown of the median run
printf '%s\n' "${times[@]}" | sort -g > /tmp/_bench_times.$$
MED=$(awk 'NR==int((n+1)/2)' n="$REPS" /tmp/_bench_times.$$)
MIN=$(head -1 /tmp/_bench_times.$$)
MAX=$(tail -1 /tmp/_bench_times.$$)
rm -f /tmp/_bench_times.$$

# find which repetition produced the median, to pull its breakdown
MEDLOG=""
for ((i=0; i<REPS; i++)); do
  LOG="$LOGDIR/${TAG}.${SIZE}.r${REORDER}.${i}.log"
  t=$(grep "^Loop time" "$LOG" | sed -n '2p' | awk '{print $4}')
  [ "$t" = "$MED" ] && { MEDLOG="$LOG"; break; }
done

# the second occurrence of the timer table is the benchmark run
BREAK=$(awk '/^Move    \|/{n++; if(n==2) m=1}
             m && /^(Move|Coll|Sort|Comm|Modify|Output) +\|/ {printf "%s=%s ", $1, $3}
             m && /^Other/{exit}' "$MEDLOG")
MOVES=$(grep "Particle-moves/CPUsec/proc" "$MEDLOG" | tail -1 | awk '{print $2}')

echo "tag=$TAG size=$SIZE reorder=$REORDER reps=$REPS"
echo "  loop_time_median=$MED  min=$MIN  max=$MAX  spread=$(awk -v a="$MIN" -v b="$MAX" -v m="$MED" 'BEGIN{printf "%.2f%%", 100*(b-a)/m}')"
echo "  moves/cpusec=$MOVES"
echo "  breakdown: $BREAK"
echo "  log: $MEDLOG"
