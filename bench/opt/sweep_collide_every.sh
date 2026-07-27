#!/bin/bash
# Sweep the collision sub-cycling period `global collide/every K` and report
# both cost and the statistics that must not move: equilibrium temperature and
# the mean collision rate per timestep.
#
# usage: sweep_collide_every.sh -b BINARY [-s SIZE] [-r REORDER] [-n REPS] [-k "1 2 4 8"]
set -u

BENCHDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN=""; SIZE="1M"; REORDER="2"; REPS=3; KLIST="1 2 4 8 16"; CORE=2
while getopts "b:s:r:n:k:c:" o; do case $o in
  b) BIN="$OPTARG";; s) SIZE="$OPTARG";; r) REORDER="$OPTARG";;
  n) REPS="$OPTARG";; k) KLIST="$OPTARG";; c) CORE="$OPTARG";;
esac; done
[ -z "$BIN" ] && { echo "need -b" >&2; exit 1; }
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

case "$SIZE" in
  10K) GX=10; GY=10; GZ=10;; 100K) GX=20; GY=20; GZ=25;;
  1M) GX=40; GY=50; GZ=50;; *,*,*) IFS=',' read -r GX GY GZ <<< "$SIZE";;
  *) echo "unknown size $SIZE" >&2; exit 1;;
esac

LOGDIR="$BENCHDIR/opt/logs"; mkdir -p "$LOGDIR"; cd "$BENCHDIR"

printf '%6s %10s %8s | %7s %7s %7s | %10s %12s %12s\n' \
  K loop_s speedup move sort coll "c_temp" "ncoll_total" "ncoll/step"
BASE=""
for K in $KLIST; do
  best=""
  for ((i=0;i<REPS;i++)); do
    LOG="$LOGDIR/sub.${SIZE}.r${REORDER}.k${K}.${i}.log"
    taskset -c "$CORE" "$BIN" -var x $GX -var y $GY -var z $GZ \
        -var reorder "$REORDER" -var cevery "$K" < in.collide.opt > "$LOG" 2>&1 \
      || { echo "FAILED $LOG" >&2; tail -5 "$LOG" >&2; exit 1; }
    t=$(grep "^Loop time" "$LOG" | sed -n '2p' | awk '{print $4}')
    [ -z "$best" ] && best="$t" && bestlog="$LOG"
    awk -v a="$t" -v b="$best" 'BEGIN{exit !(a<b)}' && { best="$t"; bestlog="$LOG"; }
  done
  # median is overkill here; the minimum of REPS is the cleanest estimator on a
  # noisy shared VM, and is used consistently for every row
  L="$bestlog"
  TEMP=$(awk '/^Step/{n++} n==2 && /^[0-9]/{v=$6} END{print v}' "$L")
  NC=$(awk '/^Step/{n++} n==2 && /^[0-9]/{if(f)s+=$5; f=1} END{print s}' "$L")
  NCPS=$(awk -v s="$NC" 'BEGIN{printf "%.0f", s/100}')
  BR=$(awk '/^Move    \|/{n++; if(n==2) m=1}
            m && /^(Move|Sort|Coll) +\|/ {printf "%s ", $3}
            m && /^Other/{exit}' "$L")
  [ -z "$BASE" ] && BASE="$best"
  SP=$(awk -v a="$BASE" -v b="$best" 'BEGIN{printf "%.2fx", a/b}')
  printf '%6s %10s %8s | %s| %10s %12s %12s\n' "$K" "$best" "$SP" "$BR" "$TEMP" "$NC" "$NCPS"
done
