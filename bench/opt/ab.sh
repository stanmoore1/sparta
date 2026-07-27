#!/bin/bash
# Interleaved A/B of two binaries on in.collide.
#
# This machine drifts by several percent over minutes, enough to invent or hide
# a 5% effect if A is measured as a block and then B. Runs are therefore
# alternated A,B,A,B,... and each binary is scored by its minimum, so both see
# the same stretch of machine.
#
# usage: ab.sh BIN_A BIN_B [-r REORDER] [-k CEVERY] [-n PAIRS] [-s SIZE]
set -u
BENCHDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
A="$(cd "$(dirname "${1:?need bin A}")" && pwd)/$(basename "$1")"
B="$(cd "$(dirname "${2:?need bin B}")" && pwd)/$(basename "$2")"
shift 2
R=2; K=1; N=4; SIZE="1M"; CORE=2
while getopts "r:k:n:s:c:" o; do case $o in
  r) R="$OPTARG";; k) K="$OPTARG";; n) N="$OPTARG";; s) SIZE="$OPTARG";; c) CORE="$OPTARG";;
esac; done
case "$SIZE" in
  100K) GX=20; GY=20; GZ=25;; 1M) GX=40; GY=50; GZ=50;;
  *,*,*) IFS=',' read -r GX GY GZ <<< "$SIZE";;
esac
cd "$BENCHDIR"; mkdir -p opt/logs
declare -A best
for tag in A B; do best[$tag]=""; done
for ((i=0;i<N;i++)); do
  for tag in A B; do
    bin=$([ "$tag" = A ] && echo "$A" || echo "$B")
    L="opt/logs/ab.$(basename $bin).r$R.k$K.$i.log"
    taskset -c "$CORE" "$bin" -var x $GX -var y $GY -var z $GZ \
        -var reorder "$R" -var cevery "$K" < in.collide.opt > "$L" 2>&1 || { echo "FAIL $L"; exit 1; }
    t=$(grep "^Loop time" "$L" | sed -n 2p | awk '{print $4}')
    cur="${best[$tag]}"
    best[$tag]=$(python3 -c "a='$cur';b='$t';print(b if not a else min(a,b,key=float))")
  done
done
python3 -c "
a=float('${best[A]}'); b=float('${best[B]}')
print('  A %-24s %8.3f s' % ('$(basename $A)', a))
print('  B %-24s %8.3f s   %.3fx' % ('$(basename $B)', b, a/b))
"
