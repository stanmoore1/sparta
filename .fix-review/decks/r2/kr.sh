#!/bin/bash
# kr.sh <kfin|kmst> <np> <timeout> <input> [args] : KOKKOS run in CWD -> log.<b>.kk.np<np>, prints RC line
source /home/user/sparta/.fix-review/tools/env.sh
b=$1; np=$2; to=$3; inp=$4; shift 4; log=log.$b.kk.np$np
OMP_NUM_THREADS=1 timeout -k 5 $to mpirun --oversubscribe -np $np $BIN/${b}_kk -k on t 1 -sf kk -in $inp -echo none "$@" > $log 2>&1 < /dev/null
rc=$?; echo "KK $b np$np rc=$rc $(grep -m1 '^ERROR' $log | cut -c1-120) $(grep -q -E 'Segmentation|signal 11' $log && echo SEGV)"
