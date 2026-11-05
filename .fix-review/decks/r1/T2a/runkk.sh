#!/bin/bash
# runkk.sh <kfin|kmst>: run all T2a decks on $BIN/<b>_kk np1 (OMP 1 thread), print rc / first ERROR / last stats line
source /home/user/sparta/.fix-review/tools/env.sh
b=$1
for d in $FR/decks/r1/T2a/in.*; do n=$(basename $d); w=$SP/v7/r1/T2a/kk.$b.$n; rm -rf $w; mkdir -p $w; cd $w
  cp $FR/decks/r1/common/air.species $FR/decks/r1/T2a/* .
  OMP_NUM_THREADS=1 timeout 120 mpirun --oversubscribe -np 1 $BIN/${b}_kk -k on t 1 -sf kk -in $n -echo none > log 2>&1 < /dev/null; rc=$?
  echo "$b $n rc=$rc $(grep -m1 ERROR log | cut -c1-110) | $(awk '/^ *Step/{f=1;next} /^Loop/{f=0} f' log | tail -1 | awk '{$1=$1;print}')"
done
