#!/bin/bash
# kq.sh <slot> <nslots> : idempotent KOKKOS example sweep queue (task 2 of r2)
# units = every examples/*/in.* (skip vtk python paraview) x {kfin,kmst} x {np1,np2}
# result: $SP/v7/r2/ex/res/<build>.np<N>.<dir>.<in>.res (written only when the unit is complete)
source /home/user/sparta/.fix-review/tools/env.sh
slot=$1; ns=$2; E=$SP/v7/r2/ex; R=$E/res; mkdir -p $R
i=0
for f in /home/user/sparta/examples/*/in.*; do
  d=$(basename $(dirname $f)); in=$(basename $f)
  case $d in vtk|python|paraview) continue;; esac
  for np in 1 2; do for b in kfin kmst; do
    i=$((i+1)); [ $((i % ns)) -ne $slot ] && continue
    res=$R/$b.np$np.$d.$in.res; [ -f $res ] && continue
    [ -x $BIN/${b}_kk ] || continue
    W=$E/$b/np$np/$d; mkdir -p $W
    [ -f $W/.copied ] || { cp -r /home/user/sparta/examples/$d/. $W/ && touch $W/.copied; }
    cd $W; log=log.$in.$b.np$np
    OMP_NUM_THREADS=1 timeout -k 5 180 mpirun --oversubscribe -np $np $BIN/${b}_kk -k on t 1 -sf kk -in $in -echo none > $log 2>&1 < /dev/null
    rc=$?
    python3 -I $FR/decks/r2/ex/stats.py $log > $res.tmp 2>&1
    { echo "RC $rc"; grep -m1 "^ERROR" $log; cat $res.tmp; } > $res.tmp2; rm -f $res.tmp; mv $res.tmp2 $res
    # keep disk small: drop big outputs of this unit
    find $W -newer $W/.copied -type f \( -name "*.ppm" -o -name "*.jpg" -o -name "dump.*" -o -name "*.restart*" -o -name "restart.*" \) -size +200k -delete 2>/dev/null
  done; done
done
echo "SLOT $slot DONE"
