#!/bin/bash
# Usage: run_examples.sh <spa binary> <outdir> <kokkos args...>
# Runs each examples/*/in.* in a scratch copy of its directory, 600 s timeout each.
BIN=$1; OUT=$2; shift 2; KARGS="$@"
mkdir -p $OUT
for inp in /home/user/sparta/examples/*/in.*; do
  d=$(dirname $inp); ex=$(basename $d); name=$(basename $inp)
  case $ex in python|paraview|vtk) continue;; esac
  w=$OUT/work/$ex; mkdir -p $w; cp -rn $d/. $w/ 2>/dev/null
  log=$OUT/$ex.$name.log
  (cd $w && timeout 600 $BIN $KARGS -in $name -echo none -screen none -log $log >/dev/null 2>$log.err); rc=$?
  st=OK; [ $rc -eq 127 ] && st=NOBIN
  grep -qiE "^ERROR|exception" $log $log.err 2>/dev/null && st=ERROR
  grep -qiE "(^|[ \t])-?nan|[ \t]-?inf([ \t]|$)" $log 2>/dev/null && st="$st,NAN"
  [ $rc -ne 0 ] && st="$st,rc=$rc"
  echo "$ex/$name $st" | tee -a $OUT/summary.txt
done
