#!/bin/bash
# usage: bisect.sh "<fix ids space-separated>" <ctest regex> <label>; result -> $FR/v/r3-state/bis.<label>.txt
source /home/user/sparta/.fix-review/tools/env.sh
W=$SP/v7/r3/F-mpi; O=$SP/v7/r3/orig; out=$FR/v/r3-state/bis.$3.txt
[ -f $out ] && exit 0
cp $O/* $W/src/
for f in $1; do python3 $FR/decks/r3/revert.py $W/src $f || exit 1; done
cd $W/build && make -j1 > $SP/v7/r3/bis.$3.build.log 2>&1 || { echo build fail; exit 1; }
ctest -R "$2" --timeout 600 > $SP/v7/r3/bis.$3.ctest.log 2>&1
cp $O/* $W/src/
grep -E "Test +#" $SP/v7/r3/bis.$3.ctest.log | sed 's/Error regular.*//' > $out.tmp; mv $out.tmp $out
