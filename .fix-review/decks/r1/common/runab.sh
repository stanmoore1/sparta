#!/bin/bash
# runab.sh <id> <deck> <kind> <np> [timeout] [bins] : copies decks/r1/common + decks/r1/<id> into
#   $SP/v7/r1/<id>/<deck>.<kind>.np<np>.<bin>, runs prep.sh if present, then the deck on each bin (default "fin mst")
# appends SPLEAK=<n>: number of SPARTA_NS frames in LeakSanitizer reports (0 = only MPI/libevent internal leaks)
source /home/user/sparta/.fix-review/tools/env.sh
id=$1; deck=$2; k=$3; np=$4; to=${5:-120}; bins=${6:-"fin mst"}
for b in $bins; do
  w=$SP/v7/r1/$id/$deck.$k.np$np.$b; rm -rf $w; mkdir -p $w; cd $w
  cp $FR/decks/r1/common/* . 2>/dev/null; cp $FR/decks/r1/$id/* .
  [ -f prep.sh ] && bash prep.sh > /dev/null 2>&1
  r=$($FR/tools/run.sh $b $k $np $to $deck)
  n=$(sed -n '/LeakSanitizer/,$p' log.$b.$k.np$np 2>/dev/null | grep -c 'SPARTA_NS')
  echo "$r" | sed "s|^RESULT|RESULT $id $deck|; s|$| SPLEAK=$n|"
done
