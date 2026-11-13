#!/bin/bash
# sw.sh <name> <kind> <step> [maxsize] : truncation sweep of single.20, prints one line per size that is not a clean error
source /home/user/sparta/.fix-review/tools/env.sh
n=$1; k=$2; st=$3; sz=${4:-$(stat -c %s single.20)}; mkdir -p sw; > sw/res.$n.$k
for ((s=0; s<sz; s+=st)); do head -c $s single.20 > sw/t.$s
  r=$($FR/tools/run.sh $n $k 1 ${TO:-30} in.read -var F sw/t.$s); e=$(grep -m1 ERROR log.$n.$k.np1 | sed 's/(.*//')
  echo "$s $(echo $r | sed 's/log=.*//;s/RESULT [^ ]* np=1 //;s/SPARTA_ERROR.*//') $e" >> sw/res.$n.$k; rm -f sw/t.$s
done
echo "SWEEP $n $k sizes=$(wc -l < sw/res.$n.$k) asan=$(grep -c ASAN sw/res.$n.$k) timeout=$(grep -c TIMEOUT sw/res.$n.$k) ubsan=$(grep -c UBSAN sw/res.$n.$k) noerr=$(grep -vc ERROR sw/res.$n.$k)"
