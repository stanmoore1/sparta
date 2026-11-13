#!/bin/bash
# run.sh <name> <serial|mpi|asan> <np> <timeout_s> <input> [spa args...]   (runs in CWD)
# prints: RESULT <name>/<kind> np=<np> rc=<rc> [TIMEOUT] [ASAN:<kind>] [LEAK] [UBSAN] [SEGV] [SPARTA_ERROR[<msg>]] log=<file>
source /home/user/sparta/.fix-review/tools/env.sh
b=$1; k=$2; np=$3; to=$4; inp=$5; shift 5
exe=$BIN/${b}_${k}; log=log.$b.$k.np$np
[ -x $exe ] || { echo "RESULT $b/$k np=$np rc=-1 NOBIN log=none"; exit 1; }
if [ $k = serial ]; then timeout -k 5 $to $exe -in $inp -echo none "$@" > $log 2>&1 < /dev/null
else timeout -k 5 $to mpirun --oversubscribe -np $np $exe -in $inp -echo none "$@" > $log 2>&1 < /dev/null; fi
rc=$?; t=""
[ $rc = 124 -o $rc = 137 ] && t="$t TIMEOUT"
a=$(grep -m1 -o -E "AddressSanitizer: [A-Za-z-]+" $log | awk '{print $2}'); [ -n "$a" ] && t="$t ASAN:$a"
grep -q "LeakSanitizer: detected" $log && t="$t LEAK"
grep -q "runtime error:" $log && t="$t UBSAN"
grep -q -E "Segmentation fault|signal 11|SIGSEGV" $log && t="$t SEGV"
e=$(grep -m1 "^ERROR" $log); [ -n "$e" ] && t="$t SPARTA_ERROR[$e]"
echo "RESULT $b/$k np=$np rc=$rc$t log=$PWD/$log"
