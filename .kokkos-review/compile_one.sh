#!/bin/bash
# Usage: .kokkos-review/compile_one.sh src/KOKKOS/foo_kokkos.cpp [more files...]
# Compiles each file with the exact flags of the Kokkos OpenMP build into a private
# temp object (safe to run concurrently). Exit code != 0 on any error.
B=/tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/build
rc=0
for f in "$@"; do
  abs=$(readlink -f "$f")
  cmd=$(python3 -c "
import json,sys
cc=json.load(open('$B/compile_commands.json'))
m=[x for x in cc if x['file']=='$abs']
print(m[0]['command'] if m else '')")
  if [ -z "$cmd" ]; then echo "no compile command for $f (header? compile a .cpp that includes it)"; rc=1; continue; fi
  out=$(mktemp -d)/o.o
  cmd=$(echo "$cmd" | sed -E "s# -o [^ ]+# -o $out#")
  (cd $B/src/KOKKOS 2>/dev/null || cd $B; eval "$cmd -fsyntax-only" 2>&1 | grep -E "error|warning: unused" | head -20; exit ${PIPESTATUS[0]}) || { echo "FAILED $f"; rc=1; }
  [ $rc -eq 0 ] && echo "OK $f"
done
exit $rc
