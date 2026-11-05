#!/bin/bash
# present.sh <commit> <file> : list added lines of <commit> in <file> that are missing from F's version
cd /home/user/sparta; F=$(cat .fix-review/F.txt)
git show $F:$2 > /tmp/claude-0/-home-user-sparta/6123e18b-95dc-57b3-8d79-a474c7818aa8/scratchpad/v7/r1/fver
tot=0; miss=0
while IFS= read -r l; do
  t=$(echo "$l" | sed 's/^+//; s/^[ \t]*//; s/[ \t]*$//')
  [ ${#t} -lt 12 ] && continue
  tot=$((tot+1))
  grep -qF -- "$t" /tmp/claude-0/-home-user-sparta/6123e18b-95dc-57b3-8d79-a474c7818aa8/scratchpad/v7/r1/fver || { miss=$((miss+1)); echo "  MISSING: $t"; }
done < <(git show --format= $1 -- $2 | grep '^+' | grep -v '^+++')
echo "$1 $2: added=$tot missing=$miss"
