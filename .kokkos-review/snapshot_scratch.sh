#!/bin/bash
# Snapshot the agents' small scripts and hand-written test decks from the session scratchpad
# into the repo (v3/scratch-files.tgz), so a fresh session/container can recover them.
# Deterministic archive (sorted names, fixed mtime/owner, gzip -n): unchanged files => unchanged tgz.
F=/tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/final
OUT=/home/user/sparta/.kokkos-review/v3/scratch-files.tgz
[ -d $F ] || exit 0
cd $F || exit 0
find build*.sh V3-* -type f \( -name "*.sh" -o -name "*.py" -o -name "*.cpp" -o -name "*.diff" -o -name "in.*" \
     -o -name "*.md" \) -size -100k \
     ! -path "*/runs/*" ! -path "*/out/*" ! -path "*/merged/*" ! -path "*/bexact*" ! -path "*/bopt/*" \
     ! -path "*/obj*" ! -path "*/CMakeFiles/*" ! -path "*/ex/*" ! -path "*/ex2/*" ! -path "*/work/*" \
     ! -path "*/lib/*" 2>/dev/null | LC_ALL=C sort > /tmp/snap.list.$$
tar --no-recursion --mtime='2026-01-01' --owner=0 --group=0 --numeric-owner -cf - -T /tmp/snap.list.$$ 2>/dev/null \
  | gzip -n -9 > $OUT.tmp && { cmp -s $OUT.tmp $OUT && rm $OUT.tmp || mv $OUT.tmp $OUT; }
rm -f /tmp/snap.list.$$
