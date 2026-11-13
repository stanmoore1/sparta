#!/bin/bash
# kbuild.sh <name> <commit> -> $BIN/<name>_kk (KOKKOS OpenMP + MPI); idempotent
source /home/user/sparta/.fix-review/tools/env.sh
name=$1; commit=$(git -C /home/user/sparta rev-parse $2); T=$SP/ab/$name; B=$T/build_kk
[ -f $BIN/${name}.stamp ] && [ "$(cat $BIN/${name}.stamp)" = "$commit" ] && [ -x $BIN/${name}_kk ] && { echo "UPTODATE $name"; exit 0; }
cd /home/user/sparta; git worktree prune
[ -d $T ] || git worktree add -q --detach $T $commit
git -C $T checkout -q --detach $commit
mkdir -p $B && cd $B
[ -f Makefile ] || cmake -C $T/cmake/presets/kokkos_omp.cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON $T/cmake > cmake.log 2>&1 || { echo CMAKE FAIL; exit 1; }
make -j${J:-3} > make.log 2>&1 || { echo MAKE FAIL; tail -20 make.log; exit 1; }
cp $B/src/spa_kokkos_omp $BIN/${name}_kk && echo $commit > $BIN/${name}.stamp && echo "BUILT $BIN/${name}_kk"
