#!/bin/bash
# build.sh <name> <commit>  -> $BIN/<name>_{serial,mpi,asan}; idempotent (skips if all 3 exist and stamp matches)
source /home/user/sparta/.fix-review/tools/env.sh
name=$1; commit=$(git -C /home/user/sparta rev-parse $2); T=$SP/ab/$name
[ -f $BIN/${name}.stamp ] && [ "$(cat $BIN/${name}.stamp)" = "$commit" ] && [ -x $BIN/${name}_asan ] && { echo "UPTODATE $name"; exit 0; }
cd /home/user/sparta; git worktree prune
[ -d $T ] || git worktree add -q --detach $T $commit
git -C $T checkout -q --detach $commit
cd $T/src
make -s yes-fft yes-python > /dev/null 2>&1
make -s mpi-stubs > /dev/null 2>&1
sed -e 's/^CCFLAGS =.*/CCFLAGS =\t-O2 -g -std=c++11/' MAKE/Makefile.mpi > MAKE/Makefile.abmpi
sed -e 's/^CCFLAGS =.*/CCFLAGS =\t-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -std=c++11/' \
    -e 's/^LINKFLAGS =.*/LINKFLAGS =\t-g -fsanitize=address,undefined/' MAKE/Makefile.mpi > MAKE/Makefile.abasan
sed -e 's/^CCFLAGS =.*/CCFLAGS =\t-O2 -g -std=c++11/' MAKE/Makefile.serial > MAKE/Makefile.abserial
for v in abserial abmpi abasan; do
  make -j${J:-3} $v > $SP/ab/build_${name}_$v.log 2>&1 || { echo "BUILD FAIL $name $v"; tail -20 $SP/ab/build_${name}_$v.log; exit 1; }
  cp spa_$v $BIN/${name}_${v#ab}; rm -rf Obj_$v
done
echo $commit > $BIN/${name}.stamp; echo "BUILT $name"
