#!/bin/bash
# build_single.sh: idempotent serial FFT_SINGLE builds of F and M -> $SP/v7/r1/fft/bin/{fin,mst}_single
# also builds a double reference only if missing (uses $BIN/fin_serial / mst_serial instead)
source /home/user/sparta/.fix-review/tools/env.sh
O=$SP/v7/r1/fft; mkdir -p $O/bin
for b in fin mst; do
  [ -x $O/bin/${b}_single ] && { echo "UPTODATE $b"; continue; }
  while [ ! -f $SP/ab/$b/src/MAKE/Makefile.abserial ]; do sleep 30; done
  rm -rf $O/$b; mkdir -p $O/$b/src; (cd $SP/ab/$b/src && tar cf - --exclude='Obj_*' --exclude='spa_*' .) | (cd $O/$b/src && tar xf -)
  mkdir -p $O/$b/lib; (cd $SP/ab/$b/lib && tar cf - --exclude=kokkos .) | (cd $O/$b/lib && tar xf -)
  cd $O/$b/src || { echo "FAIL cd $b"; continue; }
  sed -e 's/^CCFLAGS =.*/CCFLAGS =\t-O2 -g -std=c++11 -DFFT_SINGLE/' MAKE/Makefile.abserial > MAKE/Makefile.absingle
  make -j1 absingle > $O/build_$b.log 2>&1 && cp spa_absingle $O/bin/${b}_single && rm -rf Obj_absingle && echo "BUILT $b" || echo "FAIL $b"
done
echo "SINGLE DONE"
