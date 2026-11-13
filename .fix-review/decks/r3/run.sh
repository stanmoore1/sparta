#!/bin/bash
# r3 CI replica queue. Idempotent: markers in $FR/v/r3-state/<tag>.{configured,built,tested}
source /home/user/sparta/.fix-review/tools/env.sh
ST=$FR/v/r3-state; W=$SP/v7/r3; mkdir -p $ST $W
F=$(cat $FR/F.txt); M=5702a655006b76fbe41de68d64c8f65249b64d23
for job in mpi kk; do for side in F M; do
  tag=$side-$job; wt=$W/$tag; rev=$F; [ $side = M ] && rev=$M
  [ -f $ST/$tag.tested ] && continue
  if [ ! -d $wt/src ]; then rm -rf $wt; git -C /home/user/sparta worktree prune; git -C /home/user/sparta worktree add --detach $wt $rev || exit 1; rm -f $ST/$tag.configured $ST/$tag.built; fi
  if [ ! -f $ST/$tag.configured ]; then
    rm -rf $wt/build; mkdir -p $wt/build; cd $wt/build
    if [ $job = mpi ]; then
      cmake -C ../cmake/presets/mpi.cmake -DSPARTA_ENABLE_TESTING=ON -DPKG_FFT=ON -DPKG_PYTHON=ON ../cmake > $W/$tag.configure.log 2>&1 || { echo "configure failed $tag"; exit 1; }
    else
      cmake -C ../cmake/presets/kokkos_mpi_only.cmake -DSPARTA_ENABLE_TESTING=ON -DSPARTA_KOKKOS_EXACT=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF -DPKG_FFT=ON -DPKG_PYTHON=ON "-DSPARTA_SPA_ARGS=-k on -sf kk" ../cmake > $W/$tag.configure.log 2>&1 || { echo "configure failed $tag"; exit 1; }
    fi
    touch $ST/$tag.configured
  fi
  cd $wt/build
  if [ ! -f $ST/$tag.built ]; then make -j2 > $W/$tag.build.log 2>&1 || { echo "build failed $tag"; exit 1; }; touch $ST/$tag.built; fi
  rm -rf $wt/build/examples/*/olog* 2>/dev/null
  ctest -j2 --timeout 900 --output-on-failure > $W/$tag.ctest.log 2>&1
  grep -E "tests passed|^\s+[0-9]+ - " $W/$tag.ctest.log > $ST/$tag.summary.txt
  touch $ST/$tag.tested
done; done
echo ALLDONE
