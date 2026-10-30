#!/bin/bash
# Idempotent: (re)create every worktree and binary used by round V3, building only what is missing
# or out of date.  Safe to re-run after a container restart or in a fresh session (set S to the
# new session's scratchpad if it differs).  Builds run one at a time; log per build in $F/ensure_*.log.
# Usage: nohup .kokkos-review/ensure_builds.sh > $F/ensure.log 2>&1 &     (grep "rc=" $F/ensure.log)
S=${S:-/tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad}
F=$S/final
R=/home/user/sparta
J=${J:-4}
mkdir -p $F
exec 8>/tmp/kokkos-review-ensure.lock; flock -n 8 || { echo "already running"; exit 0; }

# worktrees: A source (pre-review) and HEAD + split-memory tool
[ -d $S/base/src ] || git -C $R worktree add --detach $S/base e071055f
[ -d $S/sync_A/src ] || git -C $R worktree add --detach $S/sync_A 5a2c2a52
if [ ! -d $F/src_sync/src ]; then
  git -C $R worktree add --detach $F/src_sync HEAD
  ( cd $F/src_sync && git -c user.email=x@y -c user.name=x merge -q --no-edit 5a2c2a52 ) || {
    # known conflicts: keep HEAD's code, spelled with the tool's SPARTA_NS DualView/resize wrappers
    ( cd $F/src_sync && python3 - <<'PY'
import re
def fix(p, f):
    s=open(p).read()
    s=re.sub(r'<<<<<<< [^\n]*\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', lambda m: f(m.group(1)), s, flags=re.S)
    open(p,'w').write(s)
fix('src/KOKKOS/collide_vss_kokkos.h', lambda a: a.replace('Kokkos::DualView','SPARTA_NS::DualView'))
fix('src/KOKKOS/grid_kokkos.cpp',     lambda a: a.replace('Kokkos::resize','SPARTA_NS::resize'))
PY
      git add -A && git -c user.email=x@y -c user.name=x commit -q -m "merge sync tool (local)" )
  }
fi
# keep src_sync at HEAD's src: merge any newer HEAD commits in
( cd $F/src_sync && git -c user.email=x@y -c user.name=x merge -q --no-edit ccr-3dcb1e77-fem393 >/dev/null 2>&1 || echo "WARN src_sync merge of HEAD needs manual resolution" )

KK_OMP="-DPKG_KOKKOS=on -DBUILD_KOKKOS=on -DKokkos_ENABLE_OPENMP=on -DPKG_FFT=on -DFFT=KISS -DFFT_KOKKOS=KISS"
SYNC="-DBUILD_MPI=on -DBUILD_KOKKOS=on -DPKG_KOKKOS=on -DPKG_FFT=on -DKokkos_ENABLE_SERIAL=on -DKokkos_ENABLE_OPENMP=off -DKokkos_ENABLE_ATOMICS_BYPASS=ON -DSPARTA_KOKKOS_DEBUG_SYNC=on"

# name | source cmake dir | build dir | configure options
build() {
  local name=$1 src=$2 dir=$3; shift 3
  if [ ! -f $dir/CMakeCache.txt ]; then
    cmake -S $src -B $dir -G Ninja "$@" > $F/ensure_c_$name.log 2>&1 || { echo "$name configure rc=$?"; return; }
  fi
  cmake --build $dir -j$J > $F/ensure_b_$name.log 2>&1; echo "$name rc=$?"
}
build A_opt    $S/base/cmake      $S/build_base  -DCMAKE_CXX_FLAGS="-g -O3" -DPKG_MPI_STUBS=on $KK_OMP
build B_opt    $R/cmake           $S/build       -DCMAKE_CXX_FLAGS="-g -O3" -DPKG_MPI_STUBS=on $KK_OMP
build A_mpi    $S/base/cmake      $S/bmpi_base   -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=on -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=on $KK_OMP
build B_mpi    $R/cmake           $S/bmpi_fixed  -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=on -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=on $KK_OMP
build B_sync   $F/src_sync/cmake  $F/bsync       -DCMAKE_BUILD_TYPE=Release $SYNC
build B_poison $F/src_sync/cmake  $F/bpoison     -DCMAKE_BUILD_TYPE=RelWithDebInfo $SYNC -DSPARTA_KOKKOS_DEBUG_SYNC_ASAN=on
build A_poison $S/sync_A/cmake    $S/bpoison_A   -DCMAKE_BUILD_TYPE=RelWithDebInfo $SYNC -DSPARTA_KOKKOS_DEBUG_SYNC_ASAN=on
for t in A B; do
  if [ $t = B ]; then src=$R/cmake; else src=$S/base/cmake; fi
  build cpu_$t  $src $F/bcpu_$t  -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=on -DPKG_FFT=on -DCMAKE_CXX_FLAGS="-g -Wall -Wextra"
  build stub_$t $src $F/bstub_$t -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=off -DPKG_MPI_STUBS=on -DPKG_FFT=on
done
build B_asan   $R/cmake           $F/basan       -DCMAKE_BUILD_TYPE=Debug -DBUILD_MPI=off -DPKG_MPI_STUBS=on $KK_OMP \
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=on -DCMAKE_CXX_FLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
echo ALLDONE
