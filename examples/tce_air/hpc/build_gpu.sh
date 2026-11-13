#!/bin/bash
# Build the KOKKOS GPU executable of this branch into bin/spa_gpu (and, with
# --exact, bin/spa_gpu_exact built with -DSPARTA_KOKKOS_EXACT=ON for the
# CPU/GPU comparison).  Source hpc/env.sh first (it sets KOKKOS_PRESET:
# kokkos_cuda for H100, elcapitan_kokkos for MI300A, and loads modules).
set -e
PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=$(cd "$PKG/../.." && pwd)
: "${KOKKOS_PRESET:?source hpc/env.sh first}"
JOBS=${JOBS:-16}
build () {   # build DIR EXACT(ON|OFF) NAME
  mkdir -p "$1" && cd "$1"
  cmake -C "$ROOT/cmake/presets/$KOKKOS_PRESET.cmake" \
        -DCMAKE_BUILD_TYPE=Release -DSPARTA_KOKKOS_EXACT=$2 "$ROOT/cmake"
  make -j "$JOBS"
  mkdir -p "$PKG/bin" && cp src/spa_* "$PKG/bin/$3"
  cd "$PKG"
}
build "$PKG/build/gpu" OFF spa_gpu
[ "$1" = "--exact" ] && build "$PKG/build/gpu_exact" ON spa_gpu_exact
ls -l "$PKG/bin"
