# Environment for the AMD MI300A nodes (4 APUs per node).  Copy to
# hpc/env.sh and edit the module lines for your site.  The repository
# preset elcapitan_kokkos expects MPICH_DIR and CRAY_MPICH_ROOTDIR (set by
# the cray-mpich module).

# module load cmake rocm cray-mpich           # <-- edit for your site

PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SPARTA_EXE=$PKG/bin/spa_gpu            # built by hpc/build_gpu.sh
export SPARTA_EXE_ARGS="-k on g 1 -sf kk"
export SPARTA_CMP_EXE=$PKG/bin/spa_compare    # CPU comparison build (optional)
export SPARTA_CMP_ARGS=""
export TCE_AIR_GPU_LAUNCH=""                  # e.g. "flux run -n 1 -g 1" if needed
export KOKKOS_PRESET=elcapitan_kokkos         # cmake/presets/elcapitan_kokkos.cmake (GFX942_APU)
