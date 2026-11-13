# Environment for the NVIDIA H100 nodes (4 GPUs per node).  Copy to
# hpc/env.sh and edit the module lines for your site.

# module load cmake gcc cuda openmpi          # <-- edit for your site

PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SPARTA_EXE=$PKG/bin/spa_gpu            # built by hpc/build_gpu.sh
export SPARTA_EXE_ARGS="-k on g 1 -sf kk"
export SPARTA_CMP_EXE=$PKG/bin/spa_compare    # CPU comparison build (optional)
export SPARTA_CMP_ARGS=""
# each run uses one GPU and one MPI rank; leave empty to start the
# executable directly, or set e.g. "mpirun -np 1" or "srun -n 1 --exact"
export TCE_AIR_GPU_LAUNCH=""
export KOKKOS_PRESET=kokkos_cuda              # cmake/presets/kokkos_cuda.cmake (HOPPER90)
