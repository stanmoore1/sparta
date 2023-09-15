# kokkos_hip = KOKKOS package with HIP backend, default MPI, hipcc
# compiler with OpenMPI or MPICH

include(${CMAKE_CURRENT_LIST_DIR}/kokkos_common.cmake)
# ################### BEGIN SPARTA OPTIONS ####################
set(SPARTA_MACHINE
    kokkos_cuda
    CACHE STRING
          "Descriptive string to describe \"spa_\" executable configuration"
          FORCE)
# ################### END   SPARTA OPTIONS ####################

# ################### BEGIN CMAKE OPTIONS ####################
# TODO: Should CMAKE_CXX_COMPILER be set to nvcc_wrapper
# src/KOKKOS/CMakeLists.txt? set(CMAKE_CXX_COMPILER "mpicxx" CACHE STRING "")
set(CMAKE_CXX_COMPILER
    ${CMAKE_CURRENT_LIST_DIR}/../../lib/kokkos/bin/nvcc_wrapper
    CACHE STRING "" FORCE)
# ################### END CMAKE OPTIONS ####################

# ################### BEGIN KOKKOS OPTIONS ####################
set(Kokkos_ENABLE_HIP
    ON
    CACHE STRING "")
set(Kokkos_ARCH_VEGA90A
    ON
    CACHE STRING "")
# ################### END   KOKKOS OPTIONS ####################
