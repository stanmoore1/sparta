# FX-infra fixes

- F-G21-1 | src/KOKKOS/kokkos.cpp | error if ngpus <= 0 right after parsing "g" arg (before local_rank % ngpus) | compile OK
- F-G21-2 | src/KOKKOS/kokkos.cpp | added `if (iarg+2 > narg)` bounds check to "t"/"threads" branch | compile OK
- F-G21-4 | src/KOKKOS/kokkos_type.h | Kokkos::Experimental::{HIP,SYCL,HIPHostPinnedSpace,SYCLHostUSMSpace} -> Kokkos::{...} (lines 192,197,211,213,253,258,263,268); verified non-Experimental classes exist in bundled lib/kokkos (namespace Kokkos, HIP/Kokkos_HIP.hpp, SYCL/Kokkos_SYCL.hpp, *_Space.hpp); OpenMPTarget/Scatter* Experimental names left as-is | compile OK (OpenMP build: grid_kokkos.cpp, comm_kokkos.cpp, kokkos.cpp; HIP/SYCL branches not compiled in this build)
- F-G21-5 | src/KOKKOS/kokkos_type.h | t_plevel_1d/t_host_plevel_1d now typedef'd from tdual_plevel_1d | compile OK
