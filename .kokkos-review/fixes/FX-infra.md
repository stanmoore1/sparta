# FX-infra fixes

- F-G21-1 | src/KOKKOS/kokkos.cpp | error if ngpus <= 0 right after parsing "g" arg (before local_rank % ngpus) | compile OK
- F-G21-2 | src/KOKKOS/kokkos.cpp | added `if (iarg+2 > narg)` bounds check to "t"/"threads" branch | compile OK
