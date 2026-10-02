# FX-surftally fixes

- F-G17-1 | src/KOKKOS/compute_surf_kokkos.h, src/compute_surf.cpp | TX/TY/TZ: zero pdelta_force then `if (iorig) axpy3(-origmass,vorig,...)` (same as FX branch) instead of unguarded scale3 on NULL vorig | compile OK
- F-G17-2 | src/KOKKOS/compute_surf_kokkos.h, src/KOKKOS/compute_boundary_kokkos.h | `int mvv2e` -> `double mvv2e` | compile OK
