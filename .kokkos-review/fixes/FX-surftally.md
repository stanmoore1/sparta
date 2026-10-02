# FX-surftally fixes

- F-G17-1 | src/KOKKOS/compute_surf_kokkos.h, src/compute_surf.cpp | TX/TY/TZ: zero pdelta_force then `if (iorig) axpy3(-origmass,vorig,...)` (same as FX branch) instead of unguarded scale3 on NULL vorig | compile OK
- F-G17-2 | src/KOKKOS/compute_surf_kokkos.h, src/KOKKOS/compute_boundary_kokkos.h | `int mvv2e` -> `double mvv2e` | compile OK
- F-G17-3 | src/KOKKOS/compute_surf_kokkos.cpp, src/KOKKOS/compute_isurf_grid_kokkos.cpp | init_normflux() no longer recreates surf2tally: grows it (new tail = -1) and sizes tally2surf/array_surf_tally to max(nsurf, current), never shrinking; tallyinfo() and grow_tally() use d_surf2tally.extent(0) so tallies made before a mid-cycle reallocate (fix balance) survive | compile OK
- F-G00-19 (isurf/grid part) | src/KOKKOS/compute_isurf_grid_kokkos.cpp | tallyinfo: `while (iend > 0 && h_surf2tally[iend] == -1)` operand order | compile OK
