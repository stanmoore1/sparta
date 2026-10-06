# FX-surftally fixes

- F-G17-1 | src/KOKKOS/compute_surf_kokkos.h, src/compute_surf.cpp | TX/TY/TZ: zero pdelta_force then `if (iorig) axpy3(-origmass,vorig,...)` (same as FX branch) instead of unguarded scale3 on NULL vorig | compile OK
- F-G17-2 | src/KOKKOS/compute_surf_kokkos.h, src/KOKKOS/compute_boundary_kokkos.h | `int mvv2e` -> `double mvv2e` | compile OK
- F-G17-3 | src/KOKKOS/compute_surf_kokkos.cpp, src/KOKKOS/compute_isurf_grid_kokkos.cpp | init_normflux() no longer recreates surf2tally: grows it (new tail = -1) and sizes tally2surf/array_surf_tally to max(nsurf, current), never shrinking; tallyinfo() and grow_tally() use d_surf2tally.extent(0) so tallies made before a mid-cycle reallocate (fix balance) survive | compile OK
- F-G00-19 (isurf/grid part) | src/KOKKOS/compute_isurf_grid_kokkos.cpp | tallyinfo: `while (iend > 0 && h_surf2tally[iend] == -1)` operand order | compile OK
- F-G17-4 | src/KOKKOS/compute_react_surf_kokkos.cpp | tallyinfo scans nsurf_tally_alloc (not current nlocal+nghost); `iend > 0 &&` evaluated first | compile OK
- F-G17-6 + F-G00-19 (react/isurf/grid part) | src/KOKKOS/compute_react_isurf_grid_kokkos.cpp | tallyinfo scans nsurf_tally_alloc; operand order `iend > 0 && h_surf2tally[iend] == -1` | compile OK
- F-G17-5 | src/compute_react_surf.cpp, src/compute_react_isurf_grid.cpp | init(): `return` -> `continue` for surfs not in group (warning/clear/combined=0 and collective Allreduce now always reached) | compile OK
- F-G18-2 | src/KOKKOS/compute_gas_reaction_grid_kokkos.{h,cpp} | Kokkos-only: record nlist_orig at construction; init() errors if mode!=ALL and react is NULL or react->nlist changed; SELECT device map sized from nlist_orig (CPU-side check not added: CPU files not owned) | compile OK
- F-G00-16 + F-G18-3 | src/KOKKOS/fix_temp_rescale_kokkos.cpp | per-cell kernel and end_of_step_average: vscale = 1.0 when t_current <= 0.0 (matches CPU fix_temp_rescale.cpp) | compile OK
- F-G17-7 | skipped (REFUTED); F-G18-1 | skipped (owned by another fixer)

## STATUS: COMPLETE
