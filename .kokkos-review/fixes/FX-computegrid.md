# FX-computegrid

- F-G00-5 | src/KOKKOS/compute_grid_kokkos.cpp, src/KOKKOS/compute_pflux_grid_kokkos.cpp | sorted per-cell kernels: `if (igroup < 0) return;` -> `continue;` (matches CPU) | compile OK
- F-G00-6 | src/KOKKOS/compute_tvib_grid_kokkos.{h,cpp} | post_process kernel: removed shared d_tspecies/d_tspecies_mode scratch (race); per-species Tsp now a thread-local double accumulated into numer/denom in a single loop (denom still includes skipped species, same indices as CPU second loop); modeflag==2 uses d_groupspecies(index/maxmode,isp) per CPU 456f0be9; dropped unused d_vibmode locals and the scratch view members/allocations | compile OK
- F-G00-7 | src/KOKKOS/compute_sonine_grid_kokkos.cpp | moved dup_vcom_tally contribute() before normalize_vcom kernel (inside non-sorted branch) so COM velocities are normalized after full accumulation | compile OK

## STATUS: COMPLETE
