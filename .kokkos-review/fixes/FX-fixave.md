# FX-fixave checkpoint

- F-G00-3 | fix_ave_histo_kokkos.{h,cpp}, fix_ave_histo_weight_kokkos.cpp | replaced uninitialized GridKokkos* grid_kk member with t_cinfo_1d d_cinfo set in bin_grid_cells after sync; kernels use d_cinfo[i].mask | compile OK
- F-G14-1 (also G12x-F-G14-2, same fix) | fix_ave_histo_kokkos.cpp | added #define BIG 1.0e20 and seed minmax.min_val=BIG, max_val=-BIG at irepeat==0 (matches fix_ave_histo.cpp:552-553) | compile OK
