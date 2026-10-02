# FX-fixave checkpoint

- F-G00-3 | fix_ave_histo_kokkos.{h,cpp}, fix_ave_histo_weight_kokkos.cpp | replaced uninitialized GridKokkos* grid_kk member with t_cinfo_1d d_cinfo set in bin_grid_cells after sync; kernels use d_cinfo[i].mask | compile OK
- F-G14-1 (also G12x-F-G14-2, same fix) | fix_ave_histo_kokkos.cpp | added #define BIG 1.0e20 and seed minmax.min_val=BIG, max_val=-BIG at irepeat==0 (matches fix_ave_histo.cpp:552-553) | compile OK
- F-G00-4 | fix_ave_histo_kokkos.{h,cpp} | added bin_scalar(reducer,value): stores scalar_value and runs a 1-iteration parallel_reduce (TagFixAveHisto_BinScalar -> bin_one on device) with minmax_reset/fold; replaced all host bin_one(minmax,...) calls in end_of_step (compute/fix scalar, equal-style variable; incl. dead code after error->all) | compile OK
- F-G00-12 | fix_ave_histo_weight_kokkos.cpp | re-enabled BinParticles1/2 policies in bin_particles(reducer,values,stride) and implemented kernels with d_match(i) (+ d_s2g mixture test for 1), matching base class and CPU fix_ave_histo_weight.cpp:349-360 | compile OK
