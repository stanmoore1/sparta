# FX-fixave checkpoint

- F-G00-3 | fix_ave_histo_kokkos.{h,cpp}, fix_ave_histo_weight_kokkos.cpp | replaced uninitialized GridKokkos* grid_kk member with t_cinfo_1d d_cinfo set in bin_grid_cells after sync; kernels use d_cinfo[i].mask | compile OK
- F-G14-1 (also G12x-F-G14-2, same fix) | fix_ave_histo_kokkos.cpp | added #define BIG 1.0e20 and seed minmax.min_val=BIG, max_val=-BIG at irepeat==0 (matches fix_ave_histo.cpp:552-553) | compile OK
- F-G00-4 | fix_ave_histo_kokkos.{h,cpp} | added bin_scalar(reducer,value): stores scalar_value and runs a 1-iteration parallel_reduce (TagFixAveHisto_BinScalar -> bin_one on device) with minmax_reset/fold; replaced all host bin_one(minmax,...) calls in end_of_step (compute/fix scalar, equal-style variable; incl. dead code after error->all) | compile OK
- F-G00-12 | fix_ave_histo_weight_kokkos.cpp | re-enabled BinParticles1/2 policies in bin_particles(reducer,values,stride) and implemented kernels with d_match(i) (+ d_s2g mixture test for 1), matching base class and CPU fix_ave_histo_weight.cpp:349-360 | compile OK
- G12x-F-G14-1 | fix_ave_histo_kokkos.cpp | FixAveHistoKokkos::init() (also used by ave/histo/weight/kk) now errors for a PERGRID compute input that is not KokkosBase (isurf/grid/kk, react/isurf/grid/kk) instead of NULL computeKKBase deref in end_of_step/calculate_weights | compile OK
- F-G14-6 | fix_ave_grid_kokkos.cpp | grow_percell: single maxgrid += DELTAGRID replaced by while (maxgrid < nglocal+nnew) maxgrid += DELTAGRID (fix_ave_grid.cpp:986) | compile OK
- F-G14-5 | fix_ave_grid_kokkos.cpp | added CPU tail to FixAveGridKokkos::init(): nglocal = grid->nlocal; grow_percell(0); (fix_ave_grid.cpp:444-447) | compile OK
- F-G00-20 (dup F-G14-7) | fix_ave_grid_kokkos.cpp | (latent) added CUSTOM case to init() value2index (fix_ave_grid.cpp:433-437); nvalues>1,j>0,INT custom branch now uses k_eiarray instead of k_edarray | compile OK
- G12x-F-G14-2 | n/a | duplicate of F-G14-1 (minmax BIG seed), already applied once; not double-applied
- F-G14-2 | SKIPPED | species2group staleness fixed globally by update/particle fixer
- F-G14-3, F-G14-4 | SKIPPED | REFUTED
## STATUS: COMPLETE
