# FX-computes fixes

- F-G00-2 | src/KOKKOS/compute_ke_particle_kokkos.{h,cpp} | added host-set member mvv2e (removed update-> deref in operator()), k_vector_particle.modify_device() after kernel | compile OK
- F-G00-14 (= verify/G12.md F-G16-4) | src/KOKKOS/compute_reduce_kokkos.cpp | call fkk->sync_pergrid_device_kokkos() before reading fix d_vector_grid/d_array_grid | compile OK
- F-G00-9 | src/KOKKOS/compute_property_surf_kokkos.{h,cpp} | d_cglobal sized/copied nsown (all owned), kernel over 0..nsown, rows outside surf group zeroed via mask&groupbit (matches CPU pack loops) | compile OK
- G12x-F-G16-2 (verify/G12.md F-G16-2) | src/KOKKOS/compute_lambda_grid_kokkos.cpp | reallocate() now also allocates host array_grid1 (nrho_values>1), lambda_grid, lambdainv, tauinv, temp (tempwhich!=NONE) mirroring CPU reallocate, so prewrap host compute_per_grid no longer derefs NULL; freed by base dtor | compile OK
- note: no changes needed in CPU src/compute_property_surf.cpp (already nsown semantics), src/compute_lambda_grid.cpp, src/compute_count.cpp

## STATUS: COMPLETE
