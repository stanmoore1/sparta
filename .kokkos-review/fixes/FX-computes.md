# FX-computes fixes

- F-G00-2 | src/KOKKOS/compute_ke_particle_kokkos.{h,cpp} | added host-set member mvv2e (removed update-> deref in operator()), k_vector_particle.modify_device() after kernel | compile OK
- F-G00-14 (= verify/G12.md F-G16-4) | src/KOKKOS/compute_reduce_kokkos.cpp | call fkk->sync_pergrid_device_kokkos() before reading fix d_vector_grid/d_array_grid | compile OK
- F-G00-9 | src/KOKKOS/compute_property_surf_kokkos.{h,cpp} | d_cglobal sized/copied nsown (all owned), kernel over 0..nsown, rows outside surf group zeroed via mask&groupbit (matches CPU pack loops) | compile OK
- G12x-F-G16-2 (verify/G12.md F-G16-2) | src/KOKKOS/compute_lambda_grid_kokkos.cpp | reallocate() now also allocates host array_grid1 (nrho_values>1), lambda_grid, lambdainv, tauinv, temp (tempwhich!=NONE) mirroring CPU reallocate, so prewrap host compute_per_grid no longer derefs NULL; freed by base dtor | compile OK
- note: F-G00-9 needed no CPU change (compute_property_surf.cpp already nsown semantics); CPU files later changed by F-G16-1/4/9 below

- F-G16-6 | src/KOKKOS/compute_ke_particle_kokkos.cpp | set invoked_per_particle in compute_per_particle_kokkos(); k_vector_particle.sync_host() after kokkos path in compute_per_particle() (modify_device already added under F-G00-2) | compile OK
- F-G16-7 | src/KOKKOS/compute_ke_particle_kokkos.cpp | grow also when k_vector_particle.extent(0) < nlocal (prewrap host-only alloc); ke = NULL after destroy_kokkos | compile OK
- F-G16-5 | src/KOKKOS/compute_distsurf_grid_kokkos.cpp | grid_kk->sync(Device,CELL_MASK|CINFO_MASK) before taking cell/cinfo device views | compile OK
- F-G16-1 | src/KOKKOS/compute_lambda_grid_kokkos.cpp, src/compute_lambda_grid.cpp | post-process nrho read back from column m (where it was written) instead of j-1 | compile OK
- F-G16-4 | src/KOKKOS/compute_count_kokkos.cpp, src/compute_count.cpp | compute_vector() sets invoked_vector instead of invoked_scalar | compile OK
- F-G16-8 | src/KOKKOS/compute_property_grid_kokkos.cpp | set invoked_per_grid at top of compute_per_grid_kokkos() | compile OK
- F-G16-9 | src/compute_property_surf.cpp | pack_area 3d: second sub3 writes p23 (was overwriting p12, p23 uninitialized) | compile OK
- F-G16-2, F-G16-3 (verify/G16.md) | REFUTED, skipped

## STATUS: COMPLETE
