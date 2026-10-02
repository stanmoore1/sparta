# FX-computes fixes

- F-G00-2 | src/KOKKOS/compute_ke_particle_kokkos.{h,cpp} | added host-set member mvv2e (removed update-> deref in operator()), k_vector_particle.modify_device() after kernel | compile OK
- F-G00-14 (= verify/G12.md F-G16-4) | src/KOKKOS/compute_reduce_kokkos.cpp | call fkk->sync_pergrid_device_kokkos() before reading fix d_vector_grid/d_array_grid | compile OK
- F-G00-9 | src/KOKKOS/compute_property_surf_kokkos.{h,cpp} | d_cglobal sized/copied nsown (all owned), kernel over 0..nsown, rows outside surf group zeroed via mask&groupbit (matches CPU pack loops) | compile OK
