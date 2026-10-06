# Fix-diff review B
DONE src/KOKKOS/compute_ke_particle_kokkos.{h,cpp}
DONE src/KOKKOS/compute_reduce_kokkos.cpp
DONE src/KOKKOS/compute_property_surf_kokkos.{h,cpp}
DONE src/compute_property_surf.cpp
DONE src/KOKKOS/compute_count_kokkos.cpp
DONE src/compute_count.cpp
DONE src/KOKKOS/compute_property_grid_kokkos.cpp
DONE src/KOKKOS/compute_distsurf_grid_kokkos.cpp
DONE src/KOKKOS/compute_lambda_grid_kokkos.cpp
DONE src/compute_lambda_grid.cpp
DONE src/KOKKOS/compute_grid_kokkos.cpp
DONE src/KOKKOS/compute_pflux_grid_kokkos.cpp
DONE src/KOKKOS/compute_sonine_grid_kokkos.cpp
DONE src/KOKKOS/compute_tvib_grid_kokkos.{h,cpp}
DONE src/KOKKOS/compute_fft_grid_kokkos.cpp
DONE src/KOKKOS/compute_gas_reaction_grid_kokkos.{h,cpp}
DONE src/compute_gas_reaction_grid.{h,cpp}
DONE src/KOKKOS/compute_surf_kokkos.{h,cpp}
DONE src/compute_surf.cpp
DONE src/KOKKOS/compute_boundary_kokkos.h
DONE src/KOKKOS/compute_isurf_grid_kokkos.cpp
DONE src/KOKKOS/compute_react_surf_kokkos.cpp
DONE src/KOKKOS/compute_react_isurf_grid_kokkos.cpp
DONE src/compute_react_surf.cpp
DONE src/compute_react_isurf_grid.cpp
DONE src/KOKKOS/fix_ave_histo_kokkos.{h,cpp}
DONE src/KOKKOS/fix_ave_histo_weight_kokkos.cpp
DONE src/KOKKOS/fix_ave_grid_kokkos.cpp
DONE src/KOKKOS/fix_emit_face_kokkos.cpp
DONE src/KOKKOS/fix_emit_face_file_kokkos.cpp
DONE src/KOKKOS/fix_emit_surf_kokkos.cpp
DONE src/fix_emit_surf.cpp
DONE src/KOKKOS/fix_temp_rescale_kokkos.cpp
DONE src/fix_temp_rescale.cpp
DONE src/KOKKOS/fix_grid_check_kokkos.cpp

No defects found in reviewed hunks. Informational notes (no change required):
- [R-B-1] src/KOKKOS/compute_tvib_grid_kokkos.cpp:325-410 | info | merged numer/denom loop is equivalent to CPU second loop only if emap[1]==emap[0]+1 (and per-species stride matches); true for all producers (compute tally map, fix ave/grid umap), and CPU first loop already assumes it | none
- [R-B-2] src/KOKKOS/fix_emit_{face,face_file,surf}_kokkos.cpp subsonic_sort | info | when sorted_kk==1 on entry (collide w/o reactions last step) the list is not rebuilt and omits particles inserted earlier this START_OF_STEP; identical to CPU (`if (!particle->sorted) subsonic_sort()`), so no divergence | none
- [R-B-3] src/KOKKOS/compute_ke_particle_kokkos.cpp:81 | info | ke=NULL after destroy would make a later prewrap host ComputeKEParticle::compute_per_particle write through NULL (nmax already >= nlocal); unreachable since prewrap is only re-armed by a new Update (clear deletes computes); previously ke dangled | optional: also reset nmax=0 when ke is nulled
## STATUS: COMPLETE
