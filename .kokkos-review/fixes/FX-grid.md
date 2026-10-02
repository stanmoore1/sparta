# FX-grid fixes

- F-G22-1 | src/KOKKOS/grid_kokkos.h | id_find_child: replaced ix==nx clamp with CPU id_point_child edge-check + [0,n-1] clamp (inlined, device-safe) | compile OK
- F-G22-4 | src/surf_custom.cpp, src/surf_comm.cpp | spread_inverse_custom: swapped in/out NULL guards for array branches, pass edvec[ewhich] not &edvec[ewhich]; spread_local2own n>1: idata = isurf*n (was index*n) for INT and DOUBLE | compile OK
- F-G22-5 | src/KOKKOS/fix_grid_check_kokkos.cpp | sync particles (PARTICLE_MASK) and cells (CELL_MASK) to Host before building error messages from host particles/cells | compile OK
- F-G22-3 | src/KOKKOS/surf_kokkos.cpp | SurfKokkos::grow (non-prewrap): memset host lines/tris [old,nmax) to 0 after resize, matching CPU Surf::grow (host already marked modified) | compile OK
- F-G22-2 (PLAUSIBLE, optional) | src/KOKKOS/grid_kokkos.cpp | grow_cells/grow_sinfo: dropped WithoutInitializing from k_cells/k_cinfo/k_sinfo resizes so new tail is value-initialized like CPU memset | compile OK
- F-G18-2 (CPU side) | src/compute_gas_reaction_grid.{h,cpp} | record nlist_define=react->nlist at construction; init() errors if mode!=ALL and react is NULL or react->nlist changed (same check/message as Kokkos version; distinct name to avoid shadowing the Kokkos nlist_orig member) | compile OK

## STATUS: COMPLETE
