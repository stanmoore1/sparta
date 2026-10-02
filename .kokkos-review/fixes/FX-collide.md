# FX-collide checkpoint
- F-G01-1 | src/KOKKOS/collide_vss_kokkos.cpp | ambipolar fix lookup matches "ambipolar" or "ambipolar/kk", errors if not found, errors if !kokkos_flag (as surf_collide_diffuse_kokkos) | compile OK | note: CPU collide.cpp:333 and react_bird.cpp:435 (ambi_check, called first when react defined) keep exact strcmp "ambipolar" -> same crash with explicit ambipolar/kk + react; not owned
- F-G01-2 | src/KOKKOS/collide_vss_kokkos.cpp | set vre_first=1 when vremax/remain reallocated (ngroups != oldgroups) | compile OK | note: CPU collide.cpp ~280 needs same one-liner; not owned
- F-G01-3 | src/KOKKOS/collide_vss_kokkos.cpp | react-retry path in collisions_one reallocs d_nn_last_partner with d_plist when NEARCP | compile OK
- F-G01-4 / F-G02-1 | src/KOKKOS/collide_vss_kokkos.cpp | zero-volume cell: set d_error_flag and return at all 5 collision kernels (before rand state acquire) | compile OK
