# FX-surfreact checkpoint
- F-G10-1 | src/KOKKOS/surf_react_global_kokkos.cpp | allocate d_nsingle_backup/d_tally_single_backup in ctor on live object; backup() only deep_copies | compile OK
- F-G10-2 | src/KOKKOS/surf_react_prob_kokkos.cpp | same as F-G10-1 for prob | compile OK
- F-G10-3 | src/KOKKOS/surf_react_adsorb_kokkos.{h,cpp} | factored per-slot view allocation into alloc_state_kokkos(force); grid_changed() calls alloc_state_kokkos(1) when mode==SURF && distributed (resizes nstate_ + all device per-slot views, zeroes deltas/mark, resets state_synced_to_device) | compile OK
- F-G10-4 | src/KOKKOS/surf_react_adsorb_kokkos.cpp | init_reactions_gs_kokkos() calls alloc_state_kokkos(0): k_species_delta/k_mark only re-created+zeroed on size change, so pending GS deltas survive across runs like CPU | compile OK
- F-G10-5 | src/KOKKOS/surf_react_adsorb_kokkos.cpp | backup views allocated on live object (nsingle/tally_single in ctor, species_delta/mark backups in alloc_state_kokkos); EXACT: random_backup in ctor, cmodel_random_backup[idx] in init_cmodels_kokkos; backup() only copies | compile OK (also -DSPARTA_KOKKOS_EXACT)
- F-G10-6 | src/KOKKOS/surf_react_adsorb_kokkos.cpp | pre_react() also sets state_synced_to_device=1 on live object surf->sr[this_index] | compile OK
- F-G00-18 | src/KOKKOS/surf_react_adsorb_kokkos.h | scatter_cmodel SPECULAR honors noslip flag (d_cm{ip,jp}_flags(j,0)) -> negate3, matching SurfCollideSpecular::wrapper; self-verified (no verifier verdict; G08 note only covers unused wrapper_kokkos): CPU adsorb parses specular cmodel with nflags=1 and wrapper applies noslip | compile OK (adsorb + all surf_collide_*_kokkos.cpp + update_kokkos.cpp)
## STATUS: COMPLETE
