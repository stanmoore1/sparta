# FX-emitface checkpoint

- F-G12-1 | src/KOKKOS/fix_emit_face_kokkos.cpp | subsonic_grid kernel: vstream correction guarded by `if (np && massrho_cell*soundspeed_cell > 0.0)` (CPU fix_emit_face.cpp:1109); emit/face/file and emit/surf already had it | compile OK
- F-G12-2 | src/KOKKOS/fix_emit_face_kokkos.cpp, fix_emit_face_file_kokkos.cpp, fix_emit_surf_kokkos.cpp | subsonic_sort: after a sort done on behalf of the fix, reset particle_kk->sorted_kk = 0 (plist_descending recorded before; d_plist stays valid for this fix's subsonic_grid). Checked readers: no code between start_of_step emit and move relies on sorted_kk=1 (other readers only re-sort or pick unsorted/atomic path, both correct); move resets sorted_kk=0 (update_kokkos.cpp:962) and collide/reorder re-sorts unconditionally (update_kokkos.cpp:499-500) | compile OK

## STATUS: COMPLETE
