# FX-update checkpoint
- F-G18-1 | src/KOKKOS/particle_kokkos.{h,cpp}, src/KOKKOS/update_kokkos.cpp | factored species2group build into ParticleKokkos::update_species2group() (called from wrap_kokkos) and call it in UpdateKokkos::setup() non-prewrap branch so it is rebuilt every run | compile OK
- F-G05-1 | src/KOKKOS/update_kokkos.cpp | added local use_reduce in move() mirroring kernel dispatch (Serial DeviceType never uses parallel_reduce); used for both counter zeroing and read-back | compile OK
- F-G06-1 | src/KOKKOS/update_kokkos.{h,cpp} | added tmp_compute_surf_{coll,react}_tally_kk members (FIXED_LISTS) and re-pad unused slist_active_{coll,react}_tally_copy slots | compile OK (also with -DSPARTA_KOKKOS_FIXED_LISTS)
