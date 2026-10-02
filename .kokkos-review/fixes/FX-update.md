# FX-update checkpoint
- F-G18-1 | src/KOKKOS/particle_kokkos.{h,cpp}, src/KOKKOS/update_kokkos.cpp | factored species2group build into ParticleKokkos::update_species2group() (called from wrap_kokkos) and call it in UpdateKokkos::setup() non-prewrap branch so it is rebuilt every run | compile OK
