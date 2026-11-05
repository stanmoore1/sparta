# FX-infra fixes

- F-G21-1 | src/KOKKOS/kokkos.cpp | error if ngpus <= 0 right after parsing "g" arg (before local_rank % ngpus) | compile OK
- F-G21-2 | src/KOKKOS/kokkos.cpp | added `if (iarg+2 > narg)` bounds check to "t"/"threads" branch | compile OK
- F-G21-4 | src/KOKKOS/kokkos_type.h | Kokkos::Experimental::{HIP,SYCL,HIPHostPinnedSpace,SYCLHostUSMSpace} -> Kokkos::{...} (lines 192,197,211,213,253,258,263,268); verified non-Experimental classes exist in bundled lib/kokkos (namespace Kokkos, HIP/Kokkos_HIP.hpp, SYCL/Kokkos_SYCL.hpp, *_Space.hpp); OpenMPTarget/Scatter* Experimental names left as-is | compile OK (OpenMP build: grid_kokkos.cpp, comm_kokkos.cpp, kokkos.cpp; HIP/SYCL branches not compiled in this build)
- F-G21-5 | src/KOKKOS/kokkos_type.h | t_plevel_1d/t_host_plevel_1d now typedef'd from tdual_plevel_1d | compile OK
- F-G21-7 | src/KOKKOS/comm_kokkos.cpp | under #ifdef SPARTA_KOKKOS_EXACT: sync(Host) PARTICLE(+CUSTOM), ascending test -> Particle::compress_migrate or compress_reactions (as CPU comm.cpp), then modify(Host) PARTICLE(+CUSTOM); non-EXACT path unchanged. Also added `if (ncustom) particle_kk->sync(Device,CUSTOM_MASK);` next to the existing sync(Device,PARTICLE_MASK) before the irregular exchange so the device custom-unpack kernel sees host-compacted custom data (no-op in non-EXACT builds) | compile OK (normal and with -DSPARTA_KOKKOS_EXACT -fsyntax-only)
- F-G21-8 | src/KOKKOS/CMakeLists.txt, cmake/common/set/style_file_glob.cmake | replaced the three unanchored ".*fft|pack|remap.*kokkos.*" filters with one basename-anchored `EXCLUDE REGEX "/[^/]*(fft|pack|remap)[^/]*kokkos[^/]*$"` | cmake -P test OK (mock .../packages/... and plain paths keep grid_kokkos.*, drop fft/pack/remap; real src/KOKKOS: 192 files -> 177, exactly the 14 FFT-family files + kokkos_base_fft.h removed)
- F-G21-9 | src/KOKKOS/CMakeLists.txt, cmake/common/set/style_file_glob.cmake | `list(REMOVE_ITEM style_files kokkos_base_fft.h)` -> `list(FILTER <SPARTA_PKG_KOKKOS_SRC_FILES|style_files> EXCLUDE REGEX "/kokkos_base_fft\\.h$")` | cmake -P test OK (kokkos_base_fft.h removed from both lists)
- F-G21-12 | src/KOKKOS/Install.sh | undefined `$SED` -> `sed` (4 lines in mode-0 Makefile.package cleanup); line 30 `= 1` -> `!= 0` | bash -n OK, sed expressions tested on a sample Makefile.package
- F-G21-6 | DEFERRED | PLAUSIBLE, memory-peak only, multi-file API change in memory_kokkos.h (not owned): change destroy_kokkos(TYPE data,...) at memory_kokkos.h:107,301 to `TYPE &data` and re-verify ParticleKokkos custom-array slot paths
- F-G21-10 | DEFERRED | PLAUSIBLE, unreproduced: proposed patch in kokkos_type.h after Kokkos includes `#if !defined(KOKKOS_ENABLE_IMPL_VIEW_LEGACY)` / `#error "The KOKKOS package requires Kokkos built with Kokkos_ENABLE_IMPL_VIEW_LEGACY=ON"` / `#endif`, or `kokkos_check(OPTIONS IMPL_VIEW_LEGACY)` after find_package(Kokkos) in src/KOKKOS/CMakeLists.txt:68
- F-G21-3, F-G21-11 | REFUTED, skipped

## STATUS: COMPLETE
