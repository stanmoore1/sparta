# Fix review C

DONE src/KOKKOS/kissfft_kokkos.h
DONE src/KOKKOS/fftdata_kokkos.h
DONE src/KOKKOS/fft2d_kokkos.cpp
DONE src/KOKKOS/fft3d_kokkos.cpp
DONE src/FFT/fft2d.cpp
DONE src/KOKKOS/pack2d_kokkos.h
DONE src/KOKKOS/pack3d_kokkos.h
DONE src/KOKKOS/remap3d_kokkos.cpp
DONE src/KOKKOS/kokkos.cpp
DONE src/KOKKOS/kokkos_type.h
DONE src/KOKKOS/comm_kokkos.cpp
DONE src/KOKKOS/CMakeLists.txt
DONE src/KOKKOS/Install.sh
DONE cmake/common/set/style_file_glob.cmake
DONE src/KOKKOS/grid_kokkos.h
DONE src/KOKKOS/grid_kokkos.cpp
DONE src/KOKKOS/surf_kokkos.cpp
DONE src/surf_custom.cpp
DONE src/surf_comm.cpp

## Issues
- [R-C-1] src/KOKKOS/grid_kokkos.cpp:118,133,180 | low | F-G22-2 only value-initializes the resize path; first allocation (cells/cinfo/sinfo == NULL) still goes through MemKK::realloc_kokkos which uses Kokkos::NoInit, so the initial arrays are not zeroed (CPU srealloc(NULL)+memset zeroes them). Not a regression. | if parity wanted, allocate first-time views with default init (e.g. tdual_cell_1d("grid:cells",maxcell)) or deep_copy 0 after realloc_kokkos
- [R-C-2] src/KOKKOS/kokkos_type.h:192-268 | info | Kokkos::SYCL / Kokkos::SYCLHostUSMSpace exist only in Kokkos >= 4.5 (HIP names >= 4.0); USE_EXTERNAL_KOKKOS has no minimum-version check, so an older external Kokkos with SYCL now fails to compile (bundled 5.2 is fine and needs the rename) | optionally find_package(Kokkos 4.5 REQUIRED CONFIG) or #if KOKKOS_VERSION guard
- [R-C-3] src/KOKKOS/fft2d_kokkos.cpp:197, fft3d_kokkos.cpp:199 | info (perf) | kiss_fft_functor allocates a fresh scratch view (<= data size since p_max <= length) on every FFT stage call; correct but adds an allocation per stage per call | optional: cache scratch in plan
- Verified OK: KISS scratch offsets (slice i*nscr, nscr=max(1,p_max), p <= p_max), recursion threading of d_scr/scr_offset, managed->unmanaged conversion on device, subview in KOKKOS_INLINE_FUNCTION; non-post_plan deep_copy of d_tmp->d_out; norm int->FFT_SCALAR; remap2d arg order (nqty,permute,memory,precision) matches signature; cufftDestroy/hipfftDestroy handle names match headers; pack2d/pack3d unpack index math matches CPU pack2d.h/pack3d.h; remap3d collective free matches (nsend||nrecv) alloc, value-init valid (no user ctor); CMake regex vs absolute glob paths; Install.sh $SED undefined -> sed; comm_kokkos EXACT host compress mirrors comm.cpp and Particle grow syncs device before resize; added CUSTOM_MASK device sync needed after host compress; id_find_child matches Grid::id_point_child; surf grow memset on host-modified side; spread_inverse_custom/spread_local2own in/out and idata fixes correct.

## STATUS: COMPLETE
