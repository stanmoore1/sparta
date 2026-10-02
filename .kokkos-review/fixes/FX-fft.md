# FX-fft fixer checkpoint

- F-G20-9 | kissfft_kokkos.h, fft2d_kokkos.cpp, fft3d_kokkos.cpp | removed shared st.d_scratch, added st.p_max; scratch view+offset threaded through kiss_fft_kokkos->kiss_fft_stride->kf_work->kf_bfly_generic; kiss_fft_functor takes ntransform (=total/length) and allocates a per-call scratch of ntransform*max(1,p_max), work item i uses slice i*p_max; all 10 call sites updated (incl. 1d_only) | compile OK; standalone OpenMP test (8 thr, 20000 transforms, len 7/14/49/64/30) matches naive DFT to 1e-12
- F-G00-8 / F-G19-2 | compute_fft_grid_kokkos.cpp | compute/fix array-column branches now assign member d_ingrid = k_ingrid.view_device() (lambda uses local alias l_ingrid); VARIABLE branch sets d_ingrid = k_ingrid.view_device() after sync | compile OK
- F-G19-1 | compute_fft_grid_kokkos.cpp | sumflag zeroing of d_array_grid limited to columns [startcol,ncol) via subview, preserving K-space columns (matches CPU) | compile OK
- F-G19-3 | compute_fft_grid_kokkos.cpp | prewrap path now error->all("Cannot (yet) invoke compute fft/grid/kk before the first run") instead of calling unset-up base ComputeFFTGrid::compute_per_grid | compile OK
- F-G19-9 | pack2d_kokkos.h | unpack_2d_functor: `d_data[data_offset + in] = d_buf[buf_offset + out];` (direction/indices fixed, matches CPU unpack_2d) | compile OK
- F-G20-1 / F-G00-21 | fft2d_kokkos.cpp, fft3d_kokkos.cpp | norm_functor member and ctor arg int norm -> FFT_SCALAR norm | compile OK
- F-G20-6 | pack3d_kokkos.h | unpack_3d_permute1_n: in = instart + fast*nstride_plane (drop extra nqty) | compile OK
- F-G20-7 | pack3d_kokkos.h | unpack_3d_permute2_n: in = instart + fast*nstride_line (drop extra nqty) | compile OK
- F-G19-4 | fft2d_kokkos.cpp, src/FFT/fft2d.cpp | remap plan args reordered to nqty=2, precision=FFT_PRECISION (Kokkos mid/post plans; CPU pre/mid/post plans), matching fft3d argument order | compile OK
- F-G20-8 | fftdata_kokkos.h | CUDA and SYCL branches now #undef FFT_KOKKOS_NVPL (mirrors HIP branch) | compile OK
- F-G19-6 / F-G20-3 | fft2d_kokkos.cpp, fft3d_kokkos.cpp | destroy_plan: added CUFFT (cufftDestroy) and HIPFFT (hipfftDestroy) branches for plan_fast/(plan_mid)/plan_slow | compile OK (KISS build; GPU branches not compiled here)
- F-G19-7 | fft2d_kokkos.cpp, fft3d_kokkos.cpp | removed FFTW_API(cleanup_threads)() from destroy_plan (would invalidate other live plans; CPU never calls it) | compile OK (FFTW_THREADS branch not compiled here)
- F-G20-5 | remap3d_kokkos.cpp | plan value-initialised (new ...<DeviceType>()); collective destroy frees send+recv arrays together under (nsend || nrecv), matching create | compile OK
- F-G20-4 | PARTIAL/DEFERRED | value-initialisation applied (shared with F-G20-5); remaining part (wrap store-send/recv loops, d_sendbuf alloc and self block at remap3d_kokkos.cpp ~657-772 in if (nsend||nrecv) {...} else plan->self=0, guard selfcommringloc>=0) is a block restructuring of the collective-only path, unreachable since compute fft/grid/kk forces collective_flag=0; left for owner review
