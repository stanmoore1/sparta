# FX-fft fixer checkpoint

- F-G20-9 | kissfft_kokkos.h, fft2d_kokkos.cpp, fft3d_kokkos.cpp | removed shared st.d_scratch, added st.p_max; scratch view+offset threaded through kiss_fft_kokkos->kiss_fft_stride->kf_work->kf_bfly_generic; kiss_fft_functor takes ntransform (=total/length) and allocates a per-call scratch of ntransform*max(1,p_max), work item i uses slice i*p_max; all 10 call sites updated (incl. 1d_only) | compile OK; standalone OpenMP test (8 thr, 20000 transforms, len 7/14/49/64/30) matches naive DFT to 1e-12
- F-G00-8 / F-G19-2 | compute_fft_grid_kokkos.cpp | compute/fix array-column branches now assign member d_ingrid = k_ingrid.view_device() (lambda uses local alias l_ingrid); VARIABLE branch sets d_ingrid = k_ingrid.view_device() after sync | compile OK
- F-G19-1 | compute_fft_grid_kokkos.cpp | sumflag zeroing of d_array_grid limited to columns [startcol,ncol) via subview, preserving K-space columns (matches CPU) | compile OK
- F-G19-3 | compute_fft_grid_kokkos.cpp | prewrap path now error->all("Cannot (yet) invoke compute fft/grid/kk before the first run") instead of calling unset-up base ComputeFFTGrid::compute_per_grid | compile OK
