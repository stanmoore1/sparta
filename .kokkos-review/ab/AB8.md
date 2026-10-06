# AB8 (fft) A/B results

Common setup: deterministic inputs (no particles, run 0, dump grid at step 0): compute A property/grid id (vector),
compute B property/grid xc yc vol (array), grid variable x = sin(7.3*cxlo+1.1)*cos(3.7*cylo)+cxlo*cylo^2+exp(czlo);
compute F fft/grid <args>. Reference = CPU compute fft/grid (same binary; CPU FFT=KISS, Kokkos FFT_KOKKOS=KISS in both builds).
Diff metric = max|kk-cpu| per column / max|cpu| over all columns. Inputs/scripts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/common (in.base, run.sh, cmp.py).

### F-G20-9 — KISS generic-radix butterfly shares one scratch per plan -> OpenMP/GPU race, wrong FFTs for prime factors >=7
class: race
positive control: fft of c_A only, t 4. 3D 14^3: A run1 0, run2 c1 4.8e-2 / c2 1.7e-1, run3 5.5e-2 / 5.4e-2; 2D 14x14 (3 runs): 1.3e-3, 0, 2.4e-3; 2D 98x98 (3 runs): 1.5e-2, 8.8e-4, 3.4e-1 | B: 0 (bit-identical to CPU) in all 9 runs. A with t 1 = 0 (race only) | REPRODUCED
negative control: 16^3 and 16x16 (radices 2/4 only), t 4: A = B = CPU bit-identical
necessary: yes - A t4 wrong by up to 34% of max|F| with prime factors 7/11/13 (2D+3D), t1 correct -> race
complete: B bit-identical to CPU (3 repeats each, t4) on: 14^3, 14x22x26 (radices 7,11,13 on fast/mid/slow axes, all three KISS call sites in fft3d), 14x14, 22x14, 98x98, 154x26 (2D fast+slow sites), sum mode, 3 inputs per compute; direct-API harness (fftharness/, FFT3dKokkos/FFT2dKokkos vs CPU FFT3d/FFT2d) forward AND backward, permute 0/1/2, scaled 0/1, t1/t4, 14x22x26/16x12x10/22x14/16x10: B 0 diff in all 40 cases (A: e.g. 14x22x26 t4 fwd 3.8e-2). Not exercised: fft_*_1d_only_kokkos (timing1d only, no caller in SPARTA); multi-proc (MPI builds not finished at time of writing, see later note if added)
np>1 addendum (MPI+bounds-check builds, fftharness/fftmpi_{A,B}, sweep_mpi_{A,B}_t2.txt): np 2/4 x t 2, 3D 16^3 + 14x22x26 and 2D 16x16 + 22x14, in slab/fast-split, out same/target/slab, all permutes, fwd+bwd | A: 14x22x26 wrong in 16 configs (np4 up to FWD 0.64 / BWD 0.97 of max), 22x14 np4 FWD 0.50 | B: 240/240 cases bit-identical to CPU (scaled 0/1), no bounds aborts
verdict: NECESSARY, COMPLETENESS-PARTIAL (only the 1d_only timing path, which has no caller, not executed; np 1/2/4 x t1/t2/t4 all verified)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/G20-9, /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/fftharness

### F-G00-8 / F-G19-2 — compute fft/grid/kk array/variable inputs read a stale d_ingrid (previous vector input) instead of their own data
class: cpu-observable
positive control: values "c_A c_B[2] v_x" (vector, array col, grid var), 16^3 3D t4, 16x16 2D t4, 14^3 3D t1; plus "c_A f_AV[2]" (fix ave/grid of c_B[*], run 2) | A: cols 1-2 correct, cols 3-6 wrong (rel diff 1.0/0.31; A's cols 3..6 == cols 1..2, i.e. FFT of c_A repeated); f_AV[2] path likewise wrong (1.0/0.31) | B: all columns bit-identical to CPU in every case | REPRODUCED
negative control: "c_B[2] v_x" (no preceding vector input), 3D+2D t4: A = B = CPU bit-identical
necessary: yes - A columns for c_B[2], v_x, f_AV[2] after a vector input are the vector's FFT (100%% wrong)
complete: B correct for all three branches the fix touches (compute array column, fix array column, grid variable) after a vector input, 2D and 3D, t1/t4; compute-vector and fix-vector branches assign d_ingrid directly (unchanged, correct in A and B). No other d_ingrid shadowing remains (grep: only l_ingrid locals in B)
verdict: NECESSARY+COMPLETE
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/G19-2

### F-G19-1 — sum yes: compute fft/grid/kk zeroes the whole output array each step, wiping the K-space (kx/ky/kz/kmag) columns
class: cpu-observable
positive control: "c_B[1] c_B[2] sum yes kx ky kz kmag" 16^3 3D and "... kx ky kmag" 16x16 2D, t 4 | A: all K columns exactly 0 (CPU max |kx|,|ky|,|kz| = 8, |kmag| = 13.86); summed FFT columns correct | B: all columns bit-identical to CPU | REPRODUCED
negative control: "c_B[1] c_B[2] kx yes kmag yes" (sum no), 3D+2D: A = B = CPU bit-identical
necessary: yes - A K-space columns all 0 with sum yes
complete: B correct for 3D (kx,ky,kz,kmag) and 2D (kx,ky,kmag) with sum yes; ncol==1 sum branch (sum yes + conjugate yes, no K cols) zeroes the whole vector, which is correct as it has no K columns (covered in 154x26 "sum yes kmag yes" run of G20-9 completeness: B = CPU)
verdict: NECESSARY+COMPLETE
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/G19-1

### F-G19-3 — compute fft/grid/kk invoked before first run (prewrap) falls back to base host compute_per_grid with NULL irregular/fft objects
class: cpu-observable
positive control: library driver (drv.cpp linked to A/B libs) sparta_extract_compute(F,2,2) right after "compute F fft/grid c_A c_B[2]", no run | A kk t4: SIGSEGV, gdb: Irregular::exchange_uniform(this=0x0) from ComputeFFTGrid::compute_per_grid (NULL irregular1 = the reported bug) | B kk: clean error "Cannot (yet) invoke compute fft/grid/kk before the first run", extract returns NULL | REPRODUCED
  note: the CPU style also segfaults here (A and B, different cause: input compute property/grid vector not yet allocated before init -> sendbuf=NULL), so this is a pre-existing general pre-run limitation, not a kk regression. adapt_grid value c_F trigger ends in "requires uniform one-level grid" error for CPU/A/B alike (maxlevel check precedes) -> not a usable trigger.
negative control: same driver with "run 0" before extract: B kk = CPU (F[0]=8390656,0,2048,0, sumsq 9.3859360358e13); A kk runs (cols 3-4 wrong only due to F-G19-2)
necessary: yes (A NULL-deref crash on the prewrap path)
complete: yes - the prewrap branch is the only host fallback in compute_per_grid; B errors cleanly instead of crashing. (Fix is an error, not a feature: pre-run extraction is still unsupported by design.)
verdict: NECESSARY+COMPLETE
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/G19-3

Harnesses used below:
- fftharness/fftdrv_{A,B}: links the A/B libs, opens SPARTA with -k on t N -sf kk, builds FFT3dKokkos/FFT2dKokkos and CPU FFT3d/FFT2d with identical layouts (in slab/fast-split, out same/target/slab), scaled 0/1, permute 0/1/2; runs FORWARD and BACKWARD on the same deterministic complex input; reports max|kk-cpu|/max|cpu|. Results np=1: fftharness/sweep_np1.txt.
- packunit/packtest_{A,B}: header-only unit test of every PackKokkos2d/3d unpack functor (with nonzero buf/data offsets and padded strides from the remap plan formulas) vs the CPU pack2d.h/pack3d.h routine; compares the whole data array and checks buf is not written.

### F-G00-21 / F-G20-1 — norm_functor stores the backward-FFT normalisation as int -> scaled backward FFT outputs all 0
class: unreachable (compute fft/grid/kk only does unscaled forward FFTs)
positive control: fftharness scaled=1, BACKWARD, 3D 16^3/14x22x26/16x12x10 and 2D 16x16/22x14/16x10, permute 0/1/2, t1/t4 | A: BWD maxdiff/max = 1.000 in every scaled case (output all zero) | B: 0 (bit-identical to CPU) | REPRODUCED
negative control: scaled=0 BACKWARD and all FORWARD, permute 0, 16^3/16x16: A = B = CPU (0 diff)
necessary: yes (via direct API; unreachable from compute fft/grid/kk)
complete: both norm_functor copies (fft2d and fft3d) fixed and verified for every permute/out-layout; the 1d_only (timing1d) norm call site uses the same functor type, not separately executed
verdict: NECESSARY+COMPLETE
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/fftharness

### F-G19-5 / F-G20-2 — KISS slow-axis FFT out-of-place: when post_plan is null (permute=last, out == final layout) the result never reaches out
class: unreachable (compute uses permute=0)
positive control: fftharness np=1, permute=1 (2D) / permute=2 (3D) with out = final FFT layout (post_plan == nullptr) | A: FWD maxdiff/max 0.95-1.04 and BWD 0.94-1.0 (2D 16x16, 22x14, 16x10; 3D 16^3, 14x22x26, 16x12x10) | B: 0 diff (bit-identical to CPU) in all, scaled 0/1, t1/t4 | REPRODUCED
negative control: permute 0 / (3D) permute 1 (post_plan exists): A = B = CPU (0 diff, unscaled)
necessary: yes (via direct API)
np>1 addendum (sweep_mpi_{A,B}_t1.txt, np 2/4, t1): A: 3D permute=2 out=target (and in=fast out=same) FWD 0.95/1.01, BWD 0.94-1.0; 2D permute=1 out same/target/slab FWD 0.96-1.0 (16x16, 22x14) | B: all 0 diff, bounds check clean
complete: 2D and 3D sites both verified at np 1/2/4, t1/t2/t4, scaled 0/1, both sizes incl. mixed radix
verdict: NECESSARY+COMPLETE (via direct API; unreachable from compute fft/grid/kk)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/fftharness

### F-G19-9 — unpack_2d_functor wrote data->buf with swapped indices (direction reversed)
class: unreachable (needs a permute=0 2D remap: pre_plan with fast-axis-split input, or permute=1 output not in second layout; neither used by compute fft/grid/kk)
positive control: packunit unpack_2d (nqty=2, nstride=nqty*isize padded, offsets 3/5) | A: data 24 entries differ from CPU unpack_2d and 24 buf entries overwritten (with unpadded buffers A aborts "double free or corruption (out)" = OOB heap write) | B: identical to CPU, buf untouched | REPRODUCED
negative control: unpack_2d_permute_1/_2/_n (same file): A = B = CPU
necessary: yes (unit level)
np>1 end-to-end (MPI + Kokkos DEBUG_BOUNDS_CHECK, sweep_mpi_{A,B}_t1.txt/_t2.txt): FFT2dKokkos with fast-axis-split input (pre_plan -> unpack_2d), 16x16 and 22x14, np 2/4, out same/target/slab, permute 0/1, scaled 0/1 | A: 48/48 abort "Kokkos::View ERROR: out of bounds access ... indices [330] but extents [308]" (np2 22x14; np4 [146] vs [140]; 16x16 [256] vs [256]) | B: 48/48 bit-identical to CPU FFT2d fwd+bwd, also at t2
complete: yes - unit level + end-to-end at np 2/4, all 2D out layouts/permutes
verdict: NECESSARY+COMPLETE (via direct API; unreachable from compute fft/grid/kk)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/packunit

### F-G20-6 — unpack_3d_permute1_n used in = instart + nqty*fast*nstride_plane (nstride already nqty-scaled)
class: unreachable (FFT remaps always nqty=2 -> permute1_2)
positive control: packunit unpack_3d_permute1_n nqty=3 and 4 | A: 96/765 and 128/885 entries differ from CPU (writes to wrong/out-of-box locations) | B: identical to CPU | REPRODUCED
negative control: unpack_3d_permute1_1 / permute1_2: A = B = CPU
necessary: yes (unit level)
complete: yes - sibling functors checked in the same run: all 13 unpack variants (2d: plain/perm_1/_2/_n; 3d: plain, perm1_1/_2/_n, perm2_1/_2/_n) match CPU in B; no other nqty*fast*stride pattern remains (grep)
verdict: NECESSARY+COMPLETE
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/packunit

### F-G20-7 — unpack_3d_permute2_n used in = instart + nqty*fast*nstride_line
class: unreachable (nqty=2 only)
positive control: packunit unpack_3d_permute2_n nqty=3 and 4 | A: 78/765 and 128/885 entries differ from CPU | B: identical to CPU | REPRODUCED
negative control: unpack_3d_permute2_1 / permute2_2: A = B = CPU
necessary: yes (unit level)
complete: yes (see F-G20-6 sibling sweep)
verdict: NECESSARY+COMPLETE
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/packunit

### F-G19-4 (CPU+kk) — fft2d remap plans passed (FFT_PRECISION, ..., 2) instead of (nqty=2, ..., FFT_PRECISION): FFT_SINGLE 2D moves only half of each complex value
class: build-system (FFT_SINGLE builds only)
positive control: FFT_SINGLE harness (G19-4/drv2s_{A,B}: fft2d_kokkos/remap2d_kokkos/fft2d/remap2d/fft2d_wrap compiled with -DFFT_SINGLE from A/B sources, linked to the A/B libs), forward 2D vs naive double DFT, 16x16 / 14x10 / 8x6 | A: Kokkos maxerr/max 1.02 / 1.01 / 0.98, CPU 1.02 / 1.09 / 0.94 (garbage) | B: Kokkos 3.4e-8 / 6.5e-8 / 5.0e-8 (float-exact); CPU: plan creation FAILS - remap_2d_create_plan prints "Single precision not supported" and returns NULL (src/FFT/remap2d.cpp:152 rejects precision==1, and its pack/unpack pointers are NULL for precision 1), FFT2d errors "Could not create 2d FFT plan" | REPRODUCED
negative control: 3D FFT_SINGLE (drv3s, 8x6x10, fft3d already used nqty=2,...,FFT_PRECISION): A = B, Kokkos and CPU both 8.9e-8; double-precision builds: FFT_PRECISION==2 so the arg swap is a no-op (all double runs in this file: B = CPU bit-identical)
necessary: yes (A wrong on both Kokkos and CPU in FFT_SINGLE 2D)
complete: Kokkos part yes. CPU part NO: B's CPU FFT2d now passes precision=1 to CPU remap_2d_create_plan, which refuses single precision, so CPU compute fft/grid in a 2D FFT_SINGLE build turns from silently wrong into a hard error at plan creation (CPU remap3d ignores precision, CPU remap2d does not). Making CPU 2D single work would need remap2d.cpp to ignore precision like remap3d (or the CPU call to keep precision=2 with nqty=2).
verdict: INCOMPLETE (CPU 2D FFT_SINGLE: "Single precision not supported" -> "Could not create 2d FFT plan"; Kokkos part NECESSARY+COMPLETE)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/G19-4 (results.txt)

### F-G20-8 — CUDA/SYCL branches of fftdata_kokkos.h did not #undef FFT_KOKKOS_NVPL
class: build-system
positive control: preprocessor probe of the macro block of fftdata_kokkos.h (A vs B) with -DSPARTA_KOKKOS -DKOKKOS_ENABLE_{CUDA,SYCL} -DFFT_KOKKOS_NVPL | A: FFT_KOKKOS_NVPL and FFT_KOKKOS_KISS both defined, LIB="NVPL FFT" (data-type chain picks fftw_complex -> the KISS .re/.im code cannot compile); CUDA+NVPL+CUFFT: NVPL still defined | B: NVPL undefined, KISS only, LIB="KISS FFT"; CUDA+NVPL+CUFFT -> cuFFT only | REPRODUCED (preprocessor level)
negative control: HIP+NVPL (already undef'd in A): A = B KISS; CUDA without NVPL: A = B KISS; host build +NVPL: A = B "NVPL FFT" (NVPL still allowed on host)
necessary: yes at preprocessor level (no CUDA/SYCL toolchain here to show the compile error itself)
complete: yes - all three device branches (CUDA, HIP, SYCL) now undef NVPL; MKL_GPU/CUFFT/HIPFFT/KISS selection unchanged
verdict: NECESSARY+COMPLETE (preprocessor-level; real CUDA/SYCL compile not possible here)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/G20-8

### F-G19-7 — destroy_plan called FFTW cleanup_threads() unconditionally (invalidates other live FFTW plans)
class: build-system (FFT_KOKKOS_FFTW3 + FFT_KOKKOS_FFTW_THREADS only)
positive control: FFTW3+threads harness (G19-7/drvw_{A,B}: fft2d/3d_kokkos + remap compiled with -DFFT_KOKKOS_FFTW3 -DFFT_KOKKOS_FFTW_THREADS from A/B sources, libfftw3 3.3.10 + libfftw3_threads, t 4): keep a 32^3 FFT3dKokkos, create+run+delete a second FFT3dKokkos (or a FFT2dKokkos), then rerun the surviving plan 3x | A: gdb confirms fftw_cleanup_threads() is called from fft_3d_destroy_plan_kokkos while the other plan is live and the FFTW worker threads exit; but the surviving plan's reruns are bit-identical to its pre-delete result and valgrind memcheck reports 0 errors | B: cleanup_threads never called; reruns identical; valgrind 0 errors | NOT REPRODUCED (no observable misbehaviour with this FFTW version; per FFTW docs the behaviour is undefined)
negative control: same runs - A and B outputs identical
necessary: not shown - the call is real and against the FFTW API contract, but FFTW 3.3.10 tolerates it here
complete: B removes the call from both fft2d_kokkos and fft3d_kokkos destroy (grep: only comments remain); init_threads/plan_with_nthreads unchanged
verdict: NOT-SHOWN-NECESSARY (undefined behaviour per FFTW docs, benign with FFTW 3.3.10 on this machine; fix removes the call consistently)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/G19-7 (vg_A.txt, vg_B.txt)

### F-G19-6 / F-G20-3 — cuFFT/hipFFT plans never destroyed in fft_2d/3d_destroy_plan_kokkos (leak per FFT object)
class: gpu-only
positive control: host mock (G19-6/): fft2d/3d_kokkos.cpp + remap from A/B compiled with -DFFT_KOKKOS_CUFFT or -DFFT_KOKKOS_HIPFFT against stub cufft.h / hipfft/hipfft.h that count PlanMany/Destroy (fftdata_kokkos.h force-included with only the "Must enable CUDA/HIP" #error lines removed); 10x create+delete of FFT3dKokkos(8^3) and FFT2dKokkos(8x8) | A: CUFFT and HIPFFT: created=50 destroyed=0 live=50 | B: created=50 destroyed=50 live=0 | REPRODUCED (mock)
negative control: KISS builds: destroy path unchanged (all KISS runs in this file A/B consistent; no new failures)
necessary: yes (mock shows every cuFFT/hipFFT plan leaks in A; real GPU not available)
complete: yes - 2D (fast, slow) and 3D (fast, mid, slow) plans for both cuFFT and hipFFT all destroyed (destroy count == create count)
verdict: NECESSARY+COMPLETE (host mock with stub libraries; real GPU run not possible here)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/G19-6 (results.txt)

### R-C-3 — kiss_fft_functor allocated a fresh scratch view on every FFT stage call (perf follow-up to F-G20-9)
class: cpu-observable (perf)
A here = bdbc461a fft2d/3d_kokkos + kissfft_kokkos (post F-G20-9, pre-R-C-3) compiled from git show into R-C-3/I_src (e071055f has no per-call allocation, it has the race instead); B = HEAD
positive control: drvp harness, 3D FFT3dKokkos t4, 20 forward FFTs, Kokkos Tools allocate callback counting "kissscratch" allocations during compute | I: 60 kissscratch allocations (3 per FFT, one per stage), 120 Kokkos allocations total | B: 0 kissscratch allocations, 60 total (only the per-stage d_tmp remains) | REPRODUCED (allocation count). Wall time not meaningful (load avg 15-20 from the concurrent MPI builds): I 4.63/4.45 s vs B 4.52/4.46 s
negative control: results bit-identical to CPU FFT3d for I and B (14^3, 16^3, also 14x22x26 for I); B correctness otherwise covered by F-G20-9 sweeps (40 harness cases + compute runs incl. radices 7/11/13, fwd/bwd, all permutes)
necessary: yes as a perf change (per-call allocations eliminated); no correctness bug involved
complete: all KISS call sites in fft2d/fft3d (incl. 1d_only) use plan->d_kissscr (grep); scratch sized as max over all fwd/bwd stage cfgs - covered by mixed-radix 14x22x26 fwd+bwd runs (B = CPU)
verdict: NECESSARY+COMPLETE (perf; allocation-count evidence, timing inconclusive on loaded machine)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/R-C-3 (results.txt)

### F-G20-5 — remap3d_kokkos collective destroy freed send_* only if nsend and recv_* only if nrecv, but create allocates both whenever (nsend||nrecv) -> leak on send-only/receive-only ranks; plan not value-initialised
class: mpi (collective remap path; unreachable from compute fft/grid/kk which forces collective=0)
positive control: fftharness/fftmpi_{A,B} (MPI + bounds-check builds) FFT3dKokkos 8^3 with usecollective=1 under valgrind --leak-check=full, np 2 and 4: in=rank0 out=slab (ranks>0 receive-only in pre_plan), in=slab out=rank0 (send-only in post_plan), in=rank0 out=rank0 (both) | A: definitely-lost records from remap_3d_create_plan_kokkos: receive-only -> send_offset/send_size/packplan (remap3d_kokkos.cpp:605/606/614), np2 3 recs 16 B, np4 9 recs 80 B; send-only -> recv_offset/recv_size/unpackplan (643/644/652), np2 16 B, np4 9 recs 96 B; both: np4 18 recs 192 B (per FFT object, grows with each plan) | B: 0 remap3d_kokkos leak records in all 6 cases, no invalid free | REPRODUCED
negative control: in=slab out=target, collective, np 2/4: A and B 0 leak records; all FFT results A=B=CPU (0 diff). Collective sweep (sweep_mpi_{A,B}_coll1.txt, 120 configs np2/4, 2D+3D, all layouts/permutes): B 120/120 bit-identical to CPU fwd+bwd, bounds check clean (A's 44 failures are only the F-G19-5/F-G19-9 cases)
necessary: yes for the leak part (A leaks the send or recv arrays on every send-only/receive-only rank in each collective plan). The value-initialisation part (idle rank with nsend=nrecv=0 -> free of uninitialised pointers) cannot be exercised: a rank idle in a plan (3D 2x2x1, in=rank0, np 4, collective) segfaults at create time in BOTH A and B (invalid write remap3d_kokkos.cpp:689, the store-send loop writes plan->send_size[i] which was never allocated) = the DEFERRED remainder of F-G20-4, so destroy is never reached
complete: leak part complete (send-only, receive-only, both; np 2/4). Idle-rank part not testable because of the deferred F-G20-4 crash. Sibling found (not fixed, both Kokkos and CPU src/FFT/remap3d.cpp:609-611): the collective create calls MPI_Comm_group + MPI_Group_incl and never MPI_Group_free on orig_group/new_group -> 144 B definitely lost per plan (remap3d_kokkos.cpp:791) in A AND B (and CPU remap_3d_create_plan); CPU FFT3d handles the idle-rank case without crashing (Kokkos-only crash)
verdict: NECESSARY, COMPLETENESS-PARTIAL (idle-rank/value-init half unreachable: create crashes first, deferred F-G20-4; MPI_Group leak sibling unfixed)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/G20-5 (leak_matrix.txt, leaks.py, vg_*), fftharness (fftdrv_mpi.cpp now has out=rank0 layout)

## Notes
- MPI harness sweeps (np 2/4): fftharness/sweep_mpi_{A,B}_t1.txt (scaled 0/1, t1), _t2.txt (t2), _coll1.txt (collective). B: 600/600 cases bit-identical to CPU FFT2d/FFT3d with Kokkos bounds checking on.
- Unfixed siblings/open items found: (1) remap3d_kokkos collective create crashes for a rank idle in a plan (remap3d_kokkos.cpp:689 writes unallocated send_size; = deferred F-G20-4 remainder; Kokkos only, CPU fine); (2) missing MPI_Group_free in collective create (remap3d_kokkos.cpp:789-793 and src/FFT/remap3d.cpp:609-613), 144 B/plan leak; (3) F-G19-4 CPU remap2d rejects FFT_SINGLE (see F-G19-4 entry). All unreachable from compute fft/grid/kk in a default double-precision build (it uses collective=0).

## STATUS: COMPLETE
