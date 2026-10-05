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
verdict: NECESSARY, COMPLETENESS-PARTIAL (1d_only timing path unreachable; np>1 pending)
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
complete: 2D and 3D sites both verified; np>1 post-null case pending MPI build (see later note if added)
verdict: NECESSARY, COMPLETENESS-PARTIAL (np>1 not yet run)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/fftharness

### F-G19-9 — unpack_2d_functor wrote data->buf with swapped indices (direction reversed)
class: unreachable (needs a permute=0 2D remap: pre_plan with fast-axis-split input, or permute=1 output not in second layout; neither used by compute fft/grid/kk)
positive control: packunit unpack_2d (nqty=2, nstride=nqty*isize padded, offsets 3/5) | A: data 24 entries differ from CPU unpack_2d and 24 buf entries overwritten (with unpadded buffers A aborts "double free or corruption (out)" = OOB heap write) | B: identical to CPU, buf untouched | REPRODUCED
negative control: unpack_2d_permute_1/_2/_n (same file): A = B = CPU
necessary: yes (unit level)
complete: B unpack_2d matches CPU at unit level; end-to-end FFT2dKokkos path through it requires np>1 (pending MPI build)
verdict: NECESSARY, COMPLETENESS-PARTIAL (no end-to-end np>1 run yet)
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
