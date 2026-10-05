# AB8 (fft) A/B results

Common setup: deterministic inputs (no particles, run 0, dump grid at step 0): compute A property/grid id (vector),
compute B property/grid xc yc vol (array), grid variable x = sin(7.3*cxlo+1.1)*cos(3.7*cylo)+cxlo*cylo^2+exp(czlo);
compute F fft/grid <args>. Reference = CPU compute fft/grid (same binary; CPU FFT=KISS, Kokkos FFT_KOKKOS=KISS in both builds).
Diff metric = max|kk-cpu| per column / max|cpu| over all columns. Inputs/scripts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/common (in.base, run.sh, cmp.py).

### F-G20-9 — KISS generic-radix butterfly shares one scratch per plan -> OpenMP/GPU race, wrong FFTs for prime factors >=7
class: race
positive control: fft of c_A only, t 4. 3D 14^3: A run1 0, run2 c1 4.8e-2 / c2 1.7e-1, run3 5.5e-2 / 5.4e-2; 2D 14x14 (3 runs): 1.3e-3, 0, 2.4e-3; 2D 98x98 (3 runs): 1.5e-2, 8.8e-4, 3.4e-1 | B: 0 (bit-identical to CPU) in all 9 runs. A with t 1 = 0 (race only) | REPRODUCED
negative control: 16^3 and 16x16 (radices 2/4 only), t 4: A = B = CPU bit-identical
verdict: VERIFIED
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/G20-9

### F-G00-8 / F-G19-2 — compute fft/grid/kk array/variable inputs read a stale d_ingrid (previous vector input) instead of their own data
class: cpu-observable
positive control: values "c_A c_B[2] v_x" (vector, array col, grid var), 16^3 3D t4, 16x16 2D t4, 14^3 3D t1; plus "c_A f_AV[2]" (fix ave/grid of c_B[*], run 2) | A: cols 1-2 correct, cols 3-6 wrong (rel diff 1.0/0.31; A's cols 3..6 == cols 1..2, i.e. FFT of c_A repeated); f_AV[2] path likewise wrong (1.0/0.31) | B: all columns bit-identical to CPU in every case | REPRODUCED
negative control: "c_B[2] v_x" (no preceding vector input), 3D+2D t4: A = B = CPU bit-identical
verdict: VERIFIED
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/G19-2

### F-G19-1 — sum yes: compute fft/grid/kk zeroes the whole output array each step, wiping the K-space (kx/ky/kz/kmag) columns
class: cpu-observable
positive control: "c_B[1] c_B[2] sum yes kx ky kz kmag" 16^3 3D and "... kx ky kmag" 16x16 2D, t 4 | A: all K columns exactly 0 (CPU max |kx|,|ky|,|kz| = 8, |kmag| = 13.86); summed FFT columns correct | B: all columns bit-identical to CPU | REPRODUCED
negative control: "c_B[1] c_B[2] kx yes kmag yes" (sum no), 3D+2D: A = B = CPU bit-identical
verdict: VERIFIED
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB8/G19-1
