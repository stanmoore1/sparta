# AB4 results (grid/misc computes)

Method note: inputs use `timestep 1e-12` + `create_particles ... twopass` so particles are identical in
CPU and Kokkos runs and essentially do not move; with `collide vss` the particles are sorted each step,
so `-k on t 4` (need_atomics=1, atomic_reduction=0 on CPU) selects the sorted per-cell kernels,
`t 1` / no-collide select the atomic kernels (t>1 no-collide = duplicated ScatterView path).
Kokkos and CPU outputs can then be compared value by value (dumps with %20.15g).

### F-G00-5 — sorted compute grid / pflux/grid kernels `return` instead of `continue` on species not in mixture
class: cpu-observable
positive control: N2+O2 box, 20k particles, 4^3 cells, collide vss, `compute grid all onlyO2 n u ke` and `pflux/grid all onlyO2 momxx momyy momxy` vs the O2 columns of the same computes with mixture `species`; t 4 (sorted) | A: sum of onlyO2 n = 42 vs compute count O2 = 10079; onlyO2 columns differ from species-O2 columns by rel 1.80 (grid), 1.23 (pflux), and from CPU by 1.80 | B: sum n = 10079, onlyO2 == species-O2 columns exactly, all columns bit-identical to CPU | REPRODUCED
negative control: (a) same sorted run, mixture `species` (no skipped species) columns: A vs B identical; (b) no collide, t 4 (dup atomic path): A vs B identical, both identical to CPU; (c) t 1: A, B, CPU identical
verdict: VERIFIED
artifacts: $S/ab/AB4/F-G00-5

### F-G15-1 — sonine/grid sorted kernel never tallies the mass column (moments shifted)
class: cpu-observable
positive control: N2+O2 box (20k, 4^3 cells, dt 1e-12), collide vss, `sonine/grid all air a x 3 b xy 2` + `sonine/grid all onlyO2 a y 2 b xx 2`, dump grid step 2, t 4 (sorted kernel) | A: max rel err vs CPU 1.99 (s1) / 1.92 (s2); signature exactly the column shift: A out[A1]=CPU_A2/CPU_A1 (988334 = -7.5206e12/-7.6094e6), out[A2]=CPU_A3/CPU_A1, out[A3]=CPU_B1/CPU_A1=169.5, last moment 0 | B: bit-identical to CPU | REPRODUCED
negative control: t 1 (atomic<0>, non-dup) with collide: A, B, CPU all identical
verdict: VERIFIED
artifacts: $S/ab/AB4/sonine (dump.grid.{A,B}t{1,4}c1, dump.grid.cpuc1)

### F-G00-7 — sonine/grid dup path normalizes vcom before contribute() (COM = raw mass-weighted sums)
class: cpu-observable
positive control: same input, no collide (unsorted) t 4 -> need_dup ScatterDuplicated path (also step 0 of the collide run) | A: max rel err vs CPU 1.96 (s1) / 2.00 (s2) at steps 0 and 2 | B: bit-identical to CPU | REPRODUCED
negative control: t 1 (non-dup) no collide: A, B, CPU identical
verdict: VERIFIED
artifacts: $S/ab/AB4/sonine (dump.grid.*c0)
