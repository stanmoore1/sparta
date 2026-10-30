# V3R3-1 fix verification (commit c99e6089)

Fix: compute gas/reaction/grid (CPU + kk) rejects a re-issued react only when react->nlist grows
past the count at definition (was: any change).  Deck: $F/V3-R3/t/grg.  B_opt rebuilt at HEAD+fix.

| case | A_opt | B_opt (fixed) |
|---|---|---|
| in.grg: 136 reactions -> re-issue with 1 reaction (fewer), mode every, CPU | runs | runs; stats identical to A (only CPU-time column differs) |
| same, kk t 1 | runs | runs; stats identical to A |
| in.grg_more: 1 reaction -> re-issue with 136 (more), mode every, CPU | runs, but tallies of reactions 2..136 land in array_grid[icell][icol>=ncol] = the next cell's row (contiguous 2d array, so valgrind on cpu_A reports 0 errors: silent wrong values, last cell out of bounds) | ERROR "reactions changed since compute was defined" |
| same, kk t 1 | runs (same out-of-row writes) | same ERROR |

Necessary: the fewer-reactions deck errors before the fix (V3-R3 report) and runs after.
Complete: growth is still rejected in CPU and kk; shrink and equal counts pass.
