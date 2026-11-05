# Agent prompts (launch: Agent tool, general-purpose, run_in_background). Each starts with:
"Read /home/user/sparta/.fix-review/PROTOCOL_V.md and follow it exactly. Your name: <name>."

## r1
Task 1 (lost-fix, chunks 01-14): rows.json rows with round 1 or 2, ch 00..14, v AB-CONFIRMED or AB-CONFIRMED-HARNESS,
EXCEPT these ids already verified OK on round-7 F (record them as OK-carried with that reason, no rerun):
01-2 01-3 01-6 02-4 01-5 02-6 02-7 02-8 02-9 00-5 03-7 03-8 07-6 08-2 10-2 10-3 13-2 14-1 14-3 14-4 10-4
00-1 01-1 02-1 02-2 02-3 02-5 02-10 02-11 02-12 02-13 02-14 02-15 02-16 03-1 03-2 03-3 03-4 03-5 04-1 04-5 04-6 04-7
04-8 04-11 04-13 05-1 05-2 05-3 05-4 05-6 05-7 05-8 05-11 05-12 05-13 06-1 06-2 06-3 06-4 06-10 06-11 06-12 06-13.
For each remaining id: find the fixed code (round-1 commit for its subsystem + later commits naming it); if the file was
changed by the master merge (git diff 9bab6486 F -- <file> non-empty) or after, recreate the positive control from the
row's pos text and run it on fin and mst (M lacks almost all fixes, so M should show the bug and F must not), plus the
negative control (F == M); otherwise OK by diff (state the function and that it is unchanged since the fix commit
modulo later fixes of the same finding). Prefer runtime whenever a quick deck exists.
Task 2 (new merge d06625d6): (a) compute gas/reaction/grid every/select/all, then react redefined with more, fewer, none:
F must error only for more (clean, no ASan), run otherwise; M for comparison (M ASan-overflows on more). Same on kfin_kk.
(b) FFT 2d single precision: build a serial FFT_SINGLE variant of F and M in scratch (copy $SP/ab/fin tree, add
-DFFT_SINGLE to CCFLAGS of a Makefile.abserial copy) and run examples/fft 2d decks: F single must run and agree with F
double within float tolerance; record what M single does.

## r2
Task 1 (lost-fix, phase 3 and merge 2): rows.json rows with round 3 or 4 (P-*, Q-*, 30-S, 31-*, kk *, K-*, S-1, 32-*).
Same method as r1 task 1 (fin/mst host, kfin_kk/kmst_kk KOKKOS). Files changed by the master merge include
KOKKOS/compute_surf_kokkos.cpp/.h, compute_react_surf_kokkos.cpp, react_tce(_qk)_kokkos.h, react_bird_kokkos.cpp,
grid/particle_custom_kokkos.cpp, fix_emit_surf.cpp, grid_collate.cpp, custom.cpp, surf*.cpp, modify.cpp: rerun those
items at runtime. P-9 was backed out on purpose (its decks must still pass via grid collate routing).
Task 2 (KOKKOS example sweep): every examples/*/in.* that runs with -sf kk (skip vtk/python/paraview), copied to
$SP/v7/r2/ex/<build>/<dir>, on kfin_kk and kmst_kk at np1 and np2 (timeout 180 s); compare stats ignoring CPU/time
columns; explain every difference by a fix (host differences already explained in LEDGER: 08-1, 19-8, 12-7, 19-10, 06-9,
29-11, 15-2, P-1; KOKKOS-only fixes in rows round 3) or flag REGRESSION.

## r3
CI replica: reproduce .github/workflows/main.yml jobs "mpi" (cmake mpi preset, PKG_FFT, PKG_PYTHON, testing on; skip VTK
if libvtk is unavailable) and "mpi-kokkos-exact" (kokkos_mpi_only preset, Kokkos Serial, SPARTA_KOKKOS_EXACT,
SPARTA_SPA_ARGS="-k on -sf kk") for F and for M, in $SP/v7/r3/<F|M>-<job>/ worktrees (git worktree add --detach is allowed
for this). Run ctest -j2 (export the OMPI root vars from env.sh). Stage markers in $FR/v/r3-state/ make it idempotent
(configured/built/tested per job; a killed ctest reruns the whole suite). Report: per job pass/fail counts F vs M; every
test failing on F but passing on M: is the difference explained by a fix (LEDGER lists the example-level ones) — i.e. the
gold log needs reblessing — or a REGRESSION? List exactly which gold logs (examples/*/log.*) would need rebless for CI to
pass on F, with the fix responsible. Also report tests failing on both.
