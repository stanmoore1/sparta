# Handoff: GPU testing of the review fixes for sparta/sparta#683

You are picking this up on a machine **with a GPU**. Everything below has already been
done on CPU (OpenMP + OpenMPI, no GPU). Your job is the part that needs real hardware:
check that the fixes are correct on CUDA/HIP and that they cost no performance.
Report results; **do not post anything to GitHub** (the PR or its comments) unless the
user tells you to.

## 1. What is being tested

| Item | Where |
|---|---|
| PR | [sparta/sparta#683](https://github.com/sparta/sparta/pull/683) "optimizations for GPU/cuda", Matt Bettencourt (NVIDIA). Head `21763d55`, base `e071055f`. |
| Review fixes | branch `claude/code-review-pr-683-29lhfg` on `stanmoore1/sparta`: the PR head plus three commits |
| &nbsp;&nbsp;`e7151f27` | first round: fixes findings 1-7, 10, 12-15 (below) |
| &nbsp;&nbsp;`57cb662c` | simplification of the two-pass move (section 3) |
| &nbsp;&nbsp;`9ff7c43d` | verified cleanups (section 3); **the head of the branch, and the candidate** |
| Scripts | branch `claude/great-allen-5wm0fz` on `stanmoore1/sparta`: `pr683_gpu_check.sh`, `pr683_force_gpu_path.sh`, this file |

```bash
git clone https://github.com/stanmoore1/sparta && cd sparta
```
```bash
git fetch origin claude/code-review-pr-683-29lhfg claude/great-allen-5wm0fz
```
```bash
git fetch https://github.com/sparta/sparta pull/683/head:pr683
```
```bash
git checkout claude/great-allen-5wm0fz
```

All the commits named above (`e071055f`, `21763d55`, `e7151f27`, `9ff7c43d`) must be
reachable in this clone; the script builds each one in its own `git worktree`.

### What the PR does (context)

- **Two-pass particle move on GPU** (`src/KOKKOS/update_kokkos.cpp`). `TagUpdateMoveFirstPass<DIM>`
  moves particles that stay inside a cell with no surfs; the rest are listed in `not_updated` and
  moved by the full kernel (`TagUpdateMoveIndirect`). A heuristic turns the first pass off when it
  completes < 90% of particles and retries every 100 steps. Only used on GPU, not on gfx940/942
  (`SPARTA_KOKKOS_REDUCE_ARCH`), and only when `nlocal >= concurrency`.
- On non-MI300 GPUs the move kernel is now always the `parallel_reduce` (`ATOMIC_REDUCTION = -1`)
  variant, where it used to be the atomics variant.
- **Fewer device-wide fences**: host<->device copies in the move are passed an execution space
  instance (`move_execution_space()`) and `sync_device_async()` was added to Particle/Grid/Surf.
- **Narrower host sync at output steps** (`sync_mask_for_output`), so stats-only output does not copy
  particles to the host.
- `d_plist` / `d_nn_last_partner` switched to LayoutRight.

The author's B200 numbers (PDF attached to #683; ms/step, ratio to baseline) for the PR head:
collide 0.97x/0.95x, cylinder 0.26x/0.11x, free 0.68x/0.69x, sphere 0.80x/0.64x (small/big).
The table's last row, "Atomic move kernel (no parallel_reduce)", is **not** in #683; it is the
branch the author recommends not merging. The cylinder input is not in the repo (ask the user
for it if you want to reproduce the author's table; `bench/in.free`, `in.collide`, `in.sphere`
are).

## 2. Bugs in the PR head, and what fixed them

All reproduced on CPU by forcing the GPU dispatch (section 5); none have been run on a GPU.

| # | Bug in `21763d55` | Fix |
|---|---|---|
| 1 | On CUDA and non-MI300 HIP every move counter (ntouch, ncomm, nbound, nexit, nscheck, nscollide, nreact, nstuck, naxibad) is 0: the `-1` kernels count into `d_move_reduce`, but `move()` read the atomics scalars because `atomic_reduction = 1` there. | `move()` sums both sources (both zeroed per attempt). |
| 2 | Axisymmetric runs: the first pass is instantiated for `DIM == 1` and moves r in a straight line with no `axi_remap()`, so the radial coordinate freezes. | First pass disabled for `DIM == 1`. |
| 3 | nearcp + reaction retry: `d_plist` widened but `d_nn_last_partner` not, out-of-bounds write (older than the PR). | `realloc_kokkos(d_nn_last_partner, ...)` in the retry branch. |
| 4 | Particles completed in the first pass are not counted in ntouch. | Added from the first-pass count. |
| 5 | Every Kokkos run aborts at exit (status 134, "execution space is being destructed after finalize()"): a function-local `static DeviceType`. | `e7151f27`: owned pointer; `57cb662c`: returned by value, held in locals. |
| 7 | plist resize allocated a new zero-filled view before freeing the old one. | Back to `MemKK::realloc_kokkos` (frees first, no zero fill). All plist reads stop at the cell count, so the zero fill is not needed. |
| 15 | Two-pass recheck stored as an absolute step, so `reset_timestep` disabled it for a long time. | `57cb662c`: `UpdateKokkos::setup()` re-enables the first pass at the start of every run. |
| 6, 10, 12, 14 | Dead `#if DEALLOC`, dead tally helpers, empty `{}` scopes, hard-coded loop bounds. | Cleaned up. |
| 13 | A second `const double volume` declared in the recombination branch of the collision kernel, shadowing the outer one. | Removed in `e7151f27`. **Needs a GPU timing check**: the author may have added it on purpose to lower register pressure (section 4, task P2). |

## 3. What `57cb662c` and `9ff7c43d` change (never run on a GPU at all)

`57cb662c`:


1. `move_execution_space()` returns `DeviceType()` by value. A default-constructed instance is the
   default stream, **the same one sort/collide/comm use**, which the move relies on (there are no
   fences between them). The point of passing it is only that `deep_copy(exec,...)` and
   `DualView::sync_device(exec)` skip the device-wide `Kokkos::fence()` before and after.
2. The second pass no longer waits on the host. Old: first pass, **fence**, read count, launch a
   `TeamPolicy` with `team_size = 128` sized from the count. New: first pass, then a `RangePolicy`
   over `concurrency` threads that loops `for (j = i; j < not_updated_cnt(); j += concurrency)`,
   then the count is copied back with the other move results and the 90% heuristic and the
   ntouch correction are applied after the existing end-of-move fence. One host round trip per
   step fewer; the second kernel always launches (it is cheap when the count is 0).
3. The heuristic threshold is `MOVE_TWO_PASS_KEEP` (0.90); recheck interval is still
   `MOVE_TWO_PASS_RECHECK` (100).

`9ff7c43d` (each item was checked against the code first; none changes results):

4. Particle/Grid/SurfKokkos: `sync(Device,mask)` and `sync_device_async(exec,mask)` share one
   body, `sync_device_views(mask, exec...)`. Without an execution space it calls
   `DualView::sync_device()` exactly as before, fences included (finding 9).
5. `Output::setup()` computes `sync_mask_for_output` once per run instead of six command hooks
   doing it (finding 11). Not a bug before, but the hooks missed `clear`, which keeps KokkosSPARTA
   and rebuilds Output, so the old mask stayed in force (a wider mask, so harmless).
6. Removed a block in `move()` that zeroed the device counters only when `atomic_reduction` was
   set; `zero_host_scalars()` already clears them before every attempt.
7. `not_updated` is allocated where the first pass runs, with `realloc_kokkos` (frees first, no
   zero fill), instead of on every GPU build: MI300 and axisymmetric runs never use it.

Risk to look for on GPU: the `RangePolicy` block size Kokkos picks for the big `moveOne` reduce
kernel may differ from the old 128-thread teams. Task P1 measures that.

## 3a. Compatibility with sparta/sparta#700 (bug-fix PR)

#700 ("Bug fixes from an AI line-by-line review of `src`", 272 files) is expected to merge too.
Merging it into the review branch conflicts in five files. The resolved merge is `b48f7c2f`
(parents `9ff7c43d` and `4863e6ec`), also merged into `claude/great-allen-5wm0fz`. Resolution:

- **Move counters, `update_kokkos.cpp`.** #700 fixes the counter readback with a `use_reduce`
  flag that mirrors master's kernel dispatch. #683 changes that dispatch (the reduce kernel on
  every GPU), so taking #700's version would bring back finding 1: counters 0 on CUDA. The merge
  keeps the review branch's version, which sums both sources and does not depend on the dispatch.
  **Whichever PR merges second has to keep this.**
- **`sync()` in Particle/Grid/SurfKokkos.** #700 makes `auto_sync` refresh the host before marking
  it modified (a DualView modified on both sides aborts). The merge puts that block in the shared
  `sync_device_views()`, so `sync_device_async()` gets it too. (Inside `move()` `auto_sync` is off,
  so the move path does not change.)
- **Particle/grid syncs in `move()`.** #700 moved them to the top of the loop, before the device
  handles are taken; the merge keeps that placement with #683's async copies.
- **`collide_vss_kokkos.cpp`.** Both sides contain the same `d_nn_last_partner` retry fix
  (finding 3); took #700's.
- Checked for semantic overlap in the files that merged cleanly (`kokkos.cpp`, `kokkos_type.h`,
  tally computes, `output.cpp`, `stats.cpp`, `dump_particle.cpp`, the move kernel): none. #700's
  new `PDISCARD` early exit in `moveOne()` is reached only through flags the first pass already
  passes on.

CPU results for `mrg` (4 ranks x 1 thread): 17 examples identical to `pr700` and, with the forced
GPU path, identical to its own CPU path; the only example that differs from `fix` is `ambi`,
which differs the same way between master and `pr700` (a #700 fix). Targeted inputs: forced GPU
== CPU == `pr700`, counters non-zero, two-pass on all 400 steps of `in.free3dsmall`. Recheck
trace and the output-mask test (10 dump files identical to `pr700`) pass. All runs exit 0.

## 4. Tasks, in priority order

`pr683_gpu_check.sh` does C1-C3, C6 and P1-P2 in one go. It builds these variants (one worktree and
build dir each under `$WORK`):

| variant | commit | meaning |
|---|---|---|
| `base` | `e071055f` | master before #683 |
| `pr` | `21763d55` | #683 head |
| `fix1` | `e7151f27` | first round of fixes |
| `fix` | `9ff7c43d` | **candidate** |
| `fixvol` | `9ff7c43d` + the `volume` line restored | for finding 13 |
| `mrg` | `b48f7c2f` | `fix` merged with #700 (section 3a) |
| `pr700` | `4863e6ec` | #700 alone (`git fetch https://github.com/sparta/sparta pull/700/head:pr700`) |

```bash
PRESET=kokkos_cuda EXTRA_CMAKE="-DKokkos_ARCH_HOPPER90=ON" ./pr683_gpu_check.sh 2>&1 | tee check.log
```

Set the arch flag for your GPU (for example `-DKokkos_ARCH_BLACKWELL100=ON`, `-DKokkos_ARCH_AMPERE80=ON`;
for AMD use `PRESET=kokkos_hip` and `-DKokkos_ARCH_AMD_GFX90A=ON` or `GFX942`). The presets force
`Kokkos_ARCH_HOPPER90` / `VEGA90A` on, so turn that one off when you pick another
(`-DKokkos_ARCH_HOPPER90=OFF`). `RUN="mpirun -np 1"` if the build uses MPI and your launcher
needs it. Each build is a full SPARTA + Kokkos build; expect several minutes each.

### Correctness (C)

- **C1 counters** (`in.free3ds`, 4M particles, two-pass on): the `Ntouch Nexit Nbound Ncomm`
  columns must be **non-zero** for `fix1`, `fix`, `fixvol`, and 0 for `pr` (that is the bug). `base`
  is a valid reference for the magnitudes (it never runs the two-pass code).
- **C2 axisymmetry** (`in.axir`): `c_ry` (average radius) for `fix*` must match `base` up to
  statistical noise; `pr` is expected to stay stuck near its step-0 value.
- **C3 nearcp + retry** (`in.chemnn3`, run with `react/retry yes react/extra 1.0`): must reach step
  30 with exit 0 on `fix*`. On CPU, a normal release build of `pr` dies right after step 1, the step
  where the retry widens the plist; a bounds-checked build (`-DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON`)
  reports `out of bounds ... collide:nn_last_partner`.
- **Exit status**: every `pr` run is expected to exit 134; every `fix*` run must exit 0. The script
  still times `pr`, because its abort happens after the run.
- **C4 two-pass actually engages** (not in the script). The first pass only runs when
  `nlocal >= concurrency` (about 270K-300K threads on H100/B200) and keeps running only while
  >= 90% of particles stay in their cell. Confirm with Nsight Systems that `TagUpdateMoveFirstPass`
  and `TagUpdateMoveIndirect` kernels appear every step for `in.free3ds` and for `bench/in.free`
  at size 100:
  ```bash
  nsys profile -o free_fix --stats=true ./pr683_check/build_fix/src/spa_kokkos_cuda -k on g 1 -sf kk -var x 100 -var y 100 -var z 100 -in bench/in.free
  ```
  (Run it from the `bench` directory, or copy the input and `ar.species` first.) Report the
  kernel times of the first pass, the second pass, and `TagUpdateMove` for `fix` vs `fix1`.
  Also check there is **no** `cudaStreamSynchronize`/`cudaDeviceSynchronize` between the two
  move kernels in `fix` (there is one in `fix1`).
- **C5 MI300 (gfx942), if available.** That path (`SPARTA_KOKKOS_REDUCE_ARCH`) never runs the
  two-pass code; it has never been run at all in this review. Just check counters non-zero and
  exit 0 on C1-C3.
- **C6 #700 compatibility.** `mrg` must pass C1-C3 exactly like `fix` (non-zero counters in
  particular: section 3a explains why). `in.customgrow` (particles created between runs with a
  custom attribute) is #700's positive control for the `auto_sync` fix: `pr700` and `mrg` must
  exit 0; `base`, `pr` and `fix` may abort with a DualView "modified on both sides" error. It
  could not be reproduced on CPU (OpenMP shares one memory space, so nothing aborts there).
  Also run P1 on `mrg` vs `fix`: the merge should cost nothing.

### Performance (P)

The script prints ms/step (min of `REPEAT` runs) for `bench/in.free`, `in.collide`, `in.sphere` at
`SIZES` (default 50 and 100, i.e. 1.25M and 10M particles).

- **P1 `fix` vs `pr`** is the headline: the fixes must not slow the PR down. Fixes 1 and 4 add a few
  host-side additions; fix 2 only affects axisymmetric runs; `57cb662c` removes a host wait but
  changes the second-pass launch shape. Treat > 2% slower on any case as something to investigate
  (rerun with `REPEAT=5` first; GPU timings of 0.1 ms/step steps are noisy). Compare `fix` vs `fix1`
  to isolate `57cb662c` + `9ff7c43d`; of those, only the second-pass launch should be able to
  change GPU time.
- **P2 `fixvol` vs `fix`** decides finding 13. `in.collide`/`in.sphere` have no reactions, so the
  branch is never taken; any difference would come from register allocation. If `fixvol` is
  faster by more than noise, the line should go back (renamed, for example `recomb_volume`, with a
  comment saying why). Also worth checking with `--resource-usage` / `-Xptxas -v` whether the
  register count of `TagCollideCollisionsOne` differs between the two builds.
- **P3 `fix` vs `base`** should show the PR's gains (author's table, section 1). If `fix` is not
  clearly faster than `base` on `in.free`, two-pass or the fence removal is not engaging (go to C4).

## 5. What was already verified on CPU (do not redo)

The script itself was run end to end on this CPU machine (`PRESET=kokkos_omp`, 4 MPI ranks):
all variants build, `fixvol` applies, and the correctness rows come out as described in C1-C3.
CPU timings cannot settle P2: `collide` at 270K particles gave `fix` 4.07-5.22 and `fixvol`
3.54-5.03 ms/step over 4 runs each.

Environment: 4 cores, g++ 13, OpenMPI 4.1.6, Kokkos OpenMP (`cmake/presets/kokkos_omp.cmake`).
The GPU move code was run on CPU with a test-only patch, `pr683_force_gpu_path.sh <source dir>`.
It defines `SPARTA_TEST_FORCE_GPU` in `update_kokkos.cpp` and `kokkos.cpp`, which takes the GPU
`#if` branches in `move()`, sets `atomic_reduction = 1` as on a GPU, uses `team_size = 1` where the
old TeamPolicy exists, and prints `TWOPASS step N` when `SPARTA_TEST_TRACE` is set.

Results for `e7151f27`, 4 MPI ranks x 1 thread unless noted:

- **Bugs reproduced on `pr`** (forced GPU path): counters all 0; axisymmetric `c_ry` 0.667 vs 0.789;
  exit 134 on every run (also on the plain CPU build, all 17 examples); no recheck after
  `reset_timestep`. None of these on `fix1`.
- **Forced GPU path == CPU path, bit-identical stats**, with two-pass on all 400 steps
  (`in.free3dsmall`), with gridcut and multi-iteration moves (`in.free3dcut`), and axisymmetric.
- **EXACT** (`-DSPARTA_KOKKOS_EXACT=ON`, 1 rank, 1 thread): Kokkos on the forced two-pass path is
  bit-identical to the non-Kokkos CPU code for free flow. Inputs with `fix emit/face` differ, but
  that difference is already on master (EXACT `base` Kokkos == EXACT `fix` Kokkos != plain CPU):
  a separate pre-existing gap, not caused by #683.
- **17 examples** (collide, optmove, tally computes, chem, surf_collide, ambi, circle,
  surf_react_adsorb, axi): stats identical for `base`, `pr`, `fix1`, and `fix1` with the forced GPU
  path. `ambi/in.ambi` needs `-pk kokkos react/retry yes` on every build.
- **CPU performance** (`bench` free/collide/sphere, 640K particles, 4x1 and 1x4): no difference
  beyond noise between `base`, `pr`, `fix1` (collide 1x4, 5 interleaved runs: 81.9-89.9,
  81.7-88.3, 83.6-87.7 ms/step).

Results for `57cb662c` and `9ff7c43d` (`fix`), same setup:

- Stats **bit-identical to `e7151f27`** on all targeted inputs and all 17 examples, both on the
  CPU path and with the forced GPU path (two-pass on all 400 steps of `in.free3dsmall`).
- EXACT, 1 rank, 1 thread: forced two-pass path bit-identical to the non-Kokkos CPU code (free
  flow without `emit/face`).
- Recheck (`in.recheck`: run 250 steps, `reset_timestep 0`, run 50, smaller timestep, run 20):
  first pass runs at steps 1, 101, 201 | 1 | every step 51-70. Under `e7151f27` the third run left
  it off, because the 100-step wait carried over from the previous run.
- Exit status 0 everywhere.
- `9ff7c43d`: the same checks again (targeted inputs, recheck, EXACT, 17 examples), all
  bit-identical to `e7151f27`, exit 0. Plus an output test (`clear`, `undump`, dumps added
  between runs, stats-only runs; 10 particle dump files): every dump file and the stats are
  byte-identical to `base`, on the CPU path and the forced GPU path.
- Not measured: anything about performance on GPU, and the RangePolicy launch shape. On OpenMP
  the forced path runs with `concurrency` = number of threads, so it says nothing about GPU block
  sizes.

## 6. Open questions for the user / the PR author (not for you to decide)

- Finding 7: in #623 the author deliberately replaced `realloc_kokkos` with a zero-filled
  allocation ("fixed copy in resize sort list"); the review reverted that. Ask why.
- Findings 8, 9 and 11 are now done on the branch (`57cb662c`, `9ff7c43d`); the author should
  agree with them.
- Pre-existing, not from #683: with `SPARTA_KOKKOS_EXACT`, inputs using `fix emit/face` do not
  match the non-Kokkos code (master shows the same). Worth its own issue.
- The `d_plist` LayoutRight change measured as noise in the author's table; keep it only if it
  shows a gain somewhere.

## 7. What to report back

A short table per GPU: C1-C5 pass/fail with the offending numbers, and the P1-P3 ms/step table
from the script (plus the Nsight kernel times from C4). Say which commit and arch each result is
from. If a fix needs changing, commit it on `claude/code-review-pr-683-29lhfg` with a message
that says what was measured, and say so in the report.
