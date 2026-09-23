# Handoff: the `integrate+bbox` geometry kernels are the GPU scaling ceiling

**Repo**: `stanmoore1/sparta`, branch `surf-rigid-body-owned`.
**Status**: diagnosis complete and cross-validated on CPU and GPU. The
fix is not started. This document is everything needed to start it.

---

## 1. The problem in one table

4000 bodies, 40 steps, medians of 3, `bodies owned`:

| Ranks | CPU | Eff | GPU | Eff |
|---|---|---|---|---|
| 1 | 25.11 s | 100% | 5.85 s | 100% |
| 2 | 16.93 s | 74% | 6.47 s | 45% |
| 4 | 11.96 s | 52% | 6.41 s | **23%** |

The same code strong-scales at 52% on CPU and 23% on GPU. So the cause
is **not** work replicated across ranks -- that would penalise both
equally.

CPU stage speedups, 1 -> 4 ranks:

| stage | 1 rank | 4 ranks | speedup |
|---|---|---|---|
| remove inside | 6.448 s | 2.583 s | 2.5x |
| recut | 3.664 s | 1.280 s | 2.9x |
| collision lists | 1.745 s | 0.531 s | 3.3x |
| **integrate+bbox** | 0.894 s | 0.944 s | **1.0x** |
| **apply split changes** | 0.676 s | 0.925 s | **0.7x** |

The heavy stages scale. Two stages do not. On the GPU the heavy stages
are already fast, so the non-scaling pair dominates: `integrate+bbox` is
**1.67 s of the 5.85 s single-GPU loop (29%)** against 0.89 s of 25.1 s
on CPU (4%). It is an Amdahl ceiling created by making everything else
fast.

## 2. Inside `integrate+bbox`: it is the geometry kernels, nothing else

Measured with a temporary probe on Kokkos Serial, 1000-body owned deck,
seconds per 40-step run. Host code costs the same under Serial as on a
GPU, so the host/transfer rows below are GPU numbers too:

```
pack_body_lists    0.0001      bbox DtoH+unpack   0.0005
pose host fill     0.0008      body_bins          0.0007
pose HtoD          0.0000      pack_body_device   0.0005
geometry kernels   0.2950      ftally zero        0.0076
                               body_status        0.0003
```

**95% of the stage is the four geometry kernels.** Everything host-side
or transfer-side is 1% combined. Several plausible-sounding culprits are
ruled out by this: the whole-`nbody` pose upload, the whole-`nbody` bbox
read-back, the whole-domain body-bin rebuild, `body_status`,
`pack_body_device`. Do not spend time on them.

The kernels are, in `FixRigidKokkos::device_geometry()`:

- `TagFixRigidGeometry` over `nlelem_kk` (elements of held bodies)
- `TagFixRigidBodyBox` over `nblist` (held bodies)
- `TagFixRigidInflate` over `nlelem_kk`
- `TagFixRigidGroupBox` over `nlgroup_kk` (element groups of held bodies)

plus `TagFixRigidScatterSurfs` over `nlcopy_kk` in the same timed window
when `sweepflag == 0`.

## 3. The work partitions; the kernels do not get faster

Instrumented iteration counts, same deck, per step:

| | 1 rank | 4 ranks | ratio |
|---|---|---|---|
| elements (`nlelem_kk`) | 32,000 | 8,579 | 3.73x less |
| held bodies (`nblist`) | 1,000 | 268 | 3.73x less |
| surf copies (`nlcopy_kk`) | 32,000 | 8,579 | 3.73x less |
| **geometry kernel time** | 0.3035 s | 0.2507 s | **1.21x faster** |

So owned mode feeds the kernels correctly -- this is not a partitioning
bug. Per-element cost **triples** at 4 ranks. On a 4-core CPU VM that is
consistent with shared memory bandwidth; on 4 separate H100s, each with
its own HBM, it cannot be.

## 4. The arithmetic that says this is worth an ncu run

On the GPU, `integrate+bbox` is 1.67 s / 40 = **42 ms per step** for
four kernels over 128,000 elements (4000 bodies x 32).

Those kernels stream roughly 240 bytes per element (read `displace`,
write `bodypt`, `bodynorm`, `elemlo`, `elemhi`, and the `_new` set on a
sweep step). That is about 30 MB, roughly **10 microseconds** at H100
HBM bandwidth. The observed cost is ~1000x that.

Two hypotheses, and ncu separates them in one run:

1. **Occupancy / divergence / register pressure.** One thread per
   element. At 4 GPUs that is 8,579 threads across 132 SMs, about two
   warps per SM. `TagFixRigidGeometry` does quaternion-frame rotation
   per corner point and a swept-box reduction; `Inflate` and `GroupBox`
   are short. If achieved occupancy is low and the kernels are latency
   bound, giving them less work will not make them faster -- exactly the
   1.21x above.
2. **The stage timer is catching a fence.** `FixRigid::stage_begin()`
   and `stage_end()` call `Kokkos::fence()` when `SPARTA_RIGID_TIMING`
   is set, and `bbox_to_host()` does a blocking `sync_host()`. If async
   work queued elsewhere drains here, the 42 ms is mis-attributed.

**What to collect**: per-kernel duration and achieved occupancy for the
four tags above, at 1 GPU and 4 GPUs, on the 4000-body owned deck. If
kernel durations sum to much less than 42 ms, it is hypothesis 2 and the
timer/fence placement is the thing to fix. If they sum to ~42 ms with
low occupancy, it is hypothesis 1.

New permanent sub-timers landed for exactly this, usable without ncu as
a first cut (`SPARTA_RIGID_TIMING=1`): `geometry: pose up`,
`geometry: kernels`, `geometry: bbox back`. They cut across
`integrate+bbox` and `set_xv+bounds`, since `device_geometry()` is
called from both.

## 5. If it is hypothesis 1, the shape of the fix

More parallelism per kernel, not less host work:

- A thread per **corner point** rather than per element (3x more threads
  in 3d, 2x in 2d), or a team per body with threads over its elements.
- Fuse `Geometry` + `Inflate` + `GroupBox` into one kernel so the
  element data is read once instead of three times, and one launch pays
  the latency instead of three. `Inflate` already re-reads what
  `Geometry` just wrote; `GroupBox` re-reads what `Inflate` wrote.
- Check register pressure in `TagFixRigidGeometry`: it holds a pose row
  (16 doubles) plus corner points in registers.

**Constraint**: every result must stay bit-identical. The validation
gate is in section 7. Fusing the three kernels is safe only if the
ordering dependence is respected -- `Inflate` must see `Geometry`'s
output and `GroupBox` must see `Inflate`'s, so a fused kernel needs one
thread to own a whole element (and a group box needs its group's
elements complete, which means `GroupBox` either stays separate or the
fusion is per-group).

## 6. `apply split changes` is the second non-scaling stage

0.676 -> 0.925 s on CPU, 1 -> 4 ranks. Not yet investigated. It is
`FixRigid::remap_grid()`'s tail: `grid_rebuild()` on fallback, else
`remap->apply_pending()` plus `relabel_moved_cells()`, and in the KOKKOS
path `GridKokkos::apply_changes()`. Known sub-timers `apply: restructure`
and `apply: cell counts` do not account for all of it, so the first step
is another sub-timer, not a guess.

## 7. Validation gate, non-negotiable

Every change here must be bit-identical. Run from
`tools/testing/rigid`:

```
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
EX=build-mpi/src/spa_mpi
KK=build-kkserial/src/spa_kokkos_mpi_only
python3 run_tests.py --exe $EX                       # and --mpi "mpirun --oversubscribe -np 4"
python3 run_tests.py --exe $EX --dist
python3 run_tests.py --exe $EX --owned               # and --owned --mpi ... -np 4, --owned --dist
python3 run_tests.py --exe $KK --args "-k on -sf kk" # same six combinations
```

All must report `0 test(s) failed`. Then the benchmark, comparing the
stats table with the CPU-time column stripped (`/home/user/bench/st.sh`),
for `in.bench40` and `in.bench40.owned` at 1 and 4 ranks on both builds.

Because this touches device views, also run the coherence emulation
build at `/home/user/sparta-sync` (`build-sync`) with
`SPARTA_KOKKOS_WATCH=1 SPARTA_KOKKOS_STALE=1`: Kokkos Serial aliases
host and device memory and cannot see a dropped transfer or an unclaimed
device write.

**Two traps that have already bitten here**: `mpirun` refuses to run as
root without the two environment variables above, and silently produces
empty logs, which a naive diff reports as IDENTICAL -- always assert the
log has rows. And a new pointer member must be NULLed *before*
`setup_body()` in the `FixRigid` constructor, not after; `setup_body()`
is called from the constructor and allocates.

## 8. What has already been ruled out, so nobody repeats it

- **Replicated vs owned bodies**: no measurable difference at 4000
  bodies on <= 4 ranks. Owned mode works and partitions correctly
  (section 3); it is simply not the ceiling.
- **Replicated vs distributed surfs**: identical at every point and
  identical scaling, on both CPU and GPU.
- **MPI sync points**: a barrier placed before each makes the re-cut
  Allreduce 150x cheaper, so it is skew absorption, not MPI cost. The
  two point-to-point exchanges hold ~0.16 s per 40 steps of genuine
  latency at 4 ranks; the forward one's count round trip is already
  hidden behind its packing.
- **Host install of the device cut** (`recut: install`): 6.9 ms/step at
  1 rank, 1.8 at 4, with only 0.28% of cut cells changing piece count.
  Real, and the staged design for removing it is in the plan file, but
  its ceiling is 14% at 1 GPU and 4% at 4.
- **Per-step O(global nbody) host work**: `rigid_upload`, `body_bins`
  allocation churn, the `ftally` zero -- all measured at ~1% of
  `integrate+bbox` combined. Partly fixed already; not the ceiling.

## 9. Recent commits, newest first

- `geometry sub-timers + zero only held bodies' tally rows` (this work)
- `ddcbd2c` `recut: install` sub-timer, piece-count counter
- `619cd84` R1/R2: held-row `rigid_upload`, persistent bin buffers,
  count trade covered by the packing
- `bfc3aa5` device twin of the element-group prefilter
- `134e75d` owned split cells from the split info, not a grid scan
- `21a8a59` re-cut skips candidates no body element reaches
- `b4ac4da` per-cell element scans test element groups first
- `b78f190` deletion pass classifies cells, not particles

Full history, measurements and the staged R3 design are in the plan file
`~/.claude/plans/i-benchmarked-the-surf-rigid-body-velvet-bee.md`.
