# Kokkos: the reorder buffer swap, built and measured

`ParticleKokkos::sort_kokkos()`'s `COPYPARTICLELIST` reorder scatters particles into `d_sorted`
and then does `Kokkos::deep_copy(d_particles,d_sorted)` — a second full pass over the array to put
the data back where it started. Four commented-out lines just above it show someone had tried
swapping the buffers instead and backed it out.

The host path ping-pongs its two buffers (`particle.cpp`, `sort_reorder`), which is what makes its
out-of-place counting sort cheap, so this looked like a free win. **It is not, at scale.**

## Build

No Kokkos makefile exists in `src/MAKE/`, so this used the cmake path:

```
apt-get install -y openmpi-bin libopenmpi-dev      # OpenMPI 4.1.6
rm -f src/style_*.h                                # cmake refuses to run alongside GNU-make artifacts
mkdir build && cd build
cmake -C ../cmake/presets/kokkos_mpi_only.cmake ../cmake   # Kokkos Serial backend + MPI, C++20
make -j4                                           # -> src/spa_kokkos_mpi_only
mpirun --allow-run-as-root --oversubscribe -np 4 ./spa_kokkos_mpi_only -k on -sf kk -in ...
```

## The change

`k_sorted` becomes a `tdual_particle_1d` (a DualView) rather than a bare device view, kept at
exactly `k_particles`' extent so the swap is capacity-neutral against `Particle::maxlocal`. After
the scatter, both sides of the DualView are swapped together, `Particle::particles` is rebound, and
the device side is marked modified:

```cpp
std::swap(k_particles,k_sorted);
k_particles.clear_sync_state();
d_particles = k_particles.view_device();
d_sorted    = k_sorted.view_device();
particles   = k_particles.view_host().data();
this->modify(Device,PARTICLE_MASK);
```

Rebinding the host pointer is what the earlier attempt was missing: it swapped only
`k_particles.view_device()`, and on a host-only backend the two views alias, so
`Particle::particles` would have been left pointing at the wrong buffer.

## Correctness

- **1 rank: bitwise identical** to the unmodified binary (`step/np/nattempt/ncoll/c_temp`).
- **4 ranks: cannot be checked bitwise.** The 4-rank run is not reproducible against *itself* —
  baseline vs baseline differs, and `global comm/sort yes` does not fix it. Verified statistically
  instead: `ncoll/step` 70529 (base) against 70656 (swap), temperature identical to all printed
  digits.

## Correction: the first round of these measurements was invalid

The performance numbers first recorded here were produced by running all of binary A, then all of
binary B. That is block-then-block measurement, and earlier in this same work I wrote a warning
against it into the header of `ab.sh`, because this machine drifts enough to invent or hide a
10% effect over the minutes such a block takes. Baseline loop time for one identical
configuration ranged from 6.4 s to 11.3 s across this session.

Everything below is **interleaved** — A,B,A,B... — so both binaries see the same stretch of
machine, and is the minimum of 6-10 pairs. The conclusions changed.

## Performance

`bench/in.collide`, Kokkos Serial backend, `global particle/reorder`.

| config | | loop (s) | Move | Coll | Sort | |
|---|---|---:|---:|---:|---:|---:|
| 1 rank, 1M, reorder 1 | base | 11.262 | 2.785 | 2.811 | 5.554 | |
| | **swap** | **9.606** | 2.722 | 2.826 | **3.951** | **1.172x** |
| 1 rank, 1M, reorder 2 | base | 9.389 | | 3.038 | 3.434 | |
| | **swap** | **8.547** | | 3.038 | **2.669** | **1.098x** |
| 1 rank, 250K, reorder 5 | base | 1.4525 | 0.589 | 0.489 | 0.361 | |
| | **swap** | 1.7974 | 0.653 | **0.708** | 0.408 | **0.81x** |
| 4 ranks, 1M, reorder 1 | base | 2.5174 | 0.601 | 0.461 | 1.231 | |
| | **swap** | 2.4851 | 0.699 | **0.682** | **0.965** | 1.013x |
| 4 ranks, 1M, reorder 5 | base | 1.9242 | 0.648 | 0.612 | 0.453 | |
| | **swap** | 2.0595 | 0.676 | **0.745** | 0.437 | 0.934x |

Sort improves everywhere — 22% to 29% — which is the intended effect and is never in doubt.
Whether that survives to the bottom line depends entirely on whether Move and Coll pay a penalty.

## Why Move and Coll sometimes pay: it is the per-rank working set, not the rank count

"It hurts on 4 ranks but not 1" is not a mechanism. The two configurations differ in something
else: at 1 rank/1M each buffer is 96 MB, while at 4 ranks/1M each rank's buffer is 24 MB. Running
**1 rank at 250K**, which reproduces the 4-rank per-rank footprint, reproduces the penalty exactly
(row 3 above, 0.81x, Coll +45%). So it is buffer size, and the rank count is incidental.

**A controlled experiment shows it is allocation identity alone.** Probe C does the `deep_copy`
*and then* swaps. After the copy both buffers hold byte-identical data, so the swap is
semantically a no-op against baseline; the only thing that changes is which allocation Move and
Coll read from afterwards:

| 1 rank, reorder 5 | 250K (24 MB/buffer) | 1M (96 MB/buffer) |
|---|---|---|
| base, Coll | 0.4888 | 3.2211 |
| probe C (copy **and** swap), Coll | **0.7199** | 3.2440 |
| swap, Coll | **0.7076** | 3.1842 |

Probe C reproduces the penalty at 24 MB and shows none at 96 MB. Nothing about the swap *logic* is
responsible — it is purely which buffer is being read.

**The mechanism, and it is consistent with this machine's measured cache behaviour.**
`ROOFLINE.md` measured the bandwidth-vs-size curve on this guest in round 1: working sets of
4-64 MB sustain 23-27 GB/s (L3), and larger ones fall to DRAM speed.

- At **24 MB per buffer** one buffer is L3-resident and, in the baseline, stays warm across every
  step because `deep_copy` always puts the data back in the *same* allocation. Swapping hands Move
  and Coll a cold twin after each reorder, forcing a full reload. Two 24 MB buffers plus the grid
  and plist arrays sit at or past the edge of what stays resident, so the warm copy is not
  recovered.
- At **96 MB per buffer** neither buffer can ever be resident; every access streams from DRAM
  regardless of which one it is. Alternation is free, and removing the redundant `deep_copy` is a
  clean 1.17x.

That also explains the shape of the reorder sweep: at 96 MB the benefit tracks how often the
reorder runs (1.172x at period 1, 1.098x at 2), exactly as removing one copy per reorder step
should.

## Recommendation

The `deep_copy` is genuinely redundant and worth removing, but **swapping is the wrong way to
remove it** whenever the particle array is small enough to hold cache residency between steps —
which includes the ordinary case of a decent rank count on one node.

Two honest caveats on generalising:

- This is the **Serial (CPU) backend**. On a real GPU the particle array lives in HBM with no large
  shared victim cache holding tens of MB across kernel launches, so the alternation penalty
  plausibly does not exist and the swap would be a clean win. **I could not test that here** and am
  not going to assume it.
- The right fix, which sidesteps the tradeoff entirely, is to avoid the second buffer: permute in
  place rather than scatter-and-copy. That is what the currently-disabled `FIXEDMEMORY` scheme
  attempts (`reorder_scheme` is hard-coded to `COPYPARTICLELIST` with the comment "FIXEDMEMORY
  reorder temporarily disabled due to bug on GPUs").

## Two unrelated Kokkos findings from reading this code

1. **The Kokkos reorder does not permute custom per-particle attributes.** `sort_kokkos()` never
   touches `k_eivec`/`k_edvec`/`k_eiarray`/`k_edarray`. The host path is careful about this —
   `Particle::reorder()` calls `copy_custom(dst,src)` inside its cycle permutation, and
   `sort_reorder()` refuses to run at all when `ncustom` is set. If `global particle/reorder` is
   combined with anything that creates custom particle data (`fix ambipolar`'s `ionambi`/`velambi`,
   `fix vibmode`), the particles move and their custom data does not follow. Not verified by a run
   — flagged from code reading only, and worth checking.
2. **`sorted_contiguous` is not reset on the Kokkos path.** `UpdateKokkos::move` resets
   `particle->sorted` but not `sorted_contiguous`; the host `Update::move` resets both. Harmless
   today because nothing on that path sets the flag, but it is a one-line fix.
