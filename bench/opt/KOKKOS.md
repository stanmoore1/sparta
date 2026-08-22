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

## Performance — it is a win only below ~10 MB per buffer

`bench/in.collide` with `global particle/reorder`, minimum of 3 runs, Kokkos Serial backend.

| ranks | particles | | loop (s) | Move | Coll | Sort |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 10K | base | 0.0414 | 0.0203 | 0.0144 | 0.0060 |
| 1 | 10K | **swap** | **0.0392** (1.06x) | 0.0196 | 0.0141 | 0.0048 |
| 1 | 100K | base | 0.4823 | 0.2060 | 0.1772 | 0.0941 |
| 1 | 100K | **swap** | **0.4544** (1.06x) | 0.1993 | 0.1758 | 0.0744 |
| 1 | 1M | base | 6.502 | 2.420 | 2.340 | 1.672 |
| 1 | 1M | **swap** | **7.152** (0.91x) | 2.530 | **2.884** | 1.635 |
| 4 | 100K | base | 0.1336 | 0.0519 | 0.0399 | 0.0214 |
| 4 | 100K | **swap** | **0.1273** (1.05x) | 0.0500 | 0.0391 | 0.0165 |
| 4 | 1M | base | 1.4625 | 0.5531 | 0.4776 | 0.3153 |
| 4 | 1M | **swap** | **1.6140** (0.91x) | 0.5785 | **0.5972** | 0.3264 |

The intended effect is real and shows up everywhere — Sort falls 10–23%. But at 1M particles
Move and especially Coll get slower by more than that, and the net is a 9% regression.

## Why: the reordered list buys less

Move and Coll should not be affected by this change at all, so the interesting measurement is how
much reordering is worth *inside each build* (1 rank, 1M):

| | reorder 0 | reorder 5 | reordering is worth |
|---|---:|---:|---:|
| base | 8.063 | 6.453 | **1.610 s (20.0%)** |
| swap | 8.314 | 7.324 | **0.990 s (11.9%)** |
| base, Coll only | 3.191 | 2.310 | **0.881 s** |
| swap, Coll only | 3.297 | 2.953 | **0.344 s** |

With reordering **off** — where the swap never executes — the two builds are within ~3%. With it
on, the swap build recovers only 39% of the locality benefit the baseline gets in the collide
kernel. The reordered data is correct (bitwise identical) but the kernels are not exploiting it as
well.

A bisect settles what it is *not*: a probe build carrying the new `k_sorted` member and the full
recompile, with only the `deep_copy` restored, lands exactly on baseline (loop 6.519 against
6.709). So this is the swap itself, not code layout or the added member.

**The mechanism is not established.** Two allocations holding byte-identical data, both written
sequentially, should not differ. The leading suspect is cache/page aliasing — `OnePart` is 96
bytes, so conflict-miss behaviour at that stride depends on a buffer's base-address alignment, and
that would be a constant per-buffer penalty, which matches the observation that the cost does not
scale with how often the swap runs. That is untested. The next diagnostic would be to report both
buffers' base addresses and re-run with the spare buffer deliberately re-aligned.

## Recommendation

**Do not land as-is.** It helps small problems and hurts the large ones that matter. The
`deep_copy` is genuinely redundant work, so the win is real and worth recovering — but only once
the Move/Coll penalty is understood, or by a route that avoids a second buffer entirely (an
in-place cycle permutation, which is what the currently-disabled `FIXEDMEMORY` scheme attempts).

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
