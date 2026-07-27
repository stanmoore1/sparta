# Round 9 — stepping back: the five best remaining ideas, and what happened

Eight rounds of measurement had produced a long list of things that do not work
and a short list of things that do. This round took stock, ranked what was left
by measured evidence times implementability, and tried the top five.

Three landed. One is quantified but out of scope. One remains the large item it
has been since round 5.

## The list, and what each returned

| # | idea | why it was on the list | result |
|---|---|---|---|
| 1 | Fuse the per-cell count into `Update::move` | `sort_reorder`'s first act is a full streaming read of the 96 MB particle array to extract one `int` from each 96-byte record — and `move` has just computed that `int` | **1.045x**, bitwise identical |
| 2 | Re-tune the reorder period, which #1 moves | reordering got cheaper, so the optimum should shift; reordering every step also deletes `Particle::sort()` and the standalone collide pass outright | **1.037x**, free |
| 3 | Hoist the plane-rejection test into the mover's surface loop | round 7 measured 1.16x for an AABB prefilter on `in.sphere`; SPARTA calls an out-of-line `line_tri_intersect` per triangle whose *first* act is that rejection | **1.13x on in.sphere**, bitwise identical |
| 4 | Shrink the 96-byte particle record | the largest contained lever short of SoA | **1.17x measured**, not landed — see below |
| 5 | SoA particle storage | ~1.9x on in.collide (rounds 5-7), boundary cost measured and cheap | still a multi-week change; not attempted |

## 1. Counting during the move

`Particle::sort_reorder()` began with

```cpp
for (int i = 0; i < nlocal; i++) cinfo[particles[i].icell].count++;
```

At 1M particles that is a 96 MB streaming read whose entire yield is one integer
per record. But `Update::move` has just computed each particle's destination
cell. Tallying it there costs one increment into a 400 KB array — which stays in
L2, and whose accesses are nearly sequential because the particles are in cell
order — and the pass disappears.

The counts are used only when they can be trusted: `Particle::cellcount_usable()`
requires that the number counted equals `nlocal`, which fails safe if anything
added, received, cloned or discarded particles between the move and the sort. On
any mismatch the original counting loop runs. Counting is enabled only on steps
that will actually reorder, which is why `Update::run` now decides
`reorder_flag` before the move rather than after it.

Interleaved A/B (this machine drifts several percent over minutes, enough to
invent or hide an effect this size, so runs alternate and each binary is scored
by its minimum over the same stretch):

| configuration | before | after | |
|---|---:|---:|---:|
| reorder 1, `collide/every` 1 | 6.436 s | 6.154 s | **1.046x** |
| reorder 4, `collide/every` 8 | 3.804 s | 3.644 s | **1.044x** |

Bitwise identical on `in.collide` at 1M and on all six `regress.sh` cases.

## 2. The reorder optimum moved

Round 2 measured the best reorder period as 2. Making the reorder cheaper should
move that, and it does — reordering *every* step now wins, because it also means
the fused sort+collide path runs on every step and `Particle::sort()` and the
separate collision pass over the whole array never run at all:

| reorder period at `collide/every` 1 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| loop (s) | **5.154** | 5.346 | 6.577 | 6.855 |

**1.037x**, for a changed default in the benchmark input and nothing else. With
`collide/every 8` the optimum is reorder 4 — i.e. reorder on every collide step,
which is the same rule.

## 3. The surface check: hoisting a test, not adding a data structure

Round 7 measured an AABB prefilter worth 1.16x on `in.sphere` in the mini-app and
declined to implement it, because doing it properly needs a per-surface bounding
box maintained across `read_surf`, `move_surf`, `fix ablate`, implicit-surf grid
adaptation and distributed surfs.

That turned out to be the wrong framing. `Geometry::line_tri_intersect` already
*starts* with a rejection test — whether the segment lies wholly on one side of
the triangle's plane — and at 2500 m/s and a 1e-5 timestep a particle covers a
few percent of a cell, so nearly every triangle is rejected there. The cost was
never the test; it was that the test lives behind an out-of-line call in another
translation unit, paid 2.4 times per particle-move.

Hoisting those six lines into the mover's loop in `update.cpp`, with identical
arithmetic in identical order:

| in.sphere, 8 interleaved reps | loop (s) | |
|---|---:|---:|
| original SPARTA | 0.3128 | 1.000x |
| rounds 1-2, 8 + counting during move | 0.3116 | 1.004x |
| + inline plane rejection | **0.2767** | **1.130x** |

**1.13x, bitwise identical**, with no new array and nothing to keep in sync. It
also shows that everything landed before this round was *neutral* on `in.sphere`
(1.004x) — those changes all target the `optmove` fast path, which a
surface-bearing problem never enters.

## 4. The record size: measured, and left alone

The remaining traffic is two 96 MB passes per step — the move and the scatter —
so the record's size is the last lever short of SoA. Round 3 estimated the
elasticity at 0.38 by padding 96 -> 128 bytes, predicting only ~6% for a 16-byte
saving.

**That estimate was low, and the padding method is why.** 128 bytes is exactly
two cache lines, so padding upward buys perfect alignment that partly offsets the
extra traffic. Measuring in the other direction gives a very different answer.
Building with the four trailing fields (`erot`, `evib`, `dtremain`, `weight`) at
80 bytes instead of 96:

| in.collide 1M, reorder 1 | loop (s) | |
|---|---:|---:|
| 96-byte record | 5.976 | 1.000x |
| 80-byte record | **5.096** | **1.173x** |

The mechanism is stride, not precision. The fields the mover touches — the four
`int`s, `x` and `v` — are the first 64 bytes. At a 96-byte stride two records
span three cache lines, so the mover touches 1.5 lines per particle; at an
80-byte stride four records span five lines, 1.25 per particle. 1.5/1.25 = 1.2,
and the scatter copies 1.2x fewer bytes as well.

**This is reported as a measurement of the record-size elasticity, not as a
proposed change.** The build that produced it used single precision for those
four fields, which is out of scope — the code stays FP64 and the change is
reverted. The number transfers to the FP64 route, which is to relocate two cold
doubles into side arrays: an 80-byte FP64 record has the same stride, the same
lines-per-record and the same scatter copy size, and for a monatomic unweighted
run the side arrays are touched on ~1% of moves or never. Slightly optimistic,
therefore, but the right order.

It was not landed because that relocation reaches into the emit fixes, the
surface reaction models, the restart format and the Kokkos package — a real
refactor, not a session's work. **It is now the best-quantified unimplemented
item in the whole study: ~1.17x for 96 -> 80 bytes, and more for 64.**

For the record, the single-precision build was checked before being discarded,
and nothing was wrong with it numerically: bit-identical on both benchmarks
(argon is monatomic, so `erot`/`evib` are always zero and `weight` is unused);
statistically indistinguishable on the three cases that do carry internal energy
(interspecies `c_temp` 271.77 ± 1.83 against 272.24 ± 1.40 over five seeds;
chemistry 14163 ± 135 against 14210 ± 212); and no energy drift over 20000 steps
of a closed box. It is out of scope regardless.

## 5. SoA — unchanged

Still ~1.9x on `in.collide` and ~1.2x on `in.sphere` by the round 5-7 mini-apps,
still the largest single item, still a change to the particle subsystem rather
than to three kernels. Nothing new to add this round.

## Where things stand

**`in.collide`, 1M particles, FP64 throughout, interleaved measurement:**

| | loop (s) | speedup |
|---|---:|---:|
| original SPARTA, `optmove` + reorder 2 | 7.944 | 1.00x |
| all optimizations, reorder 1 | 4.755 | **1.67x** |
| + `collide/every 8`, reorder 4 | **3.494** | **2.27x** |

**`in.sphere`:** 0.3128 s -> 0.2767 s, **1.13x**, bitwise identical.

Statistics across the three `in.collide` configurations — temperature exactly
conserved, collision rate per unit physical time flat:

| configuration | T (K) | coll/step | attempts/step |
|---|---:|---:|---:|
| original, reorder 2 | 272.96696 | 70527 | 94536 |
| optimized, reorder 1 | 272.96696 | 70583 | 94583 |
| optimized, reorder 4, K=8 | 272.96696 | 70257 | 93850 |

On exactness: rounds 1, 2, 8 and 9's code changes are each bitwise identical at a
fixed configuration, and all six `regress.sh` cases match the original binary
bitwise. The 1.67x column is not bitwise identical to the 1.00x row, because it
uses a different reorder period, and reordering every step means the fused
sort+collide path runs every step — which round 2 documented as changing the
order cells are collided in, and therefore the random number stream. That is
statistically identical, not bit-for-bit, as the table above shows.

| the five ideas | outcome |
|---|---|
| count during the move | **landed**, 1.045x, bitwise |
| reorder every step | **landed**, 1.037x, input default |
| inline plane rejection | **landed**, 1.13x on in.sphere, bitwise |
| 80-byte record | **1.17x measured**; FP64 route is a real refactor, not attempted |
| SoA storage | ~1.9x estimated; unchanged from round 5 |
