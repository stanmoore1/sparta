# Round 8 — collision sub-cycling, and why raising `dt` usually beats it

The brief for this round was "radical changes that preserve statistics but are not
necessarily bitwise equivalent, on both benchmarks." The change tried was
**collision sub-cycling**: run the collision operator every `K` steps with `K`
times the NTC attempt count, and move every step as usual. It is implemented as
`global collide/every K` (default 1, bitwise identical).

It works, on both benchmarks. It is also **dominated by simply raising the
timestep** wherever raising the timestep is legal — which on one of the two
benchmarks it very much is. That comparison is the main result of this round,
and it is the reason `collide/every` is shipped as an opt-in option with a
stated validity condition rather than recommended.

Along the way this round found and corrected **two measurement artifacts of my
own making**, both documented below, because both were on track to produce a
confidently wrong conclusion.

---

## 1. What sub-cycling is

The NTC attempt count for a cell is

```
nattempt = 0.5 * N * (N-1) * vremax * fnum * dt / volume
```

which is **linear in `dt`**. So running collisions every `K`th step with `dt`
replaced by `K*dt` produces the same number of attempts per unit physical time,
and — since the acceptance probability `vr*sigma/vremax` is untouched — the same
mean collision rate. Sorting is only needed on steps that collide, so it is
skipped too.

`src/update.cpp`:

```cpp
int docollide = (ntimestep % collide_every == 0);
if (collide) collide->nstep_collide = collide_every;
int reorder_flag = (reorder_period && docollide && ntimestep % reorder_period == 0);
```

and `CollideVSS` uses `update->dt * nstep_collide` in `attempt_collision` and in
`collide_cell_kernel`. That is the whole change, ~30 lines.

## 2. Measured: `in.collide`, 1M particles

`optmove yes`, `particle/reorder 2`, best of 3, single pinned core. `coll/step`
is `Ncoll` from the collide call divided by `K`, i.e. the collision rate per unit
physical time — the statistic that must not move.

| K | loop (s) | speedup | Move | Coll | Sort | T (K) | coll/step | vs K=1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.296 | 1.00x | 1.95 | 1.35 | 2.92 | 272.96696 | 70609 | — |
| 2 | 4.966 | 1.27x | 1.90 | 0 | 2.98 | 272.96696 | 70472 | -0.19% |
| 4 | 4.408 | 1.43x | 1.98 | 0 | 2.34 | 272.96696 | 70419 | -0.27% |
| 8 | **4.210** | **1.50x** | 2.12 | 0 | 1.99 | 272.96696 | 70257 | **-0.50%** |
| 16 | 4.056 | 1.55x | 2.26 | 0 | 1.69 | 272.96696 | 69724 | -1.25% |

`Coll` reads 0 for K>1 because with `reorder 2` every collide step is also a
reorder step, so the fused sort+collide path runs and its collide work is billed
to `Sort` (round 2's fusion). Move + Coll + Sort is the honest total.

Temperature is *exactly* conserved at every K — which is expected and therefore
uninformative, since VSS collisions conserve energy exactly and the box is
closed. **The collision rate is the statistic with teeth**, and it holds to 0.5%
at K=8, drifting to 1.25% at K=16. K=8 is the operating point.

A reorder sweep at K=8 confirms reordering should happen on every collide step:

| reorder period | 0 | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| loop (s) at K=8 | 6.306 | 4.163 | 4.139 | 4.081 | 4.133 | 4.771 |

Periods 1, 2, 4 and 8 all reorder on every collide step and land within 1.4% of
each other, which also fixes the run-to-run noise floor. Period 16 reorders on
every *other* collide step and loses; period 0 loses badly.

Most of the win is indirect: `Sort` falls from 2.92 s to 1.99 s because cell
contiguity is only needed on steps that collide.

## 3. Measured: `in.sphere`

10x10x10, ~10K particles, 1000-step benchmark run after 1000 of equilibration.
Rates are the whole-run summary counters, not the stats table (see §5).

| K | loop (s) | speedup | Move | Coll | Sort | surf coll/particle/step | gas coll/particle/step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.328 | 1.00x | 0.241 | 0.0576 | 0.0174 | 1.400e-04 | 1.186e-02 |
| 2 | 0.293 | 1.12x | 0.237 | 0.0370 | 0.0080 | 1.422e-04 | 1.188e-02 |
| 4 | 0.276 | 1.19x | 0.235 | 0.0264 | 0.0040 | 1.366e-04 | 1.177e-02 |
| 8 | 0.262 | **1.25x** | 0.229 | 0.0204 | 0.0020 | 1.390e-04 | 1.183e-02 |

Statistics hold flat. The speedup is small because `in.sphere` spends 73% of its
time in the mover doing ray-triangle intersection, and sub-cycling does not touch
the mover — it can only compress the 22% that is collide+sort. Amdahl, exactly as
expected.

## 4. The comparison that matters: just raise `dt`

Sub-cycling coarsens the collision operator's time resolution while keeping the
mover fine. The obvious alternative is to coarsen *everything* — raise the global
timestep and run fewer steps for the same physical time. Cost is then measured
per unit **physical time**, i.e. `loop_time / K`.

**`in.collide`, 1M** — 100 steps at `K*dt`, rates normalised per unit physical time:

| dt x | loop (s) | cost/phys-time | speedup | cell crossings/particle/step | coll rate vs 1x |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.54 | 6.54 | 1.00x | 0.381 | — |
| 2 | 8.90 | 4.45 | **1.47x** | 0.762 | -0.02% |
| 4 | 13.77 | 3.44 | 1.90x | **1.523** | +0.01% |
| 8 | 22.43 | 2.80 | 2.33x | **3.046** | -0.03% |

**`in.sphere`** — 1000 steps at `K*dt`:

| dt x | loop (s) | cost/phys-time | speedup | cell crossings/particle/step | surf coll/pt | gas coll/pt |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.334 | 0.334 | 1.00x | 0.026 | 1.406e-04 | 1.232e-02 |
| 2 | 0.380 | 0.190 | 1.76x | 0.052 | 1.470e-04 | 1.241e-02 |
| 4 | 0.457 | 0.114 | 2.92x | 0.104 | 1.488e-04 | 1.222e-02 |
| 8 | 0.615 | **0.077** | **4.34x** | 0.206 | 1.387e-04 | 1.198e-02 |
| 16 | 0.914 | 0.057 | 5.85x | 0.403 | 1.330e-04 | 1.177e-02 |
| 32 | 1.429 | 0.045 | 7.48x | 0.773 | 1.165e-04 (-17%) | 1.125e-02 (-9%) |

Cell crossings per particle per step is `Cell-touches/particle/step - 1`, measured
by SPARTA itself on the general (non-`optmove`) mover. It is the CFL-type
constraint that governs how far `dt` can legally be raised: **DSMC requires a
particle not to skip past cells in one step**, so this number must stay well
below 1.

That single column explains everything:

- **`in.collide` is already near its move limit.** At the shipped `dt` a particle
  crosses 0.38 cells per step. Doubling `dt` reaches 0.76; quadrupling reaches
  1.52 and is no longer a valid DSMC discretisation, whatever the equilibrium
  collision rate says. So `dt` buys about **1.47x**, and sub-cycling's **1.50x**
  is the better of the two — because the constraint that binds `in.collide` is
  the *move*, not the collisions (mean time between collisions is ~14 steps).
- **`in.sphere`'s timestep is ~38x below its move limit.** 0.026 crossings per
  step, and the mean time between gas collisions is ~84 steps. Both constraints
  have enormous slack, and raising `dt` 8x is worth **4.34x** with all rates
  within 3%, against sub-cycling's 1.25x. Past 8x the rates start to drift (-17%
  on surface collisions at 32x, where the per-step displacement approaches the
  size of a surface triangle), so 8x is the honest limit.

**So: raise `dt` when the mover has slack; sub-cycle when it does not.** For
these two benchmarks the answer is different in each case, and the deciding
number is one that SPARTA already prints.

The caveat on the `dt` route, stated plainly: it is a change to the *input deck*,
not to the code. A benchmark's timestep is part of the benchmark, and reporting
"4.34x on in.sphere" by raising `dt` would be changing the problem. It is
reported here because it is the correct engineering answer to "how do I make this
run faster", and because it bounds what sub-cycling can be worth.

## 5. Two artifacts I generated and had to correct

Both nearly produced a confident wrong answer, and both are worth recording.

**(a) The stats table aliased with the vremax reset period.** My first `in.sphere`
sub-cycling run appeared to show the collision rate exploding 30x — 4 collisions
per step at K=1 against 121 at K=8 — and I began writing that up as "sub-cycling
breaks in.sphere". It does not. `bench/in.sphere` has `collide_modify vremax 100`
and `stats 100`, so **the stats table samples `Ncoll` on exactly the timesteps
where `vremax` has just been reset to its thermal initial value** — the single
worst step of each 100-step cycle. The whole-run counter
`Collisions/particle/step` says 1.19e-2 either way, i.e. ~119 collisions/step,
and is flat across K. Every rate in this document is now taken from the whole-run
summary counters rather than the periodically-sampled stats table.

I also briefly concluded from this that `in.sphere` ships with a starved
`vremax`. **That was wrong too** and is retracted: disabling the reset entirely
changes the whole-run rate by only 3.6% (1.186e-2 -> 1.232e-2). The reset is fine;
only my sampling of it was not.

**(b) A real bug in the sub-cycling patch, found by chasing (a).**
`Collide::collisions_pre()` tested `ntimestep == vre_next` and then did
`vre_next += vre_every`. Under sub-cycling that function is no longer called every
step, so when `vre_next` is not a multiple of `collide_every` the test steps
straight over it and **`vremax` is never reset again for the rest of the run**.
That showed up as a +3.4% collision-rate drift at K=8. Fixed in `src/collide.cpp`
to test `>=` and recompute `vre_next` from the current timestep; the rate is now
flat to 0.6% across all K on `in.sphere`, as the table in §3 shows.

## 6. Validity, stated honestly

Sub-cycling applies `K*dt` worth of collisions using **one instantaneous snapshot**
of each cell's population. That is exact only where the population is
statistically stationary over `K` steps. It is *not* free in general:

- **Transients are coarsened.** Relaxation toward equilibrium now happens in
  `K`-step jumps. `in.collide` is the degenerate easy case — spatially uniform and
  already at equilibrium, so there is nothing for the coarsening to corrupt, which
  is exactly why it holds to 0.5%. A startup transient, a moving shock, or a
  relaxation measurement would need a smaller `K`, and `K` should be validated
  against the quantity being measured, not against the equilibrium rate.
- **Particles cross cells during the `K` steps**, so the population collided is
  not quite the population that occupied the cell over the interval. This is why
  the residual drift is negative (fewer collisions) and grows with `K`.
- **The requirement is `K*dt` < mean collision time.** `in.collide`: ~14 steps
  between collisions, so K=8 sits at 0.57 of that — marginal, and K=16's 1.25%
  drift is that constraint becoming visible. `in.sphere`: ~84 steps, so K=8 is
  comfortable. A denser gas needs a smaller `K`, and the check is cheap:
  `Collisions/particle/step` is printed at the end of every run.

`global collide/every 1` is the default and is **bitwise identical** to the
previous code — verified on `in.collide` at 1M (step/np/nattempt/ncoll/c_temp all
exact) and on `bench/in.sphere` against the unmodified baseline binary, plus all
six `regress.sh` cases (`collide`, `collideInterspecies`, `free`, `sphere`,
`ambi`, `chem`) identical.

## 7. Standing position after eight rounds

SPARTA performance, bitwise-identical, from rounds 1-2: **7.34 s -> 6.26 s at 1M
particles (1.17x)**. Round 8 adds an opt-in, statistics-preserving
**1.50x on top** of that at `collide/every 8` for `in.collide`, so 7.34 s -> 4.21 s
(**1.74x** cumulative), and 1.25x for `in.sphere`.

| question | answer | evidence |
|---|---|---|
| Collision sub-cycling? | **1.50x on in.collide** (K=8, rate within 0.5%); 1.25x on in.sphere | §2, §3 |
| Sub-cycle or raise `dt`? | **raise `dt` if the mover has slack** — 4.34x on in.sphere vs 1.25x; sub-cycle if not — 1.50x on in.collide vs 1.47x | §4 |
| How do you tell which? | `Cell-touches/particle/step - 1`; must stay well below 1 | §4 |
| Does sub-cycling corrupt transients? | **yes** — it is exact only for a locally stationary population | §6 |
| AoS, SoA or AoSoA? | SoA; ~1.9x on in.collide, ~1.2x on in.sphere | rounds 5-7 |
| Materialization boundary cost? | free, with and without surfaces | rounds 6-7 |
| Surface bbox prefilter? | 1.16x on in.sphere, 99.5% fewer intersection tests; SPARTA has none | round 7 |
| Vectorized mover? | 1.46x on move, but only with SoA | round 6 |
| SoA grid cells, 64 B record, mesh-free, tiling, buckets? | no / ~1.15x / no / no / no | rounds 2-4 |

The two unimplemented items with the best measured evidence remain **SoA particle
storage** (~1.9x on in.collide, boundary cost measured and cheap) and the
**surface AABB prefilter** (1.16x on in.sphere). Neither was landed here; both are
larger than a session.
