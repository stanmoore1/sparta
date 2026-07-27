# CPU optimization study of `bench/in.collide`

A profile-guided optimization pass over SPARTA's serial CPU performance on the
`in.collide` DSMC benchmark: measure a baseline, profile it, quantify the remaining
headroom with a roofline, optimize, remeasure.

**Round 1: 1.20x at 1M particles**, from changes that reproduce the baseline output
bit for bit. **Round 2** diagnosed what remains: the step is limited by the size of
the 96-byte particle record, 32 bytes of which are dead for this class of run.
Shrinking it to 64 bytes measures **1.58x** on its own — more than rounds 1 and 2
combined — while every algorithmic restructuring tried (cache tiling, pass fusion,
fixed-capacity buckets, mesh-free binning, a branchless mover) is worth at most
1.26x in a microbenchmark and ~1.05x in the real code. See `ROUND2.md`.

## Read in this order

| document | what is in it |
|---|---|
| [`ROUND2.md`](ROUND2.md) | round 2: what the bottleneck actually is (the 96-byte particle record), the full design-space exploration, and why tiling / fusion / buckets / mesh-free all lose to simply shrinking the record |
| [`RESULTS.md`](RESULTS.md) | round 1: the reorder-period sweeps, every optimization and its measured effect, verification, and the summary table |
| [`PROFILE.md`](PROFILE.md) | gprof and callgrind profiles of the baseline: instruction mix, cache misses, branch behaviour |
| [`ROOFLINE.md`](ROOFLINE.md) | measured machine ceilings, per-kernel arithmetic intensity, and what the plot says about remaining headroom |

![roofline](roofline.png)

## Scripts

| script | purpose |
|---|---|
| `run_bench.sh -b BIN -s SIZE -r PERIOD -n REPS` | time one configuration; reports median, spread, and SPARTA's own section breakdown |
| `sweep_reorder.sh BIN SIZE REPS` | sweep `global particle/reorder` over 0,1,2,5,10,20,50,100 |
| `verify.sh REF NEW SIZE PERIOD` | assert two binaries produce identical per-step physics output |
| `regress.sh REF NEW` | same check across in-tree examples covering surfaces, chemistry, ambipolar, multi-group and the non-optmove move |
| `profile.sh gprof\|callgrind BIN TAG` | run a profiler and summarize |
| `build.sh FLAVOR TAG` | build a binary with a named flag set into `bin/` |
| `make_kernels.py`, `roofline.py` | compute kernel arithmetic intensities and draw the roofline |

`SIZE` is `10K`, `100K`, `1M`, `10M` (the sizes documented in `bench/README`) or an
explicit `x,y,z`.

## Microbenchmarks

`micro/` holds standalone drivers, so a variant can be tried in seconds rather than
minutes. Build any of them with `g++ -O3 -std=c++11 -o NAME NAME.cpp`
(`machine_peak` additionally wants `-march=native`).

| driver | question it answers |
|---|---|
| `machine_peak.cpp` | what are this machine's real bandwidth and FLOP ceilings, and what does a `pow`/`sqrt`/`sin` actually cost? |
| `micro_move.cpp` | hash vs flat-array cell lookup, and does the `cells[].proc` load matter? |
| `micro_sort.cpp` | `sort()`+`reorder()` vs a fused counting sort, checked for identical ordering |
| `micro_collide.cpp` | virtual vs inlined, `plist` vs contiguous, `pow` vs fast `pow` |
| `micro_pow.cpp` | `pow` throughput *and* latency, which turned out to be the distinction that mattered |
| `micro_thp.cpp` | do huge pages help the counting sort's scattered writes? |
| `micro_design.cpp` | the round-2 design space: nine ways to structure the whole timestep, from the current three passes to tiled fusion, per-cell buckets, mesh-free binning and smaller particle records |

## Benchmark input

`bench/in.collide.opt` is `bench/in.collide` plus `global optmove yes` and
`global particle/reorder ${reorder}`, with the reorder period settable from the
command line. Its default of 2 is the measured optimum after the round-2 changes
(it was 3 after round 1, and 5 before either).

```
cd bench
../src/spa_serial -var x 40 -var y 50 -var z 50 -var reorder 3 < in.collide.opt
```
