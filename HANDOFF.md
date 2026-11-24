# Handoff notes: KOKKOS single and mixed precision

Work branch `claude/intelligent-sagan-dwvbb5`, [PR #45](https://github.com/stanmoore1/sparta/pull/45) (the same notes are posted there as a comment). This file describes the state as of `a55be49a0`; update it as the work moves on. The PR description has the design and the verification results; this file covers what is left, how to do it, and what was learned along the way. The developer documentation of the precision tools is `tools/kokkos_prec/README.md`.

## Next task: merge the latest `master`

`master` is 31 commits ahead and the PR does not merge cleanly: 30 conflicts (`git merge-tree --write-tree --name-only HEAD origin/master` lists them). Merge, do not rebase: the PR is open.

1. **Docs (5 conflicts).** Master moved the manual to Sphinx: `doc/*.txt` and `doc/*.html` are gone, and the sources are `doc/src/*.rst`. Accept the deletions.
   - Git sees `Section_start.txt` and `package.txt` as renamed to their `.rst` files. `package.rst` conflicts.
   - `Section_start.rst` **auto-merges cleanly but takes in this branch's txt2html markup** (`:pre`, `"text"_link`). Rewrite that hunk in reStructuredText; don't trust the clean merge.

   Port this branch's doc changes into:
   - `doc/src/Section_accelerate.rst`:
     - the KOKKOS "Precision" section, which replaces "Currently, there are no precision options…" (around line 873 on master);
     - its "Restrictions";
     - the CPU comm NOTE, which now gives the real defaults instead of recommending "classic".
   - `doc/src/Section_start.rst`: the mixed precision CUDA build example, and `-h` reporting the KOKKOS precision.
   - `doc/src/package.rst`:
     - `comm threaded` is the CPU default in mixed/single;
     - the paragraph on why;
     - `react/extra` keeps at least max(8, count/2) spare per-cell entries.

   The text to port is in `git show HEAD:doc/Section_accelerate.txt` etc. Build the manual with master's `doc/Makefile` to check it.
2. **KOKKOS code (25 conflicts).** Master merged a large bug-fix series (`8b35b2e85`, `831abe581`, `cba4a3e01`, ...) touching 94 KOKKOS files. Keep master's logic and this branch's precision types. Then rerun the conversion on the merged code, since master's new device code is still plain `double`:
   ```
   python3 tools/kokkos_prec/gen_kk_structs.py
   python3 tools/kokkos_prec/kk_prec_convert.py          # dry run, review the diff
   python3 tools/kokkos_prec/kk_prec_convert.py --apply
   python3 tools/kokkos_prec/kk_prec_audit.py -v
   ```
   Then build mixed and single with `-DCMAKE_CXX_FLAGS='-g -O3 -Wdouble-promotion'`, and run `kk_prec_audit.py --promotions build.log --mode mixed|single`. Update `promotions_allowlist.json` (`--update-allowlist`) only after reviewing each new promotion.
3. **Overlaps with master to resolve on purpose:**
   - **`react/extra`:** master fixed the same inversion independently (`if (!sparta->kokkos->react_retry_flag)`). Keep this branch's `cellcount_react()` per-cell headroom on top. 10% alone still overflows the `chem` example for some seeds.
   - **Seeds:** master still seeds all 15 KOKKOS random pools with `12345 + comm->me`. Keep this branch's `kk_pool_seed(random)` lines (`#ifndef SPARTA_KOKKOS_EXACT`, placed next to each `rand_pool.init(random)`), and add one to any pool master introduced.
   - **Collision velocities:** in `collide_vss_kokkos.cpp/.h`, keep `kk_cm_velocity()`, `kk_split_velocity()` and `kk_sin_from_cos()` wherever master's version computes the centre of mass or post-collision velocities. They are bit-identical to the original expressions in double and remove a momentum/energy drift in float.
   - **Geometry and comm:** check master's changes to `geometry_kokkos.h` (the axisymmetric quadratics here are solved in double) and to `comm_kokkos.cpp` (EXACT migrate) against this branch's `comm_serial = tdual_particle_1d::NEED_TRANSFORM ? 0 : 1` CPU default in `kokkos.cpp`.
4. **Gold logs.** Master reblessed examples logs. The CI `kokkos-precision` job compares against them statistically, so expect to look at any flag it raises (see the ensembles note below).

## Verification after the merge

Build each precision from scratch with:
```
cmake -C cmake/presets/kokkos_mpi_only.cmake -DKOKKOS_PREC=double|mixed|single \
      -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF -DPKG_FFT=ON \
      -DSPARTA_ENABLE_TESTING=ON -DSPARTA_SPA_ARGS="-k on -sf kk" cmake
```
Also build the merged master itself (double) as the reference.
- **Full examples, 1 and 4 ranks.** Run `ctest -j4 --timeout 2400`, then `compare_logs.py --stats MASTER_BUILD/examples NEW_BUILD/examples --pattern 'log.mpi_*'`.
  - ctest's own pass/fail is meaningless here: it diffs against the gold logs exactly, which reduced precision never matches.
  - Check that no run aborted and that no log has NaN/Inf.
  - Non-EXACT double results are no longer bit-identical to master, because this branch's seed fix changes the random streams. Bit-identity only holds against a double build of this branch from before a change. `SPARTA_KOKKOS_EXACT` builds still reproduce the non-KOKKOS gold logs (CI `mpi-kokkos-exact`).
- **A stats flag is not yet a bug.** The reference is one run, and the in-run standard error understates seed-to-seed variance. Before acting on a flag, run `tools/kokkos_prec/ensemble_test/run_ensembles.py --spa double=... --spa mixed=... --seeds 12 dir/in.NAME`, or compare against the non-KOKKOS spread over seeds. On the last round, `emit.surf.mflow`, `adapt.static` (4 ranks) and `torque` (4 ranks) were flagged and were seed noise.
- **Per-precision tests:**
  - `conservation_test/run_conservation.py --spa "mpirun -np 2 BIN -k on -sf kk" --prec P` (in CI)
  - `feature_test/run_features.py --spa BIN` (in CI)
  - `restart_test/run_restart.py --spa double=BIN_D --spa mixed=BIN_M --spa single=BIN_S`
  - `conversion_test/` (by hand)
  - `ctest -R KokkosTransformView`
- **Not rerun since the seed, collision-rounding and `react/extra` fixes:**
  - OpenMP (CI example set with `-k on t 2`);
  - ASan/UBSan with `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON` over all examples;
  - the 8-seed ensembles over all 16 default cases;
  - CPU timing (the two double-precision lines in the collision kernel cost a little in float).

  Rerun them after the merge.

## Open items not related to the merge

- **GPU:** never compiled for CUDA or HIP. This is the production blocker. Run all three precisions on hardware through the example comparison and the conservation test. Things that can only fail there:
  - device lambdas in templates;
  - TransformView paths with separate host and device memory spaces;
  - float atomics;
  - transfer cost.
- **Coverage:** 53% of KOKKOS device lines (gcov, mixed). Device region kernels and the vibmode custom update are not exercised by any input.
- **Residual rounding bias:** about 1e-10 per collision for exact 1:2 mass ratios (N/N2). It is documented in the README and well inside the conservation tolerance.
- **Pre-existing on master, not fixed here:**
  - UBSan `qsort(NULL,0)` at `marching_cubes.cpp:934`;
  - disabled examples: the custom restarts need companion files, jagged.3d needs sdata files, and cylinder is too long.

## Conventions and gotchas

- **Commits:** every commit ends with the Claude trailers plus `Co-authored-by: Stan Moore <stanmoore1@gmail.com>`. No model names in commits or the PR. Push to `claude/intelligent-sagan-dwvbb5`.
- **Error messages:** `-screen none` hides SPARTA's error message. When a run fails, rerun with the screen on.
- **Testing machine:** the cloud container has 4 cores and no GPU. Long suites hit the background-job time limit, so run ctest in halves (`-I 1,124`, `-I 125,247`).
- **CI `mpi` job:** red on `2b2f1313c` only because the GitHub runner was shut down mid-run (every test that finished had passed). Later pushes re-run it.

