# KOKKOS precision tools

Scripts used to convert the SPARTA KOKKOS package to precision-generic code,
so it can be built in double, mixed, or single precision with

    cmake -D PKG_KOKKOS=ON -D KOKKOS_PREC=double|mixed|single ...

The approach follows the LAMMPS KOKKOS package (`-D KOKKOS_PREC=...`).

## Precision model

`src/KOKKOS/kokkos_type.h` defines three floating point types:

| typedef        | double | mixed  | single | used for |
|----------------|--------|--------|--------|----------|
| `KK_FLOAT`     | double | float  | float  | velocities, rotational/vibrational energies, collision, reaction and surface collision arithmetic |
| `KK_POS_FLOAT` | double | double | float  | particle positions, remaining timestep, grid cell and surface element coordinates, particle move and geometry kernels |
| `KK_ACC_FLOAT` | double | double | double | per-grid/per-surf tallies and other accumulations, and the per-cell statistics computed from them |

`KK_ACC_FLOAT` stays double in a single precision build.  SPARTA uses SI
units, where a molecular mass is ~1e-26 kg: the per-cell moments computed
from mass weighted sums, e.g. (sum m v)^2 / sum m in a thermal temperature
or (sum m v)^3 / (sum m)^2 in `compute eflux/grid`, underflow single
precision.  For the same reason masses and reduced masses are kept double
in device code (`keep_double_identifiers`).

Global reductions (temperature, `compute reduce`, ...) always reduce into
`double`.  Random numbers are always generated in double precision and each
draw is narrowed to the precision of the expression that uses it.

The host (non-KOKKOS) classes are never changed and always hold double
precision data.  Kokkos data that the host classes also see is held in a
`TransformView` (`kokkos_type.h`): a DualView whose device side (and Kokkos
host mirror) is KK precision and whose host "legacy" side is double.  The
conversion happens on `sync_host()`/`sync_device()`.  In a double precision
build a `TransformView` is exactly a `DualView` and costs nothing.

The per-particle, per-grid-cell and per-surf structs (`Particle::OnePart`,
`Grid::ChildCell`, `Grid::SplitInfo`, `Grid::ParentCell`, `Surf::Line`,
`Surf::Tri`) have KK precision device copies (`OnePartKK`, ...) generated
into `src/KOKKOS/kokkos_structs.h`.

## Files

| file | purpose |
|------|---------|
| `precision_map.json`  | the single source of truth: struct field precisions, type renames, which identifiers are positions / accumulators / kept double, per-file overrides |
| `gen_kk_structs.py`   | regenerates `src/KOKKOS/kokkos_structs.h` from the host struct definitions; fails if a host struct gains a double field with no mapping. `--check` verifies the file is current |
| `kk_prec_convert.py`  | the bulk rewriter (dry run by default, `--apply` to write); writes `review.md` (not tracked), the list of constructs to review by hand |
| `kk_prec_audit.py`    | linter: fails if device code contains an unjustified `double`, a bare C math call, an unwrapped literal, or legacy types. Run it on new KOKKOS code |
| `compare_logs.py`     | compares thermo output of two sets of log files, exactly (double build vs. original) or statistically (`--stats`, reduced precision vs. double) |
| `run_examples.py`     | runs a list of example inputs (default `ci_examples.txt`) and collects their logs, used by the CI job |
| `conversion_test/`    | exact check of the host/device precision conversion of particle data (`check_conversion.py`) |
| `transformview_test/` | unit test of the TransformView sync state machine (ctest `KokkosTransformView`) |
| `kkprec.py`           | shared lexer: comments/strings, brace matching, device regions |

## Conventions for KOKKOS code

Device code is the signature and body of `KOKKOS_INLINE_FUNCTION`,
`KOKKOS_FUNCTION` and `KOKKOS_LAMBDA`/`SPARTA_LAMBDA`.  In device code:

- declare floating point variables as `KK_FLOAT`, `KK_POS_FLOAT` or
  `KK_ACC_FLOAT`, never `double`, unless double precision is needed; then
  keep `double` and put a `// KK_DOUBLE: reason` comment on the line
- write literals that take part in arithmetic as
  `static_cast<KK_FLOAT>(0.5)`, not `0.5` (which promotes the expression to
  double) and not `0.5f`
- call `Kokkos::sqrt()`, `Kokkos::exp()`, ... not the C functions
- narrow random draws: `static_cast<KK_FLOAT>(rand_gen.drand())`
- choose tolerances by precision with `kk_eps<T>(eps_double, eps_float)`
- floating point Kokkos arrays are `DAT::ttransform_kkfloat_*` /
  `DAT::ttransform_kkacc_*`; their `view_host()` is double

## Reduced precision pitfalls found in SPARTA

These are handled in the converted code; keep them in mind for new KOKKOS
code:

- *Underflow in SI units.*  Products of masses (~1e-26 kg), e.g. a reduced
  mass or the square of a mass weighted sum, underflow single precision.
  Masses are kept double in device code and per-cell statistics use
  `KK_ACC_FLOAT`.
- *Integer truncation.*  `int j = n*drand()` can give `j == n` when the
  draw is rounded to float; expressions converted to an integer are kept
  double.
- *Byte counts.*  `memcpy(v,ip->v,3*sizeof(double))` overruns a float
  array; use the KK type in `sizeof`.
- *Round trips.*  Host data converted to float on the device and back must
  not lose its exact double value if the device did not change it (host
  geometry compares cell corners exactly); `kk_convert()` keeps it.
- *Degenerate moves.*  With float positions `x + dt*v` can round to `x`: a
  particle with `dtremain = 0` on a reflecting face, or re-entering through
  a periodic face just outside the box, looped forever in the move kernel.
  Both are guarded, for float positions only, so a double precision build
  is unchanged.
- *Tolerances.*  Absolute tolerances tuned for double round-off are chosen
  by precision with `kk_eps<T>()`.

## Rerunning the conversion

The conversion is idempotent, so it can be rerun on converted code, e.g.
after merging new KOKKOS code:

    python3 tools/kokkos_prec/gen_kk_structs.py
    python3 tools/kokkos_prec/kk_prec_convert.py            # review the diff
    python3 tools/kokkos_prec/kk_prec_convert.py --apply
    python3 tools/kokkos_prec/kk_prec_audit.py -v

Classification of a `double` declaration in device code is by the declared
identifier: names listed in `pos_identifiers` become `KK_POS_FLOAT`, in
`acc_identifiers` `KK_ACC_FLOAT`, in `keep_double_identifiers` stay
`double` (and get a `// KK_DOUBLE` comment), anything else becomes
`KK_FLOAT`.  `file_default_class` and `file_identifier_class` override this
per file.  A declaration of several variables of different classes is split
into one declaration per variable.  Files in `skip_files` are converted by
hand.

## Testing

A double precision build must reproduce the original code exactly:

    compare_logs.py REF_LOG_DIR NEW_LOG_DIR

Mixed and single precision runs diverge from double precision ones after a
few steps, as any round-off change does in DSMC, so they are compared on
time averages:

    compare_logs.py --stats DOUBLE_LOG_DIR MIXED_LOG_DIR

The conversion of particle data between the double precision host and the
reduced precision device is checked exactly with `conversion_test/`, see
`check_conversion.py`.

The TransformView sync state machine (every host/Kokkos host/device
transition, resize, copies, the struct conversion, the abort on concurrent
modification) is unit tested by `transformview_test/test_transformview.cpp`,
built and run as the ctest `KokkosTransformView` when the KOKKOS package and
`SPARTA_ENABLE_TESTING` are enabled:

    ctest -R KokkosTransformView --output-on-failure

A reduced precision build converts Kokkos data on every host/device sync
of a TransformView (whole allocated extent), so styles that run on the
host every step (non-KOKKOS fixes or computes, frequent dumps) cost more
than in a double precision build, which has no conversion at all.
