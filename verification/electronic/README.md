# Verification suite for discrete electronic excitation

Regression and physics-verification tests for SPARTA's discrete electronic
excitation model (`collide_modify electronic discrete` + `fix elecmode` +
`species ... elecfile`), introduced in sparta/sparta PR #446.

## Requirements

A SPARTA executable (any build). Python 3 (standard library only).
The MPI mode additionally needs an `mpirun` and an MPI-enabled build.

## Usage

Run the whole physics suite against a serial build:

    python3 run_tests.py --exe ../../src/spa_serial

Run a subset:

    python3 run_tests.py --exe ../../src/spa_serial --tests boltzmann,spin

**Exact-output snapshot regression** — pin each deck's full stats table and
reaction tallies against committed reference files in `snapshots/`. This is
the primary refactor tripwire: any change to observable output fails it.

    python3 run_tests.py --exe ../../src/spa_serial --snapshot   # compare
    python3 run_tests.py --exe ../../src/spa_serial --bless      # (re)write refs

**Cross-build parity** — run every deck with two executables and require the
stats tables and reaction tallies to match bit-for-bit (CPU-time column
ignored). Use to verify a Kokkos build compiled with `-DSPARTA_KOKKOS_EXACT`
against the plain CPU code:

    python3 run_tests.py --exe ../../src/spa_serial \
        --exe2 ../../build-kk/src/spa_kokkos_mpi_only --exe2-args "-k on -sf kk"

**Parallel correctness** — run the physics suite under `mpirun -np N` (DSMC is
deterministic per rank count but not bit-identical across rank counts, so this
checks the tolerance-based physics tests still pass in parallel, plus a
same-rank run-to-run determinism check):

    python3 run_tests.py --exe ../../src/spa_mpi --mpi-np 4

The exit code is the number of failed tests. Work directories are removed on
success; pass `--keep` to keep them.

## Tests

| test          | what it pins down |
|---------------|-------------------|
| `boltzmann`   | An N₂ reservoir with all modes at 20,000 K stays there: total energy (kinetic + electronic) conserved to round-off, and electronic state populations remain Boltzmann-distributed (each within 5σ). Detailed-balance test of the relaxation algorithm, including its state-dependent relaxation-number weighting. |
| `equilibration` | Hot translation + cold electronic mode relax to the common temperature predicted by the analytic energy balance `(3/2)kT + <E_elec>(T) = (3/2)k·25000`, energy conserved throughout. |
| `spin`        | With spin conservation enforced, singlet→singlet transitions occur but the triplet states stay **exactly** empty. Pins allowed/forbidden transitions. |
| `latespecies` | Species commands after an `elecfile` species, growing the species list past its initial allocation. Memory-reallocation regression. |
| `rates`       | Frozen-composition (`compute_chem_rates yes`) equilibrium TCE rate vs the input Arrhenius rate at 10/15/20 kK: rot+vib tracks Arrhenius; adding the electronic mode **overpredicts** (documented, cf. Higdon dissertation Fig. 7.2), largest at low T; a fractional per-state dof (0.9) lowers the rate vs dof 0 (parser regression). |
| `atom`        | Electronic-only relaxation for **atomic** O (zero rot/vib dof): exercises the `elec_exchange` path where `ave_dof == 0`. Excited states populate to the Boltzmann distribution, energy conserved. |
| `specrel`     | **Species-specific** relaxation numbers: N₂ relaxes ~8× faster against a fast partner than a slow one (the per-partner `species_rel` branch of `get_elec_phi`), while the slow partner still relaxes via the `default_rel` fallback. |
| `relaxrate`   | The relaxation **rate** (not just equilibrium) tracks the input collision number: a high relaxation probability drains electronic energy faster early on, and both reach the same equilibrium. Guards the transient dynamics that equilibrium tests miss. |
| `reactions`   | Exact energy conservation across **real** dissociation (composition changes): the drop in kinetic+rot+vib+elec energy equals reactions × dissociation energy to <2%. Exercises `EEXCHANGE_ReactingEDisposal`, `relax_electronic_mode(reacting)`, `zero_elec` on products, and the 3-product path. |
| `restart`     | Electronic custom data (`elecstate`, `eelec`) survives a `write_restart`/`read_restart` round-trip exactly. (This is what caught that `fix_elecmode` did not follow the restart-safe `find_custom`-first pattern.) |
| `telec`       | `compute telec/grid` recovers the initialization electronic temperature (8,000 and 15,000 K) to within 5%. Exercises `bisectTelec` / `elec_energy` / `electronic_distribution_func`. |

Reference ratios from the `rates` test (k/k_Arrhenius, this branch):

    T=10000K  rot+vib 1.27   rot+vib+elec 1.65
    T=15000K  rot+vib 1.15   rot+vib+elec 1.46
    T=20000K  rot+vib 1.09   rot+vib+elec 1.28

## Note for a future refactor

Several assertions **pin current behavior**, not idealized physics — in
particular `rates` pins the fact that electronic energy in the `partial_energy
no` TCE collision energy overpredicts the equilibrium rate. If the model is
later changed to exclude electronic energy from the TCE reaction energy (the
subject of the PR #446 review — see the accompanying memo), the `rates`
assertions and the `snapshots/` references should be re-blessed accordingly.
The snapshot references and the example gold logs
(`examples/relax_electronic*/log.*`) must be regenerated whenever a change is
intended to alter output.

## Files

    run_tests.py       driver (all assertions live here)
    snapshots/         committed exact-output references for --snapshot
    in.*               SPARTA input decks (see header comments)
    air.species/vss    species and VSS data (from examples/relax_electronic_reacting)
    airx.vss           air.vss plus dummy species for the latespecies test
    extra.species      dummy inert species for the latespecies test
    partners.species   inert collision partners F1/F2 for the specrel test
    n2boltz.elec       real N₂ levels, single spin class, fast relaxation
    airspin.elec       N₂ with singlet/triplet structure (spin test)
    o_atom.elec        atomic O electronic levels (atom test)
    n2_specrel.elec    N₂ with per-partner relaxation numbers (specrel test)
    n2_rate.elec.tmpl  N₂ two-level template, PHI substituted per run (relaxrate)
    o2n2.elec          N₂+O₂ electronic data, per-state TCE dof = 0
    o2n2_dof09.elec    same with per-state TCE dof = 0.9
    o2n2.tce           single reaction O2 + N2 -> O + O + N2
