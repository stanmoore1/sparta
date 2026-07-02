# Verification suite for discrete electronic excitation

This directory contains regression and physics-verification tests for
SPARTA's discrete electronic excitation model (`collide_modify electronic
discrete` + `fix elecmode` + `species ... elecfile`), introduced in
sparta/sparta PR #446.

## Requirements

A SPARTA executable (any build). Python 3 (standard library only).

## Usage

Run the whole suite against a serial build:

    python3 run_tests.py --exe ../../src/spa_serial

Run a subset:

    python3 run_tests.py --exe ../../src/spa_serial --tests boltzmann,spin

Parity mode — run every deck with two executables and require the stats
tables and reaction tallies to match bit-for-bit (CPU-time columns are
ignored). Useful for verifying a Kokkos build compiled with
`-DSPARTA_KOKKOS_EXACT` against the plain CPU code:

    python3 run_tests.py --exe ../../src/spa_serial \
        --exe2 ../../build-kk/src/spa_kokkos_mpi_only --exe2-args "-k on -sf kk"

The exit code is the number of failed tests. Work directories are removed
on success; pass `--keep` to keep them.

## Tests

| test          | what it pins down |
|---------------|-------------------|
| `boltzmann`   | An N₂ reservoir initialized with every mode at 20,000 K stays there: total energy (kinetic + electronic) is conserved to round-off, and the electronic state populations remain Boltzmann-distributed (each state within 5 sigma of `g_i exp(-θ_i/T)/Q`). This is a detailed-balance test of the relaxation algorithm, including its state-dependent relaxation-number weighting. |
| `equilibration` | Translation starts at 25,000 K with a cold electronic mode. The gas must relax to the common temperature predicted by the analytic energy balance `(3/2)kT + <E_elec>(T) = (3/2)k·25000` (solved by the driver from the same electronic data file), with energy conserved throughout. |
| `spin`        | With spin conservation enforced (the default), particles starting in the singlet ground state may populate the singlet excited state but the triplet states must remain **exactly** empty. Pins the allowed/forbidden transition machinery. |
| `latespecies` | Defines an `elecfile` species first, then adds enough further species to force reallocation of the global species list, then collides the elec-species with a late-added partner. Regression test for the per-species electronic data (re)allocation and default relaxation/spin lookups. |
| `rates`       | Frozen-composition (`compute_chem_rates yes`) O₂/N₂ reservoir with a single dissociation reaction measures the equilibrium TCE rate against the input Arrhenius coefficient at 10,000/15,000/20,000 K. Asserts: (a) the rot+vib `partial_energy no` rate tracks Arrhenius (within 15% at ≥15,000 K; the discrete-vib TCE is known to run high at lower T, so 35% at 10,000 K); (b) adding the electronic mode **overpredicts** the rate at every temperature (documented behavior, cf. K. Higdon's dissertation Fig. 7.2 and the PR #446 discussion), most strongly at low T; (c) a fractional per-state TCE dof (0.9) lowers the rate relative to dof 0 — a parser regression test (integer truncation of the dof column would make the two runs identical). |

Reference numbers measured on the review branch (ratios k/k_Arrhenius):

    T=10000K  rot+vib 1.27   rot+vib+elec 1.65
    T=15000K  rot+vib 1.15   rot+vib+elec 1.46
    T=20000K  rot+vib 1.09   rot+vib+elec 1.28

Note that test (b) *pins the current behavior* of including electronic
energy in the `partial_energy no` collision energy. If the model is later
changed to exclude electronic energy from the TCE reaction energy (making
the equilibrium rate match Arrhenius), the assertions in `test_rates`
should be flipped accordingly — that change is the point of the review
discussion, and this test is the acceptance test for it.

## Files

    run_tests.py       driver (all assertions live here)
    in.*               SPARTA input decks (see header comments)
    air.species/vss    species and VSS data (from examples/relax_electronic_reacting)
    airx.vss           air.vss plus dummy species for the latespecies test
    extra.species      dummy inert species for the latespecies test
    n2boltz.elec       real N₂ level energies/degeneracies, single spin class,
                       fast relaxation — full-Boltzmann equilibrium expected
    airspin.elec       N₂ with singlet/triplet structure (from examples/relax_electronic)
    o2n2.elec          N₂+O₂ electronic data with per-state TCE dof = 0
    o2n2_dof09.elec    same with per-state TCE dof = 0.9
    o2n2.tce           single reaction O2 + N2 -> O + O + N2
