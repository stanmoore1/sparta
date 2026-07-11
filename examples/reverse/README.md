# Detailed-balance reverse reactions (issue #472)

This example demonstrates **backward reaction rates derived from forward
rates by detailed balance**, instead of requiring an independently-fitted
Arrhenius expression for each reverse reaction.  It implements the
enhancement requested in issue #472, following Bird (1994) Sec. 6.6 and
Boyd & Schwartzentruber (2017) Secs. 7.5.2-7.5.3.

## Why

In the TCE model every reaction - forward or reverse - is normally an
independent entry in the reaction file with its own Arrhenius triple
`(A, b, Ea)`.  A single Arrhenius form cannot exactly reproduce
`k_b(T) = k_f(T) / K_eq(T)`, because `K_eq` carries the non-Arrhenius
temperature dependence of the partition-function ratios; independently
fitted forward/backward rates therefore do not satisfy detailed balance,
and a reacting gas does not relax to the correct chemical equilibrium
composition.

## The B reaction style

A reaction with style **`B`** in the reaction file gets its rate derived
at run time from its forward partner (see `doc/react.txt` for the full
description).  Two kinds of pairs are supported:

- **exchange <-> exchange**: the forward partner is the exchange reaction
  with reactants and products swapped
- **recombination <-> dissociation**: `A + B -> AB + M` pairs with the
  per-partner dissociation `AB + M -> A + B + M` with the same explicit
  third body `M`

At initialization the backward activation energy (`Ea_B = Ea_F + dHf`),
reaction energy (`-dHf`), and effective DOF are seeded from the forward
line; the five numeric coefficients on a `B` line are placeholders.

An **exchange** pair is then given a temperature-free detailed-balance
table: the ratio of the two channels' microcanonical densities of
states at matched total collision energy (built with the same
machinery as the microcanonical TCE energy factors), so each
collision's backward probability follows from the forward reaction by
energy-resolved microscopic reversibility.  Its thermal average
reproduces `k_b(T) = k_f(T)/K_eq(T)` at every temperature
simultaneously, and no temperature is evaluated at run time.

A **recombination** pair scales the forward prefactor by the
partition-function ratio and the forward temperature exponent at the
local grid-cell translational temperature:

```
k_b(T_cell) = A_F * T_cell^b_F
              * q_reactants,F(T_cell) / q_products,F(T_cell)
              * exp(-(Ea_F + dHf)/kT_cell)
```

Partition functions include translational, rigid-rotor rotational (with
the symmetry number sigma from the species `rotfile`), harmonic-oscillator
vibrational, and electronic (from the species `elecfile` ladder) factors.
For a recombination the ratio carries one net translational factor (units
of volume), converting the m^3/s dissociation prefactor into the m^6/s
recombination prefactor; the third body is a spectator and cancels.

Reverse reactions require `react tce` (with the microcanonical
`partial_energy no` coupling recommended) and are not available for
ionization, whose reverse rate depends on the electron temperature and
must be supplied explicitly.  The KOKKOS `tce/kk` style produces
bit-for-bit identical results (SPARTA_KOKKOS_EXACT).

## Files

- `in.reverse` - demonstration deck: thermal air box (N2/O/NO/N) at
  20000 K with all internal modes discrete, running the two pairs in
  `rev.tce`
- `rev.tce` - exchange pair `N2 + O <-> NO + N` and
  dissociation/recombination pair `N2 + N <-> N + N + N`; forward
  parameters from `data/air.tce`
- `rev_exch.tce` - exchange pair only (used by the relaxation check)
- `in.reverse_rate` - frozen-composition reservoir
  (`compute_chem_rates yes`) for measuring rates; variables `T`, `RB`,
  `NRHO`, `FNUM`
- `in.reverse_eq` - closed reacting box for the equilibrium-relaxation
  check; variables `T`, `FN2`, `FO`, `FNO`, `FN`
- `air.rot` - rotational temperatures + symmetry numbers (N2: 2.88 K,
  sigma 2; O2: 2.07 K, sigma 2; NO: 2.44 K, sigma 1)
- `air.elec` - NIST-based low-lying electronic levels for N2, O, N, NO
- `validate_reverse.py` - quantitative validation battery (below)

## Running

```
../../src/spa_serial -in in.reverse
```

Both exchange directions fire in the tallies; the exothermic reverse is
more frequent than the endothermic forward, as expected:

```
reaction N2 + O --> NO + N: 197
reaction NO + N --> N2 + O: 506
reaction N2 + N --> N + N + N: 1004
```

Three-body recombination events are too rare to appear at this density
in a 1000-step demonstration (the rate scales as density cubed); the
validation battery measures them in a 200x denser reservoir.

## Validation

`validate_reverse.py` recomputes the partition functions and equilibrium
constants independently of the SPARTA implementation (from the same data
files) and checks the measured rates against them:

```
python3 validate_reverse.py --exe ../../src/spa_serial
```

1. **Exchange detailed balance**: the forward/backward tally ratio from
   frozen reservoirs matches the analytic `K_eq(T)` to 0.2% at 15000 K,
   1.3% at 10000 K, and 9.4% at 8000 K (within the tally statistics).
2. **Literature comparison**: the derived backward rate for
   `NO + N -> N2 + O` lies within a factor of 1.33-1.39 of the
   independently fitted literature rate for the same reaction in
   `data/air.tce` over 8000-15000 K - i.e. the derived rate agrees with
   the fitted one to well within the scatter of published rate models.
   The residual is the fit pair's own inconsistency: the equilibrium
   constant implied by the two published fits is x1.38-1.40 larger than
   the statistical-mechanics K_eq, so a thermodynamically consistent
   backward rate cannot (and should not) match the fitted one exactly.
3. **Recombination detailed balance**: in a dense reservoir the
   dissociation/recombination tally ratio times the atom number density
   matches the analytic volumetric `K_eq` to 3.8% (5.0% statistics).
4. **Equilibrium relaxation**: a closed reacting box initialized on the
   pure-reactant side and on the pure-product side relaxes toward the
   same analytic equilibrium composition (NO fraction 0.411) from both
   directions.
5. **Input sanity**: the electronic ground-state degeneracies and
   symmetry numbers that dominate the `K_eq` prefactor are present in the
   data files (for this reaction they contribute factors of ~1.8 and 2).
6. **Error paths**: a `B` reaction without a forward partner, a `B`
   recombination with a wildcard third body, and a `B` reaction under a
   QK style all abort with the intended error messages.

Accurate reverse rates need accurate partition functions: supply a
`rotfile` with symmetry numbers and an `elecfile` with the low-lying
electronic levels for every species that participates in a reverse
reaction.  Omitting them silently biases `K_eq` (here by the factors
noted in check 5).
