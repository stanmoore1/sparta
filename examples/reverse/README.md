# Detailed-balance reverse reactions — PROTOTYPE (issue #472)

This example demonstrates a prototype for deriving **backward reaction rates**
from forward rates using the principle of detailed balance, instead of
requiring an independently-fitted Arrhenius expression for each reverse
reaction. It addresses the enhancement requested in issue #472, following
Bird (1994) §6.6 and Boyd & Schwartzentruber (2017) §7.5.2–7.5.3.

## What SPARTA does today

In the TCE model every reaction — forward or reverse — is an independent
entry in the reaction file with its own Arrhenius triple `(A, b, Ea)`. A
reverse reaction is obtained only by supplying a separate fit for it (see the
forward/reverse pairs in `data/air.tce`). A single Arrhenius form cannot
exactly reproduce `k_b(T) = k_f(T) / K_eq(T)`, because `K_eq` carries the
non-Arrhenius temperature dependence of the partition-function ratios, so
independently-fit forward/backward rates do not satisfy detailed balance and
the gas does not relax to the correct chemical equilibrium at high `T`.

## What this prototype adds

A new reaction **style `B`** (Arrhenius Backward). A `B` reaction is paired at
initialization with the forward `A` reaction whose reactants/products are its
products/reactants. With `dHf` = the forward reaction energy (coeff C5):

- backward activation energy: `Ea_b = Ea_f + dHf`
- backward reaction energy:   `dHr_b = -dHf`
- backward temperature exponent and effective DOF inherited from the forward
- backward prefactor:         `A_b(T) = A_f · q_reactants,f(T) / q_products,f(T)`

The equilibrium-constant exponential cancels against the shifted activation
energy, leaving only the **partition-function ratio**, which is evaluated each
timestep at a representative **grid-cell translational temperature**. So the
reverse rate adapts to the local nonequilibrium state, exactly as the issue
requested ("Arrhenius parameters derived from forward reactions + cell-averaged
temperature + partition functions").

Partition functions used: translational `(2πmkT/h²)^{3/2}`, rotational
(rigid rotor, linear), vibrational (harmonic oscillator), electronic (ground
state, g=1).

## Running

```
../../src/spa_serial -in in.reverse
```

Both reactions fire in the tallies; the exothermic reverse reaction is more
frequent than the endothermic forward, as expected:

```
reaction N2 + O --> NO + N: 347
reaction NO + N --> N2 + O: 876
```

## Validation

The derived backward rate lands within a factor of ~5–10 of the independent
literature fit for the same reverse reaction in `data/air.tce`
(`NO + N --> N2 + O`, `A 0.0 0.0 4.059e-12 -1.359 5.175e-19`) over
5000–30000 K. The derived reverse activation energy (0) and reaction energy
(+5.175e-19) match the `air.tce` fit exactly; the remaining prefactor gap is
attributable to the prototype limitations below.

## Prototype limitations (future work for a production version)

- Only **exchange** reactions are supported. Dissociation↔recombination
  additionally requires the third-body number density that SPARTA already
  handles specially for recombination.
- **Rotational** partition functions need a rotational temperature (rotational
  data file) and a symmetry number; they default to 1 (σ=1) otherwise. This is
  the largest source of the prefactor discrepancy above.
- **Electronic** partition functions use a ground-state degeneracy of 1; the
  `species` file stores no electronic-level data.
- The backward **temperature exponent** is inherited from the forward reaction
  rather than refit to `k_f/K_eq`.
- The **cell temperature** is a translational temperature; a chemistry-relevant
  effective temperature could include internal modes.
- The **KOKKOS** accelerated styles do not yet implement this path.

## Files changed for the prototype

- `src/react.h`, `src/react.cpp` — `React::tgas`, `React::reverse_active`
- `src/react_bird.h`, `src/react_bird.cpp` — `B` style parsing; forward/reverse
  pairing and backward-coefficient seeding in `init()`
- `src/react_tce.h`, `src/react_tce.cpp` — partition functions, runtime
  partition-ratio scaling of the reverse prefactor in `attempt()`
- `src/collide.h`, `src/collide.cpp` — per-cell `cell_temperature()` feeding
  `React::tgas`
- `doc/react.txt` — documentation of the `B` style
