# PR #446 (Electronic Excitation): Physics Review and Recommendations

**Re:** [sparta/sparta PR #446](https://github.com/sparta/sparta/pull/446) — "Implemented electronic excitation"
**Scope:** The TCE/chemistry coupling question raised by @aborner1 (Mar 14, 2024), the effective-DOF problem described in the Jun 10 / Sep 13, 2024 comments.

---

## TL;DR

1. **The electronic relaxation model is sound.** It conserves energy, satisfies detailed balance within spin classes, and equilibrates correctly — your relaxation plots demonstrate this. Nothing below touches that machinery.
2. **The TCE coupling is not consistent, and cannot be made consistent with a per-state constant DOF.** The negative DOF values you found (the "−300") are not a numerical artifact to be worked around — they are derivable from the level structure and are the model telling you the per-particle electronic DOF is mathematically undefined. Details in §3.
3. **@aborner1/Higdon are right for the rate data everyone actually uses; @mgallis is right in principle.** The resolution is that the question is really about *what the input Arrhenius coefficients represent* (§4). For equilibrium-calibrated rate sets (Park, Gupta, etc.), electronic energy must be **excluded** from the TCE collision energy or the rate is double-counted and provably overpredicted.
4. **The affected code path is narrower than it looks**: only the `react_modify partial_energy no` branch of `react_tce.cpp` and the screening in `react_tce_qk.cpp`. The default rDOF path never feeds `eelec` into the reaction probability. The surgical fix is ~5 lines (§5, Stage 1).
5. A path to recover *nonequilibrium* electronic-state sensitivity without breaking equilibrium rates — the thing exclusion gives up — is sketched in §5 (Stage 2), with the fully state-specific approach as the long-term endpoint (Stage 3).

---

## 1. What the evidence already shows

- **Higdon Fig. 7.2** (posted by @aborner1): for O₂ + N₂ ⇌ O + O + N₂, "TCE: Rot.+Vib.+Electronic" sits systematically **above** the Arrhenius line, while "Rotation only" and "Rot+Vib" track it. The relative gap is largest at the low-temperature end of the figure and narrows toward 20,000 K. This is an *equilibrium* test — the cleanest possible isolation of the artifact.
- **Your May 2, 2024 species plot** (0D dry air from 25,000 K, `partial_energy no`): solid (electronic-in) vs dashed (electronic-out) agree on the major neutrals but the solid lines over-produce exactly **NO, N₂⁺, N⁺** — the high-threshold ionization/exchange channels, i.e., the reactions most sensitive to extra collision energy. You noted the same trend from 5,000 K to 50,000 K. This is the in-SPARTA reproduction of Higdon's result.
- **Your Sep 13, 2024 DOF experiment**: varying the user DOF from 0 to 2 changed results "on the same order" as the discrepancy itself — i.e., the knob cannot close the gap. §3 explains why no constant value can.

---

## 2. Why TCE works, and the consistency requirement

SPARTA's TCE probability (`react_tce.cpp`) is

```
P ∝ C1 · Γ(z + 5/2 − ω) / Γ(z + η + 3/2) · (E_c − E_a)^(η−1+ω) · (1 − E_a/E_c)^(z + 3/2 − ω)
```

with `C1, C2` computed from the Arrhenius coefficients in `react_bird.cpp` (Bird 1994, p. 127). The Γ-functions and exponents are not decoration: they are chosen so that averaging `P` over the **equilibrium distribution of E_c** — which, for translational plus *continuous* internal modes with average DOF `z`, is a Gamma distribution `f(E_c) ∝ E_c^(z+1/2−ω) e^(−E_c/kT)` — analytically reproduces `k(T) = A T^η e^(−E_a/kT)` at every temperature, with temperature-independent constants.

Here `z` is the pair-average internal DOF (`(ζ_i + ζ_j)/2` in the code), so the pair's internal energy is Gamma-distributed with shape `z`, the collision-weighted translational energy has shape `5/2 − ω`, and `E_c` is Gamma with shape `z + 5/2 − ω`, i.e. density `∝ E_c^(z+3/2−ω) e^(−E_c/kT)`.

I verified the integral against the code's coefficient definitions: averaging the `P` above over that Gamma distribution gives `⟨P⟩ = C1 · e^(−E_a/kT) · (kT)^(η−1+ω)` — the `z`-dependence cancels exactly (that is what the Γ-prefactor is for) — and multiplying by the VSS collision rate `∝ T^(1−ω)` recovers `A·T^η·e^(−E_a/kT)` identically.

This imposes a consistency requirement: **the `z` appearing in `P`'s Γ-prefactor and exponent must match the actual distribution of the energy fed into `E_c`.** If energy is added to `E_c` whose equilibrium distribution is *not* the assumed Gamma form, the cancellation breaks and the equilibrium rate departs from the input Arrhenius rate. This is why SPARTA works hard on the vibrational term (the `2·i·ln(1+1/i)` instantaneous DOF, `newtonTvib`): vibrational quanta are closely spaced relative to kT, so the SHO distribution is close enough to the continuous form for the patch to work.

---

## 3. Why the electronic mode breaks it — two independent failures

### 3(a) Distribution-shape failure (the dominant one — this is Higdon's result)

Electronic levels are *widely* spaced: N₂'s first excited state is ~72,000 K (≈ 6 eV). At any realistic temperature the per-particle electronic energy distribution is a **spike at zero plus rare, far-out spikes** — nothing resembling the smooth Gamma distribution the calibration assumes. The consequence is an exponential-compensation effect:

- A particle in state `i` has Boltzmann-suppressed population `p_i ∝ g_i·e^(−ε_i/kT)`.
- But its electronic energy effectively lowers the reaction threshold by `ε_i`, boosting its conditional rate by `≈ e^(+ε_i/kT)` (for `ε_i < E_a`; even more dramatically when `ε_i > E_a`, where the channel becomes barrierless).
- The exponentials cancel: `p_i·k_i ∝ g_i·e^(−E_a/kT)` — **every accessible excited level contributes at the same exponential order as the ground channel**, weighted by its degeneracy and an algebraic factor.

So `k_model(T) = Σ_i p_i k_i = k_ground·(1 + Σ_{i>0} O(g_i/g_0)·[algebraic])`, strictly above the calibrated rate. The *relative* overprediction is largest at **low** temperature — the `e^(+ε/kT)` threshold-lowering is strongest when `kT ≪ E_a` — and narrows as T rises, exactly the trend visible in Higdon's Fig. 7.2. Note the degeneracies involved: in your own `air.elec`, the excited N₂ states carry g = 27, 135, 45, 153 against g₀ = 9 — the excited channels are heavily weighted once this mechanism turns on.

**Numerical verification (SPARTA itself, this branch).** A frozen-composition O₂/N₂ reservoir (`react_modify compute_chem_rates yes partial_energy no`, your `air.tce` O₂+N₂ dissociation, your reacting-example electronic data with dof = 0) measures the equilibrium rate directly against the input Arrhenius coefficient:

| T (K)  | rot+vib only, k/k_Arr | rot+vib+electronic, k/k_Arr | electronic enhancement |
|--------|----------------------|------------------------------|------------------------|
| 10,000 | 1.27                 | 1.65                         | ×1.29                  |
| 15,000 | 1.15                 | 1.46                         | ×1.27                  |
| 20,000 | 1.09                 | 1.28                         | ×1.18                  |

The electronic mode overpredicts the equilibrium rate by 18–29% beyond the rot+vib baseline for this reaction, largest at low T — Higdon's result, reproduced with this PR's own inputs. (These runs are scripted in `verification/electronic/run_tests.py` on the review branch.)

No choice of `z` fixes this: `z` rescales the Γ-normalization and the `(1−E_a/E_c)` exponent, which can suppress the excited channels at *one* temperature, but the required suppression is temperature-dependent while the per-state `z` is a constant — so it cannot hold across the Arrhenius range. This is exactly why your DOF-knob experiment (0 → 2) barely moved the answer, and it is Higdon's p. 175 conclusion verbatim: the approach "cannot be extended to electronic excitation modeling since the observed errors are much greater for the much more widely spaced quantum electronic levels… ionization rates were severely overpredicted."

### 3(b) The negative-DOF pathology is derivable from your own input file

The instantaneous-DOF construction inverts the mean-energy relation `E(T)` to get a temperature, then sets `z = 2E/(kT)`. For a **finite-level** system, `E(T)` saturates as T → ∞ at the degeneracy-weighted mean of the levels:

```
E_∞ = Σ_i g_i ε_i / Σ_i g_i
```

Any particle in a state with `ε_i > E_∞` has *no positive temperature* that reproduces its energy — the inversion demands a negative (population-inversion) temperature, and `z = 2E/(kT)` goes negative.

Running the numbers on the N₂ model in `examples/relax_electronic_reacting/air.elec` (levels 0, 71600, 85293, 97478, 127998 K; degeneracies 9, 27, 135, 45, 153):

```
E_∞ = (27·71600 + 135·85293 + 45·97478 + 153·127998) / 369  ≈  101,400 K
```

The top state (127,998 K) exceeds E_∞, so it *must* produce a negative temperature and negative DOF. This even reproduces the detailed pattern you reported for your 8-state structure: a state *just above* E_∞ requires T → −∞, giving a **small** negative `z = 2E/(kT)` (your 7th state), while a state near the top of the ladder requires T → 0⁻, making `|z|` blow up (your 8th state's "−300"). The `bisectTelec` negative-temperature branch in `particle.cpp` is handling a symptom of the same thing. **The per-particle electronic DOF is not hard to compute — it is undefined.** That is strong evidence the framework, not the implementation, is the problem.

### A clarifying observation: what inclusion vs. exclusion actually asserts

Suppose you include the pair electronic energy ε in `E_c` *and* raise the threshold by the same amount, on the grounds that a barrier tabulated for ground-state reactants is referenced to the ground state. Then the excess energy — which controls whether reaction is possible at all, and the `(E_c − E_a)^(η−1+ω)` factor — is unchanged:

```
(E_cont + ε) − (E_a + ε) = E_cont − E_a
```

(The remaining `(1 − E_a/E_c)^(z+3/2−ω)` factor is not exactly invariant — its denominator becomes `E_cont + ε` — so full equivalence would also require redoing the normalization, which is precisely the recalibration a consistent per-state model must do anyway; see Stage 3.)

The point of the observation: **excluding electronic energy is not "ignoring the physics."** It encodes the statement that tabulated `E_a` is ground-state-referenced and electronic energy does not count toward crossing it. Conversely, including ε with an *unchanged* threshold implicitly asserts that every quantum of electronic energy lowers the barrier one-for-one — a strong state-specific claim that should be made deliberately (per-state thresholds and re-normalized rates, §5 Stage 3), not as a side effect of energy bookkeeping.

---

## 4. Resolving the @aborner1 / @mgallis disagreement

Both positions are correct in their own domain, and the equilibrium-consistency requirement (§2) says exactly when each applies:

- **@mgallis is right in principle**: *if* the calibration integral were re-derived using the actual discrete electronic partition function (replacing the Γ-functions with explicit sums over the real levels), the equilibrium average would reproduce Arrhenius by construction. "If the calibration included the electronic energy it must reproduce the correct rates" — true, but the current code does not do that calibration, and no closed form exists for it.
- **@aborner1/Higdon are right in practice**: the existing machinery keeps the continuous-Gamma normalization while adding a discrete energy stream, so the identity breaks and the rate overpredicts — with **no possible constant-DOF repair** (§3a).
- The "SPARTA back-calculates the TCE coefficients from the input Arrhenius rates, so it should self-correct" argument misses that the back-calculation (`react_bird.cpp:248–254`) still *assumes the continuous Gamma distribution*. It cannot correct for a distribution shape it never sees.

**The sharpest framing: this is a data-semantics question.** Standard air-chemistry rate sets (Park, Gupta, …) are equilibrium/total rates — the equilibrium excited-state populations and their enhanced reactivity are *already inside* the measured A, η, E_a. Feeding E_elec back into E_c double-counts that contribution. Only if the input rates were true ground-state-specific rates would including ε (with per-state thresholds) be correct — and then the total equilibrium rate *should* exceed the ground-state rate, by construction rather than by accident.

---

## 5. Recommendations, in order of value-per-effort

### Stage 1 — Restore equilibrium consistency (small, surgical; recommended for this PR)

The damage is confined to the `partial_energy no` branch. The default rDOF path (`partialEnergy = 1`, the default set in `react.cpp:39`) uses `ecc = pre_etrans + pre_erot·z/rotdof` and never sees `eelec`; the `pre_eelec` flowing into `pre_etotal`/`post_etotal` is pure energy bookkeeping and **must stay** (post-reaction redistribution needs it to conserve energy).

Concrete changes:

```cpp
// react_tce.cpp, attempt() — partial_energy no branch:
} else {
   ecc = pre_etotal - pre_eelec;   // exclude electronic energy from reaction energy
   z = pre_ave_rotdof;
}

// react_tce.cpp:147-159 — delete the block that adds the per-state electronic dof:
//   if (collide->elecstyle == DISCRETE) { ... z += 0.5*(zi + zj); }

// react_tce_qk.cpp:83 — same exclusion in the energetic-impossibility screen:
ecc = pre_etotal - pre_eelec;
```

Everything else stays: `pre_etotal`/`post_etotal` keep `pre_eelec`, `EEXCHANGE_ReactingEDisposal` still zeroes and redistributes electronic energy, and the whole relaxation model is untouched. The per-state `dof` column in the elec file can then be dropped (or kept but documented as unused/experimental). If desired, keep the old behavior behind an explicit `react_modify` flag defaulting to **excluded**.

What you give up: electronically hot particles no longer react faster. At equilibrium you lose nothing (that enhancement is already in the input rates); out of equilibrium, see Stage 2.

### Stage 2 — Recover nonequilibrium sensitivity without breaking equilibrium (moderate effort)

The physically real thing exclusion loses: an electronically hot but translationally cold gas (shock precursor, e-impact-pumped regions) should react faster than its heavy-particle temperatures imply. The clean way to get this is **redistribution, not augmentation** — the electronic analog of the vibrationally-favored dissociation (VFD) model:

```
P_i(E_c) = P_TCE(E_c, no elec) · w(ε_i) / ⟨w⟩(T_elec)

⟨w⟩(T) = Σ_i g_i w(ε_i) e^(−ε_i/kT) / Q_elec(T)     (a cheap sum over a handful of states)
```

with `w(ε)` a favoring function (e.g., `(1 + ε/E_a)^φ`, φ = 0 recovering exclusion). At equilibrium the bias cancels **by construction** and the total rate stays pinned to Arrhenius; out of equilibrium, excited particles are preferentially selected to react. The normalization needs the cell electronic temperature — which this PR already computes (`compute telec/grid` / `bisectTelec`), and cell-level input to reaction decisions has precedent in SPARTA (recombination density). This gives the state sensitivity @mgallis's intuition asks for without the overprediction.

### Stage 3 — State-specific electronic chemistry (larger effort; the physical endpoint)

Per-state reaction channels with thresholds `E_a,i = E_a,ground − ε_i`, calibrated so the *ground-state* rate matches ground-state data — the Liechty & Lewis treatment (electronic-level transitions and ionization linked to level structure in the QK framework). This is the only approach that genuinely captures excited-state ionization (which is dominated by stepwise excitation → ionization from high levels), and it slots most naturally into SPARTA's QK path, which already reasons in quantum levels. It requires state-resolved rate data and reinterpreting rate inputs as ground-state-specific — a future PR, not this one.

---

## 6. Minor items and questions

- **Double-appearing relaxation probability — verified correct, worth documenting.** In the non-reacting path, relaxation is gated by `phi(current state)` and then `select_elec_state` weights candidates by `degen·phi(candidate)`, so φ enters twice. I initially flagged this as a possible bug, but working through the flux balance shows it is the *right* construction: the transition probability is `T(i→f) = φ_i · g_f φ_f X_f / Z(E)` with `X_f = (1 − ε_f/E)^(3/2−ω)`, so `T(i→f)/T(f→i) = (g_f X_f)/(g_i X_i)` — the φ's cancel symmetrically and detailed balance holds exactly against the microcanonical `n_s ∝ g_s X_s` (within each spin class; `Z_i = Z_f` for same-spin states). Notably, the "obvious" alternative — gate on φ, then sample the *unweighted* `g·X` target — would **violate** detailed balance for state-dependent φ, equilibrating to `n_s ∝ g_s X_s/φ_s` instead of Boltzmann. So the double-φ is load-bearing. Two follow-ups: (a) a code comment/doc note explaining this would prevent a future "cleanup" from breaking it; (b) because transition rates scale as `φ_i·φ_f`, the input values are not literally per-state relaxation probabilities in the usual single-φ sense — users calibrating against measured relaxation times or state-to-state rates should be told how the inputs map onto effective rates.
- **`relax_electronic_mode` in the reacting path is called as `(p, p)`** (`EEXCHANGE_ReactingEDisposal`), so it uses `params[isp][isp].omega` rather than the `aveomega` used for the vibrational redistribution in the same function — a small internal inconsistency in the post-reaction LB exponent.
- **`bisectTelec`**: uses `throw 0;` for the unreachable-energy corner (should be `error->one`), and the loop tolerance `(T_high − T_low) > 0.01` is an absolute 0.01 K while the comment says "accurate to 1%" — prefer a relative tolerance.
- **`Particle::ielec()`**: `while (ran > cumulative_probabilities[ielec]) ++ielec;` can overrun the last state on floating-point rounding (probabilities summing to slightly < 1). Clamp to `nelecstate − 1`.

---

## 7. Validation checklist (acceptance tests for Stage 1)

1. **Equilibrium reservoir rate sweep** (the Higdon Fig. 7.2 test): 0D reservoir, `computeChemRates`-style rate extraction per reaction, 5,000–30,000 K. Pass criterion: "Rot+Vib+Electronic" collapses onto the Arrhenius line the way "Rot+Vib" does.
2. **Relaxation-only equilibrium** (already passing): Boltzmann electronic populations at several temperatures; equipartition with trans/rot/vib.
3. **Reacting 0D air case**: the solid/dashed gap on N⁺, N₂⁺, NO in your May 2024 species plot should close to statistical noise after Stage 1.
4. **Parity**: Kokkos vs non-Kokkos, and restart round-trip with elec custom data.

---

## References

- G. A. Bird, *Molecular Gas Dynamics and the Direct Simulation of Gas Flows*, 1994 — TCE derivation, p. 127 (the source of `C1`, `C2` in `react_bird.cpp`).
- K. J. Higdon, Ph.D. dissertation — Fig. 7.2 and p. 175: electronic energy in TCE overpredicts Arrhenius; ionization "severely overpredicted" (quoted by @aborner1 in the PR thread).
- Bird, "[Total collision energy model: 4 decades and going strong](https://pubs.aip.org/aip/pof/article/31/7/076101/1075618)," *Phys. Fluids* 31, 076101 (2019) — review of the TCE equilibrium-consistency construction.
- Gallis et al., "[Assessment of Reaction-Rate Predictions of a Collision-Energy Approach](https://www.osti.gov/servlets/purl/1141390)" — TCE rate-recovery analysis.
- Liechty & Lewis, "[Electronic Energy Level Transition and Ionization Following the Quantum-Kinetic Chemistry Model](https://arc.aiaa.org/doi/abs/10.2514/1.48826)," *J. Spacecraft & Rockets* (2011) — the state-specific endpoint (Stage 3).
- Haas & Boyd, vibrationally-favored dissociation — the redistribution-with-normalization pattern that Stage 2 adapts to discrete electronic states.
- "[Modeling of the electronic excited states in high-temperature flows](https://pubs.aip.org/aip/pof/article/36/8/086112/3306920)," *Phys. Fluids* 36, 086112 (2024) — recent survey of electronic-state DSMC modeling.
