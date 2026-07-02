# Review of the SWS/SWPM unification plan

**Subject:** `doc/sws_swpm_unification_plan.md`, commit `94585cad` on branch
`claude/nagoya-code-review-l1e63d`
**Reviewed against:** the SWS implementation on `origin/nagoya` /
`claude/nagoya-code-review-l1e63d` (reviewed + tested SWS) and the SWPM
implementation on `claude/swpm-review-tests-b8jf0s` (reviewed + tested SWPM,
this branch)
**Reviewer constraints supplied by the user:** separate implementation files
for SWS and SWPM; minimal changes to the base collide code.

## Verdict

The plan's architecture is **correct**, and — verified against the actual
code on both branches, not just the plan text — it is the **right option**
for the stated constraints.  It has one genuine design gap (the
surface/boundary tallies do not reduce to the proposed `pweight(i)` accessor)
and a handful of required amendments listed below.  Nothing in the plan needs
to be discarded; with the amendments applied it is executable as written.

---

## 1. Claims verified as correct against the code

### 1.1 The `np_eff` / `wscale` kernel seam is real

`CollideVSS::attempt_collision_SWS(icell,np,volume,count_wi,maxwi)` on the
review branch contains three variants that differ **only** in the pair-count
factor of the standard NTC formula:

| Mode | Pair-count factor |
|---|---|
| SWS (`sws==1`) | `0.5 * count_wi * (np-1)` |
| SWSmax (`sws==2`) | `0.5 * np * maxwi * (np-1)` |
| off | `0.5 * np * (np-1)` |

One `np_eff` argument (`count_wi`, `np*maxwi`, or `np`) collapses all three
into upstream's single formula with **zero weighting branches inside the
kernel** — the plan's "values, not branches" pattern is exactly achievable.
SWPM's attempt scaling (`fnum` scaled by
`max_stochastic_weight * (1 + pre_wtf*wtf)`) folds into the same parameter,
since `nattempt` is linear in it.

Likewise `test_collision_SWS` (SWSmax acceptance) uses

```
(vre/vremax) * MAX(w_i, w_j) / maxwi        // SWS, per-species weights
```

which is shape-identical to SWPM's acceptance on this branch:

```
(vre/vremax) * MAX(isw, jsw) / max_stochastic_weight   // SWPM, per-particle weights
```

A single `wscale = 1.0` default parameter on the one upstream
`test_collision` genuinely serves baseline, SWSmax, and SWPM, and ends the
two-branch textual contention on that line.  This is the strongest part of
the plan and it checks out at the code level.

(Also verified: `recomb_density` under SWS uses `count_wi * fnum / volume`;
the plan's per-cell helper returning `count_wi`/`maxwi`/`np_eff` separately
covers this — baseline behavior is recovered because `count_wi == np` when
weights are off.)

### 1.2 The SWS-folded / SWPM-separate asymmetry is the right structure

- **SWS is a modification of the standard NTC loop** and must compose with
  near-neighbor selection, group collisions, and ambipolar — which is
  precisely why the nagoya branch carries four `_SWS` loop clones
  (`collisions_one_SWS<NEARCP>`, `collisions_group_SWS`, and the two
  ambipolar variants, ~1,700 lines).  Folding it into the baseline loops as
  runtime flags + kernel parameters + helpers (bodies in a new
  `collide_sws.cpp`) is the only structure that de-duplicates those clones
  while preserving the compositions.
- **SWPM is a structurally different loop** (split before collision,
  group/reduce epilogue per cell) that is mutually exclusive with nearcp and
  ambipolar and lives in its own files already (`collide_swpm.cpp`,
  `collide_reduce.cpp`, `fix_stochastic_weight.*`).  Keeping it as its own
  dispatch branch matches the ambipolar precedent and this branch's
  `examples/swpm/REFACTOR_ANALYSIS.md` §3.4.

Both schemes end up with separate implementation files, satisfying the
first user constraint.

**On "minimal changes to base collide":** the honest statement of the trade
is — the only way to get literally zero baseline-loop edits for SWS is to
keep the four `_SWS` loop clones (merely relocated into a separate file),
which permanently duplicates ~1,700 lines of drift-prone loop code.  That is
the unsustainable option.  The plan's ~10 guarded one-line calls per baseline
loop is the price of deleting the clones, and runtime flags checked inside
these hot loops are already the house pattern (`recombflag`, `remainflag`).
The plan makes the right call; this review endorses it explicitly against
the zero-edit alternative.

The plan's rejection of a policy/strategy object with virtual hooks is also
correct (per-collision virtual dispatch, foreign to SPARTA's explicit-loop
style, and it maps poorly onto the mirrored Kokkos loop implementations).

### 1.3 The two-site weight model matches both refactor analyses

Cell weighting stays at the per-cell normalization stage untouched (keeps
master bit-exact); SWS and SWPM unify at the in-loop tally site via the
`weight_mode` + `pweight` accessor.  This is the same seam that
`doc/sws_refactor_analysis.md` and `examples/swpm/REFACTOR_ANALYSIS.md`
arrived at independently.  The single `setup_weighting()` mutual-exclusion
model correctly replaces the scattered pairwise guards — including the
misplaced exclusivity check currently inside
`Particle::stochastic_weights()` on this branch.

### 1.4 Verification discipline

The plan adopts the bit-exact vs re-bless classification (never re-bless on
a bit-exact step), per-step gates with both gold-log suites plus both
invariant verifiers, and merge-neutrality canaries (`in.sws0.box`,
`in.swpm.periodic` byte-identical after the merge).  Step 4b (folding the
loops) is correctly identified as the highest-risk step — RNG draw-order
preservation — with an isolated-commit re-bless fallback.  Step 5 (weighted
`compute_temp` under SWS) is a real correctness fix and matches SWPM's
existing behavior.  All of this mirrors the discipline that worked for the
SWPM review and is endorsed.

---

## 2. The design gap: surf/boundary tallies do not collapse to `pweight(i)`

The plan claims *"Every compute's copied weighting block collapses to
`double w = particle->pweight(i);`"*.  This is **false for three of the
eleven files** — `compute_boundary.cpp`, `compute_surf.cpp`,
`compute_isurf_grid.cpp` — and those are also where the hardest Step-1 merge
conflicts live, because both branches rewrote the *same expressions*:

```cpp
// SWS  (nagoya review branch):
vec[k++] -= weight * (ierot*wi      + jerot*wj      - iorig->erot*worig);
// SWPM (this branch):
vec[k++] -= weight * (ierot*iswfrac + jerot*jswfrac - iorig->erot*oswfrac);
```

The math is structurally identical; the unifying abstraction in these files
is the **weight triple `(worig, wi, wj)`**, not a per-index accessor.  The
sources of the triple are irreconcilable through `pweight(i)`:

- `iorig` is a **stack copy** of the pre-interaction particle, not an entry
  in the particle array — an index-based accessor cannot address it at all.
- SWS derives all three weights from **species**
  (`specwt[origspecies]`, `specwt[ip->ispecies]`, `specwt[jp->ispecies]`),
  which is exact even when a surface reaction changes the species.
- SWPM derives `wi`/`wj` from **particle indices** into the custom weight
  vector, and has no original weight left, so `worig` (= `oswfrac`) is
  reconstructed as the average of the outgoing particles' weights — valid
  under SWPM's no-boundary-splitting assumption.

**Required amendment:** add a second accessor to the plan's Architecture §A —
e.g.

```cpp
// weight triple for surface/boundary tally callbacks
void Particle::tally_weights(const OnePart *iorig, const OnePart *ip,
                             const OnePart *jp,
                             double &worig, double &wi, double &wj);
```

switching on `weight_mode` (`WEIGHT_SPECIES`: specwt by each particle's
species, including the stack copy; `WEIGHT_PARTICLE`: index-based weights
for `ip`/`jp`, outgoing-average for `worig`; default: 1/1/1) — and name the
three files explicitly in Steps 1–2 with this as their prescribed target
state.  Without it, Step 1's "resolve conflicts to a side-by-side state"
will be improvised in exactly the highest-conflict files.

---

## 3. Smaller required amendments

1. **Pin the count-keyword semantics now (Step 2), not "reconcile later".**
   Both branches already agree that `n` = raw simulation-particle count
   (SWS kept `NUM→COUNT` raw and added `numwi→COUNT_WI`; SWPM renamed the
   raw accumulator `SIMCOUNT` and made `COUNT` weighted).  The plan should
   state the outcome: one raw + one weighted accumulator; `n` raw under all
   schemes; a single weighted-count keyword (pick `numwi` or `nwt` — one
   name, both schemes); `nrho` and the mass/energy tallies weighted.  This
   is user-visible output; deciding it up front avoids a later re-bless.

2. **The scatter seam is two touches, not one.**
   `SCATTER_TwoBodyScattering_SWS` needs the pre-collision velocities
   captured *before* the scattering draw and the split-merge blend applied
   *after*.  The unified kernel therefore gets a 2–3-line guarded capture at
   the top plus the guarded `sws_scatter_merge(...)` call at the bottom.
   Same conclusion as the plan, honest accounting of the seam size.

3. **List the emission/creation files as explicitly out of unification
   scope.**  The nagoya branch also touches `create_particles.cpp`,
   `mixture.cpp/.h`, `fix_emit_face.cpp/.h`, `fix_emit_face_file.cpp/.h`,
   `fix_emit_surf.cpp/.h` (SWS scales emission/creation counts by
   `1/specwt`; SWPM initializes per-particle weights to 1.0 via
   `update_custom`).  These are scheme-specific with no counterpart to
   unify — leaving them alone is correct, but the plan's architecture and
   critical-files inventory omit them entirely, which reads as an oversight
   rather than a decision.  One sentence fixes it.

4. **Record the `ewilost_cell` lifecycle as a follow-up.**  The pool is
   per-cell state sized `nglocal`.  The plan records Ewilost *elimination*
   (Boyd's probabilistic update) as a follow-up, but not what happens to
   already-pooled energy under load balancing or grid adaptation (migrated?
   silently dropped?).  Given that reduction-scheme energy leaks were among
   the worst bugs found in the SWPM review, this deserves an explicit line
   in Step 6.  Related detail worth recording: Ewilost is re-injected only
   into collisions where *both* partners carry the max weight
   (`setup_collision_SWS`), so in strongly mixed cells the pool can sit
   unreturned for many steps.

5. **Optional tidiness for `collide.h`:** since both schemes' member blocks
   stay on the base class, group each into a small named struct
   (`SWSState`, `SWPMState`).  Cosmetic, but it keeps the base header
   near-upstream-shaped, in the spirit of the minimal-base-collide
   constraint.

---

## 4. Step ordering (considered and resolved)

An alternative ordering was considered: de-duplicate SWS **first** (Step 4
before Step 1), so the SWPM merge lands against baseline-shaped loops
instead of four clones, shrinking the `collide.cpp` conflict surface.
Recommendation: **keep the plan's merge-first order.**  The `np_eff`,
`wscale`, and tally-triple seams must serve *both* schemes; designing them
with both implementations in-tree avoids retrofitting the seams, and the
front-loaded merge conflicts are mechanical once the three tally files have
a prescribed target state (amendment §2).

---

## 5. Correctness cross-checks the unified code must preserve

From the SWPM review (this branch), the invariants that
`verify_swpm.py` pins and that the unified dispatch must not disturb:

- exact conservation of total weight, weighted momentum, and weighted energy
  through splitting and all three reduction schemes (energy/heat/stress),
  under both binary and weight grouping, serial and MPI;
- the guardrails: `fix stochastic_weight` required before
  `collide_modify stochastic_weight yes`; `Ngmin` bounds per reduction
  scheme; SWPM ⊕ nearcp, SWPM ⊕ ambipolar, SWPM ⊕ Kokkos, SWPM-reduction ⊕
  multi-species all rejected at init.

From the SWS suite (`examples/sws`, `verify_sws.py` on the review branch):
the SWS gold decks including `in.sws0.box` (weights present but scheme off),
which the plan correctly designates as the accessor-is-a-no-op canary.

The plan's end-state acceptance list (§Verification) is consistent with
these; with amendment §2.1 the `pweight`/`tally_weights` conversions remain
bit-exact by preserving each tally expression's multiplication order.

## Bottom line

Correct architecture; verified kernel seams; the right resolution of the
separate-files and minimal-base-collide constraints; honest risk
classification.  Apply the tally-triple amendment (§2), pin the
count-keyword decision (§3.1), and add the three inventory/lifecycle notes
(§3.2–3.4) — then the plan is fit to execute as written.
