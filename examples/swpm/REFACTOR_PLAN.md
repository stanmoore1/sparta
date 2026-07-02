# SWPM refactor plan

Step-by-step plan for the SWPM cleanup refactor, derived from
`REFACTOR_ANALYSIS.md`.  Every step is **bit-exact** (the pinned gold logs must
keep passing byte-for-byte) unless explicitly marked *[semantics decision]*.
Each step is one reviewable commit; the verification gate below runs after
every one.

## Verification gate (run after every step)

```bash
# from a configured build dir (cmake -C ../cmake/presets/mpi.cmake -DSPARTA_ENABLE_TESTING=ON ../cmake)
make -j
ctest -R swpm                                   # 16 gold-log tests, mpi_1 + mpi_4
python3 examples/swpm/verify_swpm.py <spa_binary>                    # serial invariants
python3 examples/swpm/verify_swpm.py "mpirun -np 4 <spa_binary>"     # parallel invariants
```

Pass criteria: all 16 ctests pass with **unchanged** gold logs, and
`verify_swpm.py` reports all checks passed in both modes.  A gold-log diff on
a bit-exact step means the step introduced a behavior change — stop and fix,
do not re-bless.

---

## Phase 0 — baseline

**Step 0.1 — record the baseline.**
Run the verification gate on the unmodified branch and keep the output.  Also
run the BKW benchmark once (`bkw_init.py` + `in.swpm.bkw`, see
`examples/swpm/README`) and save its log as the physics baseline for any
future statistically-equivalent change.

---

## Phase 1 — mechanical cleanup (no functional content)

**Step 1.1 — revert whitespace churn in `compute_reduce.cpp`.**
~200 of the 242 changed lines vs upstream are indentation-only.  Restore
upstream indentation everywhere the line's content is otherwise unchanged.
Files: `src/compute_reduce.cpp`.
Check: `git diff origin/master -- src/compute_reduce.cpp` shrinks to only the
SWPM-weighting hunks (~40 lines).

**Step 1.2 — consolidate duplicated enums.**
`enum{ENERGY,HEAT,STRESS}` and `enum{BINARY,WEIGHT,OCTREE}` are re-declared in
`collide.cpp` and `collide_swpm.cpp` (a drift hazard: the two copies must stay
in the same order); `enum{INT,DOUBLE}` duplication is a pre-existing SPARTA
idiom — leave it.  Put the two SWPM enums in `collide.h` (protected section)
and delete the per-file copies.
Files: `src/collide.h`, `src/collide.cpp`, `src/collide_swpm.cpp`.

**Step 1.3 — delete the octree stub.**
Remove `Collide::group_octree()`, its declaration, the `OCTREE` enum value,
and the `group_type == OCTREE` dispatch arm (unreachable: the parser accepts
only `binary`/`weight`).  Re-add with a real implementation if/when octree
grouping is written.
Files: `src/collide.h`, `src/collide_swpm.cpp`, `src/collide.cpp` (enum).

**Step 1.4 — drop the unused template parameters.**
`collisions_one_stochastic_weighting<NEARCP,GASTALLY>` uses neither parameter.
Make it a plain member function; collapse the four-way dispatch in
`Collide::collisions()` to a single call (nearcp+SWPM is already rejected in
`init()`).  Add a code comment at the call site that gas-collision tallies are
not implemented for SWPM (tracked limitation), so the omission is visible.
Files: `src/collide.h`, `src/collide.cpp`, `src/collide_swpm.cpp`.

---

## Phase 2 — unify the diagnostic weighting

**Step 2.1 — add the cached accessor to `Particle`.**
- Add `int index_sweight;` (init -1) and an inline
  `double *sweight_vector()` returning `edvec[ewhich[index_sweight]]` or NULL.
- `FixStochasticWeight` sets `particle->index_sweight` when it creates/finds
  the custom attribute and resets it to -1 in its destructor (it already owns
  the attribute lifetime).
- Keep `Particle::stochastic_weights()` temporarily as a thin wrapper (with
  its grid-weighting exclusivity check moved to `Collide::init()` /
  `FixStochasticWeight`), so steps 2.2.x can convert callers incrementally.
Files: `src/particle.h`, `src/particle_custom.cpp`,
`src/fix_stochastic_weight.cpp`.

**Step 2.2 — convert the computes, one file per commit.**
For each of:
`compute_grid`, `compute_thermal_grid`, `compute_temp`, `compute_eflux_grid`,
`compute_pflux_grid`, `compute_sonine_grid`, `compute_tvib_grid`,
`compute_vmom_grid`, `compute_boundary`, `compute_surf`,
`compute_isurf_grid`, `compute_reduce`:
- fetch `double *swgt = particle->sweight_vector();` once per invocation
  (per tally call for the surf/boundary tallies);
- replace `if (sweights) swfrac = sweights[i]; else if (particle->weightflag)
  swfrac = particles[i].weight;` with `if (swgt) swfrac = swgt[i];`
  (the `weightflag` branch is dead — `Particle::weightflag` is never set);
- **preserve the tally expression shape exactly** (same multiplication order)
  so results stay bit-identical.
Gate after each file.

**Step 2.3 — delete the dead plumbing.**
Remove `Particle::weightflag` and the now-unused
`Particle::stochastic_weights()` wrapper.  After this step
`git diff origin/master -- src/particle.h src/particle.cpp
src/particle_custom.cpp` should show only: `index_sweight`,
`sweight_vector()`, and the fix hookup.
Files: `src/particle.h`, `src/particle.cpp`, `src/particle_custom.cpp`.

**Step 2.4 — cache the per-cell weight array in the VSS hot path.**
In `CollideVSS::test_collision`, replace the per-pair
`particle->edvec[particle->ewhich[index_stochastic_weight]]` fetch with the
same `sweight_vector()` call, hoisted where the compiler can keep it in a
register (the array pointer is stable within a cell's attempt loop; it changes
only when `split()` grows the particle array, after which the SWPM loop
already re-fetches its own pointers).  No RNG or FP ordering change.
Files: `src/collide_vss.cpp`.

**Step 2.5 — *[semantics decision]* `compute reduce` weighting.**
Decide and document (in `doc/compute_reduce.txt`) one of:
- (recommended) weight only extensive quantities (KE, EROT, EVIB, per-particle
  custom vectors), never coordinates/velocities; normalize AVE by the summed
  weight instead of the raw count; or
- remove weighting from `compute reduce` entirely (per-grid computes are the
  weighted diagnostics path).
No gold log currently pins this behavior, so whichever choice is made, add a
stats column to one regression deck (or a check to `verify_swpm.py`) that pins
the chosen semantics, then re-run the gate.
Files: `src/compute_reduce.cpp`, `doc/compute_reduce.txt`,
`examples/swpm/` (one deck or the verifier).

---

## Phase 3 — placement (decision recorded; no code motion)

**Step 3.1 — keep SWPM on the base `Collide` class.**
Per `REFACTOR_ANALYSIS.md` §3.4: this matches the ambipolar precedent, uses
the base-class pure-virtual collision interface, and keeps the members
inheritable for a future Kokkos port.  Rename nothing.  If a future extension
(Kokkos SWPM, multi-species reduction) strains the base class, revisit a
`collide vss/sw` style subclass then — as its own project, since it changes
user input syntax.

Optional in this phase, if file organization matters to reviewers:
merge `collide_reduce.cpp` into `collide_swpm.cpp` (they implement one
feature; both are new files, so this is pure file motion) — or leave as-is.

---

## Phase 4 — tracked follow-ups (explicitly out of scope)

Not part of this refactor; listed so they are not lost:

1. Gas-collision tallies inside the SWPM loop (restores the `GASTALLY`
   behavior of the standard loops).
2. Multi-species reduction (needs per-species group moments; currently
   rejected at init).
3. Kokkos port (currently rejected at init).
4. `react` chemistry + SWPM validation (shared deletion list).
5. Octree grouping (stub deleted in step 1.3).
6. Same-timestep grouping of split-created particles (currently deferred to
   the next timestep's sort — acceptable, documented).

---

## Acceptance criteria for the whole refactor

1. All 16 `swpm` ctests pass with **byte-identical gold logs** (except the
   deck touched by step 2.5, which is re-blessed once with its semantics
   change reviewed).
2. `verify_swpm.py` passes serial and `mpirun -np 4`.
3. `git diff origin/master -- src/` shrinks materially (target: the
   `compute_*` diff roughly halves; `particle.*` diff is ~10 lines;
   `compute_reduce.cpp` diff is ~40 lines).
4. No user-facing input syntax changes; the doc pages still describe the
   behavior accurately (`compute_reduce.txt` gains the weighting semantics;
   `collide_modify.txt` needs no change — it never documented octree).
