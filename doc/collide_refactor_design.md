# Design: Refactoring `collide` / `collide_vss` for Sustainable Growth

Status: proposal / RFC
Scope: `src/collide.{h,cpp}`, `src/collide_vss.{h,cpp}`,
`src/KOKKOS/collide_vss_kokkos.{h,cpp}`; pending branches `subcell` and `swpm`.

All line numbers reference current `master` (commit 09391fe) unless otherwise
noted.

---

## 1. Problem statement

The collision code has grown combinatorially. Variants are expressed today by
two mechanisms — `template<int>` parameters (NEARCP, GASTALLY) and whole
duplicated loop bodies (`_one` / `_group` / `_one_ambipolar` /
`_group_ambipolar`) — and each new capability has been added by cloning an
entire NTC loop and editing a few dozen lines inside it.

Quantified duplication on master:

| Method | Location | Lines |
|---|---|---|
| `collisions_one<NEARCP,GASTALLY>` | collide.cpp:428-573 | ~146 |
| `collisions_group<NEARCP,GASTALLY>` | collide.cpp:580-860 | ~281 |
| `collisions_one_ambipolar<GASTALLY>` | collide.cpp:866-1163 | ~298 |
| `collisions_group_ambipolar<GASTALLY>` | collide.cpp:1170-1600 | ~431 |

That is ~1156 lines carrying four copies of one NTC skeleton. The
recombination third-body block is copy-pasted verbatim four times
(collide.cpp:509-521, 733-747, 992-1006, 1357-1372), as is the GASTALLY
snapshot/tally pair. In addition, `src/KOKKOS/collide_vss_kokkos.cpp` contains
a line-for-line second copy of the VSS physics kernels (~600 lines:
`SCATTER_*`, `EEXCHANGE_*`, `test/setup/perform_collision_kokkos`, `sample_bl`,
etc.) differing only in accessor spellings (`params[i][j]` vs `d_params(i,j)`,
`random->uniform()` vs `rand_gen.drand()`).

The pending branches continue the pattern:

- **`subcell`** adds `collisions_one_subcell<DIM,GASTALLY>` (~324 lines), a
  full clone of `collisions_one` in which only the partner-selection block and
  a per-cell binning prologue differ. It introduces a *new parallel template
  axis* (`<DIM,GASTALLY>` instead of `<NEARCP,GASTALLY>`) and is declared
  mutually exclusive with nearcp, ambipolar, and multiple groups via `init()`
  errors.
- **`swpm`** adds `collisions_one_stochastic_weighting<NEARCP,GASTALLY>`
  (~570 lines with `split()`, plus ~340 lines of reduction machinery), another
  near-clone of `collisions_one` — in which **both template parameters are
  unused in the body**, so four identical instantiations exist purely for
  dispatch symmetry. It is mutually exclusive with ambipolar, nearcp, groups,
  and gas-phase chemistry.

Meanwhile master has gained `mcflag` (`collide_modify ntc|mcf`, majorant
collision frequency; collide.cpp:1758-1759, collide_vss.cpp:147-149,185) as a
runtime branch inside `attempt_collision` — a fifth axis living in exactly the
two functions the swpm branch also modifies, guaranteeing merge conflicts.

Each new axis today either multiplies loop copies or is walled off by
`init()`-time mutual-exclusion errors. Neither scales. The goal of this design
is:

1. support an arbitrary number of variants without new loop copies;
2. make legal combinations compose by declaration, and illegal ones
   rejectable in one place;
3. zero performance regression for the base case;
4. reduce (eventually eliminate) the CPU/Kokkos physics duplication;
5. land incrementally, with every stage bit-identical for existing behavior.

## 2. What actually varies (ground truth)

All four loops share this skeleton (labels used throughout this document):

```
for icell in 0..nglocal:
  [A] cell prologue: np guard, volume, build plist
      (+p2g/glist | +elist pack | +nn reset | +subcell binning | +max-weight scan)
  [B] attempt computation: one pool (flat, :474) or gpair pool list (group, :658-671)
  for each pool:
    [C] pool prologue: ni/nj/ilist/jlist, per-group-pair nn memset (:699-705)
    for iattempt in nattempt:
      [D] pick i, pick partner j        # random | find_nn | find_nn_group | subcell search
      [E] resolve indices -> ipart/jpart # plist | plist[ilist[i]] | elist demux
                                         #   + e/e skip + electron-second swap (:956-981)
      [F] test_collision; on accept: record partner state (nn_last_partner :500-503)
      [G] recomb 3rd-body setup          # verbatim x4, only (ii,jj) exclusions differ
      [H] TALLY snapshot; setup_collision; perform_collision; TALLY emit
      [I] post-reaction bookkeeping demux   # THE messy delta (see below)
      [J] pool-exhaustion break          # np<2 | group counts | nptotal<2
  [K] cell epilogue                      # ambi velambi reconciliation
                                         #   (:1149-1161 == :1586-1598 verbatim)
```

Only `[F]→[H]` (test → setup → perform) is genuinely identical across all
copies. The deltas concentrate in `[A]`, `[D]`, `[E]`, `[I]`, `[K]` — and they
factor cleanly along **five orthogonal axes**:

| Axis | Variants today + pending | Touches |
|---|---|---|
| Addressing | flat, grouped | A, B, C, E, I, J |
| Particle source | plain, ambipolar (shadow electrons) | A, E, G, I, K |
| Partner selection | random, nearcp, subcell (pending) | A, C, D, F, I |
| Tally | off, gas-tally | H |
| Weighting | uniform, SWPM (pending) | A, F(accept), post-F split, K |

The post-reaction demux `[I]` differences, verified case by case:

- **flat/plain** (:546-570): jpart dead → dellist + plist swap-pop (+ nn
  fixup :554); kpart → plist append (+ `set_nn` :567) + `particles` re-fetch.
- **group/plain** (:772-845): additionally ipart group-migration; jpart
  group-move or delete-with-`p2g`-repair (:800-812); kpart group-add with
  glist re-fetch after every `addgroup`.
- **flat/ambi** (:1034-1140): `ambi_reset` first; kpart → plist *or*
  elist + `nlocal--`; jpart 4-way demux (neutral→electron: elist add + plist
  delete; electron→neutral: `add_particle` + new id + `ionambi=0` + elist
  swap-pop + plist append; dead electron: elist swap-pop; dead heavy:
  dellist); custom-array pointer re-fetch after each realloc.
- **group/ambi** (:1397-1565): the product of the previous two, plus egroup
  `ngroup`/`glist` maintenance on every elist mutation.

Two structural observations drive the design:

1. The group/ambi demux is not new logic — it is the *composition* of the
   group demux and the ambi demux. Written over a small set of addressing
   primitives, the ambi demux needs to exist only once.
2. nearcp's `nn_last_partner` fixups on delete/insert and subcell's
   "rebin whenever plist changes" are the *same hook* (partner-state
   maintenance on particle-list mutation). The subcell branch already reuses
   `nn_last_partner` for previous-partner exclusion, confirming that partner
   policies own this state.

## 3. Architecture

### 3.1 One skeleton, five policies

A single generic NTC cell loop, implemented as a member function template of
`Collide` in an implementation header, parameterized by five orthogonal
compile-time policies:

```cpp
// declared in collide.h, defined in collide_ntc_loop.h
template <class ADDRESS, class SOURCE, class PARTNER, int TALLY, class WEIGHT>
void Collide::ntc_collisions();
```

Policies are lightweight stack objects constructed from `Collide &` at the top
of `ntc_collisions()`, holding raw pointers (particles, ionambi, velambi,
species2group, nn arrays, subcell scratch, ...). This (a) hoists member loads
out of the loop exactly as the current code does with locals (:876-885), and
(b) mirrors Kokkos functor member capture, which eases the eventual device
port. All hook methods are plain `inline` functions.

A small shared context centralizes the bookkeeping idioms that today are
duplicated at every site:

```cpp
struct NTCContext {
  Particle::OnePart *particles;   // re-fetched via refresh_particles()
  int *next;
  int np, icell;
  double volume;
  inline void refresh_particles();   // the 8x-duplicated re-acquisition idiom
  inline int  push_dellist(int);     // DELTADELETE grow + append
  inline void grow_plist();          // DELTAPART grow
};
```

### 3.2 Hook inventory per policy

```cpp
struct AddressFlat / AddressGrouped {
  cell_build(ctx);          // plist              | plist + p2g + glist + ngroup
  npools(ctx); pool_setup(k);// 1 pool over cell  | gpair enumeration
                             //                     (attempt_collision per group pair)
  pindex_i(i); pindex_j(j); // identity           | ilist[i] / jlist[j]
  move_particle(p, newgrp); // no-op              | addgroup/delgroup + list re-fetch
  remove(j);                // plist swap-pop     | delgroup + p2g repair (:800-812)
  append(pidx);             // plist push         | plist push + addgroup
  on_electron_add/remove(); // no-op              | egroup ngroup/glist upkeep
  pool_exhausted();         // np < 2             | *ni/*nj tests (:849-856)
};

struct SourcePlain / SourceAmbi {
  cell_prologue(ctx);       // no-op | pack elist from ionambi (:919-931 / :1249-1264)
  attempt_count(np);        // np    | np + nelectron
  resolve(i,j, ipart,jpart);// plist only | elist demux + e/e short-circuit
                            //   + electron-second swap (:956-981; Flat only —
                            //   group deliberately has no e/e skip, :1339-1345,
                            //   comment must be carried over)
  recomb_excludes();        // (ii,jj) values per variant, see RNG contract
  post_reaction<ADDRESS>(); // plain demux | ambi_reset + electron demux,
                            //   written ONCE over ADDRESS primitives
  cell_epilogue(ctx);       // no-op | velambi reconciliation + conservation check
};

struct PartnerRandom / PartnerNearCP / PartnerSubcell<DIM> {
  cell_setup(ctx);          // no-op | realloc+memset nn (:447-450)
                            //       | subcell binning: nsub = np^(1/DIM), rebin
  pool_setup(ctx);          // no-op | per-group-pair nn memset (:699-705) | no-op
  select(i) -> j;           // uniform redraw loop | find_nn/find_nn_group
                            // | same-subcell pick else expanding shell search
  record(i,j);              // no-op | nn_last_partner writes (:500-503) | same
  on_delete(j, np);         // no-op | nn[j] = nn[np] (:554, :811) | rebin
  on_insert(np);            // no-op | set_nn / set_nn_group (:567, :822-832)
                            //       | rebin + subcell_alloc
};

// TALLY stays a plain int template parameter — exactly today's idiom.

struct WeightOff / WeightSWPM {
  cell_build_extra(ip);     // no-op | max-weight scan folded into plist build
  after_accept(ctx, ...);   // no-op | split(): weight transfer, may append
                            //   up to 2 copy-particles -> ctx.refresh_particles()
  cell_epilogue(ctx);       // no-op | group()/reduce merges; deletions via
                            //   ctx.push_dellist (piggybacks compress_reactions)
};
```

Under this decomposition the subcell branch's parallel `<DIM,GASTALLY>`
template axis disappears — DIM becomes the `PartnerSubcell<2|3>` policy choice
— and the swpm branch's unused template parameters disappear with it.

### 3.3 Why this mechanism (alternatives considered)

- **CRTP** loses because variants combine multiplicatively (ambipolar ×
  groups, nearcp × tally). CRTP composes along one axis; expressing
  combinations forces a class per combination — the same explosion in class
  form.
- **Runtime strategy objects** lose for hot hooks: partner selection and pair
  index resolution execute once per *attempt* (millions per step); an
  indirect call there is a measurable per-attempt cost. They would be
  acceptable for cold hooks, but mixing two composition mechanisms over one
  hook list doubles the concept count for no gain — enumerated template
  instantiations cover both at zero cost.
- **Policy structs over bare `template<int>` flags** because three of the
  axes (Address, Source, Partner) carry *state* (glist bookkeeping, elist,
  nn arrays, subcell scratch), not just a branch bit. The TALLY axis carries
  no state and stays an `int`, consistent with existing house style.
- **Per-pair virtual physics calls stay as they are.** The per-pair calls
  into the model (`test_collision`, `setup_collision`, `perform_collision`)
  are *already* virtual through `Collide *`; keeping that level of dispatch
  is the accepted baseline. The refactor adds nothing on top of it and the
  base-case instantiation must compile down to code isomorphic to today's
  `collisions_one<0,0>` (identity index maps, empty hooks, constant
  single-pool loop that folds away).

### 3.4 Dispatch: enumerated legal combinations

`Collide::collisions()` (collide.cpp:358-422) keeps its exact external
behavior; the inner if-tree becomes an explicit enumeration where each row is
one template instantiation:

```cpp
// init() has already rejected every combination not listed here
if (!ambiflag && ngroups == 1 && !swpmflag && !subcellflag) {
  if (!nearcp) {
    if (!ngas_tally) ntc_collisions<AddressFlat,SourcePlain,PartnerRandom,0,WeightOff>();
    else             ntc_collisions<AddressFlat,SourcePlain,PartnerRandom,1,WeightOff>();
  } else {
    if (!ngas_tally) ntc_collisions<AddressFlat,SourcePlain,PartnerNearCP,0,WeightOff>();
    else             ntc_collisions<AddressFlat,SourcePlain,PartnerNearCP,1,WeightOff>();
  }
}
else if (!ambiflag && ngroups > 1 && ...) { /* 4 Grouped rows */ }
else if (ambiflag)                        { /* 4 Ambi rows (PartnerRandom only) */ }
/* later: 4 Subcell rows (2D/3D x tally), 1-2 SWPM rows */
```

This yields ~15 instantiations at end state versus 12 today — the count
scales with *legal* combinations, not with 2^N. Instantiations are split
across translation units (`collide_ntc_flat.cpp`, `collide_ntc_group.cpp`,
`collide_ntc_ambi.cpp`, later `collide_subcell.cpp` / `collide_swpm.cpp`) so
no single TU compiles them all, bounding compile time and following the
one-variant-family-per-file convention.

**Legality in one place.** Each policy family contributes a small
requires/forbids capability bitmask checked once in `init()`:

```
AMBI    forbids NEARCP | SUBCELL | SWPM
SUBCELL forbids GROUPS | AMBI | NEARCP | SWPM
SWPM    forbids REACT  | AMBI | NEARCP | GROUPS
```

This replaces today's ad-hoc scattered errors and gives future variants a
single conflict-free registration point. Relaxing a restriction later means
deleting a bit and adding a dispatch row — not writing a new loop.

### 3.5 Compile-time vs runtime, per axis

| Axis | Frequency | Mechanism | Why |
|---|---|---|---|
| Partner select | per attempt | template policy | hottest hook; random path is 2-3 RNG draws total — any dispatch overhead is a measurable fraction |
| Address (flat/group) | per attempt (index resolution) | template policy | `plist[i]` vs `plist[ilist[i]]` sits in the pair-fetch path |
| Source (plain/ambi) | per attempt (resolve, e/e skip) | template policy | resolve runs per attempt; demux shares state with resolve |
| GASTALLY | per collision | `int` template | existing idiom, unchanged |
| Weight (swpm) | per accepted collision | template policy | keeps the base-case loop literally free of swpm code |
| mcflag / remainflag | per cell (inside `attempt_collision`) | runtime branch (unchanged) | per-cell branch is free; a template axis would double instantiations for nothing |
| vibstyle / rotstyle / relaxflag | inside physics | runtime (unchanged) | already inside the virtual physics call |
| recombflag | per collision | runtime, hoisted bool (unchanged) | as today |
| physics model | per pair | virtual via `Collide *` (unchanged) | accepted cost; keeps models pluggable |

If a third attempt-count scheme ever appears alongside NTC-remainder and
MCF-Poisson, promote `mcflag` to a small `AttemptScheme` enum switch inside
`attempt_collision` — never to a template axis.

### 3.6 RNG-sequence contract

The refactor must be bit-identical for every currently reachable path. The
skeleton therefore *fixes the draw order per attempt* as:

```
select-i draw(s), select-j draw(s), test draw, recomb draws,
physics draws (inside perform), demux id draw (ambi electron->neutral)
```

identical to all four current loops. The recombination helper takes
per-variant `(ii,jj)` exclusion values so the `while (k == ii || k == jj)`
redraw sequence is reproduced exactly; for flat/ambi, passing the raw `i,j`
even when `>= np` reproduces :1000-1001 bit-for-bit, since `k < np` can never
match an electron index. Every hook must document which draws it consumes;
any intentional sequence change requires maintainer sign-off and new golden
logs.

## 4. How the pending branches land

### 4.1 subcell → `PartnerSubcell<DIM>` (rebases after Stage 3)

`collide_subcell.{h,cpp}` contains the policy plus the existing
`subcell_alloc` / `subcell_rebin` machinery and the five scratch arrays:

- `cell_setup`: nsub computation + rebin + scratch sizing (the branch's
  per-cell prologue, unchanged);
- `select`: same-subcell linked-list pick, else the expanding 2D/3D shell
  search (moved verbatim);
- `on_delete` / `on_insert`: rebin + `subcell_alloc` — replacing the
  branch's inline "must rebin whenever plist changes" blocks;
- previous-partner exclusion keeps using the nearcp `nn_last_partner`
  arrays, as the branch already does.

The branch's ~324-line cloned loop is deleted; the branch shrinks to the
policy, four dispatch rows (`<2|3> x tally`), and one capability-mask line.
Its current mutual exclusions become declarations, not structure — lifting
them later (e.g. subcell × ambipolar) requires only enabling a combination
and adding its dispatch row.

### 4.2 swpm → `WeightSWPM` (rebases after Stage 4)

`collide_swpm.{h,cpp}` keeps `split()`, `group()`, `group_bt`, and
`collide_reduce.cpp`'s merge operators as-is, exposed through the policy:

- `cell_build_extra(ip)`: the skeleton passes each particle through this hook
  while building plist, so the max-weight scan needs no second pass;
- `after_accept`: `split()` using `ctx.refresh_particles()` /
  `ctx.grow_plist()` — eliminating the branch's ~8 duplicated
  pointer-re-acquisition sites;
- `cell_epilogue`: `group()`/reduce; deletions via `ctx.push_dellist()`,
  continuing to piggyback on `compress_reactions`.

The VSS-side hooks: the fnum attempt scaling
(`max_stochastic_weight * fnum * (1 + pre_wtf*wtf)`) stays a runtime branch
inside `attempt_collision` next to `mcflag` (per-cell, cold — and this is
also where the swpm/mcflag merge conflict is resolved, since both schemes now
coexist in one function). The weight-ratio thinning currently inside
`test_collision` moves into a `WEIGHT::accept()` hook *only if* the branch's
RNG draw order is preserved by doing so; otherwise it stays a runtime flag
inside `test_collision`. Decide at rebase time — both are correct designs.

## 5. Kokkos: physics-kernel sharing

### 5.1 End state

A header `src/collide_vss_kernels.h` — no Kokkos includes — containing the
physics as free-function templates over a backend system policy:

```cpp
#ifndef SPARTA_KK_INLINE            // KOKKOS TUs define this to
#define SPARTA_KK_INLINE inline     //   KOKKOS_INLINE_FUNCTION before inclusion
#endif

template<class SYS> SPARTA_KK_INLINE
int  vss_test_collision(SYS &, int icell, int igroup, int jgroup,
                        Particle::OnePart *, Particle::OnePart *, State &);
template<class SYS> SPARTA_KK_INLINE
void vss_setup_collision(SYS &, ..., State &precoln, State &postcoln);
template<class SYS> SPARTA_KK_INLINE
void vss_scatter_two_body(SYS &, ..., State &, State &);
template<class SYS> SPARTA_KK_INLINE
void vss_scatter_three_body(...);
template<class SYS> SPARTA_KK_INLINE
void vss_eexchange_nonreacting(...);   // stays separate from reacting variant:
template<class SYS> SPARTA_KK_INLINE
void vss_eexchange_reacting(...);      //   deliberately different physics
                                       //   (pairwise vs Dirichlet shared-pool);
                                       //   warning comments carried over
// + vss_sample_bl, eff_vib_dof, vib_pool_temp, rotrel, vibrel
```

`SYS` supplies `params(i,j)`, `prefactor(i,j)`, `species(i)`,
`vremax(icell,ig,jg)` (mutable), `uniform()`, `boltz/dt/fnum`, vibmode access,
and react hooks. The CPU `HostVSSSys` wraps `CollideVSS` members plus
`RanKnuth *`; `KokkosVSSSys` wraps `d_params` / `d_prefactor` / `d_species` /
`d_vremax` views plus `rand_type &`.

Two facts make this low-risk rather than speculative:

- The Kokkos signatures **already prove the target shape**:
  `test_collision_kokkos(..., State &, rand_type &)` and
  `setup_collision_kokkos(..., State &, State &)`
  (collide_vss_kokkos.h:90-97) are exactly the State-by-reference form. The
  CPU side converges to the shape Kokkos already validated, not the reverse.
- Nothing outside the collide files calls these virtuals (verified by grep);
  `collide->extract()` used by `react_*` and `compute_lambda_grid` is
  untouched. So `Collide`'s virtual signatures can gain `State &` parameters
  and the thread-unsafe `precoln`/`postcoln` members can be deleted, with the
  skeleton declaring `State precoln, postcoln` per cell exactly as the Kokkos
  functor does today (collide_vss_kokkos.cpp:664-665).

Kernel bodies are moved *verbatim* (member access → `sys.` accessor, same
floating-point operation order) so the CPU sequence stays bit-identical. This
retires ~600 duplicated lines in `collide_vss_kokkos.cpp`.

### 5.2 Optional final stage: shared per-cell loop body

The per-cell body of `ntc_collisions()` can later be extracted as
`template<class BACKEND, class... POLICIES> ntc_cell_body(...)` where BACKEND
abstracts plist access (1D pointer vs 2D `d_plist` view), dellist push (serial
vs `atomic_fetch_add` + retry signal, collide_vss_kokkos.cpp:784-798), counter
accumulation (direct vs the ATOMIC_REDUCTION demux), and capacity failure
(`memory->grow` vs set `d_retry` and bail). The Kokkos-only retry/backup
driver, `d_scalars` pack, and view management stay in the Kokkos class. This
is what lets Grouped/Subcell/SWPM reach the GPU nearly for free — valuable,
but not load-bearing for the CPU refactor. The minimum bar (do not make the
port harder) is met earlier by reshaping the Kokkos functor bodies to mirror
the hook structure 1:1.

## 6. Staged migration plan

Every stage compiles standalone, passes the full examples/CI regression suite
bit-identically (except where explicitly flagged), and is independently
revertible.

**Stage 0 — bug fix + test scaffolding.**
Fix the dispatch dead-code bug: collide.cpp:404 reads `else if (!ngas_tally)`
— the same condition as line 401 — so `collisions_one_ambipolar<1>` /
`collisions_group_ambipolar<1>` (ambipolar + gas tally) are unreachable. Fix
to `else if (ngas_tally)`; same pattern at collide_vss_kokkos.cpp:432
(currently shadowed by the :407-411 error, fix for hygiene). Since this
*enables never-executed code*, add an ambi+tally regression input and
physically validate it before trusting it as a baseline. Add scripts capturing
golden thermo logs at fixed seeds for the input matrix in §8.

**Stage 1 — extract shared helpers in place.**
Factor the verbatim-duplicated blocks into inline private members used by all
four *existing* loops: recomb third-body setup (4 copies), tally
snapshot/emit pair (4 copies), `push_dellist`, `grow_plist`, elist grow/pack
helpers (2 copies each), velambi reconciliation epilogue (2 copies). Pure
code motion; bit-identical by construction; shrinks every later diff.

**Stage 2 — skeleton + flat/plain path (the architectural PR).**
New: `collide_ntc_loop.h` (skeleton + `NTCContext`, including the pool loop
from day one), `collide_policies.h` (`AddressFlat`, `SourcePlain`,
`PartnerRandom`, `PartnerNearCP`, `WeightOff`), `collide_ntc_flat.cpp` (4
explicit instantiations). Delete `collisions_one`. Introduce the dispatch
table (other rows still call the old loops). Gate on bit-identical golden
logs plus the performance/codegen checks of §8.

**Stage 3 — grouped addressing.**
`AddressGrouped` (gpair pools, `find_nn_group` wiring, p2g repair in
`remove()`), `collide_ntc_group.cpp`; delete `collisions_group`.
**The subcell branch rebases here** — it needs only Partner hooks and the
flat path.

**Stage 4 — ambipolar source policy.**
`SourceAmbi` + `post_reaction<ADDRESS>` written once,
`collide_ntc_ambi.cpp`; delete both ambipolar loops (~730 lines). The
riskiest CPU stage — gate on bit-identical ambi, ambi 3-body, and the Stage-0
ambi+tally inputs. Consolidate init-time legality checks into the
capability-bitmask table in the same PR. **The swpm branch rebases here** (it
needs the Weight hook seams and `NTCContext` helpers).

**Stage 5 — Kokkos structural alignment.**
Reshape `CollideVSSKokkos` functor bodies into helpers named after the hook
points. No behavior change; device code generation should be unchanged.

**Stage 6 — shared physics kernels.**
`collide_vss_kernels.h` + `HostVSSSys`/`KokkosVSSSys`; virtual signatures
gain `State &`; delete `precoln`/`postcoln` members and the ~600 duplicated
Kokkos physics lines. CPU bit-identical (verbatim body motion); Kokkos
validated via a `SPARTA_KOKKOS_EXACT` build matching CPU golden logs, plus
statistical checks on GPU. Schedule after swpm lands (or coordinate the
two-line signature update with that branch).

**Stage 7 (optional) — shared per-cell body over a BACKEND policy** (§5.2).

## 7. File layout (end state)

```
src/collide.h                    Collide + ntc_collisions<> decl, NTCContext,
                                 capability masks
src/collide.cpp                  init/setup/dispatch/grid hooks/helpers
                                 (~1156 loop lines -> ~300)
src/collide_ntc_loop.h           skeleton template (implementation header,
                                 included only by *_ntc_*.cpp)
src/collide_policies.h           AddressFlat/Grouped, SourcePlain/Ambi,
                                 PartnerRandom/NearCP, WeightOff
src/collide_ntc_flat.cpp         explicit instantiations: flat/plain rows
src/collide_ntc_group.cpp        grouped/plain rows
src/collide_ntc_ambi.cpp         ambi rows (+ post_reaction<ADDRESS> def)
src/collide_vss.{h,cpp}          thin virtual wrappers + file I/O +
                                 attempt schemes (ntc/mcf/swpm-fnum)
src/collide_vss_kernels.h        shared physics templates + State + Params
                                 (Stage 6)
src/collide_subcell.{h,cpp}      PartnerSubcell<DIM> + rebin machinery
                                 (from subcell branch)
src/collide_swpm.{h,cpp}         WeightSWPM + split/group/reduce
                                 (from swpm branch)
src/KOKKOS/collide_vss_kokkos.*  driver/retry/views + KokkosVSSSys;
                                 duplicated physics deleted at Stage 6
```

All core `src/` (collide is core; no package/Install.sh changes). KOKKOS
files stay under `src/KOKKOS/`.

## 8. Verification

**Bit-identity per stage.** Golden thermo logs at fixed seeds, on 1 and 4
ranks, for: `examples/collide/in.collide`, `in.collideInterspecies`
(multi-group), `examples/ambi`, `examples/ambi_3body` (ambi demux + recomb),
`examples/chem` (react/dellist/kpart paths), `examples/vibrate` (discrete
vibration custom arrays), `examples/relax_const` + `relax_variable`,
`examples/tally_computes` (GASTALLY), `examples/thermostat`; plus new inputs
for currently uncovered axes: a nearcp variant of `in.collide`
(`collide_modify nearcp yes 10`), an mcf variant (`collide_modify mcf`), and
the Stage-0 ambi+tally input. Any byte diff in thermo columns means an RNG
sequence break and fails the stage.

**Performance.** A scaled `in.collide` (grid and particle count ×~10, ≥5k
steps, large thermo stride), 5 runs, comparing the `Coll` row of SPARTA's
timing breakdown; gate at ≤1% regression. Lower-noise codegen proxies:
`perf stat -e instructions` (base case must match within noise) and an
`objdump -d` size/shape comparison of the
`<AddressFlat,SourcePlain,PartnerRandom,0,WeightOff>` instantiation against
the old `collisions_one<0,0>` after Stage 2.

**Compile time / bloat.** Time the collide TUs and record binary size before
and after Stages 2-4 (expected roughly flat: ~15 instantiations vs 12, split
across TUs).

**Kokkos.** `SPARTA_KOKKOS_EXACT` serial-device build must match CPU golden
logs at Stage 6; standard GPU builds validated statistically (mean
collision/attempt/react counts over long runs), since the RNG differs by
design.

## 9. Risks and mitigations

1. **RNG sequence breaks** (highest likelihood). Mitigated by the per-hook
   draw-order contract (§3.6), fine stage granularity, byte-exact golden logs
   including reaction-heavy inputs, and the `(ii,jj)` recomb-exclusion
   technique.
2. **Base-case codegen regression from the pool abstraction.** Identity
   policies are designed to constant-fold (`AddressFlat::npools() == 1`);
   verified by the asm/`perf stat` gate. Fallback if a compiler refuses: a
   compile-time `if (ADDRESS::SINGLE_POOL)` constant branch hoisting the pool
   loop — dead-branch elimination is guaranteed.
3. **Ambi demux unification bugs.** The 4-way jpart demux and its p2g
   interaction (:1555-1564) are subtle. Mitigated by Stage 1 shrinking the
   diff to near-pure motion, PR comments mapping each case to old line
   numbers, ambi_3body + chem goldens, and keeping the `ambi_check()` debug
   validation callable.
4. **Template bloat / compile time.** Bounded by enumerated instantiation and
   per-family TUs; measured per stage.
5. **Kokkos divergence during CPU stages 2-4.** Zero risk by construction:
   the Kokkos path still overrides `collisions()` wholesale and is untouched
   until Stage 5.
6. **Merge conflicts with subcell/swpm.** Both branches modify loops that
   Stages 2-4 delete — rebasing is a rewrite-as-policy, not a merge. Mitigate
   by landing Stage 3 quickly after Stage 2 (unblocks subcell), sharing this
   hook map with the branch authors in advance (e.g. isolating `split()`
   behind a helper now costs nothing), and by the capability-mask table
   giving each branch a single-line, conflict-free registration point. The
   Stage-6 virtual-signature change touches swpm's VSS hooks — schedule it
   after swpm lands.

## Appendix A: current-state inventory (for reviewers)

- Class hierarchy: `Collide` (abstract; owns all NTC machinery and particle
  bookkeeping) → `CollideVSS` (only concrete model; physics via virtuals
  `vremax_init`, `attempt_collision` ×2, `test_collision`, `setup_collision`,
  `perform_collision`, `extract`) → `CollideVSSKokkos` (overrides
  `collisions()` wholesale). `CollideVSS` does **not** override
  `collisions()` — the model/driver seam this design builds on already
  exists, and no other subclass of `Collide` exists in the tree, so internal
  interfaces can be reshaped freely.
- External API that must not change: `collisions()`, `init()`, `setup()`,
  `modify_params()`, the grid pack/unpack/copy/adapt hooks (grid_comm,
  grid_surf, adapt_grid), `extract()` (react_bird/tce/qk,
  compute_lambda_grid), public counters (stats.cpp, finish.cpp), and the
  bidirectional react coupling (`react->recomb_species/part3/density` written
  from the loop; `react->attempt()` called inside `perform_collision`).
- `collide_vss.cpp` kernel families: `SCATTER_TwoBodyScattering` (~55 ln) /
  `SCATTER_ThreeBodyScattering` (~64 ln, near-twin);
  `EEXCHANGE_NonReactingEDisposal` (~147 ln) / `EEXCHANGE_ReactingEDisposal`
  (~200 ln) — the latter pair implements deliberately different physics
  (per-mode pairwise relaxation vs shared-pool Dirichlet stick-breaking) and
  must remain separate.
- House conventions respected by this design: C-style C++, no STL in hot
  loops, `memory->create/grow` with DELTA chunking, `template<int>` flags as
  the hot-loop specialization idiom, per-file enums, `copymode` guard for any
  new owned members, Kokkos DualView mirror pattern.
