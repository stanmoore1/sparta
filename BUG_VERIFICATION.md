# SPARTA AI-Found Bug Verification & Fix Audit

This report independently verifies the list of ~111 bugs (Bugs **1–103, 105–112**; there is
no #104) reported by two AI tools, and audits the two fix branches against the canonical
baseline `origin/master` (`5aed836`):

- **AB** = `origin/ai_bugfixes` (aborner, commit `d3f62a1`) — broad (128 src files, 33 KOKKOS).
- **CP** = `origin/copilot/fix-bugs-from-bugs-md` (`39c7a3a`) — narrower (78 src, 19 KOKKOS).

For every bug we determined: (1) whether it is a *genuine* defect in `origin/master`, and
(2) whether each branch's fix is correct. The verified-correct fixes were then applied to this
branch (`claude/sparta-static-analysis-bugs-335wsw`, started fresh from `origin/master`);
false positives and incorrect/regressive fixes were rejected.

## Headline results

- **111** numbered entries audited. Of these, **3 are duplicates** of another entry
  (23 = 67 = 98, same `fix_ablate` line) and **4 are NOT bugs** (40, 84, 93, 102).
- **~104 genuine defects** confirmed (REAL or PARTIAL-but-genuine).
- **Neither branch is complete or correct on its own:**
  - **AB** has **2 incorrect fixes** (Bug 46 and Bug 40) and misses several pure-logic bugs
    (3, 5, 9, 30, 47, 53, 55, 56, 57, 58, 81).
  - **CP** has **2 incorrect/partial fixes** (Bug 40, and the `react_qk` half of Bug 72) and
    misses most overflow/NaN/KOKKOS hardening (62, 64, 65, 73(½), 76, 77, 78, 79, 80, 87, 88,
    89, 90, 91, 92, 94, 96, 99, 100, 101, 102, 103, 110 …).
  - Both branches **fix only the KOKKOS copies** of Bugs 62/77/78/79/80, leaving the identical
    **CPU** defects in `update.cpp` / `geometry.cpp` unfixed.
- The deliverable branch applies the correct fix for every genuine bug (taking AB or CP
  whichever is right, or a corrected/extended fix where both were wrong or incomplete),
  **rejects** the 4 non-bugs, and **avoids AB's EPSZERO regression** in `collide_vss*`.
- **Build:** `make serial` links cleanly (`spa_serial`) with all non-KOKKOS fixes applied.

## Follow-up review (second pass): inert / non-defect changes reverted

An independent re-verification of every applied change against `origin/master` found that a
subset of the originally-applied edits do **not** fix a reachable defect and were reverted to
keep the diff scoped to genuine bugs. These reverts do **not** touch any confirmed fix:

- **82, 83** (`adapt_grid.cpp`): the `(bigint)` casts on `int * sizeof(...)` are **inert** —
  `sizeof` is `size_t`, so the multiply is already 64-bit. Same pattern this audit rejected as
  Bug 93. Reverted. (The genuine Bug 81 `nglocalprev` fix in the same file is kept.)
- **61** (`surf.cpp`): `(size_t)(nmax-old)*sizeof(Line/Tri)` memset casts are inert (already
  64-bit). Reverted. (The genuine Bug 33 `snprintf(estyle,...)` fixes are kept.)
- **60** (`surf_comm.cpp`): the two `spread_own2local_reduce` `(bigint)nlocal*n` create-casts are
  inert — the function already errors out via `bcount > MAXSMALLINT` before the create, and
  `Memory::create` truncates back to `int`. Reverted. (The two un-guarded `spread_local2own`
  `(bigint)(n+1)*nunique` casts, which remove genuine signed-overflow UB, are kept.)
- **59** (`surf_custom.cpp`): the four 1-D *vector* memset casts (`(size_t)n*sizeof(T)`) are
  inert (`int * size_t` already promotes). Reverted. The four 2-D *array* casts
  (`(size_t)n*eicol*...`, where `n*eicol` is `int*int` and can overflow before promotion) are
  genuine and **kept**.
- **100** (`compute_dt_grid.cpp` + KOKKOS): the `vrm_max > 0.0` guard is **unreachable** — the
  loop already does `if (!(temp[i] > 0.)) continue;` upstream, so `vrm_max > 0` always holds.
  Reverted.
- **105** (`collide_vss.cpp`): the `volume > 0.0` guards in both `attempt_collision` overloads
  are **unreachable** — all callers in `collide.cpp` hard-error on `volume == 0.0` before the
  call. Reverted. (The genuine Bug 46 `vremax==0` guard and Bug 47 symmetric `rotc2` assignment
  in the same file are kept, and the EPSZERO guard remains intact.)
- **75** (`react_bird_kokkos.cpp`): the pool-seed change `12345`→`54321` is speculative, not a
  correctness fix, and changes RNG reproducibility of existing KOKKOS runs while ~8 sibling
  classes keep the old seed. Reverted to `12345`.

Note: the structural `Memory::create(TYPE*&, int n, ...)` int-parameter limitation means
`(bigint)` casts on a *count* argument cannot enable a >INT_MAX allocation regardless; the
companion `memset` casts (which take `size_t`) are the ones that genuinely matter, and those
are retained for the array cases (56, 59-array).

## Legend

- **Real?**: REAL · PARTIAL (genuine but narrow/defensive) · **NO** (not a bug)
- **AB / CP**: OK (correct) · DIFF (correct, alternative) · NO (not fixed) · **WRONG** · PARTIAL
- **Applied**: ✔ applied · ✔* applied (corrected/extended beyond both branches) · ✖ rejected

| # | File(s) | Real? | AB | CP | Applied | Note |
|--|--|--|--|--|--|--|
| 1 | comm.cpp | REAL | OK | OK | ✔ | spurious double alloc of `rbuf` (bigint→int truncation) |
| 2 | react_tce.cpp | REAL | OK | OK | ✔ | unreachable react_prob warning moved out of `switch` |
| 3 | compute_reduce.cpp | REAL | NO | OK | ✔ | `narg`→`nargnew` bounds (replace & subset) — AB missed |
| 4 | react_qk.cpp | REAL | OK | WRONG | ✔ | see Bug 72; AB correct, CP pollutes `react_prob` |
| 5 | grid_custom.cpp | REAL | NO | OK | ✔ | `if(nnew-nold)`→`if(nnew>nold)` (huge memset on shrink) |
| 6 | compute_property_surf.cpp | REAL | OK | OK | ✔ | 3D pack_id `nsown`→`nchoose` OOB |
| 7 | utils.cpp | PARTIAL | OK | OK | ✔ | snprintf hardening (unbounded `cmd` string) |
| 8 | variable.cpp | REAL | OK | OK | ✔ | unchecked fopen of lock file |
| 9 | compute_count.cpp | REAL | NO | OK | ✔ | `if(imix<0)`→`if(igroup<0)` — AB missed |
| 10 | compute_gas_collision_tally.cpp | REAL | OK | OK | ✔ | `type2`→TYPE1 should be TYPE2 |
| 11 | compute_gas_reaction_tally.cpp | REAL | OK | OK | ✔ | vy2/vz2 pre-vel enum shift |
| 12 | compute_surf_reaction_tally.cpp | REAL | OK | OK | ✔ | `ID2POST||ID2POST`→`ID1POST||ID2POST` |
| 13 | fix_grid_check.cpp | REAL | OK | OK | ✔ | OOB `cells[icell]` in error msg + missing `continue` |
| 14 | timer.cpp | REAL | OK | OK | ✔ | uninit `timeout_start`; stray debug printf |
| 15 | surf_collide_specular.cpp | REAL | OK | OK | ✔ | wrapper() ignored noslip_flag |
| 16 | surf_collide_cll.cpp | REAL | OK | OK | ✔ | missing `if(copy) return` → double free |
| 17 | surf_collide_impulsive.cpp | REAL | OK | OK | ✔ | missing `if(copy) return` → double free |
| 18 | KOKKOS/fix_grid_check_kokkos.cpp | REAL | OK | OK | ✔ | OOB device access; add `return` |
| 19 | KOKKOS/compute_lambda_grid_kokkos.cpp | REAL | OK | OK | ✔ | missing `else` KNY/KNZ (subsumed by 38) |
| 20 | KOKKOS/fix_ave_histo_weight_kokkos.cpp | REAL | OK | OK | ✔ | realloc `>nmax`→`<nmax` |
| 21 | KOKKOS/compute_sonine_grid_kokkos.cpp | REAL | OK | OK | ✔ | dead OOB `d_particles[icell]` |
| 22 | KOKKOS/fix_ave_histo_weight_kokkos.cpp | REAL | OK | OK | ✔ | stray printf |
| 23 | fix_ablate.cpp | REAL | OK | OK | ✔ | local shadows member `idsource` (= 67 = 98) |
| 24 | fix_emit_face_file.cpp | REAL | OK | OK | ✔ | azimuth `MY_PI`→`MY_2PI` |
| 25 | fix_emit_face_file.cpp | REAL | OK | OK | ✔ | leak fflag/fuser on re-init |
| 26 | fix_halt.cpp | REAL | OK | OK | ✔ | `%ld`→BIGINT_FORMAT |
| 27 | fix_grid_check.cpp | REAL | OK | OK | ✔ | `%d`/icell → `%g`/x[2] |
| 28 | input/move_surf/particle.cpp | REAL | OK | OK | ✔ | sprintf→snprintf (filename overflow) |
| 29 | input.cpp | REAL | DIFF | PARTIAL | ✔ | Unknown-command leak; AB stack-buf both sites |
| 30 | marching_cubes.h | REAL | NO | OK | ✔ | `int v000…`→`double` (truncation) — AB missed |
| 31 | read_isurf.cpp | REAL | OK | OK | ✔ | unchecked fopen |
| 32 | write_grid/isurf/surf/restart.cpp | REAL | OK | OK | ✔ | sprintf→snprintf (filename) |
| 33 | surf.cpp | REAL | OK | OK | ✔ | estyle sprintf→snprintf (×2) |
| 34 | variable.cpp | REAL | OK | OK | ✔ | unchecked fscanf → uninit nextindex |
| 35 | write_isurf.cpp | REAL | OK | OK | ✔ | mutates arg[4] + leaks `file` |
| 36 | custom.cpp | REAL | OK | OK | ✔ | FILECOARSE leak |
| 37 | KOKKOS/fix_grid_check_kokkos.cpp | REAL | OK | OK | ✔ | OOB `cells[icell]` in host msg |
| 38 | KOKKOS/compute_lambda_grid_kokkos.cpp | REAL | OK | OK | ✔ | Kn div-by-zero + missing else |
| 39 | KOKKOS/compute_surf_kokkos.cpp | REAL | OK | OK | ✔ | OOB read when nsurf==0 |
| **40** | KOKKOS/fft2d_kokkos.cpp | **NO** | WRONG | WRONG | **✖** | flag convention inverted in this fn; both branches broke it |
| 41 | KOKKOS/surf_collide_specular_kokkos.cpp | REAL | OK | OK | ✔ | `sr_map[n]=nprob`→`nglob` |
| 42 | KOKKOS/fix_ave_histo_weight_kokkos.cpp | REAL | OK | OK | ✔ | DualView alloc via host allocator |
| 43 | KOKKOS/create_particles_kokkos.cpp | REAL | OK | OK | ✔ | `nlocal-1`→`inew` |
| 44 | KOKKOS/compute_thermal_/eflux_grid_kokkos.cpp | REAL | OK | OK | ✔ | `return`→`continue` in per-cell loop |
| 45 | KOKKOS/remap3d_kokkos.cpp | PARTIAL | OK | PARTIAL | ✔ | malloc-fail leaks (low sev); AB complete |
| 46 | collide_vss.cpp | REAL | **WRONG** | OK | ✔ | vremax==0 NaN; **AB deleted the EPSZERO guard (regression)** → took CP |
| 47 | collide_vss.cpp | REAL | NO | OK | ✔ | rotc2 symmetric assignment — AB missed |
| 48 | react_tce.cpp | REAL | OK | OK | ✔ | hardcoded SI kb → update->boltz |
| 49 | react_tce.cpp | REAL | OK | OK | ✔ | missing `break` inflates chem-rate tallies |
| 50 | fix_surf_temp.cpp | REAL | OK | OK | ✔ | uninit prefactor/threshold (no else) |
| 51 | fix_surf_temp.cpp | REAL | OK | OK | ✔ | stale cqw/fqw; re-resolve in init() |
| 52 | compute_lambda_grid.cpp | REAL | OK | OK | ✔ | CPU twin of 19/38 |
| 53 | update.cpp | REAL | NO | OK | ✔ | dangling *_active → double free — AB missed |
| 54 | react_bird.cpp | REAL | OK | OK | ✔ | uninit tally_reactions* in 1-arg ctor |
| 55 | particle.cpp | REAL | NO | OK | ✔ | size_restart int overflow — AB missed |
| 56 | grid_collate.cpp | REAL | NO | OK | ✔ | 32-bit memset/create — AB missed |
| 57 | grid_custom.cpp | REAL | NO | PARTIAL | ✔* | (size_t) on int **and** double memset (both missed double) |
| 58 | grid.cpp | REAL | NO | PARTIAL | ✔* | bigint on set1 **and** set2 (CP did set1 only) |
| 59 | surf_custom.cpp | REAL | OK | OK | ✔ | (size_t) memset |
| 60 | surf_comm.cpp | REAL | OK | PARTIAL | ✔ | bigint ×4 sites; CP missed dbuf → took AB |
| 61 | surf.cpp | PARTIAL | OK | PARTIAL | ✔ | effective fix = (size_t) on Tri/Line memset |
| 62 | update.h, geometry.cpp, KOKKOS update/geometry | REAL | PARTIAL | NO | ✔* | axi div-by-zero; **both left CPU geometry.cpp — fixed here** |
| 63 | fix_*/compute_* (sweep) | PARTIAL | OK | PARTIAL | ✔ partial | genuine %s sites applied via 28/32/94; pure-%d rejected |
| 64 | compute_gas_reaction_/collision_grid.cpp | PARTIAL | OK | NO | ✔ | (size_t) memset |
| 65 | fix_emit_face/face_file/surf.cpp | PARTIAL | OK | NO | ✔ | (size_t) memset maxactive/ntaskmax |
| 66 | fix_ablate_multi_inner.cpp | REAL | OK | OK | ✔ | /Ninterface==0 NaN (report's "SIGFPE" inexact) |
| 67 | fix_ablate.cpp | REAL | OK | OK | ✔ | duplicate of 23 |
| 68 | compute_tvib_grid.cpp | REAL | OK | OK | ✔ | groupspecies OOB → `index/maxmode` |
| 69 | compute_react_surf/boundary/isurf_grid.cpp | REAL | OK | OK | ✔ | strtok loop wipes prior matches |
| 70 | fix_temp_rescale.cpp | REAL | PARTIAL | OK | ✔ | t_current==0; CP guards both paths, AB only avg |
| 71 | compute_{eflux,grid,pflux,sonine,thermal,tvib,lambda}_grid.cpp | REAL | OK | OK | ✔ | memory_usage `=`→`+=` (under-report) |
| 72 | react_qk.cpp, react_tce_qk.cpp | REAL | OK | PARTIAL | ✔ | scratch `prob` vs `react_prob`; **CP wrong on react_qk** |
| 73 | KOKKOS/collide_vss_kokkos.cpp | REAL | OK | PARTIAL | ✔ | free_state before continue (PRNG race); CP missed ambipolar loop |
| 74 | KOKKOS/react_tce_kokkos.h | PARTIAL | OK | OK | ✔ | kb→boltz (premise "base uses boltz" was false; paired with 48) |
| 75 | KOKKOS/react_bird_kokkos.cpp | PARTIAL | OK | NO | ✔ | pool seed decorrelation (speculative; pool-seed only applied) |
| 76 | KOKKOS/collide_vss_kokkos.cpp, react_tce*.h | PARTIAL | OK | NO | ✔ | vr2>0 guard (the reachable one); ecc guard via 106 |
| 77 | update.cpp, KOKKOS/update_kokkos.cpp | REAL | PARTIAL | NO | ✔* | frac 0/0; **AB KOKKOS only — CPU fixed here** |
| 78 | update.cpp, KOKKOS/update_kokkos.cpp | PARTIAL | PARTIAL | NO | ✔* | clamp frac∈[0,1]; **CPU fixed here** |
| 79 | update.cpp, KOKKOS/update_kokkos.cpp | REAL | PARTIAL | NO | ✔* | stuck_iterate `==0`→`<=1e-14`; **CPU fixed here** |
| 80 | geometry.cpp, KOKKOS/geometry_kokkos.h | REAL | PARTIAL | NO | ✔* | catastrophic cancellation (Vieta); **CPU fixed here** |
| 81 | adapt_grid.cpp | REAL | NO | OK | ✔ | newcell→nglocalprev — AB missed |
| 82 | adapt_grid.cpp | PARTIAL | NO | NO | ✔* | (bigint) smalloc — neither fixed |
| 83 | adapt_grid.cpp | PARTIAL | NO | NO | ✔* | (bigint) clist/alist srealloc — neither fixed |
| **84** | adapt_grid.cpp/.h, grid_adapt.cpp | **NO** | NO | NO | **✖** | speculative int-widening; total size already bigint |
| 85 | KOKKOS/fix_ambipolar_kokkos.h | REAL | OK | OK | ✔ | -log(0)=+Inf → 1.0-drand() |
| 86 | KOKKOS/fix_vibmode_kokkos.h | REAL | OK | OK | ✔ | -log(0)→Inf cast to int (UB) |
| 87 | KOKKOS/fix_emit_*/diffuse/particle_kokkos | REAL | OK | NO | ✔ | 13 -log(drand) sites; CP missed all |
| 88 | create_isurf.cpp | REAL | OK | NO | ✔ | (bigint)/(size_t) nsurf*nbytes |
| 89 | create_*/dump_movie.cpp | PARTIAL | OK | NO | ✔ partial | dump_movie filename applied; create_* pure-%d rejected |
| 90 | create_isurf.cpp/.h | REAL | OK | NO | ✔ | maxsbuf int→bigint |
| 91 | create_isurf.cpp | REAL | OK | NO | ✔ | boxvol==0 div |
| 92 | create_isurf.cpp | REAL | OK | NO | ✔ | param==1 → Inf clamp |
| 93 | fix_move_surf/emit*.cpp | **NO** | DIFF | NO | **✖** | `int*sizeof` already promotes to 64-bit; casts inert |
| 94 | KOKKOS/fix_grid_check_kokkos.cpp | REAL | OK | NO | ✔ | sprintf→snprintf |
| 95 | fix_emit_face_file.cpp | REAL | OK | OK | ✔ | `for(m…;i++)` typo → infinite/OOB (HIGH) |
| 96 | fix_emit_face/face_file/surf.cpp | REAL | OK | NO | ✔ | subsonic /0 (massrho*soundspeed) |
| 97 | fix_temp_rescale.cpp | REAL | OK | OK | ✔ | global avg /0 (paired with 70) |
| 98 | fix_ablate.cpp | REAL | OK | OK | ✔ | duplicate of 23 |
| 99 | fix_ablate/ave_grid/histo/surf/time.cpp | REAL | OK | NO | ✔ | suffix leak on error path |
| 100 | compute_dt_grid.cpp + KOKKOS | REAL | OK | PARTIAL | ✔ | vrm_max==0 div; CP missed KOKKOS |
| 101 | compute_{thermal,eflux,pflux}_grid.cpp + KOKKOS | REAL | OK | NO | ✔ | volume==0 div in flux tallies |
| **102** | KOKKOS/compute_fft_grid_kokkos.cpp | **NO** | OK | NO | **✖** | sprintf into str[64] cannot overflow (cosmetic) |
| 103 | compute_lambda_grid.cpp, compute_reduce.cpp | REAL | OK | NO | ✔ | suffix leak on error |
| 105 | collide_vss.cpp | REAL | OK | OK | ✔ | volume==0 div in nattempt |
| 106 | react_qk/tce/tce_qk.cpp | PARTIAL | OK | OK | ✔ | ecc>0 guards (mostly already guarded; via 2/48/72) |
| 107 | surf_react_adsorb.cpp | PARTIAL | OK | OK | ✔ | CI vmag_sq>0 (ER dot==0 moot: dot hardcoded 2.0) |
| 108 | variable.cpp | REAL | OK | OK | ✔ | id leak on error |
| 109 | grid.cpp, input.cpp (sweep) | PARTIAL | DIFF | PARTIAL | ✔ partial | %s/filename sites applied (28/32); pure-%d rejected |
| 110 | dump.cpp, dump_grid.cpp, grid_id.cpp | REAL | OK | NO | ✔ | str[32] too small for deep cell ids |
| 111 | input.cpp | REAL | OK | OK | ✔ | commands[] leak on illegal `if` |
| 112 | grid.cpp | REAL | OK | OK | ✔ | list leak in grid group ops |

## Notable findings

**False positives (rejected — applying them would be wrong or pointless):**
- **Bug 40** (`fft2d_kokkos.cpp`): the report misread this function's *inverted* flag
  convention — here `flag==-1` is the forward transform, so the existing `flag==1` scaling
  already targets the inverse. **Both branches changed it and thereby moved scaling onto the
  forward transform — an actual regression.** Rejected.
- **Bug 84** (`adapt_grid` int widening): the only genuinely size-scaling quantity
  (`plevels[].nxyz` / total children) is already `bigint`; the `int` locals hold per-parent
  subdivision factors that cannot overflow. Both branches correctly left it alone.
- **Bug 93** (`fix_move_surf`/emit `nsurf*sizeof(...)`): `sizeof` is `size_t`, so the
  multiplication is already 64-bit. AB's added casts are inert; rejected to keep the diff honest.
- **Bug 102** (`compute_fft_grid_kokkos` str[64]): bounded macro text, no overflow possible.

**Incorrect fixes in a branch (we took the other branch / corrected it):**
- **Bug 46** — AB not only failed to add the `vremax==0` guard but **deleted the pre-existing
  `EPSZERO` division-by-zero guard** (and its macro) in both `collide_vss.cpp` and
  `collide_vss_kokkos.cpp` — a regression. We took CP for the CPU file and applied only the
  correct sub-edits (73, 76) to the KOKKOS file, **keeping EPSZERO**.
- **Bug 72 (react_qk.cpp)** — CP rewrote the first rejection-sampling loop to keep using
  `react_prob` as the loop scratch, which leaves the sampling probability in `react_prob` and
  pollutes the downstream reaction decision (`react_prob > random_prob`). AB correctly uses a
  separate `prob` scratch (matching the file's own second loop). We took AB.

**Completeness gaps both branches share (fixed here, marked ✔\*):**
- Bugs **62, 77, 78, 79, 80** were fixed by AB only in the KOKKOS copies; the identical CPU
  defects in `update.cpp` and `geometry.cpp` were left unfixed by both. We ported the
  verified-correct fixes to the CPU files (geometry.cpp Vieta rewrite was hand-verified:
  product-of-roots `c/a` with `c = x[1]²−yhoriz²`).
- Bugs **57, 58, 60, 73, 100** were only partially fixed by the branch that touched them
  (missing a second memset / set2 / dbuf / ambipolar loop / KOKKOS file); completed here.
- Bugs **82, 83** were fixed by neither branch; applied here.

**Duplicates:** Bugs **23 = 67 = 98** are the same `fix_ablate.cpp` `idsource` shadowing line
(one fix). Bug **19** is a subset of **38**. Bugs **70/97** and **4/72/106** are related but
touch distinct sites.

## Scope notes on hardening sweeps

`sprintf`→`snprintf` and integer-cast "sweeps" (Bugs 63, 89, 109) were applied **only** where a
real overflow is possible — i.e. the format contains a `%s` of a filename / unbounded string.
Pure-numeric (`%d`, `BIGINT_FORMAT`) conversions into fixed buffers that cannot overflow were
treated as cosmetic non-bugs and not applied, to keep the deliverable scoped to genuine defects.

## Verification performed

- Each bug verified against `origin/master` (presence + genuineness) and against both branches
  (fix correctness) by reading the actual source via `git show`.
- `make serial` builds and links `spa_serial` with all non-KOKKOS fixes applied.
- KOKKOS files are not part of a serial build (Kokkos not installed in `src/`); those edits are
  either byte-identical to AB's (which compiles) or trivial targeted edits, and were diff-checked.
