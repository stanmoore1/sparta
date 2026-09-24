# Plan: full Kokkos/GPU port of fix rigid (surf-rigid-body branch)

# ACTIVE PLAN (21 Sep 2026): owned bodies as a new option + GPU residency

## Context

The user's H100 numbers at `165a8bc` (40-step 1000-body 2d bench, 1.8M
particles): 1 GPU 2.02 s, 2 GPUs 1.93 s, 4 GPUs 2.05 s.  Particle copies
per step are now zero (the census shows 2 in both the 40- and 80-step
runs, both setup).  With fenced timers the per-stage totals did not move
(set_xv+bounds 0.35, integrate+bbox 0.27, remove inside 0.38, recut 0.48,
sum forces 0.09 s), so those stages are genuinely the fix's own work, and
they are flat in the rank count because every rank does the same O(nbody)
and O(nelem) work for all 1000 bodies: the pose loops, the geometry
kernels (which the census shows launch twice per step each:
`TagFixRigidGeometry/BodyBox/Inflate`), the swept-list enumeration over
every (body, bin) pair, the re-cut candidate enumeration over every
body's region, the contacts, and the per-body Allreduce.  `fix rigid`
today is LAMMPS `fix rigid` (every proc holds every body); the scaling
fix is LAMMPS `fix rigid/small` (each proc holds the bodies near it), as
a new option, with the replicated mode kept bit-identical.

Second goal: fewer per-step host<->device round trips and host loops in
the Kokkos fix ("Tier 1" residency): the host keeps the grid change
journal and the O(nbody) integration; everything sized by elements,
cells or particles stays on the device, and the small per-step transfers
are fused into a handful.

Decisions from the user: owned mode requires `remap incremental` only;
per-body outputs (`f_ID[i][j]`, `write_outfile`) are gathered lazily on
first read per step (an Allgatherv of the owned rows, cached, like
`compute_scalar`); residency at Tier 1 (no device-side journal, no
device restructure); follow LAMMPS `fix rigid/small` as closely as
reasonable and invent nothing new unless necessary; keeping the global
body index (nbody-sized arrays, as the Allreduce design has) is accepted
since it keeps setup, infile/restart, `f_ID[i][j]` and the replicated
mode untouched; a CPU (non-Kokkos) profiling and optimization part
(Part 2) comes before the GPU residency part (Part 3); the LAMMPS Kokkos
port `fix rigid/small/kk` is a reference for the device side.  **All
code is written by Opus 5 agents; the supervising session only briefs,
reviews, validates and commits** (see "Execution model" below).

**Follow LAMMPS `fix rigid/small` as closely as reasonable** (user's
instruction; reference checkout `/home/user/stanmoore1/lammps/src/RIGID/
fix_rigid_small.{h,cpp}`, `comm_tiled.cpp`).  The correspondence, and the
only deviations, which exist because SPARTA has no atoms in a body and
shares the code with the replicated mode:

| LAMMPS rigid/small | SPARTA `bodies owned` |
|---|---|
| body owned by the proc owning the atom closest to its centre; it migrates with that atom in `exchange()`, which on a tiled decomposition drops the point into the proc whose sub-domain contains it (`CommTiled::point_drop_tiled`, using the allgathered `rcbinfo`) | body owned by the lowest-rank proc whose owned-cell bbox contains `xcm` (allgathered `ownboxall`, the `rcbinfo` analogue); re-evaluated by every holder from the same `xcm` after the forward exchange |
| ghost bodies arrive with ghost atoms in `borders()`: every proc whose sub-domain is within the comm cutoff; the user sets a cutoff that covers the body extent, else "Rigid body atoms missing" | ghost bodies on every proc whose owned+ghost bbox overlaps `xcm ± bodycut`; `bodycut` defaults to the largest body radius plus the push cutoff and is user-settable; a body whose swept bbox leaves `xcm ± bodycut` is an `error->one` naming the keyword |
| `struct Body` copied by `memcpy` in `pack_exchange/pack_border` | `struct BodyDatum` (POD, 54 doubles) copied as bytes |
| `initial_integrate()` over `nlocal_body`, then `comm->forward_comm(this)`, then `set_xv()` over local+ghost atoms | `initial_integrate()` over `ownlist`, then the forward exchange, then geometry over `blist` |
| `compute_forces_and_torques()` sums atom forces into local+ghost bodies, `comm->reverse_comm(this,6)` adds ghost partials into the owner in swap order | `sum_tallies()` over `blist`, reverse exchange of 6 partials, owner adds them in source-rank order |
| `final_integrate()` over `nlocal_body`, `comm->forward_comm` of the velocities | `final_integrate()` over `ownlist`; the velocities ride the next step's forward exchange (no consumer needs them earlier) |
| `compute_scalar()` = Allreduce; no `compute_array`; per-body output via `compute rigid/local` | `compute_scalar()` unchanged; `compute_array`/`write_outfile`/restart via a lazy Allgatherv (`gather_all()`, user decision) |
| `body[]` = compact local-then-ghost array, `bodyown/atom2body` maps | global `nbody`-sized arrays plus `bodystatus/ownlist/blist` (deviation: the replicated mode and setup index bodies globally and must stay bit-identical) |
| `fix rigid/small/kk`: `DualView<Body*> k_body`, integrate kernels over `nlocal_body`, device-side force sum and comm packing | Tier 1: `k_bodystep` table synced once per direction per step, partials by kernel; Tier 2 (later): integrate kernels over `ownlist`, device-packed exchange buffers |

## Part 1: `bodies owned` (new option; `replicated` stays the default)

### User interface

`fix ID rigid group-ID bodystyle dstyle ... bodies replicated|owned
[cutoff value]` (default `replicated`; parsed in the optional-keyword
loop at `fix_rigid.cpp:279-359`, `bodymode = REPLICATED|OWNED` next to
the `CUTCELL,INCREMENTAL` enum at line 68).  `cutoff` (the comm-cutoff
analogue, `bodycut`) defaults to `2*rmaxmax*(1+2*EPSSURF) + pushcutoff`
with `rmaxmax` = the largest `rmaxbody` at setup: a contact partner's
COM lies within `2*rmax + pushcutoff` of an owned body's COM, and the
second radius is the motion allowance per step; a larger value lets a
body move more than its radius per step.  Errors: constructor, `owned` with `remap cutcell`
("Fix rigid bodies owned requires remap incremental"); `init()` (`:516`)
and `grid_changed()`, `!grid->clumped` ("Fix rigid bodies owned requires
a clumped grid decomposition": `create_grid ... clump/block`,
`balance_grid rcb`, `fix balance ... rcb`; `fix balance random/proc`
clears it, `fix_balance.cpp:293`; the bench uses `balance_grid rcb
cell`); per step, `error->one("Fix rigid body moved beyond the bodies
cutoff")` when an owned body's swept bbox is not inside `xcm ± bodycut`
(the "Rigid body atoms missing" analogue).  Doc `doc/fix_rigid.txt`:
syntax line (`:43`), a paragraph after the `remap` discussion
(`:459-520`), restrictions (`:836-900`), defaults (`:953`): results on
N>1 procs differ from `replicated` at round-off in the collision force
sum only (partials added in rank order instead of `MPI_Allreduce`), are
identical on 1 proc, per-body outputs are gathered on demand.

### Data model (host, both modes)

Keep every per-body array global and `nbody`-sized (`xcm, vcm, omega,
angmom, quat, xcmnew, quatnew, ex/ey/ez_space, fcm, torque, bbodylo/hi,
bboxeps, rmaxbody, ...`) so setup (`gather_body`, `setup_body*`, infile,
restart, `check_body_attributes`) and every index-based routine stay as
they are; `displace` (body frame) stays replicated (static after setup).
Add:

- `int *bodystatus` (OWNED / GHOST / FAR), `int *bodyowner`,
  `int *bodystamp` (step the last datum arrived).
- `int nown, *ownlist` (owned bodies), `int nblist, *blist` (owned +
  ghost, ascending body index), `int nnewghost, *newghost` (GHOST now,
  FAR last step), `int blistgen` (bumped when `blist` changes; the device
  mirrors re-upload on it).  In replicated mode both lists are the
  identity and every loop below runs over `blist`/`ownlist` in both
  modes with its body untouched, so replicated results are unchanged.
- `double *ownboxall, *procboxall` (nprocs x 6): each rank's owned-cell
  bbox and owned+ghost bbox, two `MPI_Allgather`s in `proc_boxes()`,
  called from `setup()` and `grid_changed()`; the owned bbox computed as
  `Grid::acquire_ghosts_near()` computes `bblo/bbhi` (`grid.cpp:591`),
  factored into `Grid::owned_bbox(lo,hi)`.
- `double bodycut, rmaxmax`; `struct BodyDatum { int ibody, pad; double
  v[54]; }` = xcm, xcmnew, quat, quatnew, vcm, omega, angmom, fcm,
  torque, fpush, tqpush, invmass, invinertia[9], and the remap's
  `prevlo/prevhi/prevxcm` (via `RigidRemap::pack_prev/unpack_prev`);
  derived on receipt as the owner derives them (`xcmmid`, `:895`;
  `ex/ey/ez_space` from `quatnew`, `:965`).  `struct PartDatum { int
  ibody, pad; double f[6]; }` for the reverse exchange.  Static per-body
  data (`massbody, inertia, moi, rmaxbody, rminbody, displace,
  bodystart`) stay replicated from setup.
- Send/receive buffers grown in chunks; one `Irregular` instance
  (`irregular.h:32,36`: `create_data_uniform(n, proclist)` +
  `exchange_uniform(sendbuf, nbytes, recvbuf)`).  `create_data_uniform`
  costs a `Reduce_scatter` and `augment/exchange` a barrier per call
  (`irregular.cpp:143,774`); accepted for the first version (it replaces
  an Allreduce).  If the census shows it, a fixed-neighbor
  point-to-point exchange (LAMMPS `CommTiled` swaps: Irecv/Send/Waitall
  over the procs whose `procboxall` is within `bodycut` of mine) replaces
  it as a follow-up package, not now.

### Ownership and ghost rules

- **Owner** of b = the lowest rank whose owned-cell bbox (`ownboxall`)
  contains `xcm[b]`, else rank 0 (a body outside every box, e.g. exited
  the domain).  Pure function of `xcm`, so every holder agrees.
- **Ghost**: rank r holds b when r's owned+ghost bbox (`procboxall`)
  overlaps `xcm[b] ± bodycut` (start-of-step COM).  This covers every
  consumer on r: the mover (swept surfs in owned and ghost cells,
  `collision_lists` iterates `nlocal+nghost`, `rigid_remap.cpp:226`),
  the re-cut region prev ∪ new, `remove_inside_all`/`inside_any_body` on
  owned cells, `mark_static`, and the contact partners of r's owned
  bodies (their COM is within `bodycut` of a COM inside r's box), as
  long as the per-step check holds: the swept bbox of every owned body
  inside `xcm ± bodycut`, else the `error->one` above.
- **Handoff** rides the next step's forward exchange (the end-of-step
  force totals exist only on the current owner, `final_integrate()` and
  the outfile need them): the owner of step n keeps b through
  `final_integrate`/outfile of n, runs `initial_integrate()` for n+1,
  then sends the complete `BodyDatum` to every rank satisfying the ghost
  rule plus, explicitly, the rank the owner rule names for `xcm`.  Each
  receiver sets OWNED if the rule names itself, else GHOST; a rank
  marks every body it holds and did not receive FAR.  Bodies that turn
  from FAR to GHOST (`newghost`) get their start-of-step geometry
  regenerated (host `body_geometry` with `posesplit = 1`, `:1118`, and
  `body_bbox(ibody,0)`) because `swept_boxes()` reads it.
- **Stale surfs of FAR bodies**: with non-distributed surfs every rank
  holds every surf and `update_surf_copies()` (`:2387`) only rewrites
  `blist` bodies, so FAR bodies' coordinates freeze in `surf->lines`.
  Harmless to the mover (no local cell lists them: a FAR body's box is
  outside the rank's owned+ghost box), wrong for anything that
  re-derives cell lists or writes surfs from the local arrays.  So
  `refresh_all()` (= `gather_all()` + geometry and `update_surf_copies()`
  for all bodies + `body_bins()`) runs exactly on: output steps
  (`output->next == update->ntimestep`), steps a balance/adapt fix will
  run (`ntimestep % nevery == 0`, the test of `modify.cpp:165`), the
  fallback branch of `remap_grid()` (after the 4-int Allreduce at
  `:1272`, before `grid_rebuild()`), and `post_run()`.  One predicate
  `FixRigid::host_surfs_needed()` serves this and the Kokkos
  `refresh_host_surfs()` gate (residency item G).
- **Per-body diagnostics** inside owned loops become `error->one`
  (`:979-990`, `:1204-1209`); the `me == 0`-guarded warnings
  (`:1004-1020`, `:1183-1192`) become per-rank flags reduced once in
  `post_run()` like `end_of_run_delete_warning()` (`:3720`).

### Per-step driver in owned mode (`fix_rigid.cpp` driver at `:738-824`)

```
start_of_step (:738):
  initial_integrate()  (:846)    loop over ownlist; error->one
  exchange_forward()   NEW       owner -> holders (ghost rule + owner
                                 rule), Irregular, unpack, bodystamp;
                                 body_status() rebuilds status/lists
  newghost geometry    NEW       start-of-step geometry of FAR->GHOST
  swept_boxes()        (:831)    over blist; then the bodycut check
  ensure_local_copies() (:2125)  [distributed] bodyneed loop (:2140) over blist
  remap->collision_lists() (rigid_remap.cpp:219, loop :260) over blist
  clear_tally()
end_of_step (:783):
  reset_collision_lists()
  sum_forces() (:1484)           sum_tallies (:1513) over blist at the
                                 global index; exchange_reverse() NEW:
                                 GHOST partials to bodyowner; owner:
                                 own partial, then received partials in
                                 source-rank order; axi_project
  remap->refresh() (:365, loop :372) over blist
  set_xv() (:1031)               set_pose (:1056), body_geometry,
                                 update_surf_copies (blist elements),
                                 body_bins (:3143) over blist
  check_bounds() (:1168)         over ownlist; error->one
  final_integrate() (:1221)      zero over blist; contact->compute()
                                 over ownlist (below); kick over ownlist
  outfile                        gather_all(); write_outfile()
  refresh_all()                  if host_surfs_needed() (output/balance/
                                 adapt steps); fallback case inside
                                 remap_grid() after the Allreduce (:1272)
  remap_grid() (:1260)           recut() region loop (:471) and prev
                                 update (:715) over blist; passes 1-2 and
                                 remove_inside_all use the bins (blist)
```

- **Contacts** (`RigidContact::compute`, `rigid_contact.cpp:221`): loop
  over `ownlist`; each owner evaluates both directions of a pair and
  keeps its own side, reproducing the replicated accumulation order
  [reactions from j<i] [own contacts] [reactions from j>i] (`:231`,
  `:597-606`) by iterating partners from `body_box()` in bin order, so
  contact forces are bit-identical to replicated; the distributed
  proc-0 rule (`:364-365`) becomes "always mine" and the distributed
  Allreduce is dropped in owned mode; static-surf bins over the
  local+ghost arrays (`surf->lines`, not `mylines`), rebuilt in
  `grid_changed()`.
- **Collectives left per step**: none in the common path (the
  `ftbuf` Allreduce at `:1488` is replaced by the reverse exchange; the
  4-int Allreduce of `remap_grid()` and the count Allreduce of
  `restructure_split_cells` stay; `surfs_changed()` stays with
  distributed surfs).  `compute_scalar` keeps its Allreduce.
- **Outputs**: `gather_all()` = `MPI_Allgather(nown)` +
  `MPI_Allgatherv` of `BodyDatum` from owners into every rank's arrays,
  `gathervalid = ntimestep`; called by `compute_array` (`:4129`),
  `compute_vector`, `write_outfile`, restart, `refresh_all()`.
- **Setup**: unchanged and replicated (`gather_body`, `setup_body`,
  checks, `read_infile`, the initial `set_xv` of every body so the grid
  cut sees all surfs), then `proc_boxes(); body_status()` in `setup()`
  (`:687`); `post_run()` ends with `refresh_all()` so the next run's
  setup sees replicated state.  `grid_changed()` (`:3346`): `refresh_all`
  has already run this step, so `proc_boxes()`, the clumped check,
  `body_status()`, then the existing distributed block.
- **Distributed surfs in owned mode** (WP4): `ensure_local_copies()`
  over `blist`, the FAR early-out in the surf scatter, contact bins over
  local+ghost static surfs (dedup by ID), and the requirement `global
  gridcut >= bodycut` with `push`.  Until WP4 lands `bodies owned` with
  `surf->distributed` is an `error->all`.
- **Replicated mode** keeps the Allreduce, the proc-0 rule and full
  loops; every new path is behind `bodymode == OWNED`, and every
  changed loop is `for (m < nblist) ibody = blist[m]` with the body
  untouched, so replicated stays bit-identical (guard: the validation
  matrix).

### Kokkos side (`src/KOKKOS/fix_rigid_kokkos.cpp`, `rigid_remap_kokkos`, `rigid_body_kokkos.h`)

- The device element table stays complete (`displace`, `body`,
  `bodystart` for all `nelem`; static).  Add `k_blist`, `k_lelem` (the
  elements of `blist` bodies, CSR order = `blist` order so per-body sums
  keep the host's order) and `k_bodystat`; rebuilt on the host when
  `blistgen` changes (O(local elements)).  Replicated: identity lists
  uploaded once.
- `device_geometry()` (`fix_rigid_kokkos.cpp:540`): `TagFixRigidGeometry`
  and `Inflate` over `nlelem` with `i = d_lelem(m)`, `BodyBox` over
  `nblist`; `ScatterSurfs` keeps its copy range but returns early for
  `d_bodystat(d_body(elem)) == FAR` (never write a FAR body's copies with
  a stale pose).  The FAR->GHOST pre-pass reuses the same kernels with
  the start-of-step frame (`posesel`, see residency item A).
- `sum_tallies()` (`:837`): kernel over `nblist` writing `d_ft` at the
  global index.  `RigidBodyKK` (device body table of the deletion kernel
  and the remap) built from `blist`; the host pair loops of
  `RigidRemapKokkos::collision_lists()/recut()` (`:113-156`, `:512-551`)
  iterate `blist` until residency item C moves them to the device.
- `refresh_host_surfs()` (`:310`) gated by `host_surfs_needed()`
  (item G); `refresh_all()` in owned mode is what makes the host surf
  arrays complete on those steps, so surf dumps and restarts work with
  non-distributed surfs too.
- `UpdateKokkos::rigid_upload()` (`update_kokkos.cpp:438`) and the
  `ComputeSurfKokkos` `xcmmid` upload keep all `nbody` rows (FAR rows
  stale but unread) until item A fuses them.

### Work packages (Part 1)

- **WP1** keyword + `cutoff`, mode enum, `bodystatus/bodyowner/
  bodystamp/ownlist/blist/newghost/blistgen`, `Grid::owned_bbox`,
  `proc_boxes()`, `body_owner()`, `body_status()`, and every fix/remap/
  contact loop listed in the driver rewritten over `blist`/`ownlist`
  (replicated: identity lists).  Gate: whole validation matrix
  bit-identical in replicated mode.
- **WP2** `BodyDatum/PartDatum`, `exchange_forward()/exchange_reverse()`
  with `Irregular`, handoff, the bodycut check, `newghost` geometry,
  `sum_forces` partials, `gather_all()/refresh_all()/host_surfs_needed()`,
  outfile/restart/`compute_array` paths, per-rank diagnostics, the
  two-way contact pass.  Gate: owned == replicated bit-for-bit on 1
  rank (CPU); the handoff deck at 4 ranks within `approx(rel=1e-10,
  abs=1e-13)` of replicated per step (the `compare_remap_modes`
  tolerance) with the `fpush` columns exactly equal; negative decks
  error as documented.
- **WP3** Kokkos: `k_blist/k_lelem/k_bodystat`, kernels over local
  elements, FAR early-out in the scatter, FAR->GHOST pre-pass,
  `RigidBodyKK` and the remap pair loops from `blist`, the
  `refresh_host_surfs()` gate.  Gate: Kokkos Serial EXACT owned == CPU
  owned bit-for-bit on 1, 2, 4 ranks; emulation build
  (`/home/user/sparta-sync`, 4 ranks) clean; the user's 1/2/4-GPU bench.
- **WP4** distributed surfs in owned mode (see above).  Gate: `--dist`
  suite in owned mode, 1 and 4 ranks, Kokkos EXACT == CPU.
- **WP5** docs (`doc/fix_rigid.txt`), `examples/rigid` owned variants,
  `tools/testing/rigid`: `--owned` flag in `run_tests.py` mirroring
  `--dist` (`:2278-2301`, appends `-var bodies owned`; `OWNED_TESTS`),
  `variable bodies index replicated` + `bodies ${bodies}` on the fix
  line of the `DIST_TESTS` decks plus `crossproc`, `nbody`, `exitbox`,
  `gridchange`, `splitbalance`, `restart.*`; new `in.test.ownership`
  (three fast pushed bodies crossing a 2x2 rcb decomposition, one
  exiting the box, one going FAR then GHOST again on some rank, stats
  every step) and the negative decks `ownedcutcell`, `ownedrandom`,
  `ownedbalrandom`, `ownedfast`.  WP5's decks are written with WP2 so
  WP2's gate can use them.

## Part 2: CPU (non-Kokkos) profiling and optimization (before Part 3)

### Where the CPU time goes today

`/home/user/bench/t_bench40_cpu.log` (`spa_mpi`, 1 rank, 40 steps,
1.8M particles, `SPARTA_RIGID_TIMING=1`) and `t_bench40_cpu_np4.log`:

| stage | 1 rank (s) | 4 ranks (s) | note |
|---|---|---|---|
| Loop | 14.63 | 4.78 | 3.06x on 4 ranks |
| Move | 4.50 | 1.61 | the mover with the swept surf lists |
| Sort | 1.43 | 0.39 | one particle sort per step |
| Modify (= the fix) | 7.79 | 2.47 | |
| collision lists | 1.59 | 0.59 | (body, bin) enumeration + per-cell element box tests, sort/merge per touched cell |
| recut | 3.56 | 1.08 | sub-timers: surf lists 1.06, cuts 0.63, retyping 0.39; ~1.5 s untimed (candidate enumeration, compare, pending apply) |
| remove inside | 2.41 | 0.65 | full particle pass, `inside_any_body()` per particle in an OVERLAP/INSIDE cell (`fix_rigid.cpp:3661-3683`) |
| set_xv+bounds | 0.11 | 0.08 | |
| integrate, forces, contacts, apply split | 0.10 | 0.13 | |

The O(nbody) replicated stages are negligible on the CPU; the CPU
problem is per-cell and per-particle work in three stages, and the
scaling loss (recut 3.3x, collision lists 2.7x on 4 ranks) is enumeration
replicated over all bodies plus imbalance.  Owned mode (Part 1) helps
the enumeration; the rest is single-core efficiency.

### Profiling protocol (a package of its own, before any optimization)

- **Stage timers** are the first cut: `SPARTA_RIGID_TIMING=1` prints the
  table above from `post_run()` (`fix_rigid.cpp:1373-1378`, max over
  procs).  Add sub-timers where a stage has untimed parts: recut
  candidate enumeration, list compare, `apply_pending`, and the
  `compress_rebalance()` inside remove inside; a per-stage
  min/avg/max over procs (imbalance) instead of max only.
- **Sampling profile**: `perf` is not installed in this container
  (install `linux-perf` if the kernel allows, else use the two below);
  `valgrind --tool=callgrind` on `in.bench4` (a 4-step cut of the deck,
  already used once: `scratchpad/kptool/callgrind.out`), reported with
  `callgrind_annotate --inclusive=yes` per function; and gdb sampling on
  the full deck (`gdb -batch -ex run -ex bt` interrupted at intervals,
  the `gdbprof` script in `/home/user/bench`) for the hot lines.
  `gprof` (`-pg`) as a check on call counts.  On the user's machine
  `perf record -g` + `perf report --no-children` on the 40-step deck.
- **Counters** that explain the numbers: per step, the number of
  particles that reach `inside_any_body()`, of candidate cells, of cells
  re-cut, of cells whose list changed, of (body, bin) pairs and of
  element box tests in the collision-list stage, printed with the
  timers.  These make each optimization's effect predictable and show
  when a stage is enumeration-bound versus test-bound.
- Deliverable: a profile table in the plan Status (function, inclusive
  %, calls per step) for 1 and 4 ranks, before and after each item.

### Candidate optimizations (each bit-identical, each its own package)

Ordered by expected gain; every item keeps the same arithmetic on the
same cells/particles in the same order, so the bench and the suites
stay bit-identical (the gate).

- **C1 remove inside: classify cells, not particles.**  Today every
  particle in an OVERLAP or INSIDE cell runs the parity test.  A cell
  cut only by static surfs, or whose bbox overlaps no local body's
  bbox, cannot hold a particle inside a body unless the whole cell is
  inside one; so build, once per step, a per-cell flag over the cells
  with `nsurf > 0` or type INSIDE: 0 = skip all its particles, 1 = test
  each (bbox overlap with a body from the COM bins, or a body element in
  its cut list), 2 = delete all (INSIDE claimed by a body via
  `rigidinside`).  The particle loop then reads one int per particle.
  Expected: most of the 2.4 s (the particles tested drop to those in
  the ~10-15k cells a body actually crosses).  Same test order for the
  particles that are still tested, so identical results.
- **C2 recut: enumerate candidates from the cell bins, not the grid.**
  Time the untimed part first (protocol).  If the region loop
  (`rigid_remap.cpp:471`) flags cells by scanning all cells of the
  bins overlapping prev ∪ new, restrict it to the cells of the bins
  overlapping each body's region (as `Grid::cells_in_box` does) and
  skip a cell whose candidate surf list is unchanged before building
  the list (compare element ids against the current cut list with an
  early exit, before `surf2grid_list`).  Keep the same candidate order.
- **C3 recut surf lists: per-cell radial prefilter before the exact
  overlap test.**  `surf2grid_list` runs the line/tri-vs-cell test for
  every candidate surf; a `rmin/rmax` shell test per (cell, body) from
  the COM bins first (the retyping already has one, `52dfb7c`) removes
  most bodies from most cells.  Same surviving surfs, same order.
- **C4 collision lists: per touched cell, not per (body, bin, cell).**
  The host stage does the 32 element box tests for every cell of every
  bin a body's swept box overlaps; the device version was rebuilt
  around touched cells (34 ms/step).  Port that structure to the host:
  per body, cells of the overlapped bins with the exact cell-vs-swept-
  bbox test first, then the element tests only for cells that pass;
  record (cell, element) hits, then sort/merge per touched cell as
  today.  Same merged rows.
- **C5 mover: shorter swept rows.**  Move is 31% and grows with the
  swept lists: check `nscheck` per step before/after C4; if the rows
  are still long, add the per-element swept-box-vs-cell test at row
  build (already exact) and nothing in the mover itself.  The mover is
  generic SPARTA code and stays untouched.
- **C6 MPI scaling.**  With owned mode in, re-measure 1/2/4/8 ranks;
  remaining loss is imbalance (recut and collision lists follow the
  bodies, particles follow the gas): `fix balance rcb` with a weight
  that includes cells near bodies (the existing `cell`/`part` weights
  plus the per-cell surf count) is the SPARTA-native answer; no fix
  code.
- **C7 allocation churn.**  `callgrind` will show `malloc/free` inside
  the step (per-cell `memory->create` in the cut, list buffers); grow
  persistent buffers in `DELTA` chunks.  Usually 5-10% once the big
  items are gone.

Gate per item: bench and suites bit-identical on 1 and 4 ranks (CPU
and Kokkos Serial EXACT, since the host code is shared), the stage
timer and counters showing the expected change.  The CPU packages are
independent of Part 1 in code (they touch `fix_rigid.cpp:
remove_inside_all`, `rigid_remap.cpp:recut/collision_lists`, and the
timers) and run before the residency packages of Part 3.

Gate for Part 2 as a whole: the CPU profile table redone at 1 and 4
ranks; the 1000-body bench on 1 rank with the fix well under the mover
(today 7.8 s fix vs 4.5 s move).

## Part 3: GPU residency, Tier 1 (after Part 2)

Reference: LAMMPS `fix rigid/small/kk` (`stanmoore1/lammps/src/KOKKOS/
fix_rigid_small_kokkos.{h,cpp}`): the body table is one
`DualView<Body*> k_body` of the same POD struct the host uses, local then
ghost; `initial_integrate`/`final_integrate` are kernels over
`nlocal_body`; `compute_forces_and_torques_kokkos()` zeroes local+ghost
bodies and sums atom forces on the device; forward and reverse comm pack
and unpack on the device (`pack_forward_comm_kokkos`, 29 doubles per
body, `pack_reverse_comm_kokkos`, 6 per body) over cached per-swap body
send lists, through `CommKokkos::forward_comm_device`; the host path is
kept for setup and exchange with `forward_comm_device = 0`.  What we take
from it now (Tier 1): one struct-shaped body table on the device
(`k_bodystep`, item A) synced once per direction per step, per-body
partials produced by a kernel and packed for the reverse exchange from
that table.  What it points to for Tier 2, once Part 1 and Tier 1 are in
and the census says the remaining host cost is the O(nown) integration:
`initial_integrate/final_integrate` as kernels over `ownlist` on the same
table, exchange buffers packed on the device and passed to `Irregular`
(CUDA-aware MPI, as `CommKokkos` does for particles), the host copy
refreshed only for `gather_all()` and outputs.

Each item is a self-contained package, bit-identical, judged by the
census tool (`tools/kokkos/census`, branch `claude/kokkos-kernel-census`)
counts of launches and deep copies per step.  Inventory today on a
non-structural incremental step (non-distributed): HtoD ≈ `k_pose` x2,
bins 2x3, pair lists 4 (collision) + 6 (recut), fallback scalar,
rigidbody, xcmmid, dellist on deleting steps; DtoH ≈ bbox 3x2,
ntouched/nhit 2, `k_ft`, ncand/fallback/nch/nent 4, 12 recut result
views, newtype, ndelete.  Target after Tier 1: ≤ 16 transfers per
non-structural step (HtoD 2, DtoH ≈ 12-14), structural steps adding
only the journal's O(records) uploads.  Order = payoff/risk:

- **G gate `refresh_host_surfs()`** (`fix_rigid_kokkos.cpp:310`): replace
  `surf->distributed ||` with `host_surfs_needed()` (Part 1).  Today a
  full local+ghost surf DtoH every step on distributed GPU runs.
- **A one fused body upload and one geometry pass per step.**
  `k_bodystep(nbody,54)`: columns 0-21 in the mover's `rigid_upload`
  layout (xcm, vcm, omega, invmass, invinertia, xcmmid), so
  `UpdateKokkos::rigid_upload()` (`update_kokkos.cpp:438`) becomes
  `d_rigidbody = fix->d_bodystep` and the move kernel (`:1862`, `:2136`)
  is untouched; then xcmnew, the end frame, the start frame from `quat`,
  rmax/rmin, prevlo/hi/prevxcm.  Uploaded once in `start_of_step()`
  after `initial_integrate` (and the exchange).  The geometry kernels
  take a `posesel` column set: the sweep pass (`device_geometry(1)`)
  already computes every end-of-step corner point inline (`:707-726`)
  and discards it, and `set_xv()` recomputes them after `set_pose()`
  copies the same pose (`fix_rigid.cpp:1052`; `ex/ey/ez_space` are the
  end frame already), so the sweep pass writes `d_bodypt_new/
  d_bodynorm_new` plus the end-only element/body boxes, and `set_xv()`
  becomes a view swap + `ScatterSurfs` + the bbox DtoH.  Removes the
  second `k_pose` upload (`:556`), three launches, the `ComputeSurfKokkos`
  xcmmid HtoD (`compute_surf_kokkos.cpp:160-168`, reads columns 19-21),
  and the remap's `k_bodyparam` (`rigid_remap_kokkos.cpp:501-556`, reads
  prev/rmin/rmax from the fused view and `bboxeps` from the device).
  Body bins: `pack_body_device()` (`:1415`) guarded by a `bodybingen`
  stamp bumped in `body_bins()`, so the three calls per step
  (collision lists `:102`, recut `:486`, remove inside `:1532`) upload
  once, `binstart`+`binlist` packed in one int DualView.
- **B one packed bbox DtoH.** `TagFixRigidBodyBox` writes
  `k_bbox(nbody,7)` (lo, hi, eps); one `sync_host` replaces three
  (`:616-628`); skip the DtoH in `swept_boxes()` unless distributed
  (`ensure_local_copies` needs it) once C is in.
- **C (body, bin) pair lists on the device.**
  `RigidRemapKokkos::pair_lists()`: kernel over `blist` computes the bin
  ranges from the device boxes (swept for collision lists, prev ∪ cur
  for recut), `parallel_scan`, one DtoH of the count, fill with the
  host's body-major z/y/x nesting (`:140-156`, `:544-551`) so the pair
  order is identical.  Removes 4 + 6 HtoD and the host O(nbody x bins)
  loops (the replicated enumeration the GPU profile shows).  Pack
  `d_ntouched/d_nhit` into one 2-int view (`:279-280`).
- **E `GridKokkos::apply_changes()` early-out and persistent staging.**
  `Grid::journal_dirty()` (`ndirtycell||ndirtysinfo||ncutrec||ncollrec||
  collreset||nbinpatch||nmoved`); return at the top (`grid_kokkos.cpp:
  338`) when clean and `ntotal == ncsurfsrows`, without bumping
  `graph_generation` (`:543` bumps on every call today); replace the
  per-call `memory->create/destroy` of `unique` (`:348/546`) and
  `sunique` (`:452/480`) with grown members.
- **J packed recut results.** One int view per changed cell (`chcand,
  chn, chloff, chnsplit, chxsub, cherr, corner[8]`), one double view
  (`xsplit[3]`), entries views `chlist/chmap` (int) and `chvols`
  (double): 12 DtoH -> 4 (`rigid_remap_kokkos.cpp:1211-1222`); fold
  `fallback` into the `nch/nent` scalar view (`:822/925/960` -> 2 DtoH).
- **F split graphs from journal records.** Today any cut record runs
  `build_csurfs_device()` (`grid_kokkos.cpp:647`) with two host loops
  over all split cells + 2 HtoD for the sub->parent map (`:688-722`),
  and any split change runs `wrap_split_graphs()` (`:279`): two host
  `count_and_fill_crs` + 4 allocations + 4 HtoD.  Journal `splitrec/
  subrec` (`journal_split(isplit)` at every `journal_sinfo()` site,
  `grid_surf.cpp:1948, 2091, 2149, 3136`, and in `split_cell_set()`
  `:1563`); `build_split_graphs_device()` from old rows + records through
  a generalized `build_crs` on double-buffered persistent views; a
  persistent `d_rowpar` scattered from `d_csubs`.  Debug check under
  `SPARTA_KOKKOS_DEBUG_SYNC`: rebuild the host CRS after each
  `apply_changes()` and compare.
- **H persistent ScatterView** in `rigid_upload()` (`update_kokkos.cpp:
  471-475`): create on reallocation only, `reset()` otherwise.
- **D device deletion list.** `remove_inside_all_kokkos()` (`:1594-1602`)
  sorts on the host: `Kokkos::sort` on the device list (check
  `Kokkos_Sort` in `lib/kokkos`; SPARTA's `sort_kokkos` uses BinSort) and
  a device `ParticleKokkos::compress_migrate_kokkos(ndelete, d_dellist)`
  that pairs the i-th hole (ascending) with the i-th survivor from the
  top, the host walk of `particle_kokkos.cpp:281-294`, so the surviving
  order is identical.  Only pays on deleting steps; last.
- **R3 contacts (K5)** stays a Tier 2 item: host geometry of the bodies
  in contact, O(nown) after Part 1.
- **Stays on the host, by design**: body integration and MPI (the
  exchanges, the 4-int and 3-bigint Allreduces), the journal apply
  (`set_cell_surfs/apply_cut/split_pending/restructure_split_cells`
  mutate the host `Grid`, authoritative for Comm, balance, dumps), the
  host cut of a cell the device cut fails on.  Explicitly **not**
  Tier 1: a device-side journal, device restructure, the deletion pass
  restricted to candidate cells, a padded CRS (reasons in the audit
  status).

Gate for Part 3: per-step census on the 1-rank bench within the
transfer target above, the geometry kernels once per step, no host
surf copy on a non-output step, bench bit-identical; the emulation
build (`/home/user/sparta-sync`) run after A, B and C especially, since
Serial aliases host and device memory and cannot see a dropped
transfer.

## Execution model (user's instruction)

- **Authoring**: every work package is implemented by an Opus 5 agent
  (`Agent` tool, `subagent_type: general-purpose`, `model: "opus"`),
  one agent per package, in a worktree (`isolation: "worktree"`) so
  packages can run in parallel where independent.  Order (user's):
  Part 1 ownership, then Part 2 CPU, then Part 3 residency.  WP1 first
  (it touches every loop); then in parallel WP2+WP5 (one agent) and the
  CPU profiling package (one agent, host files only); then WP3 and
  C1; then WP4, C4, C2, C3; then C5-C7; then Part 3 in its listed
  order (G+E+H, A+B, C, J, F, D).  Never two agents in the same file at
  once; the supervising session merges each worktree branch into
  `surf-rigid-body` and re-bases the others.
- **Brief per package** (written by the supervising session): the files
  and functions to touch, the invariants (bit-identical replicated
  mode, `SPARTA_KOKKOS_EXACT` + Serial == CPU, no host loop sized by
  cells/elements/particles in the Kokkos path, no `modify(Host,ALL_MASK)`
  inside a step, SPARTA coding conventions from this plan), the tests to
  run before reporting, and the commit message format (with the
  attribution trailers, no model identifiers in code or commits), and
  the user's comment style: short, terse, to the point (one or two
  lines per block; the rationale lives in this plan, not in the code).
- **Review**: the supervising session reads every diff line by line
  against the brief, runs the validation matrix itself (or through a
  second agent with `run_in_background`), and sends the agent back with
  findings via `SendMessage` until the package is clean.  It writes no
  bulk code; at most a few-line review fix, and preferably none.
- **Checkpoints (robustness to the session being killed)**: every
  agent brief carries a "Checkpoints" section (template in
  `scratchpad/briefs/wp1.md`): a progress file
  `scratchpad/briefs/<wp>.progress.md` (base commit, checklist of the
  scope items with done / in progress / not started, validation runs
  passed, next action) rewritten at every completed step, and a local
  "WIP <wp>: <step>" commit on `surf-rigid-body` after every step that
  compiles in both builds (never pushed), squashed into the package's
  single final commit at the end.  A resumed or replacement agent reads
  the progress file and `git log` first and continues from the
  checkpoint.  The supervising session keeps its own state in this
  file's Status section (package in flight, agent brief path, review
  findings outstanding) and in the task list, so a fresh session can
  pick up the supervision the same way.
- **Commit and push**: one commit per package on `surf-rigid-body`
  (merge the worktree branch, no squash of history needed), pushed after
  the gate passes; the plan file's Status section updated per package.

## Verification (both parts)

- Replicated mode, every package: `tools/testing/rigid/run_tests.py`
  63/63 on 1, 2 and 4 ranks, 33/33 `--dist`, CPU and Kokkos Serial EXACT
  (`-k on -sf kk`), OpenMP 2 threads; the 1000-body bench (`/home/user/
  bench/in.bench40`) bit-identical to `ref_bench40_np1.log` on 1 and 4
  ranks; the 8-sphere 3d deck bit-identical.
- Owned mode: 1-rank owned == replicated bit-for-bit (CPU and Kokkos
  EXACT); Kokkos EXACT == CPU on 2 and 4 ranks; emulation build
  (`KOKKOS_DEBUG_SYNC`, `SPARTA_KOKKOS_WATCH/STALE_STRICT`) 4 ranks clean
  on the owned suite; the handoff deck checked against replicated for
  conservation (total momentum/energy of the bodies, particle count) and
  for `f_ID` output equality on 1 rank; ASan on the handoff deck.
- Residency: census tool per-step launch/copy counts before and after
  each R item on the Serial bench; the user re-runs the H100 1/2/4-GPU
  bench after WP3 and after Part 3 (expect the flat stages to drop by
  about the rank count).

## Original plan and status follow

## Context (original)

The 1000-body benchmark (`in.movie1000`: 2d, 1131x1131 = 1.28M cells, 7.2M
particles, 1000 circles x 32 segments = 32000 non-distributed surfs,
push-off on, `remap incremental`, `gridcut 0.0`, 4 ranks / 4 H100s) runs
badly under `-k on -sf kk` because `fix rigid/kk`
(`src/KOKKOS/fix_rigid_kokkos.cpp`) is still the host fix bracketed by
device<->host transfers.  Per step today:

- **Structural split path every step.**  The bodies create and destroy
  split cells on essentially every step (fc1866a), so `end_of_step` runs
  `split_rebuild()` (`src/fix_rigid.cpp:4129-4213`: `unset_neighbors`,
  `remove_ghosts`, `remove_marked_cells`, `setup_owned` with 4 Allreduces,
  `acquire_ghosts` (MPI on 4 ranks), `reset_neighbors` = O(nlocal+nghost)
  hash lookups, `notify_changed`) and then
  `GridKokkos::resync_after_host_change()` (`grid_kokkos.cpp:330`:
  `modify(Host,ALL_MASK)` = full grid HtoD, `update_hash()` = host rebuild
  of the Kokkos UnorderedMap over every cell + copy, `wrap_kokkos_graphs()`
  = host CRS rebuild + upload, `sorted_kk = 0`).  This is the per-step
  floor, not a rare path.
- **Host work sized by cells-near-bodies and elements x cells**:
  `swept_assign_all()` (~1.4M box tests/rank), `incremental_recut()`
  (`Cut2d::surf2grid_list` + `Cut2d::split` on every cell a body surf
  overlaps, ~10-15k cut cells/rank plus ~30k candidate cells with parity
  tests), `record_oldinside()`, `push_off()` (replicated on every rank),
  geometry regeneration, bboxes, bins.
- **Bulk HtoD every step even without a structural change**:
  `host_begin/host_end` and `start_of_step` mark the whole grid and all
  surfs host-modified, so ~320k+ghost `ChildCell` (64-byte aligned) and
  `ChildInfo` records and all lines are re-uploaded up to twice per step,
  and `wrap_kokkos_graphs()` rebuilds the per-cell surf CRS on the host and
  uploads it up to three times per step.
- **Remaining particle-sized traffic**: `assign_split_kokkos()` deep-copies
  the whole `d_plist` to the host; `remove_inside_all_kokkos()` packs
  `celltype/cellnsurf` over all cells on the host every step and
  round-trips the deletion list for a host sort; any incremental fallback
  (`grid_rebuild`) pulls the full particle array to the host.
- The deck's two `dump image` commands every 15 steps force
  `particle_kk->sync(Host,ALL_MASK)` (200 MB/rank DtoH) from
  `UpdateKokkos::run()` (`update_kokkos.cpp:589`); that is output, not the
  fix, and should be dropped when timing the fix.

**Measured profile on the GPU (user, current branch head)**: host C++ is
~85% of the runtime, led by `split_rebuild()`'s four remaining whole-grid
scans and `incremental_recut()`; host<->device copies are ~15% (the
particle array is down to 36/37 crossings per 40 steps, but `grid:cells`
doubled to 81 from the sync added in ff90c93); GPU kernels are ~4%.  The
previous agent's next intended step was to make
`GridKokkos::wrap_kokkos_graphs()` incremental (it rebuilds all three CRS
graphs over every cell from scratch, on the host).

The whole-grid host scans on a structural step, all O(nlocal+nghost) or
O(nlocal), are: `Grid::unset_neighbors()`, the compaction loop of
`Grid::remove_marked_cells()` (`grid_surf.cpp:1646`), `Grid::setup_owned()`
(`grid.cpp:430`, plus 4 Allreduces), `Grid::reset_neighbors()`
(`grid.cpp:1585`, a hash lookup per neighbour), the bbox scan in
`Grid::acquire_ghosts_near()` (`grid.cpp:591`) on multi-rank runs, the
three host `count_and_fill_crs` passes of `wrap_kokkos_graphs()`
(`grid_kokkos.cpp:195-264`) and the host UnorderedMap rebuild in
`GridKokkos::update_hash()` (`grid_id_kokkos.cpp:21`).  Each of these is
removed by Phase S below, not made incremental one at a time.

Goal: the device is authoritative for everything sized by cells, surfs,
elements or particles; per-step host<->device traffic is O(nbody) plus
O(changed cells); the only host work left is O(nbody) body integration,
the MPI reductions, and O(changed cells) bookkeeping.  No host loop over
all cells on any step.

## Lessons from the branch history (fc1866a..ff90c93)

The previous agent measured on a 1000-body 2d deck (566^2 grid, 1.8M
particles, 40 steps, 1 GPU): 9.33 s -> 5.79 s, i.e. still ~145 ms/step
where the move itself is a few ms.  Its commit messages establish:

1. **The structural split path fires every step** (see Context).  The
   1-GPU measurement had no ghost cells; on 4 ranks ghost re-acquisition
   is added on top.  This gets its own phase (S) placed before the kernels.
2. **A kernel only pays once the syncs around it are gone.**  The device
   split-assign pass was correct but slower than the host pass it replaced
   (6.33 -> 7.90 s, 6847134) because it needed a `wrap_kokkos_graphs()`
   first, and the particle round trip it saved was still spent by the host
   code around it (`grid:cells` copies 41 -> 81 per 40 steps, ff90c93).
   Hence Phase A (no bulk syncs) precedes every new kernel, and each kernel
   is judged by the copy counters, not in isolation.
3. **Measurement tooling that worked**: per-DualView deep-copy counts and
   seconds (`particle:particles copies 76/77 -> 36/37`, `grid:cells 41 ->
   81`), whole-run time on a fixed 40-step deck, and counters on every
   `hashcurrent = 0` site to find the live rehash trigger after reasoning
   had pointed at the wrong routine twice (dca967b).  Phase 0 makes this a
   permanent facility.
4. **Bug classes to design out**: device structs carry raw host pointers
   (`SplitInfo::csplits/csubs`, `ChildCell::csurfs`) which a kernel must
   never dereference (cudaErrorInvalidAddressSpace, ff90c93); CRS graphs
   indexed by `isplit`/`icell` go stale the moment the host restructures
   cells (ASAN heap-buffer-overflow in `split2d_kk`, 6847134); cached
   `Crs` handles in `UpdateKokkos` (`update_kokkos.cpp:736-738`) and the
   fix (`fix_rigid_kokkos.cpp:538-540`) go stale after any rebuild; a
   functor copy of the fix double-frees host arrays without the
   `copy/copymode` guard (fc1866a).  The new design: every list access on
   the device goes through CRS graphs, every graph rebuild bumps a
   generation counter that consumers re-take handles from, and the host
   pointer fields are never read on the device.
5. **Toolchain**: ASAN on the Kokkos-Serial build needs
   `OMPI_MCA_memory=^patcher`; compute-sanitizer reports only OpenMPI
   `cuMemRetainAllocationHandle` noise, so "no device error" means host
   memory.
6. **Correctness constraints**: the split-cell reassignment is not optional
   (dropping it moves the trajectory, ff90c93); `hashcurrent` must survive
   `remove_marked_cells()`/`remove_ghosts()` (dca967b); the `crossproc`
   test differs in the last bit between distributed and non-distributed
   surfs already on the CPU, so it is not a Kokkos regression.
7. **A latent gap in the current kk path**: `sort_for_split_rebuild()` is a
   no-op for rigid/kk on the strength of "7149 cells moved, 0 particles
   relabelled" over the test suite (1e78aaa).  `remove_marked_cells()`
   (`grid_surf.cpp:1646`) fills a hole with `cells[nlocal-1]`, always a sub
   cell but not necessarily one of a *changed* split cell, and relabels its
   particles by walking the host `cinfo.first/next` list, which rigid/kk
   leaves stale.  With hundreds of split cells per rank a moved sub cell
   can hold particles whose device `icell` is then wrong.  Phase S replaces
   this with an explicit old->new cell index map applied on the device.
8. **Per-stage kill switches** (`SPARTA_NO_ASSIGN_KK` pattern) were how the
   previous agent A/B-tested; keep one env switch per new stage during
   development and remove them at the end.

## Design principles

1. **Dual-write, no bulk DualView syncs inside a step.**  The fix keeps
   the host and device copies of `cells`, `cinfo`, `sinfo`, the surf
   arrays and the per-cell lists *both* current by applying every change
   as a compact record to both sides (O(changed) each), and never raises a
   DualView modified flag on grid or surf inside a step.  Reason (from the
   review): emit fixes' `grid_changed()` runs on the host on every
   `typechanged` step (`fix_rigid.cpp:1388-1391`; `FixEmitFace::create_task`
   reads host `cinfo.type/corner`, `cells.csurfs`, `lines`,
   `fix_emit_face.cpp:320-367`), grid-style variables read host cells at
   every output (`variable_kokkos.cpp:137`), fix balance/kk packs host
   `csurfs` (`grid_comm.cpp:74,116`), and any `sync(Host,CELL_MASK|
   SINFO_MASK)` with the device flagged newer would overwrite the host
   pointer fields `csurfs/csplits/csubs` with stale values.  Dual-write
   makes every host consumer correct without a materialisation step and
   makes `sync()` in either direction a no-op.
2. **Per-cell surf graphs are built on the device, never on the host.**
   A device count/scan/fill over persistent, grown `row_map/entries` views
   (no `Kokkos::count_and_fill_crs`, which allocates three views and fences
   per call) is O(cells + entries), ~1 ms on an H100 for this grid.  Two
   graphs: `d_csurfs` = the cut-pipeline (base) list, read by `split2d/3d`
   and the re-cut; `d_csurfs_move` = base + swept elements, read only by
   the mover's collision loop.  `d_csplits/d_csubs` are rebuilt the same
   way from `d_sinfo`.
3. **The structural split change is applied in place** (Phase S): only sub
   cells are ever added or removed and only sub cells ever move
   (`remove_marked_cells` swaps `cells[nlocal-1]` into the hole), so the
   change is a set of sub-cell slot edits plus a shift of the ghost block
   by `delta = nadded - nremoved`; neighbour links, hash entries and the
   halo index that refer to ghosts are patched by `+= delta` instead of
   the unset/remove/acquire/reset cycle, on both host and device.
4. **Every O(elements) and O(cells-near-bodies) loop becomes a kernel.**
   The element table lives on the device; the per-body table (~1000 x a
   few doubles) stays on the host and moves as a few KB per step.
5. **The cut is ported in steps**: hybrid first (device candidate
   generation, compact host cut of only the cells that need it, compact
   dual-write), then `Cut2dKokkos::split` and `Cut3dKokkos::split` (both
   in scope; 2d first, `Cut3d` reuses the 2d cut per face).
6. **`SPARTA_KOKKOS_EXACT` stays bit-identical to the CPU code through
   small in-kernel bifurcations only**: the same kernel runs in both
   builds; under the macro a per-body reduction is done by one thread in
   the host's summation order, otherwise by a team reduction or atomics
   (`collide_vss_kokkos.cpp:98,177,419,600` pattern).  No host fallback
   under EXACT, so the default build is not slowed.  Lists are sorted
   (swept extras, deletion list by flag + scan) in both builds.  Note: the
   Serial-EXACT build is the bit-identical one; CUDA differs at round-off
   through FMA contraction in the geometry unless `--fmad=false`.
7. **Distributed surfs are handled by every device stage from the start**:
   the element table carries every local copy (`copy_index/copy_elem`)
   and the owned range (`olist_own/olist_elem`); force sums run over all
   local copies as the host tally loop does (`fix_rigid.cpp:1128-1133`);
   push-off pair/boundary terms follow the proc-0 rule; an
   `ensure_local_copies()` append (host, rare) re-uploads the element table.
8. **`ModifyKokkos` interface stays as today**: `kokkos_flag = 0`,
   `datamask_read/modify = EMPTY_MASK`; particle DualView flags are managed
   inside the fix (setting `PARTICLE_MASK` in the datamasks would double-
   flag the particle DualView on every host fallback and during fix balance
   pack/unpack, which is a `Kokkos::abort("Concurrent modification")`).
   `auto_sync` is forced to 1 around the host structural paths that call
   `GridKokkos::grow_cells/grow_sinfo` and `SurfKokkos` grows (they
   `resize` on the device and rely on auto-sync to bring the host back),
   and 0 around the fix's own device passes, as today.
9. **Stage-wise refactor, not a rewrite.**  `FixRigid::start_of_step()` /
   `end_of_step()` are split into virtual stages so the CPU path is
   unchanged and `FixRigidKokkos` overrides one stage at a time; each stage
   keeps a host fallback behind an env switch during development.

## Design review of the current code

Verdict: the *algorithm* of the body/grid coupling is sound and worth
keeping; the *implementation* puts Grid's and Surf's data manipulation
inside the fix, and that is what makes it 5620 lines, hard to follow, and
hard to port.  The physics parts (integration, recoil, contact law,
density properties, watertight/enclosed checks, axisymmetric handling)
are careful, well commented, and tested; they need reshaping, not
rethinking.

### What the body/grid interaction does today, and what is right about it

Per step, with the grid always holding the geometry of the bodies at the
*start* of the step (= the end of the previous one):

1. `start_of_step`: integrate to the end-of-step pose; compute per-element
   swept boxes (start U end position, plus the arc bulge); add every
   body element to the *collision* list of every cell its swept box
   overlaps (`swept_assign_all`), so a particle anywhere on the body's
   path is tested against the analytically moving surf in the mover.
   The *cut* list (what `split2d/3d` index in lockstep with `csplits`)
   is left alone; for a split cell the augmented list goes on the sub
   cells only.
2. mover: moving-surf intersection with the body-frame path, collision
   in the wall rest frame, recoil on the finite-mass body.
3. `end_of_step`: sum forces; commit the pose and write the new geometry
   into the Surf arrays; contacts; then re-cut only the cells in the
   union R of each body's old and new bbox (`incremental_recut`): cells
   whose surf list changed or which contain a moving surf are re-cut with
   `Cut2d/3d::split` (volume, corner marks, split map); uncut cells in R
   are typed INSIDE/OUTSIDE by a parity ray cast of their centre against
   the body elements; cells a body interior vacated (recorded pre-move by
   `record_oldinside`) go back to OUTSIDE; piece-count changes are queued
   and applied by `split_rebuild`; ghost split cells are re-derived
   locally from the replicated bodies.  Finally particles inside a body
   are deleted (a safety net; swept coverage should leave none).

This is the right decomposition: the moving-surf test makes the grid's
half-step lag harmless for collisions, the swept list keeps the cut list
consistent with `csplits`, the parity test types cells locally without
the collective flood fill, and replicated bodies let every rank re-derive
the same ghost split.  Keep all of it.

### What is wrong with how it is built

1. **The fix mutates Grid's data structures directly** (`cells[].nsurf/
   csurfs`, `cinfo[].type/volume/corner`, `sinfo[].csplits/csubs/xsub/
   xsplit`) and owns the memory behind them through three `std::map`
   registries keyed by cell ID, with copy-back logic in the destructor
   and in `grid_changed()` (`fix_rigid.cpp:4728-4900`).  Grid has no
   primitive for "replace this cell's surf list" or "re-split this cell
   in place", so the fix invented them outside Grid.  Every later bug in
   the Kokkos work (stale graphs, pointer fields on the device) is a
   consequence: the device side had no way to know what changed.
2. **One field carries two meanings.**  `cells[].csurfs` is both the cut
   list and the collision list; the swept mechanism overwrites it for a
   step and restores it (`modified/nsurf_saved/csurfs_saved/cpage`,
   `:3754-3915`), with the "split parent keeps its length" rule as a
   comment rather than a type.  The device port already had to separate
   the two (`d_csurfs` vs a move graph), so the host should too.
3. **Piece-count changes are a hand-rolled transaction** (`PendingSplit`,
   `split_pending/split_rebuild/split_update/split_ghost_drop`, `:4047-
   4258`) because Grid only knows how to restructure split cells inside a
   full rebuild.  The rebuild machinery it then calls (`unset_neighbors`,
   `remove_ghosts`, `acquire_ghosts`, `reset_neighbors`, `notify_changed`)
   is the measured per-step floor.  Only sub cells are ever added or
   removed, so Grid can do this in place.
4. **A pre-move snapshot pass exists only to remember state.**
   `record_oldinside()` (`:3989-4036`) ray-casts cell centres against the
   pre-move geometry so pass 2 can tell "INSIDE because of a body" from
   "INSIDE because of static surfs".  A per-cell `int rigidinside`
   (body index that owns the cell's interior, -1 otherwise), set wherever
   a cell is typed INSIDE by a body and cleared when it leaves, records
   the same fact with no pre-move pass and no ordering constraint on the
   geometry commit.
5. **A fallback for a case the parity test can resolve.**
   `FALLBACK_UNKNOWN` (`:4540`) triggers a full re-map when the cut
   leaves the corner marks UNKNOWN because every surf only touches the
   cell faces.  For a cell whose surfs are body elements, the parity test
   of the cell centre gives the marking exactly (the bodies are closed);
   only a cell whose *static* surfs produce UNKNOWN needs the flood fill,
   and those cells were already resolved by the initial full pipeline and
   are not re-cut unless a body surf enters them.  Resolve locally; keep
   the full re-map only for `FALLBACK_SURFMAX` and `remap cutcell`.
6. **Two indexing schemes for the same thing.**  Non-distributed surfs
   use `slist`, distributed use `copy_index/copy_elem` + `olist_own/
   olist_elem`, and `lblist` serves both; `update_surf_copies`,
   `gather_body`, `check_body_attributes`, `push_bins` each branch on
   `surf->distributed`.  One scheme (every local copy as `copy_index/
   copy_elem`, `ncopy = nsurf` for non-distributed) removes the branches.
   `ensure_local_copies()` (`:1966-2113`) restructures Surf's arrays and
   re-indexes ghost cells' lists from inside the fix; that is Surf's job.
7. **Per-surf maps in three places.**  `irigid` (fix), `rigidmap`
   (Update), `body[]`/`idmap` (fix): the mover needs surf -> body, the
   force sum needs surf -> element.  Build both as flat arrays in one
   place when the surf arrays change.
8. **Update is the wrong owner** for `rigid_cell_box` (a box -> cell bin
   index, a Grid utility), `init_rigid` (calls back into the fix for
   distributed copies), and the rigid bins' invalidation.
9. **Force gathering through a user-defined compute** with six
   validation checks at `init()` (`:629-648`), `xcmmid` pushed into the
   compute, and a per-row hash lookup at every step.
10. **Flag soup**: `listschanged`, `typechanged`, `splitchanged`,
    `copiesappended`, `insplitrebuild`, `pbodyflag`, `npending` exist to
    pass state between stages and to the kk subclass; the kk subclass
    adds three virtual hooks (`particles_to_host`, `sort_for_split_
    rebuild`, `combine_split_all`) whose only purpose is to dodge syncs.
11. `start_of_step()` (200 lines) and `end_of_step()` (360 lines) mix
    integration, MPI, geometry, checks, contacts and the re-map policy.

### Target design of the body/grid interaction

Same algorithm, with each piece of data owned by one class and each
per-step change made through a primitive that both the host and the
device can apply.

```
Grid / GridKokkos
  cells_in_box(lo,hi,&list)          box -> candidate owned+ghost cells
                                     (moved from Update::rigid_cell_box)
  set_cell_surfs(icell,n,list)       cut list; Grid-owned pages,
                                     compacted by threshold like compress()
  set_collision_surfs(icell,n,list)  per-step collision list (default =
                                     cut list; mover reads it, split2d/3d
                                     and the cut read the cut list)
  reset_collision_surfs()            all cells back to the cut list
  set_cell_type(icell,type,volume)   uncut cell: type, corners, volume
  recut_cell(icell)                  Cut2d/3d split of the cell's cut list;
                                     same piece count -> applied in place;
                                     changed count -> returned as pending
  restructure_split_cells(n,pending) piece-count changes in place: sub
                                     cells appended/removed, ghost block
                                     shifted, hash/neighbour/halo patched,
                                     old->new index map returned
  changelist                         journal of the above, consumed by
                                     GridKokkos::apply_changes()
  rigidinside[nlocal]                body owning a cell's interior, or -1

Surf / SurfKokkos
  add_local_copies(...)              distributed copies + ghost re-index
                                     (from FixRigid::ensure_local_copies)
  rigidmap[], elem_of_surf[]         per local+ghost surf, built together

RigidRemap / RigidRemapKokkos (helper class, like Cut2d)
  collision_lists()                  swept boxes -> cells_in_box -> per
                                     element box test -> set_collision_surfs
  recut()                            R per body -> cells_in_box -> candidate
                                     surfs (static + binned bodies + radial
                                     prefilter) -> surf2grid_list -> compare
                                     -> recut_cell / set_cell_type by parity
                                     -> rigidinside retyping -> pending list
                                     -> grid->restructure_split_cells()

RigidContact / RigidContactKokkos (helper class)
  bins over static surfs, compute(fpush,tqpush)

FixRigid / FixRigidKokkos
  body table, dynamics, force tally, driver, output
```

The fix's per-step driver then reads:

```
start_of_step:  initial_integrate(); set_xv(XNEW);
                if (distributed) surf->add_local_copies(...);
                remap->collision_lists();
end_of_step:    grid->reset_collision_surfs(); sum_forces();
                commit pose; set_xv(XCUR); contact->compute(); final_integrate();
                remap->recut(); remove_inside();
```

`FixRigidKokkos`, `RigidRemapKokkos` and `RigidContactKokkos` override the
same methods with kernels and nothing else; `GridKokkos::apply_changes()`
is called once at the end of each of the three stages.

## Structural refactors (Plimpton style; LAMMPS fix rigid as reference)

LAMMPS `FixRigid` is the model for the fix itself: per-body `double**`
arrays owned by the fix, `initial_integrate()` / `final_integrate()` /
`set_xv()` as the per-step skeleton, forces gathered by the fix, and the
KOKKOS port keeping the same arrays as DualViews with per-body kernels.
SPARTA adds the grid the surfs cut through, and that piece belongs in
`Grid` (section above).  Each item says what it removes and where it lands.

1. **Grid owns every cell mutation; the fix never touches `cells[]`,
   `cinfo[]`, `sinfo[]` or their pointers.**  The primitives listed in
   the target design, all O(1) per cell and all with a `GridKokkos` twin.
   Retires the fix's `registry/csplitreg/csubreg` maps, `registry_replace/
   remove`, `copy_registry_to_grid`, `free_registry`, the split registry
   twins, `csplits_alloc/csubs_alloc`, `split_pending/split_rebuild/
   split_update/split_ghost_drop` (~900 lines), `insplitrebuild`, and
   `record_oldinside` (replaced by `rigidinside`).  `fix move/surf` becomes
   a second client of the same primitives.
2. **Collision list as a Grid concept, not a pointer swap.**  Host:
   per-cell `ncoll/ccoll` beside `nsurf/csurfs`, equal to the cut list
   unless augmented; device: `d_csurfs_move`.  `Update::move` reads the
   collision list.  The `modified/nsurf_saved/csurfs_saved/cpage` machinery
   (~150 lines) goes away; CPU and device paths become the same design.
3. **The fix gathers its own forces; no `compute surf` in the loop.**
   LAMMPS `FixRigid` sums per-atom forces into `fcm/torque` itself.  Here
   the mover already calls `surf_tally()` on every active surf compute
   per collision (`update.cpp:1694`, `update_kokkos.cpp:2077-2098`); give
   the fix a `surf_tally(isurf, icell, iorig, ipart, jpart)` of the same
   shape (host: per-surf 6-value array `ftally[nsurf][6]` summed per body
   in element order; device: the same array as a ScatterView, exactly
   `ComputeSurfKokkos::surf_tally_kk` minus the column dispatch, then the
   K3 per-body reduce).  This removes the compute-ID argument, `csurf`,
   `force_torque_colcheck()`, `com_rigid()`, `mixture_covers_all_species()`,
   the `xcmmid` upload into `ComputeSurfKokkos`, the `idmap` lookup per
   tally row, and the `tallyinfo()` compression from the per-step path;
   bit-identical to today's numbers because the per-collision expression
   is the same and the per-body sum order is fixed.  **User-visible**:
   `fix ID rigid group-ID bodystyle dstyle ...` (compute-ID dropped; a
   `compute surf ... com rigid` still works for output).  Decided: do it
   (user: refactors that clean up the code or ease the port are wanted).
   Update `doc/fix_rigid.txt`, `examples/rigid/in.*` and their gold logs,
   every `tools/testing/rigid/in.test.*`, `tools/rigid/replicate.py`
   output and the 1000-body deck accordingly, in the same commit.
4. **One per-body state table, shared by everyone.**  Today the body
   state is copied three times per step: `UpdateKokkos::rigid_upload()`
   packs 19 doubles/body, `ComputeSurfKokkos::pre_surf_tally()` uploads
   `xcmmid`, and `pack_body_device()` uploads the fix's own arrays.  Do it
   as LAMMPS `fix rigid/kk` does: the fix owns `k_xcm, k_vcm, k_omega,
   k_quat, k_xcmnew, k_invmass, k_invinertia, k_ex/ey/ez_space, k_bbox`
   DualViews of the same `double**` arrays (`memoryKK->create_kokkos`),
   flagged host-modified after `initial_integrate()`/`final_integrate()`
   and synced once; the mover and any compute read the fix's device views.
   `rigid_upload()` and its packed layout are deleted.
5. **Element table as the single space-frame geometry.**  `displace`
   (body frame) is the source; `bodypt/bodynorm` (space frame) is derived
   by `set_xv()`, and the Surf copies are a scatter of it.  Rename
   `update_surf_copies()` -> `set_xv()` and fold the geometry regeneration
   loop of `end_of_step()` (`fix_rigid.cpp:1209-1245`) and
   `body_bbox()` into it, so there is one routine that turns a pose into
   geometry (host loop / device kernel K1+K4).  `pbodylo/pbodyhi` become
   the previous step's `bbox`, kept by a swap of two arrays.
6. **Per-surf arrays instead of a hash, one indexing scheme.**  `idmap`
   (`unordered_map<surfint,int>`) maps surf ID -> element; keep it for
   the rare surf-array rebuilds only, and build `rigidmap[]` and
   `elem_of_surf[]` over local+ghost surfs together in `Surf` when the
   surf arrays change.  Every local copy of a body element is listed once
   as `copy_index/copy_elem` (non-distributed: `ncopy = nsurf`), so
   `slist`, the `surf->distributed` branches in `gather_body`,
   `update_surf_copies`, `check_body_attributes`, `push_bins`, and
   `irigid` go away.  `ensure_local_copies()` moves to
   `Surf::add_local_copies()`, `rigid_cell_box` to `Grid::cells_in_box()`,
   and `Update::init_rigid()` shrinks to building the maps.
   `check_watertight()`'s `std::map<std::array>` edge counting becomes a
   sort-and-scan over edge records.
7. **Contacts as a helper class.**  `push_bins/push_off/push_contact` and
   their bins are a self-contained DEM contact model (spring/dashpot,
   linear or Hertz, static surfs + bodies + walls).  Move them to
   `RigidContact` (`rigid_contact.h/.cpp`, a `Pointers`-derived helper
   like `Cut2d`) with `RigidContactKokkos` for K5.  The fix calls
   `contact->compute(fpush,tqpush)`.  ~600 lines out of the fix, and the
   contact model is testable on its own.
8. **The per-step skeleton reads like LAMMPS** (driver in the target
   design above).  The three sync-avoidance virtuals (`particles_to_host`,
   `sort_for_split_rebuild`, `combine_split_all`) and the flag soup of
   review item 10 disappear: with the device authoritative there is no
   host particle access to avoid, and the change journal carries what
   changed.  `FixRigidKokkos` overrides the stages that have kernels and
   nothing else.
9. **Cell change journal in Grid, consumed by GridKokkos.**  Dual-write
   is expressed once: every Grid primitive above appends a record to
   `Grid::changelist` (cell index, kind, POD payload); `GridKokkos::
   apply_changes()` scatters it to the device and rebuilds the graphs at
   the end of the fix's stage.  This replaces `resync_after_host_change()`
   for every bounded change and is reusable by fix adapt/kk and fix
   move/surf/kk.
10. **What to keep as is**: the body dynamics (Verlet halves, Euler/
    Richardson quaternion, `set_recoil`, axisymmetric projection), the
    inside-body parity test, the radial prefilters, the COM bins, the
    incremental re-cut's per-cell logic (it becomes `RigidRemap::recut()`
    calling the Grid primitives), the local re-derivation of ghost splits,
    and the moving-surf collision tests in the mover.  They are already
    understood and tested.
11. **Resolve `FALLBACK_UNKNOWN` locally** (review item 5): when the cut
    of a cell whose surfs are all body elements returns UNKNOWN corners,
    type it by the parity test of its centre.  The full re-map remains
    for `FALLBACK_SURFMAX` and `remap cutcell`.

Coding conventions to hold to (the existing SPARTA code, e.g.
`grid_surf.cpp`, `fix_move_surf.cpp`, `cut2d.cpp`): `memory->create/
grow/destroy` flat arrays and `double**`; small POD structs; `enum{}`
for modes; `MAX/MIN`; `error->all/one` with the message documented at the
foot of the header; `/* ---- */` separators with a lowercase comment block
stating the invariant each routine relies on; no STL containers in
per-step paths; no callbacks or lambdas in host code; one owner per data
structure and every allocation grown in `DELTA` chunks; Kokkos code as
`operator()(Tag, i)` functors with `copymode` around the launch, following
`collide_vss_kokkos.cpp` and `update_kokkos.cpp`.

Where these land: all of 1-11 are the CPU-side restructuring, done first
as Phase R (bit-identical results, the regression suite as the guard),
because the Kokkos port targets the new structure; Phase S then adds the
in-place `restructure_split_cells()` and its device twin on top of the
Grid primitives; A, B, C add the kernels stage by stage.

### Target file layout

- `src/fix_rigid.{h,cpp}` (~1800 lines): command, body table
  (`gather_body`), body setup (`setup_body*`, properties, checks,
  infile/outfile), dynamics (`initial_integrate`, `final_integrate`,
  `set_xv`, `set_recoil`), `surf_tally` + `sum_forces`, the driver,
  `compute_*`.
- `src/rigid_remap.{h,cpp}` (~1000): `RigidRemap`, a `Pointers`-derived
  helper given the body table by the fix (element ranges, bboxes old/new,
  COM, `rmin/rmax`, bins): `collision_lists()`, `recut()`, the candidate
  and typing passes, the pending list handed to Grid.
- `src/rigid_contact.{h,cpp}` (~600): `RigidContact`: static-surf bins,
  `compute(fpush,tqpush)`.
- `src/grid.{h,cpp}`, `src/grid_surf.cpp`: the primitives, the collision
  list, `rigidinside`, `cells_in_box`, `restructure_split_cells`,
  `changelist`, page compaction.
- `src/surf.{h,cpp}`: `add_local_copies`, `rigidmap`, `elem_of_surf`.
- `src/update.cpp`: `move()` reads the collision list and the fix's body
  views; `init_rigid` reduced to map building; `rigid_cell_box` removed.
- KOKKOS: `fix_rigid_kokkos`, `rigid_remap_kokkos` (K2, K6/K7, uses
  `cut2d_kokkos.h`/`cut3d_kokkos.h`), `rigid_contact_kokkos` (K5),
  `grid_kokkos` (`apply_changes`, graphs, ghost shift, hash patch),
  `surf_kokkos` (copies scatter), `update_kokkos` (graph selection).

### Stage names the device classes override

`FixRigid`: `initial_integrate`, `final_integrate`, `set_xv`,
`sum_forces`, `remove_inside`.  `RigidRemap`: `collision_lists`,
`recut` (with `candidates`, `cut_cells`, `retype`, `apply` as the
overridable pieces).  `RigidContact`: `compute`.  Each has a host and a
device implementation of the same method; nothing else is virtual.

## Device data model

New members of `FixRigidKokkos` (all `double`, never `SPARTA_FLOAT`, like
`tdual_rigidbody_2d`, `update_kokkos.h:180-186`):

- Element table, uploaded at `setup()` and whenever `gather_body()` /
  `ensure_local_copies()` / `grid_changed()` change it (`nelem =
  bodystart[nbody]`): `d_displace(nelem,3,3)`, `d_body(nelem)`,
  `d_bodystart(nbody+1)`, `d_lblist(nelem)`, `d_bodytrans(nelem)`;
  distributed: `d_copy_index/d_copy_elem(ncopy)`, `d_olist_own/
  d_olist_elem(nolist)`, and a per-body CSR over copies (`d_copystart`,
  copies sorted by element then local index) for team-per-body loops.
- Per-step device outputs: `d_bodypt(nelem,3,3)`, `d_bodypt_new`,
  `d_bodynorm(nelem,3)`, `d_elemlo/d_elemhi(nelem,3)`.
- Per-body state as DualViews of the fix's own `double**` arrays
  (refactor item 4): `k_xcm, k_xcmnew, k_vcm, k_omega, k_quat,
  k_invmass, k_invinertia, k_ex/ey/ez_space, k_rmax/k_rmin`; the mover
  and computes read these device views; `UpdateKokkos::rigid_upload()`
  is deleted.
- Per-body download `k_bodybox(nbody,7)`: `bbodylo, bbodyhi, bboxeps`.
- Body bins (host-built, uploaded, as `pack_body_device()` does today).
- Device cell-bin index (port of `Update::rigid_cell_box`,
  `src/update.cpp:348-458`): `d_cellbinstart/d_cellbinlist` over
  local+ghost cells, built on the device from `d_cells` lo/hi;
  invalidated exactly where `update->rigid_bins_clear()` is called today
  and patched (not rebuilt) by the Phase S ghost shift.
- Push bins for static surfs (host-built once per run in `push_bins()`,
  uploaded) plus per-static-surf bbox.
- Per-cell scratch, grown not reallocated: `d_swcount/d_swrow`,
  `d_rcandflag/d_rcand`, `d_newlist(ncand,maxsurfpercell)`, `d_newn`,
  `d_cellflag`, `d_rigidinside` (device twin of `Grid::rigidinside`),
  cut outputs, and the change-journal record buffers (device and host
  mirrors).
- Device scalars for the step's reductions: `d_fallback`, `d_typechanged`,
  `d_npending`, `d_splitchanged`, cut error code.

`GridKokkos` additions (`src/KOKKOS/grid_kokkos.{h,cpp}`):

- `d_csurfs_move` (same `Crs<int,DeviceType,void,crs_size_type>` type as
  `d_csurfs`, `grid_kokkos.h:174`), `int csurfs_move_flag`, and
  `int graph_generation` bumped by every graph rebuild (including
  `wrap_kokkos_graphs()`), so `UpdateKokkos` and the fix re-take their
  handles when it changed.
- `build_crs_device(nrows, count_functor, fill_functor, crs)`: count ->
  `parallel_scan` -> fill on persistent views; functor index type is
  `crs_size_type` (`kokkos_type.h:34-39`, `bigint` under BIGBIG).
- `patch_cells_device(records)`, `patch_cinfo_device`, `patch_sinfo_device`:
  scatter kernels for the compact records (indices and POD fields only;
  the pointer fields are left untouched on the device).
- `shift_ghost_block_device(delta, nlocal_old)`: Phase S device mirror.

`UpdateKokkos` change (`src/KOKKOS/update_kokkos.cpp`): at the handle
refresh (`:736-738`) take `d_csurfs_coll = csurfs_move_flag ? d_csurfs_move
: d_csurfs`; in the move kernel use `nsurf = row_map(icell+1) -
row_map(icell)` of that graph for the collision loop (`:1697` feeds both
the loop at `:1753-1763` and `nscheck_one` at `:1703-1708`, as the host's
merged `nsurf` does); keep the `d_cells[icell].nsurf < 0` empty-ghost test
(`:2440`) and keep `split2d/3d` (`:2510-2609`) on `d_csurfs` (base).  Rows
of `d_csurfs_move` for untouched cells, including ghosts, equal the base
rows; `nsurf < 0` counts as 0 (`grid_kokkos.cpp:204`).

`ComputeSurfKokkos`: public accessor for `d_array_surf_tally` (private
today, `compute_surf_kokkos.h:461-468`; `F_FLOAT`, reduce in double).

## Execution order (driven by the measured profile)

Host C++ at 85% means the order is: kill the whole-grid host scans first,
then the host re-cut, then the remaining host loops, then polish the
copies.  Concretely:

| order | phase | what it removes | share it targets |
|---|---|---|---|
| 1 | 0 | nothing; makes the profile reproducible per stage | - |
| 2 | R (CPU restructure: Grid/Surf primitives, `RigidRemap`, `RigidContact`, fix tallies forces, one indexing scheme, `rigidinside`, local UNKNOWN typing) | the registries, pointer swaps, pending transaction, pre-move pass, compute coupling, flag soup; bit-identical results | none directly; it is what makes 3-6 small and safe |
| 3 | S + the graph part of A (`apply_changes`, `build_crs_device`, generation counter) | `split_rebuild`'s whole-grid scans, `wrap_kokkos_graphs` x3, `update_hash`, `resync_after_host_change`, the doubled `grid:cells` copies | the largest host item + most of the copies |
| 4 | rest of A (no bulk syncs, K1-K4, particle-side cleanups) | the remaining O(cells)/O(surfs) syncs and host packs | copies |
| 5 | C (device re-cut, 2d then 3d) | `RigidRemap::recut` on the host (the second host item) | host |
| 6 | B (K2, K5) | collision lists, contacts | host |
| 7 | D | leftovers | - |

On the per-cell graphs: a full device rebuild is O(cells + entries) at
GPU bandwidth (~1 ms for 1.28M cells) and needs no bookkeeping, so it is
preferred over an incremental host-side `wrap_kokkos_graphs()`; a CRS
cannot be patched in place when row lengths change anyway.  The host
side never builds a graph again after step 2.

## Phases

### Phase 0: build, instrument, baseline

- Build Kokkos Serial with `-DSPARTA_KOKKOS_EXACT=ON` (preset
  `cmake/presets/kokkos_mpi_only.cmake`, as CI) and Kokkos OpenMP
  (`cmake/presets/kokkos_omp.cmake`); install OpenMPI for 4-rank runs.
- Permanent instrumentation behind env `SPARTA_RIGID_TIMING`:
  `Kokkos::Profiling::pushRegion/popRegion` per stage; per-stage
  `MPI_Wtime` accumulators; per-DualView deep-copy counters and bytes
  (hook in `GridKokkos::sync/modify`, `SurfKokkos`, `ParticleKokkos`) and
  a count of `wrap_kokkos_graphs()`/`update_hash()`/`resync_after_host_
  change()` calls; per run counts of steps with `fallback`, `structural`,
  `splitchanged`, `typechanged`, and of `ensure_local_copies()` appends.
  Printed from `post_run()`.  This is the previous agent's ad hoc tooling
  made permanent; every later phase is accepted on these numbers.
- Regenerate the benchmark inputs with the uploaded `repro.sh`; keep the
  previous agent's 40-step harness (566^2 grid, 1.8M particles) and a
  100-body/358^2 deck for CPU timing under Serial-EXACT and OpenMP.  The
  user runs the same instrumented binary on the H100s to confirm the
  breakdown before and after each phase.

### Phase R: CPU restructure (bit-identical)

The refactor items 1-11 above, as a sequence of commits each of which
keeps `tools/testing/rigid` and `examples/rigid` bit-identical (except
the force-tally commit, which changes the command and re-blesses the
decks with the same numbers):

1. Grid primitives + collision list + `changelist` + page compaction;
   fix rewired to call them (registries deleted).
2. `RigidRemap` split out; `rigidinside` replaces `record_oldinside`;
   local UNKNOWN typing.
3. `RigidContact` split out.
4. One indexing scheme; `Surf::add_local_copies`, `rigidmap`/
   `elem_of_surf` in Surf; `Grid::cells_in_box`; `Update::init_rigid`
   reduced.
5. Fix tallies its own forces; compute-ID argument removed; docs,
   examples, tests updated.
6. `initial_integrate/final_integrate/set_xv` shape of the driver;
   `start_of_step/end_of_step` reduced to the driver listed above.
7. `FixRigidKokkos` rewritten as the thin stage-override class of the
   new structure with the existing device passes (RemoveInside,
   CombineSplit/AssignSplit) attached to the new stage names; the three
   sync-avoidance virtuals and env switches removed.  Still host-bracketed
   at this point (Phase S/A remove the brackets).

### Phase A: dual-write data model, compact transfers (host cut kept)

Removes every O(cells)/O(surfs) transfer and host CRS rebuild outside the
structural path.

1. Stage refactor of `fix_rigid.cpp` (above).
2. `FixRigidKokkos` stops calling `host_begin()/host_end()` and never
   `modify(Host,ALL_MASK)`s grid or surf inside a step.  `auto_sync`
   handling per principle 8.
3. Geometry on device (K1/K4), `TeamPolicy(nbody)`.  K1 at start_of_step:
   end-of-step corner points from `d_displace` and the pose into
   `d_bodypt_new`, swept per-element boxes, team min/max -> per-body bbox,
   then the `eps` inflation of `body_bbox()` (`fix_rigid.cpp:4912-4992`).
   K4 at end_of_step: swap `d_bodypt <- d_bodypt_new`, recompute
   `d_bodynorm`, scatter into `d_lines/d_tris` (via `d_lblist`;
   distributed: every `d_copy_index/d_copy_elem` pair plus `k_mylines/
   k_mytris` via `d_olist_own/d_olist_elem`; device twin of
   `update_surf_copies()`, `:2289-2345`), current per-element boxes,
   per-body bbox; DtoH `k_bodybox`.  The host surf copies are refreshed
   from a DtoH of `d_bodypt/d_bodynorm` (9 doubles/element) only on steps
   with a host consumer: emit fixes with `typechanged`, an output step
   (`dump grid/surf/image`, restart, grid variables), fix balance/adapt,
   or a structural/fallback step; otherwise no surf transfer at all.
   Axisymmetric branch (x-shift, keep r bitwise) ported verbatim.
4. Swept lists on device (K2), each start_of_step: (a) team per body over
   the cells of the bins overlapping the body's swept bbox, exact
   cell-vs-body box test, then per element `box_overlap(cell,
   elemlo/hi)`; count per *target* cell (the cell itself if unsplit, else
   each of its sub cells from `d_csubs`; a split parent keeps its base
   row, `fix_rigid.cpp:3874-3892`) with `atomic_inc`; (b) scan; (c) fill
   with an atomic cursor; (d) per target cell drop entries already in its
   base row and insertion-sort the extras by local surf index; (e)
   `build_crs_device` of (base row + extras) into `d_csurfs_move`, flag
   set, generation bumped.  `swept_restore_stage()` is a no-op.  Covers
   ghost rows (the mover follows particles into ghost cells).
5. Force/torque on device (K3): `d_array_surf_tally` is dense by local
   surf (`itally = isurf`, `compute_surf_kokkos.h:113-115`), zeroed in
   `clear()` at the top of the step (`update_kokkos.cpp:513`), 6 columns
   enforced by `init()`.  `TeamPolicy(nbody)` sums the six columns over
   the body's rows (`d_lblist(e)`, or every local copy via `d_copystart`
   for distributed) into `d_ft(nbody,6)`; EXACT: one thread in ascending
   local-surf order.  DtoH into `ftbuf_mine`; the single `MPI_Allreduce`
   (`fix_rigid.cpp:1139`) is unchanged.  `tallyinfo()` is not called
   (`compute_surf_kokkos.cpp:268-314` compresses on the host); keep
   `csurf->addstep()`, `invoked_flag`, and the `tallyinfo()` calls that
   `grid_rebuild()` makes before a reallocation (`:3677-3685`).
6. Hybrid re-cut: host `incremental_recut()` keeps running over its host
   candidate cells but writes into compact changed-cell records (cell
   index, `nsurf`, `type`, `volume`, `corner[8]`, and for split cells
   `xsub`, `xsplit`, `csplits`, sub-cell `nsurf/volume`; ghost split
   cells: `nsplit/isplit` from `split_ghost_drop`, `:4047-4051`) plus a
   concatenated list buffer.  The host arrays are patched as today
   (registry lists), the device by the `patch_*_device` scatter kernels,
   then `build_crs_device` of the new base (old rows for unchanged cells,
   record rows for changed) -> `grid_kk->d_csurfs`, and of `d_csplits`
   when a split cell changed.  `d_cinfo` has no ghost rows (`maxlocal`,
   `grid_kokkos.cpp:130-140`): ghost records carry no cinfo fields.  The
   host cut needs current host body surfs: DtoH of `d_bodypt` per step in
   this phase only (2.3 MB for the benchmark); Phase C removes it.
7. Split-cell particle work without host reads: `assign_split_kokkos()`
   builds its work list on the device from `d_cellcount/d_plist`
   (count/scan/fill over the `nsplitlocal` split cells uploaded from
   `sinfo`), no whole-`d_plist` copy (`fix_rigid_kokkos.cpp:452-517`);
   `combine_split_kokkos()` loops over `sinfo` instead of all cells.
8. `remove_inside_all_kokkos()` reads `d_cinfo.type` and `d_cells.nsurf`
   directly (drop the per-step host packs, `:855-870`); deletion list by
   flag + exclusive scan (ascending by construction, which both
   `ParticleKokkos::compress_migrate` and the host EXACT variant accept;
   drop the host `std::sort`, `:921-926`).  Factor the inside test into a
   shared `inside_any_body_kk(x)` for K6/K7.

Exit: rigid suite (incl. `--dist`) passes under Serial-EXACT with `-k on
-sf kk`; on a non-structural step the copy counters show only the
per-body tables, the element table (until Phase C) and the changed-cell
records.

### Phase S: structural split changes in place (the per-step floor)

Replaces `split_rebuild()` for rigid/kk with `structural_stage()`:

1. Host, O(pending) only: `split_cell_unset()` for each pending cell
   (marks its sub cells), `split_cell_set()` for the new piece counts
   (appends sub cells at `nlocal`, `grid_surf.cpp:1559-1606`, growing
   `k_cells/k_cinfo` in `DELTA = 8192` chunks so a resize is rare), then a
   new `Grid::remove_listed_cells(list)` that does the swap-with-last
   compaction of `remove_marked_cells()` (`:1646`) driven by the list of
   detached sub cells instead of scanning all `nlocal` cells for
   `proc == -1`, with the hash patched and `hashcurrent` kept, and a sinfo
   compaction driven by the pending list.  `setup_owned()`'s eps scan is
   skipped (sub cells do not change `cell_epsilon`).  Record every slot
   edit: (old index -> new index) for moved sub cells, the appended
   sub-cell records, the split parents' `nsplit/isplit`, the sinfo rows.
2. Ghost block shift: `delta = nlocal_new - nlocal_old`; memmove the
   ghost block on the host, `+= delta` on every `neigh[i]` whose nmask
   says child index and whose value `>= nlocal_old`, on the host hash
   values of ghost IDs, on `halo_index` entries `>= nlocal_old`, and on
   the ghost entries of the fix's cell bins.  No `unset_neighbors/
   remove_ghosts/acquire_ghosts/reset_neighbors/rehash`, no
   `comm->reset_neighbors()`.  Device mirror: `shift_ghost_block_device`
   (memmove via `Kokkos::deep_copy` on subviews, a kernel over cells for
   `neigh`, a kernel over ghost IDs updating `hash_kk` values in place, a
   kernel over `d_halo_index`), then the sub-cell/sinfo record scatters,
   then `build_crs_device` for `d_csurfs`, `d_csplits`, `d_csubs`,
   generation bumped.
3. Particle relabel on the device: an old->new index map `d_cellmap(nlocal_
   old)` (identity except moved sub cells) applied by a kernel over all
   particles (`icell = d_cellmap[icell]`), replacing the host list walk
   (lesson 7).  Particles of *changed* split cells were already pulled up
   into the parent by `combine_split_kokkos()` and are pushed back down by
   `assign_split_kokkos()` in `remove_inside_all()`.
4. Ghost copies of a pending cell on other ranks (only with ghost cells
   that carry surf info, i.e. `gridcut > 0`; with `gridcut 0.0` ghosts are
   EMPTY and nothing is needed): the owner sends (cell ID, `nsplit`,
   `xsub`, `xsplit`, `csplits` map, sub-cell `ilocal` list) records with
   `Irregular` to the ranks whose ghost box overlaps the cell (the boxes
   `acquire_ghosts_near` allgathers, `grid.cpp:678-683`, kept on the
   Grid); receivers append/drop ghost sub cells for that ghost split cell
   and patch its `nsplit/isplit/sinfo`, on both sides.  The migration
   contract (`Comm::migrate_particles` resolves the destination sub cell on
   the sending rank from the ghost copy) is unchanged.  Alternative if
   this proves fragile: route into the ghost split parent, migrate with
   the parent's `ilocal`, and let the owner run the split-assign kernel on
   arrivals (the mover already resolves a particle starting in a split
   parent, `update_kokkos.cpp:1344-1349`).
5. Replace `setup_owned()`'s four Allreduces by one fused Allreduce of
   the four counts; `cell_epsilon` is unchanged by sub cells.
6. Notify only what needs it: per-grid computes `reallocate()`, dumps'
   `reset_grid_count()`, per-grid fixes via the existing
   `add_grid_one/copy_grid_one/reset_grid_count` hooks that
   `split_cell_set/remove_marked_cells` already call; do not call
   `Grid::notify_changed()` (which sets `grid->changed` and triggers
   `resync_after_host_change()` in `modify_kokkos.cpp:99`).  Emit fixes get
   the same `grid_changed()` they get on a `typechanged` step.  Check that
   collide/kk's `add_grid_one/copy_grid_one` (`collide_vss_kokkos.cpp:
   4920-4981`) do not force a host sync of its per-cell arrays; if they
   do, give them a device path.
7. `sorted_kk = 0` stays (the next sort is needed anyway).

Exit: a structural step shows no `resync_after_host_change()`,
`update_hash()` or `wrap_kokkos_graphs()` in the counters, no MPI beyond
the fused count Allreduce and the (small) ghost record exchange, and no
transfer larger than the records; the split-cell tests (`splitcell`,
`splitbalance`, `splitmany`, `nbody`) bit-identical under Serial-EXACT on
1 and 4 ranks with `gridcut 0.0` and with a positive cutoff.

### Phase B: remaining O(elements x cells) host loops to device

1. `record_oldinside_stage()` (K6): kernel over the candidate cells of
   each body's pre-move bbox (owned, `nsplit == 1`, not cut, `type ==
   INSIDE`, centre inside via `inside_any_body_kk`) -> `d_oldinside` by
   flag + scan; runs before K4 (pre-move geometry), as `fix_rigid.cpp:1154`.
2. `push_off_stage()` (K5): `TeamPolicy(nbody)`, no atomics: body j's team
   computes everything that lands in `fpush[j]/tqpush[j]` in the host's
   order: reactions from every i < j (i's corners vs j's elements, the
   negated `push_contact()` force at the same contact point, `:3320`),
   its own static-surf contacts from the device push bins (`:3507-3552`),
   its own pair contacts in body-bin order (`:3569-3587`), boundary faces
   (`:3592-3649`), reactions from every i > j.  Each pair is evaluated
   twice (cheap: neighbours x 32 x 32); EXACT: one thread per body in that
   order; otherwise the team splits the loops and reduces.  DtoH 6 x
   nbody.  Distributed: owned-surf bins, pair/boundary on proc 0 only
   (`:3566-3567`), merged by the existing Allreduce.  Needs
   `box_overlap` and closest-point variants of `distsq_point_line/tri`
   in `geometry_kokkos.h` (`:1160,1199` return distance only).
3. Body bins stay host (O(nbody)) and are uploaded once per step.

Exit: no host loop over elements or cells in a steady step other than
the hybrid cut.

### Phase C: device incremental re-cut (2d then 3d)

1. Candidate cells (K7a): region R = old bbox U new bbox per body
   (`fix_rigid.cpp:4314-4341`) -> device cell bins -> flag owned cells and
   ghost split cells with exact box overlap; scan -> `d_rcand`.
2. Candidate surf lists + `surf2grid_list` (K7b), one thread per candidate
   cell (the per-cell work is serial and divergent, so no teams): static
   surfs of the base row (`d_rigidmap < 0`), elements of bodies from the
   device body bins with the radial prefilter (`:4397-4418`), device
   overlap test (`GeometryKokkos::line_quad_intersect` `:588`, 3d
   `tri_hex_intersect` `:725`) into `d_newlist`; `n > maxsurfpercell` ->
   `FALLBACK_SURFMAX` (max-reduce); insertion sort; "unchanged and no
   moving surf" -> skip; transparent test (`cell_cut()`).
3. `Cut2dKokkos` (`src/KOKKOS/cut2d_kokkos.h`): device port of
   `Cut2d::split()` (`src/cut2d.cpp:163-311`) and helpers `build_clines`
   (`:474`), `weiler_build` (`:584`), `weiler_loops` (`:816`), `loop2pg`
   (`:888`), `create_surfmap` (`:976`), `split_point_explicit` (`:1007`;
   the implicit variant is dead here, `fix_rigid.cpp:111-112`),
   `cliptest/clip` (`:1086,1147`), `ptflag/whichside` (`:1198,1214`),
   exposing `weiler_build/weiler_loops/loop2pg` for the 3d face cut.
   Fixed-capacity per-cell scratch rows from `maxsurfpercell = cap`:
   `clines <= cap`, `points <= 2*cap+4`, `loops/pgs/used/areas <=
   points`; no recursion.  Per-cell state (`grazecount/touchcount/
   touchmark/axisymmetric`) as locals.  The 7 `errflag` paths and
   `failed_cell()` become a per-cell error code max-reduced; the host
   raises the identical message from the host cell record.  Outputs per
   cell: `nsplit`, `vols[]`, `corner[4]`, `newmap[n]`, `xsub`, `xsplit`.
   ~1.0-1.2k LOC including scratch/error plumbing and the harness.
4. Apply (K7c/K7d) per changed cell exactly as `:4465-4580`, writing the
   same compact records as the hybrid path (so the dual-write scatter and
   the host patch of Phase A.6 are reused unchanged); pending-split
   records feed Phase S; passes 2 and 3 (`:4587-4631`) as kernels over
   `d_oldinside`/`d_rcand` with `inside_any_body_kk`; the three ints of
   the `MPI_Allreduce(MAX)` come from device scalars.  Never index
   `d_cinfo` for `icell >= nlocal`.
5. `Cut3dKokkos` (`src/KOKKOS/cut3d_kokkos.h`): port of `Cut3d::split`
   (`src/cut3d.cpp:229-281`: `split_try` (`:488`) once, again on the
   `SHRINK`-shrunk cell on failure), `add_tris` (`:691`), `clip_tris`
   (`:788`), `clip_adjust` (`:966`), `ctri_volume` (`:1166`), `edge2face`
   (`:1220`), `edge2clines` (`:1287`), `add_face_pgons` (`:1332`, 2d cut
   per face), `add_face` (`:1453`), `remove_faces`, `check` (`:1579`),
   `walk` (`:1654`, explicit stack), `loop2ph`, `split_point_*`, and the
   edge/vertex bookkeeping (`:1925-2056`).  Capacities from the `grow()`
   sites (`:702-703, 908, 1233, 1301, 1358, 1381, 1461, 1506, 1665, 1675,
   1730, 1770`); the 86 error/printf sites become per-cell codes;
   `ntiny/nshrink` become per-cell counts summed into the host `bigint`s.
   Roughly 3-4x the 2d port.
6. Drop the per-step `d_bodypt` DtoH of Phase A.6 (host consumers only).

Exit: the 1000-body deck (2d) and a 3d many-body deck run a steady step
with no host loop sized by cells or elements and no transfer larger than
the per-body tables and the changed-cell records; rigid suite
bit-identical under Serial-EXACT, including `--dist`; cell-by-cell cut
harness bitwise clean.

### Phase D: leftovers

- `ensure_local_copies()` stays host (it restructures the surf arrays
  with `surf->remove_ghosts()`); on the steps it appends, the surf
  DualViews are uploaded through the existing `modify(Host,...)`, the
  element/copy table re-uploaded, `build_rigidmap()` refreshes
  `d_rigidmap`.
- `grid_rebuild()` (incremental fallback) stays host + `resync_after_
  host_change()`; it is a warning-level event, counted by Phase 0.
- `resync_after_host_change()` triggered by *another* host fix (ablate,
  move/surf): the fix re-takes graphs through the generation counter at
  its next stage and invalidates `d_csurfs_move`.
- Merge `UpdateKokkos::rigid_upload()` into the fix's pose upload
  (cosmetic); a device `create_tasks` for emit/face/kk so the per-step
  host surf DtoH disappears for decks with emission.
- Docs: `doc/fix_rigid.txt` KOKKOS notes (lines ~926-941) currently say
  integration, force summation, re-assignment and deletion run on the host.

## Files

See "Target file layout" above.  In addition: `src/KOKKOS/grid_kokkos.
{h,cpp}` and `grid_id_kokkos.cpp` get `apply_changes`, `d_csurfs_move`,
the generation counter, `build_crs_device`, `shift_ghost_block_device`
and the in-place `hash_kk` value patch; `src/KOKKOS/update_kokkos.
{h,cpp}` the graph selection and row_map-based `nsurf`, reading the
fix's body views instead of `rigid_upload()`; `src/KOKKOS/geometry_
kokkos.h` gains `box_overlap` and closest-point helpers; `src/KOKKOS/
cut2d_kokkos.h`, `cut3d_kokkos.h`, `rigid_remap_kokkos.{h,cpp}`,
`rigid_contact_kokkos.{h,cpp}` are new; `src/KOKKOS/Install.sh`,
`src/KOKKOS/modify_kokkos.cpp` (verify the `grid->changed` hook is not
reached on a Phase S step), `doc/fix_rigid.txt`, `examples/rigid/*`,
`tools/testing/rigid/*`.

## Verification

- CPU regression after the stage refactor: `tools/testing/rigid/
  run_tests.py --exe src/spa_serial` and `--mpi "mpirun -np 4"`, plus the
  `examples/rigid` ctest suite, bit-identical to the gold logs.
- Kokkos Serial `-DSPARTA_KOKKOS_EXACT=ON` with `--args "-k on -sf kk"`
  (the CI configuration): identical to non-Kokkos after every phase,
  including `--dist`, on 1 and 4 ranks; ASAN build of the Serial variant
  (`OMPI_MCA_memory=^patcher`) for the split-cell tests after Phase S.
- Kokkos OpenMP: same suite for statistical agreement.
- Cut harness (Phase C): drive `Cut2d::split`/`Cut3d::split` and the
  device functors over every cell of the split decks and the first steps
  of the 1000-body deck, compare bitwise.
- Performance gates from the Phase 0 counters on the 40-step harness
  (OpenMP here, H100 by the user): after A no O(cells) copies on a
  non-structural step; after S none on a structural step and no
  `update_hash()`; after B no host stage scaling with elements x cells;
  after C fix time per step comparable to the move.
- Determinism: two identical Serial-EXACT runs of the 1000-body deck give
  identical `f_1[*]` columns.

## Decisions (from the user)

- Refactors that clean up the code or make the port simpler are wanted
  (items 1-9 above), coded the way Steve Plimpton would: fully
  understood, well designed, clean, performant.

- Both the 2d and the 3d cut are ported to the device (Phase C).
- `SPARTA_KOKKOS_EXACT` stays bit-identical through small in-kernel
  bifurcations only, never a host fallback.
- Distributed surfs are supported by every device stage from the start.
- Build and run Kokkos Serial (EXACT) and OpenMP locally for validation;
  OpenMPI and other dependencies may be installed (root available).

## Status (18 Sep 2026)

Phase R is complete and pushed (`860677e`, `2558697`): Grid primitives,
`RigidRemap`, `RigidContact`, one indexing scheme, the fix tallies its
own forces (compute-ID argument dropped), LAMMPS-shaped driver.  Bench
bit-identical, rigid suite 63/63 (1 and 4 ranks, `--dist`, Kokkos Serial
EXACT).

Phase S committed and pushed as `4e9e03e`:

- **S1, host (done, verified)**: `Grid::restructure_split_cells()` applies
  the piece-count changes in place: holes = the old sub cells, new sub
  cells fill them in the order an append + compaction would (so the
  owned layout, and every result, is bit-identical to before), the ghost
  block steps aside or closes up by moving a few ghost cells, and
  `Grid::move_cell()` repairs every reference to a moved cell (sinfo,
  hash, halo index, cell bins, neighbor back links, particles, per-cell
  data of collide and the per-grid fixes).  Only sub cells move among the
  owned cells; ghost sub cells route migrating particles to their split
  cell (`Grid::subroute`, resolved by the owner at PENTRY), so no other
  proc's records go stale and nothing is communicated except one
  Allreduce of the cell counts.  `restructure_check()` detects the rare
  case (after a balance the owned tail holds non-sub cells) and the
  collective ghost rebuild is taken for that step.  Recut no longer
  touches ghost split cells at all.  Bench bit-identical on 1 rank with
  the fix time down from 17.8 s to 9.8 s on the CPU; suites 63/63 on 1
  rank, `--dist`, 4 ranks, and the Kokkos Serial EXACT build.
  Also fixed a real bug from R1 in the Kokkos mover: it sized the surf
  loop by the cut count, not the graph row, so swept surfs were skipped.
- **S2, device (done, verified)**: `Grid` keeps a change journal when
  `journalflag` is set (dirty cells/sinfo, cut-list and collision-list
  records, moves); `GridKokkos::apply_changes()` scatters the records,
  patches `hash_kk` and `d_halo_index` in place, rebuilds `d_csurfs` on
  the device from its old rows + records (`build_csurfs_device`,
  `build_crs`) and `d_csurfs_move` from the collision records; the split
  graphs are rebuilt from the host sinfo (small).  `FixRigidKokkos` calls
  `apply_changes()` instead of `modify(Host,ALL_MASK)` + `wrap` +
  `update_hash`; the deletion kernel reads `d_cinfo/d_cells` directly;
  `UpdateKokkos` reads `d_csurfs_move` for the collision loop and
  `d_csurfs` for the split tests.  The device resizes now start from the
  host copy (`modify(Host)` before `sync(Device)` in `grow_*`).
  Verified: Kokkos Serial EXACT bench bit-identical to the CPU run with
  all 40 steps in place; kk suites 63/63 and 33/33 (`--dist`); CPU
  benches bit-identical on 1 and 4 ranks.

Then (all pushed): `f38d806` per-stage timers + device-built split-assign
work list; `52dfb7c` cheaper host re-cut (interior cells skipped, shell
test before the ray cast, element box prefilter).  1-rank bench 23.6 s
-> 12.6 s, 4-rank 22.0 s -> 10.5 s, all bit-identical.

Phase B K2 in the working tree (verified bit-identical on the kk bench):
`RigidRemapKokkos::collision_lists()` builds the mover's graph on the
device from the device cell bins (`GridKokkos::sync_cell_bins`, bin
patches through the journal), the fix's device body table and the cut
graph, with the host's merge order.  The KOKKOS CMake exclusion regex
for the FFT remap files is anchored at the file name so
`rigid_remap_kokkos.cpp` is built.

`e42cbce` (pushed): the hybrid device re-cut (`RigidRemapKokkos::
recut()`: candidate cells, candidate surf lists with the exact
cliptest/clip tests (`cut_kokkos.h`), re-typing on the device; only the
changed cells cut on the host by `RigidRemap::recut_cell()`), and the
device body table `RigidBodyKK` (`rigid_body_kokkos.h`) shared by the
deletion kernel and the remap.

Phase C (19 Sep 2026), in the working tree, being validated:

- `cut2d_kokkos.h` / `cut3d_kokkos.h`: device twins of `Cut2d::split()`
  and `Cut3d::split()` (all helpers ported line for line; the 3d cut
  embeds a 2d cut for its faces), one instance per thread on scratch
  rows sized by the cell's surf count (bounds documented at the head of
  `cut3d_kokkos.h`, overflow = error 9 -> host cut).  `RigidRemapKokkos::
  recut()` packs the changed lists as CRS rows, cuts every changed cell
  on the device (3d in chunks under a 256 MB scratch budget), brings
  back only per-cell results (nsplit, corners, xsub, xsplit) and
  per-entry rows (piece map, volumes), and installs them through
  `RigidRemap::apply_cut()` (factored out of `recut_cell()`).  A cell
  the device cut fails on is re-cut on the host, which raises the same
  message.  Verified cell by cell against the host cut (2d suite decks,
  the 1000-body bench, an 8-sphere 3d deck with 48696 device cuts):
  identical.
- Bugs found on the way: `GridKokkos::~GridKokkos()` freed two host
  arrays before its functor-copy guard (double free on every kernel
  launch; the earlier suites passed by luck); `Update::rigid_maps_
  changed()` read the fix through a stale cached pointer after a fix
  re-definition (use after free in `refix`); `Cut3d::walk()` reused the
  outer loop index in its inner edge loop, so the walk could skip the
  start of a second flow component (fixed on host and device).
- Per-body force sums: the host summed tally rows in first-hit order,
  the device per surf in index order, so Kokkos EXACT differed from the
  CPU at the last bit (`momentum`, `crossproc`) and distributed differed
  from non-distributed.  Now one tally row per body element on both
  sides (device: `d_surfelem` map in `UpdateKokkos`), summed per body in
  element order: Kokkos Serial EXACT == CPU bit for bit on every deck
  tried, distributed == non-distributed.
- Pushed as `3e895f3`; then `f88a787`+`49a2c76`: on 2+ procs in
  `remap cutcell` mode (and the incremental fallback) the device
  deletion/split-assign pass ran against the device grid the host full
  re-map had just replaced (particles sent through the bodies, then
  deleted); `FixRigidKokkos::grid_rebuild()` re-establishes the device
  grid before that pass.  Kokkos == CPU at 2 ranks in incremental mode
  and on `crossproc`; `cutcell` at 2 ranks differs from the CPU in the
  last bits (no deletions), a Phase D leftover.
- Validation of `49a2c76`: Kokkos Serial EXACT suites 63/63 at 1 and 2
  ranks, 33/33 distributed; CPU suites 63/63 and 33/33 at 1 rank; the
  bench bit-identical Kokkos vs CPU and vs the original reference.
- `b339ad7`: the 20-body example logs refreshed (the element-order sums
  shift its chaotic trajectory at round-off); every other example log
  changed only in timings and was kept.
- Also: CPU suites 63/63 at 4 ranks (one test re-run after a timeout
  under load) and 33/33 distributed; Kokkos Serial 63/63 at 4 ranks; Kokkos OpenMP (2 threads)
  63/63; AddressSanitizer clean on the 3d deck, crossproc (dist),
  splitmany, refix and the 2-rank cutcell deck.
Phase A K1/K4 (19 Sep 2026), in the working tree, being validated:

- The body geometry lives on the device: `FixRigidKokkos::set_xv()`
  moves the bodies to the new pose on the host (O(nbody)) and
  regenerates points, normals, element boxes, body bboxes and the surf
  arrays on the device (`device_geometry(0)`, four kernels);
  `swept_boxes()` does the swept boxes of the step the same way
  (`device_geometry(1)`).  Per step only the poses go up and the body
  bboxes come down.  The host copies of the geometry are regenerated per
  body on demand (`host_geometry()`, contacts, inside tests) and the host
  surf arrays on output steps, grid rebuilds, distributed copy appends
  and after the run (`refresh_host_surfs()`).  `FixRigid::set_xv()` was
  split into `set_pose()` + `body_geometry(ibody)`; `posesplit` marks
  the window between `initial_integrate()` and `set_pose()` in which a
  host regeneration must take the start-of-step frame from `quat`.
  Bit-identical Kokkos vs CPU on the bench and on 2d, axisymmetric and
  3d decks.  Note: `SurfKokkos::sync(Device)` under auto-sync flags the
  host copy newer, so the fix never relies on auto-sync for the surfs.
- Left for B/D: K5 contacts on the device (the contacts regenerate the
  host geometry of the bodies in contact), the per-step `rigid_upload`
  and pose uploads could be merged, K2's bins upload; the 2-rank
  `cutcell` last-bit difference; a stage split of
  `RigidRemapKokkos::recut()`.
- Left for A/B/D (older list): K1/K4 geometry on the device (the per-step host
  `set_xv` loop and the surf upload), K5 contacts, surf-copy DtoH; the
  2-rank `cutcell` last-bit difference; a stage split of
  `RigidRemapKokkos::recut()` for readability.

GPU coherence audit (20 Sep 2026), after the user's CUDA session reported
`in.test.splitmany -var mode cutcell -var gas 1` losing 5 particles on
one H100 at `ceba432` while Serial, OpenMP and the CPU keep all 20000:

- Tooling: the `KOKKOS_DEBUG_SYNC` build from
  `claude/kokkos-detector-verify` (a checked `SPARTA_NS::DualView` that
  gives the host side its own allocation on a CPU build and drives the
  coherence counters itself; `SPARTA_KOKKOS_WATCH/STALE/STALE_STRICT/
  TRACE/PARANOID`, and a poison mode under AddressSanitizer), merged onto
  this branch in the worktree `/home/user/sparta-sync` (builds
  `build-sync` EXACT, `build-sync-ne` non-EXACT, `build-poison`).  The
  rigid dual views declared as `Kokkos::DualView` had to be respelled
  `SPARTA_NS::DualView` there to be covered.  The deck loses a particle
  under the emulation exactly as on the GPU, so every finding below is a
  host/device transfer bug that Serial's aliased memory hides.
- Three defects, all in the `remap cutcell` path or the deletion pass:
  1. `GridKokkos::resync_after_host_change()` claimed the host copies of
     the grid and surfs but never copied them to the device, so the
     deletion and split-assign pass which follows the full re-map read
     last step's cells, split info and surfs (poison mode trapped every
     read: `split2d_kk`, the K2 count kernel, `box_overlap`).  It now
     ends with `sync(Device,ALL_MASK)` for the grid and the surfs.
  2. `remove_inside_all_kokkos()` compacted the particle array through
     `Particle::compress_migrate()`, the host routine: in an EXACT build
     `ParticleKokkos::compress_migrate()` is compiled out, so the host
     copy was compacted while the device copy, the claimed side, kept
     the deleted particle, and the last particle vanished instead (found
     with a hardware watchpoint on the deleted slot).  New
     `ParticleKokkos::compress_migrate_kokkos()`, compiled in every
     build, applies the host routine's (destination,source) pairs on the
     device, so the layout is the host's and EXACT stays bit-identical.
  3. The device scatter of the body surfs (`set_xv`) never claimed the
     surf dual views, so the next host claim discarded it (watch mode:
     "device side written, never claimed, and is now lost", every step).
     `device_geometry()` now claims the device (auto-sync off around it),
     the scatter also writes the owned arrays of distributed surfs
     (`k_olist_own/elem` uploaded by `pack_body_static()`), and
     `refresh_host_surfs()` is the `sync(Host)` that claim calls for
     instead of a host regeneration; `host_begin()` no longer syncs the
     surfs to the host every step; `check_body_attributes()` regenerates
     the host body geometry it compares against.
- The remaining stale reports (`grid:cinfo`, `grid:sinfo`) differ only in
  host-side fields (particle list heads, pointer fields): forcing copies
  of them (`SPARTA_KOKKOS_PARANOID`) changes nothing.
- Pushed as `65d9285`.  Verified: the deck keeps 20000 particles under
  the emulation, EXACT and non-EXACT, bit-identical to the plain Serial
  EXACT run; rigid suites 63/63 (1 and 2 ranks) and 33/33 (`--dist`) on
  the main Serial EXACT build and on the emulation build; CPU 63/63;
  OpenMP (2 threads) 63/63; the 1000-body bench and the 8-sphere 3d deck
  bit-identical to the CPU.  The debug worktree `/home/user/sparta-sync`
  (detector branch merged, the rigid dual views respelled
  `SPARTA_NS::DualView`) is kept for the next audit; its merge commit is
  local only.

Per-step device pass audit (20 Sep 2026), 1-rank 40-step bench, Serial:

- Tooling: a 60-line Kokkos Tools library (`scratchpad/kptool/kp_count.cpp`,
  loaded with `KOKKOS_TOOLS_LIBS`) counts and times every kernel by name
  and every deep copy by label; per-step numbers are the difference of
  a 40- and an 80-step run.  Every lambda kernel in the remap, the grid
  journal and the fix now carries a label (`rigid_remap:sw_count`,
  `grid:crs_fill`, ...), which the tool needs to attribute time.
- Findings (ms per step, Serial): mover 124, particle sort 49, collide 19;
  the fix: deletion pass 57 (a full particle pass whose cost is the
  parity tests of the particles in surf cells, the same test as the
  CPU's), swept lists 94 (two enumeration passes over the (body,bin)
  pairs with 32 element box tests each, and a whole-grid CRS assembly:
  rowcount, scan, fill of every base entry, plus three 5 MB zeroings
  and a host-built sub-cell table), re-cut 52 (candidate enumeration
  21, new lists 16, cuts 7, compare 6), cut-graph rebuild 16 (whole-grid
  count/scan/fill whenever a cut list changed, every step here), hash
  patch 4, relabel 7 on 40% of steps.  gdb sampling shows the fix's
  host work is under 5% of its time on Serial: the journal packing of
  ~53k changed cells per step and the pair lists.
- Done: the swept-list stage rebuilt around the touched cells only.  The
  count pass records every (cell, element) hit and lists each touched
  cell once (atomics; the hit buffer grows and the pass re-runs when it
  overflows, once at the first step); the hits are scattered into one
  row per touched cell, sorted and merged against the cut row as
  before; the per-cell count/row arrays are reset from the previous
  step's touched list, never rewritten.  The mover reads the cut graph
  plus the rows through `GridKokkos::d_swrow/d_swoff/d_swext/d_swelem`
  (`swextras` flag; an unsplit cell its own row, a sub cell its split
  cell's, a split cell none), so no whole-grid graph is assembled and
  the host sub-cell table is gone.  Stage cost 94 -> 34 ms/step; bench
  20.1 -> 16.9 s, bit-identical to the reference; 3d deck identical.
- Left as is, with reasons: the deletion pass (its cost is the inside
  tests the CPU also pays; restricting it to candidate cells' sorted
  particle lists is unsafe on restructure steps); the cut-graph rebuild
  (a padded CRS would let rows be patched in place, ~16 ms Serial, a
  few whole-grid launches on a GPU, not worth the consumer changes);
  the re-cut candidate flag/scan/list over the grid (5 ms plus the
  enumeration, which is the real cost); the whole-grid memsets (trivial
  on a GPU).  `apply_changes()` is called three times per step but its
  kernels run once (records exist once); the geometry kernels run twice
  by design (swept and committed poses).

Follow-up (21 Sep 2026), after the user's H100 run of the fixes (1.97 s /
40 steps at 1 GPU, flat at 2 and 4 GPUs; "36/37 particle crossings per
40 steps"):

- The particle crossings: under the emulation with the device comm path
  (`-pk kokkos comm threaded`) the loop makes no particle copy at all in
  incremental mode; the 36/37 are `ParticleKokkos::grow()` during
  `create_particles` (growth by max(16384, 10%) -> ~36 growths for 1.8M
  particles, each a sync to the device and, under auto-sync, a copy
  back), i.e. setup, not the loop.  `CreateParticles::create_local()`
  and `create_local_twopass()` now grow the array once to the final
  per-proc count.
- One device sort per step instead of two: `assign_split_kokkos()` used
  to re-sort the particles after the combine (which invalidates the
  sort) to get the split cells' rows; it now reads the rows the combine
  relabelled (the combine's sub cell -> split cell pairs of the cells
  split now, plus the own row of a cell split now but not then; a cell
  split then but not now keeps its particles).  Bench 16.9 -> 15.9 s,
  bit-identical; the first attempt (the combine's pairs verbatim) lost
  particles on cells whose piece count changed, which the bench caught.
- The Kokkos Tools counter (`scratchpad/kptool/kp_count.cpp`) gained
  `KP_COUNT_BT=<label>`: copies of that array are keyed by the SPARTA
  frames of their call stack, and `KP_COUNT_OUT` gets a rank suffix
  under MPI, for the user's GPU runs.

Supervision state (21 Sep 2026, ACTIVE PLAN execution):

- **Branch**: the user asked for the ACTIVE PLAN work to go to a new
  branch: `surf-rigid-body-owned` (created from `surf-rigid-body` at
  `165a8bc`; all packages commit and push there with
  `git push -u origin surf-rigid-body-owned`).  `surf-rigid-body` stays
  at `165a8bc`.
- WP1 landed as `533e00f` on `surf-rigid-body-owned` (reviewed, comments
  trimmed to the user's terse style, full matrix green, pushed).

- 22 Sep: the user's H100 runs of owned mode.  **Strong scaling** (1000
  bodies): 1 GPU 2.06 s, 2 GPUs 2.07, 4 GPUs 1.90 -- the first
  configuration where 4 GPUs beats 1, but only 1.09x.  Contacts (0.40x),
  recut (0.53x) and collision lists (0.69x) now scale; `set_xv+bounds`
  (0.35 s) and `sum forces` are flat and `integrate+bbox` is 1.38x
  worse.  Transfers are no longer a factor: 31.7 MB and 106 transfers
  per step, 2.8% of the loop, zero particle copies (down from 571
  MB/step); what is left is `collide:vremax/remain` both ways every step
  (10.3 MB, collide code, not the fix), `grid:stagecell` 5.2 MB, and the
  body surfs coming back to the host 3.4 MB/step.  GPU kernel time is
  ~7% of the loop, so ~90% is host-side CPU work.
  **Weak scaling** (1000 bodies + 1.8M particles per GPU, boxes
  566/800/1132 cells square): 2.51 s, 5.30 s, 25.32 s -- 10% efficiency
  at 4 GPUs, and `sum forces` alone is 14.16 s of the 25.3, of which
  **13.18 s is inside `Irregular::create_data_uniform`** (325 ms per
  call, 40 calls) while `exchange_uniform` is 2 ms and the payload is
  under 100 KB/step.  Not O(nbody) and not O(nprocs): 0.006 s at
  1k/4 ranks, 0.130 at 4k/2 ranks, 13.18 at 4k/4 ranks.
  Reproduced here on the CPU with the same deck and rank count (4000
  bodies, 1132^2, 7.2M particles, 4 ranks): `sum forces` 0.16 s,
  `integrate+bbox` 0.19 s, loop 12.38 s owned vs 12.83 s replicated,
  mover balanced to 3%.  So the blow-up is that node's MPI reacting to
  the routine's unexpected-message handshake (a `Reduce_scatter`, a
  blocking `MPI_Send` per destination, then `MPI_ANY_SOURCE` receives
  with nothing pre-posted), not the fix's arithmetic.  Package **PERF1**
  (in flight, `scratchpad/briefs/perf1.md`) replaces it with the
  fixed-neighbor Irecv/Isend exchange this plan already named as the
  follow-up: the neighbor set comes from `procboxall` and `bodycut` in
  `proc_boxes()`, is symmetric, and changes only when the grid does.
  Proc 0 is in every neighbor list, since `body_owner()` names it for a
  body outside every owned box (the ownership deck's third body leaves
  the box mid-run, so that path is covered; `exitbox` cannot cover it,
  it uses `remap cutcell`).  Verified: 11 suites green (CPU
  1/4/dist/owned 1/4, kk 1/2/dist/owned 1/2/4), the owned bench and the
  4000-body weak deck bit-identical to the previous binary at 1 and 4
  procs on both builds, replicated bench tables unchanged, ASan clean on
  the ownership deck and the owned bench at 2 ranks, CPU loop time
  unchanged.  `sum forces` and `integrate+bbox` now each print an
  `exchange` sub-timer: on the CPU weak deck at 4 ranks the exchange is
  0.18 s of the 0.19 s `sum forces` stage, so the user's next GPU run
  shows directly whether the 325 ms per call is gone.
  The flat `set_xv+bounds` and the `integrate+bbox` regression are next
  (Part 3 item A: the census shows the three geometry kernels launching
  twice per step at ~3.3 ms each, which the fused geometry pass removes).
- P0 landed as `5f9d34b` (pushed): per-stage sub-timers with
  min/avg/max over procs and per-run work counters, every stage fully
  accounted.  Bench tables identical, suites green.
- WP3 landed as `e41b104` (pushed): owned mode in the Kokkos fix,
  kk owned == CPU owned bit-for-bit at 1, 2, 4 ranks; plus a hardening
  in `FixRigid::grid_changed()` (refresh_all before re-deciding
  ownership; `fix move/surf` reaches it outside the dump/balance
  triggers).  CPU performance after the refactor checked by an
  interleaved A/B against the pre-refactor binary (worktree
  `/home/user/sparta-base` at `165a8bc`, `build-mpi`): 1 rank 15.7 vs
  15.8 s, 4 ranks 5.3-5.9 vs 5.3-5.4 s replicated and 5.2-5.3 s owned,
  i.e. unchanged within noise (the older 13.2 s log reflects a faster
  machine state, not the code).  Keep that worktree for A/B timing of
  every later package.  In flight: P0 (CPU profiling protocol), Opus 5
  agent from `scratchpad/briefs/p0.md`; baselines `p0base_*.log`.
- WP2 (+WP5) landed as `c9b9183` on `surf-rigid-body-owned` (pushed):
  owned mode on N procs, bit-identical to replicated at 1 and 4 ranks
  on the bench and the ownership deck.  In flight: WP3 (Kokkos owned
  mode), Opus 5 agent from `scratchpad/briefs/wp3.md` (progress file
  `wp3.progress.md`, WIP commits "WIP wp3: ..." above `c9b9183`);
  baselines `/home/user/bench/wp3base_*.log`.
- Earlier: WP2 (+WP5), Opus 5 agent from `scratchpad/briefs/wp2.md`
  (progress file `wp2.progress.md`, WIP commits "WIP wp2: ..." on
  `surf-rigid-body-owned` above base `533e00f`).  21 Sep 19:13 UTC: the
  agent was killed by the session usage limit with all 12 scope items
  done (WIP `467d774`, `6fca3ea` + 4 files of uncommitted follow-ups),
  the replicated matrix and bench runs still to finish, then the
  squash.  Resume the same agent by message (it continues from its
  transcript) or start a fresh one on the brief; either reads the
  progress file and `git log` first.  Agent notes to check in review:
  the record grew to 63 doubles (ex/ey/ez_space travel, not derived from
  quatnew); the default bodycut leaves only the push range for per-step
  motion (decks that need more set `cutoff`); `create_grid`'s default
  decomposition is stride (the doc says block), so owned decks use
  `balance_grid rcb cell`.
  22:00 UTC: WP2 delivered as `ee20be1`; supervisor validation all
  green (9 suites, replicated bench identical, owned == replicated
  bit-for-bit at 1 and 4 ranks).  Two follow-ups sent back for an
  amended commit: default cutoff = `2*rmaxmax*(1+2*EPSSURF) +
  pushcutoff` (covers contact partners, one radius of motion per step;
  the plan's Part 1 text is updated below) and `host_surfs_needed()`
  on dumps/restarts only, not stats.  Pre-existing oddities the agent
  found, to look at later: (a) `global gridcut ... comm/sort yes`
  placed after `create_grid` makes 4-rank gas runs non-deterministic
  run to run (placing it before `create_box` fixes it); (b) a body
  crossing a reflecting boundary at 400 m/s in a dense gas came back
  at 1667 m/s in both modes; (c) `create_grid` defaults to a stride
  decomposition while `doc/create_grid.txt` says block.  Baselines for the
  review: `/home/user/bench/wp1rev_{cpu_np1,cpu_np4,kk_np1,kk_np2}.log`
  (replicated stats tables must stay identical); owned gates in the
  brief.
- Review procedure per package: read the full diff against the brief;
  rerun the validation matrix independently (CPU 63/63 at 1 and 4
  ranks, `--dist` 33/33, Kokkos Serial EXACT 63/63 at 1 and 2 ranks,
  bench stats diffs empty); send findings back to the agent by message
  until clean; then push.
- Next packages, briefs to write after WP1 lands: WP2+WP5 (exchanges,
  handoff, gather/refresh, contacts by owner, docs, tests) and the CPU
  profiling package (Part 2 P0); they touch the same files as WP1 and
  therefore wait for it.

CPU profile (P0), 21 Sep 2026, on top of the WP3 commit `e41b104`
(branch `surf-rigid-body-owned`, `build-mpi` `-O3 -g`, 1000-body bench
`in.bench40`, 1.8M particles, 40 steps).  Instrumentation only: the
`SPARTA_RIGID_TIMING=1` table now prints min/avg/max over procs, every
stage is fully accounted by sub-timers, and per-run/per-step counters
are printed with it (`bigint`, MPI_SUM, only under the flag).  Bench
tables and suites unchanged.  Raw outputs in `/home/user/bench/prof/`.

Stage table (s, max over procs; 1 rank loop 16.08 s, 4 ranks 5.80 s):

| stage | 1 rank | 4 ranks (min/max) |
|---|---|---|
| Move / Sort / Coll (SPARTA) | 4.83 / 1.56 / 0.94 | 1.92 / 0.44 / 0.25 |
| Modify (= the fix) | 8.75 | 3.03 |
| integrate+bbox | 0.041 | 0.040/0.043 |
| collision lists | 1.661 | 0.613/0.662 |
| collision: enumerate | 1.403 | 0.551/0.598 |
| collision: merge | 0.208 | 0.045/0.051 |
| collision: reset | 0.042 | 0.012/0.012 |
| sum forces | 0.007 | 0.036/0.068 |
| set_xv+bounds | 0.119 | 0.081/0.085 |
| contacts+kick | 0.023 | 0.024/0.025 |
| recut | 3.958 | 1.301/1.311 |
| recut: candidates | 0.915 | 0.303/0.317 |
| recut: surf lists | 1.040 | 0.377/0.398 |
| recut: compare | 0.143 | 0.035/0.036 |
| recut: cuts | 0.646 | 0.169/0.178 |
| recut: retyping | 0.448 | 0.129/0.140 |
| recut: reduce (the 4-int Allreduce) | 0.000 | 0.031/0.097 |
| recut: split combine | 0.567 | 0.132/0.141 |
| apply split changes | 0.031 | 0.017/0.026 |
| apply: restructure / cell counts | 0.030 / 0.000 | 0.006 / 0.011-0.020 |
| remove inside | 2.908 | 0.799/0.847 |
| remove: split assign | 0.582 | 0.127/0.142 |
| remove: particle pass | 2.326 | 0.672/0.704 |
| remove: compress | 0.000 | 0.000 |

Counters, per step (sum over procs; 1 rank, 4 ranks nearly identical):
particles tested by `inside_any_body()` 42,348 of 1.80M (2.4%);
particles deleted 0; candidate cells 90,453; cut lists changed 53,650;
cells cut 51,098; pending split changes 175; (body, cell) pairs in the
collision stage 224,300 (253,217 at 4 ranks, ghost cells); element box
tests 6.15M, yielding 107,044 swept entries in 70,180 touched cells,
23,035 of which get extra surfs.  The mover's `nscheck` is the
end-of-run stats line `SurfColl checks = 5,166,633` (129k/step,
`Surface-checks/particle/step: 0.0718`).

Function table, callgrind on `in.bench4` with `--collect-atstart=no
--toggle-collect=Update::run`, i.e. the 4 timesteps only (inclusive Ir,
4.659 G total in the loop):

| % | function |
|---|---|
| 43.6 | `Modify::end_of_step` = `FixRigid::end_of_step` |
| 34.5 | `FixRigid::remap_grid` |
| 31.6 | `Update::move<2,1,0,1>` |
| 20.1 | `RigidRemap::recut` |
| 18.4 | `FixRigid::start_of_step` |
| 17.7 | `RigidRemap::collision_lists` |
| 16.1 | `FixRigid::inside_any_body` |
| 14.5 | `FixRigid::inside_body` |
| 12.6 | `FixRigid::remove_inside_all` |
| 10.7 | `Grid::cells_in_box` |
| 10.5 | `Geometry::line_line_intersect` |
| 9.2 | `RigidRemap::recut_cell` |
| 8.3 | `Grid::build_cell_bins` |
| 7.9 | `RigidRemap::refresh` |
| 7.8 | `RigidRemap::mark_static` |

(4 steps over-weight once-per-run work: `build_cell_bins` and
`mark_static` are rebuilt at the start of the run, and the 40-step
`perf`/gdb profiles show neither as a per-step cost.)  `perf record -e
cpu-clock` on the 40-step deck (self, whole process): `Update::move`
23.3%, `FixRigid::remove_inside_all` 12.5%, `Particle::sort` 8.5%,
`RigidRemap::recut` 7.3%, `RigidRemap::collision_lists` 6.4%,
`Grid::cells_in_box` 4.1%, `line_line_intersect` 3.2%,
`FixRigid::inside_body` 0.8%.  86 gdb backtraces inside the loop agree
(mover 29%, `remove_inside_all` 19%, `Particle::sort` 13%,
`cells_in_box` 7%, `collision_lists` and `recut` 6% each).  No
`malloc`/`free` frame reaches 1% of the loop.

What this says about C1-C7.  **C1** (classify cells, not particles) is
the right stage - remove inside is 2.91 s of 16.08 s - but the profile
does not support the plan's "most of the 2.4 s": only 2.4% of the
particles reach `inside_any_body()` today, and the parity tests
themselves are cheap in cycles (`inside_body` 0.8% of the run, though
14.5% of the loop's instructions: dense FP code, high IPC).  The 2.33 s
pass is the scan itself, two random loads per particle (`cinfo[icell].
type` and `cells[icell].nsurf`); a per-cell flag replaces them with one
load into a 4-byte-per-cell array, so expect about half, 1.0-1.3 s
(6-8% of the loop), not 2.4 s.  **C2** is half in already: candidates
come from the cell bins (`Grid::cells_in_box`), and that enumeration is
now measured at 0.92 s (`recut: candidates`), with the per-cell list
build at 1.04 s of which the 41% of candidate cells whose list does not
change (90.5k enumerated, 53.7k changed per step) could be skipped by an
early compare - together 0.5-0.9 s (3-5%).  **C3** (radial prefilter
before the exact overlap test) is already in the host re-cut since
`52dfb7c` (the rmin/rmax shell test per (cell, body) in `recut()`); what
is left under `recut: surf lists` is `Grid::surfs_in_cell` itself, so
little remains.  **C4** is the best-supported item after C1: collision
list enumeration is 1.40 s of the 1.66 s stage and runs 6.15M element
box tests per step for 107k entries (1.7% yield) over 224k (body, cell)
pairs; the exact cell-vs-swept-box test C4 asks for is already done
before the element loop, so the win must come from the same rmin/rmax
shell prefilter as C3 (cells deep inside a body pass the box test and
fail all 32 element tests), worth 0.7-1.0 s (4-6%).  **C5** is not
supported as a priority: `nscheck` is 0.072 per particle per step
(129k/step) and only 23k cells of 1.28M carry extra swept surfs;
`line_line_moving_intersect` is 3.0% of loop Ir, so shorter rows cannot
buy more than 1-2% - the 4.83 s Move is the generic particle advance.
**C6**: the fix scales 8.75 -> 3.03 s (2.9x) on 4 ranks and the new
min/max columns show little imbalance (recut 1.301/1.311, remove inside
0.799/0.847, collision lists 0.613/0.662); the 4-int Allreduce that
opens the re-cut is 0.031/0.097 s, i.e. the collective already absorbs
what imbalance there is.  Not worth a package at 4 ranks; revisit at 8+.
**C7** is not supported: no allocation frame appears in the loop-only
callgrind profile above 1%.  One item the plan does not list is now the
cheapest: `recut: split combine` (0.57 s) and `remove: split assign`
(0.58 s), 7% of the loop together, are two O(all owned cells) scans of
1.28M cells to find the ~175 split cells that changed in the step
(`FixRigid::combine_split_all()` and the `splitflag` loop at the head of
`remove_inside_all()`); walking the split-cell list instead visits the
same cells in the same order and is bit-identical by construction.
Priority for the CPU packages from this profile: C8 (split-cell scans,
~1.1 s), C1 (~1.1 s), C4 (~0.9 s), C2 (~0.7 s), then C3/C5/C6/C7.

CPU scaling curves at `78c40a7` (this 4-core container, `spa_mpi`,
40-step decks, `SPARTA_RIGID_TIMING=1`), measured because every earlier
CPU number was a single-point A/B:

- **Strong** (1000 bodies, 1131^2, 1.8M particles): replicated 14.97 /
  9.49 / 5.64 s and owned 14.63 / 9.06 / 5.09 s at 1 / 2 / 4 ranks, so
  2.87x on 4 cores in owned mode (2.65x replicated).  Owned is 2-9%
  faster than replicated at every rank count.
- **Weak** (1000 bodies + 1.8M particles per rank: 1k/566^2 at 1 rank,
  2k/800^2 at 2, 4k/1132^2 at 4; decks `in.w{1,2,4}k.{rep,owned}` in
  `/home/user/bench`): replicated 8.65 / 10.21 / 13.17 s, owned 8.60 /
  10.00 / 12.95 s, i.e. 66% efficiency at 4 ranks.
- **The weak loss is the machine, not the fix.**  The mover does
  constant work per rank and communicates nothing, and it grows
  2.74 -> 3.17 -> 4.45 s (1.62x), so 62% is this box's own weak-scaling
  ceiling from memory-bandwidth contention.  Against that reference the
  fix's stages scale at or better than the mover: remove inside 1.36x,
  collision lists 1.36x, recut 1.28x, contacts 1.66x, set_xv 1.77x.
- **The only superlinear stages are the two exchanges**, and they cost
  the same as the collective they replaced: owned `sum forces` 0.0061 ->
  0.0341 -> 0.1675 s against replicated's `MPI_Allreduce` at 0.0065 ->
  0.0455 -> 0.1523 s.  Both are the first synchronization point after
  the mover, so both absorb the inter-rank skew accumulated over the
  step; at 4 ranks that is 4 ms/step each, 2.5% of the loop.  This is
  the CPU confirmation that the mechanism is skew absorption, and that
  the GPU node's 325 ms per call was its MPI's unexpected-message
  handling, which PERF1 removes.
- Consequence for the next packages: on the CPU there is no
  communication problem left to fix; the remaining weak-scaling
  headroom is per-rank efficiency in `remove inside` (2.76 s) and
  `recut` (1.97 s), which is exactly Part 2's C1-C4.

**PERF2** landed as `ad1442e` (pushed, written inline): Part 3 item A,
the flat geometry stage.  The three geometry kernels ran twice per step
because `swept_boxes()` computed every element's end-of-step points on
its way to the swept box and discarded them, and `set_xv()` recomputed
them from the same pose (`set_pose()` only copies xcmnew/quatnew into
xcm/quat and leaves `ex/ey/ez_space` alone).  Each pass also uploaded
the pose and read three per-body arrays back.  The sweep pass now keeps
the end-of-step points, normals, element boxes and body boxes beside the
swept ones (`k_bodypt_new` and friends), and `set_xv()` swaps the handles
(`commit_geometry()`), writes the surfs and reads one packed `(lo,hi,eps)`
array back.  Per step: 3 launches and 3 transfers instead of 6 and 8,
confirmed with the census tool (`TagFixRigidGeometry/BodyBox/Inflate`
2.00 -> 1.00 launches per step).

Bit-identity argument, verified empirically: the sweep's pose row is
built from `xcmnew` and the end-of-step axes, `set_pose()` then makes
`xcm` equal `xcmnew` and does not touch the axes, and the 2d and
axisymmetric branches are enforced on `xcmnew` in `initial_integrate()`,
so the stored points equal the recomputed ones bit for bit.  Verified:
6 kk suites green (1/2 ranks, `--dist`, owned 1/2/4), the benchmark
identical at 1, 2 and 4 ranks in both modes, the 8-sphere 3d deck
identical, and `set_xv+bounds` down from 0.285 s to 0.088 s per 40 steps
under Serial (where a launch is free), with `integrate+bbox` unchanged at
0.35 s -- the point recomputation is what disappeared.

Next on the GPU: the body surfs still copy to the host every step
(3.4 MB) through the `output->next` condition the kk refresh gate kept
for replicated-mode safety (Part 3 item G), and `collide:vremax/remain`
moves both ways every step (10.3 MB, collide code, not the fix).

**PERF3** landed as `00b5106` (pushed, inline): the last two per-step
round trips from the user's GPU transfer table.

- *Surf copy, 3.4 MB/step.*  Not the `refresh_host_surfs()` gate in
  `FixRigidKokkos::end_of_step()` (measured: all three of its conditions
  were false on a plain step).  It came from `FixRigid::remap_grid()`,
  which after an incremental re-cut that changed cells or markings
  refreshes the host surfs and then notifies the emit fixes, whose
  per-cell tasks read host surfs and cells.  With no emit fix defined
  that loop does nothing, so the copy was waste, and the 1000-body deck
  changes cells every step.  `nemitfix`, counted once in `init()`, now
  gates both: 46 refreshes per 40 steps down to 6 (the output steps).
- *Collide, 10.3 MB/step both ways.*  `CollideVSSKokkos::add_grid_one()`
  and `copy_grid_one()` are per-cell callbacks that each synced
  `vremax`/`remain` to the host, wrote one cell and marked the host
  modified, so any step that restructures the grid paid a full round
  trip of both arrays; the bodies restructure ~340 cells per step
  (measured).  This is the hazard the plan's Phase S item 6 flagged
  ("check that collide/kk's add_grid_one/copy_grid_one do not force a
  host sync; if they do, give them a device path").  They now record a
  `(src,dst)` pair, `src < 0` = initial values, and
  `apply_cellops()` replays the pairs on the device in one pass at the
  top of `sync()`, `modified()`, `collisions()`, `backup()` and
  `reset_vremax()`.  Single threaded, since a later pair may read a cell
  an earlier one wrote and the work is a few values per pair.
  Recording is the only path, so every backend exercises it and the host
  branch is gone.

Verified bit-identical: the rigid suite on 1, 2 and 4 procs (replicated,
owned, distributed; CPU and kk), the benchmark, and -- because this
touches core collide code -- `examples/adapt`'s three decks and
`examples/ablation/in.ablation.2d` at 1 and 4 procs, which refine,
coarsen and migrate cells with collisions active.

Remaining per-step transfers on the GPU, by the user's table: 
`grid:stagecell` 5.2 MB (the journal records, Part 3 item E/F) and
`fix_rigid:ftally`, `particle:cellcount`, `rigid_remap:candflag` 4.1 MB
(zero-fills, trivial on a GPU).

# PLAN: Part 2 CPU per-rank work (C1-C4), 22 Sep 2026

## Context

Four packages have landed since owned bodies (`78c40a7` neighbor
exchange, `ad1442e` one geometry pass, `00b5106` the surf and collide
transfers).  What they removed was communication, transfers and launches.
What is left on both the CPU and the GPU is per-rank compute: on the GPU
the host side was ~90% of the loop at the last measurement, and on the
CPU the fix is 48% of the loop with no communication problem left (the
weak-scaling study showed every stage at or better than the mover's own
1.62x growth, which is this box's memory-bandwidth ceiling).

So the target is the work itself, in the three stages that carry it.
All numbers below are the 4-rank weak deck (4000 bodies, 1132^2 cells,
7.2M particles, 40 steps, `spa_mpi`, `SPARTA_RIGID_TIMING=1`), stage
seconds as avg over ranks and counters as totals across ranks per step:

| stage | s / 40 steps | counters per step |
|---|---|---|
| remove inside | 2.76 (particle pass 2.64) | 351,575 particles tested, **0 deleted** |
| recut | 1.97 (surf lists 0.78, candidates 0.39, cuts 0.30, retyping 0.15) | 141,486 candidate cells, 104,632 lists changed, 102,078 cells cut |
| collision lists | 0.83 (enumerate 0.72) | 357,944 (body,cell) pairs, **7,080,514 element box tests** |
| everything else in the fix | 0.63 | |
| loop / Modify | 12.95 / 6.19 | |

Two facts drive the design.  The deletion pass costs 21% of the loop and
deleted nothing in the entire run: it is a safety net for a mover that is
now correct, and it pays a COM-bin query plus a 32-element ray cast for
every one of the 88k particles per rank that sit in a cut cell.  And the
collision-list enumeration tests every element of a body against every
cell whose box overlaps the body's, with no test in between.

Each item below is applied to the host routine **and** its device twin in
the same commit, since the two must stay in lockstep, and each is
bit-identical unless it says otherwise.

## C1: the deletion pass (`FixRigid::remove_inside_all`, `fix_rigid.cpp`)

Per particle today: two random struct reads (`cinfo[icell].type`,
`cells[icell].nsurf`), then for the survivors `inside_any_body(x)` =
`body_box(x,x)` bin query + per candidate body `inside_body()`, a parity
ray cast over all of the body's elements.  Device twin:
`TagFixRigidRemoveInside` in `fix_rigid_kokkos.cpp` with
`RigidBodyKK::inside_any_body/inside_body` in `rigid_body_kokkos.h`.

1. **Measure the split first.**  Add a sub-timer separating the sweep
   from the tests, and a counter of ray-cast element tests, using the P0
   machinery (`T_*`/`C_*` enums, `add_time`/`add_count`).  The estimate
   is ~2/3 tests, ~1/3 sweep; the rest of C1 is ordered by what this
   shows.  One commit, no behaviour change.
2. **Hoist the per-cell work out of the particle loop.**  Once per step,
   over the candidate cells only, build a byte per cell (skip / test /
   test-and-delete, i.e. the compressed form of the two struct reads)
   and cache each cell's candidate body list from a single `body_box`
   query on the **cell box**.  That set is a superset of the per-particle
   set, and `inside_body()` is exact, so the boolean is unchanged.  The
   particle loop then reads one byte from a ~320 KB array and reuses the
   cached list, removing ~73k bin queries per rank per step and the two
   random reads into ~20 MB arrays.
3. **Reject before the ray cast.**  Apply the radial shell test the
   re-typing pass already uses (`rminbody`, `rmaxbody + bboxeps`, and the
   `cominside` flag for the inner case, `rigid_remap.cpp:735-750`) before
   calling `inside_body()`.  Exact, and it skips the 32-element cast for
   any particle inside the body's bbox but away from its surface.
4. **Iterate cells, not particles** (gate on item 1).  Walk the sorted
   per-cell particle lists of the candidate cells instead of sweeping all
   1.8M particles per rank.  Requires running the pass before
   `assign_split_cell_particles()` and testing every particle of a split
   cell unconditionally, since its sub-cell assignment is stale until
   then; split cells are few (612 pending per step).  Worth doing only if
   the sweep turns out to be a large share.
5. **Let a validated run pay less** (opt-in, changes results only when
   used): a `deletecheck N` value on the fix, default 1 = every step as
   now, documented in `doc/fix_rigid.txt` as a safety net whose cost a
   converged setup may not need every step.

## C2/C3: the re-cut (`RigidRemap::recut`, `rigid_remap.cpp`)

The candidate pass flags cells in each body's old union new region; the
surf-list pass rebuilds every candidate cell's list; the compare pass
then finds 105k of 141k changed.

- **C3, the same shell prefilter per (cell, body)** before testing a
  body's elements when building a candidate cell's list.  Shares the
  helper added in C1.3.
- **C2, skip cells whose list cannot have changed**: stamp each candidate
  cell with the set of bodies that reached it last step (a small hash of
  the body indices is enough) and skip rebuilding when the set and the
  static part are unchanged and no element of those bodies moved into or
  out of the cell.  The existing compare pass stays as the guard, so a
  wrong skip shows up immediately as a changed result.

## C4: the collision-list enumeration (`RigidRemap::collision_lists`)

`rigid_remap.cpp:280-312`: for every (cell, body) pair whose boxes
overlap it tests **all** of the body's elements, which is where the
7.08M element box tests per step come from.  Add the shell test between
the two: a cell whose centre is farther than `rmax + cell diagonal` from
the COM, or nearer than `rmin - cell diagonal`, cannot touch any element.
For a circle that should cut the tests roughly in half; the counter says
exactly how much.  Device twin: the `sw_count` kernel in
`rigid_remap_kokkos.cpp`.

C5 (mover row length) follows from C4 and is re-measured after it; C6
(balance weights) is a deck-level change, not code; C7 (allocation churn)
waits on a callgrind pass, which P0 already produced under
`/home/user/bench/prof`.

## Verification

- **Bit-identical is the gate for C1.1-C1.4, C2, C3, C4.**  Snapshot the
  current binaries first; then for every item: the rigid suite at 1, 2
  and 4 ranks, replicated, owned and `--dist`, on the CPU and Kokkos
  Serial EXACT builds; the 1000-body benchmark and the 4000-body weak
  deck identical at 1 and 4 ranks in both modes on both builds; the
  3d deck identical.  C1.5 adds tests for the new value and its default.
- **Each item must show its counter move**: C1.2/C1.3 the ray-cast
  element count, C2 the lists-changed count, C3/C4 the element box test
  count.  An item whose counter does not move is reverted, not kept.
- **Timing**: interleaved A/B against the snapshot binary, three runs
  each, on the weak deck at 4 ranks and the benchmark at 1 and 4 ranks,
  reporting loop time and the affected stage lines.
- Sanitizer run on the ownership deck and the owned benchmark at 2 ranks
  after C1.4, which changes the iteration order over particles.

## Execution

Inline, no agents, per the standing instruction.  One commit per item on
`surf-rigid-body-owned`, WIP checkpoints while a step is in flight,
pushed once its gate passes, with the plan's Status updated as each
lands.

**WP4** landed as `8d81671` (pushed, inline), completing Part 1: owned
mode now supports distributed surfs.

The only real blocker was the push-off contacts.  `RigidContact` bins
the static surfs a proc *owns*, which with distributed surfs is a
round-robin share, and replicated mode sums a partial push force from
every proc through an Allreduce.  An owner producing a body's complete
force alone cannot use that: the surfs it owns are not the surfs near
its bodies.  `RigidContact::local_surfs()` now selects this proc's local
and ghost copies whenever an owner computes alone (and always without
distributed surfs, where the arrays coincide), so replicated mode keeps
the owned arrays, the proc-0 rule and the reduction unchanged.  The
proc-0 rule for the pair and boundary terms also yields to the owner
rule.  A grid change and a local-copy append both restructure the local
arrays, so `FixRigid::rebin_contacts()` rebuilds the bins at each of the
four sites that do it -- the hazard that would otherwise leave `stamp`
undersized and the bins pointing at moved surfs.

Requirement added: with distributed surfs and `push`, the ghost layer
must reach as far as a contact, so `global gridcut` must be at least the
bodies cutoff (or negative).  Documented in `doc/fix_rigid.txt` in place
of the restriction removed.

New test `owneddist` compares owned against replicated on
`in.test.staticdist` and `staticdist3d` (a body drifting past a
200-segment circle and a 1200-triangle sphere, mostly ghost surfs on
several procs).  It is a real gate: with the previous binning it fails
at 4 procs, the body's y position differing in the 8th digit by step
500.  Verified: every suite combination green on 1, 2 and 4 procs (CPU
and kk, replicated/owned x plain/distributed), benchmark bit-identical
on both builds, ASan clean on both decks in owned distributed mode at 2
and 4 procs.

Also landed just before: `425bf21`, the surf scatter over the local
copies.  The GPU session's data had shown `set_xv+bounds` flat in the
proc count; a local diagnostic confirmed the body lists themselves are
partitioned correctly (each of 4 ranks owned ~250 of 1000 bodies and
held ~265), so the flatness was the scatter kernel running over every
copy of every body with a skip inside.  It now runs over the copies of
held bodies only.  The remaining flat cost is what is sized by the surf
arrays themselves, which only distributed surfs make scale -- now
available in owned mode.

**Part 2 C1-C4 landed** (22 Sep 2026, inline), three commits on
`surf-rigid-body-owned`:

- `b78f190` **C1**, the deletion pass classifies cells, not particles.
  Once per step every cell gets one byte (skip / test / delete-all) from
  its type and surf count, and the candidate body list of the cell a
  particle sits in is cached from a single `body_box()` query reused for
  the whole cell, with the radial shell test before the ray cast.  The
  particle loop then reads one byte per particle instead of two random
  struct fields.
- `b4ac4da` **C3+C4**, the per-cell element scans test groups first.
  Both scans which offer a body's elements to a cell -- the collision
  list enumeration and the re-cut candidate list -- tested every element
  of every body whose box reached the cell.  The elements now carry a
  box per group of `EGROUP = 8`, rebuilt from the element boxes at the
  end of `body_bbox()`, so a group box is exactly as fresh as the body
  box the same call computes and skipping a group is exact.  `EGROUP` is
  at its optimum for 32-element bodies: the counter says about one group
  per (body, cell) pair survives, so halving the group size would trade
  element tests for group tests one for one.
- `21a8a59` **C2**, the re-cut skips a candidate cell whose list holds
  no body surf and which no body element box reaches: `surfs_in_cell()`
  would return the same static surfs in the same order.  36,212 of the
  141,486 candidate cells per step, with `lists changed` unaltered,
  which is the guard.  Nothing more is skippable there: of the 105,274
  candidates a body does reach, 104,632 change.

Weak deck (4000 bodies, 1132^2, 7.2M particles, 40 steps, 4 ranks),
before C1 -> after C2:

| stage | before | after |
|---|---|---|
| Loop / Modify | 12.95 / 6.19 | 10.56 / 4.38 |
| remove inside (particle pass) | 2.76 (2.64) | 1.44 (1.35) |
| recut (surf lists / candidates / cuts) | 1.97 (0.78/0.39/0.30) | 1.73 (0.69/0.35/0.27) |
| collision lists (enumerate) | 0.83 (0.72) | 0.63 (0.55) |
| element box tests per step | 7,080,514 | 2,295,955 |

Interleaved A/B of C2+C3/C4 over the C1 binary, three runs each:
1000-body bench 1 rank 15.67 -> 14.67 s (-6%), owned 4 ranks 4.72 ->
4.20 s (-11%), weak deck 4 ranks 11.37 -> 10.17 s (-11%).

Bit-identical throughout: the rigid suite on 1 and 4 ranks in
replicated, owned and distributed modes on both the CPU and Kokkos
Serial EXACT builds (10 suite runs per commit), and the benchmark in
both modes on both builds at 1 and 4 ranks.

C5-C7 stay unsupported by the profile, for the reasons in the P0
section (mover rows can buy 1-2%, the fix already scales 2.9x on 4
ranks, no allocation frame reaches 1%).  The largest CPU item left is
the one P0 found and the plan does not list, **C8**: `recut: split
combine` and `remove: split assign` are two O(all owned cells) scans of
the grid to find the ~600 split cells which changed in the step; on the
1000-body bench they are 0.49 + 0.47 s of a 14.5 s loop.  Walking the
split-cell list visits the same cells in the same order, so it is
bit-identical by construction.

The device twins of C2 and C3/C4 are not in: the prefilters are exact,
so the device results are unchanged without them, and the device needs
`d_groupstart`/`d_elemglo`/`d_elemghi` with their own sweep-pass copies
and handle swap in `commit_geometry()` before the `sw_count` and re-cut
candidate kernels can use them.

**C8 landed** as `134e75d` (22 Sep 2026, inline).  `Grid::owned_split_
cells()` builds the owned split cells from `sinfo`, which holds them
before the ghost ones, and sorts the list, so the three callers (the
split-cell reassignment at the head of both deletion passes and
`combine_split_all()`) walk the same cells in the same order.  The
KOKKOS fix already enumerated them this way and its guard against an
entry a restructure abandoned (`icell < 0 || cells[icell].isplit != i ||
cells[icell].nsplit <= 1`) is the one used here.

`recut: split combine` 0.489 -> 0.0005 s and `remove: split assign`
0.474 -> 0.0002 s per 40 steps at 1 rank.  Validated first with a
temporary check that the list equals the full scan's cells in the
scan's order, which held across every suite and both benchmarks, then
bit-identical on the usual matrix.

Interleaved A/B of C2+C3/C4+C8 over the C1 binary, three runs each:
1000-body bench 1 rank -8.4%, owned 4 ranks -15.9%, weak deck 4 ranks
-11.7%.

**None of C1-C4 reaches the GPU path yet.**  `remove_inside_all()`
(C1), `RigidRemap::collision_lists()` (C4) and `RigidRemap::recut()`
(C2, C3) are all virtual and overridden by the KOKKOS classes, so the
host code they changed runs under `-sf kk` only in the rare host
fallback of `FixRigidKokkos::remove_inside_all()`.  Nothing is needed
for correctness there -- each is an exact filter, which is why the kk
suites and benches were bit-identical -- but the device twins are the
work that would make the GPU see these wins:

- C3/C4 on the device: `d_groupstart`/`d_elemglo`/`d_elemghi` in
  `RigidBodyKK` with a group-box kernel after `TagFixRigidInflate`,
  their own sweep-pass copies and a handle swap in `commit_geometry()`,
  then the group test in the `sw_count` kernel and in the re-cut
  candidate kernel of `rigid_remap_kokkos.cpp`.
- C2 on the device: the same all-static/no-body-reaches early-out in
  the re-cut candidate kernel.
- C1 on the device: the per-cell classification and cached body list in
  `remove_inside_all_kokkos()`.
- C8 needs nothing: the KOKKOS path was already `sinfo`-based.

**C3/C4 on the device** landed as `bfc3aa5` (22 Sep 2026, inline).  A
box per element group rides beside the element boxes: `k_elemglo/
k_elemghi` with `_new` copies, built by `TagFixRigidGroupBox` after
`TagFixRigidInflate` over `k_lgroup` (the groups of the blist bodies,
packed with `k_lelem`), and swapped with the element boxes in
`commit_geometry()`, so a group box is always as fresh as the boxes it
bounds.  The group's element range is a CSR `k_groupelem(ngroup+1)`
over all bodies -- the last group of a body ends where the next body
starts -- uploaded once with `k_groupstart` in `pack_body_static()`, so
the device needs no copy of `EGROUP`.  `RigidBodyKK` gains
`d_groupstart/d_groupelem/d_elemglo/d_elemghi` and `group_overlap()`;
the `sw_count` kernel and the re-cut candidate kernel scan groups first.
The new-ghost pre-pass builds the group boxes of the bodies it
regenerates, over its own `k_newgroup`.

Order is preserved because the groups are consecutive element ranges,
so the elements are still visited ascending and the hit order the host
merge depends on is unchanged.

Kokkos Serial, owned bench at 1 rank: collision lists 1.79 -> 1.63 s per
40 steps, recut surf lists 2.63 -> 2.58, loop 17.94 -> 17.43 s.  A
kernel launch is a plain loop under Serial, so that is the arithmetic
only; the GPU effect needs the user's H100 run.

Gates: the rigid suite at 1 and 4 ranks in replicated, owned and
distributed modes (the 3d decks among them), the benchmark in both
modes at 1, 2 and 4 ranks, and -- because this adds device arrays and a
handle swap, which Serial's aliased memory cannot check -- the
coherence emulation build (`/home/user/sparta-sync`, merged to this
branch as local commit `50744cb`) with `SPARTA_KOKKOS_WATCH/STALE` on:
suites green at 1 and 4 ranks plain, owned and distributed, no watch or
stale report, and the 2-rank owned bench bit-identical to the plain
Kokkos run.

Still unported to the device: C1 (the per-cell classification and
cached body list in `remove_inside_all_kokkos()`) and C2 (the
all-static/no-body-reaches early-out in the re-cut candidate kernel).

# STRUCTURAL ASSESSMENT (22 Sep 2026): why it still does not scale

Measured on this 4-core VM with the current head (`bfc3aa5`), 40-step
decks, `SPARTA_RIGID_TIMING=1`.

## 1. The work is partitioned correctly.  This is settled.

Weak series (1k/566^2 at 1 rank, 2k/800^2 at 2, 4k/1132^2 at 4, owned),
counters summed over ranks, so exactly 4.00x means per-rank work is
constant:

| counter | 1 rank | 4 ranks | ratio |
|---|---|---|---|
| particles tested | 3,509,653 | 14,062,988 | 4.01 |
| candidate cells | 1,415,197 | 5,659,420 | 4.00 |
| lists changed | 1,046,189 | 4,185,278 | 4.00 |
| cells cut | 1,020,596 | 4,083,137 | 4.00 |
| element box tests | 22,961,016 | 91,838,208 | 4.00 |
| swept entries | 2,586,636 | 10,344,454 | 4.00 |
| (body,cell) pairs | 2,842,615 | 14,317,754 | 5.04 |

Only the (body,cell) pairs rise faster, by the 26% of ghost cells a
rank also enumerates.  Owned mode did its job; there is no replicated
enumeration left to remove.

## 2. This VM cannot weak-scale, for any code.  Stop measuring it here.

Same series, per-rank work constant for every row:

| stage | 1 rank | 4 ranks | degradation |
|---|---|---|---|
| Move (pure SPARTA, no fix, no MPI) | 2.443 | 3.809 | **1.56x** |
| Coll (pure SPARTA) | 0.464 | 0.604 | 1.30x |
| Sort | 0.925 | 1.056 | 1.14x |
| **Modify (the fix)** | 2.701 | 3.826 | **1.42x** |
| Loop | 6.536 | 9.632 | 1.47x |

The particle mover communicates nothing and does identical work per
rank, and it loses 1.56x.  That is this box's shared memory bandwidth,
and it is the ceiling.  The fix degrades *less* than the mover.  Strong
scaling says the same: loop 2.53x on 4 ranks against the mover's 2.75x.
So the CPU VM's poor scaling is the machine, not the rigid code, and no
change to the fix will move it.

## 3. Distributed surfs are not the lever (tested, not assumed).

The benchmark decks use replicated surfs, so every rank holds all
128,000 lines and the per-step tally array is sized by the global surf
count.  Weak series re-run with `global surfs explicit/distributed` and
`gridcut 0.02` (decks `in.w{1,2,4}k.owndist`): 6.73 / 8.13 / 10.15 s,
i.e. 66% efficiency against 68% replicated.  No gain on the CPU.  It
may still matter on a GPU, where `d_ftally` is a 6 MB zero-fill and a
ScatterView duplication per step, but it is not an algorithmic cap.

## 4. The one fix-specific term that does not scale: the sync points.

| stage | weak 1 rank | weak 4 ranks | kk strong 1 rank | kk strong 4 ranks |
|---|---|---|---|---|
| integrate: exchange | 0.001 | 0.138 | 0.001 | 0.229 |
| forces: exchange | 0.000 | 0.186 | 0.000 | 0.591 |
| recut: reduce (4-int Allreduce) | 0.000 | 0.133 | 0.000 | 0.244 |
| total | 0.001 | **0.457** | 0.001 | **1.064** |

1.06 s of a 7.01 s Kokkos loop at 4 ranks, from nothing at 1 rank.  The
compute stages around them are balanced to 2-3% (Modify min/max
3.759/3.856), so this is not gross load imbalance; it is three
synchronous points per step on the critical path, each fencing the
device and absorbing whatever skew accumulated since the last one.

## 5. So the GPU's problem is what the CPU cannot show

The work partitions, and on the CPU the fix is already at the machine's
ceiling.  Whatever caps the GPU must be a term that is free or absent
under Serial:

- **Host work sized by the global body count, per step, on every rank.**
  `UpdateKokkos::rigid_upload()` loops over all `nbody` and uploads an
  `nbody x 22` array every step, FAR bodies included.  `body_bins()`
  runs a `for i < nbody` scan for `rmaxall`, then destroys and creates
  a whole-domain bin grid of about `2*nbody` bins plus an `nbody` list,
  every step.  `k_bbox` comes back and `k_pose` goes up as whole
  `nbody` arrays.  `body_status()` is O(nbody).  On the CPU these are
  microseconds and invisible in the table above.  On a GPU, where the
  kernels are ~7% of the loop, they are a fixed per-step host cost that
  four GPUs do not divide -- and in a weak study they grow 4x on every
  rank.
- **Per-step fences.** About ten stage boundaries, two or three
  `apply_changes()` calls, and the three MPI points: a fixed sequence
  of synchronizations whose length is independent of the GPU count.
- **The host owns the grid.** The device cuts the cells, the results
  come back, the host mutates `Grid` (pages, pointers, split info), the
  journal goes up again.  It partitions, but the constant is
  latency-bound pointer work that no GPU accelerates, and it is what
  the last GPU profile measured as ~90% of the loop.

In one sentence: **the fix is compute-partitioned but latency-serial.**
Everything left is per-step fixed cost, and adding GPUs divides only
the part that is already small.

## 6. What to do, in order

- **R0. Re-measure the GPU.**  The last profile predates seven
  packages (PERF1-3, 425bf21, WP4, C1-C4, C8, the device group
  prefilter).  Needed: per-stage times at 1, 2 and 4 GPUs on the same
  deck, plus the census launch/copy counts, plus the host-vs-device
  split.  Without it the three bullets in section 5 cannot be ranked,
  and each implies a different next package.
- **R1. Localize the O(nbody) per-step work.**  `rigid_upload` over
  held rows into a persistent view; `body_bins()` over the rank's own
  box with the held bodies and grown buffers instead of a malloc/free
  per step; `k_bbox`/`k_pose` transfers over held rows; `body_status()`
  over the neighbor set.  Contained, keeps the global body index the
  user accepted, and removes the only per-rank work that grows with the
  global problem size.
- **R2. Take the MPI points off the critical path.**  The 4-int
  Allreduce reports rare events (fallback, structural change): make it
  an `MPI_Iallreduce` posted at the end of step n and acted on at the
  start of n+1, one step late.  The forward exchange depends only on
  `final_integrate`, so post it there and wait at the next step's
  geometry, hiding it behind the re-cut and the deletion pass.  The
  reverse exchange genuinely needs the mover's tallies and stays.
  Three synchronous points become one.
- **R3. Device-authoritative incremental grid (Tier 2).**  Keep
  `csurfs/cinfo/sinfo` on the device for the incremental path and
  materialize the host copy only when `host_surfs_needed()` says so.
  This is the item that removes the host from the per-step loop
  entirely, and on the evidence so far it is the largest GPU win left.
  It is also the largest package.
- **R4. Balance on the right weight.**  The fix's work follows the
  cells near bodies, the mover's follows the particles.  On this deck
  the bodies are uniform and the imbalance is 2-3%, so it does not
  matter yet; for any clustered configuration `fix balance rcb` needs a
  weight including the per-cell surf count.  Deck-level, no code.
- **R5 (a question, not a proposal).**  Each rank re-cuts about 8% of
  its cells every step (25,520 of ~320,000 at 4 ranks).  That is the
  dominant compute and it exists to keep volumes, split pieces and
  inside/outside typing current; the collisions themselves do not need
  it, since the swept lists make the mover exact against the
  analytically moving surf.  Amortizing it (re-cut every K steps, or
  only when a cell's cut topology rather than its surf list changes)
  would remove most of that compute, but it changes results and is a
  method change, so it belongs to the user, not to a package.

# R1 + R2 landed (23 Sep 2026) as `619cd84`, and what the barrier probe found

**R1, per-step work sized by the global body count.**
`UpdateKokkos::rigid_upload()` built and uploaded a row for every body,
FAR ones included, on every rank every step; the move kernel only ever
indexes a body by a surf it hit, and a surf in a local cell belongs to a
held body, so it now fills the held rows.  `exchange_reverse()` zeroed
all `6*nbody` partials though only held rows are written and read.
`body_bins()` destroyed and created its two arrays on every call.  The
bin grid geometry and the global `rmaxall` are deliberately unchanged:
`body_box()` returns partners in bin order and the contact sum depends
on that order, so a per-rank bin grid would break owned == replicated.

**R2, the count trade covered by the packing.**  Each exchange traded
its per-slot counts in one round trip and the records in a second.  The
counts are final after the counting pass, so the trade is posted there
and finished after the packing pass.  The counts travel in `nsendcount`,
since the packing pass reuses `nsendslot` as its fill cursor.  Receives
stay pre-posted -- no probe-based or unexpected-message receive, which
is the pattern that cost 325 ms a call on the user's GPU node before
PERF1.

**Measured**, weak deck, 4000 bodies at 4 ranks, three interleaved runs:
forward exchange 0.251 -> 0.086 s per 40 steps (about 4 ms a step);
reverse exchange unchanged.

## The barrier probe: what the sync points actually cost

A temporary timed `MPI_Barrier` before each of the three sync points
(removed before the commit), weak deck at 4 ranks on an idle machine,
max over ranks per 40 steps:

| sync point | no barrier | barrier | barrier absorbed |
|---|---|---|---|
| forward exchange | 0.122 | 0.048 | 0.028 |
| reverse exchange | 0.168 | 0.115 | 0.073 |
| re-cut Allreduce | 0.214 | **0.0014** | 0.062 |

The re-cut Allreduce gets **150x cheaper** once a barrier precedes it,
so it is skew absorption almost entirely, and no restructuring of the
fix can remove it -- only balance can.  The reverse exchange keeps most
of its cost after a barrier, so about 0.11 s of it is genuine
point-to-point latency.  The forward exchange is roughly half and half,
and it is the half R2 removed.

This settles the R2 question: of the three sync points, only the two
point-to-point exchanges had latency to attack, the forward one had
packing work to hide it behind, and the collective had nothing to
overlap with (the flags are final at the very end of `recut()` and are
used immediately).  The remaining cost at those points is imbalance
created elsewhere in the step -- the Sort spread alone is 8.9% on the
Kokkos strong run -- which is R4's territory, not the fix's.

## Where that leaves the scaling question

R1 and R2 together are worth 1-2% on the CPU.  They were worth doing
because R1 removes the last per-rank work that grows with the global
problem size, and R2 removes a round trip whose absolute cost is the
same on a GPU while the step is five times shorter.  Neither is the
answer to the GPU's strong scaling, and the probe now says why the
sync points cannot be: they are paying for imbalance, not for MPI.

So the open items are unchanged and R0 is still the gate: a fresh
per-stage GPU profile at 1, 2 and 4 GPUs.  At 1 GPU the loop was 50 ms a
step and at 4 GPUs 47.5 ms; if the compute partitions, roughly 45 ms a
step is not partitioning, and nothing measured locally accounts for a
term that large.  The candidates remain R3 (the host owns the grid, so
every step round-trips the changed cells through host pointer work) and
per-step launch and fence latency, and only the GPU can tell them apart.

# R3 sized, not yet implemented (23 Sep 2026), instrumentation as `ddcbd2c`

I went to write R3 and measured its target first, because host code costs
the same under Kokkos Serial as it does on a GPU, so the saving can be
read off a local run exactly.

`recut: install` is now a permanent sub-timer covering the host install
of the device cut results (it was charged to `recut: cuts` with the
device cut itself), and the per-run re-cut line reports how many cells
changed their piece count, counted in `RigidRemap::apply_cut()` so both
paths fill it.  Kokkos Serial, owned benchmark:

| | 1 rank | 4 ranks |
|---|---|---|
| recut: cuts (the device cut) | 0.6257 s | 0.1695 s |
| **recut: install (host)** | **0.2775 s** | **0.0734 s** |
| cells cut | 2,043,927 | 516,732 |
| piece counts changed | 5,671 | 1,433 |
| install per step | **6.9 ms** | **1.8 ms** |
| exception rate | 0.28% | 0.28% |

**R3 is well conditioned.**  Only 0.28% of cut cells change their piece
count, and those are the only ones that must reach the host, because
only they add or remove sub cells and need `restructure_split_cells()`
and its Allreduce.  142 cells a step at 1 rank, 36 at 4.  The other
99.7% are in-place edits of fields the device already mirrors.

**But its ceiling is 14% at 1 GPU and 4% at 4.**  6.9 ms of a 50 ms step,
1.8 ms of 47.5 ms.  It improves absolute speed more than scaling, since
the install partitions cleanly with the rank count.  Against the ~45 ms
a step that is not partitioning on the GPU, R3 is not the answer.

**The alternative that fits the flat strong scaling better, and costs
one GPU run to test: occupancy.**  The device cut is one thread per
cell running a serial clipping algorithm on scratch rows.  At 1 GPU the
bench cuts 51,098 cells a step; at 4 GPUs 12,918 per GPU.  An H100 has
132 SMs, so 12,918 threads is about 3 warps per SM of long, divergent,
register-heavy work -- far below what fills the device, and the kernel
would take nearly the same wall time as at 1 GPU.  That is exactly the
shape of 2.02 / 1.93 / 2.05 s at 1 / 2 / 4 GPUs.  The test is to compare
per-kernel times, not loop times, at 1 and 4 GPUs: if the per-GPU kernel
time is flat while its cell count falls 4x, occupancy is the wall, and
the fix is more parallelism per kernel (a thread per candidate surf or
per cell corner rather than per cell, or several cells per warp), not
less host work.

## R3 staged design, if it goes ahead

1. **Split the install by exception.**  The device already knows each
   changed cell's new `nsplit`; partition the changed list into the
   99.7% whose piece count is unchanged and the 0.28% that must go to
   the host.  Only the second list comes back.  Gate: bit-identical,
   `recut: install` falls by the same share.
2. **Apply the 99.7% on the device.**  `d_cells[icell].nsurf` and the
   `d_csurfs` row from `d_chlist` (which `build_csurfs_device()` already
   assembles, but today from host journal records that are a round trip
   of data the device had), `d_cinfo[icell].type/volume/corner`, and the
   `d_csplits` row for a split cell whose piece count did not change.
3. **Mark the host copy stale and materialize on demand.**  The
   predicate exists: `host_surfs_needed()` already names the steps with
   a host consumer (dumps, restarts, balance, adapt, emit).  Add the
   grid fields to what it covers and rebuild the host pages from the
   device CRS there.  This is the step with the correctness surface:
   every host reader of `cells[].nsurf/csurfs`, `cinfo[].type/volume/
   corner` and `sinfo[].csplits` has to be audited, `Comm::migrate_
   particles` included.
4. **Keep the 0.28% path exactly as it is.**  It already works and it
   is 36 cells a step at 4 ranks.

The emulation build (`/home/user/sparta-sync`) is the gate for step 3,
since Serial's aliased memory cannot see a host copy that was never
refreshed.

# Correction and the real scaling answer (23 Sep 2026)

**My error**: I read the GPU session's "replicated vs distributed"
tables as a body-mode axis and told the user to switch to owned mode.
They were in `bodies owned` throughout; the axis was the surf mode.
The advice was wrong and the diagnosis with it.

**Settled locally instead**, Kokkos Serial, 1000-body benchmark, 1/2/4
ranks, sequential runs on an idle machine, speedup 1 -> 4:

| stage | owned | replicated |
|---|---|---|
| LOOP | **2.83x** | 2.54x |
| integrate+bbox, less the exchange | **1.24x** | 0.95x |
| set_xv+bounds | **1.63x** | 1.24x |
| forces: tallies | **1.12x** | 0.96x |
| collision lists | 2.99x | 2.65x |
| recut | 3.37x | 3.03x |
| recut: install | 4.26x | 3.89x |
| remove inside | 3.28x | 3.07x |

Two conclusions, and they are the answer to the GPU curve.

1. **Owned mode works.**  It beats replicated on every stage and on the
   loop.  It is not failing to partition what it claims to.
2. **It partitions the cell work and not the body/element work.**  The
   cell-sized stages scale 3.0-4.3x.  The three body/element-sized
   stages scale 1.1-1.6x, which is why the user's GPU table shows
   `integrate+bbox`, `set_xv+bounds` and `forces: tallies` flat while
   `recut` scales.  At 4 GPUs and 4000 bodies those three are 2.475 s of
   a 5.85 s loop -- **42% of the step**, which caps strong scaling at
   about 1.5x however good the rest is.

**Why they do not partition**: the per-step structures behind them are
still sized by the global problem, on every rank.

- `d_ftally` is sized by the global element count and
  `Kokkos::deep_copy(d_ftally,0.0)` zeroes all of it every step on every
  rank (6.1 MB at 4000 bodies), while `TagFixRigidSumTallies` correctly
  runs over `nblist`.  Only held bodies' element rows are ever written
  (a tallied surf lies in a local cell) or read, so zeroing `d_lelem`'s
  rows is exactly equivalent.
- `k_pose` goes up and `k_bbox` comes back as whole `nbody` arrays every
  step, though only held rows are filled and read.
- `body_bins()` rebuilds a whole-domain grid of about `2*nbody` bins and
  `pack_body_device()` uploads `bodybinstart(nbins+1)` plus
  `bodybinlist(nbody)` every step.  The bin *geometry* must stay global
  and identical on every rank, because `body_box()` returns contact
  partners in bin order and the contact sum depends on that order; the
  per-step *cost* need not be.
- `body_status()` is O(nbody) per step, and `rigid_upload()` still syncs
  the whole `nbody x 22` view even though R1 made its host fill local.

**More bodies is the wrong direction.**  Every item above grows with the
global body count on every rank, which is why `integrate+bbox` went from
about 0.35 s at 1000 bodies to 1.67 s at 4000.

**Next step, and it is a measurement first.**  I have twice picked a
target by reasoning and been wrong, so before touching any of the five
items above: sub-timers inside `integrate+bbox` and `set_xv+bounds`
separating the host fill, the transfers, the kernels and the read-back.
That names which of the five carries the 42%, and it is an hour's work
against a package that would otherwise be guesswork.

# The functor copy, and the launch census (23 Sep 2026)

**The body/element stages did not partition because every `*this`
launch deep-copied a global hash map.**  `FixRigid::idmap` was a
`std::unordered_map` held by value, one entry per body element in the
global problem.  `parallel_for(..., *this)` copy-constructs the whole
fix on the host at every launch, so each of fix rigid/kk's launches
copied and freed it: 1.3 ms at 32,000 entries, 6.7 ms at 128,000 on the
VM.  That is the whole of `integrate+bbox`, `set_xv+bounds` and
`forces: tallies` staying flat with the rank count and growing with the
body count, and it explains the 42 ms a step the GPU showed.  The Kokkos
launch mechanism plays no part: the copy happens before it.  Fixed by
holding it by pointer (`805ca2ae`).  Kokkos
Serial, in.bench40.owned, 4 ranks: geometry kernels 0.288 -> 0.034 s,
loop 6.67 -> 5.37 s.

**Census, not estimates.**  A Kokkos Tools library counting launches,
deep copies, fences and allocations per step (difference of a 20 and a
40 step run, rigid path only, 1000-body deck, 1 rank):

| | before | after |
|---|---|---|
| kernel launches | 41.5 | 30.5 |
| View allocations | 20.3 | 2.1 (growth still settling) |
| fences | 109 | 60 |

Under Serial a DualView sync is a no-op and is not counted; on a GPU
each is a full-extent copy.  Those were cut as well: the re-cut's 12
changed-cell read-backs are 3 copies of the part in use, its 6 uploads
2, the collision lists' 4 uploads 1, the split graphs' 4 uploads 2, and
the 5 scalar read-backs 2.

What changed, all bit-identical:

- geometry: Geometry + ZeroTally per element, then one team per body for
  BodyBox + Inflate + GroupBox (5 launches -> 2); a team per body also
  parallelizes the body box of a large body, which was one thread
- grid patch: the cell/hash/halo/sinfo scatters are one kernel; the CSR
  rebuild's two init kernels are one, its set/last record passes one
  atomic max, its count folded into the scan (11 -> 6)
- re-cut: one scan of a packed (row, offset) prefix and one pack replace
  scan/pack/offscan/fill (4 -> 2), and the fallback flag comes back with
  the totals, so three read-backs are one
- split assign: the max reduce is folded into the scan (2 -> 1 launches,
  2 -> 1 read-backs)
- the per-step allocations: the split graphs are subviews of persistent
  buffers, the re-cut's fallback flag and the swept counters one scalar
  view, and every buffer grown by a per-step count gets 10% extra

# R3 stage 2: the device installs the unsplit cells alone (24 Sep 2026)

Stage 1 set the in-place cells' fields on the device but still installed
every changed cell on the host (`set_cell_surfs` + `apply_cut`), and
journaled its list.  Stage 2 drops the host install for the cells that
are unsplit before and after (nsplit 1 -> 0 or 1), nearly all of the
~53,000 changed cells a step on the 1000-body deck.

- **Device**: `rc_devinstall` also names those cells (`o_ipcell` range of
  `d_chint`, device only), and `GridKokkos::defer_cut_lists()` holds the
  records.  The next rebuild of `d_csurfs` applies them, which is the
  one `apply_changes()` does for the 0.28% exceptions anyway
  (`build_csurfs_device()` takes a device record as `-2-m` in `d_rowrec`,
  and a journal record overrides it).  A first version rebuilt the CRS
  on its own (`replace_cut_lists`): +0.31 s of `recut: cuts` on Serial,
  which ate the whole gain.  `flush_cut_lists()` still does that, only
  for a reader which cannot wait.  Pass 2 runs on the device too
  (`rc_devtype`) and returns only the types which changed.
- **Host**: the cells are marked stale (`GridKokkos::mark_host_stale`);
  their `nsurf/csurfs` and `cinfo` type, corners and volume are the
  device's until `refresh_host_cells()` packs them back.  A cell the
  host installs later (exception, failed cut) is unmarked first.
  Stale cells are owned, never split and never move (only sub cells do
  in place).
- **Refresh points**: `GridKokkos::sync(Host, CELL|CINFO)` (every Kokkos
  style that reads the host grid syncs first: balance, adapt, move_surf,
  create_particles, compute reduce, variables) except fix rigid/kk's own
  per-step syncs (`refresh = 0`); `sync(Device)` under auto_sync;
  `grow_cells`; `compact_surf_lists`; `RigidRemap::refresh` (static
  flags); `apply_pending(rebuild)` (ghosts are sent); the emit-fix
  notification; the host deletion fallback; `grid_rebuild`, setup,
  post_run; and end_of_step when output, a dump/restart, or a balance or
  adapt fix on this or the next step reads the host.
  `wrap_kokkos_graphs()` discards the marks (host rebuilt wholesale).
- **Journaled stale cells**: the in-place restructure repairs neighbor
  links of cells next to moved ghosts and journals them.  `apply_changes`
  stages such a cell with a flag and the scatter keeps the device's
  `nsurf`, type, corners and volume.
- `device_lists_ok()` does not reject `auto_sync`: fix rigid/kk is not a
  `kokkos_flag` fix, so ModifyKokkos runs it with auto_sync on, and the
  first cut of this check disabled stage 2 entirely.

Serial, in.bench40.owned, EXACT: bit-identical at 1 and 4 ranks;
`recut: install` 0.295 -> 0.080 s, `recut: retyping` 0.013 -> 0.004 s,
`recut: cuts` unchanged.  Serial aliases the two copies, so the stale
host fields are only exercised by the split-memory debug build
(`SPARTA_KOKKOS_DEBUG_SYNC`, applied in a scratch worktree only).
