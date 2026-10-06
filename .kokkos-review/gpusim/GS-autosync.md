# GS-autosync: auto_sync claims the host over a standing device claim

Scope: `ParticleKokkos::sync()`, `GridKokkos::sync()`, `SurfKokkos::sync()` each did
`if (auto_sync) modify(Host,mask);` in the Device branch -- the shape fixed in
`CollideVSSKokkos::sync()`. If the pair is device-claimed, `modify_host` aborts
("concurrent modification of host and device views in DualView").

Binaries (S = scratchpad):
- `$S/gpusim/GS-autosync/spa_B_unpatched`: private build of gpusim-B (0778ebfc), worktree
  `$S/gpusim/GS-autosync/src` (detached), build dir `$S/gpusim/GS-autosync/build`, same
  options as bsync_B (Serial, MPI, KISS, Release, SPARTA_KOKKOS_DEBUG_SYNC=on).
- `spa_B_fixed`: + the first patch (auto_sync refresh in all three sync()s).
- `spa_B_fixed2`: + the final patch (that, plus the GridKokkos prewrap guard change, below).
  This is exactly `git diff` of the three files in /home/user/sparta (`$S/gpusim/GS-autosync/autosync.diff`).
- A_sync = `$S/bsync_A/src/spa_`.
Decks are in `$S/gpusim/GS-autosync/decks`. Stock examples were copied to `$S/gpusim/GS-autosync/ex/<dir>`.
"cpu" means the same binary run without `-k on -sf kk` (plain styles, no DualViews). It is only
compared on deterministic columns: decks p*/g* have no collisions (ncoll = 0), and in the stock
examples I compare grid/surf custom columns, or step-0 rows.

## Where the device claim comes from (static analysis)

auto_sync is 0 only inside `UpdateKokkos::run()`'s loop. Everywhere else it is 1: setup, between
runs, non-Kokkos fix hooks in ModifyKokkos, and comm/serial migrate. Inside an auto_sync region
every `modify(Device)` immediately syncs the host, and every non-Kokkos fix hook (or the
Kokkos-wrapped adapt/balance/move_surf fixes) syncs particles/grid to the host first. Particle,
grid and surf data are therefore never left with a *modify*-made device claim inside an auto_sync
region. The collide case differed because nothing synced collide's arrays.

The claim comes from **`DualView::resize()`**. Kokkos resizes on the device when the counters
tie and marks the device modified; the tool models this. The resizes involved are the per-custom
inner views:
- `ParticleKokkos::grow_custom()` (non-prewrap): `sync(Device,CUSTOM_MASK)`, then `grow_kokkos()` resizes on the device and leaves (0,1).
- `GridKokkos::reallocate_custom()` (non-prewrap): the same.
- `GridKokkos::reallocate_custom()` (prewrap): it intends `sync(Host)+modify(Host)` to force a host resize, but
  **GridKokkos::sync/modify returned early for Host under prewrap**, so these were no-ops. The resize went to the device,
  and read_restart then wrote the restart values to the host without a claim.
- Surf: `SurfKokkos::reallocate_custom()` always claims the host first, so the resize happens on the host. Nothing calls
  `modify(Device)` on lines/tris/mylines/mytris/custom. No device claim on SurfKokkos-managed data was found.

## Particle -- REPRODUCED (B and A), FIXED, VERIFIED

positive control: `in.p2` (2 custom particle vectors created after the first create_particles; run 20;
`create_particles air n 30000` between runs, which forces `grow()`; run 20). Also `in.p3`: p2 plus `custom particle set`
of both vectors and `compute reduce sum p_pv p_pi` in stats. Also `in.p33d` (3d).
- B_unpatched and A_sync: abort, `modify_host ERROR ... DualView "particle:dvector"`. Trace (`SPARTA_KOKKOS_TRACE=particle:dvector`):
  `resize flags=(0,0)` then `modify_host flags=(0,1)` -> abort. Backtrace: ParticleKokkos::modify <- ParticleKokkos::sync
  <- ParticleKokkos::grow_custom <- ParticleKokkos::grow <- Particle::add_particle <- CreateParticles::create_local.
  That is, the second grow_custom() of one grow(): its auto_sync `modify(Host,CUSTOM)` hits vector 0, which the first
  grow_custom() just resized on the device.
- np 2 / np 4 (`-var NADD 40000/80000`): B_unpatched aborts the same way.
- `in.p1` (1 custom vector) does not abort: the next add_particle does `sync(Host)` before anything syncs to the device.
negative control: `in.p0` (no custom). B_unpatched = B_fixed = B_fixed2 = cpu.
necessary: yes. The abort comes from `ParticleKokkos::sync()`'s auto_sync `modify(Host)` over a device claim.
complete: B_fixed2 runs p0, p1, p2, p3, p33d at np1 and p0, p3 at np2 and np4 to completion. Stats (np, c_pr sums of the
custom values) equal cpu on all of them. WATCH/STALE/STRICT: no watch reports. The stale reports beyond the
p0/g0 controls are `particle:{d,i}vector <- ParticleKokkos::grow_custom` (see follow-ups).
verdict: bug real and reachable with only stock commands (custom particle + create_particles between runs). The fix is
correct for this path: the device holds the resized data and the host is refreshed from it.

## Grid -- REPRODUCED (B and A, including 4 stock examples), FIXED, VERIFIED

positive controls:
- `in.g1a`: 50x50 grid, `custom grid create g1 float 0`, run 20, `custom grid set g1 v_gx`, `adapt_grid all refine
  random 1.0 0.0 cells 2 2 1` (two grow_cells in one refine), `custom grid create g2`, run 20.
- `in.g1b` / `in.g1c`: g1a with no second custom; g1c sets g1 again after the adapt. `compute reduce sum g_g1` in stats.
- `in.g3`: in-run `fix adapt 10 all refine random ...` (non-Kokkos fix -> auto_sync).
- `in.g1c3d` (3d).
- **Stock examples** `examples/custom/in.custom.{cube,step}.{read,set}.restart`.
Results:
- B_unpatched and A_sync abort on all of the decks above (np1; and g1a/g1c/g3 at np2 with NX=71, np4 with NX=100).
  The DualView is `grid:dvector`; for the stock restart examples it is `grid:ivector`.
- Decks: trace `resize flags=(0,0)` then `modify_host flags=(0,1)`. Backtrace GridKokkos::sync <- reallocate_custom <-
  grow_cells <- Grid::add_child_cell <- Grid::refine_cell (both adapt_grid and fix adapt).
- Stock restart examples: backtrace GridKokkos::sync <- ComputeReduceKokkos::setup_values <- Stats <- Output::setup
  <- Run::command. The resize came from read_restart: Grid::grow_cells -> GridKokkos::reallocate_custom, prewrap
  branch (gdb backtrace).
negative control: `in.g0` (no custom), and the non-restart examples/custom and examples/adapt decks: identical to unpatched.
first patch (spa_B_fixed, auto_sync refresh only): the decks stop aborting and match cpu. **But the 4 stock restart examples
then gave WRONG grid custom values**: c_2[1] and c_2[4] (g_ivec, g_dvec averages) were 0, against cpu 4945.65 and 4949.11.
WATCH: `grid:ivector: the host side was written, never claimed, and is now lost -- the write is between resize and
sync_host`. The restart values were written to the host after a device-side resize, and the refresh then copied the
device's zeros over them. Without the patch, a release Kokkos (no abort check) would have pushed the host values down
correctly. So the refresh alone is wrong whenever the device claim comes from a resize and the host holds unclaimed writes.
second change (grid_kokkos.cpp, prewrap guard in sync() and modify()): under prewrap, the Host direction now keeps
CUSTOM_MASK bits instead of returning: `mask &= CUSTOM_MASK; if (!mask) return;`. Device under prewrap still errors.
The cell arrays are still skipped before wrap. reallocate_custom()/allocate_custom() prewrap branches now really claim
the host, the resize happens on the host and keeps its values, and restart writes land on the host claim.
complete: B_fixed2
- runs every g* deck at np1 (and g1a, g1c, g3 at np2 and np4) to completion; stats match cpu (np, c_gr sums).
- runs the 4 restart examples to completion: grid and surf custom columns equal cpu on all rows, and the step-0 rows equal cpu.
  The grid:{i,d}vector watch reports are gone.
- On g1a/g1c/g3 no watch reports. The stale reports beyond the controls are `grid:dvector <- GridKokkos::reallocate_custom`
  (follow-up), and `grid:{cells,cinfo} <- ComputePropertyGridKokkos::compute_per_grid_kokkos` (benign: it takes
  `view_device()` one line before its `sync(Device,...)`).
verdict: bug real, and reachable from stock examples. Fixed by the refresh plus the prewrap guard change.

## Surf -- UNEXERCISED (fixed on the precedent only)

Tried (B_unpatched and B_fixed2, `SPARTA_KOKKOS_TRACE=surf:`, looking for any SurfKokkos-managed view with flags (h,d>0)):
- `in.s1`: 2d circle, 2 custom surf vectors, `fix ave/surf` (non-Kokkos, auto_sync, datamask_read EMPTY) over compute surf/kk,
  `fix custom` surf set every 10 steps, then remove_surf + read_surf + custom surf set between runs, run again.
- examples/surf in.surf.{move,add,remove}, examples/ablation in.ablation.2d, and all examples/custom decks.
No device claim ever appeared on surf:lines/tris/mylines/mytris or surf custom vectors. The only device-claimed `surf:*`
labels are compute tally scratch (`surf:tally2surf`, `surf:array_surf_tally`), which SurfKokkos::sync does not manage.
Statically, no code calls SurfKokkos::modify(Device,..), and every surf resize claims the host first.
B_fixed2 output equals B_unpatched (excluding the CPU-time column) on all of these.
verdict: fixed for consistency (LAMMPS AtomKokkos::sync precedent); no reproducer exists.

## Stock-example regression (B_fixed2 vs B_unpatched, np1, stats excluding the CPU column)

examples/adapt (3), custom (28), ambi (3), surf (5), ablation (4), and deck s1: identical to B_unpatched on all of them,
except the 4 `*.restart` custom decks. B_unpatched aborts on those; B_fixed2 matches cpu, as above. in.ambi stops with
"Ran out of space in Kokkos collisions" on both binaries, so it is pre-existing and unrelated.

## Compile check (real tree)

`.kokkos-review/compile_one.sh src/KOKKOS/{particle,grid,surf}_kokkos.cpp`: OK (after both changes).

## Follow-ups found (outside my three files)

1. **Root of particle/grid: device-side resize of custom inner views.** `ParticleKokkos::grow_custom()` and
   `GridKokkos::reallocate_custom()` (non-prewrap) do `sync(Device,CUSTOM_MASK)` and then resize. That resizes on the
   device and leaves a device claim, and the host pointer is then re-taken from `view_host()`. On a real GPU,
   Kokkos rebuilds the host mirror uninitialized on a device-side resize, so host reads through `eivec/edvec` see
   garbage until the next `sync_host`. Host writes in that window are lost at the refresh (the detector reports
   the stale host reads). Recommended: resize on the host as `SurfKokkos::reallocate_custom()` does
   (`sync(Host)+modify(Host)` first), in particle_custom_kokkos.cpp and grid_custom_kokkos.cpp.
2. Restart examples: `[watch] surf:{iarray,darray}: the host side was written without a claim and this sync_device
   has nothing to copy` at `SurfKokkos::sync <- UpdateKokkos::move<3,1,0,0>` (auto_sync off, so not this patch).
   Surf custom *arrays* restored by read_restart are never claimed on the host. Results still match cpu here, but the device copy is stale.
3. VERIFY noise: with `SPARTA_KOKKOS_WATCH` and `SPARTA_KOKKOS_VERIFY` set, the patched binaries print
   `[verify] ... differ at byte 0 ... (at sync_host)` on clean decks. This comes from the extra `sync_host()` calls now
   made on in-sync views; the first one is on a freshly allocated, uninitialized view (flags (0,0), claims (0,0)).
   It is not a data change: B_unpatched and B_fixed stats are byte-identical on p0/g0/p1.

## STATUS: COMPLETE
