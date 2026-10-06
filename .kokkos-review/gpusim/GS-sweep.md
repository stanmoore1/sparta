# GS-sweep: every example on B_sync (39e1c1f7 + tool) with the coherence detectors

Setup: S=$S (session scratchpad). Work dir $S/gpusim/GS-sweep (scripts in bin/, outputs in out/<variant>.np<N>/).
- Inputs copied to work/_tmpl/<ex>, `run N` shortened by bin/shorten.py (target ~0.3 s stock loop time, >=20 steps,
  >= one period of periodic fixes when affordable, rounded to the stats period; plans in out/plan.txt.<ex>).
  torque: averaging/stats periods 3000->100, run 200; surf_react_heatflux: periods 1000->50, run 150; cylinder: run 10.
- Variants (np1): stock = $S/spa_C4_opt -k on t 1 -sf kk (39e1c1f7, OpenMP, MPI stubs);
  Bw = B_sync + WATCH= STALE= STALE_STRICT=1; Ba = B_sync + AUDIT=1; Aw/Aa = same on A_sync (e071055f);
  Bp/Ap = poison builds (when present).
- Stats compare: thermo rows (CPU column dropped) of stock vs Bw vs Ba.

## Per-example results

(Progress notes, written while the np1 sweep runs; the per-example table follows at the end.)

### Report classes seen so far on B (np1), with first analysis
- R1 `[stale] grid:cells / grid:pcells / grid:sinfo device read while host newer, from UpdateKokkos::move<2,1,0,0>`
  (adapt.*). Same report on A_sync (adapt.static, same counts 8/9) -> pre-existing. Cause: update_kokkos.cpp:674-676
  take `d_cells/d_sinfo/d_pcells = grid_kk->k_*.view_device()` BEFORE `grid_kk->sync(Device,CELL_MASK|PCELL_MASK|SINFO_MASK|PLEVEL_MASK)`
  at :717. The handle is the same allocation after the sync (sync never reallocates), so the kernel reads fresh data:
  accessor-order false positive, not a GPU fault. Cosmetic fix: move the grid_kk->sync (and particle_kk->sync) above the
  view_device() grabs at :671-676.
- R2 `[stale] grid:cinfo device read while host newer, from ComputePropertyGridKokkos::compute_per_grid_kokkos()`
  (ablation.multi.inner.3d): compute_property_grid_kokkos.cpp:111-113 same handle-before-sync order; benign. Cosmetic fix: sync first.
- R3 `[stale] update:tally_models` (UpdateKokkos::setup_surf_tally_copies), `surf_react:models` (SurfCollideDiffuseKokkos::pre_collide),
  `collide:gas_tally_models` (CollideVSSKokkos::setup_gas_tally): blit to view_host() without modify_host, then
  `k.view_device().extent(0)` in *_buf_sync()/resize (kokkos_type.h:866-886, update_kokkos.cpp:86-108) runs while the pair
  differs with nothing owed (STALE_STRICT). Only metadata is read; the sync that follows copies everything. Benign.
  Cosmetic fix: call k.modify_host() inside *_buf_blit(), or test k.extent(0) instead of k.view_device().extent(0).
- R4 `[stale] ave/grid:tally, ave/grid:array_grid from FixAveGridKokkos::grow_percell / ~FixAveGridKokkos`,
  `surf:dvector from SurfKokkos::remove_custom`, `surf:tally2surf/array_surf_tally from ~ComputeSurfKokkos`,
  `particle:particles from ParticleKokkos::grow`: memory_kokkos.h grow_kokkos()/destroy_kokkos() take
  `data.view_host()(i,0)` addresses / `view_device().data()` right after a DualView resize (device marked newer), and
  the caller syncs straight after (fix_ave_grid_kokkos.cpp:641-654). Pointer arithmetic only; benign.
- R5 `[watch] react/isurf/grid:tally2surf & array_surf_tally: host side written, never claimed, now lost (between
  sync_host and modify_device)` (ablation.3d.reactions): ComputeReactISurfGridKokkos::tallyinfo()
  (compute_react_isurf_grid_kokkos.cpp:150-187) compacts the host copy in place without a claim; the next step's
  post_surf_tally() modify_device() discards it. The device copy is re-zeroed by clear() each step and the compacted
  host copy is a per-step product consumed on the host, so nothing is lost that is needed: benign by design.
  Annotation fix: after the compaction loop, `k_tally2surf.clear_sync_state(); k_array_surf_tally.clear_sync_state();`
  (or modify_host() followed by nothing else) to document the deliberate divergence.
- R6 audit `<style> starts with <array> stale on the device, not covered by datamask_read` for emit/face (start_of_step,
  reset_grid_count, copy_grid_one), grid/check (end_of_step), ambipolar (update_custom): all three declare
  EMPTY_MASK read and sync what they use themselves (fix_emit_face_kokkos.cpp:325,591,595,678,689;
  fix_grid_check_kokkos.cpp:70,73; fix_ambipolar_kokkos.cpp:96,117 - update_custom is a host routine). Informational.
- R7 audit `end_of_step ablate declares every array in datamask_modify` (FixAblate keeps Fix's ALL_MASK): uncheckable by the audit, not a fault.
