# FX-emitsurf checkpoint

- F-G00-1 | src/KOKKOS/fix_emit_surf_kokkos.cpp | grid_changed: realloc k_cummulative_custom when extent(0) < max_cummulative (was inverted); assign d_cummulative_custom = k_cummulative_custom.view_device() after modify_host | compile OK
- F-G11-1 | src/KOKKOS/fix_emit_surf_kokkos.cpp | perform_task: on ncands==0 early return, call post_surf_tally() on all slist_active computes first (releases refs taken by pre_surf_tally) | compile OK
- F-G11-2 | src/fix_emit_surf.cpp | grid_changed: force cummulative_custom[isurf][nspecies-1] = 1.0 after each row's running sum (mirrors mixture.cpp); Kokkos copies this array in its grid_changed | compile OK
- F-G11-3 | src/KOKKOS/fix_emit_surf_kokkos.cpp | perform_task kernel: `cummulative = fractions_custom_flag ? &d_cummulative_custom(isurf,0) : d_cummulative_mix.data();` so the possibly-unallocated mix view is never indexed | compile OK
- F-G11-4 | src/KOKKOS/fix_emit_surf_kokkos.cpp | subsonic_grid kernel: guard vstream correction with `if (np && massrho_cell*soundspeed_cell > 0.0)` matching CPU fix_emit_surf.cpp | compile OK
- F-G11-5 | DEFERRED | per orchestrator instruction: enabling the Kokkos create_local changes particle placement for create_particles/kk (regression baselines), needs maintainer decision. Proposed patch: src/create_particles.h:65-66 -> `virtual void create_local();` `virtual void create_local_twopass();`; src/KOKKOS/create_particles_kokkos.h:33-34 -> `void create_local() override;` `void create_local_twopass() override { create_local(); }`; src/KOKKOS/create_particles_kokkos.cpp:52 -> `void CreateParticlesKokkos::create_local()` (drop bigint param so `np` is the member, used by the nglobal-nprevious != np warning at create_particles.cpp:439). Requires a test run.

## STATUS: COMPLETE
