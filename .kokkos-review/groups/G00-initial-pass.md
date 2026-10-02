# G00 — findings from initial single-pass review (not line-by-line); to be verified
## Progress
(n/a — seed findings)
## Findings
- [F-G00-1] src/KOKKOS/fix_emit_surf_kokkos.cpp:236 | high | grid_changed() inverted realloc test for k_cummulative_custom; d_cummulative_custom never assigned from k_cummulative_custom but read in kernel line ~732
- [F-G00-2] src/KOKKOS/compute_ke_particle_kokkos.cpp:99 | high | device operator() reads update->mvv2e (host pointer deref on GPU)
- [F-G00-3] src/KOKKOS/fix_ave_histo_kokkos.cpp:725 | high | kernel uses member grid_kk never set (shadowed by local in bin_grid_cells() line 635); same in fix_ave_histo_weight_kokkos.cpp:536/451; also host pointer on device
- [F-G00-4] src/KOKKOS/fix_ave_histo_kokkos.cpp:203 | high | end_of_step calls bin_one() on host for global inputs; bin_one does atomic_add on device views d_bin/d_stats
- [F-G00-5] src/KOKKOS/compute_grid_kokkos.cpp:232 | high | sorted kernel 'if (igroup<0) return;' should be continue; same compute_pflux_grid_kokkos.cpp:214
- [F-G00-6] src/KOKKOS/compute_tvib_grid_kokkos.cpp:353 | high | post_process parallel_for shares scratch views d_tspecies/d_tspecies_mode across threads (race); modeflag==2 branch line ~432 indexes d_groupspecies(index,...) instead of index/maxmode (CPU fixed in 456f0be9)
- [F-G00-7] src/KOKKOS/compute_sonine_grid_kokkos.cpp:136 | med | unsorted path normalizes d_vcom before contribute(d_vcom, dup_vcom_tally) when need_dup
- [F-G00-8] src/KOKKOS/compute_fft_grid_kokkos.cpp:195 | high | local 'auto d_ingrid' shadows member in compute-array/fix-array branches (195,225); variable branch fills only k_ingrid; exchange uses stale member d_ingrid
- [F-G00-9] src/KOKKOS/compute_property_surf_kokkos.cpp:79 | med | copies only first nchoose entries of cglobal / kernel over nchoose w/o group mask; base class now nsown-sized cglobal (9bf5ab59)
- [F-G00-10] src/KOKKOS/surf_collide_cll_kokkos.cpp:380 | med | backup() for cll/td/impulsive/adiabatic missing pre_update_custom_kokkos + re-copy of fix_ambi/vibmode copies (unlike diffuse/specular/piston)
- [F-G00-11] src/KOKKOS/update_kokkos.cpp:812 | med | move() retry does not roll back per-step surf/boundary tally computes; non-SURFACE boundary branch (~2108) never resets jpart
- [F-G00-12] src/KOKKOS/fix_ave_histo_weight_kokkos.cpp:419 | med | regionflag branches have parallel_reduce commented out -> no binning with region
- [F-G00-13] src/KOKKOS/collide_vss_kokkos.cpp:3485 | med | collisions_one_ambipolar disables recombination third body when np<=2; CPU (collide.cpp:1552) still picks one when np==2
- [F-G00-14] src/KOKKOS/compute_reduce_kokkos.cpp:481 | med | per-grid FIX branch reads fkk d_vector_grid/d_array_grid without sync_pergrid_device_kokkos()
- [F-G00-15] src/KOKKOS/collide_vss_kokkos.cpp:5139 | med | backup/restore don't save fix vibmode per-particle mode levels
- [F-G00-16] src/KOKKOS/fix_temp_rescale_kokkos.cpp:140 | low | missing CPU guard t_current<=0 -> vscale=1 (NaN)
- [F-G00-17] src/KOKKOS/collide_vss_kokkos.cpp:3843 | low | test_collision_kokkos missing CPU vremax==0 guard
- [F-G00-18] src/KOKKOS/surf_react_adsorb_kokkos.h:536 | low | adsorb SPECULAR post-reaction scatter ignores noslip
- [F-G00-19] src/KOKKOS/compute_isurf_grid_kokkos.cpp:194 | low | reads h_surf2tally[-1] when rank has no surfs; same compute_react_isurf_grid_kokkos.cpp:172
- [F-G00-20] src/KOKKOS/fix_ave_grid_kokkos.cpp | low | CUSTOM branch reads etype[-1], wrong dual view for INT (currently unreachable)
- [F-G00-21] src/KOKKOS/fft*_kokkos | low | norm_functor stores int norm (unused path)
