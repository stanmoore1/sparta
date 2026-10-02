# Fix-diff review A
DONE src/KOKKOS/collide_vss_kokkos.h
DONE src/KOKKOS/collide_vss_kokkos.cpp
DONE src/collide.cpp
DONE src/react_bird.cpp
DONE src/KOKKOS/react_bird_kokkos.{h,cpp}, react_tce_kokkos.h
- [R-A-1] src/KOKKOS/collide_vss_kokkos.cpp:599 / react_bird_kokkos.cpp:197 | low (perf) | check_prob_warn() does a blocking device->host deep_copy of d_prob_warn every collide step for the whole run when no warning ever fires | fold the flag into an existing per-step h_scalars read-back, or only check every N steps / at end of run
- [R-A-2] src/KOKKOS/react_tce_kokkos.h:278 | info | unsynchronized concurrent writes of 1 vs 2 to d_prob_warn; which warning is printed is nondeterministic when both cases occur (benign, CPU prints whichever comes first) | optional: atomic_max or bit-or (1|2) and print both
DONE src/KOKKOS/update_kokkos.{h,cpp}
DONE src/KOKKOS/particle_kokkos.{h,cpp}
DONE src/KOKKOS/particle_custom_kokkos.cpp
- [R-A-3] src/KOKKOS/update_kokkos.cpp:562 | info | use_reduce = need_atomics && !atomic_reduction mirrors the #if dispatch only because kokkos.cpp ties atomic_reduction to SPARTA_KOKKOS_REDUCE_ARCH and forbids ngpus==0 on GPU builds; correct in every reachable config but derived indirectly | optional: compute use_reduce with the same #if ladder as the dispatch (GPU: SPARTA_KOKKOS_REDUCE_ARCH; else need_atomics && !Serial)
DONE src/KOKKOS/surf_collide_*_kokkos.{h,cpp}
DONE src/surf_collide_{piston,transparent,vanish}.cpp
- [R-A-4] src/KOKKOS/surf_collide_piston_kokkos.h:210,221 (with update_kokkos.cpp:1239-1307) | high | the new outside-box discard marks the orphaned reaction product PDISCARD and returns jp=NULL, so UpdateKokkos never adds it to the migrate list; the product sits at index >= pstop, and because surf->nsr && pstop < nlocal the move loop runs another pass over [pstop,nlocal). TagUpdateMove has no PDISCARD branch (only PDONE/PKEEP/PINSERT/PENTRY/PEXIT/>=PSURF), so dtremain/xnew stay uninitialized and the particle is advected with garbage (UB: wrong cell, spurious "sent to self" error, out-of-range cell index). Before the fix the orphan had a defined PKEEP flag. The same latent defect already exists in the pre-existing ambipolar jp-delete path (d_particles[j_orig].flag = PDISCARD) of every surf_collide_*_kokkos model, so the CPU-parity nlocal-- has no working device equivalent | in TagUpdateMove, right after the PDONE block add `if (pflag == PDISCARD) { int indx = (ATOMIC_REDUCTION==0) ? d_nmigrate()++ : Kokkos::atomic_fetch_add(&d_nmigrate(),1); k_mlist.view_device()[indx] = i; return; }` (migration deletes PDISCARD, see the comment at update_kokkos.cpp:1345), so every model's PDISCARD product is removed and never advected; also prefer `jp->flag = PDISCARD` over d_particles[jp - d_particles.data()] in piston
- [R-A-5] src/KOKKOS/surf_collide_specular_kokkos.h:239 | info | wrapper_kokkos() has no callers in src/KOKKOS, so the noslip change cannot be exercised or tested; it does not persist flags[0] into noslip_flag the way CPU wrapper() does (that is a CPU side effect, acceptable on device) | none required
