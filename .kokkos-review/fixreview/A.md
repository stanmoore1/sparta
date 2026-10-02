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
