# AB9 results (infra/grid/surf)
S=/tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad ; artifacts under $S/ab/AB9/<ID>/

### F-G21-2 — "t" Kokkos arg had no bounds check (atoi(NULL)/next argv)
class: cpu-observable
positive control: `-in in.x -sf kk -k on t` (t last) | A: SIGSEGV rc=139 | B: "ERROR: Invalid Kokkos command-line args (kokkos.cpp:126)" rc=1 | REPRODUCED
positive control 2: `-k on t -sf kk -in in.x` | A: silently "requested 0 thread(s)" and runs | B: same clean error | REPRODUCED
negative control: `-k on t 2 -sf kk` 1000 particles 10 steps | A vs B: identical (2 threads, Np 1000, both run)
verdict: VERIFIED
artifacts: $S/ab/AB9/F-G21-2

### F-G21-1 — "-k on g 0" -> local_rank % ngpus division by zero
class: unreachable (GPU-build-only arg path; tested by instrumentation)
method: scratch copies of base and fixed kokkos.cpp with only the `#ifndef SPARTA_KOKKOS_GPU` error line blanked, compiled with build_base flags and relinked against build_base libs (spa_A/spa_B); kokkos.cpp is the only file touched by the fix.
positive control: OMPI_COMM_WORLD_LOCAL_RANK=1, `-k on g 0 t 1 -sf kk` | A: SIGFPE (rc=136) | B: "ERROR: Invalid Kokkos command-line args" rc=1 | REPRODUCED
negative control: OMPI_COMM_WORLD_LOCAL_RANK=0, `-k on g 1 t 1`, 1000 particles 10 steps | A vs B: identical (both run, 1 GPU requested, Np 1000). Also unmodified A_opt/B_opt both reject `g` in a non-GPU build with the same message.
note: `g -2` is never reached (sparta.cpp treats "-2" as a new switch -> "Invalid command-line argument" in both).
verdict: VERIFIED
artifacts: $S/ab/AB9/F-G21-1

### F-G21-4 — kokkos_type.h used deprecated Kokkos::Experimental::{HIP,SYCL,HIPHostPinnedSpace,SYCLHostUSMSpace}
class: build-system (HIP/SYCL-only; no such toolchain here)
method: extracted every `#if/#elif defined(KOKKOS_ENABLE_HIP|SYCL)` block of A and B kokkos_type.h into a TU over a mock of Kokkos 5.2.2 decl/Kokkos_Declare_{HIP,SYCL}.hpp (Kokkos::HIP etc.; Experimental:: aliases only under KOKKOS_ENABLE_DEPRECATED_CODE_5); real classes confirmed at lib/kokkos/core/src/HIP/Kokkos_HIP.hpp:21, Kokkos_HIP_Space.hpp:117, SYCL/Kokkos_SYCL.hpp:30, Kokkos_SYCL_Space.hpp:103.
positive control: DEPRECATED_CODE_5 OFF | A: HIP and SYCL TUs fail, 11 errors each ("'HIP' is not a member of 'Kokkos::Experimental'") | B: both compile, 0 errors | REPRODUCED
negative control: DEPRECATED_CODE_5 ON (the default) | A compiles (4 deprecation warnings), B compiles with 0 warnings; OpenMP/CUDA branches unchanged (A_opt/B_opt both build)
verdict: VERIFIED
artifacts: $S/ab/AB9/F-G21-4

### F-G21-5 — t_plevel_1d/t_host_plevel_1d typedef'd from tdual_pcell_1d (ParentCell) instead of tdual_plevel_1d
class: unreachable (latent typedef; no users) -> unit compile test with real kokkos_type.h
positive control: TU `t_plevel_1d d = tdual_plevel_1d(...).view_device(); t_host_plevel_1d h = ...view_host(); h(1).nx=7` built with A/B include flags + Kokkos libs | A: compile fails, 4 errors ("conversion from View<ParentLevel*> to View<ParentCell*>", "ParentCell has no member nx") | B: compiles, prints "extent 3 nx 7" | REPRODUCED
negative control: same TU using tdual_pcell_1d/t_pcell_1d/t_host_pcell_1d | A vs B: identical (both compile, "pcell extent 3 3")
verdict: VERIFIED
artifacts: $S/ab/AB9/F-G21-5

