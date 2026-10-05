# AB9 results (infra/grid/surf)
S=/tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad ; artifacts under $S/ab/AB9/<ID>/

### F-G21-2 — "t" Kokkos arg had no bounds check (atoi(NULL)/next argv)
class: cpu-observable
positive control: `-in in.x -sf kk -k on t` (t last) | A: SIGSEGV rc=139 | B: "ERROR: Invalid Kokkos command-line args (kokkos.cpp:126)" rc=1 | REPRODUCED
positive control 2: `-k on t -sf kk -in in.x` | A: silently "requested 0 thread(s)" and runs | B: same clean error | REPRODUCED
negative control: `-k on t 2 -sf kk` 1000 particles 10 steps | A vs B: identical (2 threads, Np 1000, both run)
necessary: A segfaults (rc=139) with `-k on t` last, `-k on threads` last and `-k on t 1 t`; A silently takes atoi("-sf")=0 threads when t is followed by another switch.
complete: B errors cleanly on all 4 variants (t-last, threads-last, `t 1 t`, t followed by -sf). Sibling sites: d/g branches already had the check (d-last: A and B both error), all `package kokkos` args (kokkos.cpp:255-279) are bounds-checked; no remaining unchecked arg[iarg+1].
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB9/F-G21-2

### F-G21-1 — "-k on g 0" -> local_rank % ngpus division by zero
class: unreachable (GPU-build-only arg path; tested by instrumentation)
method: scratch copies of base and fixed kokkos.cpp with only the `#ifndef SPARTA_KOKKOS_GPU` error line blanked, compiled with build_base flags and relinked against build_base libs (spa_A/spa_B); kokkos.cpp is the only file touched by the fix.
positive control: OMPI_COMM_WORLD_LOCAL_RANK=1, `-k on g 0 t 1 -sf kk` | A: SIGFPE (rc=136) | B: "ERROR: Invalid Kokkos command-line args" rc=1 | REPRODUCED
negative control: OMPI_COMM_WORLD_LOCAL_RANK=0, `-k on g 1 t 1`, 1000 particles 10 steps | A vs B: identical (both run, 1 GPU requested, Np 1000). Also unmodified A_opt/B_opt both reject `g` in a non-GPU build with the same message.
note: `g -2` is never reached (sparta.cpp treats "-2" as a new switch -> "Invalid command-line argument" in both).
necessary: A SIGFPE (rc=136) with `g 0` under EACH of the 6 local-rank env vars (SLURM_LOCALID, FLUX_TASK_LOCAL_ID, MPT_LRANK, MV2_COMM_WORLD_LOCAL_RANK, OMPI_COMM_WORLD_LOCAL_RANK, PMI_LOCAL_RANK).
complete: B rejects g 0 with "Invalid Kokkos command-line args" for all 6 env vars and for the skip-gpu form `g 0 1` (no env); the check precedes all six `% ngpus` sites (no other modulo/division by ngpus in kokkos.cpp). Caveat: only reachable in a GPU build, tested via instrumented relink.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB9/F-G21-1

### F-G21-4 — kokkos_type.h used deprecated Kokkos::Experimental::{HIP,SYCL,HIPHostPinnedSpace,SYCLHostUSMSpace}
class: build-system (HIP/SYCL-only; no such toolchain here)
method: extracted every `#if/#elif defined(KOKKOS_ENABLE_HIP|SYCL)` block of A and B kokkos_type.h into a TU over a mock of Kokkos 5.2.2 decl/Kokkos_Declare_{HIP,SYCL}.hpp (Kokkos::HIP etc.; Experimental:: aliases only under KOKKOS_ENABLE_DEPRECATED_CODE_5); real classes confirmed at lib/kokkos/core/src/HIP/Kokkos_HIP.hpp:21, Kokkos_HIP_Space.hpp:117, SYCL/Kokkos_SYCL.hpp:30, Kokkos_SYCL_Space.hpp:103.
positive control: DEPRECATED_CODE_5 OFF | A: HIP and SYCL TUs fail, 11 errors each ("'HIP' is not a member of 'Kokkos::Experimental'") | B: both compile, 0 errors | REPRODUCED
negative control: DEPRECATED_CODE_5 ON (the default) | A compiles (4 deprecation warnings), B compiles with 0 warnings; OpenMP/CUDA branches unchanged (A_opt/B_opt both build)
necessary: A's HIP and SYCL blocks fail to compile against Kokkos 5.2.2 decls once DEPRECATED_CODE_5 is OFF (mock of the real decl headers; no HIP/SYCL toolchain here, so not a real-backend build).
complete: all 8 renamed sites (ExecutionSpaceFromDevice HIP/SYCL, SPAPinnedHostType HIP/SYCL, AtomicDup<+-1> HIP/SYCL) compile in B; grep of src/KOKKOS/*.{h,cpp} finds no other Experimental::HIP/SYCL use. OpenMPTarget Experimental names left (valid in Kokkos 5, out of scope).
verdict: NECESSARY+COMPLETE (mock-header level; real HIP/SYCL build untestable here)
artifacts: $S/ab/AB9/F-G21-4

### F-G21-5 — t_plevel_1d/t_host_plevel_1d typedef'd from tdual_pcell_1d (ParentCell) instead of tdual_plevel_1d
class: unreachable (latent typedef; no users) -> unit compile test with real kokkos_type.h
positive control: TU `t_plevel_1d d = tdual_plevel_1d(...).view_device(); t_host_plevel_1d h = ...view_host(); h(1).nx=7` built with A/B include flags + Kokkos libs | A: compile fails, 4 errors ("conversion from View<ParentLevel*> to View<ParentCell*>", "ParentCell has no member nx") | B: compiles, prints "extent 3 nx 7" | REPRODUCED
negative control: same TU using tdual_pcell_1d/t_pcell_1d/t_host_pcell_1d | A vs B: identical (both compile, "pcell extent 3 3")
necessary: A cannot compile any use of t_plevel_1d/t_host_plevel_1d with a plevel DualView (4 errors). No current users, so latent only.
complete: both typedefs (device and host) tested in B and compile/run; scripted scan of all 54 `typedef tdual_X::t_dev|t_host t_[host_]Y` lines: A has exactly 2 mismatches (the plevel pair), B has 0.
verdict: NECESSARY+COMPLETE (latent)
artifacts: $S/ab/AB9/F-G21-5

### F-G21-8 — unanchored ".*fft|pack|remap.*kokkos.*" CMake filters drop all KOKKOS files when the checkout path contains pack/fft/remap
class: build-system
method: `cmake -P` driver including the exact filter block extracted from A/B src/KOKKOS/CMakeLists.txt (only CONFIGURE_DEPENDS removed, invalid in script mode) and A/B cmake/common/set/style_file_glob.cmake (minus configure_file loop), run in mock trees with the real src/KOKKOS file names under .../packages/sparta/, .../fftstage/sparta/, .../remapper/x/, .../plain/sparta/.
positive control: PKG_FFT=OFF, "packages"/"fft"/"remap" paths | A: SRC_FILES 192->2 (only rand_pool_wrap.cpp/.h survive), style_files 100->1, grid_kokkos.cpp/.h dropped | B: SRC_FILES 177, style_files 90, grid_kokkos.* kept, identical to plain path | REPRODUCED
negative control: plain path PKG_FFT=OFF -> A and B remove the same 14 FFT-family files (B additionally kokkos_base_fft.h, see F-G21-9); PKG_FFT=ON all paths -> A and B identical (192 src / 100 headers)
necessary: A loses 190/192 sources and 99/100 headers for any path containing packages/fft/remap with PKG_FFT=OFF.
complete: B covers both sites (src/KOKKOS/CMakeLists.txt SRC list and style_file_glob.cmake header list), all three keywords (pack, fft, remap paths), and the result equals the plain-path result; removed set is exactly the 14 FFT-family files + kokkos_base_fft.h. Other style_file_glob users (src, FFT, VTK, PYTHON) have no such filters.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB9/F-G21-8 (cmake_results.txt, drv.cmake, dump.cmake)

### F-G21-9 — list(REMOVE_ITEM style_files kokkos_base_fft.h) never matched (absolute paths / wrong list)
class: build-system
method: same cmake -P harness as F-G21-8
positive control: plain path PKG_FFT=OFF | A: kokkos_base_fft.h kept in both SPARTA_PKG_KOKKOS_SRC_FILES (n=178) and style_files (n=91) | B: removed from both (n=177 / 90) | REPRODUCED
negative control: PKG_FFT=ON | A vs B: identical (kokkos_base_fft.h present in both lists, 192/100)
necessary: A keeps kokkos_base_fft.h in both lists with PKG_FFT=OFF.
complete: B removes it from both SPARTA_PKG_KOKKOS_SRC_FILES and style_files, in plain and "packages" paths.
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB9/F-G21-8

### F-G21-12 — KOKKOS/Install.sh uninstall used undefined $SED; `test $KOKKOS_INSTALLED = 1` missed counts >1
class: build-system
method: scratch src trees ($D/{A,B}_<case>/src) with each variant's Install.sh, mock Makefile.package/.settings, `env -u SED /bin/sh Install.sh 0` (as src/Makefile does)
positive control (case one: Makefile.package with -DSPARTA_KOKKOS/-I.../kokkos/-L.../kokkos/-lkokkos, settings with `CXX = $(CC)` + `include .../Makefile.kokkos`) | A: "Install.sh: 240/241/245/246: -i: not found", rc=127, both files unchanged | B: rc=0, all kokkos/KOKKOS tokens stripped (-lfft kept), CXX and kokkos include lines deleted, other include kept | REPRODUCED
positive control 2 (case two: 2 lines with DSPARTA_KOKKOS -> KOKKOS_INSTALLED=2) | A: accelerator_kokkos.h NOT touched (mtime stays 2020-01-01) + sed failures | B: touched + cleaned | REPRODUCED
negative control (case none: no kokkos entries) | A vs B: Makefile.package/.settings identical and unchanged, accelerator_kokkos.h untouched in both; action-file removal (fft2d_kokkos.cpp deleted, grid.cpp kept) identical in all cases (A only differs by its rc=127)
necessary: A leaves KOKKOS flags in legacy Makefile.package(.settings) on `make no-kokkos` and fails the >1-count touch.
complete: all 4 $SED sites + the :30 test fixed and exercised; grep finds no other $SED in src/ (no other Install.sh uses it). (Pre-existing regex limitation, not in scope: a kokkos token at end-of-line without trailing space is not stripped by either A or B.)
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB9/F-G21-12 (results.txt)

### F-G22-1 — GridKokkos::id_find_child lacked CPU id_point_child round-off edge correction/clamp
class: cpu-observable (at function level; end-to-end masked, see below)
positive control: 2d box 0..1, `create_grid 2 2 1 levels 2 subset 2 2 2 1 5 5 1`; single N2 particles moving +x / +y / -x(periodic) with the transverse coord exactly on child edges 0.6,0.7,0.8,0.9 (=0.5+j*0.5/5); gdb breakpoint after the index computation in id_find_child (both call sites NPARENT and NPBPARENT hit, 12 calls) | A: wrong child for 6/12 calls (y=0.6 -> iy=0 instead of 1, y=0.7 -> 1 instead of 2, same for x) | B: all 12 equal CPU id_point_child | REPRODUCED (function level)
positive control 3d/3 levels: `create_grid 2 2 2 levels 3 subset 2 2 2 2 5 5 5 subset 3 * * * 3 3 3`, 20 particles with y on level-2 edges and z on level-3 edges; gdb hits (40) checked against a Python emulation of CPU id_point_child recursion | A: 22/40 mismatches incl. level-3 recursion done with the wrong child's lo/hi (e.g. (0,2,1) vs (0,0,1)) | B: 0/40 mismatches | REPRODUCED
end-to-end: final particle cellIDs (dump id cellID x y [z], every step) for all three inputs: A_kk == B_kk == CPU (0 differing particle-steps). Reason: A always errs to child j-1 with the point exactly on that child's hi face, and the move loop's `xnew >= hi` test immediately does a zero-length face crossing into the correct child. Also tried particles ending the step exactly on the face (x0=0.45, vx*dt=0.05): still self-corrected.
negative control: (a) same inputs with transverse coords +0.03 off-edge: A/B/CPU identical cellIDs; (b) 3d 3-level random gas with VSS collisions, 20000 particles, 100 steps, t1: A_kk and B_kk thermo identical (np, nattempt, ncoll, max/ave cell counts)
necessary: shown at function level (A returns a different child than CPU for exactly-on-edge points, at both NPARENT/NPBPARENT call sites, both levels); NOT shown end-to-end — no observable change in cell assignment, counts or thermo found on CPU/OpenMP 1 proc (only an extra zero-length crossing; could matter when the wrong child is a ghost/unknown on another proc - MPI not tested).
complete: B matches CPU at every call/level tested (2d x/y, 3d y/z, 2 and 3 levels, periodic NPBPARENT path); no other inverse-index child lookup in src/KOKKOS (only id_find_child has this formula).
verdict: NECESSARY, COMPLETENESS-PARTIAL (function-level necessity only, end-to-end masked by self-correction; MPI/ghost-child variant untested)
artifacts: $S/ab/AB9/F-G22-1 (in.pos*, gdbA/gdbB/g3A/g3B hits, emu3d.py, cmp.py)

### F-G22-5 — fix grid/check/kk built error messages from stale host particles/cells
class: gpu-only
positive control: none possible here — on this OpenMP build the particle and cell DualViews alias (tiny compile test against kokkos_type.h: host ptr == device ptr for tdual_particle_1d and tdual_cell_1d), so host data can never be stale. Also could not provoke any grid/check problem on CPU (create_particles single only into OUTSIDE cells; read_surf `particle keep` errors "Particles are inside new surfaces"; transparent circle + emit/face 100 steps -> no flagged particle in A or B)
negative control: transparent-circle flow, `fix grid/check 1 error`, 100 steps, CPU and kk t1 | A vs B: identical (no error, Np 84334 cpu / 84409 kk in both)
necessary: NOT shown (gpu-only; staleness impossible with aliased host/device views).
complete: by inspection the single sync(Host, PARTICLE_MASK|CELL_MASK) precedes all 5 error-message branches (invalid/outside/split/interior/zero-volume), so all are covered; untested at runtime.
verdict: NOT-SHOWN-NECESSARY (gpu-only: needs separate host/device memory)
artifacts: $S/ab/AB9/F-G22-5

### F-G18-2-cpu — compute gas/reaction/grid (CPU) kept columns/reaction2col sized for the react command at definition; re-issued/removed react -> OOB
class: cpu-observable
setup: 3d 4x4x4 cells, 5-species air at 40000 K, collide vss, `react tce small.tce` (2 reactions), compute g gas/reaction/grid every|select 1 2, plus compute gall ... all as reference; run 30; then `react tce air.tce` (45 reactions) or `react none`; run 30; dump grid c_gall c_g[*]
positive control (CPU): every | A: runs silently, 4 cells at step 60 where sum(c_g columns) != c_gall (reactions >2 written into neighbouring cells' rows) | B: "ERROR: Compute gas/reaction/grid reactions changed since compute was defined" | REPRODUCED
positive control (CPU): select 1 2 | A: SIGSEGV (reaction2col OOB) | B: same clean error | REPRODUCED
react none (CPU, every) | A: runs (stale compute, all zeros) | B: clean error (intended behaviour change; matches Kokkos)
Kokkos (with `package kokkos react/retry yes`): every A SIGSEGV, select A SIGSEGV, none A abort "SharedAllocationRecord failed increment" | B: clean error for all 3 (Kokkos side fixed in 605d2aee, init delegates to the same check)
negative control: (a) react unchanged between runs (in.neg) and (b) react re-issued with the SAME 45-reaction file (in.negsame) | CPU A vs B: identical stats and grid dumps; per-cell sum(cols)==all in every dump. kk negsame: both run, per-cell consistent; A/B differ statistically only (B carries other collide/react fixes; step-30 counts 4 vs 1)
necessary: A corrupts per-cell counts (every) or segfaults (select) on CPU, and crashes on Kokkos.
complete: B errors for every, select, and react none, on CPU and Kokkos; mode all (no columns) unaffected and still allowed; re-issuing a same-size react is still allowed (negsame runs). Note the check compares only nlist, so a re-issued react with the same count but a different reaction order is still accepted (columns then mean different reactions; not OOB).
verdict: NECESSARY+COMPLETE
artifacts: $S/ab/AB9/F-G18-2-cpu

### F-G22-3 — SurfKokkos::grow (non-prewrap) did not zero lines/tris [old,nmax) when nmax unchanged (CPU Surf::grow memsets)
class: cpu-observable (unit level); end-to-end needs ghost surfs (MPI)
method: driver linked against A/B libs (unit/drv.cpp): read surfs, `run 0` (prewrap=0 so the Kokkos path is used), fill tail slots [nlocal,nmax) with stale copies (id=1000+i, as remove_ghosts() leaves ghost lines there), call surf->grow(nlocal) with nmax unchanged, exactly as Surf::add_surfs does
positive control: 2d circle (50 lines, nmax 1024) and 3d cube (12 tris) | A Kokkos: 974 / 1012 stale nonzero-ID slots remain | B Kokkos: 0 / 0 | REPRODUCED. CPU Surf::grow (same driver without -k): 0 in A and B (reference)
1-proc end-to-end (read_surf circle, then read_surf triangle with duplicate line ID 1,1,3): "Missing read_surf IDs = 1" in A and B, CPU and kk — no ghosts on 1 proc so tail already zero; stale-ghost variant needs MPI (see below if run)
negative control: valid second surface (IDs 1,2,3), emit/face flow 100 steps | A vs B: identical (cpu: Np 19147 nscoll 134; kk: Np 19181 nscoll 119)
necessary: shown at unit level (A keeps stale lines/tris in new slots, defeating the "Missing read_surf IDs" check); end-to-end not shown on 1 proc.
complete: both branches (2d lines, 3d tris) fixed and tested; live slots [0,old) untouched. Sibling SurfKokkos::grow_own (mylines/mytris) has no memset either, but it resizes exactly nown_old->nown so Kokkos resize value-initializes the new tail; only an extent>nown_old state (not reachable via add_surfs, clear_explicit resets) would leave stale data - not tested.
verdict: NECESSARY, COMPLETENESS-PARTIAL (unit-level; MPI ghost end-to-end pending/untested)
artifacts: $S/ab/AB9/F-G22-3 (drv.cpp, results.txt, *.log)

### F-G22-2 — GridKokkos grow_cells/grow_sinfo resized k_cells/k_cinfo/k_sinfo WithoutInitializing (CPU memsets new tail)
class: unreachable (hygiene; verify found no reader of the uninitialized fields) -> unit test
method: driver (#define protected public) linked to A/B libs; 2d 20x20 grid, `run 0` (prewrap=0), then force grow_cells(maxcell,maxlocal) and grow_sinfo twice; count nonzero bytes in the new tail; glibc perturb (GLIBC_TUNABLES=glibc.malloc.perturb=171, mmap_threshold 32MB) to expose uninitialized memory
positive control (resize branch) | A Kokkos: cells tail 983040 nonzero bytes, cinfo 524288, sinfo resize 393216 | B Kokkos: 0 / 0 / 0 | CPU (A and B): 0 | REPRODUCED (memory-content level)
sibling not covered: first allocation branch (`cells/cinfo/sinfo == NULL` -> MemKK::realloc_kokkos, NoInit) | B Kokkos: sinfo first alloc 0->8192 still 393216 nonzero bytes with perturb and 4000 nonzero bytes even with default malloc (A identical); CPU 0. Fix only touched the Kokkos::resize branch.
negative control: CPU path (no -k) identical A vs B (all zero); behaviour: no reader of the garbage exists per verify, so no run-level difference expected (grid examples covered by other clusters)
necessary: NOT shown behaviourally (no consumer of uninitialized fields known); shown only as memory-content divergence from CPU.
complete: resize branches of k_cells, k_cinfo, k_sinfo all now zeroed; the initial-allocation (realloc_kokkos) branch of the same three functions still returns uninitialized memory (sinfo demonstrated), so CPU-parity is not complete.
verdict: INCOMPLETE (first-allocation realloc_kokkos branch still uninitialized: sinfo 0->8192 nonzero bytes in B) — and NOT-SHOWN-NECESSARY behaviourally (hygiene only)
artifacts: $S/ab/AB9/F-G22-2 (drv.cpp, results.txt)

