# AB6 (emit) A/B results

### F-G00-1 — emit/surf/kk custom fractions: k_cummulative_custom never allocated (inverted realloc test), d_cummulative_custom never assigned
class: cpu-observable
positive control: 2d circle, emit/surf normal yes perspecies no twopass custom fractions s_fr (fr = 0.9/-1 -> 0.9 N, 0.1 O), 100 steps, compute count N O | A: kk t1 SEGFAULT (rc 139) before step 0; CPU ref step100 np 6686 N 6004 O 682 (N frac 0.898) | B: kk t1 np 6760 N 6143 O 617 (0.909), kk t4 np 6763 N 6052 O 711 (0.895); CPU A==B | REPRODUCED
negative control: same deck without custom fractions | A vs B: identical (kk: 6574/3282/3292 both; CPU 6552 both)
necessary: YES - A kk segfaults (rc 139) on every custom-fractions/perspecies-no run (2d t1, t4; 3d; per-surf split), CPU reference fine.
complete: B correct on: 2d constant 0.9/-1 (t1 N 0.909, t4 0.895 vs CPU 0.898); per-surf split via custom file (surfs 1-25 pure N, 26-50 pure O; dump after 5 steps: 0 particles of the wrong species on the wrong half, kk t1/t4 and CPU); 3d sphere (1200 tris) 0.9/-1 with 2 runs + adapt_grid between (re-runs grid_changed): kk B N frac 0.906/0.903 vs CPU 0.897/0.897; perspecies yes (not touched by fix): A==B identical. Not tested: multi-proc (ghost surfs / nslocal growth -> realloc branch) - MPI builds not finished. No sibling sites (only emit/surf has custom fractions).
verdict: NECESSARY, COMPLETENESS-PARTIAL (multi-proc realloc path untested: MPI build unavailable)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB6/F-G00-1

### F-G11-1 — emit/surf/kk: ncands==0 early return skipped post_surf_tally(), orphaning a surf-react particle-list reference per step
class: cpu-observable (memory leak)
positive control: 2d circle, surf_react prob + emit/surf from mixture with nrho 1e-30 (ncands==0 every step) + compute surf/fix ave/surf every step; 40x {create_particles 20000; run 2} so the particle array reallocates ~40 times; kk t1, max RSS via getrusage | A: maxRSS 820 MB (old particle arrays never freed) | B: 175 MB; final np 787357 in both, stats identical | REPRODUCED
negative control: same deck, emit mixture nrho 1.0 (ncands>0, normal post_surf_tally path) | A vs B: identical stats (np 791785), maxRSS 172 vs 173 MB
necessary: YES - A leaks one particle array per realloc while ncands==0 (maxRSS 820 MB vs 175 MB, identical physics).
complete: B flat RSS on: 1 compute t1 (175 MB), 2 surf tally computes (nsurf_tally=2) t1 113 MB vs A 489 MB, t2 (need_dup scatter path) 113 MB vs A 489 MB; stats identical A/B in all. The ncands>0 path already called post_surf_tally (neg control). Sibling: emit/face(/file)/kk have no surf-tally pre/post pairing; no other early return between pre_surf_tally and post_surf_tally in perform_task.
verdict: NECESSARY+COMPLETE
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB6/F-G11-1

### F-G11-2 — emit/surf (CPU+kk): custom-fraction cumulative row not forced to 1.0; roundoff (>=2 unset entries) lets a draw near 1.0 walk isp past nspecies
class: unreachable (in a run: ~1e-16 per draw) -> unit-tested
positive control: standalone unit replicating init unset-fill + grid_changed row (A vs B code) and the selection loop, rn = Kokkos XorShift64 drand max (=1.0) and rn = nextafter(last) | A: 0.3/-1/-1 -> last=0.99999999999999989, rn=1.0 -> isp=6 (valid 0..2, walks through next surf's row and off the end); 0.07+7x-1 -> last=1-2.2e-16, isp=16 (valid 0..7) | B: last=1.0 exactly, isp=2 / 7 | REPRODUCED (unit)
negative control: 2d circle, 3 species N O NO, custom fractions 0.3/-1/-1, emit/surf twopass, 100 steps | CPU A vs B: identical (np 6035; 1786/2106/2143); kk B np 6039 1784/2168/2087 (fractions 0.30/0.36/0.35, matches); kk A segfaults (F-G00-1, same path)
necessary: YES at unit level (A selects isp past nspecies for draws in (last,1.0], which Kokkos drand can return); not observable in a run (p~1e-16 per draw).
complete: B forces last=1.0 in the only builder (FixEmitSurf::grid_changed; Kokkos copies it). Sibling cumulative builders: Mixture::init_fraction already forces 1.0 (used by emit/face, emit/face/file per-task rows), so no remaining site.
verdict: NECESSARY+COMPLETE (unit-level necessity; run-level probability ~1e-16)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB6/F-G11-2 (unit.cpp, unit.out)

### F-G11-4 — emit/surf/kk subsonic PONLY: vstream correction not guarded by massrho*soundspeed > 0 (CPU has the guard) -> inf/NaN vstream when cell thermal KE <= 0
class: unreachable (no observable consumer)
positive control: 2d circle, emit/surf subsonic 1.38e-22 NULL (PONLY) twopass, cells pre-filled by create_particles from a mixture with temp 0 (all particles share vstream -> ke == 0 or roundoff +-), vstream 1/3.3/10, normal yes/no, 5 steps, CPU + kk t1, A and B | A == B in every variant (np, f_in, vx/vy min/max identical, e.g. v1.0 normal yes kk: np 4007 both); v3.3/v10 normal no: kk A AND B abort (ntarget >= 2^31 / 27 GiB alloc), CPU v10 also OOM/timeout in A and B | NOT REPRODUCED
necessary: NOT SHOWN. Whenever the guard fires (soundspeed_cell == 0 or NaN) the same task's nrho = nrho_cell + (p-press)/soundspeed^2 is inf/NaN in both CPU and Kokkos, so ntarget is NaN/inf: the task inserts nothing (int(NaN) < 0) or the run aborts, and the task vstream (reset from vcom every step) is never consumed. The guard is a parity/hygiene change with no observable effect.
complete: B has the guard identical to CPU fix_emit_surf.cpp:1533; sibling sites: emit/face/kk fixed by F-G12-1, emit/face/file/kk already guarded. Residual shared (CPU+KK, A and B) issue on the same zero-thermal-energy path: the unguarded nrho division -> inf/NaN ntarget -> CPU OOM/hang (v10 normal no) and KK abort (v3.3/v10 normal no); out of this fix's scope, reported for follow-up.
negative control: examples/emit/in.emit.surf.subsonic (300 steps, collisions, window 100) kk t1 | A vs B: identical stats (np 37957 at step 300); t4 runs of both exceeded the 120 s timeout under machine load (not compared)
verdict: NOT-SHOWN-NECESSARY (guarded value has no consumer when the guard fires; nrho already inf/NaN)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB6/F-G11-4

### F-G12-1 — emit/face/kk subsonic PONLY: vstream correction missing the CPU massrho*soundspeed > 0 guard
class: unreachable (no observable consumer)
positive control: 2d box, emit/face xlo subsonic 1.38e-22 NULL twopass, cells pre-filled from a temp-0 mixture (identical velocities -> ke == 0 / roundoff), vstream 1/3.3/10, 5 steps | v1/v10: CPU A, CPU B, kk A, kk B ALL stop with "subsonic insertion count exceeds 32-bit int" (nrho inf); v3.3: all four identical (np 3999, no insertion, vx range 3.3..3.3) | NOT REPRODUCED
necessary: NOT SHOWN. As for F-G11-4: when soundspeed_cell is 0/NaN, nrho = nrho_cell + (p-press)/soundspeed^2 is already inf/NaN in CPU and KK, so the task either errors (same error in CPU) or inserts nothing; the NaN vstream never reaches a particle.
complete: B guard identical to CPU fix_emit_face.cpp:1109; sibling sites emit/surf/kk (F-G11-4, fixed) and emit/face/file/kk (already guarded) - no remaining unguarded site.
negative control: single emit/face xlo subsonic 0.414 NULL, nrho 1e20 fnum 1e18, create_particles, 200 steps | A vs B: kk identical (np 8392, f_in 4892), CPU identical (8565/5129)
verdict: NOT-SHOWN-NECESSARY (degenerate-cell nrho already inf/NaN; guarded value unobservable)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB6/F-G12-1

### F-G11-3 — emit/surf/kk perform_task: &d_cummulative_mix[0] evaluated on the unallocated (0-extent) mix view whenever custom fractions + perspecies no
class: bounds-check
positive control: MPI/bounds-check SPARTA builds still compiling (23% under load 25-30), so tested the unit: standalone program built against build_base's Kokkos (OpenMP) with KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK, state as in init() (DualView mix never allocated, custom view allocated), kernel pointer selection copied from A and B | A custom: "Kokkos::View ERROR: out of bounds access ... indices [0] but extents [0]" abort | B custom: sum 1.5 (correct), no abort | REPRODUCED (unit)
necessary: YES (unit-level): A's line always indexes the empty view in the custom/perspecies-no case and aborts under bounds checking; release builds only form a null pointer (no wrong values). Full SPARTA A cannot isolate it because F-G00-1 crashes first on this path.
complete: B correct in both branches (custom: 1.5; mix allocated: 0.75 = A's value). Sibling sites: emit/face/kk and emit/face/file/kk always allocate their cumulative arrays (mixture or per-task); no other conditional-allocation + unconditional-index pattern found. Release-build B on the real path: F-G00-1 runs (2d/3d, t1/t4) correct. Bounds-checked full-SPARTA B run not done (build unfinished).
negative control: unit mix-mode (allocated mix view): A == B (0.75); full runs without custom fractions identical A vs B (see F-G00-1)
verdict: NECESSARY, COMPLETENESS-PARTIAL (bounds-checked full SPARTA B run pending the DEBUG_BOUNDS_CHECK build; unit + release runs pass)
artifacts: /tmp/claude-0/-home-user-sparta/890c9580-1a31-5f7e-91e6-8571cb0d6a4f/scratchpad/ab/AB6/F-G11-3 (unit.cpp, inc/ with bounds-check config)
