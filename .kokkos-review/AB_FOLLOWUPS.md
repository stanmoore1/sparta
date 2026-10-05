# Follow-up fixes found by A/B testing (to fix after A/B completes, then A/B again)
- FU-1 (from AB4, G12x-F-G16-2 INCOMPLETE): compute sonine/grid/kk and dt/grid/kk reallocate() never create host arrays used by the CPU fallback before first run (sonine: vcom; dt: tau,temp,usq,vsq,wsq) -> segfault with adapt_grid value c_X before first run (compute_sonine_grid.cpp:173, compute_dt_grid.cpp:460).
- FU-2 (from AB4, F-G16-9 sibling, CPU): src/compute_property_surf.cpp pack_v3y/pack_v3z (~427, ~444) read p1 instead of p3.
- FU-3 (from AB6, CPU+kk): subsonic pressure-only emit (face/surf) adjacent to a cell with zero thermal energy -> nrho inf/NaN -> int overflow error / huge allocation / OOM in both A and B, CPU and Kokkos. Root cause upstream of F-G11-4/F-G12-1 guards.
