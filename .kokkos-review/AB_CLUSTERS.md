# A/B clusters (IDs from FIXES.md; details in verify/*.md, fixes/*.md, fixreview/*.md)
AB1 collide/react: F-G21V-1 (react/extra inversion), F-G01-1, F-G01-2 (incl CPU collide.cpp), F-G01-3, F-G01-4/F-G02-1, F-G02-2, F-G00-13, F-G00-15, F-G00-17, F-G04-1, F-G04-2 (+R-A-1/R-A-2)
AB2 surf_collide/update/particle: F-G00-10/F-G09-1, F-G08-1, F-G08-2 (piston CPU+kk), F-G08-3 (vanish/transparent CPU+kk), F-G09-3, F-G09-4 (a: surf_collide files, b: update_kokkos), F-G18-1, F-G05-1, F-G00-11b, F-G06-1, R-A-4, F-G13-4
AB3 surf_react: F-G10-1, F-G10-2, F-G10-3, F-G10-4, F-G10-5, F-G10-6, F-G00-18
AB4 grid/misc computes: F-G15-1, F-G00-5, F-G00-6, F-G00-7, F-G00-2, F-G00-9, F-G00-14, G12x-F-G16-2, F-G16-1 (CPU+kk), F-G16-4 (CPU+kk), F-G16-5, F-G16-6, F-G16-7, F-G16-8, F-G16-9 (CPU)
AB5 surf tally computes/temp rescale: F-G17-1 (CPU+kk), F-G17-2, F-G17-3, F-G17-4, F-G17-5 (CPU), F-G17-6, F-G00-19, F-G18-2 (kk + CPU), F-G00-16, F-G18-3, F-G00-16-note (n_current==0, CPU+kk)
AB6 emit: F-G00-1, F-G11-1, F-G11-2 (CPU), F-G11-3, F-G11-4, F-G12-1, F-G12-2
AB7 fix ave: F-G00-3, F-G00-4, F-G00-12, F-G00-20, F-G14-1, F-G14-5, F-G14-6, G12x-F-G14-1
AB8 fft: F-G20-9, F-G00-8/F-G19-2, F-G19-1, F-G19-3, F-G19-9, F-G20-6, F-G20-7, F-G00-21/F-G20-1, F-G19-4 (CPU+kk), F-G19-5/F-G20-2, F-G19-6/F-G20-3, F-G19-7, F-G20-8, F-G20-5, R-C-3
AB9 infra/grid/surf: F-G21-1, F-G21-2, F-G21-4, F-G21-5, F-G21-7, F-G21-8, F-G21-9, F-G21-12, F-G22-1, F-G22-2, F-G22-3, F-G22-4 (CPU), F-G22-5, F-G18-2-cpu
AB10 follow-ups (A = B_opt = $S/spa_new_final [before follow-ups], C = $S/spa_C_opt [with follow-ups]; for the ORIGINAL bug also compare to A_opt): FU-1, FU-2, FU-3, FU-4, FU-5, FU-6 (perf: re-measure in.one2 react/retry slowdown from ab/AB1), FU-8 (deck $S/ab/AB2/F-G09-4/in.sr_kk), FU-9, FU-10
AB10 add: FU-13 (needs a binary built after its commit: spa_C2_opt / C_mpi)
AB11 final follow-ups (A = $S/spa_C_opt [before FU-13/FU-3b/FU-18], C = $S/spa_C3_opt [HEAD]; A0 = build_base): FU-3b (relative cold-cell test; decks $S/ab/AB10/FU-3), FU-13 (react/isurf/grid grid group; deck $S/ab/AB5/sibling_gridgroup), FU-18 (CPU dt/grid post-processed input; deck $S/ab/AB10/FU-4/dtpp). Also: full examples t1 C vs final = 141/141 IDENTICAL (before FU-3b/FU-13/FU-18).
