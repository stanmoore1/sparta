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
