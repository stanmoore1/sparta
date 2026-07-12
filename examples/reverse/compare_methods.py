#!/usr/bin/env python3
"""Cell-temperature-FREE (this implementation) vs cell-temperature (traditional)
reverse-rate methods.

The reverse reaction rate coefficient depends on the collision-energy
distribution of the reacting pair.  Two ways to obtain it:

  * cell-temperature (traditional): compute one temperature T_cell from the
    cell, then k_rev = k_fwd(T_cell)/Keq(T_cell) with an analytic Keq.  A DSMC
    cell temperature is a translational temperature, so this model is blind to
    how the internal (rot/vib/elec) energy is distributed.

  * cell-temperature-free (this branch): a per-collision reverse probability
    built from the microcanonical density of states, evaluated on the actual
    collision energy of each pair.  No cell temperature is formed.

At equilibrium the two agree (checks 1-3 of validate_reverse.py already show
the measured detailed-balance ratio equals the analytic Keq).  This script
demonstrates the DIFFERENCE out of equilibrium: a two-temperature reservoir
(fixed T_tr, swept T_int) where the true reverse rate swings by >10x while a
translational cell-temperature model cannot move at all.

Run:  python3 compare_methods.py --exe ../../src/spa_serial
"""
import argparse, math, os, sys
import numpy as np
import validate_reverse as V

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--ns", type=int, default=4_000_000)
    a = ap.parse_args()
    exe = os.path.abspath(a.exe)

    NRHO, FNUM = 7.07043e22, 1.767e6
    spB, spF = ("N2","O"), ("NO","N")
    Fcoeff = (0.0, -1.359, 5.175e-19)
    NONEQ_TCE = ("NO + N --> N2 + O\nE A 0.0 0.0 4.059e-12 -1.359 5.175e-19\n\n"
                 "N2 + O --> NO + N\nE B 0.0 0.0 0.0 0.0 0.0\n")

    def _frozen_species():
        out = []
        for line in open(os.path.join(V.HERE,"air.species")):
            w = line.split()
            if len(w) >= 10 and not line.strip().startswith("#"):
                w[4] = "0.0"; w[6] = "0.0"; out.append("  ".join(w))
            else: out.append(line.rstrip("\n"))
        return "\n".join(out) + "\n"
    def _frozen_elec():
        out = []
        for line in open(os.path.join(V.HERE,"air.elec")):
            w = line.split()
            if len(w) > 2 and not line.strip().startswith("#"):
                nl = int(w[1])
                for k in range(nl): w[2+5*k+1] = "0.0"
                out.append(" ".join(w))
            else: out.append(line.rstrip("\n"))
        return "\n".join(out) + "\n"
    xf = {"air_frozen.species": _frozen_species(),
          "air_frozen.elec":   _frozen_elec(),
          "rev_noneq.tce":     NONEQ_TCE}

    # cell-temperature-free reconstruction (microcanonical integral)
    du, ntab, mtab = V.db_table_shape(np, spB, spF, Fcoeff)
    def _interp(ecc):
        x = ecc/du; k = np.clip(x.astype(int), 0, ntab-2); f = x - k
        return (1.0-f)*mtab[k] + f*mtab[k+1]
    rng = np.random.default_rng(2024)
    def _avg(Ttr, Tint):
        return _interp(V.sample_ecc(np, rng, spB, Ttr, Tint, a.ns)).mean()

    omB = 0.5*(V.VSS[spB[0]] + V.VSS[spB[1]])
    def krev_an(T): return V.kb_lit(T)*V.keq_exchange(T)

    print("="*74)
    print("PART A  equilibrium: does cell-temp-free reproduce the analytic "
          "(cell-temp) rate?")
    print("="*74)
    print("  the reconstructed microcanonical rate vs analytic k_rev(T)="
          "k_fwd(T)/Keq(T):")
    r0 = None
    for T in (8000.,12000.,16000.,20000.):
        micro = _avg(T,T)*T**(1-omB)
        r = micro/krev_an(T)
        if r0 is None: r0 = r
        print("    T=%6.0fK   micro/analytic (norm) = %.4f   (dev %+.2f%%)"
              % (T, r/r0, 100*(r/r0-1)))
    print("  -> flat across 8-20 kK  =>  the two methods coincide in "
          "equilibrium (as required)")

    print()
    print("="*74)
    print("PART B  NON-equilibrium: two-temperature reservoir, T_tr=8000 K "
          "fixed, T_int swept")
    print("="*74)
    Ttr = 8000.0
    beq = None
    rows = []
    for Tint in (4000.0, 8000.0, 12000.0, 20000.0):
        log = V.run(exe, "in.reverse_noneq",
                    {"T":Ttr, "TINT":Tint, "NRHO":NRHO, "FNUM":FNUM},
                    "cmp%d"%Tint, [],
                    subs={"run             8000":"run             %d"%a.steps},
                    extra_files=xf)
        b = V.tallies(log).get("N2 + O --> NO + N", 0.0)
        if abs(Tint-Ttr) < 1: beq = b
        rows.append((Tint, b))
    print("  R = reverse-rate(T_tr,T_int) / reverse-rate(T_tr,T_tr)")
    print("  %-9s %-14s %-16s %-16s" %
          ("T_int", "R measured", "R cell-temp-free", "R cell-temp(T_tr)"))
    print("  %-9s %-14s %-16s %-16s" %
          ("(K)", "(SPARTA)", "(micro predict)", "(translational)"))
    print("  " + "-"*57)
    for Tint, b in rows:
        rm = b/beq if beq else float("inf")
        rp = _avg(Ttr, Tint)/_avg(Ttr, Ttr)
        rc = 1.0    # a translational cell-temperature is fixed at T_tr
        sig = 100*math.sqrt(1.0/max(b,1)+1.0/max(beq,1))
        print("  %-9.0f %-14s %-16.3f %-16.3f  (stat +-%.1f%%)"
              % (Tint, "%.3f"%rm, rp, rc, sig))
    print()
    print("  measured (SPARTA cell-temp-free) tracks the microcanonical")
    print("  prediction; a translational cell-temperature model is pinned at")
    print("  R=1 and misses the swing entirely.")

if __name__ == "__main__":
    main()
