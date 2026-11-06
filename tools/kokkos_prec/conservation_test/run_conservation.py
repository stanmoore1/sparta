#!/usr/bin/env python3
"""
Physical conservation and seed checks for a KOKKOS build of SPARTA.

usage: run_conservation.py --spa "path/to/spa_kokkos -k on -sf kk" --prec P
                           [--out OUTDIR]

  P = double, mixed or single (KOKKOS_PREC of the build)

Conservation: in.cons.* run a closed periodic box (no surfaces), where total
mass, momentum and energy must be conserved by the collision (and reaction)
kernels.  Each run fails if the largest relative change over the run of
  mass       sum m
  momentum   |sum m v|, against |sum m v| at step 0
  energy     sum (ke + erot + evib) (+ dissociation energy for in.cons.chem)
exceeds the tolerance for the precision.  Double precision conserves to
round-off (~1e-14).  In reduced precision the float round-off of the
velocities leaves 3e-9 (in.cons.vib) to 5e-8 (in.cons.mix: the exact 1:2
N/N2 mass ratio keeps a residual tie bias of ~1e-10 per collision).  The
rounding bias fixed in the collision kernels (float mass fractions in the
centre of mass velocity, ~4e-9 per collision) gave 2e-7 to 1.4e-6 here and
fails.

Seed: in.seed resets the seed before defining collisions and inflow, which
draw from the KOKKOS random pools.  The same seed must reproduce the run
exactly, a different seed must change it.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = ["co2.species", "co2.species.vib", "co2.vss",
        "air.species", "air.vss", "air.tce"]
INPUTS = ["in.cons.vib", "in.cons.mix", "in.cons.chem"]

# largest relative change allowed, for mass and for momentum and energy
TOL = {"double": (1e-12, 1e-12), "mixed": (1e-12, 1.5e-7), "single": (1e-12, 1.5e-7)}


def run(spa, inp, log, outdir, extra=()):
    cmd = shlex.split(spa) + ["-pk", "kokkos", "react/retry", "yes",
                              "-in", inp, "-log", log, "-echo", "none",
                              "-screen", "none"] + list(extra)
    r = subprocess.run(cmd, cwd=outdir, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FAILED {inp}: exit code {r.returncode}\n{r.stdout}{r.stderr}")
    return os.path.join(outdir, log)


def rows(log):
    out, on = [], False
    for line in open(log):
        w = line.split()
        if w[:2] == ["Step", "Np"]:
            on = True
            continue
        if on:
            if not w or not w[0].isdigit():
                on = False
                continue
            out.append([float(x) for x in w])
    if not out:
        sys.exit(f"FAILED: no stats output in {log}")
    return out


def drift(log):
    # stats columns: step np mass px py pz etot
    r = rows(log)
    r0 = r[0]
    p0 = sum(x * x for x in r0[3:6]) ** 0.5
    dm = max(abs(a[2] - r0[2]) for a in r) / r0[2]
    dp = max(sum((a[i] - r0[i]) ** 2 for i in (3, 4, 5)) ** 0.5 for a in r) / p0
    de = max(abs(a[6] - r0[6]) for a in r) / r0[6]
    return dm, dp, de


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spa", required=True, help="command to run SPARTA with KOKKOS")
    ap.add_argument("--prec", required=True, choices=sorted(TOL))
    ap.add_argument("--out", default="conservation_run")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for f in DATA + INPUTS + ["in.seed"]:
        shutil.copy(os.path.join(HERE, f), args.out)

    tol_m, tol_pe = TOL[args.prec]
    nfail = 0
    for inp in INPUTS:
        dm, dp, de = drift(run(args.spa, inp, "log." + inp[3:], args.out))
        ok = dm <= tol_m and dp <= tol_pe and de <= tol_pe
        nfail += not ok
        print(f"{'ok    ' if ok else 'FAILED'} {inp:14s} mass {dm:.1e}  "
              f"momentum {dp:.1e}  energy {de:.1e}  "
              f"(tolerance {tol_m:.0e}, {tol_pe:.0e})")

    seed = {}
    for tag, s in (("a", "777"), ("b", "777"), ("c", "4242")):
        seed[tag] = rows(run(args.spa, "in.seed", "log.seed." + tag, args.out,
                             ("-var", "cseed", s)))
    same = seed["a"] == seed["b"]
    differ = seed["a"] != seed["c"]
    nfail += not (same and differ)
    print(f"{'ok    ' if same and differ else 'FAILED'} in.seed        "
          f"same seed reproduces: {same}, different seed changes the run: {differ}")

    if nfail:
        sys.exit(f"{nfail} check(s) failed")
    print("all checks passed")


if __name__ == "__main__":
    main()
