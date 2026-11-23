#!/usr/bin/env python3
"""
KOKKOS device code that no example exercises, compared with the non-KOKKOS
code.

usage: run_features.py --spa "path/to/spa_kokkos" [--np N] [--out OUTDIR]
                       [names ...]

Each in.feat.NAME input (or only the NAMEs given) is run twice with the same
executable: without KOKKOS (the reference) and with "-k on -sf kk".  A run
fails if it stops early, or if the time average of any thermo column over
the second half of the run differs from the reference by more than 2% and
4 standard errors (compare_logs.py --stats).  Works for any KOKKOS_PREC.

  in.feat.emitfrac   fix emit/surf perspecies no with custom per-surf
                     species fractions (crashed in KOKKOS before this fix)
  in.feat.computes   per grid cell computes reduced to global values
  in.feat.histo      fix ave/histo and ave/histo/weight
  in.feat.nngroup    near-neighbor (nearcp) collisions with several groups
  in.feat.region.*   fix emit/face restricted to a block, sphere, cylinder,
                     plane, and union/intersect region
  in.feat.subsonic   fix emit/face subsonic boundary
  in.feat.surfglobal surf_react global
  in.feat.tceqk      react tce/qk chemistry
  in.feat.vibemit    discrete vibration (fix vibmode) with fix emit/face
"""

import argparse
import glob
import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from compare_logs import compare_stats  # noqa: E402


def run(cmd, inp, log, outdir):
    r = subprocess.run(cmd + ["-in", inp, "-log", log, "-echo", "none",
                              "-screen", "none"],
                       cwd=outdir, capture_output=True, text=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spa", required=True,
                    help="SPARTA executable built with KOKKOS (without -k/-sf)")
    ap.add_argument("--np", type=int, default=1, help="MPI ranks (default 1)")
    ap.add_argument("--out", default="feature_run")
    ap.add_argument("names", nargs="*", help="feature names, default all")
    args = ap.parse_args()

    inputs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(HERE, "in.feat.*")))
    if args.names:
        inputs = [i for i in inputs if i[len("in.feat."):] in args.names]
    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(HERE):
        if not f.endswith(".py"):
            shutil.copy(os.path.join(HERE, f), args.out)

    base = shlex.split(args.spa)
    if args.np > 1:
        base = ["mpirun", "-np", str(args.np)] + base
    nfail = 0
    for inp in inputs:
        name = inp[len("in.feat."):]
        rc_h = run(base, inp, "log.host." + name, args.out)
        rc_k = run(base + ["-k", "on", "-sf", "kk"], inp, "log.kk." + name, args.out)
        if rc_h or rc_k:
            msgs = ["exit code: non-KOKKOS %d, KOKKOS %d" % (rc_h, rc_k)]
        else:
            msgs = compare_stats(os.path.join(args.out, "log.host." + name),
                                 os.path.join(args.out, "log.kk." + name),
                                 0.5, 0.02, 4.0)
        nfail += bool(msgs)
        print("%s %-16s %s" % ("FAILED" if msgs else "ok    ", name, "; ".join(msgs)))
    if nfail:
        sys.exit("%d of %d feature tests failed" % (nfail, len(inputs)))
    print("all %d feature tests passed" % len(inputs))


if __name__ == "__main__":
    main()
