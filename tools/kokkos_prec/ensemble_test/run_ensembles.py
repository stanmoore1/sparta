#!/usr/bin/env python3
"""
Seed ensembles: reduced precision vs double precision means.

usage: run_ensembles.py --spa PREC=CMD [--spa PREC=CMD ...] [--seeds N]
                        [--out OUTDIR] [case ...]

  e.g. --spa double=build-double/src/spa_kokkos_mpi_only
       --spa mixed=build-mixed/src/spa_kokkos_mpi_only

A single reduced precision run is one sample of a noisy process, so a
difference from a double precision run says little.  Here each case (an
example input, default list below) is run with N seeds (default 8) in each
precision.  For every thermo column, the time average over the second half
of each run gives N samples per precision, and a Welch t-test compares each
precision's mean with the first precision given (the reference, normally
double).  The test fails on a failed run, NaN/Inf, or |t| > 4; |t| > 3
is expected about once per 100 comparisons by chance.

Takes a few hours on one CPU core for the default list and 8 seeds in three
precisions.  Not run in CI.
"""

import argparse
import glob
import math
import os
import re
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from compare_logs import thermo_blocks, TIMING_COLUMNS  # noqa: E402

EXAMPLES = os.path.join(HERE, "..", "..", "..", "examples")

# example dir/input; very short runs are run 4x longer (LONGER)
CASES = ["collide/in.collide", "collide/in.collideInterspecies",
         "thermostat/in.thermostat", "chem/in.chem", "chem/in.chem.gastally",
         "qk/in.qk", "ambi/in.ambi.group.react", "vibrate/in.vibrate",
         "emit/in.emit.face", "circle/in.circle", "surf_collide/in.circle.td",
         "surf_collide/in.beam.cll", "surf_react_adsorb/in.beam.surf.gs_ps",
         "optmove/in.optmove.axi", "variable_timestep/in.variable_dt",
         "torque/in.torque"]
LONGER = ("vibrate", "emit.face", "beam.cll")


def averages(path):
    out = {}
    for b, (h, rows) in enumerate(thermo_blocks(path)):
        half = rows[len(rows) // 2:]
        for c in range(1, len(h)):
            if half and h[c] not in TIMING_COLUMNS:
                out[(b, h[c])] = sum(r[c] for r in half) / len(half)
    return out


def mean_var(xs):
    n = len(xs)
    m = sum(xs) / n
    return n, m, sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spa", action="append", required=True, metavar="PREC=CMD")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default="ensemble_run")
    ap.add_argument("cases", nargs="*", help="dir/in.NAME, default list in this script")
    args = ap.parse_args()
    spa = dict(s.split("=", 1) for s in args.spa)
    cases = args.cases or CASES

    nfail = 0
    ens = {}
    for p, cmd in spa.items():
        for case in cases:
            d, inp = os.path.split(case)
            name = inp[len("in."):]
            w = os.path.join(args.out, p, name)
            shutil.copytree(os.path.join(EXAMPLES, d), w, dirs_exist_ok=True)
            for f in glob.glob(os.path.join(w, "log.*")):
                os.remove(f)
            text = open(os.path.join(w, inp)).read()
            if name in LONGER:
                text = re.sub(r"^(run\s+)(\d+)", lambda m: m.group(1) + str(4 * int(m.group(2))),
                              text, flags=re.M)
            for k in range(1, args.seeds + 1):
                seeded = re.sub(r"^(seed\s+)\d+", r"\g<1>%d" % (1000 + k), text, flags=re.M)
                open(os.path.join(w, "in.ens%d" % k), "w").write(seeded)
                r = subprocess.run(shlex.split(cmd) + ["-k", "on", "-sf", "kk",
                                                       "-in", "in.ens%d" % k,
                                                       "-log", "log.ens%d" % k,
                                                       "-screen", "none"],
                                   cwd=w, capture_output=True, text=True)
                if r.returncode != 0:
                    print("FAILED %s %s seed %d: exit code %d" % (p, name, k, r.returncode))
                    nfail += 1
                    continue
                ens.setdefault((p, name), []).append(averages(os.path.join(w, "log.ens%d" % k)))

    ref = next(iter(spa))
    worst = []
    for case in cases:
        name = os.path.basename(case)[len("in."):]
        a_runs = ens.get((ref, name), [])
        for p in list(spa)[1:]:
            b_runs = ens.get((p, name), [])
            if len(a_runs) < 2 or len(b_runs) < 2:
                continue
            for key in a_runs[0]:
                a = [e[key] for e in a_runs if key in e]
                b = [e[key] for e in b_runs if key in e]
                if len(a) < 2 or len(b) < 2:
                    continue
                if any(math.isnan(x) or math.isinf(x) for x in b):
                    print("FAILED %s %s %s: NaN/Inf" % (p, name, key[1]))
                    nfail += 1
                    continue
                na, ma, va = mean_var(a)
                nb, mb, vb = mean_var(b)
                se = math.sqrt(va / na + vb / nb)
                t = (mb - ma) / se if se else (0.0 if ma == mb else float("inf"))
                worst.append((abs(t), name, p, key, ma, mb, t))

    worst.sort(key=lambda w: w[0], reverse=True)
    n3 = sum(1 for w in worst if w[0] > 3)
    n4 = sum(1 for w in worst if w[0] > 4)
    print("%d comparisons vs %s: |t| > 3: %d (about %.1f expected by chance), |t| > 4: %d"
          % (len(worst), ref, n3, 0.0096 * len(worst), n4))
    for w in worst[:10]:
        print("  %-22s %-6s block %d %-14s %s %.6g  %s %.6g  t %+.2f"
              % (w[1], w[2], w[3][0], w[3][1], ref, w[4], w[2], w[5], w[6]))
    if nfail or n4:
        sys.exit("ensemble test failed")
    print("ensemble test passed")


if __name__ == "__main__":
    main()
