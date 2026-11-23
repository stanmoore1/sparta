#!/usr/bin/env python3
"""
Restart files across KOKKOS precisions.

usage: run_restart.py --spa PREC=CMD [--spa PREC=CMD ...] [--out OUTDIR]

  e.g. --spa double=build-double/src/spa_kokkos_mpi_only
       --spa mixed=build-mixed/src/spa_kokkos_mpi_only
       --spa single=build-single/src/spa_kokkos_mpi_only

examples/circle (flow around a circle, diffuse surface, inflow) is run in
each given precision for 2000 steps uninterrupted, and for 1000 steps
ending in write_restart.  Each restart file is then read in every given
precision and run for 1000 more steps.  Restart files are always double
precision, so every continuation must agree with the second half of the
uninterrupted run of the first precision given: each thermo column's mean
within 2% or 4 standard errors (Poisson floor for counts).
"""

import argparse
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

CIRCLE = os.path.join(HERE, "..", "..", "..", "examples", "circle")

READ = """seed                12345
global              gridcut 0.0 comm/sort yes
read_restart        restart.{wp}
surf_collide        1 diffuse 300.0 0.0
surf_modify         all collide 1
collide             vss air air.vss
fix                 in emit/face air xlo twopass
timestep            0.0001
stats               100
stats_style         step cpu np nattempt ncoll nscoll nscheck
run                 1000
"""


def run(cmd, inp, log, outdir):
    r = subprocess.run(shlex.split(cmd) + ["-k", "on", "-sf", "kk", "-in", inp,
                                           "-log", log, "-echo", "none",
                                           "-screen", "none"],
                       cwd=outdir, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FAILED {inp}: exit code {r.returncode}\n{r.stdout}{r.stderr}")


def rows_after(path, step0):
    h, rows = thermo_blocks(path)[-1]
    return h, [r for r in rows if r[0] > step0]


def mean_var(xs):
    n = len(xs)
    m = sum(xs) / n
    return m, sum((x - m) ** 2 for x in xs) / max(1, n - 1), n


def compare(ref, new):
    (hr, rr), (hn, rn) = ref, new
    msgs = []
    for c, name in enumerate(hn):
        if c == 0 or name in TIMING_COLUMNS or name not in hr:
            continue
        a = [r[hr.index(name)] for r in rr]
        b = [r[c] for r in rn]
        ma, va, na = mean_var(a)
        mb, vb, nb = mean_var(b)
        sem = math.sqrt(va / na + vb / nb)
        if all(x >= 0 and x == int(x) for x in a + b):
            sem = max(sem, math.sqrt(max(ma, 1.0) / na + max(mb, 1.0) / nb))
        d = abs(mb - ma)
        if d > 0.02 * abs(ma) and d > 4 * sem:
            msgs.append("%s %.4g vs %.4g" % (name, ma, mb))
    return msgs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spa", action="append", required=True, metavar="PREC=CMD")
    ap.add_argument("--out", default="restart_run")
    args = ap.parse_args()
    spa = dict(s.split("=", 1) for s in args.spa)

    os.makedirs(args.out, exist_ok=True)
    for f in ("air.species", "air.vss", "data.circle"):
        shutil.copy(os.path.join(CIRCLE, f), args.out)
    circle = open(os.path.join(CIRCLE, "in.circle")).read()
    full = re.sub(r"^run\s+\d+", "run 2000", circle, flags=re.M)
    open(os.path.join(args.out, "in.full"), "w").write(full)

    for p, cmd in spa.items():
        write = re.sub(r"^run\s+\d+", "run 1000\nwrite_restart restart.%s" % p,
                       circle, flags=re.M)
        open(os.path.join(args.out, "in.write." + p), "w").write(write)
        run(cmd, "in.write." + p, "log.write." + p, args.out)
        run(cmd, "in.full", "log.full." + p, args.out)

    first = next(iter(spa))
    ref = rows_after(os.path.join(args.out, "log.full." + first), 1000)
    nfail = 0
    for wp in spa:
        for rp, cmd in spa.items():
            inp = "in.read.%s.%s" % (wp, rp)
            open(os.path.join(args.out, inp), "w").write(READ.format(wp=wp))
            run(cmd, inp, "log.read.%s.%s" % (wp, rp), args.out)
            new = rows_after(os.path.join(args.out, "log.read.%s.%s" % (wp, rp)), 1000)
            msgs = compare(ref, new) if new[1] else ["no stats output after the restart"]
            nfail += bool(msgs)
            print("%s written %-6s read %-6s %s" % ("FAILED" if msgs else "ok    ", wp, rp,
                                                    "; ".join(msgs)))
    if nfail:
        sys.exit("%d restart combination(s) failed" % nfail)
    print("all restart combinations agree with the uninterrupted %s run" % first)


if __name__ == "__main__":
    main()
