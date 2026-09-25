#!/usr/bin/env python3
"""Compare the thermo output of two sets of SPARTA log files.

Two modes:

  exact   (default) every thermo row must be identical, used to check that
          a KOKKOS_PREC=double build reproduces the original code exactly
  stats   compare the time average of each thermo column over the last
          --fraction of the run, for reduced precision builds whose
          trajectories diverge from the double precision ones; a column
          passes if the averages agree within --rtol relative, or within
          --nsigma standard errors of the double precision run

usage: compare_logs.py [--stats] [--gold] REFDIR NEWDIR [--pattern 'log.mpi_1.*']
Log files are matched by path relative to REFDIR and NEWDIR.
"""

import argparse
import fnmatch
import math
import os
import re
import sys

# wall clock columns, never compared

TIMING_COLUMNS = ("CPU", "Elapsed", "Time", "TpS", "CPULeft")

# with fewer thermo rows than this in the averaging window, a column also
#   passes within FEW_RTOL relative difference

MIN_SAMPLES = 5
FEW_RTOL = 0.1


def thermo_blocks(path):
    """list of (header, rows) for every thermo block in a log file"""
    blocks = []
    header = None
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            w = line.split()
            if w and w[0] == "Step":
                header = w
                rows = []
                continue
            if header is None:
                continue
            if w and w[0] == "Loop":
                blocks.append((header, rows))
                header = None
                continue
            if len(w) == len(header):
                try:
                    rows.append([float(x) for x in w])
                except ValueError:
                    pass
    if header is not None:
        blocks.append((header, rows))
    return blocks


def find_logs(root, pattern, gold=False):
    """log files under root matching pattern, keyed by relative path; with
    gold=True the gold standard names log.DATE.mpi_N.NAME in the examples
    directories are keyed as log.mpi_N.NAME, the name a test run writes"""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            name = f
            if gold:
                m = re.match(r"log\.\w+\.(mpi_\d+\..*)$", f)
                if not m:
                    continue
                name = "log." + m.group(1)
            if fnmatch.fnmatch(name, pattern):
                p = os.path.join(d, f)
                out[os.path.relpath(os.path.join(d, name), root)] = p
    return out


def compare_exact(ref, new):
    rb, nb = thermo_blocks(ref), thermo_blocks(new)
    if len(rb) != len(nb):
        return "different number of thermo blocks (%d vs %d)" % (len(rb), len(nb))
    for (rh, rr), (nh, nr) in zip(rb, nb):
        if rh != nh:
            return "different thermo columns"
        if len(rr) != len(nr):
            return "different number of thermo rows (%d vs %d)" % (len(rr), len(nr))
        cols = [c for c, h in enumerate(rh) if h not in TIMING_COLUMNS]
        for a, b in zip(rr, nr):
            if any(a[c] != b[c] for c in cols):
                return "rows differ at step %g" % a[0]
    return None


def compare_stats(ref, new, fraction, rtol, nsigma):
    rb, nb = thermo_blocks(ref), thermo_blocks(new)
    if len(rb) != len(nb):
        return ["different number of thermo blocks (%d vs %d)" % (len(rb), len(nb))]
    msgs = []
    for b, ((rh, rr), (nh, nr)) in enumerate(zip(rb, nb)):
        if rh != nh:
            msgs.append("thermo block %d: different columns" % b)
            continue
        if not rr and not nr:
            continue
        if not rr or not nr:
            msgs.append("thermo block %d: no data rows in %s log"
                        % (b, "reference" if not rr else "new"))
            continue
        if len(nr) < len(rr):
            msgs.append("thermo block %d: run stopped early (%d of %d rows)"
                        % (b, len(nr), len(rr)))
            continue
        for c in range(1, len(rh)):
            if rh[c] in TIMING_COLUMNS:
                continue
            ra = [r[c] for r in rr[int(len(rr) * (1 - fraction)):]]
            na = [r[c] for r in nr[int(len(nr) * (1 - fraction)):]]
            if not ra or not na:
                continue
            rm, nm = sum(ra) / len(ra), sum(na) / len(na)
            if any(math.isnan(x) or math.isinf(x) for x in na):
                msgs.append("%s: NaN/Inf" % rh[c])
                continue
            var = sum((x - rm) ** 2 for x in ra) / max(1, len(ra) - 1)
            nvar = sum((x - nm) ** 2 for x in na) / max(1, len(na) - 1)
            # standard error of the difference of the two means
            sem = math.sqrt(var / len(ra) + nvar / len(na)) if min(len(ra), len(na)) > 1 else 0.0
            diff = abs(nm - rm)
            # relative to the mean, or to the fluctuation of a quantity
            #   that averages to about zero
            scale = max(abs(rm), math.sqrt(var), 1e-300)
            if diff <= rtol * scale or diff <= nsigma * sem or (rm == 0 and nm == 0):
                continue
            # too few samples for a meaningful standard error
            if min(len(ra), len(na)) < MIN_SAMPLES and diff <= FEW_RTOL * scale:
                continue
            msgs.append("%s: ref %.6g new %.6g (rel %.3g, %.1f sem)"
                        % (rh[c], rm, nm, diff / scale, diff / sem if sem else float("inf")))
    return msgs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refdir")
    ap.add_argument("newdir")
    ap.add_argument("--pattern", default="log.mpi_1.*")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--gold", action="store_true",
                    help="REFDIR holds gold standard logs named log.DATE.mpi_N.NAME")
    ap.add_argument("--only", nargs="*", help="compare only these log names")
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--rtol", type=float, default=0.02)
    ap.add_argument("--nsigma", type=float, default=4.0)
    args = ap.parse_args()

    ref = find_logs(args.refdir, args.pattern, args.gold)
    if args.only:
        ref = {k: v for k, v in ref.items() if os.path.basename(k) in args.only
               or os.path.basename(k).split(".", 2)[-1] in args.only}
    new = find_logs(args.newdir, args.pattern)
    nfail = 0
    for rel in sorted(ref):
        if rel not in new:
            print("MISSING  %s" % rel)
            nfail += 1
            continue
        if args.stats:
            msgs = compare_stats(ref[rel], new[rel], args.fraction, args.rtol, args.nsigma)
            if msgs:
                nfail += 1
                print("DIFFER   %s\n           %s" % (rel, "\n           ".join(msgs)))
            else:
                print("OK       %s" % rel)
        else:
            msg = compare_exact(ref[rel], new[rel])
            if msg:
                nfail += 1
                print("DIFFER   %s: %s" % (rel, msg))
            else:
                print("SAME     %s" % rel)
    print("%d of %d logs %s" % (len(ref) - nfail, len(ref),
                                "agree" if args.stats else "identical"))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
