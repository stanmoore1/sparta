#!/usr/bin/env python3
"""Run a set of SPARTA example inputs and collect their log files.

Each example directory is copied under OUTDIR, and each input in.NAME is run
there, writing log.mpi_1.NAME, the same name the regression tests use, so
the logs can be compared with compare_logs.py against another run or (with
--gold) against the gold standard logs in examples/.

usage: run_examples.py --spa "path/to/spa_kokkos -k on -sf kk" --out OUTDIR
                       [--list FILE | NAME ...] [--examples DIR] [-j N]
The default list is ci_examples.txt next to this script.
"""

import argparse
import concurrent.futures
import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def find_input(examples, name):
    for d, _, files in os.walk(examples):
        if "in." + name in files:
            return d
    return None


def run_one(spa, examples, out, name, timeout):
    src = find_input(examples, name)
    if src is None:
        return name, "no input in.%s" % name
    dst = os.path.join(out, os.path.relpath(src, examples))
    log = "log.mpi_1." + name
    cmd = shlex.split(spa) + ["-in", "in." + name, "-log", log]
    try:
        p = subprocess.run(cmd, cwd=dst, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        return name, "timeout"
    with open(os.path.join(dst, "stdout.mpi_1." + name), "wb") as f:
        f.write(p.stdout)
    if p.returncode != 0:
        return name, "exit code %d" % p.returncode
    return name, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--spa", required=True, help="command to run SPARTA")
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", default=os.path.join(HERE, "ci_examples.txt"))
    ap.add_argument("--examples", default=os.path.join(ROOT, "examples"))
    ap.add_argument("-j", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--timeout", type=float, default=1800)
    args = ap.parse_args()

    names = args.names or open(args.list).read().split()
    # copy each example directory once, without old logs
    for name in names:
        src = find_input(args.examples, name)
        if src is None:
            continue
        dst = os.path.join(args.out, os.path.relpath(src, args.examples))
        if not os.path.isdir(dst):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("log.*"))

    nfail = 0
    with concurrent.futures.ThreadPoolExecutor(args.j) as ex:
        futs = [ex.submit(run_one, args.spa, args.examples, args.out, n, args.timeout)
                for n in names]
        for f in concurrent.futures.as_completed(futs):
            name, err = f.result()
            if err:
                nfail += 1
            print("%-8s %s%s" % ("FAILED" if err else "ran", name,
                                 ": " + err if err else ""), flush=True)
    print("%d of %d examples ran" % (len(names) - nfail, len(names)))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
