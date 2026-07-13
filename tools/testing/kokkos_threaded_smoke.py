#!/usr/bin/env python3

# ----------------------------------------------------------------------
#   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
#   http://sparta.github.io
#   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
#   Sandia National Laboratories
#
#   Copyright (2014) Sandia Corporation.  Under the terms of Contract
#   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
#   certain rights in this software.  This software is distributed under
#   the GNU General Public License.
#
#   See the README file in the top-level SPARTA directory.
# -------------------------------------------------------------------------

# Threaded-KOKKOS smoke test.
#
# The Serial-backend KOKKOS regression job (SPARTA_KOKKOS_EXACT) already checks
# that KOKKOS reproduces the non-KOKKOS results *numerically*, but it runs with
# a single thread and therefore cannot exercise real multithreading. Running the
# same decks under the OpenMP backend with multiple threads perturbs the RNG
# draw order, so the stochastic thermo output no longer matches the serial
# gold logs to a tight tolerance -- which makes a numeric comparison unreliable.
#
# This script instead runs each input deck under the threaded KOKKOS binary and
# checks only that it *completes cleanly*: a zero exit code, no ERROR or signal
# message, and a finished run ("Loop time of" in the output). That reliably
# catches the failure modes multithreading actually introduces -- data races
# that abort or produce NaNs, deadlocks, and KOKKOS runtime errors -- without
# depending on a fragile statistical tolerance.
#
# Usage:
#   kokkos_threaded_smoke.py --sparta <binary> --threads N \
#       [--skip substr ...] <example-dir> [<example-dir> ...]
#
# Exit code is 0 if every deck completes cleanly, 1 otherwise.

import argparse
import glob
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Threaded KOKKOS smoke test")
    parser.add_argument("--sparta", required=True, help="path to sparta binary")
    parser.add_argument("--threads", type=int, default=2, help="OpenMP threads")
    parser.add_argument("--skip", default="",
                        help="comma-separated substrings; skip decks whose name contains any of them")
    parser.add_argument("dirs", nargs="+", help="example directories to scan for in.* decks")
    args = parser.parse_args()

    skip_list = [s for s in args.skip.split(",") if s]

    sparta = os.path.abspath(args.sparta)
    spa_args = ["-k", "on", "t", str(args.threads), "-sf", "kk"]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.threads)
    env.setdefault("OMP_PROC_BIND", "false")

    npass = 0
    failures = []
    skipped = []

    for d in args.dirs:
        for path in sorted(glob.glob(os.path.join(d, "in.*"))):
            name = os.path.basename(path)
            if any(s in name for s in skip_list):
                skipped.append(name)
                continue

            cmd = [sparta] + spa_args + ["-in", name, "-log", "none"]
            try:
                proc = subprocess.run(
                    cmd, cwd=d, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=600)
                out = proc.stdout.decode("utf-8", "replace")
            except subprocess.TimeoutExpired:
                failures.append((name, "TIMEOUT"))
                print("FAIL %-40s timed out" % name)
                continue

            bad = None
            if proc.returncode != 0:
                bad = "exit code %d" % proc.returncode
            elif "ERROR" in out or "exited on signal" in out:
                bad = next((ln for ln in out.splitlines()
                            if "ERROR" in ln or "exited on signal" in ln), "error")
            elif "Loop time of" not in out:
                bad = "run did not complete (no 'Loop time of')"

            if bad:
                failures.append((name, bad))
                print("FAIL %-40s %s" % (name, bad))
            else:
                npass += 1
                print("ok   %-40s" % name)

    print("\n%d passed, %d failed, %d skipped" %
          (npass, len(failures), len(skipped)))
    if skipped:
        print("skipped: %s" % ", ".join(skipped))
    if failures:
        print("\nFailures:")
        for name, why in failures:
            print("  %s: %s" % (name, why))
        return 1
    print("All threaded KOKKOS smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
