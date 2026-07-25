#!/usr/bin/env python3
"""Regression tests for fix ablate mode = deposit.

The feature under test lets an implicit surface grow into the gas.  A growing
surface runs into gas particles, and the whole point of the implementation is
that those particles are reflected off the advancing front rather than
deleted.  These tests assert that property directly.

Usage:
    python3 run_tests.py [--exe ../../../src/spa_serial]
    python3 run_tests.py --exe ../../../src/spa_mpi --ranks 4
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

RANKS = 1

# stats_style columns: step np f_ablate f_ablate[3] f_ablate[4] f_ablate[8]
#   f_ablate     = summed corner point values, i.e. how much material exists
#   f_ablate[3]  = nburied        (particles incorporated into the film)
#   f_ablate[4]  = buried mass
#   f_ablate[8]  = nfrontreflect  (salvage reflections after a regeneration)

MASS_N = 2.325e-26      # from air.species, used to check the mass bookkeeping


def run(exe, infile, variables=None):
    cmd = [exe, "-in", infile]
    for k, v in (variables or {}).items():
        cmd += ["-var", k, str(v)]
    if RANKS > 1:
        cmd = ["mpirun", "-n", str(RANKS), "--oversubscribe", "--bind-to", "none"] + cmd
    env = dict(os.environ)
    # OpenMPI refuses to run as root unless told otherwise
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr


def stats_rows(out):
    """Return the stats table as a list of float lists."""
    rows, inside = [], False
    for line in out.splitlines():
        if line.startswith("Step "):
            inside = True
            continue
        if inside:
            if line.startswith("Loop time") or not line.strip():
                break
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                break
    return rows


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -- " + detail) if detail else ""))
    return ok


def test_conserve(exe, rate, nevery):
    """No particle may vanish without being counted as buried."""
    label = "conserve (rate=%s nevery=%s)" % (rate, nevery)
    rc, out = run(exe, "in.test.conserve", {"RATE": rate, "NEVERY": nevery})
    if rc != 0:
        err = next((l for l in out.splitlines() if "ERROR" in l), "no stats")
        return check(label, False, err.strip()[:100])

    rows = stats_rows(out)
    if len(rows) < 2:
        return check(label, False, "no stats rows")

    np0, npN = rows[0][1], rows[-1][1]
    nburied, buried_mass, nreflect = rows[-1][3], rows[-1][4], rows[-1][5]

    lost = np0 - npN
    ok = True

    # the headline invariant: every particle that left was accounted for
    ok &= check(label + " : lost == buried", lost == nburied,
                "lost=%d buried=%d" % (lost, nburied))

    # the mass ledger must match the count exactly (single species, no weighting)
    expect = nburied * MASS_N
    tol = max(1e-12 * max(expect, 1e-30), 1e-30)
    ok &= check(label + " : buried mass ledger", abs(buried_mass - expect) <= tol,
                "got=%g expect=%g" % (buried_mass, expect))

    # fix grid/check ran with the error setting, so completing means no
    # particle was ever left inside the growing surface
    print("      (reflections salvaged: %d)" % nreflect)
    return ok


def test_momentum(exe, rate, nevery, infile="in.test.momentum", label=""):
    """In a periodic box the surface is the only place gas momentum can go."""
    name = "momentum (rate=%s nevery=%s)%s" % (rate, nevery, label)
    rc, out = run(exe, infile, {"RATE": rate, "NEVERY": nevery})
    if rc != 0:
        err = next((l for l in out.splitlines() if "ERROR" in l), "no stats")
        return check(name, False, err.strip()[:90])

    rows = stats_rows(out)
    if len(rows) < 2:
        return check(name, False, "no stats rows")
    f, l = rows[0], rows[-1]

    # cols: 0 step, 1 np, 2-4 summed velocity, 5-7 surface, 8-10 buried,
    #       11-13 reflected
    total = []
    for k in range(3):
        dgas = MASS_N * (l[2+k] - f[2+k])
        total.append(dgas + (l[5+k]-f[5+k]) + (l[8+k]-f[8+k]) + (l[11+k]-f[11+k]))

    scale = max(max(abs(MASS_N*f[2+k]), abs(MASS_N*l[2+k])) for k in range(3))
    rel = max(abs(t) for t in total) / scale

    # round-off over ~10^6 collisions, not a physics tolerance
    return check(name, rel < 1e-5, "relative residual %.2e" % rel)


def test_no_wall_work(exe):
    """A growing surface must not do work on the gas.

    A depositing front advances by accretion, so the atoms a molecule strikes
    are at rest and the front velocity must not enter the reflection.  Off a
    stationary plane, specular reflection preserves speed exactly, so the gas
    temperature must not move -- however often the front catches a particle.
    A wall physically translating at the same speed would be a piston and
    would heat the gas.
    """
    temps, ok = {}, True
    for rate in (0.0, 0.02, 0.1):
        rc, out = run(exe, "in.test.wall", {"RATE": rate, "WALL": "specular"})
        if rc != 0:
            return check("growing wall does no work", False, "run failed")
        rows = stats_rows(out)
        if len(rows) < 2:
            return check("growing wall does no work", False, "no stats rows")
        # cols: 0 step, 1 np, 2 temperature, 3 nscoll, 4 nburied, 5 nsalvaged
        temps[rate] = (rows[-1][2], rows[-1][3], rows[-1][5], rows[-1][4])

    base = temps[0.0][0]
    for rate in (0.02, 0.1):
        t, nscoll, nsalv, nburied = temps[rate]
        # only meaningful while nothing is buried: burial removes the slowest
        # particles, which shifts the mean of what is left for reasons that
        # have nothing to do with the wall
        ok &= check("growing wall does no work (rate=%s)" % rate,
                    nburied == 0 and t == base,
                    "T=%.8g vs %.8g, %d front collisions salvaged"
                    % (t, base, nsalv))
    print("      (surface collision rate rose from %d to %d with no change "
          "in T)" % (temps[0.0][1], temps[0.1][1]))
    return ok


def test_guard(exe):
    """A front that outruns its collision lists must be refused."""
    rc, out = run(exe, "in.test.guard")
    saw = "front advances" in out
    return check("guard rejects an over-fast front", rc != 0 and saw,
                 "rc=%d" % rc if not saw else "")


def test_ablate_unaffected(exe):
    """Ablation must not touch any of the deposition machinery."""
    rc, out = run(exe, "in.test.ablate")
    if rc != 0:
        return check("ablate mode unaffected", False, "run failed")
    rows = stats_rows(out)
    if len(rows) < 2:
        return check("ablate mode unaffected", False, "no stats rows")
    nburied, nreflect = rows[-1][3], rows[-1][4]
    return check("ablate mode unaffected", nburied == 0 and nreflect == 0,
                 "buried=%d reflect=%d" % (nburied, nreflect))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(HERE, "../../../src/spa_serial"))
    ap.add_argument("--ranks", type=int, default=1,
                    help="MPI ranks; >1 requires an MPI build (spa_mpi)")
    args = ap.parse_args()

    global RANKS
    RANKS = args.ranks

    exe = os.path.abspath(args.exe)
    if not os.path.exists(exe):
        sys.exit("executable not found: %s (build it with 'make serial')" % exe)
    print("running with %d rank(s): %s\n" % (RANKS, exe))

    ok = True
    # sweep the growth rate: slow enough that nothing is buried, up to fast
    # enough that some particles are, and check the ledger balances throughout
    for rate in (0.02, 0.05, 0.1, 0.2, 0.3):
        ok &= test_conserve(exe, rate, 1)
    # the same physics with the isosurface regenerated less often
    for nevery in (2, 5):
        ok &= test_conserve(exe, 0.05 * nevery, nevery)
    # momentum: gas + surface + buried + reflected must balance
    for rate, nevery in ((0.2, 1), (0.5, 5), (1.0, 20)):
        ok &= test_momentum(exe, rate, nevery)
    # the same ledger in 3d, which uses marching cubes and the triangle
    # intersection rather than marching squares and the line one
    ok &= test_momentum(exe, 0.5, 5, "in.test.momentum.3d", label=" 3d")
    # the front velocity places the collision but must not enter the rebound
    ok &= test_no_wall_work(exe)
    ok &= test_guard(exe)
    ok &= test_ablate_unaffected(exe)

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
