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

# stats_style columns:
#   step np f_ablate f_ablate[3] f_ablate[4] f_ablate[8] f_ablate[20]
#   f_ablate     = summed corner point values, i.e. how much material exists
#   f_ablate[3]  = nburied        (particles incorporated into the film)
#   f_ablate[4]  = buried mass
#   f_ablate[8]  = nfrontreflect  (salvage reflections after a regeneration)
#   f_ablate[20] = nfrontmigrate  (of those, pushed across a proc boundary)

MASS_N = 2.325e-26      # from air.species, used to check the mass bookkeeping


def run(exe, infile, variables=None, ranks=None, cwd=None):
    cmd = [exe, "-in", infile]
    for k, v in (variables or {}).items():
        cmd += ["-var", k, str(v)]
    n = RANKS if ranks is None else ranks
    if n > 1:
        cmd = ["mpirun", "-n", str(n), "--oversubscribe", "--bind-to", "none"] + cmd
    env = dict(os.environ)
    # OpenMPI refuses to run as root unless told otherwise
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    p = subprocess.run(cmd, cwd=cwd or HERE, capture_output=True, text=True, env=env)
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


def test_fill(exe, rate):
    """The film may grow until it runs out of box, and still balance.

    Here the corner point grid is the whole box, so nothing stops the surface
    short.  That covers the case in.test.conserve cannot: a surface ending on
    the simulation box is legal, and the group boundary check has to allow it.
    """
    label = "fill (rate=%s)" % rate
    rc, out = run(exe, "in.test.fill", {"RATE": rate})
    if rc != 0:
        err = next((l for l in out.splitlines() if "ERROR" in l), "no stats")
        return check(label, False, err.strip()[:100])

    rows = stats_rows(out)
    if len(rows) < 2:
        return check(label, False, "no stats rows")

    lost = rows[0][1] - rows[-1][1]
    nburied, buried_mass, nreflect = rows[-1][3], rows[-1][4], rows[-1][5]
    material = rows[-1][2]

    ok = check(label + " : lost == buried", lost == nburied,
               "lost=%d buried=%d" % (lost, nburied))
    expect = nburied * MASS_N
    tol = max(1e-12 * max(expect, 1e-30), 1e-30)
    ok &= check(label + " : buried mass ledger", abs(buried_mass - expect) <= tol,
                "got=%g expect=%g" % (buried_mass, expect))
    # 101 x 101 corner points, 255 max each
    print("      (film reached %.0f%% of a solid box, %d reflections salvaged)"
          % (100.0 * material / (101 * 101 * 255), nreflect))
    return ok


def test_variable(exe):
    """fix ablate must be drivable by a grid-style variable.

    Checked against references the suite already trusts: a constant variable
    must reproduce the dedicated "uniform" source exactly, and a rate applied
    over half the domain must deposit twice what a quarter deposits.
    """
    def material(infile, variables):
        rc, out = run(exe, infile, variables)
        if rc != 0:
            return None
        rows = stats_rows(out)
        if len(rows) < 2:
            return None
        return rows[-1][2] - rows[0][2]

    ok = True

    const = material("in.test.variable", {"RATE_EXPR": "0.2"})
    ref = material("in.test.conserve", {"RATE": 0.2, "NEVERY": 1})
    ok &= check("variable source : constant matches the uniform source",
                const is not None and ref is not None and const == ref,
                "variable=%s uniform=%s" % (const, ref))

    half = material("in.test.variable", {"RATE_EXPR": "0.2*(cxlo<75)"})
    quarter = material("in.test.variable", {"RATE_EXPR": "0.2*(cxlo<50)"})
    if half is None or quarter is None:
        return check("variable source : rate is applied per cell", False,
                     "run failed")
    # cells 25-74 vs 25-49, so twice the width for twice the material
    rel = abs(half - 2.0 * quarter) / half
    ok &= check("variable source : rate is applied per cell", rel < 0.02,
                "half=%.0f quarter=%.0f, half/quarter=%.3f"
                % (half, quarter, half / quarter))
    return ok


def test_distance_units(exe):
    """A rate in length/time must not depend on how often the surface is rebuilt.

    Nevery chops the same physical growth into different sized pieces; it is
    not part of the rate.  Applying it both in set_delta's prefactor and again
    as the elapsed interval made a distance rate Nevery times too strong.
    """
    gains = {}
    for nevery in (1, 2, 5, 10, 20):
        rc, out = run(exe, "in.test.variable",
                      {"RATE_EXPR": 1.0, "NEVERY": nevery, "UNITS": "distance"})
        if rc != 0:
            err = next((l for l in out.splitlines() if "ERROR" in l), "no stats")
            return check("units distance : Nevery invariance", False,
                         "nevery=%d: %s" % (nevery, err.strip()[:70]))
        rows = stats_rows(out)
        if len(rows) < 2:
            return check("units distance : Nevery invariance", False,
                         "no stats rows")
        gains[nevery] = rows[-1][2] - rows[0][2]

    lo, hi = min(gains.values()), max(gains.values())
    spread = (hi - lo) / lo
    return check("units distance : Nevery invariance", spread < 0.01,
                 "gain %.0f..%.0f over Nevery 1..20, spread %.2f%%"
                 % (lo, hi, 100 * spread))


def test_rate_calibration(exe):
    """A rate given in length/time must be the rate the surface moves at.

    Not a fitted factor: the corner point increment is solved for from the
    field as it stands.  Raising both ends of a crossing edge slides the
    isosurface by the level set amount, but where the solid side is pinned at
    255 -- which is the whole of a binary 0/255 field, i.e. most real inputs
    -- only the gas side can rise and the crossing moves by half as much.
    front_response() accounts for that, so both geometries come out right.

    Checked on a flat front, where the answer is arithmetic, and on the curved
    blob the rest of the suite uses, where it is not.
    """
    ok = True

    for rate in (0.5, 1.0, 2.0, 4.0):
        rc, out = run(exe, "in.test.flat", {"RATE": rate})
        rows = stats_rows(out) if rc == 0 else []
        if not rows:
            ok &= check("rate calibration flat (s=%s)" % rate, False, "run failed")
            continue
        got = rows[-1][2]
        ok &= check("rate calibration, flat front (s=%s)" % rate,
                    abs(got - rate) / rate < 0.05,
                    "realized %.5f, ratio %.4f" % (got, got / rate))

    for rate in (1.0, 4.0):
        rc, out = run(exe, "in.test.variable",
                      {"RATE_EXPR": rate, "NEVERY": 1, "UNITS": "distance"})
        rows = stats_rows(out) if rc == 0 else []
        if not rows:
            ok &= check("rate calibration curved (s=%s)" % rate, False, "run failed")
            continue
        got = rows[-1][6]
        ok &= check("rate calibration, curved front (s=%s)" % rate,
                    abs(got - rate) / rate < 0.05,
                    "realized %.5f, ratio %.4f" % (got, got / rate))
    return ok


def test_stick(exe):
    """Flux driven deposition, against the closed form growth rate.

    A film that captures a fraction STICK of the impingement mass flux and has
    bulk density RHOFILM must grow at

        s = STICK * rho_gas * vbar / 4 / rho_film

    with no fitted constants anywhere.  One number checks mflux_incident, the
    norm flow normalization, the per cell area sum, the sticking weighting and
    the length/time conversion at once -- which is why it is worth the run
    time.

    Sticking is swept as well: the rate has to be linear in it, and zero at
    zero.  That separates a wrong capture weighting from a wrong flux.
    """
    import math
    k, T, nrho, rhofilm = 1.380649e-23, 300.0, 1.0e20, 2.0e3
    vbar = math.sqrt(8 * k * T / (math.pi * MASS_N))
    unit = nrho * MASS_N * vbar / 4.0 / rhofilm     # the rate at STICK = 1

    ok = True
    for stick in (0.0, 0.5, 1.0):
        rc, out = run(exe, "in.test.stick",
                      {"STICK": stick, "NRHO": nrho, "TEMP": T,
                       "RHOFILM": rhofilm})
        rows = stats_rows(out) if rc == 0 else []
        if not rows:
            err = next((l for l in out.splitlines() if "ERROR" in l), "no stats")
            ok &= check("flux source (stick %g)" % stick, False,
                        err.strip()[:90])
            continue

        # skip step 0, which is before the first regeneration
        got = sum(r[3] for r in rows[1:]) / (len(rows) - 1)
        want = stick * unit

        if stick == 0.0:
            ok &= check("flux source : no flux, no growth", got == 0.0,
                        "got %.4g" % got)
        else:
            # the flux is a Monte Carlo estimate over a finite window, so the
            # tolerance is set by the sampling rather than by the conversion
            ok &= check("flux source : s == stick*rho*vbar/4/rhofilm "
                        "(stick %g)" % stick,
                        abs(got - want) / want < 0.06,
                        "got %.4g want %.4g" % (got, want))
    return ok


def test_react(exe):
    """Reaction driven deposition, against the same closed form.

    A capture probability applied to the impingement flux is the same physics
    whether the probability lives in fix ablate or in a surf_react model, so
    this must land on

        s = P * rho_gas * vbar / 4 / rho_film

    with P read from stick.surf.  What it exercises that test_stick does not
    is the mass weighting of compute react/isurf/grid, which is what makes
    the rate available for any surf_react model -- and, with the sign the
    other way, for ablation.

    The reaction captures molecules and the gas is not replenished, so the
    rate drifts down with the density.  Np is in the stats table, so the
    expectation is corrected by it rather than fudged into the tolerance.
    """
    import math
    k, T, nrho, rhofilm, prob = 1.380649e-23, 300.0, 1.0e20, 2.0e3, 0.4
    vbar = math.sqrt(8 * k * T / (math.pi * MASS_N))
    want0 = prob * nrho * MASS_N * vbar / 4.0 / rhofilm

    rc, out = run(exe, "in.test.react",
                  {"NRHO": nrho, "TEMP": T, "RHOFILM": rhofilm})
    rows = stats_rows(out) if rc == 0 else []
    if not rows:
        err = next((l for l in out.splitlines() if "ERROR" in l), "no stats")
        return check("reaction source", False, err.strip()[:90])

    np0 = rows[0][1]
    got = sum(r[3] for r in rows[1:]) / (len(rows) - 1)
    # each window sees the density it was sampled at
    want = sum(want0 * r[1] / np0 for r in rows[1:]) / (len(rows) - 1)

    ok = check("reaction source : particles were captured",
               rows[-1][1] < np0, "np %g -> %g" % (np0, rows[-1][1]))
    ok &= check("reaction source : s == P*rho*vbar/4/rhofilm",
                abs(got - want) / want < 0.07,
                "got %.4g want %.4g" % (got, want))
    return ok


def test_equal_variable(exe):
    """An equal-style variable source.

    A rate that depends only on time is one number for the whole surface, not
    a field, and writing it as a grid-style variable is busywork.  On a
    uniform rate the two paths have to agree exactly -- if they do not, the
    equal-style value is not reaching every cell in the group.
    """
    rows = {}
    for style in ("grid", "equal"):
        rc, out = run(exe, "in.test.flat", {"RATE": 2.0, "STYLE": style})
        r = stats_rows(out) if rc == 0 else []
        if not r:
            err = next((l for l in out.splitlines() if "ERROR" in l), "no stats")
            return check("equal-style variable source", False, err.strip()[:90])
        rows[style] = r
    return check("equal-style variable source : same as grid-style",
                 rows["grid"] == rows["equal"],
                 "material %g vs %g" % (rows["grid"][-1][1], rows["equal"][-1][1]))


def test_flux(exe):
    """The incident flux keywords, against the closed form.

    For a Maxwellian at rest against a wall at the same temperature the
    one-sided impingement rate is n*vbar/4 with vbar = sqrt(8kT/pi m), so
    there is nothing to compare against but arithmetic.

    compute isurf/grid had only NET mflux, which is identically zero at a wall
    that does not react.  nflux_incident and mflux_incident are what an
    impingement driven rate is built from.
    """
    import math
    k, T, nrho = 1.380649e-23, 300.0, 1.0e20
    vbar = math.sqrt(8 * k * T / (math.pi * MASS_N))
    want_n = nrho * vbar / 4.0
    want_m = nrho * MASS_N * vbar / 4.0

    rc, out = run(exe, "in.test.flux", {"NRHO": nrho, "TEMP": T})
    rows = stats_rows(out) if rc == 0 else []
    if not rows:
        err = next((l for l in out.splitlines() if "ERROR" in l), "no stats")
        return check("incident flux", False, err.strip()[:90])

    # average the sampled windows, skipping step 0 which has none
    got_n = sum(r[2] for r in rows[1:]) / (len(rows) - 1)
    got_m = sum(r[3] for r in rows[1:]) / (len(rows) - 1)

    ok = check("incident flux : nflux_incident == n*vbar/4",
               abs(got_n - want_n) / want_n < 0.05,
               "got %.4g want %.4g" % (got_n, want_n))
    ok &= check("incident flux : mflux_incident == rho*vbar/4",
                abs(got_m - want_m) / want_m < 0.05,
                "got %.4g want %.4g" % (got_m, want_m))
    # every incident molecule of one species carries exactly that mass, so
    # this holds however poor the statistics are; the tolerance is the width
    # of the printed stats field, not a physical one
    ok &= check("incident flux : mflux/nflux == species mass",
                abs(got_m / got_n - MASS_N) / MASS_N < 1e-6,
                "got %.6g want %.6g" % (got_m / got_n, MASS_N))
    return ok


def test_grow(exe):
    """A growing film must be able to leave the block it was read into.

    Corner point values exist only on the fix's grid group, so that group is
    the room the surface has.  For ablation the block the surface file
    describes is always enough; for deposition it is not, so fix ablate may be
    defined on a larger group than read_isurf reads into.

    Same input twice at a rate that decides it: with the fix on the read block
    the surface reaches its edge and fix ablate says so, and with the fix on
    the whole grid it grows on out of the block.
    """
    ok = True

    rc, out = run(exe, "in.test.conserve", {"RATE": 0.65, "NEVERY": 1})
    saw = "grown the surface out to the edge" in out
    ok &= check("fix group = read block : surface stops at its edge",
                rc != 0 and saw,
                "rc=%d" % rc if not saw else "")

    rc, out = run(exe, "in.test.grow", {})
    if rc != 0:
        err = next((l for l in out.splitlines() if "ERROR" in l), "no stats")
        return check("fix group = whole grid : surface grows past it", False,
                     err.strip()[:100])
    rows = stats_rows(out)
    if len(rows) < 2:
        return check("fix group = whole grid : surface grows past it", False,
                     "no stats rows")
    lost = rows[0][1] - rows[-1][1]
    nburied, buried_mass, nreflect = rows[-1][3], rows[-1][4], rows[-1][5]

    ok &= check("fix group = whole grid : surface grows past it", True,
                "ran to completion, material %g vs %g at the start"
                % (rows[-1][2], rows[0][2]))
    ok &= check("fix group = whole grid : lost == buried", lost == nburied,
                "lost=%d buried=%d" % (lost, nburied))
    expect = nburied * MASS_N
    tol = max(1e-12 * max(expect, 1e-30), 1e-30)
    ok &= check("fix group = whole grid : buried mass ledger",
                abs(buried_mass - expect) <= tol,
                "got=%g expect=%g" % (buried_mass, expect))
    print("      (%d reflections salvaged)" % nreflect)
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


def test_crossproc(exe):
    """A molecule must not be buried just because the gas next door is off-proc.

    salvage_to_neighbor can only push a molecule into a cell this proc owns.
    Left at that, one whose only gas neighbor happens to sit on another proc
    is buried, so the burial count grows with the number of procs -- an
    artifact of how the grid was divided up, not physics.  Those molecules are
    now handed to the owner, which decides.

    Asserts the mechanism is live and inert in serial, and that the burial
    count no longer walks away from the serial answer as ranks are added.
    Before this, in.test.fill went 10574 / 10592 / 10649 / 10753 / 10891 at
    1 / 2 / 4 / 8 / 16 ranks -- monotone, and still climbing at 16.
    """
    ok = True

    rc, out = run(exe, "in.test.fill", {"RATE": 1.0}, ranks=1)
    if rc != 0:
        return check("cross-proc salvage", False, "serial run failed")
    rows = stats_rows(out)
    serial_buried, serial_migrated = rows[-1][3], rows[-1][6]

    ok &= check("cross-proc salvage : inert on one rank", serial_migrated == 0,
                "%d pushed across a boundary" % serial_migrated)

    if RANKS < 2:
        print("      (run with --ranks 2 or more to exercise the rest)")
        return ok

    rc, out = run(exe, "in.test.fill", {"RATE": 1.0})
    if rc != 0:
        return check("cross-proc salvage", False, "parallel run failed")
    rows = stats_rows(out)
    par_buried, par_migrated = rows[-1][3], rows[-1][6]

    ok &= check("cross-proc salvage : live on %d ranks" % RANKS,
                par_migrated > 0,
                "%d molecules handed to another proc" % par_migrated)

    # the intrinsic seed-to-seed spread of this count is about 2%, measured by
    # varying the seed in serial, so anything inside that is noise
    rel = abs(par_buried - serial_buried) / serial_buried
    ok &= check("cross-proc salvage : burials match the serial answer",
                rel < 0.02,
                "%d on %d ranks vs %d on one, %.2f%%"
                % (par_buried, RANKS, serial_buried, 100 * rel))
    return ok


def test_balance(exe):
    """Repartitioning the grid must carry everything fix ablate keeps per cell.

    fix balance moves cells between procs while the film grows.  The snapshot
    of the corner point field that the front speed and the per-step refreshed
    collision geometry are measured against has to move with its cell; left
    behind, each cell is paired with another cell's history.

    Also the only test that exercises pack_grid_one / unpack_grid_one at all,
    where a byte-count mismatch would corrupt every per-cell array at once.
    """
    label = "grid repartitioned under a growing film"
    rc, out = run(exe, "in.test.balance")
    if rc != 0:
        err = next((l for l in out.splitlines() if "ERROR" in l), "no stats")
        return check(label, False, err.strip()[:100])

    rows = stats_rows(out)
    if len(rows) < 2:
        return check(label, False, "no stats rows")

    lost = rows[0][1] - rows[-1][1]
    nburied, buried_mass, nreflect = rows[-1][3], rows[-1][4], rows[-1][5]

    ok = check(label + " : lost == buried", lost == nburied,
               "lost=%d buried=%d" % (lost, nburied))
    expect = nburied * MASS_N
    tol = max(1e-12 * max(expect, 1e-30), 1e-30)
    ok &= check(label + " : buried mass ledger",
                abs(buried_mass - expect) <= tol,
                "got=%g expect=%g" % (buried_mass, expect))
    print("      (ran under fix grid/check error, %d reflections salvaged)"
          % nreflect)
    return ok


def test_energy(exe, rate, nevery):
    """In a periodic box the surface is the only place gas energy can go.

    The companion to test_momentum.  E_gas is translational plus rotational
    plus vibrational and the gas moves energy between those reservoirs on its
    own, so this only closes if every mode is accounted for at the surface --
    including for the molecules the film buries.
    """
    name = "energy (rate=%s nevery=%s)" % (rate, nevery)
    rc, out = run(exe, "in.test.energy", {"RATE": rate, "NEVERY": nevery})
    if rc != 0:
        err = next((l for l in out.splitlines() if "ERROR" in l), "no stats")
        return check(name, False, err.strip()[:90])

    rows = stats_rows(out)
    if len(rows) < 2:
        return check(name, False, "no stats rows")
    f, l = rows[0], rows[-1]

    # cols: 0 step, 1 np, 2 ke, 3 erot, 4 evib, 5 surf, 6 reflect,
    #       7 buried ke, 8 buried erot, 9 buried evib, 10 nburied
    egas0 = f[2] + f[3] + f[4]
    egasN = l[2] + l[3] + l[4]
    ebur = (l[7] - f[7]) + (l[8] - f[8]) + (l[9] - f[9])
    total = (egasN - egas0) + (l[5] - f[5]) + (l[6] - f[6]) + ebur

    rel = abs(total) / max(abs(egas0), abs(egasN))
    return check(name, rel < 1e-5,
                 "relative residual %.2e, %d buried" % (rel, l[10]))


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


AXIDIR = os.path.join(HERE, "../../../examples/explicit2implicit")


def test_axisymmetric(exe):
    """Deposition onto an axisymmetric implicit surface.

    Axisymmetric runs use the refreshed geometry but solve the intersection
    with the surface-of-revolution test, because the particle path there is an
    arc in the r-z plane rather than a straight line.  That test resolves a
    re-hit of the element the particle just bounced off itself, so unlike 2d
    and 3d that element must NOT be skipped -- getting this wrong lets a
    molecule walk through the surface, which is what this catches.

    The example runs under fix grid/check with the error setting, so
    completing at all is the assertion.
    """
    label = "axisymmetric deposition"
    rc, out = run(exe, "in.deposit.axi.spherecone", cwd=AXIDIR)
    if rc != 0:
        err = next((l for l in out.splitlines() if "ERROR" in l), "run failed")
        return check(label, False, err.strip()[:110])

    rows = stats_rows(out)
    if len(rows) < 2:
        return check(label, False, "no stats rows")

    # cols: 0 step, 1 cpu, 2 np, 3 nscoll, 4 nscheck, 5 material, 6 nburied,
    #       7 nfrontreflect
    return check(label, True,
                 "ran clean under fix grid/check, %d buried, %d salvaged"
                 % (rows[-1][6], rows[-1][7]))


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
    # the top of this sweep is deliberately close to the point where marching
    # squares itself gives up: above about 0.65 per step SPARTA's own
    # watertight check fails, in ablate mode as well as deposit
    for rate in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6):
        ok &= test_conserve(exe, rate, 1)
    # the same physics with the isosurface regenerated less often
    for nevery in (2, 5):
        ok &= test_conserve(exe, 0.05 * nevery, nevery)
    # the surface must be drivable by a variable, not just a compute or fix
    ok &= test_variable(exe)
    # the incident flux keywords, against the free molecular closed form
    ok &= test_flux(exe)
    ok &= test_stick(exe)
    ok &= test_react(exe)
    ok &= test_equal_variable(exe)
    # a rate in length/time must be the rate the surface actually moves at
    ok &= test_rate_calibration(exe)
    # a rate in length/time must not depend on the rebuild interval
    ok &= test_distance_units(exe)
    # a growing film must be able to leave the block it was read into
    ok &= test_grow(exe)
    # with the corner point grid on the whole box the film can grow until it
    # runs out of box, which is legal and has to stay accounted for
    ok &= test_fill(exe, 1.0)
    # momentum: gas + surface + buried + reflected must balance
    for rate, nevery in ((0.2, 1), (0.5, 5), (1.0, 20)):
        ok &= test_momentum(exe, rate, nevery)
    # the same ledger in 3d, which uses marching cubes and the triangle
    # intersection rather than marching squares and the line one
    ok &= test_momentum(exe, 0.5, 5, "in.test.momentum.3d", label=" 3d")
    # energy: the same ledger for translational + rotational + vibrational,
    # at a rate that buries nothing and one that buries plenty
    for rate, nevery in ((0.2, 1), (0.6, 1)):
        ok &= test_energy(exe, rate, nevery)
    # a molecule must not be buried because the gas next door is off-proc
    ok &= test_crossproc(exe)
    # everything fix ablate keeps per cell must move when the cell does
    ok &= test_balance(exe)
    # axisymmetry: the refreshed geometry with the surface-of-revolution test
    ok &= test_axisymmetric(exe)
    # the front velocity places the collision but must not enter the rebound
    ok &= test_no_wall_work(exe)
    ok &= test_guard(exe)
    ok &= test_ablate_unaffected(exe)

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
