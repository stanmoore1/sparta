#!/usr/bin/env python3
"""Regression tests for SPARTA rigid-body surface objects (fix rigid).

Runs a set of small input decks and checks the results against analytic
expectations with tolerances.  Deterministic tests (no particles) must
produce identical results on any number of procs; statistical tests use
loose tolerances.

Usage:
  python3 run_tests.py --exe /path/to/spa_serial
  python3 run_tests.py --exe /path/to/spa_mpi --mpi "mpirun -np 4"
  python3 run_tests.py --exe /path/to/spa_kokkos --args "-k on -sf kk"

Exit code = number of failed tests.
"""

import argparse
import glob
import math
import os
import shlex
import subprocess
import sys

THISDIR = os.path.dirname(os.path.abspath(__file__))

# mass of N in air.species, fnum from the decks

MASS_N = 2.325e-26
FNUM = 0.001


def run_deck(exe_cmd, deck, extra=None):
    """Run one deck, return (returncode, stdout+stderr)."""
    cmd = exe_cmd + ["-in", deck] + (extra or [])
    try:
        proc = subprocess.run(cmd, cwd=THISDIR, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              timeout=600)
    except subprocess.TimeoutExpired as e:
        # a hang (e.g. a collective entered by only some ranks) is a
        # failure, not a reason to stall the suite
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else \
            (e.stdout or "")
        return -1, out + "\nTIMEOUT after 600 s\n"
    return proc.returncode, proc.stdout


def parse_stats(output):
    """Parse the last stats table in the output into a list of dicts."""
    header = None
    rows = []
    for line in output.splitlines():
        toks = line.split()
        if not toks:
            continue
        if toks[0] == "Step":
            header = toks
            rows = []
            continue
        if header is None:
            continue
        if toks[0].startswith("Loop"):
            header = None
            continue
        try:
            vals = [float(t) for t in toks]
        except ValueError:
            continue
        if len(vals) == len(header):
            rows.append(dict(zip(header, vals)))
    return rows


def approx(a, b, rel=0.0, abs_=0.0):
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_)


# ----------------------------------------------------------------------
# individual tests: each returns a list of failure strings (empty = pass)
# ----------------------------------------------------------------------

def test_ballistic(exe_cmd):
    rc, out = run_deck(exe_cmd, "in.test.ballistic")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if not rows:
        return ["no stats output"]
    last = rows[-1]
    fails = []
    # tolerances limited by the ~8 significant digits of stats output
    # com = com0 + v * t,  t = 1000 * 1e-4 = 0.1
    if not approx(last["f_1[1]"], 3.0 + 12.0 * 0.1, rel=1e-7):
        fails.append("xcm = %.12g, expected 4.2" % last["f_1[1]"])
    if not approx(last["f_1[2]"], 4.0 + 7.0 * 0.1, rel=1e-7):
        fails.append("ycm = %.12g, expected 4.7" % last["f_1[2]"])
    if not approx(last["f_1[4]"], 12.0, rel=1e-7):
        fails.append("vx = %.12g, expected 12" % last["f_1[4]"])
    if not approx(last["f_1[5]"], 7.0, rel=1e-7):
        fails.append("vy = %.12g, expected 7" % last["f_1[5]"])
    # omega_z = Lz / Izz = 1e-22 / 1.6666667e-23
    womega = 1.0e-22 / 1.6666667e-23
    if not approx(last["f_1[15]"], womega, rel=1e-7):
        fails.append("omega = %.12g, expected %.12g" % (last["f_1[15]"], womega))
    return fails


def test_rotation(exe_cmd):
    # torque-free tumbling of an asymmetric body: angular momentum is
    # exactly conserved, so the rotational kinetic energy T = L.w/2 can
    # be formed from the reported angular velocity and the known L
    Lx = Ly = Lz = 1.0e-23
    fails = []
    drift = {}
    for rot in ("euler", "richardson"):
        rc, out = run_deck(exe_cmd, "in.test.rotation",
                           extra=["-var", "rot", rot])
        if rc:
            fails.append("%s: run failed with exit code %d" % (rot, rc))
            continue
        rows = parse_stats(out)
        if not rows:
            fails.append("%s: no stats output" % rot)
            continue
        ke = [0.5 * (Lx * r["f_1[13]"] + Ly * r["f_1[14]"] + Lz * r["f_1[15]"])
              for r in rows]
        ke0 = ke[0]
        drift[rot] = max(abs(k - ke0) for k in ke) / abs(ke0)

        # the body must actually tumble: for an asymmetric free body the
        # angular velocity is not constant.  a constant w would mean the
        # rotational dynamics are not being integrated at all
        wx = [r["f_1[13]"] for r in rows]
        if max(wx) - min(wx) < 0.01 * abs(max(wx)):
            fails.append("%s: angular velocity is nearly constant; an "
                         "asymmetric torque-free body must tumble" % rot)
    if fails:
        return fails

    # euler is first order, richardson second: at this timestep the
    # energy drift should differ by orders of magnitude
    if drift["euler"] > 1.0e-2:
        fails.append("euler: rotational energy drift %.3e is too large"
                     % drift["euler"])
    if drift["richardson"] > 1.0e-5:
        fails.append("richardson: rotational energy drift %.3e, expected "
                     "second-order accuracy" % drift["richardson"])
    if drift["richardson"] > drift["euler"]:
        fails.append("richardson drift %.3e exceeds euler %.3e; the "
                     "higher-order scheme is not better"
                     % (drift["richardson"], drift["euler"]))
    return fails


def test_force(exe_cmd):
    # constant external force on the COM: velocity Verlet is exact,
    # x = x0 + v0*t + a*t^2/2, v = v0 + a*t
    fx = 1.0e-21
    rc, out = run_deck(exe_cmd, "in.test.ballistic",
                       extra=["-var", "fx", repr(fx)])
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if not rows:
        return ["no stats output"]
    last = rows[-1]
    fails = []
    a = fx / 1.0e-22
    dt = 1.0e-4
    n = 1000
    t = n * dt
    xexp = 3.0 + 12.0 * t + 0.5 * a * t * t
    vexp = 12.0 + a * t
    if not approx(last["f_1[1]"], xexp, rel=1e-7):
        fails.append("xcm = %.12g, expected %.12g" % (last["f_1[1]"], xexp))
    if not approx(last["f_1[4]"], vexp, rel=1e-7):
        fails.append("vx = %.12g, expected %.12g" % (last["f_1[4]"], vexp))
    if not approx(last["f_1[2]"], 4.7, rel=1e-7):
        fails.append("ycm = %.12g, expected 4.7 (force is x only)"
                     % last["f_1[2]"])
    return fails


def test_bounce(exe_cmd):
    fails = []
    # elastic cases: both force laws must rebound at -50 within 1%
    # damped case: DEM spring-dashpot must rebound slower, with a
    #   coefficient of restitution in a loose band around the analytic
    #   value for the effective 2-contact linear spring-dashpot
    cases = (
        ("linear", "1.0e-18", "0.0", (-50.5, -49.5)),
        ("hertz", "4.0e-18", "0.0", (-50.5, -49.5)),
        ("linear-damped", "1.0e-18", "3.0e-21", (-32.5, -17.5)),
    )
    for label, pk, pdamp, vband in cases:
        pstyle = label.split("-")[0]
        rc, out = run_deck(exe_cmd, "in.test.bounce",
                           extra=["-var", "pstyle", pstyle,
                                  "-var", "pk", pk,
                                  "-var", "pdamp", pdamp])
        if rc:
            fails.append("%s: run failed with exit code %d" % (label, rc))
            continue
        rows = parse_stats(out)
        if not rows:
            fails.append("%s: no stats output" % label)
            continue
        # force engages when body corner is cutoff=0.5 from wall at x=8,
        # i.e. xcm <= 7.5; no tunneling means xcm never reaches the wall
        xmax = max(r["f_1[1]"] for r in rows)
        if xmax > 7.55:
            fails.append("%s: max xcm = %.6g, tunneled past push-off zone"
                         % (label, xmax))
        vfinal = rows[-1]["f_1[4]"]
        if not vband[0] <= vfinal <= vband[1]:
            fails.append("%s: final vx = %.6g, expected within [%g,%g]"
                         % (label, vfinal, vband[0], vband[1]))
    return fails


def test_restitution(exe_cmd):
    # analytic properties of the push-off contact model, measured as the
    # rebound speed of a body launched at a static wall with no gas
    def bounce(pstyle, pk, pdamp, v0):
        rc, out = run_deck(exe_cmd, "in.test.restitution",
                           extra=["-var", "pstyle", pstyle, "-var", "pk", pk,
                                  "-var", "pdamp", pdamp, "-var", "v0",
                                  repr(v0)])
        if rc:
            return None
        rows = parse_stats(out)
        if not rows:
            return None
        vout = rows[-1]["f_1[4]"]
        if vout >= 0.0:          # never rebounded: passed through the wall
            return None
        return abs(vout) / v0

    fails = []

    # elastic contact conserves energy exactly: e = 1 for both force laws
    for pstyle, pk in (("linear", "1.0e-18"), ("hertz", "4.0e-18")):
        for v0 in (10.0, 25.0):
            e = bounce(pstyle, pk, "0.0", v0)
            if e is None:
                fails.append("%s elastic v0=%g: no rebound" % (pstyle, v0))
            elif not approx(e, 1.0, abs_=1e-3):
                fails.append("%s elastic v0=%g: restitution %.6f, expected 1"
                             % (pstyle, v0, e))

    # the wall is 0.2 thick, less than 2*cutoff, so a corner pt which
    # crosses the near face is within range of the far face too; contact
    # is one-sided, so the far face must not push the body on through
    # (v0=90 is within the 4-contact capacity of the linear spring)
    e = bounce("linear", "1.0e-18", "0.0", 90.0)
    if e is None:
        fails.append("linear elastic v0=90: no rebound, thin wall pushed "
                     "the body through instead of repelling it")
    elif not approx(e, 1.0, abs_=1e-3):
        fails.append("linear elastic v0=90: restitution %.6f, expected 1" % e)

    # a linear spring-dashpot has a restitution independent of impact
    # speed; this is the property which distinguishes it from Hertzian
    lin = [bounce("linear", "1.0e-18", "3.0e-21", v0) for v0 in (10.0, 25.0)]
    if None in lin:
        fails.append("linear damped: no rebound")
    elif not approx(lin[0], lin[1], rel=1e-4):
        fails.append("linear damped: restitution %.6f at v0=10 vs %.6f at "
                     "v0=25, should not depend on impact speed"
                     % (lin[0], lin[1]))

    # a Hertzian spring-dashpot restitution does depend on impact speed
    her = [bounce("hertz", "4.0e-18", "3.0e-21", v0) for v0 in (10.0, 25.0)]
    if None in her:
        fails.append("hertz damped: no rebound")
    elif approx(her[0], her[1], rel=1e-3):
        fails.append("hertz damped: restitution %.6f at v0=10 and %.6f at "
                     "v0=25 are equal; a Hertzian contact must depend on "
                     "impact speed" % (her[0], her[1]))

    # damping must actually remove energy, else the checks above are vacuous
    if lin[0] is not None and lin[0] > 0.9:
        fails.append("linear damped restitution %.4f is too close to elastic "
                     "for the test to be meaningful" % lin[0])
    return fails


def test_momentum(exe_cmd):
    rc, out = run_deck(exe_cmd, "in.test.momentum")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if not rows:
        return ["no stats output"]
    mass_body = 1.0e-22

    dt = 1.0e-4                      # timestep of the deck

    def gas_plus_body(r):
        return MASS_N * FNUM * r["c_r"] + mass_body * r["f_1[4]"]

    # compute surf tallies the momentum the gas gave up on a step; with
    # velocity Verlet the body receives half of it at the end of that step
    # and half at the start of the next, so half a step's impulse
    # 0.5*dt*fcm is always in flight.  Including it makes the invariant
    # exact rather than accurate to the statistical size of one step's
    # transfer.

    def total_px(r):
        return gas_plus_body(r) + 0.5 * dt * r["f_1[7]"]

    p0 = total_px(rows[0])
    fails = []
    for r in rows:
        if not approx(total_px(r), p0, rel=1.0e-11):
            fails.append("step %d: lag-corrected total px = %.15g vs initial "
                         "%.15g" % (int(r["Step"]), total_px(r), p0))
            break

    # the uncorrected sum should drift by roughly one step's transfer and
    # no more: a bounded offset, not a leak
    raw = [abs(gas_plus_body(r) - gas_plus_body(rows[0])) for r in rows]
    if max(raw) > 0.05 * abs(gas_plus_body(rows[0])):
        fails.append("uncorrected momentum drifted by %.3g, far more than "
                     "one step's impulse; this looks like a leak"
                     % (max(raw) / abs(gas_plus_body(rows[0]))))
    # body must have absorbed a significant momentum fraction by the end
    pbody = mass_body * rows[-1]["f_1[4]"]
    if pbody < 0.25 * p0:
        fails.append("body momentum %.3g < 25%% of gas momentum %.3g, "
                     "coupling too weak" % (pbody, p0))
    return fails


def test_overrun(exe_cmd):
    rc, out = run_deck(exe_cmd, "in.test.overrun")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if not rows:
        return ["no stats output"]
    fails = []
    # f_1 (scalar) = cumulative particles deleted inside the body
    # the body sweeps 40% of a grid cell per step through nearly
    # stationary gas.  swept collision coverage adds the body surfs to
    # the collision lists of every cell they sweep into during the step,
    # so no particle is overtaken undetected: every particle in the path
    # is reflected off the moving surf rather than deleted.  the deletion
    # count must therefore not grow at all after the initial setup.  any
    # growth means a particle tunneled into the body and was deleted.
    ndel = rows[-1]["f_1"] - rows[0]["f_1"]
    nptotal = rows[0]["Np"]
    if ndel != 0:
        fails.append("deleted %g of %g particles after setup: a particle "
                     "was overtaken by the moving body instead of being "
                     "reflected (swept collision coverage failed)"
                     % (ndel, nptotal))
    return fails


def compare_remap_modes(exe_cmd, deck, keys, labels):
    """Run deck in both remap modes, require identical final values."""
    results = {}
    fails = []
    for mode in ("cutcell", "incremental"):
        rc, out = run_deck(exe_cmd, deck, extra=["-var", "mode", mode])
        if rc:
            fails.append("mode %s: run failed with exit code %d" % (mode, rc))
            continue
        rows = parse_stats(out)
        if not rows:
            fails.append("mode %s: no stats output" % mode)
            continue
        results[mode] = rows
    if fails:
        return fails, results
    last_c = results["cutcell"][-1]
    last_i = results["incremental"][-1]
    for key, name in zip(keys, labels):
        if not approx(last_i[key], last_c[key], rel=1e-10, abs_=1e-13):
            fails.append("%s: incremental %.15g differs from cutcell %.15g"
                         % (name, last_i[key], last_c[key]))
    return fails, results


def test_remap(exe_cmd):
    # one gas-driven body: cutcell and incremental must give identical
    # trajectories, verifying the incremental re-cut against the full
    # rebuild; the body must actually have moved
    fails, results = compare_remap_modes(
        exe_cmd, "in.test.remap",
        ("f_1[1]", "f_1[2]", "f_1[15]"), ("xcm", "ycm", "omega"))
    if fails:
        return fails
    if abs(results["cutcell"][-1]["f_1[1]"] - 5.0) < 0.01:
        fails.append("body barely moved (xcm = %.6g), test is too weak"
                     % results["cutcell"][-1]["f_1[1]"])
    return fails


def test_staticdist(exe_cmd):
    # body next to a 200-segment static circle: the incremental re-cut
    # must handle cells holding many static surfs; with --dist on several
    # procs most static surfs are ghost surfs on any one proc, exercising
    # the mover's ghost-cell collision tests and the local body copies
    fails, results = compare_remap_modes(
        exe_cmd, "in.test.staticdist",
        ("f_1[1]", "f_1[2]", "f_1[15]"), ("xcm", "ycm", "omega"))
    if fails:
        return fails
    for mode in ("cutcell", "incremental"):
        ndel = results[mode][-1]["f_1"] - results[mode][0]["f_1"]
        if ndel != 0:
            fails.append("mode %s: %g particles deleted inside the body "
                         "during the run" % (mode, ndel))
    # the gas must actually have moved the body (it drifts ~0.006 in x
    # over the run), else the two modes agree trivially
    if abs(results["cutcell"][-1]["f_1[1]"] - 6.0) < 0.001:
        fails.append("body barely moved (xcm = %.6g), test is too weak"
                     % results["cutcell"][-1]["f_1[1]"])
    return fails


def test_staticdist3d(exe_cmd):
    # 3d: cube body drifting past a 1200-triangle static sphere, both
    # remap modes must agree and no particle may be deleted; with --dist
    # on several procs most sphere triangles are ghost surfs
    fails, results = compare_remap_modes(
        exe_cmd, "in.test.staticdist3d",
        ("f_1[1]", "f_1[2]", "f_1[3]", "f_1[13]", "f_1[14]", "f_1[15]"),
        ("xcm", "ycm", "zcm", "wx", "wy", "wz"))
    if fails:
        return fails
    for mode in ("cutcell", "incremental"):
        ndel = results[mode][-1]["f_1"] - results[mode][0]["f_1"]
        if ndel != 0:
            fails.append("mode %s: %g particles deleted inside the body "
                         "during the run" % (mode, ndel))
    if abs(results["cutcell"][-1]["f_1[1]"] - 4.0) < 0.001:
        fails.append("body barely moved (xcm = %.6g), test is too weak"
                     % results["cutcell"][-1]["f_1[1]"])
    return fails


def test_restart(exe_cmd):
    # restart continuation: a deterministic push-off run split across a
    # write_restart/read_restart must reproduce the one-shot trajectory.
    # the split points straddle the contact with the wall, which is the
    # sensitive case: the body moves on a step under the force and torque
    # accumulated on the previous one, so a continuation which resumed
    # with zero force would lose that impulse.  two bodies, each with its
    # own outfile, so that the state written for a body other than the
    # last-defined one is also checked to be the complete end-of-step state
    total = 1500
    fails = []
    rc, out = run_deck(exe_cmd, "in.test.restart.oneshot",
                       extra=["-var", "nrun", str(total)])
    if rc:
        return ["one-shot run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if not rows:
        return ["one-shot run produced no stats output"]
    ref = rows[-1]

    # the body must actually reach the wall and rebound, else the test
    # never exercises a restart under load

    if ref["f_1[1][4]"] > -40.0:
        return ["one-shot final vx = %.6g, body did not rebound; test "
                "geometry is broken" % ref["f_1[1][4]"]]

    # 480 falls inside body 1's contact with the wall (steps ~405-565),
    # so its stored force and torque are nonzero at that split
    for split in (300, 480, 700, 1100):
        rc, out1 = run_deck(exe_cmd, "in.test.restart.part1",
                            extra=["-var", "nrun", str(split)])
        if rc:
            fails.append("split %d: first half failed with exit code %d"
                         % (split, rc))
            continue
        rows1 = parse_stats(out1)
        rc, out2 = run_deck(exe_cmd, "in.test.restart.part2",
                            extra=["-var", "nrun", str(total - split)])
        if rc:
            fails.append("split %d: continuation failed with exit code %d"
                         % (split, rc))
            continue
        rows2 = parse_stats(out2)
        if not rows1 or not rows2:
            fails.append("split %d: a half produced no stats output"
                         % split)
            continue
        keys = (("f_1[1][1]", "xcm"), ("f_1[1][4]", "vx"), ("f_1[1][15]", "omega"),
                ("f_1[2][1]", "xcm2"), ("f_1[2][4]", "vx2"), ("f_1[2][15]", "omega2"))

        # the state read back from the outfile at the start of the
        # continuation must equal the state at the end of the first half
        # exactly: the outfile stores 17 digits, which round-trip a double
        # (15 digits, the previous format, lost the last few ulp)

        first = rows2[0]
        end1 = rows1[-1]
        for key, name in keys:
            if name.startswith("omega"):
                # omega is not stored: it is re-derived from the angular
                # momentum through the inertia eigensolver, to round-off
                if not approx(first[key], end1[key], rel=1e-12, abs_=1e-300):
                    fails.append("split %d: %s re-derived from the outfile "
                                 "as %.17g, was %.17g"
                                 % (split, name, first[key], end1[key]))
            elif first[key] != end1[key]:
                fails.append("split %d: %s read back from the outfile as "
                             "%.17g, written from %.17g"
                             % (split, name, first[key], end1[key]))

        # the rest of the continuation re-derives the body-frame geometry
        # from the restarted surfs, so it tracks the one-shot run to
        # round-off rather than exactly

        last = rows2[-1]
        for key, name in keys:
            if not approx(last[key], ref[key], rel=1e-12, abs_=1e-300):
                fails.append("split %d: %s = %.12g differs from one-shot "
                             "%.12g" % (split, name, last[key], ref[key]))

    for f in glob.glob(os.path.join(THISDIR, "tmp.rigid*")):
        os.remove(f)
    return fails


def test_gridchange(exe_cmd):
    # the grid changing underneath the body must not perturb it: a
    # no-particle push-off trajectory is identical with and without
    # fix balance (random style, full rebuild every 25 steps) and
    # fix adapt (refine/coarsen on the body surfs every 100 steps)
    results = {}
    fails = []
    for pert in ("none", "balance", "balancecell", "adapt"):
        rc, out = run_deck(exe_cmd, "in.test.gridchange",
                           extra=["-var", "pert", pert])
        if rc:
            fails.append("pert %s: run failed with exit code %d" % (pert, rc))
            continue
        rows = parse_stats(out)
        if not rows:
            fails.append("pert %s: no stats output" % pert)
            continue
        results[pert] = rows
    if fails:
        return fails
    ref = results["none"]
    for pert in ("balance", "balancecell", "adapt"):
        if len(results[pert]) != len(ref):
            fails.append("pert %s: %d stats rows vs %d unperturbed"
                         % (pert, len(results[pert]), len(ref)))
            continue
        for r, r0 in zip(results[pert], ref):
            for key in ("f_1[1]", "f_1[4]", "f_1[15]", "f_1[20]"):
                if not approx(r[key], r0[key], rel=1e-12, abs_=1e-30):
                    fails.append("pert %s, step %d: %s = %.15g differs from "
                                 "unperturbed %.15g"
                                 % (pert, int(r["Step"]), key, r[key], r0[key]))
                    break
            if fails:
                break
    # the body must actually have bounced, else the test is vacuous
    if ref[-1]["f_1[4]"] > -40.0:
        fails.append("final vx = %.6g, body did not rebound; test geometry "
                     "is broken" % ref[-1]["f_1[4]"])
    return fails


def test_splitcell(exe_cmd):
    # body sweeping alongside a diagonal wall that creates split cells:
    # particles entering swept split cells must be reflected, not
    # overrun, so the deletion count must not grow; both remap modes
    # must agree (incremental falls back to a full re-map near split
    # cells, which must not change the result)
    fails, results = compare_remap_modes(
        exe_cmd, "in.test.splitcell",
        ("f_1[1]", "f_1[2]"), ("xcm", "ycm"))
    if fails:
        return fails
    for mode in ("cutcell", "incremental"):
        ndel = results[mode][-1]["f_1"] - results[mode][0]["f_1"]
        if ndel != 0:
            fails.append("mode %s: %g particles overrun by the body in "
                         "split cells" % (mode, ndel))
        if results[mode][-1]["Nscoll"] == 0:
            fails.append("mode %s: no surface collisions, test geometry "
                         "is broken" % mode)
    return fails


def test_splitbalance(exe_cmd):
    # split cells plus a fix balance in the same run: after a full re-map
    # fix rigid reassigns split-cell particles to sub cells, which leaves
    # them unsorted, and the balance later in the step must re-sort
    # before migrating cells.  the box is periodic with no emission and
    # no deletion, so the particle count must stay at its initial value;
    # with stale lists a random balance quadrupled it on 4 ranks
    fails = []
    for mode in ("cutcell", "incremental"):
        rc, out = run_deck(exe_cmd, "in.test.splitbalance",
                           extra=["-var", "mode", mode])
        if rc:
            fails.append("mode %s: run failed with exit code %d" % (mode, rc))
            continue
        rows = parse_stats(out)
        if not rows:
            fails.append("mode %s: no stats output" % mode)
            continue
        np0 = rows[0]["Np"]
        for row in rows[1:]:
            if row["Np"] != np0:
                fails.append("mode %s: step %d has %d particles, started "
                             "with %d" % (mode, row["Step"], row["Np"], np0))
                break
        if rows[-1]["f_1"] != 0:
            fails.append("mode %s: %g particles deleted inside the body"
                         % (mode, rows[-1]["f_1"]))
    return fails


def test_transplane(exe_cmd):
    # a fast body sweeps over and then vacates cells cut only by a
    # transparent plane; the total flow volume per step must agree
    # between the incremental and cutcell remap modes
    vols = {}
    fails = []
    for mode in ("cutcell", "incremental"):
        rc, out = run_deck(exe_cmd, "in.test.transplane",
                           extra=["-var", "mode", mode])
        if rc:
            fails.append("mode %s: run failed with exit code %d" % (mode, rc))
            continue
        rows = parse_stats(out)
        if len(rows) < 15:
            fails.append("mode %s: expected 15 stats rows, got %d"
                         % (mode, len(rows)))
            continue
        vols[mode] = [r["c_tvol"] for r in rows]
    if fails:
        return fails
    for i, (vi, vc) in enumerate(zip(vols["incremental"], vols["cutcell"])):
        if not approx(vi, vc, rel=1e-12):
            fails.append("step %d: incremental flow volume %.17g differs "
                         "from cutcell %.17g" % (i, vi, vc))
    return fails


def test_multiremap(exe_cmd):
    # two gas-driven bodies: cutcell and incremental must give identical
    # trajectories, verifying multi-body incremental re-cut
    results = {}
    fails = []
    for mode in ("cutcell", "incremental"):
        rc, out = run_deck(exe_cmd, "in.test.multiremap",
                           extra=["-var", "mode", mode])
        if rc:
            fails.append("mode %s: run failed with exit code %d" % (mode, rc))
            continue
        rows = parse_stats(out)
        if not rows:
            fails.append("mode %s: no stats output" % mode)
            continue
        last = rows[-1]
        results[mode] = tuple(last[k] for k in
                              ("f_1[1][1]", "f_1[1][2]", "f_1[1][15]",
                               "f_1[2][1]", "f_1[2][2]", "f_1[2][15]"))
    if fails:
        return fails
    labels = ("b1 xcm", "b1 ycm", "b1 omega", "b2 xcm", "b2 ycm", "b2 omega")
    for i, name in enumerate(labels):
        if not approx(results["incremental"][i], results["cutcell"][i],
                      rel=1e-10, abs_=1e-13):
            fails.append("%s: incremental %.15g differs from cutcell %.15g"
                         % (name, results["incremental"][i],
                            results["cutcell"][i]))
    return fails


def test_pushpair(exe_cmd):
    # ASYMMETRIC body-body contact: heavy large body overtakes a light
    # small one, corner-vs-face contact. Total momentum of the pair must
    # be conserved (contact forces are equal-and-opposite on both
    # bodies); tolerance is set by the 8-digit stats output, not physics
    rc, out = run_deck(exe_cmd, "in.test.pushpair")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if not rows:
        return ["no stats output"]
    m1, m2 = 4.0e-22, 1.0e-22
    px0 = m1 * rows[0]["f_1[1][4]"] + m2 * rows[0]["f_1[2][4]"]
    py0 = m1 * rows[0]["f_1[1][5]"] + m2 * rows[0]["f_1[2][5]"]
    fails = []
    for r in rows:
        px = m1 * r["f_1[1][4]"] + m2 * r["f_1[2][4]"]
        py = m1 * r["f_1[1][5]"] + m2 * r["f_1[2][5]"]
        if not approx(px, px0, rel=1e-6):
            fails.append("step %d: px = %.10e vs initial %.10e, body-body "
                         "contact violates momentum conservation"
                         % (int(r["Step"]), px, px0))
        if abs(py - py0) > 1e-6 * abs(px0):
            fails.append("step %d: py = %.3e drifted from %.3e"
                         % (int(r["Step"]), py, py0))
    # the collision must actually have happened
    if rows[-1]["f_1[2][4]"] < 20.0:
        fails.append("final body2 vx = %.6g, no significant collision "
                     "occurred; test geometry is broken"
                     % rows[-1]["f_1[2][4]"])
    return fails


def test_twobody(exe_cmd):
    rc, out = run_deck(exe_cmd, "in.test.twobody")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if not rows:
        return ["no stats output"]
    last = rows[-1]
    v1, v2 = last["f_1[1][4]"], last["f_1[2][4]"]
    fails = []
    # head-on symmetric collision: velocities reverse, ~elastic
    if not approx(v1, -30.0, rel=0.02):
        fails.append("body1 final vx = %.6g, expected -30 within 2%%" % v1)
    if not approx(v2, 30.0, rel=0.02):
        fails.append("body2 final vx = %.6g, expected +30 within 2%%" % v2)
    # total momentum ~ 0
    if abs(v1 + v2) > 0.05:
        fails.append("momentum asymmetry |v1+v2| = %.4g > 0.05" % abs(v1 + v2))
    return fails


def test_recoil(exe_cmd):
    """single specular collision of a particle with a body only 10x heavier:
    exact two-body elastic result, frictionless normal impulse"""
    m = 1.0e10 * 2.325e-26          # fnum * m_N
    u = 1000.0
    fails = []

    # 2d: square, I = M/6, hit at r = (-0.5, yhit-5, 0), n = (-1,0,0)
    # mratio = 1 is the equal-mass exchange: particle stops, body takes u
    for yhit, mratio in ((5.0, 10.0), (5.4, 10.0), (5.0, 1.0)):
        M = mratio * m
        I2 = M / 6.0
        ry = yhit - 5.0
        J = 2.0 * m * u / (1.0 + m * (1.0 / M + ry * ry / I2))
        vx = u - J / m
        vcm = J / M
        omega = -ry * J / I2
        label = "2d yhit=%g M/m=%g" % (yhit, mratio)
        rc, out = run_deck(exe_cmd, "in.test.recoil",
                           ["-var", "yhit", str(yhit),
                            "-var", "mratio", str(mratio)])
        if rc != 0:
            fails.append("%s: run failed with exit code %d" % (label, rc))
            continue
        rows = parse_stats(out)
        if not rows:
            fails.append("%s: no stats output" % label)
            continue
        row = rows[-1]
        if row["Np"] != 1:
            fails.append("%s: particle lost" % label)
        for key, want in (("c_rvx", vx), ("c_rvy", 0.0), ("f_1[4]", vcm),
                          ("f_1[5]", 0.0), ("f_1[15]", omega)):
            if not approx(row[key], want, rel=1.0e-10, abs_=1.0e-10 * u):
                fails.append("%s: %s = %.12g, expected %.12g"
                             % (label, key, row[key], want))
        e0 = 0.5 * m * u * u
        e1 = 0.5 * m * (row["c_rvx"] ** 2 + row["c_rvy"] ** 2) \
            + 0.5 * M * (row["f_1[4]"] ** 2 + row["f_1[5]"] ** 2) \
            + 0.5 * I2 * row["f_1[15]"] ** 2
        if not approx(e1, e0, rel=1.0e-10):
            fails.append("%s: energy %.12g vs %.12g" % (label, e1, e0))

    # 3d: unit cube, I = M/6, hit at r = (-0.5, 0.3, 0.2), n = (-1,0,0)
    M = 10.0 * m
    I3 = M / 6.0
    ry, rz = 0.3, 0.2
    J = 2.0 * m * u / (1.0 + m * (1.0 / M + (ry * ry + rz * rz) / I3))
    vx = u - J / m
    vcm = J / M
    wy, wz = rz * J / I3, -ry * J / I3
    rc, out = run_deck(exe_cmd, "in.test.recoil3d")
    if rc != 0:
        fails.append("3d: run failed with exit code %d" % rc)
        return fails
    rows = parse_stats(out)
    if not rows:
        fails.append("3d: no stats output")
        return fails
    row = rows[-1]
    for key, want in (("c_rvx", vx), ("c_rvy", 0.0), ("c_rvz", 0.0),
                      ("f_1[4]", vcm), ("f_1[5]", 0.0), ("f_1[6]", 0.0),
                      ("f_1[13]", 0.0), ("f_1[14]", wy), ("f_1[15]", wz)):
        if not approx(row[key], want, rel=1.0e-10, abs_=1.0e-10 * u):
            fails.append("3d: %s = %.12g, expected %.12g"
                         % (key, row[key], want))
    e0 = 0.5 * m * u * u
    e1 = 0.5 * m * (row["c_rvx"] ** 2 + row["c_rvy"] ** 2 + row["c_rvz"] ** 2) \
        + 0.5 * M * (row["f_1[4]"] ** 2 + row["f_1[5]"] ** 2 + row["f_1[6]"] ** 2) \
        + 0.5 * I3 * (row["f_1[13]"] ** 2 + row["f_1[14]"] ** 2
                      + row["f_1[15]"] ** 2)
    if not approx(e1, e0, rel=1.0e-10):
        fails.append("3d: energy %.12g vs %.12g" % (e1, e0))
    return fails


def test_exitbox(exe_cmd):
    # two bodies leaving through opposite faces: the run must complete
    # (no false "surfs may enclose the box" error) and the motion is
    # ballistic, ycm = y0 + vy * t with t = 200 * 1e-4
    rc, out = run_deck(exe_cmd, "in.test.exitbox")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if not rows:
        return ["no stats output"]
    last = rows[-1]
    fails = []
    if not approx(last["f_1[1][2]"], 3.0 - 400.0 * 0.02, rel=1e-10):
        fails.append("body 1 ycm = %.12g, expected %.12g"
                     % (last["f_1[1][2]"], 3.0 - 400.0 * 0.02))
    if not approx(last["f_1[2][2]"], 7.0 + 400.0 * 0.02, rel=1e-10):
        fails.append("body 2 ycm = %.12g, expected %.12g"
                     % (last["f_1[2][2]"], 7.0 + 400.0 * 0.02))
    return fails


def negative_test(exe_cmd, deck, message):
    rc, out = run_deck(exe_cmd, deck)
    fails = []
    if rc == 0:
        fails.append("run succeeded but an error was expected")
    if message not in out:
        fails.append("expected error message not found: '%s'" % message)
    return fails


def test_badmoi(exe_cmd):
    return negative_test(exe_cmd, "in.test.badmoi", "triangle inequality")


def test_notwatertight(exe_cmd):
    return negative_test(exe_cmd, "in.test.notwatertight", "not watertight")


def test_facetbounce(exe_cmd):
    # a square rebounding from a 200-segment static circle: kinetic
    # energy in free flight after the contact must equal the launch
    # energy (contacts with a faceted surface must be conservative)
    rc, out = run_deck(exe_cmd, "in.test.facetbounce")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if len(rows) < 21:
        return ["expected 21 stats rows, got %d" % len(rows)]
    free = [r for r in rows if r["f_1[20]"] == 0.0 and r["f_1[21]"] == 0.0]
    if len(free) == len(rows):
        return ["the body never touched the circle, test geometry is broken"]
    if rows[-1]["f_1[4]"] >= 0.0:
        return ["body did not rebound (vx = %.6g)" % rows[-1]["f_1[4]"]]
    e0 = rows[0]["v_ke"]
    e1 = free[-1]["v_ke"]
    if not approx(e1, e0, rel=1e-4):
        return ["kinetic energy after the rebound %.10g vs %.10g before "
                "(%.3g%%)" % (e1, e0, 100.0 * (e1 / e0 - 1.0))]
    return []


def test_vacate(exe_cmd):
    # gas collisions with a fast body spanning whole interior cells: the
    # cells the body partly vacates within a step hold particles at zero
    # flow volume until the re-cut; the run must complete with the
    # particle count constant and nothing deleted after step 0
    rc, out = run_deck(exe_cmd, "in.test.vacate")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if len(rows) < 11:
        return ["expected 11 stats rows, got %d" % len(rows)]
    fails = []
    np0 = rows[0]["Np"]
    for r in rows[1:]:
        if r["Np"] != np0:
            fails.append("step %d: Np %g != %g" % (r["Step"], r["Np"], np0))
            break
    if rows[-1]["f_1"] != rows[0]["f_1"]:
        fails.append("%g particles deleted during the run"
                     % (rows[-1]["f_1"] - rows[0]["f_1"]))
    if rows[-1]["f_1[1]"] < 5.5:
        fails.append("body barely moved (xcm = %.6g), test is too weak"
                     % rows[-1]["f_1[1]"])
    return fails


def test_prenofix(exe_cmd):
    return negative_test(exe_cmd, "in.test.prenofix",
                         "not initialized before the run")


def test_refix(exe_cmd):
    # a fix rigid re-defined between runs, then balance_grid before the
    # next run: the rigid map rebuild must not reach the deleted fix;
    # the second run continues the ballistic body from where the
    # re-definition placed it (xcm 5.08 + 20*0.0001*20)
    rc, out = run_deck(exe_cmd, "in.test.refix")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if not rows or rows[-1]["Step"] != 60:
        return ["run did not reach step 60"]
    x = rows[-1]["f_1[1]"]
    if not approx(x, 5.08 + 20.0 * 1.0e-4 * 20, rel=1e-12):
        return ["xcm after the second run %.17g, expected %.17g"
                % (x, 5.08 + 20.0 * 1.0e-4 * 20)]
    return []


def test_inward(exe_cmd):
    # a body traversed the wrong way round (normals pointing into its
    # interior, a container) is rejected in 2d and 3d
    fails = negative_test(exe_cmd, "in.test.inward", "normals point inward")
    fails += negative_test(exe_cmd, "in.test.inward3d",
                           "normals point inward")
    return fails


def test_modifyafter(exe_cmd):
    return negative_test(exe_cmd, "in.test.modifyafter",
                         "attributes were changed")


def test_wallmotion(exe_cmd):
    return negative_test(exe_cmd, "in.test.wallmotion", "own wall motion")


def timestep_independence(exe_cmd, deck, keys, coarse, fine, tol=1.0e-9,
                          hitkey="c_rvx"):
    """run the same physical problem at two timesteps and require the same
    answer: the deck is built so that the only timestep-dependent piece is
    the moving-surf collision test; the particle is launched in +x and
    must have been turned around by the body (hitkey < 0), else the
    moving-surf test was never exercised"""
    fails = []
    last = {}
    for label, (dt, nsteps) in (("coarse", coarse), ("fine", fine)):
        rc, out = run_deck(exe_cmd, deck, ["-var", "dt", dt,
                                           "-var", "nsteps", nsteps])
        if rc:
            fails.append("%s: run failed with exit code %d" % (label, rc))
            continue
        rows = parse_stats(out)
        if not rows:
            fails.append("%s: no stats output" % label)
            continue
        if rows[-1]["Np"] != 1:
            fails.append("%s: the particle was lost" % label)
            continue
        if rows[-1][hitkey] >= 0.0:
            fails.append("%s: %s = %.6g, the particle never hit the body"
                         % (label, hitkey, rows[-1][hitkey]))
            continue
        last[label] = rows[-1]
    if fails:
        return fails
    for key in keys:
        c, f = last["coarse"][key], last["fine"][key]
        if not approx(c, f, rel=tol, abs_=tol):
            fails.append("%s: %.14g at the coarse timestep, %.14g at the "
                         "fine one" % (key, c, f))
    return fails


def test_rotwall(exe_cmd):
    # the body is heavy enough that the collision does not perturb it and
    # spins at a constant rate, so its pose is exact at any timestep, and
    # the particle is ballistic on either side of the collision.  the hit
    # fraction of the moving-surf test is then the only thing that can
    # make the two runs differ
    return timestep_independence(
        exe_cmd, "in.test.rotwall",
        ("c_rx", "c_ry", "c_rvx", "c_rvy"),
        ("1.0e-3", "5"), ("2.0e-5", "250"))


def test_rotwall3d(exe_cmd):
    return timestep_independence(
        exe_cmd, "in.test.rotwall3d",
        ("c_rx", "c_ry", "c_rz", "c_rvx", "c_rvy", "c_rvz"),
        ("1.0e-3", "5"), ("2.0e-5", "250"))


def test_customemit(exe_cmd):
    # a fix emit/surf which spreads a custom per-surf attribute, defined
    # before fix rigid, across two runs with distributed surfs: the
    # per-surf status flags fix rigid resets after a grid rebuild gate a
    # collective re-spread in the emit fix's init, so they must be reset
    # on every rank or none, else the second run's init hangs
    rc, out = run_deck(exe_cmd, "in.test.customemit")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if len(rows) < 2:
        return ["fewer than two stats rows, the second run did not start"]
    return []


def test_emitsurf(exe_cmd):
    return negative_test(exe_cmd, "in.test.emitsurf",
                         "cannot emit from fix rigid body surfs")


def test_renumber(exe_cmd):
    fails = negative_test(exe_cmd, "in.test.renumber", "were renumbered")
    fails += negative_test(exe_cmd, "in.test.renumber2", "were renumbered")
    return fails


def test_axistuck(exe_cmd):
    # no rigid body: the mover's moving-surf re-hit rule must leave the
    # legitimate repeated hits on one static line alone (axisymmetric)
    rc, out = run_deck(exe_cmd, "in.test.axistuck")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if not rows:
        return ["no stats output"]
    fails = []
    for row in rows:
        if row["Np"] != 2000:
            fails.append("step %d: np = %d, particles were deleted"
                         % (row["Step"], row["Np"]))
            break
    if rows[-1]["Nscoll"] < 1000:
        fails.append("too few surf collisions (%d) for the test to be "
                     "meaningful" % rows[-1]["Nscoll"])
    return fails


def test_tallyorder(exe_cmd):
    # a fix ave/surf on a static wall must see the same per-step hit
    # counts whether it is defined before or after fix rigid
    results = {}
    fails = []
    for order in ("0", "1"):
        rc, out = run_deck(exe_cmd, "in.test.tallyorder",
                           extra=["-var", "order", order])
        if rc:
            fails.append("order %s: run failed with exit code %d"
                         % (order, rc))
            continue
        rows = parse_stats(out)
        if not rows:
            fails.append("order %s: no stats output" % order)
            continue
        results[order] = rows
    if fails:
        return fails
    for r0, r1 in zip(results["0"], results["1"]):
        if r0["c_red"] != r1["c_red"]:
            fails.append("step %d: wall hits %g (ave/surf before fix rigid)"
                         " vs %g (after)" % (r0["Step"], r0["c_red"],
                                             r1["c_red"]))
            break
    if results["0"][-1]["c_red"] < 10:
        fails.append("too few wall hits (%g) for the test to be meaningful"
                     % results["0"][-1]["c_red"])
    return fails


def test_badinfile(exe_cmd):
    return negative_test(exe_cmd, "in.test.badinfile",
                         "Invalid floating point number")


def test_mixture(exe_cmd):
    return negative_test(exe_cmd, "in.test.mixture",
                         "mixture must contain all species")


def test_zerothick(exe_cmd):
    fails = negative_test(exe_cmd, "in.test.zerothick", "encloses zero area")
    fails += negative_test(exe_cmd, "in.test.zerothick3d",
                           "encloses zero volume")
    return fails


def read_state(name):
    """Read the 16 (or 22) body parameters of the first body from a fix
       rigid outfile, skipping its leading body ID.
       Returns the list of floats, or None if the file is missing."""
    path = os.path.join(THISDIR, name)
    if not os.path.exists(path):
        return None
    for line in open(path):
        line = line.split('#')[0].strip()
        if not line:
            continue
        return [float(w) for w in line.split()][1:]
    return None


def test_react(exe_cmd):
    """A surface reaction on body surfs which leaves exactly one particle.

    Species A becomes species B, whose mass is twice A's, so the impulse is
    mpost*v - mpre*vpre with two different masses and the recoil correction
    must use the post-collision mass.  A reaction is not an elastic normal
    impulse, so it must take the matrix branch of rigid_recoil, which also
    induces a tangential component through the off-diagonal terms of K.
    Checked against the exact rigid-body impulse solution."""
    FN = 1.0e-3
    mpre = FN * 2.325e-26
    mpost = FN * 4.650e-26
    MB = 1.0e-24
    IZZ = 1.6e-25
    V0 = 1000.0
    rx, ry = -0.5, 0.4
    rxn = -ry * -1.0                      # z of r x n for n = (-1,0)
    # K as a 3x3 acting in the plane; build the two in-plane rows we need
    kxx = 1.0/MB + ry*ry/IZZ
    kxy = -rx*ry/IZZ
    kyy = 1.0/MB + rx*rx/IZZ
    # Jinf from a specular reflection off a wall at rest, with the mass change
    jinf_x = mpost*(-V0) - mpre*V0
    # solve (1 + mpost K) J = Jinf in the plane
    a11 = 1.0 + mpost*kxx; a12 = mpost*kxy
    a21 = mpost*kxy;       a22 = 1.0 + mpost*kyy
    det = a11*a22 - a12*a21
    jx = ( a22*jinf_x) / det
    jy = (-a21*jinf_x) / det
    vout_x = (jx + mpre*V0) / mpost
    vout_y = jy / mpost
    dvcm_x = -jx/MB
    dom_z = (rx*(-jy) - ry*(-jx))/IZZ      # z of Iinv (r x -J)

    rc, out = run_deck(exe_cmd, "in.test.react")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if not rows:
        return ["no stats output"]
    last = rows[-1]
    fails = []
    if last["Np"] != 1:
        fails.append("Np = %d, an exchange reaction must leave exactly one "
                     "particle" % last["Np"])
    if last["f_1"] != 0:
        fails.append("%d particles deleted" % last["f_1"])
    for nm, col, ex in (("particle vx", "c_pv[1]", vout_x),
                        ("particle vy", "c_pv[2]", vout_y),
                        ("body vcom x", "f_1[4]", dvcm_x),
                        ("body omega z", "f_1[15]", dom_z)):
        if not approx(last[col], ex, rel=1.0e-11):
            fails.append("%s %.17g != exact %.17g" % (nm, last[col], ex))

    # momentum must be conserved exactly despite the mass change
    dp = mpost*last["c_pv[1]"] + MB*last["f_1[4]"] - mpre*V0
    if abs(dp) > 1.0e-12 * abs(mpre*V0):
        fails.append("momentum not conserved across the reaction: %.3e" % dp)
    return fails


def test_badreact(exe_cmd):
    """A reaction model which does not leave exactly one particle must be
    rejected on body surfs: the recoil correction is undefined and the body
    would gain or lose mass, which this fix holds fixed."""
    rc, out = run_deck(exe_cmd, "in.test.badreact")
    if rc == 0:
        return ["a recombination reaction on the body surfs was accepted"]
    if "one particle" not in out and "surf_react" not in out:
        return ["rejected, but not with the expected message"]
    return []


def test_recoil_spin(exe_cmd):
    """The recoil correction must include the body's rotational response.

    A collision's impulse is reduced by the body's inverse-mass matrix at the
    contact, K = 1/M - [r x] Iinv [r x].  in.test.recoil covers the 1/M part
    on a head-on hit; here the strike is off-centre and Izz is chosen so the
    rotational term (r x n)^2/Izz equals 1/M exactly, making it half the
    body's response.  Checked against the exact rigid-body impulse solution."""
    MP = MASS_N * 1.0e-3
    MB = 1.0e-24
    IZZ = 1.6e-25
    V0 = 1000.0
    # left face of the unit square at x=4.5, outward normal (-1,0); COM (5,5)
    rx, ry = -0.5, 0.4
    nx, ny = -1.0, 0.0
    rxn = rx*ny - ry*nx                      # z component of r x n
    nKn = 1.0/MB + rxn*rxn/IZZ
    p = -2.0 * (V0*nx) / (1.0/MP + nKn)      # elastic normal impulse
    vcm_ex = -(p/MB)*nx
    om_ex = -p*rxn/IZZ

    rc, out = run_deck(exe_cmd, "in.test.recoil.spin")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if not rows:
        return ["no stats output"]
    last = rows[-1]
    fails = []
    if last["f_1"] != 0:
        fails.append("%d particles deleted" % last["f_1"])
    if not approx(last["f_1[4]"], vcm_ex, rel=1.0e-12):
        fails.append("vcom x %.17g != exact rigid-body impulse result %.17g"
                     % (last["f_1[4]"], vcm_ex))
    if not approx(last["f_1[15]"], om_ex, rel=1.0e-12):
        fails.append("omega z %.17g != exact %.17g -- the rotational part of "
                     "the recoil correction is wrong" % (last["f_1[15]"], om_ex))
    if abs(last["f_1[5]"]) > 1.0e-12 * abs(vcm_ex):
        fails.append("vcom y %.3e != 0 (impulse is along x)" % last["f_1[5]"])

    # exact momentum conservation, and an elastic collision conserves energy
    vp_out = V0 + (p/MP)*nx
    dp = MP*(vp_out - V0) + MB*last["f_1[4]"]
    if abs(dp) > 1.0e-12 * abs(MP*V0):
        fails.append("momentum not conserved: %.3e" % dp)
    e0 = 0.5*MP*V0*V0
    e1 = (0.5*MP*vp_out*vp_out + 0.5*MB*last["f_1[4]"]**2 +
          0.5*IZZ*last["f_1[15]"]**2)
    if abs(e1-e0) > 1.0e-10 * e0:
        fails.append("energy not conserved: rel %.3e" % ((e1-e0)/e0))
    return fails


def test_couple(exe_cmd):
    """Accuracy of the explicit gas-body coupling, measured on one impulse.

    A single particle strikes a free body, so the collision set is fixed and
    only the placement of the impulse within the step varies with dt.  The
    velocity must be exact at every dt; the position carries an offset of at
    most one step's body motion, falling linearly with dt."""
    MP = MASS_N * 1.0e-3          # species mass x fnum
    MB = 1.0e-24
    V0 = 1000.0
    X0 = 2.037
    T = 5.0e-3
    thit = (4.5 - X0) / V0        # body is at rest until struck
    vb = 2.0 * MP * V0 / (MP + MB)
    xex = 5.0 + vb * (T - thit)

    fails = []
    seen = []
    for ns in (1250, 5000, 20000):
        dt = T / ns
        rc, out = run_deck(exe_cmd, "in.test.couple",
                           extra=["-var", "dt", "%.17g" % dt,
                                  "-var", "nstep", str(ns)])
        if rc:
            return ["run failed with exit code %d" % rc]
        rows = parse_stats(out)
        if not rows:
            return ["no stats output"]
        last = rows[-1]
        if last["f_1"] != 0:
            fails.append("dt=%g: %d particles deleted" % (dt, last["f_1"]))

        # velocity: exact, independent of dt
        vgot = last["f_1[4]"]
        if not approx(vgot, vb, rel=1.0e-13):
            fails.append("dt=%g: body velocity %.17g != two-body result "
                         "%.17g -- the impulse must not depend on where in "
                         "the step the hit lands" % (dt, vgot, vb))

        # position: offset bounded by one step's body motion
        xerr = abs(last["f_1[1]"] - xex)
        if xerr > 1.05 * vb * dt:
            fails.append("dt=%g: position error %.3e exceeds one step's body "
                         "motion %.3e" % (dt, xerr, vb * dt))
        seen.append((dt, xerr))

    # and that offset must actually shrink with dt, not sit at a fixed value
    if len(seen) == 3 and all(e > 0 for _, e in seen):
        if seen[0][1] < seen[2][1]:
            fails.append("position error does not decrease with dt: %s"
                         % ", ".join("dt=%g err=%.3e" % x for x in seen))
    return fails


def test_density(exe_cmd):
    """dstyle = density on a unit cube: mass, COM and moi from geometry.

    A cube is exactly representable by flat triangles, so every value is
    analytic and the only error is round-off.  The tolerance is set by the
    outfile's 17-digit format, not by any property of the shape."""
    fails = []
    out_name = "tmp.density.state"
    path = os.path.join(THISDIR, out_name)
    if os.path.exists(path):
        os.remove(path)

    rc, out = run_deck(exe_cmd, "in.test.density")
    if rc:
        return ["run failed with exit code %d" % rc]

    v = read_state(out_name)
    if v is None:
        return ["no state file written"]
    if len(v) < 16:
        return ["state file has %d values, expected at least 16" % len(v)]

    # unit cube, density 1
    TOL = 1.0e-12
    if abs(v[0] - 1.0) > TOL:
        fails.append("mass %.17g != 1 (unit cube, density 1)" % v[0])
    for k, nm in enumerate("xyz"):
        if abs(v[1+k]) > TOL:
            fails.append("com %s %.3e != 0 (cube is centred on the origin)"
                         % (nm, v[1+k]))
    for k, nm in enumerate(("ixx", "iyy", "izz")):
        if abs(v[4+k] - 1.0/6.0) > TOL:
            fails.append("%s %.17g != M/6 = %.17g" % (nm, v[4+k], 1.0/6.0))
    for k, nm in enumerate(("ixy", "ixz", "iyz")):
        if abs(v[7+k]) > TOL:
            fails.append("%s %.3e != 0 (cube has no products of inertia)"
                         % (nm, v[7+k]))

    os.remove(path)
    return fails


def test_density_far(exe_cmd):
    """dstyle = density must give the same answer wherever the body sits.

    The volume integrals reduce to sums over the body elements.  Taken about
    the coordinate origin those sums are O(V R^2) for a body of size L a
    distance R from the origin, while the answer is O(V L^2), so the inertia
    loses relative precision like (R/L)^2 -- this unit cube at 1e5 kept only
    about 5 digits before the sums were referenced to the body itself.  Mass
    and COM are unaffected either way; only the second moments degrade, so
    this test is what distinguishes the two reductions."""
    fails = []
    out_name = "tmp.densityfar.state"
    path = os.path.join(THISDIR, out_name)
    if os.path.exists(path):
        os.remove(path)

    rc, out = run_deck(exe_cmd, "in.test.density.far")
    if rc:
        return ["run failed with exit code %d" % rc]

    v = read_state(out_name)
    if v is None:
        return ["no state file written"]
    if len(v) < 16:
        return ["state file has %d values, expected at least 16" % len(v)]

    C = 100000.5
    # the same tolerances as in.test.density: placement must not cost digits
    if abs(v[0] - 1.0) > 1.0e-12:
        fails.append("mass %.17g != 1 (unit cube, density 1)" % v[0])
    for k, nm in enumerate("xyz"):
        if abs(v[1+k] - C) > 1.0e-9:
            fails.append("com %s %.17g != %.17g" % (nm, v[1+k], C))
    for k, nm in enumerate(("ixx", "iyy", "izz")):
        err = abs(v[4+k] - 1.0/6.0) / (1.0/6.0)
        if err > 1.0e-12:
            fails.append("%s %.17g != M/6, rel err %.3e -- the moment sums "
                         "are losing precision to the body's distance from "
                         "the origin" % (nm, v[4+k], err))
    for k, nm in enumerate(("ixy", "ixz", "iyz")):
        if abs(v[7+k]) > 1.0e-12:
            fails.append("%s %.3e != 0 (cube has no products of inertia)"
                         % (nm, v[7+k]))

    os.remove(path)
    return fails


def test_density2d(exe_cmd):
    """dstyle = density in 2d, plus the vcom/angmom keywords.

    Unit square plate: izz = M/6, ixx = iyy = izz/2, all products zero,
    and ixz = iyz = 0 exactly as 2d requires.  vcom and angmom are user
    input even under density style, so they must round-trip unchanged."""
    fails = []
    out_name = "tmp.density2d.state"
    path = os.path.join(THISDIR, out_name)
    if os.path.exists(path):
        os.remove(path)

    rc, out = run_deck(exe_cmd, "in.test.density2d")
    if rc:
        return ["run failed with exit code %d" % rc]

    v = read_state(out_name)
    if v is None:
        return ["no state file written"]
    if len(v) < 16:
        return ["state file has %d values, expected at least 16" % len(v)]

    TOL = 1.0e-12
    if abs(v[0] - 1.0) > TOL:
        fails.append("mass %.17g != 1 (unit square, density 1)" % v[0])

    # the body moves during the one step it is run, so the COM is compared
    # against its start-of-step value plus the drift from vcom
    if abs(v[3]) > TOL:
        fails.append("com z %.3e != 0 for a 2d body" % v[3])

    if abs(v[4] - 1.0/12.0) > TOL:
        fails.append("ixx %.17g != M/12 = %.17g" % (v[4], 1.0/12.0))
    if abs(v[5] - 1.0/12.0) > TOL:
        fails.append("iyy %.17g != M/12 = %.17g" % (v[5], 1.0/12.0))
    if abs(v[6] - 1.0/6.0) > TOL:
        fails.append("izz %.17g != M/6 = %.17g" % (v[6], 1.0/6.0))
    if abs(v[7]) > TOL:
        fails.append("ixy %.3e != 0 (square has no product of inertia)" % v[7])

    # a 2d body must have these exactly zero, not merely small: setup_body()
    # requires a principal axis along z
    if v[8] != 0.0 or v[9] != 0.0:
        fails.append("ixz,iyz = %.3e,%.3e must be exactly 0 for 2d"
                     % (v[8], v[9]))

    # vcom and angmom are user input, unchanged by the geometry integration
    for k, (got, want, nm) in enumerate(
            ((v[10], 12.0, "vxcm"), (v[11], -3.0, "vycm"),
             (v[12], 0.0, "vzcm"), (v[15], 5.0e-3, "lz"))):
        if abs(got - want) > 1.0e-12 * max(abs(want), 1.0):
            fails.append("%s %.17g != %.17g as given" % (nm, got, want))
    if v[13] != 0.0 or v[14] != 0.0:
        fails.append("lx,ly = %.3e,%.3e must be exactly 0 for 2d"
                     % (v[13], v[14]))

    os.remove(path)
    return fails


def test_baddensity(exe_cmd):
    return negative_test(exe_cmd, "in.test.baddensity",
                         "body density must be positive")


# ----------------------------------------------------------------------
# axisymmetric domains
#
# a rigid body in an axisymmetric domain is a body of revolution about the
# x axis, so its only symmetry-preserving motions are translation along x
# and spin about x.  the tests below check that reduction from three
# directions: the free-flight integration, the volume integrals that give
# a body of revolution its mass and inertia, and the two exact
# conservation laws the collision machinery has to satisfy
# ----------------------------------------------------------------------

def test_axiballistic(exe_cmd):
    """Free body of revolution: linear drift along the axis, constant spin.

    The transverse degrees of freedom do not exist, so ycm, zcm, vy, vz and
    the transverse angular velocities must be identically zero, not merely
    small.  With -var fx a constant axial force is applied and the COM must
    follow the exact constant-force trajectory."""
    fails = []

    rc, out = run_deck(exe_cmd, "in.test.axiballistic")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if not rows:
        return ["no stats output"]
    last = rows[-1]

    # com = com0 + v t, t = 1000 * 1e-4 = 0.1;  omega_x = Lx / ixx
    if not approx(last["f_1[1]"], 1.5 + 2.0 * 0.1, rel=1e-7):
        fails.append("xcm = %.12g, expected 1.7" % last["f_1[1]"])
    if not approx(last["f_1[4]"], 2.0, rel=1e-7):
        fails.append("vx = %.12g, expected 2" % last["f_1[4]"])
    if not approx(last["f_1[13]"], 7.5e-21 / 1.0e-23, rel=1e-7):
        fails.append("omega_x = %.12g, expected 750" % last["f_1[13]"])

    for r in rows:
        for col, nm in (("f_1[2]", "ycm"), ("f_1[3]", "zcm"),
                        ("f_1[5]", "vy"), ("f_1[6]", "vz"),
                        ("f_1[14]", "omega_y"), ("f_1[15]", "omega_z")):
            if r[col] != 0.0:
                fails.append("%s = %.3e at step %d, must be exactly zero for "
                             "a body of revolution" % (nm, r[col], r["Step"]))
                break
        else:
            continue
        break

    # constant axial force: xcm = xcm0 + v0 t + f t^2 / (2 M)
    FX = 1.0e-21
    MB = 1.0e-22
    rc, out = run_deck(exe_cmd, "in.test.axiballistic",
                       extra=["-var", "fx", repr(FX)])
    if rc:
        fails.append("forced run failed with exit code %d" % rc)
        return fails
    rows = parse_stats(out)
    if not rows:
        fails.append("forced run: no stats output")
        return fails
    last = rows[-1]
    t = 0.1
    xexp = 1.5 + 2.0 * t + 0.5 * FX * t * t / MB
    vexp = 2.0 + FX * t / MB
    if not approx(last["f_1[1]"], xexp, rel=1e-7):
        fails.append("forced xcm = %.12g, expected %.12g"
                     % (last["f_1[1]"], xexp))
    if not approx(last["f_1[4]"], vexp, rel=1e-7):
        fails.append("forced vx = %.12g, expected %.12g"
                     % (last["f_1[4]"], vexp))
    return fails


def test_axidensity(exe_cmd):
    """dstyle = density for a body of revolution.

    A straight profile segment generates a frustum, and the volume integrals
    which give mass, COM and inertia reduce to exact polynomials in its end
    points, so a cylinder and a cone are both represented exactly and the
    only error is round-off.  The cone is the case that exercises the
    dr != 0 terms; a cylinder profile has none."""
    fails = []
    out_name = "tmp.axidensity.state"
    path = os.path.join(THISDIR, out_name)

    RHO = 1000.0

    # (geometry, volume, xcm, ixx/M, iyy/M)
    # cylinder R = 0.1 from x = 0.4 to 0.6:
    #   V = pi R^2 L, ixx = M R^2 / 2, iyy = M (3 R^2 + L^2) / 12
    # cone base R = 0.1 at x = 0.4, apex at x = 0.7:
    #   V = pi R^2 h / 3, xcm = base + h/4, ixx = 3 M R^2 / 10,
    #   iyy = 3 M (4 R^2 + h^2) / 80
    # torus of rectangular section, r = R1 to R2 over the same length:
    #   V = pi (R2^2 - R1^2) L, ixx = M (R1^2 + R2^2) / 2,
    #   iyy = M L^2 / 12 + ixx / 2
    # it is the only case whose profile does NOT terminate on the axis, so
    # it is the one that exercises the negative-dx segments of the moment
    # sums and the watertight check without the axis exception
    R = 0.1
    L = 0.2
    H = 0.3
    R1 = 0.06
    cases = [
        ("cylinder", math.pi * R * R * L, 0.5,
         R * R / 2.0, (3.0 * R * R + L * L) / 12.0),
        ("cone", math.pi * R * R * H / 3.0, 0.4 + H / 4.0,
         3.0 * R * R / 10.0, 3.0 * (4.0 * R * R + H * H) / 80.0),
        ("torus", math.pi * (R * R - R1 * R1) * L, 0.5,
         (R1 * R1 + R * R) / 2.0,
         L * L / 12.0 + (R1 * R1 + R * R) / 4.0),
    ]

    TOL = 1.0e-12
    for geom, vol, xcm, ixx_m, iyy_m in cases:
        if os.path.exists(path):
            os.remove(path)
        rc, out = run_deck(exe_cmd, "in.test.axidensity",
                           extra=["-var", "geom", geom])
        if rc:
            fails.append("%s: run failed with exit code %d" % (geom, rc))
            continue
        v = read_state(out_name)
        if v is None or len(v) < 16:
            fails.append("%s: no usable state file" % geom)
            continue

        mass = RHO * vol
        for got, want, nm in ((v[0], mass, "mass"),
                              (v[1], xcm, "xcm"),
                              (v[4], mass * ixx_m, "ixx"),
                              (v[5], mass * iyy_m, "iyy"),
                              (v[6], mass * iyy_m, "izz")):
            if not approx(got, want, rel=TOL):
                fails.append("%s: %s %.17g != %.17g" % (geom, nm, got, want))

        # the COM of a body of revolution is on the axis, and its products
        # of inertia vanish, both exactly
        for k, nm in ((2, "ycm"), (3, "zcm"),
                      (7, "ixy"), (8, "ixz"), (9, "iyz")):
            if v[k] != 0.0:
                fails.append("%s: %s = %.3e, must be exactly zero"
                             % (geom, nm, v[k]))

    if os.path.exists(path):
        os.remove(path)
    return fails


def test_aximomentum(exe_cmd):
    """Axial momentum of gas + body is conserved in an axisymmetric domain.

    Same invariant as test_momentum, including the half-step impulse which
    velocity Verlet has in flight, so it holds to round-off rather than to
    the statistical size of one step's transfer.  Only the axial component
    is conserved: the axisymmetric model rotates each particle's (vy,vz)
    into the plane on every move, so transverse gas momentum is not a
    conserved quantity of the model at all."""
    MB = 1.0e-22
    DT = 1.0e-5
    rows_by_mode = {}
    for mode in ("incremental", "cutcell"):
        rc, out = run_deck(exe_cmd, "in.test.aximomentum",
                           extra=["-var", "remap", mode])
        if rc:
            return ["%s: run failed with exit code %d" % (mode, rc)]
        rows_by_mode[mode] = parse_stats(out)
    rows = rows_by_mode["incremental"]
    if len(rows) < 3:
        return ["not enough stats output"]

    p = [MASS_N * FNUM * r["c_r"] + MB * r["f_1[4]"] + 0.5 * DT * r["f_1[7]"]
         for r in rows]
    drift = max(abs(x - p[0]) for x in p) / abs(p[0])
    fails = []
    if drift > 1.0e-12:
        fails.append("axial momentum drifted by %.3e (relative)" % drift)

    # compute surf tallies the raw per-element force, whose radial part is
    # large and does not cancel in the plane; it is fix rigid that takes
    # the azimuthal average, so the force the body actually sees must have
    # no transverse part at all
    for r in rows:
        if r["f_1[8]"] != 0.0 or r["f_1[9]"] != 0.0:
            fails.append("transverse force on the body %.3e %.3e is not "
                         "exactly zero at step %d"
                         % (r["f_1[8]"], r["f_1[9]"], r["Step"]))
            break

    # the incremental re-cut must reproduce the full rebuild, which for an
    # axisymmetric domain includes the annular volumes it installs for the
    # cells the body vacates.  in serial with non-distributed surfs the two
    # agree bit for bit; on several procs the cutcell path rebuilds the
    # grid every step and so reorders the summation of the gas momentum,
    # which moves its last bits
    other = rows_by_mode["cutcell"]
    if len(other) != len(rows):
        fails.append("cutcell remap produced %d stats rows, incremental %d"
                     % (len(other), len(rows)))
    else:
        for a, b in zip(rows, other):
            for col in ("f_1[4]", "c_r"):
                if not approx(a[col], b[col], rel=1.0e-12, abs_=1.0e-30):
                    fails.append("cutcell and incremental remap diverged at "
                                 "step %d: %s %.17g vs %.17g"
                                 % (a["Step"], col, a[col], b[col]))
                    break
            else:
                continue
            break
    # the body must actually be pushed, or the invariant is vacuous
    if abs(rows[-1]["f_1[4]"]) < 1.0e-3:
        fails.append("body barely moved: vcm_x = %.3e" % rows[-1]["f_1[4]"])
    return fails


def test_axispin(exe_cmd):
    """Axial angular momentum of gas + body is conserved.

    r * v_theta is a particle's angular momentum about the axis; it does not
    change during free flight, the remap into the plane is a rotation about
    that axis, and the reflecting outer wall reverses only v_r.  So
      m fnum sum(r v_theta) + ixx omega_x + 0.5 dt tx
    holds to round-off.  This is the invariant which pins down the azimuthal
    channel of the axisymmetric recoil, kmat[2][2] = r^2/ixx: a body which
    took the whole impulse, or none of it, would break it.  The wall is
    diffuse, so the matrix branch of the recoil solve is the one used."""
    IXX = 5.0e-27
    DT = 1.0e-5
    rc, out = run_deck(exe_cmd, "in.test.axispin")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if len(rows) < 3:
        return ["not enough stats output"]

    lz = [MASS_N * FNUM * r["c_r"] + IXX * r["f_1[13]"] + 0.5 * DT * r["f_1[10]"]
          for r in rows]
    drift = max(abs(x - lz[0]) for x in lz) / abs(lz[0])
    fails = []
    if drift > 1.0e-12:
        fails.append("axial angular momentum drifted by %.3e (relative)"
                     % drift)
    # the body must actually spin down, or the invariant is vacuous
    w0 = rows[0]["f_1[13]"]
    wend = rows[-1]["f_1[13]"]
    if abs(wend - w0) < 0.05 * abs(w0):
        fails.append("body barely spun down: omega_x %.4g -> %.4g"
                     % (w0, wend))
    return fails


def test_axidrag(exe_cmd):
    """Free-molecular drag on a body of revolution.

    A cold beam at speed U on a sphere of radius R with specular reflection
    gives Fx = rho U^2 pi R^2 exactly, and the faceted profile reproduces
    the smooth value because its widest vertex sits at R.

    The sharper half of this test is the comparison between the same sphere
    run as a fix rigid body and as a static surf.  The body is heavy enough
    not to move, and the moving-surf test reduces to the static one exactly
    in that case, because the body frame is a Galilean boost along the
    symmetry axis: the two runs must agree to round-off, not merely to the
    statistical error.  A third run repeats the rigid case with radial cell
    weighting, which must not change the drag."""
    U = 1000.0
    R = 0.5
    NRHO = 1.0
    analytic = MASS_N * NRHO * U * U * math.pi * R * R

    fails = []
    means = {}
    for label, extra in (("rigid", ["-var", "rigid", "1"]),
                         ("static", ["-var", "rigid", "0"]),
                         ("weighted", ["-var", "rigid", "1",
                                       "-var", "weight", "1"])):
        rc, out = run_deck(exe_cmd, "in.test.axidrag", extra=extra)
        if rc:
            fails.append("%s: run failed with exit code %d" % (label, rc))
            continue
        rows = parse_stats(out)
        if len(rows) < 10:
            fails.append("%s: not enough stats output" % label)
            continue
        fx = [r["c_red[1]"] for r in rows[1:]]
        means[label] = sum(fx) / len(fx)

    if "rigid" in means and "static" in means:
        rel = abs(means["rigid"] - means["static"]) / abs(means["static"])
        if rel > 1.0e-9:
            fails.append("moving body and static surf differ by %.3e "
                         "(relative); with the body at rest the moving test "
                         "must reduce to the static one" % rel)
    if "rigid" in means and "weighted" in means:
        rel = abs(means["weighted"] - means["rigid"]) / abs(means["rigid"])
        if rel > 0.06:
            fails.append("radial cell weighting changed the drag by %.3e"
                         % rel)
    for label in ("rigid", "weighted"):
        if label not in means:
            continue
        # 1.6% seed-to-seed scatter was measured over five seeds, and the
        # proc count changes the random stream, so this is a 5-sigma bound
        # rather than a precision claim: the precision content of this test
        # is the exact rigid-vs-static comparison above.  a real error in
        # the drag would be tens of percent
        rel = abs(means[label] - analytic) / analytic
        if rel > 0.08:
            fails.append("%s: drag %.6e differs from the free-molecular "
                         "value %.6e by %.2f%%"
                         % (label, means[label], analytic, 100.0 * rel))
    return fails


def test_axipair(exe_cmd):
    """Push-off contact between two bodies of revolution on the same axis.

    No particles.  A cylinder slides along the axis into a heavier one at
    rest.  The contact force is ring-weighted by 2*pi*r and the reaction on
    the second body uses the SAME weighted force, so the total axial
    momentum of the pair is conserved exactly however that weighting is
    chosen -- that is the property this test pins down, and it is
    independent of how well the contact is resolved in time.  The
    individual velocities converge to the elastic two-body result only to
    the one-step resolution of the release instant.

    This is the body-body branch of push_contact in an axisymmetric domain,
    which the single-body axipush deck does not reach."""
    M1 = 1.0e-22
    M2 = 2.0e-22
    V0 = 100.0
    rc, out = run_deck(exe_cmd, "in.test.axipair")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if len(rows) < 5:
        return ["not enough stats output"]

    fails = []
    for r in rows:
        for col, nm in (("f_1[1][5]", "body 1 vcm_y"), ("f_1[2][5]", "body 2 vcm_y"),
                        ("f_1[1][21]", "body 1 fpush_y"),
                        ("f_1[2][21]", "body 2 fpush_y")):
            if r[col] != 0.0:
                fails.append("%s = %.3e at step %d, must be exactly zero for "
                             "a body of revolution" % (nm, r[col], r["Step"]))
                break
        else:
            continue
        break

    last = rows[-1]
    v1, v2 = last["f_1[1][4]"], last["f_1[2][4]"]
    if v2 <= 0.0:
        fails.append("the bodies never made contact: body 2 vcm_x = %.4g" % v2)
        return fails

    # exact: the ring weighting cancels between action and reaction
    p0 = M1 * V0
    p1 = M1 * v1 + M2 * v2
    if not approx(p1, p0, rel=1.0e-12):
        fails.append("pair momentum %.17g != %.17g (relative %.3e)"
                     % (p1, p0, abs(p1 - p0) / abs(p0)))

    # approximate: elastic two-body, to the one-step contact resolution
    if not approx((v2 - v1) / V0, 1.0, rel=1.0e-3):
        fails.append("restitution %.6f, expected 1 for an undamped contact"
                     % ((v2 - v1) / V0))
    return fails


def test_axireact(exe_cmd):
    """A mass-changing surface reaction on a spinning body of revolution.

    Every hit flips the species between A and B, whose masses differ by 2x.
    The wall is specular and the body spins, which is what makes this test
    the azimuthal channel of the axisymmetric recoil rather than only the
    axial one: specular reflection leaves the azimuthal velocity relative to
    the wall unchanged, so the azimuthal impulse is exactly
    (mpost - mpre) * v_theta, non-zero only because the mass changed.  The
    body answers it through kmat[2][2] = r^2/ixx.

    Both exact invariants must hold, with the per-particle mass taken from
    the current species so the sums follow the reactions.  Neither is
    conserved by the reaction itself -- an exchange between species of
    different mass does not conserve gas momentum -- but the impulse the
    body is given is exactly the momentum the gas lost, so the sum is."""
    MB = 1.0e-22
    IXX = 5.0e-28
    DT = 1.0e-5
    rc, out = run_deck(exe_cmd, "in.test.axireact")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if len(rows) < 5:
        return ["not enough stats output"]

    fails = []
    if any(r["f_1"] != 0 for r in rows):
        fails.append("%d particles were deleted; a reacting particle must "
                     "not be lost" % rows[-1]["f_1"])

    # both drifts are measured against the size of the PARTS of each sum,
    # not against the sum itself.  an invariant that is a difference of two
    # larger quantities has no meaningful relative measure of its own: the
    # angular pieces here nearly cancel, and the axial ones would too if
    # the gas had no net drift, at which point the reduction-order noise of
    # an MPI sum swamps the closure.  the deck gives the gas a drift so the
    # axial sum is well conditioned as well, but the measure below does not
    # rely on that
    pg = [FNUM * r["c_rp"] for r in rows]
    pb = [MB * r["f_1[4]"] for r in rows]
    p = [g + b + 0.5 * DT * r["f_1[7]"]
         for g, b, r in zip(pg, pb, rows)]
    lg = [FNUM * r["c_rl"] for r in rows]
    lb = [IXX * r["f_1[13]"] for r in rows]
    lz = [g + b + 0.5 * DT * r["f_1[10]"]
          for g, b, r in zip(lg, lb, rows)]

    def drift(vals, parts):
        scale = max(max(abs(x) for x in q) for q in parts)
        return max(abs(x - vals[0]) for x in vals) / scale

    pdrift = drift(p, (pg, pb))
    ldrift = drift(lz, (lg, lb))
    if pdrift > 1.0e-12:
        fails.append("axial momentum drifted by %.3e (relative to the "
                     "size of its parts)" % pdrift)
    if ldrift > 1.0e-12:
        fails.append("axial angular momentum drifted by %.3e (relative to "
                     "the size of its parts)" % ldrift)

    # the spin must actually respond, or the azimuthal channel is untested
    w = [r["f_1[13]"] for r in rows]
    span = (max(w) - min(w)) / abs(w[0])
    if span < 0.05:
        fails.append("body spin barely moved (%.3g of omega_0): the "
                     "azimuthal impulse is not being exercised" % span)
    return fails


def test_axipush(exe_cmd):
    """Push-off contact for a body of revolution.

    No particles.  A solid cylinder slides along the axis into a static
    disk and is pushed back elastically, so the axial speed must return
    unchanged.  Meanwhile the narrow box keeps the body's outer radius
    inside the cutoff of the yhi boundary the whole run, so there is a live
    radial contact throughout: its force must never reach the body.  That
    is the discriminating check -- with the azimuthal average removed, the
    radial push shows up in fpush_y and leaks straight into fcm_y."""
    rc, out = run_deck(exe_cmd, "in.test.axipush")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if len(rows) < 5:
        return ["not enough stats output"]

    fails = []
    for r in rows:
        for col, nm in (("f_1[2]", "ycm"), ("f_1[5]", "vcm_y"),
                        ("f_1[8]", "fcm_y"), ("f_1[9]", "fcm_z"),
                        ("f_1[21]", "fpush_y"), ("f_1[22]", "fpush_z")):
            if r[col] != 0.0:
                fails.append("%s = %.3e at step %d, must be exactly zero for "
                             "a body of revolution" % (nm, r[col], r["Step"]))
                break
        else:
            continue
        break

    # the axial contact must actually have happened
    if not any(r["f_1[20]"] != 0.0 for r in rows):
        fails.append("no axial contact force was ever applied")

    # elastic contact with no damping: the speed comes back
    v0 = rows[0]["f_1[4]"]
    vend = rows[-1]["f_1[4]"]
    if vend >= 0.0:
        fails.append("body did not bounce: vcm_x %.6g -> %.6g" % (v0, vend))
    elif not approx(abs(vend), abs(v0), rel=1.0e-4):
        fails.append("contact was not elastic: |vcm_x| %.9g -> %.9g"
                     % (abs(v0), abs(vend)))
    return fails


def test_axibadvcom(exe_cmd):
    return negative_test(exe_cmd, "in.test.axibadvcom",
                         "vcom must be zero for an axisymmetric domain")


def test_axiopen(exe_cmd):
    """A free end off the axis is still a hole in an axisymmetric body.

    The axis exception lets a profile terminate at r = 0, where the surface
    of revolution closes itself.  It must not let through a free end
    anywhere else, even one which sits on the box surface and so passes the
    Surf class's own watertight check."""
    return negative_test(exe_cmd, "in.test.axiopen", "not watertight")


def test_badvcom(exe_cmd):
    return negative_test(exe_cmd, "in.test.badvcom",
                         "vcom keyword requires density style")


def test_nonfinite(exe_cmd):
    """A body state which stops being a finite number is caught here.

    Inf or NaN in the pose reaches the surf coords, the moving-surf
    collision tests, and the cut/split geometry, where Cut2d walks off the
    end of its point list and writes out of bounds instead of failing."""
    return negative_test(exe_cmd, "in.test.nonfinite",
                         "no longer a finite number")


def test_crossproc(exe_cmd):
    """Bodies crossing proc boundaries with distributed surfs.

    With distributed surfs a proc stores local copies only of the bodies
    which can reach its cells, appended as a body approaches.  Three
    gas-driven bodies fast enough to cross the whole box, with contacts
    between them and with the box boundaries, must follow exactly the
    same trajectories as with non-distributed surfs on the same procs,
    where every proc stores every body.  Run it on several procs."""
    results = {}
    fails = []
    for dist in ("0", "1"):
        rc, out = run_deck(exe_cmd, "in.test.crossproc",
                           extra=["-var", "dist", dist])
        if rc:
            fails.append("dist %s: run failed with exit code %d" % (dist, rc))
            continue
        rows = parse_stats(out)
        if not rows:
            fails.append("dist %s: no stats output" % dist)
            continue
        results[dist] = rows
    if fails:
        return fails
    if len(results["0"]) != len(results["1"]):
        return ["stats row counts differ: %d vs %d"
                % (len(results["0"]), len(results["1"]))]
    keys = [k for k in results["0"][0] if k.startswith("f_1")]
    for r0, r1 in zip(results["0"], results["1"]):
        for k in keys:
            if r0[k] != r1[k]:
                fails.append("step %d %s: distributed %.15g differs from "
                             "non-distributed %.15g"
                             % (int(r0["Step"]), k, r1[k], r0[k]))
        if fails:
            break
    # the bodies must actually have collided, else the contacts on
    # bodies stored by only some procs are not exercised
    if results["0"][-1]["f_1[1][4]"] > 100.0:
        fails.append("final body1 vx = %.6g, no body-body collision "
                     "occurred; test geometry is broken"
                     % results["0"][-1]["f_1[1][4]"])
    return fails


def test_nbody(exe_cmd):
    """Conservation laws of many-body push-off contacts.

    Four squares of different masses collide with each other in several
    pair and multi-body contacts, with no particles and no wall
    contacts.  The contact forces are equal-and-opposite and act at the
    same point on both bodies, so the total linear momentum and the total
    angular momentum about the origin (orbital plus spin) are conserved
    exactly; the elastic springs conserve the kinetic energy once the
    bodies separate."""
    rc, out = run_deck(exe_cmd, "in.test.nbody")
    if rc:
        return ["run failed with exit code %d" % rc]
    rows = parse_stats(out)
    if not rows:
        return ["no stats output"]
    mass = (1.0e-22, 2.0e-22, 3.0e-22, 4.0e-22)
    izz = (1.6667e-23, 3.3333e-23, 5.0e-23, 6.6667e-23)

    def invariants(r):
        px = py = lz = ke = 0.0
        for i in range(4):
            b = i + 1
            x, y = r["f_1[%d][1]" % b], r["f_1[%d][2]" % b]
            vx, vy = r["f_1[%d][4]" % b], r["f_1[%d][5]" % b]
            w = r["f_1[%d][15]" % b]
            px += mass[i] * vx
            py += mass[i] * vy
            lz += mass[i] * (x * vy - y * vx) + izz[i] * w
            ke += 0.5 * mass[i] * (vx * vx + vy * vy) + 0.5 * izz[i] * w * w
        return px, py, lz, ke

    px0, py0, lz0, ke0 = invariants(rows[0])
    pscale = sum(m * 40.0 for m in mass)
    lscale = sum(m * 40.0 * 10.0 for m in mass)
    fails = []
    for r in rows:
        px, py, lz, ke = invariants(r)
        if abs(px - px0) > 1e-9 * pscale or abs(py - py0) > 1e-9 * pscale:
            fails.append("step %d: momentum (%.10e,%.10e) drifted from "
                         "(%.10e,%.10e)"
                         % (int(r["Step"]), px, py, px0, py0))
        if abs(lz - lz0) > 1e-9 * lscale:
            fails.append("step %d: angular momentum %.10e drifted from "
                         "%.10e" % (int(r["Step"]), lz, lz0))
    px, py, lz, ke = invariants(rows[-1])
    if not approx(ke, ke0, rel=0.02):
        fails.append("final kinetic energy %.6e vs initial %.6e, not "
                     "conserved within 2%%" % (ke, ke0))
    # every body must have been deflected, else the contacts did not happen
    for i in range(4):
        b = i + 1
        dv = abs(rows[-1]["f_1[%d][4]" % b] - rows[0]["f_1[%d][4]" % b]) + \
            abs(rows[-1]["f_1[%d][5]" % b] - rows[0]["f_1[%d][5]" % b])
        if dv < 1.0:
            fails.append("body %d velocity changed by only %.3g, it was "
                         "not hit; test geometry is broken" % (b, dv))
    return fails


def test_idperm(exe_cmd):
    """The physics must not depend on how the bodies are numbered.

    Three squares colliding via push-off, run with body IDs (1,2,3) and
    with the same bodies numbered (3,1,2), which changes the order of the
    body element table and of every per-body loop.  Each body must follow
    the same trajectory in both runs, to the round-off of summing its
    contact forces in a different order."""
    runs = {}
    fails = []
    for perm, extra in (("123", []),
                        ("312", ["-var", "ta", "2", "-var", "tb", "0",
                                 "-var", "tc", "1", "-var", "bodies",
                                 "data.idperm.312.bodies"])):
        rc, out = run_deck(exe_cmd, "in.test.idperm", extra)
        if rc:
            fails.append("IDs %s: run failed with exit code %d" % (perm, rc))
            continue
        rows = parse_stats(out)
        if not rows:
            fails.append("IDs %s: no stats output" % perm)
            continue
        runs[perm] = rows
    if fails:
        return fails
    if len(runs["123"]) != len(runs["312"]):
        return ["stats row counts differ"]
    # body A is ID 1 then 3, B is 2 then 1, C is 3 then 2
    for name, b123, b312 in (("A", 1, 3), ("B", 2, 1), ("C", 3, 2)):
        for r1, r2 in zip(runs["123"], runs["312"]):
            for col in (1, 2, 4, 5, 15):
                v1 = r1["f_1[%d][%d]" % (b123, col)]
                v2 = r2["f_1[%d][%d]" % (b312, col)]
                if not approx(v1, v2, rel=1e-12, abs_=1e-12):
                    fails.append("step %d body %s column %d: %.15g with "
                                 "IDs (1,2,3) vs %.15g with IDs (3,1,2)"
                                 % (int(r1["Step"]), name, col, v1, v2))
            if fails:
                return fails
    # the bodies must have collided
    last = runs["123"][-1]
    if last["f_1[1][4]"] > 30.0:
        fails.append("final body A vx = %.6g, no collision occurred; "
                     "test geometry is broken" % last["f_1[1][4]"])
    return fails


def test_missingid(exe_cmd):
    return negative_test(exe_cmd, "in.test.missingid",
                         "Fix rigid body 2 has no surface elements")


TESTS = [
    ("ballistic", test_ballistic),
    ("force", test_force),
    ("rotation", test_rotation),
    ("bounce", test_bounce),
    ("restitution", test_restitution),
    ("momentum", test_momentum),
    ("recoil", test_recoil),
    ("overrun", test_overrun),
    ("remap", test_remap),
    ("multiremap", test_multiremap),
    ("transplane", test_transplane),
    ("splitbalance", test_splitbalance),
    ("staticdist", test_staticdist),
    ("staticdist3d", test_staticdist3d),
    ("splitcell", test_splitcell),
    ("gridchange", test_gridchange),
    ("exitbox", test_exitbox),
    ("restart", test_restart),
    ("twobody", test_twobody),
    ("pushpair", test_pushpair),
    ("badmoi", test_badmoi),
    ("notwatertight", test_notwatertight),
    ("zerothick", test_zerothick),
    ("inward", test_inward),
    ("refix", test_refix),
    ("prenofix", test_prenofix),
    ("vacate", test_vacate),
    ("facetbounce", test_facetbounce),
    ("modifyafter", test_modifyafter),
    ("wallmotion", test_wallmotion),
    ("customemit", test_customemit),
    ("emitsurf", test_emitsurf),
    ("renumber", test_renumber),
    ("mixture", test_mixture),
    ("badinfile", test_badinfile),
    ("rotwall", test_rotwall),
    ("rotwall3d", test_rotwall3d),
    ("axistuck", test_axistuck),
    ("tallyorder", test_tallyorder),
    ("react", test_react),
    ("badreact", test_badreact),
    ("recoilspin", test_recoil_spin),
    ("couple", test_couple),
    ("density", test_density),
    ("densityfar", test_density_far),
    ("density2d", test_density2d),
    ("baddensity", test_baddensity),
    ("badvcom", test_badvcom),
    ("axiballistic", test_axiballistic),
    ("axidensity", test_axidensity),
    ("aximomentum", test_aximomentum),
    ("axispin", test_axispin),
    ("axidrag", test_axidrag),
    ("axipush", test_axipush),
    ("axireact", test_axireact),
    ("axipair", test_axipair),
    ("axibadvcom", test_axibadvcom),
    ("axiopen", test_axiopen),
    ("nonfinite", test_nonfinite),
    ("crossproc", test_crossproc),
    ("nbody", test_nbody),
    ("idperm", test_idperm),
    ("missingid", test_missingid),
]

# tests whose decks support -var dist 1 (global surfs explicit/distributed)
# remap, multiremap, staticdist, and splitcell verify the incremental
# re-cut against the full rebuild in distributed mode; staticdist is the
# one whose static surfs are not local on every proc when run on
# several procs

DIST_TESTS = {"ballistic", "force", "rotation", "bounce", "restitution",
              "momentum",
              "overrun",
              "remap", "multiremap", "transplane", "staticdist",
              "staticdist3d",
              "splitcell", "gridchange", "exitbox", "twobody", "pushpair",
              "tallyorder", "rotwall", "rotwall3d", "customemit",
              "splitbalance",
              "vacate", "facetbounce", "nbody",
              "axiballistic", "axidensity", "aximomentum", "axispin",
              "axipush", "axireact", "axipair"}


def main():
    global run_deck
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True,
                        help="path to SPARTA executable")
    parser.add_argument("--mpi", default="",
                        help='MPI launcher prefix, e.g. "mpirun -np 4"')
    parser.add_argument("--args", default="",
                        help='extra SPARTA command-line args placed after '
                             'the executable, e.g. "-k on -sf kk"')
    parser.add_argument("--tests", default="",
                        help="comma-separated subset of tests to run")
    parser.add_argument("--dist", action="store_true",
                        help="run with distributed surfs "
                             "(global surfs explicit/distributed)")
    args = parser.parse_args()

    exe_cmd = (shlex.split(args.mpi) + [os.path.abspath(args.exe)] +
               shlex.split(args.args))

    subset = None
    if args.tests:
        subset = set(args.tests.split(","))

    if args.dist:
        if subset is None:
            subset = set(DIST_TESTS)
        else:
            subset &= DIST_TESTS
        base_run_deck = run_deck

        def dist_run_deck(exe_cmd, deck, extra=None):
            extra = (extra or []) + ["-var", "dist", "1"]
            return base_run_deck(exe_cmd, deck, extra)

        run_deck = dist_run_deck

    nfail = 0
    for name, func in TESTS:
        if subset and name not in subset:
            continue
        fails = func(exe_cmd)
        if fails:
            nfail += 1
            print("FAIL %s" % name)
            for f in fails:
                print("     %s" % f)
        else:
            print("PASS %s" % name)

    print("%d test(s) failed" % nfail)
    return nfail


if __name__ == "__main__":
    sys.exit(main())
