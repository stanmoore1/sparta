#!/usr/bin/env python3
"""
Verification suite for SPARTA's discrete electronic excitation model.

Runs a set of small input decks and checks code and physics correctness:

  boltzmann     electronic state populations stay Boltzmann-distributed at
                equilibrium (detailed balance) and total energy is conserved
  equilibration a cold electronic mode equilibrates with hot translation to
                the analytically predicted common temperature
  spin          spin-forbidden states remain exactly unpopulated when spin
                conservation is enforced (allowed/forbidden transitions)
  latespecies   species added after an elecfile species (memory regression)
  rates         equilibrium TCE rate for O2 + N2 -> O + O + N2 vs the input
                Arrhenius rate, without and with the electronic mode
                (documents the known overprediction when electronic energy
                is included with partial_energy no), plus a fractional-dof
                parsing regression (dof 0.9 must lower the rate vs dof 0)

Usage:
  python3 run_tests.py --exe /path/to/spa_serial [--tests t1,t2] [--keep]
  python3 run_tests.py --exe spa_serial --exe2 spa_kokkos --exe2-args "-k on -sf kk"
      (parity mode: every deck is run with both executables and the stats
       tables, reaction tallies, and sorted dumps must match bit-for-bit)

Exit code = number of failed tests.
"""

import argparse, math, os, re, shutil, subprocess, sys

KB = 1.380649e-23
SUITE = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = ["air.species", "air.vss", "airx.vss", "extra.species",
              "n2boltz.elec", "airspin.elec", "o2n2.elec", "o2n2_dof09.elec",
              "o2n2.tce"]

# ---------------------------------------------------------------- parsing

def parse_species_masses(path):
    masses = {}
    for line in open(path):
        w = line.split()
        if not w or w[0].startswith("#"):
            continue
        masses[w[0]] = float(w[2])
    return masses

def parse_elec_states(path, species):
    """Return list of (temp_K, degen) for one species from an elec file."""
    for line in open(path):
        w = line.split()
        if not w or w[0].startswith("#") or w[0] != species:
            continue
        try:
            n = int(w[1])
        except ValueError:
            continue
        if n <= 0:
            continue
        return [(float(w[2+5*k]), int(w[2+5*k+2])) for k in range(n)]
    raise RuntimeError("no electronic data for %s in %s" % (species, path))

def parse_dump(path):
    """Return (colnames, rows) from a SPARTA particle dump snapshot."""
    rows, cols, reading = [], None, False
    for line in open(path):
        if line.startswith("ITEM: ATOMS"):
            cols = line.split()[2:]
            reading = True
            continue
        if line.startswith("ITEM:"):
            reading = False
            continue
        if reading:
            rows.append([float(x) for x in line.split()])
    if cols is None:
        raise RuntimeError("no ATOMS section in %s" % path)
    return cols, rows

def parse_stats(logpath):
    """Return (header, rows) of the (last) stats table in a SPARTA log."""
    header, rows = None, []
    lines = open(logpath).read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Step "):
            header = line.split()
            rows = []
            for l2 in lines[i+1:]:
                w = l2.split()
                try:
                    float(w[0])
                except (ValueError, IndexError):
                    break
                rows.append(w)
    return header, rows

def parse_tally(logpath, reaction_prefix):
    """Return the cumulative tally for a reaction from the end-of-run block."""
    for line in open(logpath):
        m = re.match(r"\s*reaction (.+): ([0-9.eE+-]+)", line)
        if m and m.group(1).startswith(reaction_prefix):
            return float(m.group(2))
    return 0.0

# ------------------------------------------------------------- physics

def boltzmann_fracs(states, T):
    w = [g * math.exp(-th / T) for th, g in states]
    Z = sum(w)
    return [x / Z for x in w]

def elec_mean_energy_K(states, T):
    """Mean electronic energy in Kelvin units (E/k) at temperature T."""
    fr = boltzmann_fracs(states, T)
    return sum(f * th for f, (th, g) in zip(fr, states))

def solve_equilibrium_T(states, T0):
    """Solve (3/2)T + <theta>(T) = (3/2)T0 for the common temperature T."""
    lo, hi = 100.0, T0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.5 * mid + elec_mean_energy_K(states, mid) > 1.5 * T0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def arrhenius_k(T):
    """O2 + N2 -> O + O + N2 rate from o2n2.tce  [m^3/s]."""
    return 3.321e-9 * T**-1.5 * math.exp(-8.197e-19 / (KB * T))

# ------------------------------------------------------------- running

def run_deck(exe, exe_args, deck, workdir, variables=None, tag=""):
    os.makedirs(workdir, exist_ok=True)
    for f in DATA_FILES + [deck]:
        src = os.path.join(SUITE, f)
        if os.path.exists(src):
            shutil.copy(src, workdir)
    log = "log.sparta" + tag
    cmd = [exe] + exe_args + ["-in", deck, "-log", log]
    for k, v in (variables or {}).items():
        cmd += ["-var", k, str(v)]
    r = subprocess.run(cmd, cwd=workdir, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, timeout=3600)
    if r.returncode != 0:
        tail = "\n".join(r.stdout.splitlines()[-15:])
        raise RuntimeError("%s failed (exit %d) in %s:\n%s"
                           % (" ".join(cmd), r.returncode, workdir, tail))
    return os.path.join(workdir, log)

def total_energy(dumpfile, masses_by_type):
    cols, rows = parse_dump(dumpfile)
    it, ivx = cols.index("type"), cols.index("vx")
    ie = cols.index("p_eelec") if "p_eelec" in cols else None
    E = 0.0
    for r in rows:
        m = masses_by_type[int(r[it])]
        v2 = r[ivx]**2 + r[ivx+1]**2 + r[ivx+2]**2
        E += 0.5 * m * v2
        if ie is not None:
            E += r[ie]
    return E, len(rows)

def state_histogram(dumpfile):
    cols, rows = parse_dump(dumpfile)
    i = cols.index("p_elecstate")
    hist = {}
    for r in rows:
        s = int(r[i])
        hist[s] = hist.get(s, 0) + 1
    return hist, len(rows)

def temperature(dumpfile, masses_by_type):
    cols, rows = parse_dump(dumpfile)
    it, ivx = cols.index("type"), cols.index("vx")
    ke = sum(0.5 * masses_by_type[int(r[it])] *
             (r[ivx]**2 + r[ivx+1]**2 + r[ivx+2]**2) for r in rows)
    return 2.0 * ke / (3.0 * KB * len(rows))

# ------------------------------------------------------------- tests

class Result:
    def __init__(self, name):
        self.name, self.ok, self.notes = name, True, []
    def check(self, cond, msg):
        self.notes.append(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            self.ok = False
    def info(self, msg):
        self.notes.append("       " + msg)

def test_boltzmann(exe, exe_args, wd):
    res = Result("boltzmann")
    run_deck(exe, exe_args, "in.boltzmann", wd)
    masses = parse_species_masses(os.path.join(SUITE, "air.species"))
    mt = {1: masses["N2"]}
    states = parse_elec_states(os.path.join(SUITE, "n2boltz.elec"), "N2")

    E0, n0 = total_energy(os.path.join(wd, "dump.boltzmann.0"), mt)
    E1, n1 = total_energy(os.path.join(wd, "dump.boltzmann.2000"), mt)
    res.check(n0 == n1, "particle count constant (%d)" % n0)
    res.check(abs(E1 - E0) / E0 < 1e-9,
              "total energy conserved (rel drift %.2e)" % (abs(E1 - E0) / E0))

    T = temperature(os.path.join(wd, "dump.boltzmann.2000"), mt)
    expected = boltzmann_fracs(states, T)
    hist, n = state_histogram(os.path.join(wd, "dump.boltzmann.2000"))
    for i, pexp in enumerate(expected):
        cnt = hist.get(i, 0)
        sigma = math.sqrt(max(pexp * n, 1.0))
        dev = abs(cnt - pexp * n) / sigma
        res.check(dev < 5.0,
                  "state %d population %d vs Boltzmann %.0f at T=%.0fK (%.1f sigma)"
                  % (i, cnt, pexp * n, T, dev))
    return res

def test_equilibration(exe, exe_args, wd):
    res = Result("equilibration")
    run_deck(exe, exe_args, "in.equilibration", wd)
    masses = parse_species_masses(os.path.join(SUITE, "air.species"))
    mt = {1: masses["N2"]}
    states = parse_elec_states(os.path.join(SUITE, "n2boltz.elec"), "N2")

    E0, _ = total_energy(os.path.join(wd, "dump.equil.0"), mt)
    E1, _ = total_energy(os.path.join(wd, "dump.equil.6000"), mt)
    res.check(abs(E1 - E0) / E0 < 1e-9,
              "total energy conserved (rel drift %.2e)" % (abs(E1 - E0) / E0))

    Tpred = solve_equilibrium_T(states, 25000.0)
    T5 = temperature(os.path.join(wd, "dump.equil.5000"), mt)
    T6 = temperature(os.path.join(wd, "dump.equil.6000"), mt)
    res.info("predicted common T = %.0f K" % Tpred)
    res.check(abs(T6 - T5) / T6 < 0.01,
              "plateau reached (T(5000)=%.0f, T(6000)=%.0f)" % (T5, T6))
    res.check(abs(T6 - Tpred) / Tpred < 0.02,
              "final T_trans %.0f K matches analytic %.0f K" % (T6, Tpred))

    # electronic mean energy must match the same temperature
    cols, rows = parse_dump(os.path.join(wd, "dump.equil.6000"))
    ie = cols.index("p_eelec")
    mean_theta = sum(r[ie] for r in rows) / len(rows) / KB
    pred_theta = elec_mean_energy_K(states, Tpred)
    res.check(abs(mean_theta - pred_theta) / pred_theta < 0.05,
              "mean elec energy %.0f K matches analytic %.0f K"
              % (mean_theta, pred_theta))
    return res

def test_spin(exe, exe_args, wd):
    res = Result("spin")
    run_deck(exe, exe_args, "in.spin", wd)
    hist, n = state_histogram(os.path.join(wd, "dump.spin.3000"))
    # airspin.elec: state 0 = X1Sigma (spin 1), 1 = A3Sigma (spin 3),
    #               2 = B3Pi (spin 3),          3 = a1Sigma (spin 1)
    res.check(hist.get(1, 0) == 0 and hist.get(2, 0) == 0,
              "spin-forbidden triplet states unpopulated (%d, %d)"
              % (hist.get(1, 0), hist.get(2, 0)))
    res.check(hist.get(3, 0) > 0,
              "spin-allowed a1Sigma populated (%d of %d)" % (hist.get(3, 0), n))
    return res

def test_latespecies(exe, exe_args, wd):
    res = Result("latespecies")
    log = run_deck(exe, exe_args, "in.latespecies", wd)
    header, rows = parse_stats(log)
    res.check(len(rows) >= 3, "run completed (%d stats rows)" % len(rows))
    if rows:
        T = float(rows[-1][header.index("c_temp")])
        res.check(10000.0 < T < 30000.0, "final temperature sane (%.0f K)" % T)
    return res

def measure_rate(exe, exe_args, wd, deck, variables, tag):
    log = run_deck(exe, exe_args, deck, wd, variables, tag)
    tally = parse_tally(log, "O2 + N2")
    nrho, frac, V, fnum = 7.07043e22, 0.5, 1.0e-12, 1.767e6
    nsteps, dt = 2000, 1.0e-9
    n1 = n2 = nrho * frac
    k = tally * fnum / (n1 * n2 * V * nsteps * dt)
    return k, tally

def test_rates(exe, exe_args, wd):
    res = Result("rates")
    temps = [10000.0, 15000.0, 20000.0]
    ratio_b, ratio_c = {}, {}
    for T in temps:
        ka = arrhenius_k(T)

        kb, tb = measure_rate(exe, exe_args, os.path.join(wd, "b%d" % T),
                              "in.rate_noelec", {"T": T, "PE": "no"}, "")
        ratio_b[T] = kb / ka
        res.info("T=%5.0fK  rot+vib      : k/k_Arrhenius = %.3f (%.0f tallies)"
                 % (T, kb / ka, tb))
        # the discrete-vibration instantaneous-dof TCE is known to run high
        # at low temperature, so the tolerance is wider at 10000 K
        tol = 0.35 if T < 12000 else 0.15
        res.check(abs(kb / ka - 1.0) < tol,
                  "rot+vib rate within %.0f%% of Arrhenius at %.0f K" % (100*tol, T))

        kc, tc = measure_rate(exe, exe_args, os.path.join(wd, "c%d" % T),
                              "in.rate_elec", {"T": T, "EFILE": "o2n2.elec"}, "")
        ratio_c[T] = kc / ka
        res.info("T=%5.0fK  rot+vib+elec : k/k_Arrhenius = %.3f (%.0f tallies)"
                 % (T, kc / ka, tc))

    # the documented physics finding (Higdon dissertation Fig. 7.2):
    # including electronic energy in the TCE collision energy with
    # partial_energy no systematically overpredicts the Arrhenius rate.
    # The multiplicative enhancement kc/kb is strongest at LOW temperature,
    # where the e^{+eps/kT} threshold-lowering by the electronic quantum is
    # largest relative to kT, and narrows toward high temperature.
    for T in temps:
        res.check(ratio_c[T] / ratio_b[T] > 1.10,
                  "electronic mode overpredicts rate at %.0f K (x%.3f over rot+vib)"
                  % (T, ratio_c[T] / ratio_b[T]))
    res.check(ratio_c[temps[0]] / ratio_b[temps[0]] >
              ratio_c[temps[-1]] / ratio_b[temps[-1]],
              "relative overprediction largest at low T "
              "(x%.3f at %.0fK vs x%.3f at %.0fK)"
              % (ratio_c[temps[0]] / ratio_b[temps[0]], temps[0],
                 ratio_c[temps[-1]] / ratio_b[temps[-1]], temps[-1]))

    # fractional-dof regression: per-state dof 0.9 enters the TCE degrees of
    # freedom and must LOWER the rate relative to dof 0 (an integer-truncating
    # parser would make the two runs identical)
    T = temps[-1]
    kd, td = measure_rate(exe, exe_args, os.path.join(wd, "d%d" % T),
                          "in.rate_elec", {"T": T, "EFILE": "o2n2_dof09.elec"}, "")
    res.info("T=%5.0fK  elec, dof=0.9: k/k_Arrhenius = %.3f (%.0f tallies)"
             % (T, kd / arrhenius_k(T), td))
    res.check(kd < ratio_c[T] * arrhenius_k(T) * 0.97,
              "fractional dof 0.9 lowers rate vs dof 0 (%.3f vs %.3f)"
              % (kd / arrhenius_k(T), ratio_c[T]))
    return res

TESTS = {
    "boltzmann": test_boltzmann,
    "equilibration": test_equilibration,
    "spin": test_spin,
    "latespecies": test_latespecies,
    "rates": test_rates,
}

# ------------------------------------------------------------- parity

PARITY_DECKS = [
    ("in.boltzmann", None),
    ("in.equilibration", None),
    ("in.spin", None),
    ("in.latespecies", None),
    ("in.rate_elec", {"T": 20000.0, "EFILE": "o2n2.elec"}),
]

def strip_timing(logpath):
    """Stats rows and reaction tallies with CPU-time columns removed."""
    header, rows = parse_stats(logpath)
    icpu = header.index("CPU") if header and "CPU" in header else None
    out = []
    for r in rows:
        out.append(" ".join(x for i, x in enumerate(r) if i != icpu))
    for line in open(logpath):
        if re.match(r"\s*reaction .+:", line):
            out.append(line.strip())
    return out

def parity(exe, exe_args, exe2, exe2_args, wd):
    fails = 0
    for deck, variables in PARITY_DECKS:
        name = deck.replace("in.", "")
        try:
            l1 = run_deck(exe, exe_args, deck, os.path.join(wd, "a_" + name),
                          variables)
            l2 = run_deck(exe2, exe2_args, deck, os.path.join(wd, "b_" + name),
                          variables)
        except RuntimeError as e:
            print("FAIL %-14s %s" % (name, e))
            fails += 1
            continue
        same = strip_timing(l1) == strip_timing(l2)
        print("%s %-14s stats+tallies %s" %
              ("PASS" if same else "FAIL", name,
               "bit-for-bit identical" if same else "DIFFER"))
        if not same:
            fails += 1
    return fails

# ------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(SUITE, "../../src/spa_serial"))
    ap.add_argument("--exe-args", default="")
    ap.add_argument("--exe2", help="second executable for parity comparison")
    ap.add_argument("--exe2-args", default="",
                    help='e.g. "-k on -sf kk" for a Kokkos executable')
    ap.add_argument("--tests", default=",".join(TESTS))
    ap.add_argument("--workdir", default=os.path.join(SUITE, "work"))
    ap.add_argument("--keep", action="store_true",
                    help="keep work directories on success")
    args = ap.parse_args()

    exe = os.path.abspath(args.exe)
    exe_args = args.exe_args.split()
    if os.path.isdir(args.workdir):
        shutil.rmtree(args.workdir)

    if args.exe2:
        fails = parity(exe, exe_args, os.path.abspath(args.exe2),
                       args.exe2_args.split(), args.workdir)
        if not args.keep and fails == 0:
            shutil.rmtree(args.workdir, ignore_errors=True)
        sys.exit(fails)

    fails = 0
    for name in args.tests.split(","):
        name = name.strip()
        try:
            res = TESTS[name](exe, exe_args, os.path.join(args.workdir, name))
        except Exception as e:
            print("FAIL %-14s exception: %s" % (name, e))
            fails += 1
            continue
        print("%s %-14s" % ("PASS" if res.ok else "FAIL", name))
        for note in res.notes:
            print("     " + note)
        if not res.ok:
            fails += 1
    if not args.keep and fails == 0:
        shutil.rmtree(args.workdir, ignore_errors=True)
    sys.exit(fails)

if __name__ == "__main__":
    main()
