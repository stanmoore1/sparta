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
              "partners.species", "n2boltz.elec", "airspin.elec",
              "o2n2.elec", "o2n2_dof09.elec", "o_atom.elec", "o_atom.vss",
              "n2_specrel.elec", "specrel.vss", "n2_rate.elec.tmpl",
              "o2n2.tce"]

DISS_ENERGY = 8.197e-19  # O2 + N2 -> O + O + N2 reaction energy (J), o2n2.tce coeff[4]

LAUNCHER = []  # process launcher prefix (e.g. mpirun); set by --mpi-np

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

def run_deck(exe, exe_args, deck, workdir, variables=None, tag="", launcher=None,
             extra_files=None):
    os.makedirs(workdir, exist_ok=True)
    for f in DATA_FILES + (extra_files or []) + [deck]:
        src = os.path.join(SUITE, f)
        if os.path.exists(src):
            shutil.copy(src, workdir)
    log = "log.sparta" + tag
    if launcher is None:
        launcher = LAUNCHER
    cmd = launcher + [exe] + exe_args + ["-in", deck, "-log", log]
    for k, v in (variables or {}).items():
        cmd += ["-var", k, str(v)]
    r = subprocess.run(cmd, cwd=workdir, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, timeout=3600)
    if r.returncode != 0:
        tail = "\n".join(r.stdout.splitlines()[-20:])
        raise RuntimeError("%s failed (exit %d) in %s:\n%s"
                           % (" ".join(cmd), r.returncode, workdir, tail))
    return os.path.join(workdir, log)

def total_energy_full(dumpfile, masses_by_type):
    """Kinetic + rotational + vibrational + electronic energy over all particles."""
    cols, rows = parse_dump(dumpfile)
    it, ivx = cols.index("type"), cols.index("vx")
    ir = cols.index("erot") if "erot" in cols else None
    iv = cols.index("evib") if "evib" in cols else None
    ie = cols.index("p_eelec") if "p_eelec" in cols else None
    E = 0.0
    for r in rows:
        m = masses_by_type[int(r[it])]
        E += 0.5 * m * (r[ivx]**2 + r[ivx+1]**2 + r[ivx+2]**2)
        if ir is not None: E += r[ir]
        if iv is not None: E += r[iv]
        if ie is not None: E += r[ie]
    return E, len(rows)

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

def eelec_series(logpath):
    """Return {step: c_Eelec} from a stats table that includes c_Eelec."""
    header, rows = parse_stats(logpath)
    istep, ie = header.index("Step"), header.index("c_Eelec")
    return {int(float(r[istep])): float(r[ie]) for r in rows}

def test_atom(exe, exe_args, wd):
    # #4: electronic-only relaxation for an atomic species (elec_exchange path,
    # ave_dof == 0). O has zero rot/vib dof, so relaxation happens only if the
    # atomic electronic path is taken.
    res = Result("atom")
    run_deck(exe, exe_args, "in.atom", wd)
    masses = parse_species_masses(os.path.join(SUITE, "air.species"))
    mt = {1: masses["O"]}
    states = parse_elec_states(os.path.join(SUITE, "o_atom.elec"), "O")

    E0, n0 = total_energy(os.path.join(wd, "dump.atom.0"), mt)
    E1, n1 = total_energy(os.path.join(wd, "dump.atom.4000"), mt)
    res.check(abs(E1 - E0) / E0 < 1e-9,
              "total energy conserved (rel drift %.2e)" % (abs(E1 - E0) / E0))

    hist0, _ = state_histogram(os.path.join(wd, "dump.atom.0"))
    res.check(set(hist0) == {0},
              "all atoms start in ground state (hist0=%s)" % hist0)
    T = temperature(os.path.join(wd, "dump.atom.4000"), mt)
    expected = boltzmann_fracs(states, T)
    hist, n = state_histogram(os.path.join(wd, "dump.atom.4000"))
    res.check(hist.get(1, 0) + hist.get(2, 0) > 0,
              "excited atomic states populated by collisions (%d in states>0)"
              % (n - hist.get(0, 0)))
    for i, pexp in enumerate(expected):
        cnt = hist.get(i, 0)
        sigma = math.sqrt(max(pexp * n, 1.0))
        res.check(abs(cnt - pexp * n) / sigma < 5.0,
                  "O state %d population %d vs Boltzmann %.0f at T=%.0fK (%.1f sigma)"
                  % (i, cnt, pexp * n, T, abs(cnt - pexp * n) / sigma))
    return res

def test_specrel(exe, exe_args, wd):
    # #2: species-specific relaxation numbers (get_elec_phi species_rel branch).
    # N2 relaxes faster against fast partner F1 than slow partner F2.
    res = Result("specrel")
    ef = ["partners.species", "n2_specrel.elec", "specrel.vss"]
    lf = run_deck(exe, exe_args, "in.specrel", os.path.join(wd, "fast"),
                  {"PARTNER": "F1"}, extra_files=ef)
    ls = run_deck(exe, exe_args, "in.specrel", os.path.join(wd, "slow"),
                  {"PARTNER": "F2"}, extra_files=ef)
    ef_fast = eelec_series(lf)
    ef_slow = eelec_series(ls)
    early = sorted(ef_fast)[1]           # first stats output after step 0
    res.info("at step %d: eelec(fast F1)=%.3e  eelec(slow F2)=%.3e"
             % (early, ef_fast[early], ef_slow[early]))
    res.check(ef_fast[early] > 3.0 * ef_slow[early],
              "N2 relaxes much faster vs fast partner (ratio %.1fx)"
              % (ef_fast[early] / max(ef_slow[early], 1e-30)))
    res.check(ef_slow[max(ef_slow)] > ef_slow[early],
              "slow partner keeps relaxing over time (default_rel fallback works)")
    return res

def test_relaxrate(exe, exe_args, wd):
    # #1: relaxation RATE tracks the input collision number. Generate elec files
    # with high vs low relaxation probability and check the transient.
    res = Result("relaxrate")
    tmpl = open(os.path.join(SUITE, "n2_rate.elec.tmpl")).read()
    series = {}
    for name, phi in [("hi", 0.5), ("lo", 0.05)]:
        d = os.path.join(wd, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "n2_rate.elec"), "w") as f:
            f.write(tmpl.replace("PHI", repr(phi)))
        log = run_deck(exe, exe_args, "in.rate_relax", d,
                       extra_files=["n2_rate.elec"])
        series[name] = eelec_series(log)
    # at an early step the high-phi gas must have relaxed more electronic energy
    early = sorted(series["hi"])[1]
    hi, lo = series["hi"][early], series["lo"][early]
    res.info("at step %d: eelec(phi=0.5)=%.3e  eelec(phi=0.05)=%.3e" % (early, hi, lo))
    res.check(hi > 2.0 * lo,
              "higher collision number relaxes faster early (ratio %.2fx)"
              % (hi / max(lo, 1e-30)))
    # both approach the same equilibrium by the end
    last = max(series["hi"])
    he, le = series["hi"][last], series["lo"][last]
    res.check(abs(he - le) / he < 0.15,
              "both reach ~same equilibrium eelec (%.3e vs %.3e)" % (he, le))
    return res

def test_reactions(exe, exe_args, wd):
    # #3: exact energy conservation across REAL reactions with electronic energy.
    # Each dissociation removes DISS_ENERGY from kinetic+internal energy.
    res = Result("reactions")
    log = run_deck(exe, exe_args, "in.react", wd)
    masses = parse_species_masses(os.path.join(SUITE, "air.species"))
    mt = {i + 1: masses[s] for i, s in enumerate(["N2", "O2", "O", "N"])}
    E0, n0 = total_energy_full(os.path.join(wd, "dump.react.0"), mt)
    E1, n1 = total_energy_full(os.path.join(wd, "dump.react.1000"), mt)
    nreact = parse_tally(log, "O2 + N2")
    res.check(nreact > 100, "reactions actually occurred (%d)" % nreact)
    res.check(n1 > n0, "particle count grew from dissociation (%d -> %d)" % (n0, n1))
    drop = E0 - E1
    predicted = nreact * DISS_ENERGY
    rel = abs(drop - predicted) / predicted
    res.info("KE+int drop = %.4e J ; reactions x E_diss = %.4e J" % (drop, predicted))
    res.check(rel < 0.02,
              "energy drop equals reactions x dissociation energy (rel err %.3f)" % rel)
    return res

def test_restart(exe, exe_args, wd):
    # #5: electronic custom data (elecstate, eelec) survives write/read restart.
    # Continuous run vs restart+continue must give identical particle state.
    res = Result("restart")
    run_deck(exe, exe_args, "in.restart_write", wd)
    run_deck(exe, exe_args, "in.restart_read", wd)  # same wd: reads elec.restart
    pre = parse_dump(os.path.join(wd, "dump.pre.100"))
    post = parse_dump(os.path.join(wd, "dump.post.100"))
    res.check(pre[0] == post[0], "dump columns match")
    ci = pre[0].index("id"); ce = pre[0].index("p_eelec"); cs = pre[0].index("p_elecstate")
    a = sorted(pre[1], key=lambda r: r[ci])
    b = sorted(post[1], key=lambda r: r[ci])
    res.check(len(a) == len(b) and len(a) > 0, "same particle count (%d)" % len(a))
    nexc = sum(1 for r in a if int(r[cs]) > 0)
    res.info("%d of %d particles are in an excited electronic state" % (nexc, len(a)))
    res.check(nexc > 0, "some particles excited (else the test is vacuous)")
    mism = sum(1 for ra, rb in zip(a, b)
               if int(ra[cs]) != int(rb[cs]) or ra[ce] != rb[ce])
    res.check(mism == 0,
              "elecstate+eelec identical after restart round-trip (%d mismatches)" % mism)
    return res

def test_telec(exe, exe_args, wd):
    # #6: compute telec/grid recovers the initialization temperature.
    res = Result("telec")
    for T in [8000.0, 15000.0]:
        log = run_deck(exe, exe_args, "in.telec", os.path.join(wd, "t%d" % int(T)),
                       {"TELEC": T})
        header, rows = parse_stats(log)
        # c_Telec[1] is the per-species electronic temperature at step 0
        icol = [i for i, h in enumerate(header) if h.startswith("c_Telec")][0]
        t0 = float(rows[0][icol])
        res.info("init telec=%.0fK -> compute telec/grid=%.0fK" % (T, t0))
        res.check(abs(t0 - T) / T < 0.05,
                  "telec/grid recovers %.0f K (got %.0f K)" % (T, t0))
    return res

TESTS = {
    "boltzmann": test_boltzmann,
    "equilibration": test_equilibration,
    "spin": test_spin,
    "latespecies": test_latespecies,
    "rates": test_rates,
    "atom": test_atom,
    "specrel": test_specrel,
    "relaxrate": test_relaxrate,
    "reactions": test_reactions,
    "restart": test_restart,
    "telec": test_telec,
}

# ---------------------------------------------- parity / snapshot / mpi

# single-run decks (name, variables) used for exact-output comparisons.
# These cover the main code paths: relaxation, equilibration, spin,
# late-species, atomic electronic-only, and the reacting/TCE path.
SINGLE_RUN_DECKS = [
    ("in.boltzmann", None),
    ("in.equilibration", None),
    ("in.spin", None),
    ("in.latespecies", None),
    ("in.atom", None),
    ("in.rate_elec", {"T": 20000.0, "EFILE": "o2n2.elec"}),
    ("in.react", None),
]

def strip_timing(logpath):
    """Stats rows and reaction tallies with the CPU-time column removed."""
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
    for deck, variables in SINGLE_RUN_DECKS:
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

SNAP_DIR = os.path.join(SUITE, "snapshots")

def snapshot(exe, exe_args, wd, bless):
    """Pin each deck's exact stats+tallies against a committed reference.
    With bless=True, (re)write the references instead of comparing.
    A refactor that changes any observable output trips this."""
    if bless:
        os.makedirs(SNAP_DIR, exist_ok=True)
    fails = 0
    for deck, variables in SINGLE_RUN_DECKS:
        name = deck.replace("in.", "")
        ref = os.path.join(SNAP_DIR, name + ".txt")
        try:
            log = run_deck(exe, exe_args, deck, os.path.join(wd, name), variables)
        except RuntimeError as e:
            print("FAIL %-14s %s" % (name, e)); fails += 1; continue
        got = strip_timing(log)
        if bless:
            with open(ref, "w") as f:
                f.write("\n".join(got) + "\n")
            print("BLESS %-14s (%d lines)" % (name, len(got)))
            continue
        if not os.path.exists(ref):
            print("FAIL %-14s no reference (run --bless first)" % name); fails += 1; continue
        want = open(ref).read().splitlines()
        if got == want:
            print("PASS %-14s matches snapshot (%d lines)" % (name, len(got)))
        else:
            nd = sum(1 for a, b in zip(got, want) if a != b) + abs(len(got) - len(want))
            print("FAIL %-14s differs from snapshot (%d lines changed)" % (name, nd))
            fails += 1
    return fails

def mpi_suite(exe, exe_args, wd, np):
    """Run the physics test suite under mpirun -np NP. DSMC is not bit-identical
    across rank counts, so this checks that the (tolerance-based) physics tests
    still pass in parallel. Also checks same-rank determinism run-to-run."""
    launcher = ["mpirun", "--oversubscribe", "-np", str(np)]
    fails = 0
    # (a) physics suite under MPI
    global LAUNCHER
    LAUNCHER = launcher
    for name, fn in TESTS.items():
        try:
            res = fn(exe, exe_args, os.path.join(wd, name))
            ok = res.ok
        except Exception as e:
            print("FAIL %-14s exception: %s" % (name, e)); fails += 1; continue
        print("%s %-14s (np=%d)" % ("PASS" if ok else "FAIL", name, np))
        if not ok:
            for n in res.notes:
                print("     " + n)
            fails += 1
    LAUNCHER = []
    # (b) same-rank determinism: identical physics run-to-run at NP ranks
    a = run_deck(exe, exe_args, "in.boltzmann", os.path.join(wd, "det_a"),
                 launcher=launcher)
    b = run_deck(exe, exe_args, "in.boltzmann", os.path.join(wd, "det_b"),
                 launcher=launcher)
    det = strip_timing(a) == strip_timing(b)
    print("%s %-14s (np=%d run-to-run identical)"
          % ("PASS" if det else "FAIL", "determinism", np))
    if not det:
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
    ap.add_argument("--snapshot", action="store_true",
                    help="compare exact stats+tally output against committed snapshots")
    ap.add_argument("--bless", action="store_true",
                    help="(re)write snapshot reference files instead of comparing")
    ap.add_argument("--mpi-np", type=int, default=0,
                    help="run the physics suite under mpirun -np N (parallel correctness)")
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

    if args.snapshot or args.bless:
        fails = snapshot(exe, exe_args, args.workdir, args.bless)
        if not args.keep and fails == 0:
            shutil.rmtree(args.workdir, ignore_errors=True)
        sys.exit(fails)

    if args.mpi_np:
        fails = mpi_suite(exe, exe_args, args.workdir, args.mpi_np)
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
