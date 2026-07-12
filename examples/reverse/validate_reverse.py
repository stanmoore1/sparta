#!/usr/bin/env python3
"""Quantitative validation of detailed-balance reverse reactions (issue #472).

Recomputes the species partition functions and equilibrium constants
INDEPENDENTLY of the SPARTA implementation (from the same data files), then
checks the measured forward/backward rates from frozen-composition
reservoirs against them:

  1  exchange pair N2 + O <-> NO + N: the measured tally ratio must equal
     the analytic equilibrium constant Keq(T) at several temperatures
  2  the derived backward rate must lie close to the independent
     literature Arrhenius fit for NO + N -> N2 + O (data/air.tce)
  3  dissociation/recombination pair N2 + N <-> N + N + N: the measured
     tally ratio must equal the analytic volumetric equilibrium constant
  4  a reacting (non-frozen) box initialized off-equilibrium must relax
     toward the analytic equilibrium composition from both sides

Usage:  python3 validate_reverse.py --exe ../../src/spa_serial
Exit code = number of failed checks.
"""

import argparse, math, os, re, shutil, subprocess, sys, tempfile

KB = 1.380649e-23
H  = 6.62607015e-34
HERE = os.path.dirname(os.path.abspath(__file__))

FAIL = 0
def check(name, ok, detail=""):
    global FAIL
    if not ok: FAIL += 1
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  " + detail) if detail else ""))

# ---------------------------------------------------------------- data

def parse_species(path):
    sp = {}
    for line in open(path):
        w = line.split()
        if not w or w[0].startswith("#"): continue
        # id molwt molmass rotdof rotrel vibdof vibrel vibtemp specwt charge
        sp[w[0]] = {"mass": float(w[2]), "rotdof": int(w[3]),
                    "vibdof": int(w[5]), "vibtemp": float(w[7])}
    return sp

def parse_rot(path):
    rot = {}
    for line in open(path):
        w = line.split()
        if not w or w[0].startswith("#"): continue
        n = int(w[1])
        temps = [float(x) for x in w[2:2+n]]
        sigma = float(w[2+n]) if len(w) > 2+n else 1.0
        rot[w[0]] = (temps, sigma)
    return rot

def parse_elec(path):
    elec = {}
    for line in open(path):
        w = line.split()
        if not w or w[0].startswith("#"): continue
        n = int(w[1])
        states = []
        for k in range(n):
            t = float(w[2+5*k]); g = float(w[2+5*k+2])
            states.append((t, g))
        elec[w[0]] = states
    return elec

SPECIES = parse_species(os.path.join(HERE,"air.species"))
ROT = parse_rot(os.path.join(HERE,"air.rot"))
ELEC = parse_elec(os.path.join(HERE,"air.elec"))

def q_total(name, T, vibsmooth=False):
    """partition function per unit volume, matching ReactBird exactly;
    with vibsmooth the classical harmonic vib form (q = T/theta per mode),
    matching partition_function under collide_modify vibrate smooth"""
    s = SPECIES[name]
    q = (2.0*math.pi*s["mass"]*KB*T/(H*H))**1.5
    if s["rotdof"] == 2 and name in ROT:
        temps, sigma = ROT[name]
        q *= T/(sigma*temps[0])
    if s["vibdof"] >= 2 and s["vibtemp"] > 0.0:
        if vibsmooth: q *= T/s["vibtemp"]
        else:         q *= 1.0/(1.0 - math.exp(-s["vibtemp"]/T))
    if name in ELEC:
        q *= sum(g*math.exp(-t/T) for t,g in ELEC[name])
    return q

def park_fit(func, Ts):
    """solve for Park coefficients c0..c4 with
    ln f(T) = c0/Z + c1 + c2 ln Z + c3 Z + c4 Z^2,  Z = 1e4/T,
    exactly through the 5 given temperatures (pure-python 5x5 solve)"""
    A, y = [], []
    for T in Ts:
        Z = 1e4/T
        A.append([1.0/Z, 1.0, math.log(Z), Z, Z*Z])
        y.append(math.log(func(T)))
    # Gaussian elimination with partial pivoting
    n = 5
    M = [A[i][:] + [y[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        for j in range(col, n+1): M[col][j] /= d
        for r in range(n):
            if r == col: continue
            f = M[r][col]
            for j in range(col, n+1): M[r][j] -= f*M[col][j]
    return [M[i][n] for i in range(n)]

# reaction data (must match rev.tce)
A_F, B_F, EA_F, DH_F = 1.069e-12, -1.0, 5.175e-19, -5.175e-19   # N2+O -> NO+N
A_D, B_D, EA_D, DH_D = 4.980e-8, -1.5, 1.561e-18, -1.561e-18    # N2+N -> N+N+N
# independent literature fit for NO + N -> N2 + O (data/air.tce)
A_L, B_L, EA_L = 4.059e-12, -1.359, 0.0

def kf(T):  return A_F*T**B_F*math.exp(-EA_F/(KB*T))
def kb_lit(T): return A_L*T**B_L*math.exp(-EA_L/(KB*T))

def keq_exchange(T):
    """N2 + O <-> NO + N (dimensionless)"""
    return (q_total("NO",T)*q_total("N",T)) / \
           (q_total("N2",T)*q_total("O",T)) * math.exp(DH_F/(KB*T))

def keq_dissoc(T):
    """N2 <-> N + N (per volume, 1/m^3); third body cancels"""
    return (q_total("N",T)**2/q_total("N2",T)) * math.exp(DH_D/(KB*T))

def kb_derived(T):
    return kf(T)/keq_exchange(T)

# ---------------------------------------------------------------- running

def run(exe, deck, varz, tag, extra_args=None, subs=None, extra_files=None):
    wd = os.path.join(HERE, "work_validate", tag)
    os.makedirs(wd, exist_ok=True)
    for f in ("air.species","air.vss","air.rot","air.elec","rev.tce",
              "rev_exch.tce","rev_mol.tce",deck):
        shutil.copy(os.path.join(HERE,f), wd)
    for name, text in (extra_files or {}).items():
        open(os.path.join(wd,name),"w").write(text)
    if subs:
        dk = open(os.path.join(wd,deck)).read()
        for old,new in subs.items(): dk = dk.replace(old,new)
        open(os.path.join(wd,deck),"w").write(dk)
    cmd = [exe] + (extra_args or []) + ["-in", deck, "-log", "log.sparta"]
    for k,v in varz.items(): cmd += ["-var", k, str(v)]
    r = subprocess.run(cmd, cwd=wd, capture_output=True, text=True,
                       timeout=3600)
    if r.returncode != 0:
        tail = "\n".join((r.stdout or "").splitlines()[-15:])
        raise RuntimeError("run failed (%s):\n%s" % (tag, tail))
    return os.path.join(wd,"log.sparta")

def tallies(log):
    t = {}
    for line in open(log):
        m = re.match(r"\s*reaction (.+): ([0-9.eE+-]+)", line)
        if m: t[m.group(1).strip()] = float(m.group(2))
    return t

def stats_rows(log):
    rows, header = [], None
    lines = open(log).read().splitlines()
    for i,l in enumerate(lines):
        if l.startswith("Step "):
            header = l.split(); rows = []
            for l2 in lines[i+1:]:
                w = l2.split()
                try: float(w[0])
                except (ValueError,IndexError): break
                rows.append(w)
    return header, rows

# ---------------------------------------------------------------- checks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exe", required=True)
    p.add_argument("--exe-args", default="")
    p.add_argument("--exe2", default=None,
                   help="second binary (e.g. KOKKOS) for the keq parity check")
    p.add_argument("--exe2-args", default="")
    args = p.parse_args()
    exe = os.path.abspath(args.exe)
    extra = args.exe_args.split() if args.exe_args else []
    exe2 = os.path.abspath(args.exe2) if args.exe2 else None
    extra2 = args.exe2_args.split() if args.exe2_args else []

    FNUM, NRHO, V = 1.767e6, 7.07043e22, 1.0e-12
    NRHO_HI, FNUM_HI = 1.414086e25, 3.534e8  # dense reservoir for 3-body rates
    NSTEP, DT = 2000, 1.0e-9
    nfrac = 0.25

    print("check 1: exchange pair tally ratio vs analytic Keq(T)")
    for T in (8000.0, 10000.0, 15000.0):
        log = run(exe,"in.reverse_rate",
                  {"T":T,"RB":1000.0,"NRHO":NRHO,"FNUM":FNUM},
                  "rate%d"%T,extra)
        t = tallies(log)
        f = t.get("N2 + O --> NO + N",0.0)
        b = t.get("NO + N --> N2 + O",0.0)
        keq = keq_exchange(T)
        ratio = f/b if b else float("inf")
        sig = math.sqrt(1.0/max(f,1)+1.0/max(b,1))     # relative stat error
        dev = abs(ratio/keq - 1.0)
        check("T=%6.0fK  f/b=%.4f  Keq=%.4f (dev %.1f%%, stat %.1f%%)"
              % (T, ratio, keq, 100*dev, 100*sig),
              dev < max(4*sig, 0.06) and f > 200 and b > 200)

        # check 2 piggybacks on the same runs
        kb_meas = b*FNUM/((NRHO*nfrac)**2*V*NSTEP*DT)
        check("T=%6.0fK  derived kb=%.3e  literature fit=%.3e (x%.2f)"
              % (T, kb_meas, kb_lit(T), kb_meas/kb_lit(T)),
              0.5 < kb_meas/kb_lit(T) < 2.0)

    print("check 3: dissociation/recombination pair vs volumetric Keq")
    # longer run + tight bound: a missing constant in the 3-body table
    # calibration (e.g. the Gamma(3/2) norm of the third body's
    # translational energy, ~13%) must fail this check
    T = 15000.0
    log = run(exe,"in.reverse_rate",
              {"T":T,"RB":1.0,"NRHO":NRHO_HI,"FNUM":FNUM_HI},
              "recomb%d"%T,extra,
              subs={"run             2000":"run             6000"})
    t = tallies(log)
    d = t.get("N2 + N --> N + N + N",0.0)
    r = t.get("N + N --> N2 + N",0.0)
    # rate_d = kd n_N2 n_N V ; rate_r = kr n_N^2 n_N V  (third body N)
    # kd/kr = (d/r) * n_N  must equal keq_dissoc
    nN = NRHO_HI*nfrac
    keqd = keq_dissoc(T)
    ratio = (d/r)*nN if r else float("inf")
    sig = math.sqrt(1.0/max(d,1)+1.0/max(r,1))
    dev = abs(ratio/keqd - 1.0)
    check("T=%6.0fK  (d/r)*n_N=%.3e  Keq=%.3e (dev %.1f%%, stat %.1f%%)"
          % (T, ratio, keqd, 100*dev, 100*sig),
          dev < max(3*sig, 0.05) and d > 500 and r > 500)

    print("check 4: reacting box relaxes toward analytic equilibrium")
    # closed system N2/O/NO/N with only the exchange reaction active:
    # element totals conserved; equilibrium satisfies
    #   (nNO nN)/(nN2 nO) = Keq(T);  start from pure reactant side and
    # pure product side and require the compositions to converge toward
    # the analytic root and toward each other
    T = 15000.0
    keq = keq_exchange(T)
    logs = {}
    for tag, fr in (("fwd",(0.45,0.45,0.05,0.05)),
                    ("bwd",(0.05,0.05,0.45,0.45))):
        logs[tag] = run(exe,"in.reverse_eq",
                        {"T":T,"FN2":fr[0],"FO":fr[1],"FNO":fr[2],"FN":fr[3]},
                        "eq_"+tag,extra)
    # analytic equilibrium for x: N2,O start a; NO,N start b; extent x
    def resid(x,a,b):
        return (b+x)*(b+x) - keq*(a-x)*(a-x)
    for tag,(a,b) in (("fwd",(0.45,0.05)),("bwd",(0.05,0.45))):
        lo,hi = -b+1e-9, a-1e-9
        for _ in range(200):
            mid = 0.5*(lo+hi)
            if resid(mid,a,b) > 0: hi = mid
            else: lo = mid
        xeq = 0.5*(lo+hi)
        frac_eq = (b+xeq)/((a-xeq)+(b+xeq))   # NO fraction of the NO+N2 pool
        header, rows = stats_rows(logs[tag])
        iNO = header.index("c_spcount[3]"); iN2 = header.index("c_spcount[1]")
        n0 = (float(rows[0][iNO]), float(rows[0][iN2]))
        nT = (float(rows[-1][iNO]), float(rows[-1][iN2]))
        frac0 = n0[0]/(n0[0]+n0[1])
        fracT = nT[0]/(nT[0]+nT[1])
        # moved at least 60% of the way from the start to the analytic value
        prog = (fracT-frac0)/(frac_eq-frac0) if frac_eq != frac0 else 1.0
        check("%s: NO/(NO+N2) %.3f -> %.3f (analytic eq %.3f, progress %.0f%%)"
              % (tag, frac0, fracT, frac_eq, 100*prog),
              prog > 0.6 and abs(fracT-frac_eq) < abs(frac0-frac_eq)*0.5)

    print("check 5: analytic sanity of the implementation inputs")
    # ground-state electronic degeneracy ratio for this reaction is
    # (4*4)/(1*9) = 1.78; confirm the elec file carries it
    g = lambda name: ELEC[name][0][1]
    check("elec ground degeneracies N2=1 O=9 NO=4 N=4",
          g("N2")==1 and g("O")==9 and g("NO")==4 and g("N")==4)
    sig = lambda name: ROT[name][1]
    check("symmetry numbers N2=2 NO=1", sig("N2")==2 and sig("NO")==1)

    print("check 6: error paths reject invalid B-style inputs")
    def expect_error(tag, tce_text, msg_frag, deck="in.reverse_rate",
                     react_style_line=None):
        wd = os.path.join(HERE,"work_validate","err_"+tag)
        os.makedirs(wd, exist_ok=True)
        for f in ("air.species","air.vss","air.rot","air.elec",
                  "in.reverse_rate"):
            shutil.copy(os.path.join(HERE,f), wd)
        open(os.path.join(wd,"rev.tce"),"w").write(tce_text)
        dk = open(os.path.join(wd,deck)).read()
        if react_style_line:
            dk = dk.replace("react           tce rev.tce", react_style_line)
        open(os.path.join(wd,deck),"w").write(dk)
        cmd = [exe] + extra + ["-in", deck, "-log", "log.sparta",
               "-var","T","15000.0","-var","RB","1000.0",
               "-var","NRHO","7.07043e22","-var","FNUM","1.767e6"]
        r = subprocess.run(cmd, cwd=wd, capture_output=True, text=True,
                           timeout=600)
        out = (r.stdout or "") + (r.stderr or "")
        check("%s: run fails with '%s'" % (tag, msg_frag),
              r.returncode != 0 and msg_frag in out)

    expect_error("nopartner",
        "NO + N --> N2 + O\nE B 0.0 0.0 0.0 0.0 0.0\n",
        "No forward partner")
    expect_error("wildcard",
        "N2 + N --> N + N + N\nD A 1.0 1.561e-18 4.980e-8 -1.5 -1.561e-18\n"
        "\nN + N --> N2 + atom\nR B 0.0 0.0 0.0 0.0 0.0\n",
        "explicit third-body species")
    expect_error("qkstyle",
        "N2 + O --> NO + N\nE A 0.0 5.175e-19 1.069e-12 -1.0 -5.175e-19\n"
        "\nNO + N --> N2 + O\nE B 0.0 0.0 0.0 0.0 0.0\n",
        "require react tce",
        react_style_line="react           tce/qk rev.tce")

    # forward-only reaction file for the auto-reverse checks

    FWD_TCE = (
        "N2 + O --> NO + N\n"
        "E A 0.0 5.175e-19 1.069e-12 -1.0 -5.175e-19\n\n"
        "N2 + N --> N + N + N\n"
        "D A 1.0 1.561e-18 4.980e-8 -1.5 -1.561e-18\n")

    print("check 7: reverse auto generates B partners from a forward-only file")
    T = 15000.0
    log = run(exe,"in.reverse_rate",
              {"T":T,"RB":1000.0,"NRHO":NRHO,"FNUM":FNUM},
              "auto%d"%T,extra,
              subs={"react           tce rev.tce":
                    "react           tce fwd.tce",
                    "rboost ${RB}":"rboost ${RB} reverse auto"},
              extra_files={"fwd.tce":FWD_TCE})
    gen = any("Generated 2 reverse reaction" in l for l in open(log))
    check("init reports 2 generated reverse reactions", gen)
    t = tallies(log)
    f = t.get("N2 + O --> NO + N",0.0)
    b = t.get("NO + N --> N2 + O",0.0)
    keq = keq_exchange(T)
    ratio = f/b if b else float("inf")
    sig = math.sqrt(1.0/max(f,1)+1.0/max(b,1))
    dev = abs(ratio/keq - 1.0)
    check("generated exchange reverse holds detailed balance "
          "(f/b=%.4f Keq=%.4f dev %.1f%%)" % (ratio,keq,100*dev),
          dev < max(4*sig, 0.06) and f > 200 and b > 200)

    print("check 8: external Keq fit reproduces the fitted reverse rate")
    # Park-form fit of the Keq implied by the air.tce forward/reverse pair:
    # matching it must reproduce the literature backward rate exactly
    c1 = math.log(A_F/A_L) + (B_F-B_L)*math.log(10000.0)
    c3 = -EA_F/KB/10000.0
    PARK = ("N2 + O --> NO + N\n"
            "park 0.0 %.6f %.6f %.6f 0.0\n" % (c1, -(B_F-B_L), c3))
    log = run(exe,"in.reverse_rate",
              {"T":T,"RB":1000.0,"NRHO":NRHO,"FNUM":FNUM},
              "keqfit%d"%T,extra,
              subs={"react           tce rev.tce":
                    "react           tce fwd.tce",
                    "rboost ${RB}":
                    "rboost ${RB} reverse auto keq_file park.keq"},
              extra_files={"fwd.tce":FWD_TCE,"park.keq":PARK})
    t = tallies(log)
    b = t.get("NO + N --> N2 + O",0.0)
    kb_meas = b*FNUM/((NRHO*nfrac)**2*V*NSTEP*DT)
    check("derived kb matches the literature fit via the Keq file "
          "(kb=%.3e lit=%.3e x%.2f)" % (kb_meas,kb_lit(T),kb_meas/kb_lit(T)),
          0.9 < kb_meas/kb_lit(T) < 1.1 and b > 200)

    print("check 9: external Keq fit on a dissociation/recombination pair")
    # a Park fit of the analytic volumetric dissociation Keq (1/m^3), fed
    # via keq_file, must reproduce that Keq as the recombination backward
    # rate -- the recombination (m^6/s) analogue of check 8, exercising the
    # m^3->m^6 unit path that the exchange check cannot
    T = 15000.0
    cd = park_fit(keq_dissoc, (8000.,11000.,14000.,17000.,20000.))
    PARKD = ("N2 + N --> N + N + N\n"
             "park %.8g %.8g %.8g %.8g %.8g\n" % tuple(cd))
    log = run(exe,"in.reverse_rate",
              {"T":T,"RB":1.0,"NRHO":NRHO_HI,"FNUM":FNUM_HI},
              "keqrecomb%d"%T,extra,
              subs={"react           tce rev.tce":
                    "react           tce fwd.tce",
                    "rboost ${RB}":
                    "rboost ${RB} reverse auto keq_file park.keq",
                    "run             2000":"run             6000"},
              extra_files={"fwd.tce":FWD_TCE,"park.keq":PARKD})
    t = tallies(log)
    d = t.get("N2 + N --> N + N + N",0.0)
    r = t.get("N + N --> N2 + N",0.0)
    nN = NRHO_HI*nfrac
    keqd = keq_dissoc(T)
    ratio = (d/r)*nN if r else float("inf")
    sig = math.sqrt(1.0/max(d,1)+1.0/max(r,1))
    dev = abs(ratio/keqd - 1.0)
    check("T=%6.0fK  (d/r)*n_N=%.3e  Keq=%.3e (dev %.1f%%, stat %.1f%%)"
          % (T, ratio, keqd, 100*dev, 100*sig),
          dev < max(3*sig, 0.05) and d > 500 and r > 500)

    print("check 10: standard eta=-1.5 dissociation raises no bounds warning")
    # the ubiquitous eta = -3/2 dissociation with one rotor sits exactly on
    # the low-energy trend bound; check_tce_bounds must not warn about it
    log = run(exe,"in.reverse_rate",{"T":15000.,"RB":1000.,
              "NRHO":NRHO,"FNUM":FNUM},"bounds",extra)
    warned = any("does not vanish" in l for l in open(log))
    check("no spurious 'does not vanish' warning on eta=-1.5", not warned)

    print("check 11: reverse detailed balance under vibrate smooth")
    # with classical (smooth) vibration the calibration target and the
    # detailed-balance table must share the same vib temperature
    # dependence: the table must not warn that it drifts, and the exchange
    # reverse must reproduce the classical-vib Keq
    T = 15000.0
    log = run(exe,"in.reverse_rate",
              {"T":T,"RB":1000.,"NRHO":NRHO,"FNUM":FNUM},
              "smooth%d"%T,extra,
              subs={"vibrate discrete":"vibrate smooth"})
    drift = any("detailed-balance table drifts" in l for l in open(log))
    t = tallies(log)
    f = t.get("N2 + O --> NO + N",0.0)
    b = t.get("NO + N --> N2 + O",0.0)
    keqs = (q_total("NO",T,True)*q_total("N",T,True)) / \
           (q_total("N2",T,True)*q_total("O",T,True))*math.exp(DH_F/(KB*T))
    ratio = f/b if b else float("inf")
    sig = math.sqrt(1.0/max(f,1)+1.0/max(b,1))
    dev = abs(ratio/keqs - 1.0)
    check("smooth-vib exchange reverse: no drift warning and f/b matches "
          "classical Keq (f/b=%.4f Keq=%.4f dev %.1f%%)" % (ratio,keqs,100*dev),
          (not drift) and dev < max(4*sig, 0.06) and f > 200 and b > 200)

    print("check 13: 3-body recombination with a molecular third body")
    # N + N -> N2 + N2: the third body N2 carries discrete vibrational and
    # electronic ladders (folded into the density of states) and continuum
    # rotation (a flat measure variable) -- the general molecular-M case of
    # build_db3_table, where check 3 uses an atomic third body N
    T = 15000.0
    molsub = {"react           tce rev.tce":"react           tce rev_mol.tce",
              "run             2000":"run             8000"}
    log = run(exe,"in.reverse_rate",
              {"T":T,"RB":1.0,"NRHO":NRHO_HI,"FNUM":FNUM_HI},
              "molrecomb",extra,subs=molsub)
    t = tallies(log)
    d = t.get("N2 + N2 --> N + N + N2",0.0)
    r = t.get("N + N --> N2 + N2",0.0)
    nN = NRHO_HI*nfrac
    keqd = keq_dissoc(T)
    ratio = (d/r)*nN if r else float("inf")
    sig = math.sqrt(1.0/max(d,1)+1.0/max(r,1))
    dev = abs(ratio/keqd - 1.0)
    check("molecular third body N2: (d/r)*n_N=%.3e Keq=%.3e (dev %.1f%%, stat %.1f%%)"
          % (ratio, keqd, 100*dev, 100*sig),
          dev < max(3*sig, 0.05) and d > 500 and r > 500)
    if exe2:
        tk = tallies(run(exe2,"in.reverse_rate",
                     {"T":T,"RB":1.0,"NRHO":NRHO_HI,"FNUM":FNUM_HI},
                     "molrecomb_kk",extra2,subs=molsub))
        check("molecular third body: CPU/second-binary tallies identical",
              t == tk and bool(t))

    if exe2:
        print("check 12: external-Keq path is bit-for-bit across binaries")
        # tgas feeds the keq prefactor; verify the two binaries (e.g. CPU
        # and KOKKOS) produce identical reaction tallies on the keq path
        c1 = math.log(A_F/A_L) + (B_F-B_L)*math.log(10000.0)
        c3 = -EA_F/KB/10000.0
        PARK = ("N2 + O --> NO + N\n"
                "park 0.0 %.6f %.6f %.6f 0.0\n" % (c1, -(B_F-B_L), c3))
        sub = {"react           tce rev.tce":"react           tce fwd.tce",
               "rboost ${RB}":"rboost ${RB} reverse auto keq_file park.keq"}
        xf = {"fwd.tce":FWD_TCE,"park.keq":PARK}
        vz = {"T":15000.,"RB":1000.,"NRHO":NRHO,"FNUM":FNUM}
        t1 = tallies(run(exe ,"in.reverse_rate",vz,"keqpar1",extra ,sub,xf))
        t2 = tallies(run(exe2,"in.reverse_rate",vz,"keqpar2",extra2,sub,xf))
        same = t1 == t2 and t1
        check("CPU and second-binary keq tallies identical (%s)" %
              (",".join("%s=%g"%(k.split("-->")[0].strip(),v)
                        for k,v in sorted(t1.items())) if t1 else "no tallies"),
              same)

    print("%d failures" % FAIL)
    sys.exit(FAIL)

if __name__ == "__main__":
    main()
