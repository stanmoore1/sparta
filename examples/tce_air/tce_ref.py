"""Independent reference calculations for the tce_air reproduction package.

Nothing here calls SPARTA or reuses its tables: every reference value the
package compares a simulation against is computed from the data files by
the formulas of the paper, with numerical quadrature where the paper says
"quadrature".  The functions are adapted from the independent recomputation
of the paper's analytic numbers (paper-artifacts branch,
wip/scratch/recompute/common.py and c01-c04).

Conventions follow SPARTA's reaction files: coefficient C5 is the reaction
energy in J, negative for an endothermic reaction (the paper's Delta H_0 is
-C5).  Energies of discrete states are in kelvin in the data files.
"""
import math
import os

import numpy as np
from scipy import integrate, special

KB = 1.380649e-23        # J/K
HP = 6.62607015e-34      # J s
EV = 1.602176634e-19

HERE = os.path.dirname(os.path.abspath(__file__))
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # numpy 1 and 2
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data"))


# ------------------------------------------------------------------ parsers
def _datalines(text):
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        yield s


def parse_species(text):
    """ID molwt mass rotdof rotrel vibdof vibrel vibtemp specwt charge"""
    out = {}
    for s in _datalines(text):
        w = s.split()
        if len(w) < 10:
            continue
        out[w[0]] = dict(molwt=float(w[1]), mass=float(w[2]),
                         rotdof=int(w[3]), vibdof=int(w[5]),
                         vibtemp=float(w[7]), charge=float(w[9]))
    return out


def parse_vss(text):
    """ID diam omega tref alpha ..."""
    out = {}
    for s in _datalines(text):
        w = s.split()
        try:
            out[w[0]] = dict(diam=float(w[1]), omega=float(w[2]),
                             tref=float(w[3]), alpha=float(w[4]))
        except (ValueError, IndexError):
            pass
    return out


def parse_rot(text):
    """ID N theta_rot [sigma] -> (theta, sigma)"""
    out = {}
    for s in _datalines(text):
        w = s.split()
        n = int(w[1])
        out[w[0]] = (float(w[2]), float(w[2 + n]) if len(w) > 2 + n else 1.0)
    return out


def parse_elec(text):
    """ID N (temp relax degen spin dof)*N -> list of (temp_K, g)"""
    out = {}
    for s in _datalines(text):
        w = s.split()
        try:
            n = int(w[1])
        except (ValueError, IndexError):
            continue
        if len(w) != 2 + 5 * n:
            continue
        out[w[0]] = [(float(w[2 + 5 * k]), float(w[2 + 5 * k + 2]))
                     for k in range(n)]
    return out


def parse_tce(text, keep_reverse=False):
    """'A + B --> C + D [+ E]' then 'TYPE STYLE C1 Ea A eta C5'"""
    lines = [ln.rstrip() for ln in text.splitlines()]
    rx, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        if s and not s.startswith("#") and "-->" in s:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            c = lines[j].split()
            lhs, rhs = s.split("-->")
            r = dict(label=" ".join(s.split()),
                     reac=[x.strip() for x in lhs.split(" + ")],
                     prod=[x.strip() for x in rhs.split(" + ")],
                     type=c[0], style=c[1], c1=float(c[2]), Ea=float(c[3]),
                     A=float(c[4]), eta=float(c[5]), C5=float(c[6]))
            if c[1] != "B" or keep_reverse:
                rx.append(r)
            i = j + 1
        else:
            i += 1
    return rx


def _read(path):
    with open(path) as fh:
        return fh.read()


class Data:
    """species, collision, rotation, electronic and reaction data read from
    files; by default the air files distributed with SPARTA (data/air.*).
    Any file can be replaced, e.g. Data(elec="o2n2.elec")."""

    def __init__(self, species=None, vss=None, rot=None, elec=None, tce=None):
        d = lambda f, dflt: f if f else os.path.join(DATA, dflt)
        self.sp = parse_species(_read(d(species, "air.species")))
        self.vss = parse_vss(_read(d(vss, "air.vss")))
        self.rot = parse_rot(_read(d(rot, "air.rot")))
        self.elec = parse_elec(_read(d(elec, "air.elec")))
        self.rx = parse_tce(_read(d(tce, "air.tce")))
        self.rx_all = parse_tce(_read(d(tce, "air.tce")), keep_reverse=True)

    def reaction(self, label):
        for r in self.rx_all:
            if r["label"] == " ".join(label.split()):
                return r
        raise KeyError(label)

    # pair collision parameters: arithmetic means (collide_vss.cpp)
    def pair(self, a, b):
        va, vb = self.vss[a], self.vss[b]
        d = 0.5 * (va["diam"] + vb["diam"])
        om = 0.5 * (va["omega"] + vb["omega"])
        tref = 0.5 * (va["tref"] + vb["tref"])
        ma, mb = self.sp[a]["mass"], self.sp[b]["mass"]
        return d, om, tref, ma * mb / (ma + mb)

    def zrot(self, a, b):
        return 0.5 * (self.sp[a]["rotdof"] + self.sp[b]["rotdof"])

    def kcoll(self, a, b, T):
        """VHS/VSS pair collision rate coefficient <sigma g>/eps [m^3/s]"""
        d, om, tref, mr = self.pair(a, b)
        eps = 2.0 if a == b else 1.0
        return (2.0 * math.sqrt(math.pi) * d * d *
                math.sqrt(2.0 * KB * tref / mr) * (T / tref) ** (1.0 - om) / eps)

    # ---- partition functions per unit volume, all modes modeled
    def q_trans(self, s, T):
        return (2.0 * math.pi * self.sp[s]["mass"] * KB * T / HP ** 2) ** 1.5

    def q_rot(self, s, T):
        if self.sp[s]["rotdof"] == 2 and s in self.rot:
            th, sig = self.rot[s]
            return T / (sig * th)
        return 1.0

    def q_vib(self, s, T):
        th = self.sp[s]["vibtemp"]
        if self.sp[s]["vibdof"] > 0 and th > 0.0:
            return 1.0 / (1.0 - math.exp(-th / T))
        return 1.0

    def q_elec(self, s, T):
        if s in self.elec:
            return sum(g * math.exp(-t / T) for t, g in self.elec[s])
        if self.sp[s]["charge"] < 0 and self.sp[s]["mass"] < 1e-29:
            return 2.0                     # free electron: spin degeneracy
        return 1.0

    def q(self, s, T):
        return (self.q_trans(s, T) * self.q_rot(s, T) * self.q_vib(s, T) *
                self.q_elec(s, T))

    def keq(self, r, T):
        """partition-function equilibrium constant of reaction r in
        number-density units, k_f/k_r = prod q(products)/prod q(reactants)
        * exp(C5/kT)"""
        num = np.prod([self.q(s, T) for s in r["prod"]])
        den = np.prod([self.q(s, T) for s in r["reac"]])
        return num / den * math.exp(r["C5"] / (KB * T))


    def e_mean(self, s, T):
        """mean energy per particle in J (translation, rotation, vibration,
        electronic states) at equilibrium at T, consistent with q(s, T)"""
        e = 1.5 * KB * T
        if self.sp[s]["rotdof"] >= 2:
            e += 0.5 * self.sp[s]["rotdof"] * KB * T
        th = self.sp[s]["vibtemp"]
        if self.sp[s]["vibdof"] > 0 and th > 0.0:
            e += KB * th / math.expm1(th / T)
        if s in self.elec:
            w = [(g * math.exp(-t / T), t) for t, g in self.elec[s]]
            e += KB * sum(a * t for a, t in w) / sum(a for a, t in w)
        return e


# formation energies (J) of the closed air set: the table in the header of
# data/air.tce (0-K enthalpies of formation of the NASA Glenn database);
# every reaction energy C5 of data/air.tce is a difference of these
EFORM = {"N2": 0.0, "O2": 0.0, "e": 0.0, "N": 7.81811e-19, "O": 4.09804e-19,
         "NO": 1.50723e-19, "N2+": 2.49630e-18, "O2+": 1.93453e-18,
         "NO+": 1.63499e-18, "N+": 3.11044e-18, "O+": 2.59166e-18}


def closure_residual(r):
    """C5 minus the formation-energy difference, relative to |C5|"""
    d = sum(EFORM[s] for s in r["reac"]) - sum(EFORM[s] for s in r["prod"])
    return (r["C5"] - d) / max(abs(r["C5"]), 1e-30)


def arrhenius(r, T):
    return r["A"] * T ** r["eta"] * math.exp(-r["Ea"] / (KB * T))


# -------------------------------------------- TCE equilibrium quadratures
def _lgam(x):
    return special.gammaln(x)


def _glnodes(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def _graded_rule(tmax, npanel=48, norder=16, power=1.6):
    edges = tmax * (np.linspace(0.0, 1.0, npanel + 1) ** power)
    xs, ws = [], []
    for k in range(npanel):
        x, w = _glnodes(edges[k], edges[k + 1], norder)
        xs.append(x)
        ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


_YMAX, _SUBP = 90.0, 4
_T, _W = _graded_rule(_YMAX ** (1.0 / _SUBP))
_Y = _T ** _SUBP
_DY = _SUBP * _T ** (_SUBP - 1) * _W


def avgP_scaled(eps, zP, z0, om, eta, ea, chunk=4000):
    """J = e^{ea-eps} int_0^inf g_{s0}(x) p_{zP}(x+eps) dx, vectorized over
    discrete-state energies eps (in kT) and the z used inside P for each
    state; g_{s0} is the Gamma(z0+5/2-om) density of the continuum energy x
    (in kT) and p_z the standard TCE probability, Eq. (3) of the paper,
    divided by C1 (kT)^(eta-1+om).  For eps = 0 and zP = z0 the calibration
    identity gives J = 1.  The Boltzmann factor e^{-eps} is applied by the
    caller as w e^{eps}, so that R = sum_p w_p e^{eps_p} J_p."""
    eps = np.atleast_1d(np.asarray(eps, float))
    zP = np.broadcast_to(np.asarray(zP, float), eps.shape)
    s0 = z0 + 2.5 - om
    out = np.empty(eps.shape)
    lg_s0 = _lgam(s0)
    for i0 in range(0, eps.size, chunk):
        e = eps[i0:i0 + chunk, None]
        z = zP[i0:i0 + chunk, None]
        lo = np.maximum(ea - e, 0.0)
        x = lo + _Y[None, :]
        xm = np.maximum(e - ea, 0.0) + _Y[None, :]
        ec = xm + ea
        lnp = (_lgam(z + 2.5 - om) - _lgam(z + eta + 1.5) +
               (eta - 1.0 + om) * np.log(xm) +
               (z + 1.5 - om) * (np.log(xm) - np.log(ec)))
        lng = (s0 - 1.0) * np.log(x) - xm - lg_s0
        out[i0:i0 + chunk] = np.sum(np.exp(lnp + lng) * _DY[None, :], axis=1)
    return out


def avgP_quad(eps, zP, z0, om, eta, ea):
    """scalar adaptive-quadrature version of avgP_scaled (cross-check)"""
    s0 = z0 + 2.5 - om
    lo = max(ea - eps, 0.0)
    cz = _lgam(zP + 2.5 - om) - _lgam(zP + eta + 1.5) - _lgam(s0)

    def f(t):
        y = t ** 4
        x = lo + y
        ec = x + eps
        xm = ec - ea
        if xm <= 0.0 or x <= 0.0:
            return 0.0
        v = (cz + (eta - 1.0 + om) * math.log(xm) +
             (zP + 1.5 - om) * (math.log(xm) - math.log(ec)) +
             (s0 - 1.0) * math.log(x) - xm)
        return math.exp(v) * 4.0 * t ** 3
    val, _ = integrate.quad(f, 0.0, 120.0 ** 0.25, epsabs=0.0,
                            epsrel=1e-12, limit=400)
    return val


def R_inclusion(T, ladA, ladB, z0, om, eta, Ea, dofA=None, dofB=None,
                Tpop=None):
    """equilibrium rate / input rate when the pair's discrete (electronic)
    energy is added to the collision energy and P keeps its standard form,
    Eq. (5) of the paper.  dofA/dofB: per-state DOF added to z inside P
    (zP = z0 + (dA+dB)/2); None means zero.  Tpop: the discrete states are
    populated at Tpop instead of T (frozen populations); the result is still
    relative to the input rate at T."""
    ea = Ea / (KB * T)
    Tp = T if Tpop is None else Tpop
    QA = sum(g * math.exp(-t / Tp) for t, g in ladA)
    QB = sum(g * math.exp(-t / Tp) for t, g in ladB)
    eps, gg, zz = [], [], []
    for i, (ta, ga) in enumerate(ladA):
        for j, (tb, gb) in enumerate(ladB):
            eps.append((ta + tb) / T)
            # population weight at Tp times e^{+eps} (applied with J below)
            gg.append(ga * gb * math.exp((ta + tb) * (1.0 / T - 1.0 / Tp)))
            zz.append(z0 + 0.5 * ((dofA[i] if dofA else 0.0) +
                                  (dofB[j] if dofB else 0.0)))
    J = avgP_scaled(np.array(eps), np.array(zz), z0, om, eta, ea)
    return float(np.sum(np.array(gg) * J) / (QA * QB))


def _zinst(i):
    i = np.asarray(i, float)
    out = np.zeros_like(i)
    m = i > 0
    out[m] = i[m] * np.log(1.0 + 1.0 / i[m])
    return out


def _sho(theta, T, Ea_K, Tpop=None, extra=45.0):
    """SHO levels (K), populations at Tpop (default T), instantaneous z"""
    Tpop = T if Tpop is None else Tpop
    if theta <= 0:
        return np.array([0.0]), np.array([1.0]), np.array([0.0])
    n = int((Ea_K + extra * max(T, Tpop)) / theta) + 2
    i = np.arange(n)
    q = 1.0 / (1.0 - math.exp(-theta / Tpop))
    return i * theta, np.exp(-i * theta / Tpop) / q, _zinst(i)


def R_instdof(T, thA, thB, z0, om, eta, Ea, Tvib=None):
    """equilibrium rate / input rate of the standard instantaneous-DOF
    treatment of discrete (SHO) vibration: Ec includes the vibrational
    energy and z inside P is z0 + i ln(1+1/i) + j ln(1+1/j) (Bird).  With
    Tvib, vibrational populations are frozen at Tvib (rate relative to the
    input rate at T)."""
    ea = Ea / (KB * T)
    eA, pA, zA = _sho(thA, T, Ea / KB, Tvib)
    eB, pB, zB = _sho(thB, T, Ea / KB, Tvib)
    E = (eA[:, None] + eB[None, :]).ravel() / T
    with np.errstate(divide="ignore"):
        lw = (np.log(pA)[:, None] + np.log(pB)[None, :]).ravel() + E
    Z = (z0 + zA[:, None] + zB[None, :]).ravel()
    keep = lw > -80.0
    J = avgP_scaled(E[keep], Z[keep], z0, om, eta, ea)
    return float(np.sum(np.exp(lw[keep]) * J))


# ------------------------- microcanonical factor out of equilibrium
def micro_ratio(T, z0, om, eta, Ea, ladders_eq, ladders_frozen=(), du_K=None):
    """rate with the microcanonical factor, Eq. (11), when the pair's
    discrete ladders are populated at their own temperatures, relative to
    the input rate at the translational-rotational temperature T.

    ladders_eq: ladders [(temp_K, g)] populated at T
    ladders_frozen: (ladder, Tfrozen) pairs populated at Tfrozen
    The pair state p of the joint ladder has energy eps_p (sum) and weight
    prod_i g e^{-eps_i/T_i}/Q_i; the rate is
      <P~>/<P~>_eq = sum_p w_p int f_s0(E) P~(E+eps_p) dE / (C1 (kT)^(eta-1+om) e^{-Ea/kT})
    evaluated on an energy grid with P~ = C1 Gamma(s0)/Gamma(z+eta+3/2) N/D."""
    lads = [(lad, T) for lad in ladders_eq] + list(ladders_frozen)
    # joint ladder with Boltzmann weights at each ladder's own temperature
    st = [(0.0, 1.0, 1.0)]                     # (energy K, g, weight)
    for lad, Tl in lads:
        Ql = sum(g * math.exp(-t / Tl) for t, g in lad)
        new = []
        for e0, g0, w0 in st:
            for t, g in lad:
                new.append((e0 + t, g0 * g, w0 * g * math.exp(-t / Tl) / Ql))
        st = new
    eps = np.array([s[0] for s in st])
    gg = np.array([s[1] for s in st])
    ww = np.array([s[2] for s in st])
    s0 = z0 + 2.5 - om
    Ea_K = Ea / KB
    emax = Ea_K + 80.0 * T + eps.max()
    du = du_K or min(T / 400.0, 25.0)
    u = np.arange(1, int(emax / du) + 2) * du          # total energy (K)
    num = np.zeros_like(u)
    den = np.zeros_like(u)
    for e, g in zip(eps, gg):
        x = u - e
        m = x > 0
        den[m] += g * x[m] ** (s0 - 1.0)
        xr = u - e - Ea_K
        mr = xr > 0
        num[mr] += g * xr[mr] ** (z0 + eta + 0.5)
    ratio = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    # thermal average over continuum x and discrete state p:
    # sum_p w_p int g_s0(x;T) ratio(x+eps_p) dx, with ratio interpolated
    tot = 0.0
    for e, w in zip(eps, ww):
        if w < 1e-300:
            continue
        x = u - e
        m = x > 0
        gx = np.exp((s0 - 1.0) * np.log(x[m] / T) - x[m] / T - _lgam(s0)) / T
        tot += w * _trapz(gx * ratio[m], x[m])
    # normalization: the same average with every ladder at T is the input
    # rate up to the constant C1 Gamma(s0)/Gamma(z+eta+3/2) (kT)^... ;
    # compute it the same way so that discretization cancels
    if ladders_frozen:
        ref = micro_ratio(T, z0, om, eta, Ea,
                          list(ladders_eq) + [l for l, _ in ladders_frozen],
                          (), du_K=du)
        return tot / ref
    return tot


if __name__ == "__main__":
    # self-test: calibration identity and quadrature agreement
    for (z0, om, eta, ea) in [(2.0, 0.755, -1.5, 4.0), (1.0, 0.77, -1.5, 9.0),
                              (0.0, 0.8, 0.0, 3.0)]:
        j = avgP_scaled([0.0], [z0], z0, om, eta, ea)[0]
        jq = avgP_quad(0.0, z0, z0, om, eta, ea)
        print("identity z0=%g om=%g eta=%g ea=%g : GL %.10f quad %.10f"
              % (z0, om, eta, ea, j, jq))


# ------------------------------------------- Arrhenius refit of a biased model
def refit_inclusion(lad, z0, om, eta, Ea, A, dofA, dofB, Tmin=5000.0,
                    Tmax=40000.0, n=71, Ea_min=None):
    """best Arrhenius input (A', eta', Ea') for the inclusion model with
    per-state DOF, such that the model's equilibrium rate
    A' T^eta' e^{-Ea'/kT} R(T; eta', Ea') reproduces the target
    A T^eta e^{-Ea/kT} in least squares of the logarithm over Tmin..Tmax.
    The model is re-evaluated with the refit coefficients (R depends on
    eta and Ea), not only rescaled.  lad = (ladA, ladB).  Ea_min bounds the
    refit activation energy from below (a dissociation threshold cannot lie
    below the bond energy; SPARTA gates a reaction at its endothermicity).
    Returns (A', eta', Ea', Ts, residual ratio model/target)."""
    from scipy.optimize import least_squares
    ladA, ladB = lad
    Ts = np.linspace(Tmin, Tmax, n)

    def ratio(p, T):
        a, b, c = p
        e2, E2 = eta + b, Ea + c * KB
        R = R_inclusion(T, ladA, ladB, z0, om, e2, E2, dofA, dofB)
        return R * math.exp(a) * T ** b * math.exp(-c / T)

    cmin = -np.inf if Ea_min is None else (Ea_min - Ea) / KB
    res = least_squares(lambda p: [math.log(ratio(p, T)) for T in Ts],
                        x0=[0.0, 0.0, max(0.0, cmin) + 100.0],
                        x_scale=[1.0, 0.1, 1000.0],
                        bounds=([-np.inf, -np.inf, cmin],
                                [np.inf, np.inf, np.inf]))
    a, b, c = res.x
    return (A * math.exp(a), eta + b, Ea + c * KB, Ts,
            np.array([ratio(res.x, T) for T in Ts]))


# -------------------- joint vibrational-electronic factor, frozen ladders
def _conv(arr, eps, wts, du):
    """shift-and-add convolution of arr (on grid du) with a ladder"""
    out = np.zeros_like(arr)
    n = arr.size
    for e, w in zip(eps, wts):
        if w == 0.0:
            continue
        sh = e / du
        i0 = int(math.floor(sh))
        fr = sh - i0
        if i0 < n:
            out[i0:] += w * (1 - fr) * arr[:n - i0]
        if i0 + 1 < n:
            out[i0 + 1:] += w * fr * arr[:n - i0 - 1]
    return out


class JointLadder:
    """microcanonical factor N/D of a pair on an energy grid (kelvin):
    continuum = translation + rotation (z, omega), composite ladder =
    harmonic vibrational ladders (thetas, untruncated up to umax) times
    electronic ladders [(temp_K, g)].  rate(T, Tv, Te) is the equilibrium-
    calibrated rate with vibration populated at Tv and electronic states at
    Te, divided by the input Arrhenius rate at T (independent of A)."""

    def __init__(self, z, om, eta, Ea_K, thetas=(), elecs=(), du=10.0,
                 umax=2.0e6):
        self.z, self.om, self.eta, self.Ea = z, om, eta, Ea_K
        self.du = du
        self.u = np.arange(0.0, umax, du)
        u = self.u
        self.thetas = list(thetas)
        self.vibs = [np.arange(0, int(umax / th) + 1) * th for th in thetas]
        self.elecs = [(np.array([t for t, g in L]), np.array([g for t, g in L]))
                      for L in elecs]
        num = np.clip(u - Ea_K, 0.0, None) ** (z + eta + 0.5)
        den = u ** (z + 1.5 - om)
        for lv in self.vibs:
            num = _conv(num, lv, np.ones_like(lv), du)
            den = _conv(den, lv, np.ones_like(lv), du)
        for t, g in self.elecs:
            num = _conv(num, t, g, du)
            den = _conv(den, t, g, du)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.fac = np.where(den > 0, num / den, 0.0)

    def rate(self, T, Tv=None, Te=None):
        Tv = T if Tv is None else Tv
        Te = T if Te is None else Te
        u = self.u
        sr = self.z + 2.5 - self.om
        with np.errstate(divide="ignore"):
            kern = np.where(u > 0, np.exp((sr - 1) * np.log(np.where(u > 0, u, 1))
                                          - u / T), 0.0)
        for lv, th in zip(self.vibs, self.thetas):
            p = np.exp(-lv / Tv) * (1 - math.exp(-th / Tv))
            kern = _conv(kern, lv, p, self.du)
        for t, g in self.elecs:
            w = g * np.exp(-t / Te)
            kern = _conv(kern, t, w / w.sum(), self.du)
        lhs = _trapz(self.fac * kern, dx=self.du)
        ref = math.exp(_lgam(self.z + self.eta + 1.5) +
                       (self.z + self.eta + 1.5) * math.log(T) - self.Ea / T)
        return lhs / ref


def park_ratio(T, Tv, q, eta, Ea_K):
    """Park two-temperature rate k(Ta)/k(T), Ta = T^q Tv^(1-q)"""
    Ta = T ** q * Tv ** (1 - q)
    return (Ta / T) ** eta * math.exp(-Ea_K / Ta + Ea_K / T)


def qk_over_kcoll(T, Tv, theta, Ea_K, om):
    """Bird's quantum-kinetic dissociation probability averaged over the
    VHS collision ensemble: a molecule at SHO level i (levels below the
    dissociation energy) dissociates if E_tr + i k theta >= E_d, with the
    collision-weighted E_tr Gamma(5/2-omega) distributed at T and the
    levels populated at Tv"""
    from scipy.special import gammaincc
    lv = np.arange(int(Ea_K / theta) + 1)
    p = np.exp(-lv * theta / Tv)
    p /= p.sum()
    thr = np.clip((Ea_K - lv * theta) / T, 0.0, None)
    return float(np.sum(p * gammaincc(2.5 - om, thr)))


def park_fit(func, Ts):
    """Park equilibrium-constant coefficients c0..c4 with
    ln f(T) = c0/Z + c1 + c2 ln Z + c3 Z + c4 Z^2, Z = 1e4/T, through the
    five given temperatures (the form react_modify keq_file reads)"""
    A = [[1e4 / (1e4 / T) * 0 + 1.0 / (1e4 / T), 1.0, math.log(1e4 / T),
          1e4 / T, (1e4 / T) ** 2] for T in Ts]
    y = [math.log(func(T)) for T in Ts]
    return list(np.linalg.solve(np.array(A), np.array(y)))


# ------------------------------------------------ derived reverse reactions
def pair_states(D, species, emax, vib="discrete"):
    """discrete states of one species or a pair, as numpy arrays of the
    vibrational part, the electronic part (J) and the degeneracy; states
    above emax (J) are dropped.  vib='smooth' leaves vibration out of the
    ladder (it is then a continuum mode, see pair_z)."""
    ev, ee, gg = [0.0], [0.0], [1.0]
    for s in species:
        th = D.sp[s]["vibtemp"]
        if vib == "discrete" and D.sp[s]["vibdof"] > 0 and th > 0.0:
            lv = [l * KB * th for l in range(int(emax / (KB * th)) + 1)]
        else:
            lv = [0.0]
        if s in D.elec:
            le = [(KB * t, float(g)) for t, g in D.elec[s]]
        elif D.sp[s]["charge"] < 0 and D.sp[s]["mass"] < 1e-29:
            le = [(0.0, 2.0)]                     # free electron: spin
        else:
            le = [(0.0, 1.0)]
        nv, ne, ng = [], [], []
        for a, b, c in zip(ev, ee, gg):
            for v in lv:
                for e, g in le:
                    if a + v + b + e <= emax:
                        nv.append(a + v)
                        ne.append(b + e)
                        ng.append(c * g)
        ev, ee, gg = nv, ne, ng
    return np.array(ev), np.array(ee), np.array(gg)


def pair_z(D, species, vib="discrete"):
    """continuum internal DOF / 2 of a pair: rotation, plus vibration when
    it is continuous"""
    z = sum(0.5 * D.sp[s]["rotdof"] for s in species)
    if vib == "smooth":
        z += sum(0.5 * D.sp[s]["vibdof"] for s in species)
    return z


def _powp(x, p):
    """(x)_+^p with (x)_+^0 = step function"""
    xp = np.maximum(x, 0.0)
    if p == 0.0:
        return (x > 0.0).astype(float)
    with np.errstate(divide="ignore"):
        return np.where(x > 0.0, xp ** p, 0.0)


def _ladder_sum(u, e, g, p, chunk=2000000):
    """sum_k g_k (u - e_k)_+^p for an array u"""
    u = np.asarray(u, float)
    out = np.zeros(u.size)
    flat = u.ravel()
    step = max(1, chunk // max(e.size, 1))
    for i0 in range(0, flat.size, step):
        x = flat[i0:i0 + step, None] - e[None, :]
        out[i0:i0 + step] = _powp(x, p) @ g
    return out.reshape(u.shape)


def _cont_density(c, a1, th1, a2, th2, n=48):
    """density of x1 + x2 with x1 ~ Gamma(a1, th1), x2 ~ Gamma(a2, th2);
    a2 = 0 means x2 = 0"""
    c = np.asarray(c, float)
    if a2 == 0.0 or th1 == th2:
        a = a1 + a2
        with np.errstate(divide="ignore"):
            return np.where(c > 0, np.exp((a - 1) * np.log(c) - c / th1 -
                                          special.gammaln(a) -
                                          a * math.log(th1)), 0.0)
    # x1 = c y: Gauss-Jacobi in y with weight y^(a1-1) (1-y)^(a2-1)
    y, w = special.roots_jacobi(n, a2 - 1.0, a1 - 1.0)
    y = 0.5 * (y + 1.0)
    w = w * 0.5 ** (a1 + a2 - 1.0)
    cc = c[..., None]
    expo = np.exp(-cc * (y / th1 + (1.0 - y) / th2))
    pref = np.exp(-(special.gammaln(a1) + special.gammaln(a2) +
                    a1 * math.log(th1) + a2 * math.log(th2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(c > 0, pref * c ** (a1 + a2 - 1.0) *
                        (expo @ w), 0.0)


class ExchangeReverse:
    """The derived reverse of a two-body forward reaction fwd (exchange,
    charge exchange, associative ionization), Eq. (exch) of the paper:

        P_B(u) = S N_F(u + dH0) / D_B(u),
        N_F(E) = sum_p g_p (E - e_p - Ea_F)_+^(z_F + eta_F + 1/2)   (forward
                 pair ladder), D_B(u) = sum_q g_q (u - e_q)_+^(s_B - 1)
                 (reverse pair ladder), s_B = z_B + 5/2 - omega_B,

    with S fixed so that the equilibrium average reproduces k_f/K_eq, in
    closed form from the Laplace transforms of N_F and D_B.  Computed from
    the data files only.  The cap correction of Eq. (capnum) is not
    included: it changes nothing unless the forward reaction is capped.

    mean_p(Ttr, Trot, Tvib, Telec, cap) is the average of P_B (capped at
    one if cap) over the collision ensemble of the reverse pair:
    collision-weighted translation Gamma(5/2 - omega_B) at Ttr, continuum
    rotation at Trot, and the vibrational and electronic ladders populated
    at Tvib and Telec.  The reverse rate is k_coll,B(Ttr) mean_p."""

    def __init__(self, D, fwd, vib="discrete", emax_ev=40.0, Tcal=10000.0,
                 forward=False):
        self.D, self.f, self.vib, self.forward = D, fwd, vib, forward
        self.F, self.B = list(fwd["reac"]), list(fwd["prod"])
        self.dH = -fwd["C5"]                       # positive: endothermic
        self.EaB = max(fwd["Ea"] - self.dH, 0.0)
        if forward:
            # the forward reaction's own microcanonical probability, Eq.
            # (microfactor): N_F/D_F of the forward pair, threshold Ea_F,
            # calibrated to the Arrhenius rate; averaged over the forward
            # pair's ensemble (self.B is the pair that is averaged over)
            self.B, self.dH, self.EaB = list(fwd["reac"]), 0.0, fwd["Ea"]
        self.zF, self.zB = pair_z(D, self.F, vib), pair_z(D, self.B, vib)
        self.a = self.zF + fwd["eta"] + 0.5
        self.omB = D.pair(*self.B)[1]
        self.sB = self.zB + 2.5 - self.omB
        self.emax = self.EaB + emax_ev * EV
        evF, eeF, self.gF = pair_states(D, self.F, self.emax + self.dH, vib)
        evB, eeB, self.gB = pair_states(D, self.B, self.emax, vib)
        self.eF = evF + eeF
        self.evB, self.eeB, self.eB = evB, eeB, evB + eeB
        self.S = 1.0
        self.S = self.calibration(Tcal)
        # fine grid of P_B: geometric near zero, then uniform
        g1 = np.geomspace(1e-7 * EV, 0.05 * EV, 600)
        g2 = np.arange(0.05 * EV, 2.0 * EV, 2e-4 * EV)
        g3 = np.arange(2.0 * EV, self.emax, 1e-3 * EV)
        self.ug = np.concatenate([g1, g2[1:], g3[1:]])
        self.pg = self.p_exact(self.ug)

    def n_f(self, u):
        return _ladder_sum(np.asarray(u) - self.EaB, self.eF, self.gF, self.a)

    def d_b(self, u):
        return _ladder_sum(u, self.eB, self.gB, self.sB - 1.0)

    def p_exact(self, u):
        with np.errstate(divide="ignore", invalid="ignore"):
            d = self.d_b(u)
            return np.where(d > 0, self.S * self.n_f(u) / d, 0.0)

    def p(self, u):
        return np.interp(u, self.ug, self.pg, left=self.pg[0], right=np.nan)

    def mean_unit(self, T):
        """equilibrium average of N_F(u + dH0)/D_B(u), closed form"""
        kT = KB * T
        QF = np.sum(self.gF * np.exp(-self.eF / kT))
        QB = np.sum(self.gB * np.exp(-self.eB / kT))
        return (QF * math.exp(special.gammaln(self.a + 1.0) +
                              (self.a + 1.0) * math.log(kT) - self.EaB / kT) /
                (QB * math.exp(special.gammaln(self.sB) +
                               self.sB * math.log(kT))))

    def kb_target(self, T):
        if self.forward:
            return arrhenius(self.f, T)
        return arrhenius(self.f, T) / self.D.keq(self.f, T)

    def calibration(self, T):
        """S such that k_coll,B(T) <P_B>_T = k_f(T)/K_eq(T)"""
        return self.kb_target(T) / (self.D.kcoll(*self.B, T) *
                                    self.mean_unit(T))

    def mean_p(self, Ttr, Trot=None, Tvib=None, Telec=None, cap=True,
               wmin=1e-13):
        Trot = Ttr if Trot is None else Trot
        Tvib = Ttr if Tvib is None else Tvib
        Telec = Ttr if Telec is None else Telec
        lw = (np.log(self.gB) - self.evB / (KB * Tvib) -
              self.eeB / (KB * Telec))
        w = np.exp(lw - lw.max())
        keep = w > wmin * w.sum()
        w, eq = w[keep] / w.sum(), self.eB[keep]
        th1, th2 = KB * Ttr, KB * Trot
        cmax = 40.0 * max(th1, th2)
        t, wt = _graded_rule(cmax ** 0.5, npanel=64, norder=16, power=1.0)
        c, wc = t * t, 2.0 * t * wt                     # c = t^2
        rho = _cont_density(c, 2.5 - self.omB, th1, self.zB, th2)
        uu = c[None, :] + eq[:, None]
        pp = self.p(np.minimum(uu, self.ug[-1]))
        if cap:
            pp = np.minimum(pp, 1.0)
        return float(w @ (pp @ (wc * rho)))

    def rate(self, Ttr, **kw):
        return self.D.kcoll(*self.B, Ttr) * self.mean_p(Ttr, **kw)


class Recombination3:
    """The derived recombination A + B + M -> AB + M, the reverse of the
    dissociation fwd (AB + M -> A + B + M), Eq. (pr3) of the paper:

        P_r = S n_M N_F(w + dH0) / (D_AB(u) c3 V3(w)),
        w   = u + e_t3 + (third-body internal energy),

    for an atomic third body: c3 = e_t3^(1/2), V3(w) = sum_m g_m (w - e_m)_+
    over the electronic ladder of M (two flat variables, u and e_t3), N_F
    the reactive density of states of the forward pair AB + M and D_AB that
    of the pair A + B.  The sampling: u from the collision-weighted ensemble
    of the pair (translation Gamma(5/2 - omega) at Ttr plus the electronic
    ladders of A and B at Tel), e_t3 Maxwellian at Ttr (the third particle
    is drawn uniformly, not by relative speed), the ladder of M at Tel.

    ratio(Ttr, Tel) is <P_r>(Ttr, Tel)/<P_r>(Ttr, Ttr) without the cap, by
    quadrature: with x = w, J(y) = int_y^inf e^{-x/kT} N_F(x + dH0)/V3(x) dx,

        <P_r> ~ sum_q sum_m W_q W_m e^{(e_q + e_m)/kT}
                int_{e_q}^inf (u - e_q)^(s-1) / D_AB(u) J(u + e_m) du.

    cap_excess(Ttr, Tel, n_M) bounds the part of <P_r> above one (the
    capped loss), which comes from soft collisions with A, B and M in their
    ground states."""

    def __init__(self, D, fwd):
        self.D, self.f = D, fwd
        AB, M = fwd["reac"]
        A, B = fwd["prod"][0], fwd["prod"][1]
        assert D.sp[M]["rotdof"] == 0 and D.sp[M]["vibdof"] == 0, \
            "atomic third body only"
        assert D.sp[A]["rotdof"] == 0 and D.sp[B]["rotdof"] == 0
        self.AB, self.M, self.pair = AB, M, (A, B)
        self.dH = -fwd["C5"]
        self.EaB = max(fwd["Ea"] - self.dH, 0.0)
        zF = pair_z(D, [AB, M])
        self.a = zF + fwd["eta"] + 0.5
        self.om = D.pair(A, B)[1]
        self.s = 2.5 - self.om                    # z_p = 0 for two atoms
        emax = 60.0 * EV
        evF, eeF, self.gF = pair_states(D, [AB, M], emax + self.dH)
        self.eF = evF + eeF
        _, self.eq, self.gq = pair_states(D, [A, B], emax)
        _, self.em, self.gm = pair_states(D, [M], emax)

    def _G(self, x):
        """N_F(x + dH0)/V3(x)"""
        # N_F(E) = sum_p g_p (E - e_p - Ea_F)_+^a at E = x + dH0, i.e.
        # sum_p g_p (x - e_p - Ea_B)_+^a with Ea_B = Ea_F - dH0
        nf = _ladder_sum(np.asarray(x) - self.EaB, self.eF, self.gF, self.a)
        v3 = _ladder_sum(x, self.em, self.gm, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(v3 > 0, nf / v3, 0.0)

    def _grid(self, kT, top=45.0):
        """graded grid in energy: geometric near zero, then fine uniform"""
        g1 = np.geomspace(1e-9 * kT, 0.02 * kT, 400)
        g2 = np.arange(0.02 * kT, top * kT, 0.002 * kT)
        return np.concatenate([g1, g2[1:]])

    def mean_unit(self, Ttr, Tel):
        kT = KB * Ttr
        x = self._grid(kT)
        f = np.exp(-x / kT) * self._G(x)
        # J(y) = int_y^inf f dx, cumulative trapezoid from the top
        seg = 0.5 * (f[1:] + f[:-1]) * np.diff(x)
        J = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        # the 1/x singularity of f at x -> 0: J(y) ~ -c ln y, add the
        # missing tail int_0^x0 analytically through the grid's first node
        Jf = lambda y: np.interp(y, x, J, left=np.nan, right=0.0)
        wq = self.gq * np.exp(-self.eq / (KB * Tel))
        wq /= wq.sum()
        wm = self.gm * np.exp(-self.em / (KB * Tel))
        wm /= wm.sum()
        keepq = wq > 1e-14
        keepm = wm > 1e-14
        tot = 0.0
        t, wt = _graded_rule(30.0 ** 0.5, npanel=48, norder=16, power=1.0)
        r, wr = t * t * kT, 2.0 * t * wt * kT    # u - e_q in J
        for eq, w1 in zip(self.eq[keepq], wq[keepq]):
            u = eq + r
            dab = _ladder_sum(u, self.eq, self.gq, self.s - 1.0)
            base = r ** (self.s - 1.0) / dab
            for em, w2 in zip(self.em[keepm], wm[keepm]):
                y = u + em
                jy = np.where(y < x[0], J[0] + (f[0] * x[0]) *
                              np.log(x[0] / np.maximum(y, 1e-300)), Jf(y))
                tot += w1 * w2 * math.exp((eq + em) / kT) * np.sum(
                    wr * base * jy)
        return tot

    def mean_unit_closed(self, T):
        """equilibrium value of mean_unit(T, T) in closed form (the Laplace
        transform of N_F; Eq. (pravg3) of the paper): a check of the
        quadrature"""
        kT = KB * T
        QF = np.sum(self.gF * np.exp(-self.eF / kT))
        Qp = np.sum(self.gq * np.exp(-self.eq / kT))
        QM = np.sum(self.gm * np.exp(-self.em / kT))
        return QF * math.exp(special.gammaln(self.a + 1.0) +
                             (self.a + 1.0) * math.log(kT) -
                             self.EaB / kT) / (Qp * QM)

    def ratio(self, Ttr, Tel):
        return self.mean_unit(Ttr, Tel) / self.mean_unit(Ttr, Ttr)


def _recomb_capped_fraction(self, T, n_M, nq=200, ne=200):
    """fraction of the uncapped equilibrium recombination rate that the cap
    P_r <= 1 leaves, at temperature T and third-body density n_M (m^-3):
    E[min(1, P_r)]/E[P_r] over the equilibrium sampling of Eq. (pr3), with
    the absolute calibration S fixed by k_coll,AB(T) <P_r> = n_M k_f/K_eq"""
    D, kT = self.D, KB * T
    A, B = self.pair
    a1 = special.gammaln(self.s) + self.s * math.log(kT)
    a2 = special.gammaln(1.5) + 1.5 * math.log(kT)
    Qp = np.sum(self.gq * np.exp(-self.eq / kT))
    QM = np.sum(self.gm * np.exp(-self.em / kT))
    mean_closed = self.mean_unit_closed(T) / math.exp(a1 + a2)  # / (Qp QM) in it
    kr = arrhenius(self.f, T) / D.keq(self.f, T)
    S = kr / (D.kcoll(A, B, T) * mean_closed)
    # G(w) = N_F(w + dH0)/V3(w) on a fine grid in w
    wg = self._grid(kT)
    Gw = self._G(wg)
    t, wt = _graded_rule(40.0 ** 0.5, npanel=40, norder=8, power=1.0)
    x, wx = t * t * kT, 2.0 * t * wt * kT          # energies above a threshold
    capped = uncapped = 0.0
    wq = self.gq * np.exp(-self.eq / kT) / Qp
    wm = self.gm * np.exp(-self.em / kT) / QM
    for eq, w1 in zip(self.eq, wq):
        if w1 < 1e-8:
            continue
        u = eq + x                                   # pair energy
        dab = _ladder_sum(u, self.eq, self.gq, self.s - 1.0)
        rho_u = np.exp((self.s - 1.0) * np.log(x) - x / kT - a1)
        for em, w2 in zip(self.em, wm):
            if w2 < 1e-8:
                continue
            eps = x                                  # third-body translation
            rho_e = np.exp(0.5 * np.log(eps) - eps / kT - a2)
            W = u[:, None] + eps[None, :] + em
            G = np.interp(W, wg, Gw, left=Gw[0], right=0.0)
            P = S * n_M * G / (dab[:, None] * np.sqrt(eps)[None, :])
            wgt = (w1 * w2) * (rho_u * wx)[:, None] * (rho_e * wx)[None, :]
            uncapped += np.sum(wgt * P)
            capped += np.sum(wgt * np.minimum(P, 1.0))
    return capped / uncapped, uncapped


Recombination3.capped_fraction = _recomb_capped_fraction
