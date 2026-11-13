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


def R_inclusion(T, ladA, ladB, z0, om, eta, Ea, dofA=None, dofB=None):
    """equilibrium rate / input rate when the pair's discrete (electronic)
    energy is added to the collision energy and P keeps its standard form,
    Eq. (5) of the paper.  dofA/dofB: per-state DOF added to z inside P
    (zP = z0 + (dA+dB)/2); None means zero."""
    ea = Ea / (KB * T)
    QA = sum(g * math.exp(-t / T) for t, g in ladA)
    QB = sum(g * math.exp(-t / T) for t, g in ladB)
    eps, gg, zz = [], [], []
    for i, (ta, ga) in enumerate(ladA):
        for j, (tb, gb) in enumerate(ladB):
            eps.append((ta + tb) / T)
            gg.append(ga * gb)
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
