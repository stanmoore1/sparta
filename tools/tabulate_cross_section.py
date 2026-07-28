#!/usr/bin/env python3

"""
tabulate_cross_section.py

Generate tabulated data files for the SPARTA "collide table" style.

Modes:

  vss        emit the analytic VHS/VSS total cross section for a species
             pair, read from a SPARTA species file and VSS parameter file.
             Useful for verification: "collide table" with such a table
             must reproduce "collide vss" to within the interpolation error.

  csv        convert a two-column file of (x, sigma) values, e.g. published
             or computed cross section data, into the table format.

  potential  compute collision cross sections directly from an intermolecular
             potential, by numerically integrating the classical deflection
             angle.  Supports Lennard-Jones 12-6 and a tabulated V(r), and
             emits any of the total cross section, the energy-dependent VSS
             alpha, and the full angular distribution.

  lxcat      convert a file exported from the LXCat project (www.lxcat.net)
             into the table format.  LXCat is the standard source of
             electron-neutral cross sections.  The database name, process
             description and the database's own "HOW TO REFERENCE" text are
             copied into the generated file, so the attribution required by
             LXCat travels with the data.

Examples:

  tabulate_cross_section.py vss --species data/ar.species --vss data/ar.vss \\
      --pair Ar Ar --emin 1.0e-4 --emax 1.0e3 -n 200 \\
      --keyword AR_AR_VHS -o data/ar_ar.tab

  tabulate_cross_section.py potential --form lj --eps-k 119.8 --sigma 3.405e-10 \\
      --species data/ar.species --pair Ar Ar --emit sigma alpha scatter \\
      --keyword AR_AR -o ar_lj.tab

  tabulate_cross_section.py potential --form file --vfile ar_ar_potential.txt \\
      --species data/ar.species --pair Ar Ar --emit sigma -o ar_ab.tab

  tabulate_cross_section.py lxcat N2_phelps.txt --list
  tabulate_cross_section.py lxcat N2_phelps.txt --process ELASTIC --target N2 \\
      --keyword E_N2_MT -o e_n2.tab

The potential mode validates itself against the standard Lennard-Jones
collision integrals with --selftest.

See doc/collide.html for the table file format.
"""

import argparse
import math
import sys

BOLTZ = 1.380649e-23
EV2J = 1.602176634e-19

XUNIT_SCALE = {"eV": EV2J, "J": 1.0, "K": BOLTZ, "m/s": 1.0}
YUNIT_SCALE = {"m^2": 1.0, "cm^2": 1.0e-4, "A^2": 1.0e-20}


# ----------------------------------------------------------------------
# SPARTA file readers
# ----------------------------------------------------------------------

def read_species(fname):
    """Return {id: mass} from a SPARTA species file."""
    species = {}
    with open(fname) as fp:
        for line in fp:
            line = line.split("#")[0].strip()
            if not line:
                continue
            words = line.split()
            if len(words) < 3:
                continue
            species[words[0]] = float(words[2])
    return species


def reduced_mass(species, isp, jsp):
    for name in (isp, jsp):
        if name not in species:
            sys.exit("error: species %s not found in the species file" % name)
    mi, mj = species[isp], species[jsp]
    return mi / 2.0 if isp == jsp else mi * mj / (mi + mj)


def read_vss(fname, isp, jsp):
    """Return (diam, omega, tref) for the isp/jsp pair from a VSS param file.

    An explicit cross-species line wins; otherwise the two self lines are
    averaged, exactly as CollideVSS::read_param_file() does.
    """
    self_params = {}
    cross = None
    with open(fname) as fp:
        for line in fp:
            line = line.split("#")[0].strip()
            if not line:
                continue
            words = line.split()
            if len(words) >= 5 and words[2] in ("table", "alpha", "scatter"):
                continue
            try:
                float(words[1])
                is_self = True
            except ValueError:
                is_self = False
            if is_self:
                if len(words) < 5:
                    continue
                self_params[words[0]] = tuple(float(w) for w in words[1:4])
            else:
                if len(words) < 6:
                    continue
                if {words[0], words[1]} == {isp, jsp}:
                    cross = tuple(float(w) for w in words[2:5])

    if cross:
        return cross
    for name in (isp, jsp):
        if name not in self_params:
            sys.exit("error: species %s not found in %s" % (name, fname))
    a = self_params[isp]
    b = self_params[jsp]
    return tuple(0.5 * (a[k] + b[k]) for k in range(3))


# ----------------------------------------------------------------------
# analytic VHS/VSS cross section
# ----------------------------------------------------------------------

def vss_sigma(g, diam, omega, tref, mr):
    """VHS/VSS total cross section at relative speed g, in m^2.

    Matches CollideVSS: sigma*g = prefactor * (g^2)^(1-omega) with
    prefactor = pi*d^2 * (2*k*Tref/mr)^(omega-1/2) / Gamma(5/2-omega).
    """
    prefactor = (math.pi * diam * diam *
                 (2.0 * BOLTZ * tref / mr) ** (omega - 0.5) /
                 math.gamma(2.5 - omega))
    return prefactor * g ** (1.0 - 2.0 * omega)


# ----------------------------------------------------------------------
# classical scattering from an intermolecular potential
# ----------------------------------------------------------------------

class Potential:
    """Classical deflection angle and transport cross sections for a
    spherically symmetric potential V(r).

    Works in reduced units throughout: x = r/rscale, V* = V/escale,
    E* = E/escale, b* = b/rscale.  The turning point is the smallest
    positive root of

        f(w) = 1 - b*^2 w^2 - V*(1/w)/E*,   w = rscale/r

    and the deflection angle is

        chi = pi - 2 b* int_0^{w_m} dw / sqrt(f(w))

    The substitution w = w_m (1 - s^2) removes the inverse-square-root
    endpoint singularity, and Gauss-Legendre nodes are interior so the
    integrand is never evaluated at the singular point.
    """

    def __init__(self, vstar, rscale, escale, np):
        self.vstar = vstar          # V*(x), vectorized over x
        self.rscale = rscale        # m
        self.escale = escale        # J
        self.np = np
        self.glx, self.glw = np.polynomial.legendre.leggauss(160)
        self.glx = 0.5 * (self.glx + 1.0)
        self.glw = 0.5 * self.glw

    def _f(self, w, b, E):
        np = self.np
        with np.errstate(all="ignore"):
            return 1.0 - (b * b) * w * w - self.vstar(1.0 / w) / E

    def chi(self, bs, E, nscan=4000, wmax=3.0):
        """Deflection angle for an array of impact parameters at energy E."""
        np = self.np
        bs = np.asarray(bs, float)
        w = np.linspace(1e-9, wmax, nscan)[None, :]
        F = self._f(w, bs[:, None], E)
        neg = F <= 0.0
        has = neg.any(axis=1)
        idx = np.maximum(np.where(has, neg.argmax(axis=1), 1), 1)
        lo = w[0, idx - 1].copy()
        hi = w[0, idx].copy()
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            pos = self._f(mid, bs, E) > 0.0
            lo = np.where(pos, mid, lo)
            hi = np.where(pos, hi, mid)
        wm = 0.5 * (lo + hi)

        s = self.glx[None, :]
        ww = wm[:, None] * (1.0 - s * s)
        with np.errstate(all="ignore"):
            f = self._f(ww, bs[:, None], E)
            integ = 2.0 * wm[:, None] * s / np.sqrt(np.where(f > 0, f, np.nan))
        val = np.nansum(self.glw[None, :] * integ, axis=1)
        return np.where(has, math.pi - 2.0 * bs * val, 0.0)

    def bgrid(self, E, nb, bmax):
        np = self.np
        return (np.arange(nb) + 0.5) * (bmax / nb)

    def Qstar(self, E, l, nb=2400, bmax=8.0):
        """Reduced transport cross section Q^(l)/(pi rscale^2)."""
        np = self.np
        b = self.bgrid(E, nb, bmax)
        c = np.cos(self.chi(b, E))
        return 2.0 * np.sum((1.0 - c ** l) * b) * (bmax / nb)

    def Omega(self, Tstar, l, s_, ng=48, **kw):
        """Reduced collision integral Omega^(l,s)*."""
        np = self.np
        x, w = np.polynomial.legendre.leggauss(ng)
        g = 0.5 * 4.0 * (x + 1)
        wt = 0.5 * 4.0 * w
        norm = 1.0 - (1.0 + (-1.0) ** l) / (2.0 * (1.0 + l))
        pref = 2.0 / (math.factorial(s_ + 1) * norm)
        tot = sum(wi * math.exp(-gi * gi) * gi ** (2 * s_ + 3) *
                  self.Qstar(gi * gi * Tstar, l, **kw)
                  for gi, wi in zip(g, wt) if gi > 1e-6)
        return pref * tot

    def bmax_for_chimin(self, E, chimin, blo=1e-3, bhi=40.0, nscan=600):
        """Largest impact parameter that still deflects by at least chimin.

        sigma_T is formally infinite for a potential of infinite range, so a
        cutoff is unavoidable.  Cutting on deflection angle rather than on a
        fixed b_max keeps sigma_T as small as the requested angular
        resolution allows, which is what controls how many near-zero
        deflection collisions the simulation has to perform.
        """
        np = self.np
        b = np.linspace(blo, bhi, nscan)
        c = np.abs(self.chi(b, E))
        big = np.where(c >= chimin)[0]
        if len(big) == 0:
            return blo
        i = big[-1]
        if i >= nscan - 1:
            return bhi
        lo, hi = b[i], b[i + 1]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if abs(self.chi(np.array([mid]), E)[0]) >= chimin:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def cos_cdf(self, E, ncos, nb=4000, bmax=8.0):
        """Inverse CDF of cos(chi) for impact parameters sampled uniformly
        in b^2 up to bmax, i.e. the angular distribution which accompanies
        a total cross section of pi*(bmax*rscale)^2."""
        np = self.np
        b = self.bgrid(E, nb, bmax)
        c = np.cos(self.chi(b, E))
        wgt = 2.0 * b                                  # dP proportional to b db
        order = np.argsort(c)
        c = c[order]
        wgt = wgt[order]
        cdf = np.cumsum(wgt)
        cdf = (cdf - 0.5 * wgt) / cdf[-1]
        probs = (np.arange(ncos) + 0.5) / ncos
        return np.interp(probs, cdf, c)


def lj_potential(np):
    """Lennard-Jones 12-6 in reduced units, V*(x) = 4(x^-12 - x^-6)."""
    def vstar(x):
        with np.errstate(all="ignore"):
            return 4.0 * (x ** -12 - x ** -6)
    return vstar


def file_potential(np, fname, rscale, escale):
    """Tabulated V(r) from a two-column file of r (Angstrom) and V (eV).

    Interpolated with a cubic spline in r, and continued outside the
    tabulated range by the power law implied by the two end points, which
    is the right behaviour for both the repulsive wall and the attractive
    tail of a physically motivated potential.
    """
    rs, vs = [], []
    with open(fname) as fp:
        for line in fp:
            line = line.split("#")[0].strip()
            if not line:
                continue
            w = line.replace(",", " ").split()
            if len(w) < 2:
                continue
            rs.append(float(w[0]) * 1.0e-10)
            vs.append(float(w[1]) * EV2J)
    if len(rs) < 4:
        sys.exit("error: need at least 4 points in the potential file")
    r = np.array(rs) / rscale
    v = np.array(vs) / escale
    if np.any(np.diff(r) <= 0):
        sys.exit("error: r values in the potential file must increase")

    def powfit(r0, r1, v0, v1):
        if v0 * v1 <= 0 or r0 <= 0:
            return None
        p = math.log(abs(v1 / v0)) / math.log(r1 / r0)
        a = v0 / r0 ** p
        return a, p

    lo = powfit(r[0], r[1], v[0], v[1])
    hi = powfit(r[-2], r[-1], v[-2], v[-1])

    def vstar(x):
        x = np.asarray(x, float)
        out = np.interp(x, r, v)
        if lo is not None:
            out = np.where(x < r[0], lo[0] * x ** lo[1], out)
        if hi is not None:
            out = np.where(x > r[-1], hi[0] * x ** hi[1], out)
        else:
            out = np.where(x > r[-1], 0.0, out)
        return out

    return vstar


# ----------------------------------------------------------------------
# LXCat export format
# ----------------------------------------------------------------------

LXCAT_PROCESSES = ("ELASTIC", "EFFECTIVE", "MOMENTUM", "EXCITATION",
                   "IONIZATION", "ATTACHMENT", "ROTATIONAL", "VIBRATIONAL")


def read_lxcat(fp):
    """Parse an LXCat export file into a list of process blocks.

    The format is a sequence of blocks, each opening with a process-type
    keyword on a line of its own, followed by the target species, an
    optional parameter, a set of KEY: value header lines, and the data
    bracketed by lines of dashes.  Database-level headers that precede the
    blocks supply the attribution text.
    """
    lines = fp.read().splitlines()
    database = ""
    reference = ""
    blocks = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if line.startswith("DATABASE:"):
            database = line.split(":", 1)[1].strip()
        elif line.startswith("HOW TO REFERENCE:"):
            reference = line.split(":", 1)[1].strip()
        elif line.startswith("PERMLINK:") and not reference:
            reference = line.split(":", 1)[1].strip()

        if line in LXCAT_PROCESSES:
            blk = {"process": line, "database": database,
                   "reference": reference, "header": [], "data": []}
            i += 1
            if i < len(lines):
                blk["target"] = lines[i].strip()
                i += 1
            # header lines up to the first dashed separator
            while i < len(lines) and not lines[i].strip().startswith("---"):
                h = lines[i].strip()
                if h:
                    blk["header"].append(h)
                    if h.startswith("PROCESS:"):
                        blk["desc"] = h.split(":", 1)[1].strip()
                    elif h.startswith("HOW TO REFERENCE:"):
                        blk["reference"] = h.split(":", 1)[1].strip()
                i += 1
            i += 1                       # skip the opening dashes
            while i < len(lines) and not lines[i].strip().startswith("---"):
                w = lines[i].replace(",", " ").split()
                if len(w) >= 2:
                    try:
                        blk["data"].append((float(w[0]), float(w[1])))
                    except ValueError:
                        pass
                i += 1
            if blk["data"]:
                blocks.append(blk)
        i += 1
    return blocks


def cmd_lxcat(args):
    blocks = read_lxcat(args.input)
    if not blocks:
        sys.exit("error: no cross section blocks found; is this an LXCat export?")

    if args.list:
        print("# %-3s %-12s %-8s %-24s %6s  %s" %
              ("idx", "process", "target", "database", "points", "description"))
        for n, b in enumerate(blocks):
            print("  %-3d %-12s %-8s %-24s %6d  %s" %
                  (n, b["process"], b.get("target", "?"), b["database"][:24],
                   len(b["data"]), b.get("desc", "")[:50]))
        return

    sel = [b for b in blocks
           if (args.index is None or blocks.index(b) == args.index)
           and (args.process is None or b["process"] == args.process.upper())
           and (args.target is None or b.get("target") == args.target)]
    if not sel:
        sys.exit("error: no block matched; run with --list to see what is available")
    if len(sel) > 1:
        sys.exit("error: %d blocks matched, narrow it with --index/--process/--target"
                 % len(sel))
    b = sel[0]

    xs = [e for e, s in b["data"]]
    ys = [s for e, s in b["data"]]

    # LXCat tables normally start at exactly 0 eV, which the table format
    # cannot index; drop leading non-positive energies
    while xs and xs[0] <= 0.0:
        xs.pop(0)
        ys.pop(0)
    if len(xs) < 2:
        sys.exit("error: fewer than 2 positive-energy points in the block")
    for k in range(1, len(xs)):
        if xs[k] <= xs[k - 1]:
            sys.exit("error: energies are not increasing at row %d" % (k + 1))

    header = ["converted from an LXCat export by tools/tabulate_cross_section.py",
              "",
              "LXCat, www.lxcat.net, retrieved from the file %s" % args.input.name,
              "database: %s" % (b["database"] or "unknown"),
              "process : %s" % b.get("desc", b["process"]),
              ""]
    if b["reference"]:
        header += ["HOW TO REFERENCE (from the LXCat file):", "  " + b["reference"], ""]
    header += ["Cite the database above and the LXCat project when publishing",
               "results obtained with this data."]

    write_table(args.output, args.keyword, xs, ys, "energy", "eV",
                args.yunits, args.extrap, header)


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------

def write_table(fp, keyword, xs, values, xvar, xunits, yunits, extrap,
                header, ncol=1):
    for line in header:
        fp.write("# %s\n" % line)
    fp.write("\n%s\n" % keyword)
    mstr = "" if ncol == 1 else " M %d" % ncol
    ystr = "" if yunits is None else " YUNITS %s" % yunits
    fp.write("N %d%s X %s XUNITS %s%s EXTRAP %s %s\n\n" %
             (len(xs), mstr, xvar, xunits, ystr, extrap[0], extrap[1]))
    for i, x in enumerate(xs):
        row = values[i] if ncol > 1 else [values[i]]
        fp.write("%-6d %- 20.12g %s\n" %
                 (i + 1, x, " ".join("%- 14.8g" % v for v in row)))


def grid(lo, hi, n, logspace):
    if n < 2:
        sys.exit("error: need at least 2 points")
    if logspace:
        if lo <= 0.0:
            sys.exit("error: log spacing requires a positive lower bound")
        step = math.log(hi / lo) / (n - 1)
        return [lo * math.exp(i * step) for i in range(n)]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


# ----------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------

def cmd_vss(args):
    species = read_species(args.species)
    isp, jsp = args.pair
    mr = reduced_mass(species, isp, jsp)
    diam, omega, tref = read_vss(args.vss, isp, jsp)

    xs = grid(args.emin, args.emax, args.n, not args.linear)
    sigmas = []
    for e in xs:
        g = math.sqrt(2.0 * e * EV2J / mr)
        sigmas.append(vss_sigma(g, diam, omega, tref, mr) / YUNIT_SCALE[args.yunits])

    header = [
        "VHS/VSS total cross section for %s + %s" % (isp, jsp),
        "generated by tools/tabulate_cross_section.py from %s" % args.vss,
        "diam = %g m, omega = %g, Tref = %g K, m_r = %g kg" %
        (diam, omega, tref, mr),
    ]
    write_table(args.output, args.keyword, xs, sigmas, "energy", "eV",
                args.yunits, args.extrap, header)


def cmd_csv(args):
    xs, sigmas = [], []
    for line in args.input:
        line = line.split("#")[0].strip()
        if not line:
            continue
        words = line.replace(",", " ").split()
        if len(words) < 2:
            sys.exit("error: expected two columns, got: %s" % line)
        xs.append(float(words[0]))
        sigmas.append(float(words[1]))

    if len(xs) < 2:
        sys.exit("error: need at least 2 data points")
    for i in range(1, len(xs)):
        if xs[i] <= xs[i - 1]:
            sys.exit("error: x values must increase monotonically "
                     "(row %d: %g <= %g)" % (i + 1, xs[i], xs[i - 1]))

    header = ["converted from %s by tools/tabulate_cross_section.py" %
              args.input.name]
    write_table(args.output, args.keyword, xs, sigmas, args.xvar, args.xunits,
                args.yunits, args.extrap, header)


LJ_REFERENCE = [
    # T*,  Omega(2,2)*, Omega(1,1)*   Hirschfelder, Curtiss & Bird
    (1.0, 1.593, 1.440),
    (2.0, 1.175, 1.075),
    (4.0, 0.9700, 0.8836),
    (10.0, 0.8242, 0.7424),
]


def cmd_potential(args):
    try:
        import numpy as np
    except ImportError:
        sys.exit("error: the potential mode requires numpy")

    if args.form == "lj":
        if args.eps_k is None or args.sigma is None:
            sys.exit("error: --form lj requires --eps-k and --sigma")
        rscale, escale = args.sigma, args.eps_k * BOLTZ
        vstar = lj_potential(np)
        desc = "Lennard-Jones 12-6, eps/k = %g K, sigma = %g m" % (
            args.eps_k, args.sigma)
    else:
        if args.vfile is None:
            sys.exit("error: --form file requires --vfile")
        rscale = args.sigma if args.sigma else 1.0e-10
        escale = (args.eps_k * BOLTZ) if args.eps_k else EV2J
        vstar = file_potential(np, args.vfile, rscale, escale)
        desc = "tabulated potential from %s" % args.vfile

    pot = Potential(vstar, rscale, escale, np)

    if args.selftest:
        if args.form != "lj":
            sys.exit("error: --selftest only applies to --form lj")
        print("Lennard-Jones collision integrals vs Hirschfelder et al.:")
        print("  T*     Omega(2,2)*  ref      Omega(1,1)*  ref")
        worst = 0.0
        for Ts, r22, r11 in LJ_REFERENCE:
            v22 = pot.Omega(Ts, 2, 2, nb=args.nb, bmax=args.bmax)
            v11 = pot.Omega(Ts, 1, 1, nb=args.nb, bmax=args.bmax)
            worst = max(worst, abs(v22 - r22) / r22, abs(v11 - r11) / r11)
            print("  %-6.2f %-12.4f %-8.4f %-12.4f %.4f" % (Ts, v22, r22, v11, r11))
        print("  worst relative error: %.2f%%" % (100 * worst))
        if worst > 0.01:
            sys.exit("error: self test exceeded 1%")
        return

    species = read_species(args.species)
    isp, jsp = args.pair
    mr = reduced_mass(species, isp, jsp)

    energies = grid(args.emin, args.emax, args.n, True)
    estars = [e * EV2J / escale for e in energies]

    emit = set(args.emit)

    # in cutoff mode a deflection-angle cutoff sets b_max per energy
    bmax_of = {}
    if args.mode == "cutoff" and args.chimin > 0.0:
        chimin = math.radians(args.chimin)
        for Es in [e * EV2J / escale for e in grid(args.emin, args.emax, args.n, True)]:
            bmax_of[Es] = pot.bmax_for_chimin(Es, chimin)

    # a scatter table is the angular distribution which accompanies a
    # hard cutoff at b_max, so it is only consistent with that sigma_T
    if "scatter" in emit and args.mode != "cutoff":
        sys.exit("error: --emit scatter requires --mode cutoff, since the "
                 "tabulated angular distribution corresponds to "
                 "sigma_T = pi*b_max^2")
    need_q1 = ("alpha" in emit) or (args.mode == "transport")
    sigmas, alphas, rows = [], [], []

    for Es in estars:
        q2 = pot.Qstar(Es, 2, nb=args.nb, bmax=args.bmax) * math.pi * rscale ** 2
        q1 = (pot.Qstar(Es, 1, nb=args.nb, bmax=args.bmax) * math.pi * rscale ** 2
              if need_q1 else 0.0)

        if args.mode == "viscosity":
            # isotropic scattering realizes Q2 = (2/3) sigma_T, so this
            # choice reproduces the viscosity at every temperature
            sig = 1.5 * q2
            alpha = 1.0
        elif args.mode == "transport":
            # VSS gives Q1/sigma_T = 2/(1+a) and Q2/Q1 = 2a/(2+a), so
            # sigma_T(E) with alpha(E) matches both transport cross sections
            R = q2 / q1
            R = min(R, 2.0 - 1.0e-9)
            alpha = 2.0 * R / (2.0 - R)
            sig = q1 * (1.0 + alpha) / 2.0
        else:
            # cutoff: sigma_T = pi*b_max^2 with the true angular distribution
            # with --chimin, b_max is set by the smallest deflection angle
            #   worth simulating, which keeps sigma_T as small as possible
            bm = bmax_of[Es] if bmax_of else args.bmax
            sig = math.pi * (bm * rscale) ** 2
            alpha = 1.0

        sigmas.append(sig)
        alphas.append(alpha)

    if "scatter" in emit:
        for Es in estars:
            bm = bmax_of[Es] if bmax_of else args.bmax
            rows.append(list(pot.cos_cdf(Es, args.ncos, nb=args.nb, bmax=bm)))

    base = [
        "%s for %s + %s" % (desc, isp, jsp),
        "generated by tools/tabulate_cross_section.py",
        "m_r = %g kg, sigma_T convention: %s" % (mr, args.mode),
    ]

    if "sigma" in emit:
        write_table(args.output, args.keyword, energies, sigmas, "energy",
                    "eV", "m^2", args.extrap,
                    base + ["total collision cross section"])
    if "alpha" in emit:
        args.output.write("\n")
        write_table(args.output, args.keyword + "_ALPHA", energies, alphas,
                    "energy", "eV", None, ("constant", "constant"),
                    base + ["energy-dependent VSS alpha"])
    if "scatter" in emit:
        args.output.write("\n")
        write_table(args.output, args.keyword + "_SCATTER", energies, rows,
                    "energy", "eV", None, ("constant", "constant"),
                    base + ["cos(chi) at equally spaced cumulative probability"],
                    ncol=args.ncos)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")
    sub.required = True

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--keyword", default="XSEC",
                        help="section keyword to write into the table file")
    common.add_argument("--yunits", default="m^2", choices=sorted(YUNIT_SCALE),
                        help="cross section units to write (default m^2)")
    common.add_argument("--extrap", nargs=2, default=["constant", "constant"],
                        metavar=("LO", "HI"),
                        choices=["constant", "powerlaw", "vss", "error"],
                        help="extrapolation modes below/above the table range")
    common.add_argument("-o", "--output", type=argparse.FileType("w"),
                        default=sys.stdout, help="output file (default stdout)")

    p = sub.add_parser("vss", parents=[common],
                       help="tabulate the analytic VHS/VSS cross section")
    p.add_argument("--species", required=True, help="SPARTA species file")
    p.add_argument("--vss", required=True, help="SPARTA VSS parameter file")
    p.add_argument("--pair", nargs=2, required=True, metavar=("SP1", "SP2"))
    p.add_argument("--emin", type=float, default=1.0e-4,
                   help="lowest relative translational energy, eV")
    p.add_argument("--emax", type=float, default=1.0e3,
                   help="highest relative translational energy, eV")
    p.add_argument("-n", type=int, default=200, help="number of values")
    p.add_argument("--linear", action="store_true",
                   help="space values linearly instead of logarithmically")
    p.set_defaults(func=cmd_vss)

    p = sub.add_parser("csv", parents=[common],
                       help="convert a two-column (x,sigma) data file")
    p.add_argument("input", type=argparse.FileType("r"))
    p.add_argument("--xvar", default="energy", choices=["energy", "speed"])
    p.add_argument("--xunits", default="eV", choices=sorted(XUNIT_SCALE))
    p.set_defaults(func=cmd_csv)

    p = sub.add_parser("lxcat", parents=[common],
                       help="convert a file exported from www.lxcat.net")
    p.add_argument("input", type=argparse.FileType("r"))
    p.add_argument("--list", action="store_true",
                   help="list the cross section blocks in the file and exit")
    p.add_argument("--process", default=None,
                   help="select by process type, e.g. ELASTIC or MOMENTUM")
    p.add_argument("--target", default=None, help="select by target species")
    p.add_argument("--index", type=int, default=None,
                   help="select by the index shown by --list")
    p.set_defaults(func=cmd_lxcat)

    p = sub.add_parser("potential", parents=[common],
                       help="compute cross sections from an intermolecular potential")
    p.add_argument("--form", default="lj", choices=["lj", "file"])
    p.add_argument("--eps-k", type=float, default=None,
                   help="potential well depth eps/k_B, K")
    p.add_argument("--sigma", type=float, default=None,
                   help="potential length scale, m")
    p.add_argument("--vfile", default=None,
                   help="two-column r (Angstrom) and V (eV) file, --form file")
    p.add_argument("--species", help="SPARTA species file")
    p.add_argument("--pair", nargs=2, metavar=("SP1", "SP2"))
    p.add_argument("--mode", default="viscosity",
                   choices=["viscosity", "transport", "cutoff"],
                   help="viscosity: sigma_T = (3/2)Q2, exact viscosity with "
                        "isotropic scattering; transport: sigma_T and alpha(E) "
                        "matching both Q1 and Q2; cutoff: sigma_T = pi*bmax^2 "
                        "with the true angular distribution")
    p.add_argument("--emit", nargs="+", default=["sigma"],
                   choices=["sigma", "alpha", "scatter"],
                   help="which tables to write")
    p.add_argument("--emin", type=float, default=1.0e-4)
    p.add_argument("--emax", type=float, default=1.0e1)
    p.add_argument("-n", type=int, default=120, help="number of energies")
    p.add_argument("--ncos", type=int, default=64,
                   help="angular resolution of a scatter table")
    p.add_argument("--nb", type=int, default=2400,
                   help="impact parameter samples per energy")
    p.add_argument("--bmax", type=float, default=8.0,
                   help="largest impact parameter, in units of the "
                        "potential length scale")
    p.add_argument("--chimin", type=float, default=0.0,
                   help="cutoff mode only: set b_max per energy from the "
                        "smallest deflection angle in degrees worth "
                        "simulating, instead of a fixed --bmax.  A larger "
                        "value gives a smaller sigma_T and so far fewer "
                        "near-zero deflection collisions")
    p.add_argument("--selftest", action="store_true",
                   help="check the numerics against tabulated LJ collision "
                        "integrals and exit")
    p.set_defaults(func=cmd_potential)

    args = parser.parse_args()
    if args.cmd == "lxcat" and args.list:
        cmd_lxcat(args)
        return
    if args.cmd == "potential" and not args.selftest:
        if not args.species or not args.pair:
            sys.exit("error: --species and --pair are required")
    args.func(args)


if __name__ == "__main__":
    main()
