"""Step 6 analysis: reverse rates without a cell temperature.

Produces, in results/step6_noneq_reverse/:
  table11_cellT.tex/.txt  Table XI: rate relative to its value at
                          T_int = T_tr = 8000 K, measured and predicted, for
                          the reverse of each configuration (and the forward
                          of data/air.tce), and the value of a model
                          evaluated at the translational cell temperature
  numbers.json, results.txt

References (tce_ref.py): ExchangeReverse, the derived reverse probability
of Eq. (exch) with its absolute calibration and the cap at one, averaged
over the collision ensemble of the reverse pair with each mode at its own
temperature; the same construction for the forward microcanonical
probability (forward=True); Recombination3, the three-body average of
Eq. (pr3) with the electronic ladders at T_el.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import pkg                 # noqa: E402
import tce_ref as R        # noqa: E402
sys.modules.pop("runs", None)  # this step's runs.py, not another step's
from runs import (runs, STEP, TTR, TINT, MODES, MODE_T, TEL,  # noqa: E402
                  PAIRS)

LAB = {"rel": ("N2 + O --> NO + N", "NO + N --> N2 + O"),
       "bar": ("NO + N --> N2 + O", "N2 + O --> NO + N")}


def counts(rl, prefix, label):
    sel = [r for r in rl if r.tag.startswith(prefix + "_s") and r.done()]
    per = [pkg.tallies(r.log).get(label, 0) for r in sel]
    return sum(per), per


def ratio(rl, pre, pre0, label):
    """tally ratio of two run sets (same particles and steps) with its
    error: Poisson, or the spread over seeds if larger"""
    n, per = counts(rl, pre, label)
    n0, per0 = counts(rl, pre0, label)
    if n == 0 or n0 == 0:
        return None, None
    r = n / n0
    e = r * math.sqrt(1.0 / n + 1.0 / n0)
    if len(per) > 1 and len(per) == len(per0):
        rs = [a / b for a, b in zip(per, per0) if b > 0]
        if len(rs) > 1:
            e = max(e, np.std(rs, ddof=1) / math.sqrt(len(rs)))
    return r, e


def check_frozen_files(rep):
    def rows(path, skip):
        out = []
        for l in open(path):
            w = l.split()
            if l.strip() and not l.lstrip().startswith("#") and len(w) > 2:
                out.append([x for i, x in enumerate(w) if i not in skip(w)])
        return out
    sp_skip = lambda w: {4, 6}
    el_skip = lambda w: {2 + 5 * k + 1 for k in range(int(w[1]))}
    for f, ref, sk in (("air_frozen.species", "air.species", sp_skip),
                       ("air_frozen.elec", "air.elec", el_skip)):
        a = rows(os.path.join(HERE, f), sk)
        b = rows(os.path.join(R.DATA, ref), sk)
        same = len(a) == len(b) and all(
            len(x) == len(y) and all(float(p) == float(q) if
                                     _isnum(p) else p == q
                                     for p, q in zip(x, y))
            for x, y in zip(a, b))
        rep.check(same, "%s equals data/%s except the relaxation "
                  "probabilities" % (f, ref))


def _isnum(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def analyze(size="quick"):
    rep = pkg.Report(STEP + " (" + size + ")")
    out = os.path.join(pkg.results_dir(), STEP)
    os.makedirs(out, exist_ok=True)
    rl = runs(size)
    check_frozen_files(rep)
    D = R.Data()
    num, table = {}, []
    refs = {}
    for pair, tce in PAIRS.items():
        Dp = R.Data(tce=os.path.join(HERE, tce))
        fwd = Dp.reaction(LAB[pair][0])
        refs[pair] = (R.ExchangeReverse(Dp, fwd),
                      R.ExchangeReverse(Dp, fwd, forward=True))

    # ---- all internal modes at T_int
    for pair in PAIRS:
        rev, fw = refs[pair]
        base_r, base_f = rev.mean_p(TTR), fw.mean_p(TTR)
        for Ti in TINT:
            pre, pre0 = "tint_%s_TI%d" % (pair, Ti), "tint_%s_TI%d" % (
                pair, TTR)
            pr = rev.mean_p(TTR, Trot=Ti, Tvib=Ti, Telec=Ti) / base_r
            pf = fw.mean_p(TTR, Trot=Ti, Tvib=Ti, Telec=Ti) / base_f
            mr, er = ratio(rl, pre, pre0, LAB[pair][1])
            mf, ef = ratio(rl, pre, pre0, LAB[pair][0])
            if mr is None:
                continue
            table.append((pair, Ti, mr, er, pr, mf, ef, pf))
            num["tint_%s_%d" % (pair, Ti)] = dict(
                reverse=[mr, 2 * er, pr], forward=[mf, 2 * ef, pf])
            if Ti == TTR:
                continue
            rep.check(abs(mr / pr - 1) <= 3 * er / pr + 0.005,
                      "%s T_int=%5d K: reverse %-18s x%.3f +/- %.3f, "
                      "predicted x%.3f" % (pair, Ti, LAB[pair][1], mr,
                                           2 * er, pr))
            rep.check(abs(mf / pf - 1) <= 3 * ef / pf + 0.005,
                      "%s T_int=%5d K: forward %-18s x%.3f +/- %.3f, "
                      "predicted x%.3f" % (pair, Ti, LAB[pair][0], mf,
                                           2 * ef, pf))

    # ---- one mode at a time (rel pair)
    rev, fw = refs["rel"]
    base_r = rev.mean_p(TTR)
    for mode, var in MODES.items():
        for lab, Tm in MODE_T.items():
            kw = {"Trot": TTR, "Tvib": TTR, "Telec": TTR}
            kw[{"TROT": "Trot", "TVIB": "Tvib", "TELEC": "Telec"}[var]] = Tm
            pr = rev.mean_p(TTR, **kw) / base_r
            mr, er = ratio(rl, "mode_%s_%s" % (mode, lab),
                           "tint_rel_TI%d" % TTR, LAB["rel"][1])
            if mr is None:
                continue
            num["mode_%s_%s" % (mode, lab)] = [mr, 2 * er, pr]
            rep.check(abs(mr / pr - 1) <= 3 * er / pr + 0.005,
                      "%-4s at %5d K, others at %d K: reverse x%.3f +/- "
                      "%.3f, predicted x%.3f" % (mode, Tm, TTR, mr, 2 * er,
                                                 pr))

    # ---- recombination with the electronic ladders at T_el
    rc = R.Recombination3(D, D.reaction("N2 + N --> N + N + N"))
    for Te in TEL:
        if Te == TTR:
            continue
        mr, er = ratio(rl, "recel_TE%d" % Te, "recel_TE%d" % TTR,
                       "N + N --> N2 + N")
        if mr is None:
            continue
        pr = rc.ratio(TTR, Te)
        num["recel_%d" % Te] = [mr, 2 * er, pr]
        rep.check(abs(mr / pr - 1) <= 3 * er / pr + 0.005,
                  "recombination N + N + N, T_el=%5d K: x%.3f +/- %.3f, "
                  "three-body quadrature x%.4f" % (Te, mr, 2 * er, pr))
    _table(out, table)
    with open(os.path.join(out, "numbers.json"), "w") as fh:
        json.dump(num, fh, indent=1, default=float)
    return rep.write()


def _table(out, table):
    L = ["configuration & $T_{\\mathrm{int}}$ (K) & reverse measured & "
         "microcanonical & forward measured & microcanonical & "
         "translational cell temperature \\\\", "\\hline"]
    T = ["%-4s %8s %10s %8s %10s %8s %6s" % ("pair", "T_int", "rev meas",
                                             "rev pred", "fwd meas",
                                             "fwd pred", "cell")]
    for pair, Ti, mr, er, pr, mf, ef, pf in table:
        L.append("%s & %d & %.3f & %.3f & %.3f & %.3f & 1.000 \\\\"
                 % ({"rel": "air.tce", "bar": "reversed"}[pair], Ti, mr, pr,
                    mf, pf))
        T.append("%-4s %8d %10.3f %8.3f %10.3f %8.3f %6.3f"
                 % (pair, Ti, mr, pr, mf, pf, 1.0))
    open(os.path.join(out, "table11_cellT.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(out, "table11_cellT.txt"), "w").write("\n".join(T) + "\n")


if __name__ == "__main__":
    sys.exit(analyze(sys.argv[1] if len(sys.argv) > 1 else "quick"))
