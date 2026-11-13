"""Step 6 run list: reverse rates without a cell temperature (Sec. VII E,
Table XI, and the non-equilibrium response of recombination in Sec. VII C).

  tint  frozen two-temperature reservoirs, translation at 8000 K and all
        internal modes at T_int = 4000, 8000, 12 000, 20 000 K, for the
        Zeldovich pair of data/air.tce (rel: derived reverse NO + N ->
        N2 + O, barrierless) and for the pair written the other way round
        (bar: derived reverse N2 + O -> NO + N, barriered; the paper's
        Table XI)
  mode  translation at 8000 K, one internal mode (rotation, vibration or
        electronic states) at 4000 K or 20 000 K and the others at 8000 K,
        for the rel pair
  recel N-rich dense reservoir at 8000 K with the electronic states frozen
        at T_el = 4000, 8000, 20 000 K: N + N + N -> N2 + N
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pkg import Run, DATA, size_params  # noqa: E402

STEP = "step6_noneq_reverse"
# NSTEP: rel pair; NSTB: bar pair (its reverse is rarer); NSTR: recel
SIZES = {"quick": dict(seeds=2, NP=40000, NSTEP=4000, NSTB=12000,
                       NSTR=1000)}
TTR = 8000.0
TINT = (4000.0, 8000.0, 12000.0, 20000.0)
MODES = {"rot": "TROT", "vib": "TVIB", "elec": "TELEC"}
MODE_T = {"cold": 4000.0, "hot": 20000.0}
TEL = (4000.0, 8000.0, 20000.0)
NHIGH = 1.414086e25
PAIRS = {"rel": "zeldovich.tce", "bar": "zeldovich_barrier.tce"}
FILES = ("zeldovich.tce", "zeldovich_barrier.tce", "recomb_atom.tce",
         "air_frozen.species", "air_frozen.elec")


def runs(size="quick"):
    sz = size_params(STEP, SIZES, size)
    out = []
    for s in range(sz["seeds"]):
        seed = 62345 + 1000 * s
        for pair, tce in PAIRS.items():
            for Ti in TINT:
                v = dict(DATA=DATA, SEED=seed, T=TTR, TROT=Ti, TVIB=Ti,
                         TELEC=Ti, TCE=tce, NP=sz["NP"],
                         NSTEP=sz["NSTEP"] if pair == "rel" else sz["NSTB"])
                out.append(Run(STEP, "tint_%s_TI%d_s%d" % (pair, Ti, s),
                               "in.res2T", v, files=FILES))
        for mode, var in MODES.items():
            for lab, Tm in MODE_T.items():
                v = dict(DATA=DATA, SEED=seed, T=TTR, TROT=TTR, TVIB=TTR,
                         TELEC=TTR, TCE=PAIRS["rel"], NP=sz["NP"],
                         NSTEP=sz["NSTEP"])
                v[var] = Tm
                out.append(Run(STEP, "mode_%s_%s_s%d" % (mode, lab, s),
                               "in.res2T", v, files=FILES))
        for Te in TEL:
            v = dict(DATA=DATA, SEED=seed, T=TTR, TELEC=Te, NRHO=NHIGH,
                     NP=sz["NP"], NSTEP=sz["NSTR"])
            out.append(Run(STEP, "recel_TE%d_s%d" % (Te, s), "in.recel", v,
                           files=FILES))
    return out
