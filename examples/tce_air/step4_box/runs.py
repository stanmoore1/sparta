"""Step 4 run list: a closed reacting box with the Zeldovich pair
(Sec. VII C, Fig. 9).

  iso  thermostatted at 15 kK, started on the N2 + O side (NO fraction of
       the NO + N2 pool 0.1) and on the NO + N side (0.9)
  adi  the same two starts without thermostat (adiabatic box)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pkg import Run, DATA, size_params  # noqa: E402

STEP = "step4_box"
SIZES = {"quick": dict(seeds=1, NP=40000, NSTEP=80000)}
T0 = 15000.0
STARTS = {"fwd": (0.45, 0.05), "bwd": (0.05, 0.45)}   # (FN2 = FO, FNO = FN)


def runs(size="quick"):
    sz = size_params(STEP, SIZES, size)
    out = []
    for s in range(sz["seeds"]):
        seed = 42345 + 1000 * s
        for mode, thermo in (("iso", 1), ("adi", 0)):
            for side, (fa, fb) in STARTS.items():
                v = dict(DATA=DATA, SEED=seed, T=T0, FN2=fa, FNO=fb,
                         THERMO=thermo, NP=sz["NP"], NSTEP=sz["NSTEP"])
                out.append(Run(STEP, "%s_%s_s%d" % (mode, side, s), "in.box",
                               v, files=("zeldovich.tce",)))
    return out
