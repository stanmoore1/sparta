"""Step 2 run list: non-equilibrium response of O2 + N2 -> O + O + N2 at a
translational-rotational temperature of 15 000 K (Sec. VII B).

  vib      vibrational populations frozen at TV (vibrational relaxation
           off, air_vfrozen.species), electronic thermal: microcanonical
           factor (release) and the instantaneous-DOF treatment (compare)
  el       electronic populations frozen at TE (o2n2_frozen.elec),
           vibration thermal: microcanonical factor (release) and direct
           inclusion with instantaneous-DOF vibration (compare)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pkg import Run as _Run, DATA, size_params  # noqa: E402


def Run(step, tag, deck, variables, kind="release", files=()):
    """comparison-build runs use the deck variant without rotfile"""
    if kind == "compare":
        deck = deck + "_cmp"
    return _Run(step, tag, deck, variables, kind=kind, files=files)

STEP = "step2_noneq"
SIZES = {"quick": dict(seeds=2, NP=40000, NSTEP=2000)}
FILES = ("o2n2.tce", "o2n2.elec", "o2n2_frozen.elec", "air_vfrozen.species")
T = 15000.0
TVS = (300.0, 1000.0, 3000.0, 6000.0, 10000.0, 15000.0, 25000.0, 40000.0)
TES = (100.0, 3000.0, 8000.0, 15000.0, 25000.0, 40000.0)
PE = "partial_energy no"


def runs(size="quick"):
    sz = size_params(STEP, SIZES, size)
    out = []
    for s in range(sz["seeds"]):
        base = dict(DATA=DATA, SEED=22345 + 1000 * s, T=T, ROT="smooth",
                    VIB="discrete", TCE="o2n2.tce", NP=sz["NP"],
                    NSTEP=sz["NSTEP"])
        for TV in TVS:
            v = dict(base, TV=TV, TE=T, EFILE="o2n2.elec",
                     SPFILE="air_vfrozen.species", RMOD=PE)
            out.append(Run(STEP, "vib_micro_TV%d_s%d" % (TV, s), "in.rate", v,
                           files=FILES))
            v = dict(base, TV=TV, SPFILE="air_vfrozen.species",
                     RMOD=PE + " vib_energy dof")
            out.append(Run(STEP, "vib_inst_TV%d_s%d" % (TV, s),
                           "in.rate_noelec", v, kind="compare", files=FILES))
        for TE in TES:
            v = dict(base, TV=T, TE=TE, EFILE="o2n2_frozen.elec",
                     SPFILE=os.path.join(DATA, "air.species"), RMOD=PE)
            out.append(Run(STEP, "el_micro_TE%d_s%d" % (TE, s), "in.rate", v,
                           files=FILES))
            v = dict(v, RMOD=PE + " vib_energy dof elec_energy yes")
            out.append(Run(STEP, "el_incl_TE%d_s%d" % (TE, s), "in.rate", v,
                           kind="compare", files=FILES))
    return out
