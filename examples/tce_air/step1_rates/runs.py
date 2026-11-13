"""Step 1 run list: equilibrium rates of O2 + N2 -> O + O + N2 (the
paper's test case) under the TCE treatments of Secs. III-IV.

Two executables are used.  'release' is this branch: its microcanonical
factor is selected by react_modify partial_energy no.  'compare' is the
comparison build of the earlier treatments (react_modify vib_energy dof:
Bird's instantaneous vibrational DOF; elec_energy yes|no|micro: electronic
energy added to the collision energy, left out, or through the
microcanonical factor of the electronic ladder alone).  Runs that need
'compare' are skipped when it is not built.

Groups (each a figure or table of the paper):
  fig1    vibration left out (VIB no, so the analytic curves of Fig. 1,
          z = 2, apply): electronic treatments at 5, 10, 20, 40 kK
  chan    electronically driven reactions at Ttr = 2000 K with electronic
          populations frozen at 40 000 K (Fig. 1 right)
  tab6    discrete vibration at 10, 15, 20 kK (Table VI top, Fig. 3)
  frz     discrete vibration, electronic populations frozen at 100 K and
          40 000 K, Ttr = 15 000 K (Table VI bottom)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pkg import Run, DATA  # noqa: E402

STEP = "step1_rates"
SIZES = {"quick": dict(seeds=2, NP=40000, NSTEP=2000),
         "paper": dict(seeds=5, NP=1000000, NSTEP=2000)}

FILES = ("o2n2.tce", "o2n2_refit.tce", "o2n2.elec", "o2n2_dof1.elec",
         "o2n2_frozen.elec", "o2n2_frozen_dof1.elec")

PE = "partial_energy no"
TREAT = {
    # name: (executable, deck, react_modify, electronic file, tce file)
    "micro":     ("release", "in.rate", PE, "o2n2.elec", "o2n2.tce"),
    "incl":      ("compare", "in.rate", PE + " vib_energy dof elec_energy yes",
                  "o2n2.elec", "o2n2.tce"),
    "dof1":      ("compare", "in.rate", PE + " vib_energy dof elec_energy yes",
                  "o2n2_dof1.elec", "o2n2.tce"),
    "dof1refit": ("compare", "in.rate", PE + " vib_energy dof elec_energy yes",
                  "o2n2_dof1.elec", "o2n2_refit.tce"),
    "excl":      ("compare", "in.rate", PE + " vib_energy dof elec_energy no",
                  "o2n2.elec", "o2n2.tce"),
    "microel":   ("compare", "in.rate",
                  PE + " vib_energy dof elec_energy micro",
                  "o2n2.elec", "o2n2.tce"),
    "inst":      ("compare", "in.rate_noelec", PE + " vib_energy dof",
                  None, "o2n2.tce"),
    "vibonly":   ("release", "in.rate_noelec", PE, None, "o2n2.tce"),
}


def _run(group, treat, T, seed, size, VIB="discrete", TE=None, efile=None,
         ROT="smooth"):
    kind, deck, rmod, ef, tce = TREAT[treat]
    ef = efile or ef
    v = dict(DATA=DATA, SEED=12345 + 1000 * seed, T=T, TV=T,
             TE=TE if TE is not None else T, ROT=ROT, VIB=VIB,
             RMOD=rmod.replace(" ", " "), TCE=tce,
             NP=SIZES[size]["NP"], NSTEP=SIZES[size]["NSTEP"])
    if ef:
        v["EFILE"] = ef
    tag = "%s_%s_T%d%s%s_s%d" % (group, treat, T,
                                 "_TE%d" % TE if TE is not None else "",
                                 "_rot" if ROT != "smooth" else "", seed)
    return Run(STEP, tag, deck, v, kind=kind, files=FILES)


def runs(size="quick"):
    n = SIZES[size]["seeds"]
    out = []
    for s in range(n):
        for T in (5000.0, 10000.0, 20000.0, 40000.0):
            for tr in ("micro", "incl", "dof1", "dof1refit"):
                out.append(_run("fig1", tr, T, s, size, VIB="no"))
        for tr, ef in (("micro", "o2n2_frozen.elec"),
                       ("excl", "o2n2_frozen.elec"),
                       ("incl", "o2n2_frozen.elec"),
                       ("dof1", "o2n2_frozen_dof1.elec")):
            out.append(_run("chan", tr, 2000.0, s, size, VIB="no",
                            TE=40000.0, efile=ef))
        for T in (10000.0, 15000.0, 20000.0):
            for tr in ("inst", "incl", "excl", "microel", "micro", "vibonly"):
                out.append(_run("tab6", tr, T, s, size))
        out.append(_run("tab6", "micro", 15000.0, s, size, ROT="discrete"))
        for TE in (100.0, 40000.0):
            for tr in ("incl", "excl", "microel"):
                out.append(_run("frz", tr, 15000.0, s, size, TE=TE,
                                efile="o2n2_frozen.elec"))
    return out
