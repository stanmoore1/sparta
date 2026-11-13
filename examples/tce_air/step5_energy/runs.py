"""Step 5 run list: energy conservation and the consistency conditions
(Sec. VII D, Table VIII).

Five combinations of reaction energies and species masses in the dense
5-species reactor, and the 11-species reservoir with the released data:

  unclosed_unconserved  data/air.tce and data/air.species before closure
                        (commit d6cecfe6 of this branch)
  unclosed_conserved    old reaction energies, released masses
  closed_unconserved    released reaction energies, old masses
  closed_norecomb       ... with recombination switched off
  closed_conserved      released data
  full11                11 species with ions and electrons, released data
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pkg import Run, DATA, size_params  # noqa: E402

STEP = "step5_energy"
SIZES = {"quick": dict(seeds=1, NP=10000, NSTEP=400)}
DT, NOUT = 2.0e-11, 20
OLD_TCE, OLD_SP = "unclosed_air.tce", "unconserved_air.species"
NEW_TCE, NEW_SP = os.path.join(DATA, "air.tce"), os.path.join(DATA,
                                                              "air.species")
CASES = {  # name: (reaction file, species file, react_modify)
    "unclosed_unconserved": (OLD_TCE, OLD_SP, "rboost 1.0"),
    "unclosed_conserved": (OLD_TCE, NEW_SP, "rboost 1.0"),
    "closed_unconserved": (NEW_TCE, OLD_SP, "rboost 1.0"),
    "closed_norecomb": (NEW_TCE, OLD_SP, "rboost 1.0 recomb no"),
    "closed_conserved": (NEW_TCE, NEW_SP, "rboost 1.0"),
}


def runs(size="quick"):
    sz = size_params(STEP, SIZES, size)
    out = []
    for s in range(sz["seeds"]):
        seed = 52345 + 1000 * s
        for name, (tce, sp, rmod) in CASES.items():
            v = dict(DATA=DATA, SEED=seed, SPFILE=sp, TCE=tce, RMOD=rmod,
                     NP=sz["NP"], NSTEP=sz["NSTEP"], NOUT=NOUT, DT=DT)
            out.append(Run(STEP, "%s_s%d" % (name, s), "in.energy", v,
                           files=(OLD_TCE, OLD_SP)))
        v = dict(DATA=DATA, SEED=seed, NP=sz["NP"], NSTEP=sz["NSTEP"],
                 NOUT=NOUT, DT=DT)
        out.append(Run(STEP, "full11_s%d" % s, "in.energy11", v))
    return out
