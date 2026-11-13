"""Step 7 run list: the probability cap with reverse reactions (Sec. IV D
and Sec. VII G, Table X), in NO/N/O gas with only NO + O and NO + N
dissociation and their derived recombinations.

  frz   frozen reservoirs (compute_chem_rates yes) at 5600, 8000, 10 000
        and 15 000 K: the realized forward rates relative to their fits,
        and the recombination rates relative to the uncapped
        detailed-balance values, against the capped fractions computed
        from the microcanonical forward probability
  rx    adiabatic reactors at 1e25 m^-3 relaxing to the NO = N + O
        equilibrium (about 5630 K): the realized equilibrium constant
        relative to the partition-function value, for three seeds (cap),
        a tenth of the density (lown), rates divided by 30 (slow, no cap),
        and relaxation of NO at every collision (fast); and, with the
        build before the cap correction (kind 'precap', optional), the
        forward-only cap
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pkg import Run, DATA, size_params  # noqa: E402

STEP = "step7_cap"
# NPF/NSTF: frozen reservoirs; NPR/NSTR: reactors (steps of 5e-11 s; the
# low-density reactor uses steps of 5e-10 s and NSTL steps)
SIZES = {"quick": dict(seeds=3, NPF=20000, NSTF=4000, NPR=10000,
                       NSTR=150000, NSTL=150000)}
FILES = ("cap.tce", "cap_slow.tce", "nofast.species")
NRHO = 1.0e25
X_EQ = dict(XNO=0.3326, XO=0.3348, XN=0.3326)     # frozen reservoirs
# the equilibrium at 5600 K and 1e25 m^-3 of the data files (tce_ref
# Data.keq): the reactor without cap starts there, since with rates
# divided by 30 it relaxes too slowly to reach equilibrium from elsewhere
X_SLOW = dict(XNO=0.3233, XO=0.3383, XN=0.3383)
X_RX = dict(XNO=0.5, XO=0.25, XN=0.25)            # reactor start, 9000 K
FRZ_T = (5600.0, 8000.0, 10000.0, 15000.0)
SP = os.path.join(DATA, "air.species")


def runs(size="quick"):
    sz = size_params(STEP, SIZES, size)
    out = []
    for T in FRZ_T:
        v = dict(DATA=DATA, SEED=72345, SPFILE=SP, TCE="cap.tce", T=T,
                 NRHO=NRHO, NP=sz["NPF"], NSTEP=sz["NSTF"], DT=5.0e-11,
                 NOUT=1000, CHEM="compute_chem_rates yes", **X_EQ)
        out.append(Run(STEP, "frz_T%d" % T, "in.cap", v, files=FILES))
    base = dict(DATA=DATA, SPFILE=SP, TCE="cap.tce", T=9000.0, NRHO=NRHO,
                NP=sz["NPR"], NSTEP=sz["NSTR"], DT=5.0e-11, NOUT=1000,
                CHEM="compute_chem_rates no", **X_RX)
    for s in range(sz["seeds"]):
        out.append(Run(STEP, "rx_cap_s%d" % s, "in.cap",
                       dict(base, SEED=73345 + 1000 * s), files=FILES))
    out.append(Run(STEP, "rx_lown_s0", "in.cap",
                   dict(base, SEED=73345, NRHO=NRHO / 10, DT=5.0e-10,
                        NSTEP=sz["NSTL"]), files=FILES))
    out.append(Run(STEP, "rx_slow_s0", "in.cap",
                   dict(base, SEED=73345, TCE="cap_slow.tce", T=5600.0,
                        **X_SLOW), files=FILES))
    out.append(Run(STEP, "rx_fast_s0", "in.cap",
                   dict(base, SEED=73345, SPFILE="nofast.species"),
                   files=FILES))
    out.append(Run(STEP, "rx_precap_s0", "in.cap", dict(base, SEED=73345),
                   kind="precap", files=FILES))
    return out
