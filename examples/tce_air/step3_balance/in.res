# Frozen-composition reservoir of N2, O, NO and N (one quarter each), all
# internal modes and electronic states modeled, air data of data/air.*:
# reactions are tallied but not performed (compute_chem_rates yes), so the
# ratio of forward to reverse tallies of a pair measures k_f/k_r at T.
#
# Variables: DATA, SEED, T, NRHO (number density), NP (particles), NSTEP
# (steps of 1 ns), TCE (reaction file), VIB (discrete or smooth), RMOD
# (extra react_modify settings: "recomb yes" by default, or
# "reverse auto keq_file park.keq")

seed                ${SEED}
dimension           3
global              gridcut 1.0e-5 comm/sort yes
boundary            rr rr rr
create_box          0 1.0e-4 0 1.0e-4 0 1.0e-4
create_grid         2 2 2
balance_grid        rcb part

species             ${DATA}/air.species N2 O NO N rotfile ${DATA}/air.rot elecfile ${DATA}/air.elec
mixture             air N2 O NO N vstream 0.0 0.0 0.0 temp ${T}
mixture             air N2 frac 0.25
mixture             air O  frac 0.25
mixture             air NO frac 0.25
mixture             air N  frac 0.25

variable            fn equal ${NRHO}*1.0e-12/${NP}
global              nrho ${NRHO}
global              fnum ${fn}

collide             vss air ${DATA}/air.vss relax constant
collide_modify      rotate discrete vibrate ${VIB} electronic discrete
fix                 electronic elecmode
collide_modify      remain no

react               tce ${TCE}
react_modify        partial_energy no compute_chem_rates yes rboost 1.0 ${RMOD}

create_particles    air n ${NP} twopass
compute             temp temp
stats               1000
stats_style         step cpu np nattempt ncoll c_temp
timestep            1.0e-9
run                 ${NSTEP}
