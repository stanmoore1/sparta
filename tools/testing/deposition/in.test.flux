# Incident flux on an implicit surface, against the closed-form answer
#
# For a Maxwellian gas at rest against a wall at the same temperature, the
# one-sided impingement rate is known exactly:
#
#   number flux = n * vbar / 4        vbar = sqrt(8 k T / (pi m))
#   mass flux   = rho * vbar / 4
#
# with n the number density and rho = n*m.  So this needs no reference run:
# the answer is arithmetic.
#
# compute isurf/grid only had NET mflux, which is identically zero at a wall
# that does not react -- the molecule that leaves carries the mass of the one
# that arrived.  mflux_incident and nflux_incident are what an impingement
# driven deposition rate is built from, and this checks them.
#
# The surface is held still (rate 0), so this measures the flux and nothing
# else.  Two independent checks fall out:
#
#   nflux_incident              must equal n*vbar/4
#   mflux_incident/nflux_incident must equal the species mass, since every
#                               incident molecule of a single species carries
#                               exactly that
#
# Both are per unit area, the default "norm flux".  norm flow drops the area
# and is what fix ablate consumes; see the flux source on the doc page.
#
# The averaging window has to be long enough that essentially every surface
# element records a collision in it.  Averaging over the cells that did record
# one is biased high otherwise, since the cells left out are genuine zeros of
# the sample rather than places with no flux.

variable            NRHO index 1.0e20
variable            TEMP index 300.0

seed                12345
dimension           2
global              gridcut 0.0 comm/sort yes

boundary            r r p

create_box          0 20 0 80 -0.5 0.5
create_grid         20 80 1
balance_grid        rcb cell

global              nrho ${NRHO} fnum 1.0e18

species             air.species N
mixture             air N temp ${TEMP} vstream 0.0 0.0 0.0

variable            zero grid 0.0
fix                 ablate ablate all 0 1.0 v_zero mode deposit

global              surfs implicit
read_isurf          all 20 80 1 ramp.20x80 127.5 ablate push no

surf_collide        1 diffuse ${TEMP} 1.0
surf_modify         all collide 1

create_particles    air n 0

compute             FLUX isurf/grid all all nflux_incident mflux_incident
fix                 AVG ave/grid all 1 500 500 c_FLUX[*]

# average over the cells that hold surface only: a plain reduce ave would
# divide by every cell in the box, most of which see no flux at all
variable            hit grid (f_AVG[1]>0)
compute             NSUM reduce sum f_AVG[1]
compute             MSUM reduce sum f_AVG[2]
compute             NCELL reduce sum v_hit
variable            NF equal c_NSUM/(c_NCELL+1.0e-30)
variable            MF equal c_MSUM/(c_NCELL+1.0e-30)

timestep            1.0e-6

stats               500
stats_style         step np v_NF v_MF c_NCELL

run                 2000
