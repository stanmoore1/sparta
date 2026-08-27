################################################################################
# SPARTA logo, idea 11: the emblem spinning in a flow
#
# The lettering is a rigid body with angular momentum rather than translation,
# so the mark itself turns while the gas streams past it.  The wake sweeps
# around with the letters instead of sitting still behind them.
#
# fix rigid integrates the orientation quaternion; richardson is the more
# accurate of the two schemes it offers and is worth it here, since the body
# turns through a full revolution during the run.
################################################################################

dimension           2
seed                12345
global              gridcut 0.0 comm/sort yes
boundary            oo rr p

create_box          -600e-1 600e-1 -600e-1 600e-1 -0.5 0.5
create_grid         100 100 1
balance_grid        rcb cell

read_surf           letters.surf group letters
surf_collide        skin diffuse 300.0 1.0
surf_modify         all collide skin

species             logo.species N O B Nh Oh Nm Om
global              fnum 8.4e16

mixture             flow N O nrho 1.17e18 temp 300 vstream 1200.0 0.0 0.0
mixture             flow N frac 0.5
mixture             flow O frac 0.5

create_particles    flow n 0
fix                 in emit/face flow xlo

# No collide.  A rotating body sweeps cells to zero flow volume as it turns and
# collide rejects those outright with "Collision cell volume is zero", so this
# one runs free molecular - which is a real rarefied regime, and the wake it
# leaves is a shadow rather than a shock.

timestep            1.0e-4
stats               400
stats_style         step cpu np ncoll nscoll

# angmom lz = izz * omega.  izz 100 with lz 6280 is omega = 62.8 rad/s, one
# revolution in 0.1 s, which is 1000 steps at this timestep.
compute             csurf surf letters all fx fy fz tx ty tz com 0.0 0.0 0.0
fix                 spin rigid letters csurf body &
                    mass 1.0e-3 com 0.0 0.0 0.0 vcom 0.0 0.0 0.0 &
                    moi 100.0 100.0 100.0 0.0 0.0 0.0 angmom 0.0 0.0 6280.0 &
                    rotate richardson
global              rigid spin

dump                3 image all 20 spin.*.ppm type type pdiam 0.9 &
                    surf one 0.005 box no 1 size 500 500 &
                    view 0 0 center s 0.5 0.5 0.5 zoom 1.98
dump_modify         3 pad 5 backcolor white &
                    color shieldred 0.75 0 0 color shieldgold 0.95 0.75 0.05 &
                    color ink 0.0 0.0 0.0 &
                    pcolor 1 shieldgold pcolor 2 shieldred scolor * ink

run                 1200
