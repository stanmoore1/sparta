################################################################################
# TEST: spin-conservation (allowed/forbidden transitions)
# N2-only gas using airspin.elec: ground X1Sigma and a1Sigma are spin 1,
# A3Sigma and B3Pi are spin 3. With spin conservation enforced (the default
# for N2-N2 collisions) and all particles starting in the ground state,
# the spin-3 states must remain EXACTLY unpopulated while the spin-1
# excited state (a1Sigma) is populated by collisions.
################################################################################
seed                12345
dimension           3
global              gridcut 1.0e-5 comm/sort yes
boundary            rr rr rr
create_box          0 0.0001 0 0.0001 0 0.0001
create_grid         2 2 2
balance_grid        rcb part

species             air.species N2 elecfile airspin.elec
mixture             gas N2 vstream 0.0 0.0 0.0 temp 30000.0 trot 100.0 tvib 100.0 telec 300.0

global              nrho 1.4141E24
global              fnum 3.5352E7

collide             vss gas air.vss relax constant
collide_modify      rotate no vibrate no electronic discrete
fix                 electronic elecmode
collide_modify      remain no

create_particles    gas n 40000 twopass

compute             temp temp
stats               1000
stats_style         step cpu np nattempt ncoll c_temp
dump                1 particle all 3000 dump.spin.* id type vx vy vz p_eelec p_elecstate
dump_modify         1 format float %.17e

timestep            1.00E-9
run                 3000
