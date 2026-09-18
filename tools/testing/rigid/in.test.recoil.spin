# TEST: the recoil correction includes the body's ROTATIONAL response.
# The impulse a collision delivers is reduced by the body's inverse-mass
# matrix at the contact, K = 1/M - [r x] Iinv [r x].  The translational
# part (1/M) is covered by in.test.recoil; this deck exercises the
# rotational part (r x n)^2/Izz, with the geometry and inertia chosen so
# that term equals 1/M exactly -- half the body's response.
#
# One particle strikes the left face of a free unit square 0.4 above the
# COM.  Everything is deterministic, so exact rigid-body impulse theory
# applies and run_tests.py checks vcom AND omega against it, plus exact
# momentum conservation and elastic energy conservation.

seed                12345
dimension           2
global              gridcut 0.0 comm/sort yes
boundary            r r p

create_box          0 10 0 10 -0.5 0.5
create_grid         20 20 1
balance_grid        rcb cell

global              nrho 1.0 fnum 1.0e-3

species             air.species N
mixture             air N vstream 1000.0 0 0 temp 1.0e-6

read_surf           data.square group body
surf_collide        1 specular
surf_modify         all collide 1
collide             none

timestep            1.0e-6

global              rigid yes

# Izz = 1.6e-25 makes (r x n)^2/Izz = 0.4^2/1.6e-25 = 1e24 = 1/M exactly
fix                 1 rigid body single body mass 1.0e-24 com 5 5 0 &
                    vcom 0 0 0 moi 1.0e-25 1.0e-25 1.6e-25 0 0 0 &
                    angmom 0 0 0 rotate richardson

create_particles    air single N 2.0 5.4 0.0 1000.0 0.0 0.0

stats               4000
stats_style         step np f_1[4] f_1[5] f_1[15] f_1
stats_modify        format float %.17g
run                 4000
