# 23-7: surf_collide piston(/kk) + adsorb gs face with ER product (velreset): products must not be piston-reflected
# -var F er.surf|aa.surf -var WALL xlo|xhi|ylo
seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary ${BX} ${BY} p
create_box 0 10 0 10 -0.5 0.5
create_grid 20 20 1
balance_grid rcb cell
global nrho 1.0 fnum 0.001
species air.species O CO CO2 C
mixture air O vstream 0.0 0 0 temp 300.0
mixture air O frac 1.0
mixture air CO frac 0.0
mixture air CO2 frac 0.0
mixture air C frac 0.0
surf_collide 1 piston 10.0
surf_react ads adsorb gs ${F} nsync 1 face 1000 6.022e9 O CO
bound_modify ${WALL} collide 1 react ads
create_particles air n 10000 twopass
compute c count O CO
compute k reduce sum ke
timestep 0.0001
stats 100
stats_style step np c_c[*] c_k
run 300
