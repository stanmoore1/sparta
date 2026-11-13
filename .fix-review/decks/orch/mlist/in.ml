seed                12345
dimension           3
global              gridcut ${gc} comm/sort yes
boundary            so pp pp
create_box          0 0.01 0 0.01 0 0.01
create_grid         ${nc} 4 4
balance_grid        rcb cell
species             air.species N2 N
mixture             air N2 N vstream -1000.0 0.0 0.0 temp 1.0
mixture             air N2 frac 1.0
mixture             air N frac 0.0
global              nrho 1.0e20 fnum ${fnum}
surf_collide        1 specular
surf_react          1 prob d1.surf
bound_modify        xlo collide 1 react 1
create_particles    air n ${np} twopass
stats               1
stats_style         step np nsreact nexit
timestep            1.0e-4
run                 2
