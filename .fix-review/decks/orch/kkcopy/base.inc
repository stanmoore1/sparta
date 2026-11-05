seed                12345
dimension           2
global              gridcut 0.0 comm/sort yes
boundary            o r p
create_box          0 10 0 10 -0.5 0.5
create_grid         10 10 1
balance_grid        rcb cell
global              nrho 1.0 fnum 0.001
species             air.species N O
mixture             air N O vstream 100.0 0 0
read_surf           data.circle
surf_collide        1 diffuse 300.0 0.0
surf_modify         all collide 1
collide             vss air air.vss
fix                 in emit/face air xlo
timestep            0.0001
