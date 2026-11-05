# 31-3: Surf::init "N surface elements not assigned to a collision model" counts each surf once (true 50)
seed 12345
dimension 2
global gridcut 0.0 surfs explicit/distributed
boundary o r p
create_box 0 10 0 10 -0.5 0.5
create_grid 10 10 1
balance_grid rcb cell
global nrho 1.0 fnum 0.001
species air.species N O
mixture air N O vstream 10.0 0 0 temp 10.0
read_surf data.circle
run 0
