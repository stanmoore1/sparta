# Q-6: multivalue ablation must not abort (variants of examples/ablation/in.ablation.multi.inner.3d)
# -var SRC "uniform 3" | "random 5" ; -var TH 100.5 | 150.5
seed 9763
dimension 2
global gridcut 0.04
global comm/sort yes
boundary p p p
create_box 0.0 22.0 0.0 10.0 -0.5 0.5
create_grid 44 20 1
balance_grid rcb cell
global nrho 1.0 fnum 0.001
global surfs explicit/distributed
species air.species O2
mixture air O2 vstream 0.0 0 0
read_surf data.etch2d
surf_collide 1 diffuse 300.0 0.0
surf_modify all collide 1
fix fablate ablate all 1 1 ${SRC} multiple yes
create_isurf all fablate ${TH} multi
timestep 0.001
compute 1 property/grid all vol
compute 1r reduce sum c_1
stats 10
stats_style step c_1r
run ${N}
