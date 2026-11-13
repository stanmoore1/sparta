# 31-1/S-1: fix emit/surf insertion must follow fix dt/reset (resetflag 1)
seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary o r p
create_box 0 10 0 10 -0.5 0.5
create_grid 10 10 1
balance_grid rcb cell
global nrho 1.0 fnum 0.001
species air.species N
mixture air N vstream 10.0 0 0 temp 10.0
read_surf data.circle
surf_collide 1 diffuse 300.0 0.0
surf_modify all collide 1
collide vss air air.vss
fix in emit/surf air all perspecies ${PS} twopass
timestep 0.0001
compute 1 grid all air nrho temp usq vsq wsq
fix 1 ave/grid all 1 10 10 c_1[*]
compute mct lambda/grid f_1[1] f_1[2] tau
compute tstep dt/grid all 0.25 0.1 c_mct f_1[2] f_1[3] f_1[4] f_1[5]
fix DT dt/reset 20 c_tstep 0.1 ${RESET}
stats 10
stats_style step np dt
run 100
