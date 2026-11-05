# K-1: collide_vss_kokkos vibdof==2 DISCRETE: N2 vibmode level must follow evib (0 mismatches)
seed 54345
dimension 3
timestep 5e-9
global nrho 1.E22
global fnum 2500 comm/sort yes
boundary r r r
create_box 0 1e-5 0 1e-5 0 1e-5
create_grid 2 2 2
species co2.species N2 CO2 vibfile co2.species.vib
mixture mix CO2 frac 0.9
mixture mix N2 frac 0.1
mixture mix group all
mixture mix vstream 0.0 0.0 0.0 temp 20000 trot 20000.00 tvib 10.0
collide vss all co2.vss
collide_modify vibrate discrete rotate smooth
fix 1 vibmode
create_particles mix n 0 twopass
stats 50
stats_style step np ncoll
dump 1 particle all 100 d.nr.* id type evib p_vibmode[1]
run 100
