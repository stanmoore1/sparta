# 19-10/19-11: react qk/kk and tce/qk/kk with atom-first reactant order "N + N2" (-var R qk|tce/qk)
seed ${S}
dimension 3
global gridcut 1.0e-5 comm/sort yes
boundary rr rr rr
create_box 0 0.0001 0 0.0001 0 0.0001
create_grid 10 10 10
balance_grid rcb part
species air.species N2 N
mixture air N2 N vstream 0.0 0.0 0.0 temp 20000.0
mixture air N2 frac 0.5
mixture air N frac 0.5
global nrho 7.07043E22
global fnum 7.07043E5
collide vss air air.vss
react ${R} nn2.tce
create_particles air n 10000 twopass
stats 100
stats_style step np nreact
timestep 7.00E-9
run 400
