seed                12345
dimension           2
global              gridcut 0.0 comm/sort yes
boundary            o r p
create_box          0 150 0 150 -0.5 0.5
create_grid         150 150 1
balance_grid        rcb cell
global              nrho 1.0 fnum 0.01
species             air.species N O
mixture             air N O vstream 100.0 0 0
region              inner block 25.5 124.5 25.5 124.5 INF INF
group               inner grid region inner one
compute             COMP isurf/grid all all n
fix                 FIX ave/grid all 10 10 100 c_COMP[*]
fix                 ablate ablate inner 100 0.0 f_FIX
global              surfs implicit
read_isurf          inner 100 100 1 binary.101x101 180.5 ablate
