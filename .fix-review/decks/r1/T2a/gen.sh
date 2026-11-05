#!/bin/bash
# gen.sh: writes in.<mode>.<redef> decks (mode every/select/all, redef more/fewer/none/same)
for mode in every select all; do
  case $mode in select) m="select 1 3";; *) m=$mode;; esac; c="c_1[*]"; cs="c_s[*]"; [ $mode = all ] && c="c_1" && cs="c_s"
  for redef in more fewer none same; do
    case $redef in more) r="react tce air.full.tce";; fewer) r="react tce rB.tce";; none) r="react none";; same) r="react tce rA.tce";; esac
    cat > in.$mode.$redef <<EOD
seed 12345
dimension 3
global gridcut 1.0e-5 comm/sort yes
boundary rr rr rr
create_box 0 0.0001 0 0.0001 0 0.0001
create_grid 4 4 4
balance_grid rcb part
species air.species N2 N O2 O NO
mixture air N2 N O2 O NO vstream 0.0 0.0 0.0 temp 20000.0
global nrho 7.07043E22 fnum 7.07043E6
collide vss air air.vss
react tce rA.tce
create_particles air n 2000 twopass
compute 1 gas/reaction/grid all air $m
compute s reduce sum $c
stats 10
stats_style step np nreact $cs
timestep 7.00E-9
run 20
$r
run 20
EOD
  done
done
