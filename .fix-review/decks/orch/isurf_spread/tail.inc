surf_modify         all collide 1
collide             vss air air.vss
fix                 in emit/face air xlo twopass
timestep            0.0001
fix                 bal balance 50 1.001 rcb part
compute             s isurf/grid all all etot
compute             es reduce sum c_s[1]
stats               50
stats_style         step np nscoll nscheck c_es f_bal
run                 300
