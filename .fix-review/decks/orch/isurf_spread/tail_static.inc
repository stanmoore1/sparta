surf_modify         all collide 1
collide             vss air air.vss
fix                 in emit/face air xlo twopass
timestep            0.0001

compute             s isurf/grid all all etot
compute             es reduce sum c_s[1]
stats               50
stats_style         step np nscoll nscheck c_es
run                 300
