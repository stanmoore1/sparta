SPARTA (24 Sep 2025)
Running on 4 MPI task(s)
################################################################################
# electron-neutral collisions with a tabulated cross section, in a form the
# KOKKOS package can run
#
# This is in.table.electron with one change: every species is in a single
# collision group.  The KOKKOS collide styles do not yet support more than one
# group, for any collide style, so the grouped form of this problem cannot run
# with the kk suffix.  This one runs with it and reproduces the non-KOKKOS
# build bit for bit, at 1 and 4 procs.
#
# Two costs come with that change.
#
# It is slower.  SPARTA draws its NTC collision attempt count from the largest
# sigma*g in a group, and an ambipolar electron is about 200x faster than a
# nitrogen molecule at the same temperature.  Putting the electrons in with the
# heavy species inflates the attempt count for every pair in the cell, and
# nearly all of those extra attempts are rejected: this input makes about 5.6x
# the attempts of in.table.electron for the same number of collisions, and runs
# about 3.3x slower.  SPARTA warns about this at setup.
#
# It is also not quite the same calculation.  SPARTA reaches ambipolar
# electrons through different code for one group than for several, and the two
# do not select the same electron collisions: this input records about 14% more
# collisions than in.table.electron, consistently across seeds.  That gap is
# not an artifact of the tabulated cross section -- replacing "collide table"
# with "collide vss" here reproduces it -- so it is a property of the ambipolar
# model's grouping, and it is why the grouped input remains the reference.
#
# Prefer in.table.electron whenever the KOKKOS package is not being used.
#
# Electrons are handled with the ambipolar approximation ("fix ambipolar"):
# each electron stays with its parent ion and the pair is advected together,
# so no self-consistent electric field is solved and a normal timestep can be
# used.  At collision time SPARTA gives each ambipolar electron its own
# velocity, so a tabulated sigma(E) is evaluated at the true electron energy.
#
# This is valid for a quasi-neutral, weakly ionised gas on scales much larger
# than the Debye length.  It cannot represent sheaths, charge separation, or
# an electron energy distribution driven by an applied field, since the
# electron's trajectory is its ion's.  Those need a PIC code.
#
# NOTE ON THE CROSS SECTION: en_demo.tab is a SYNTHETIC model with a
# Ramsauer-type minimum and a resonance peak.  It is not data for nitrogen.
# It is here so this example runs and so the shape exercises structure that
# no VHS power law can reproduce.  See in.table.electron for how to replace it
# with data from www.lxcat.net.
#
# Note:
#  - The "comm/sort” option to the “global” command is used to match MPI runs.
#  - The “twopass” option is used to match Kokkos runs.
# The "comm/sort" and "twopass" options should not be used for production runs.
################################################################################

seed	    	    12345
dimension   	    3
global              gridcut 1.0e-5 comm/sort yes

boundary	    rr rr rr

create_box  	    0 0.0001 0 0.0001 0 0.0001
Created orthogonal box = (0 0 0) to (0.0001 0.0001 0.0001)
create_grid 	    10 10 10
WARNING: Could not acquire nearby ghost cells b/c grid partition is not clumped (../grid.cpp:473)
Created 1000 child grid cells
  CPU time = 0.0015718 secs
  create/ghost percent = 93.3009 6.69911

balance_grid        rcb part
Balance grid migrated 740 cells
  CPU time = 0.00161993 secs
  reassign/sort/migrate/ghost percent = 66.5552 0.281741 14.1182 19.0448

species		    en.species N2 N2+ e

# no "group" keyword, so every species lands in one group named "default".
# that is what lets this input run with the kk suffix

mixture		    plasma N2 N2+ e vstream 0.0 0.0 0.0 temp 10000.0
mixture             plasma N2 frac 0.90
mixture             plasma N2+ frac 0.10
mixture             plasma e frac 0.0

global              nrho 1.0E22
global              fnum 1.0E6

fix                 ambi ambipolar e N2+
collide		    table plasma en.tab.vss linear 128
Tabulated cross section EN_RAMSAUER_DEMO from en_demo.tab for e N2:
  300 values, E = 0.001 to 1000 eV, max 2.793e-19, log-log interpolation
  2553 bins, 128 per factor of 2 in energy, reproduces input to 0.0056%
collide_modify      ambipolar yes

create_particles    plasma n 20000 twopass
Created 20000 particles
  CPU time = 0.0043141 secs

stats		    50
compute             temp temp
stats_style	    step cpu np nattempt ncoll c_temp

timestep 	    1.0E-9
run 		    500
WARNING: Single-group ambipolar collisions are inefficient; grouping electrons separately (e.g. collide ... species) is recommended (../collide.cpp:344)
Larsen-Borgnakke detailed balance corrected for 2 tabulated species pairs
  mean acceptance over the tabulated range >= 0.309
Memory usage per proc in Mbytes:
  particles (ave,min,max) = 2 2 2
  grid      (ave,min,max) = 1.51379 1.51379 1.51379
  surf      (ave,min,max) = 0 0 0
  total     (ave,min,max) = 3.51379 3.51379 3.51379
Step CPU Np Natt Ncoll c_temp 
       0            0    20000        0        0    9928.2195 
      50  0.084951029    20000    62076     3726    9908.8965 
     100   0.16894671    20000    62320     3741    9911.8594 
     150   0.25312194    20000    62389     3820    9908.3429 
     200   0.34070608    20000    62318     4022    9951.3453 
     250   0.42376621    20000    62437     3738    9966.5489 
     300   0.49241011    20000    62582     3990    9973.2248 
     350   0.55655114    20000    62226     3646    9976.3148 
     400   0.62572983    20000    62224     3868     9956.583 
     450   0.69466759    20000    62453     3844    9918.6748 
     500   0.75762988    20000    62571     3769    9917.7519 
Loop time of 0.757745 on 4 procs for 500 steps with 20000 particles
Performance: 659.852 timesteps/s, 13.197 Mparticle-step/s

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Move    | 0.048801   | 0.053125   | 0.057359   |   1.8 |  7.01
Coll    | 0.51193    | 0.55906    | 0.60339    |   5.8 | 73.78
Sort    | 0.0049389  | 0.0062889  | 0.0073398  |   1.3 |  0.83
Comm    | 0.031391   | 0.032263   | 0.033032   |   0.4 |  4.26
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0.001429   | 0.0027245  | 0.0036561  |   1.6 |  0.36
Other   |            | 0.1043     |            |       | 13.76

Particle moves    = 10000000 (10M)
Cells touched     = 13700939 (13.7M)
Particle comms    = 272149 (0.272M)
Boundary collides = 411283 (0.411M)
Boundary exits    = 0 (0K)
SurfColl checks   = 0 (0K)
SurfColl occurs   = 0 (0K)
Surf reactions    = 0 (0K)
Collide attempts  = 31020877 (31M)
Collide occurs    = 1886088 (1.89M)
Reactions         = 0 (0K)
Particles stuck   = 0
Axisymm bad moves = 0

Particle-moves/CPUsec/proc: 3.29926e+06
Particle-moves/step: 20000
Cell-touches/particle/step: 1.37009
Particle comm iterations/step: 1
Particle fraction communicated: 0.0272149
Particle fraction colliding with boundary: 0.0411283
Particle fraction exiting boundary: 0
Surface-checks/particle/step: 0
Surface-collisions/particle/step: 0
Surf-reactions/particle/step: 0
Collision-attempts/particle/step: 3.10209
Collisions/particle/step: 0.188609
Reactions/particle/step: 0

Particles: 5000 ave 5072 max 4910 min
Histogram: 1 0 0 0 0 1 1 0 0 1
Cells:      250 ave 250 max 250 min
Histogram: 4 0 0 0 0 0 0 0 0 0
GhostCell: 172.5 ave 240 max 110 min
Histogram: 1 0 0 0 2 0 0 0 0 1
EmptyCell: 62.5 ave 130 max 0 min
Histogram: 1 0 0 0 2 0 0 0 0 1
