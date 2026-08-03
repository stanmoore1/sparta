SPARTA (24 Sep 2025)
Running on 1 MPI task(s)
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
Created 1000 child grid cells
  CPU time = 0.00120552 secs
  create/ghost percent = 78.3808 21.6192

balance_grid        rcb part
Balance grid migrated 0 cells
  CPU time = 0.000326958 secs
  reassign/sort/migrate/ghost percent = 59.2299 1.02001 7.81568 31.9344

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
  CPU time = 0.00656371 secs

stats		    50
compute             temp temp
stats_style	    step cpu np nattempt ncoll c_temp

timestep 	    1.0E-9
run 		    500
WARNING: Single-group ambipolar collisions are inefficient; grouping electrons separately (e.g. collide ... species) is recommended (../collide.cpp:344)
Larsen-Borgnakke detailed balance corrected for 2 tabulated species pairs
  mean acceptance over the tabulated range >= 0.309
Memory usage per proc in Mbytes:
  particles (ave,min,max) = 4 4 4
  grid      (ave,min,max) = 1.51379 1.51379 1.51379
  surf      (ave,min,max) = 0 0 0
  total     (ave,min,max) = 5.51379 5.51379 5.51379
Step CPU Np Natt Ncoll c_temp 
       0            0    20000        0        0    10003.386 
      50   0.26463937    20000    61868     3678    9989.9917 
     100   0.54697161    20000    62273     3724    9978.2314 
     150   0.82777698    20000    62238     3689     10009.88 
     200    1.1129169    20000    62083     3660    10058.437 
     250    1.3948773    20000    62233     3688    10063.333 
     300    1.6718559    20000    62270     3778    10030.443 
     350    1.9192527    20000    62207     3757    10016.551 
     400    2.2008155    20000    62175     3695    10012.886 
     450     2.471744    20000    62391     3797    9997.2357 
     500    2.7293644    20000    62153     3670    9977.4256 
Loop time of 2.72943 on 1 procs for 500 steps with 20000 particles
Performance: 183.188 timesteps/s, 3.664 Mparticle-step/s

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Move    | 0.23255    | 0.23255    | 0.23255    |   0.0 |  8.52
Coll    | 2.443      | 2.443      | 2.443      |   0.0 | 89.51
Sort    | 0.045883   | 0.045883   | 0.045883   |   0.0 |  1.68
Comm    | 0.0024653  | 0.0024653  | 0.0024653  |   0.0 |  0.09
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0.0023684  | 0.0023684  | 0.0023684  |   0.0 |  0.09
Other   |            | 0.003148   |            |       |  0.12

Particle moves    = 10000000 (10M)
Cells touched     = 13712577 (13.7M)
Particle comms    = 0 (0K)
Boundary collides = 412536 (0.413M)
Boundary exits    = 0 (0K)
SurfColl checks   = 0 (0K)
SurfColl occurs   = 0 (0K)
Surf reactions    = 0 (0K)
Collide attempts  = 30913761 (30.9M)
Collide occurs    = 1847511 (1.85M)
Reactions         = 0 (0K)
Particles stuck   = 0
Axisymm bad moves = 0

Particle-moves/CPUsec/proc: 3.66377e+06
Particle-moves/step: 20000
Cell-touches/particle/step: 1.37126
Particle comm iterations/step: 1
Particle fraction communicated: 0
Particle fraction colliding with boundary: 0.0412536
Particle fraction exiting boundary: 0
Surface-checks/particle/step: 0
Surface-collisions/particle/step: 0
Surf-reactions/particle/step: 0
Collision-attempts/particle/step: 3.09138
Collisions/particle/step: 0.184751
Reactions/particle/step: 0

Particles: 20000 ave 20000 max 20000 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Cells:      1000 ave 1000 max 1000 min
Histogram: 1 0 0 0 0 0 0 0 0 0
GhostCell: 0 ave 0 max 0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
EmptyCell: 0 ave 0 max 0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
