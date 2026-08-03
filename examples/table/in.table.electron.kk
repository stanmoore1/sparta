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
# the attempts of in.table.electron and runs about 3.3x slower.  SPARTA warns
# about this at setup.
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
create_grid 	    10 10 10

balance_grid        rcb part

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
collide_modify      ambipolar yes

create_particles    plasma n 20000 twopass

stats		    50
compute             temp temp
stats_style	    step cpu np nattempt ncoll c_temp

timestep 	    1.0E-9
run 		    500
