.. _start_8:

SPARTA screen output
====================

As SPARTA reads an input script, it prints information to both the
screen and a log file about significant actions it takes to setup a
simulation.  When the simulation is ready to begin, SPARTA performs
various initializations and prints the amount of memory (in MBytes per
processor) that the simulation requires.  It also prints details of
the initial state of the system.  During the run itself, statistical
information is printed periodically, every few timesteps.  When the
run concludes, SPARTA prints the final state and a total run time for
the simulation.  It then appends statistics about the CPU time and
size of information stored for the simulation.  An example set of
statistics is shown here:

Loop time of 0.639973 on 4 procs for 1000 steps with 45792 particles


.. parsed-literal::

   MPI task timing breakdown:
   Section \|  min time  \|  avg time  \|  max time  \|%varavg\| %total
   ---------------------------------------------------------------
   Move    \| 0.10948    \| 0.26191    \| 0.42049    \|  27.6 \| 40.92
   Coll    \| 0.013711   \| 0.041659   \| 0.070985   \|  13.5 \|  6.51
   Sort    \| 0.01733    \| 0.040286   \| 0.063573   \|  10.6 \|  6.29
   Comm    \| 0.02276    \| 0.023555   \| 0.02493    \|   0.6 \|  3.68
   Modify  \| 0.00018167 \| 0.024758   \| 0.051345   \|  15.6 \|  3.87
   Output  \| 0.0002172  \| 0.0007354  \| 0.0012152  \|   0.0 \|  0.11
   Other   \|            \| 0.2471     \|            \|       \| 38.61

   Particle moves    = 38096354 (38.1M)
   Cells touched     = 43236871 (43.2M)
   Particle comms    = 146623 (0.147M)
   Boundary collides = 182782 (0.183M)
   Boundary exits    = 181792 (0.182M)
   SurfColl checks   = 7670863 (7.67M)
   SurfColl occurs   = 177740 (0.178M)
   Surf reactions    = 124169 (0.124M)
   Collide attempts  = 1232 (1K)
   Collide occurs    = 553 (0.553K)
   Gas reactions     = 23 (0.023K)
   Particles stuck   = 0

   Particle-moves/CPUsec/proc: 1.4882e+07
   Particle-moves/step: 38096.4
   Cell-touches/particle/step: 1.13493
   Particle comm iterations/step: 1.999
   Particle fraction communicated: 0.00384874
   Particle fraction colliding with boundary: 0.00479789
   Particle fraction exiting boundary: 0.0047719
   Surface-checks/particle/step: 0.201354
   Surface-collisions/particle/step: 0.00466554
   Surface-reactions/particle/step: 0.00325934
   Collision-attempts/particle/step: 1.232
   Collisions/particle/step: 0.553
   Gas-reactions/particle/step: 0.023

Gas reaction tallies:
  style tce #-of-reactions 45
  reaction O2 + N --> O + O + N: 10
  reaction O2 + O --> O + O + O: 5
  reaction N2 + O --> N + N + O: 8

Surface reaction tallies:
  id 1 style global #-of-reactions 2
    reaction all: 124025
    reaction delete: 53525
    reaction create: 70500


.. parsed-literal::

   Particles: 11448 ave 17655 max 5306 min
   Histogram: 2 0 0 0 0 0 0 0 0 2
   Cells:     100 ave 100 max 100 min
   Histogram: 4 0 0 0 0 0 0 0 0 0
   GhostCell: 21 ave 21 max 21 min
   Histogram: 4 0 0 0 0 0 0 0 0 0
   EmptyCell: 21 ave 21 max 21 min
   Histogram: 4 0 0 0 0 0 0 0 0 0
   Surfs:     50 ave 50 max 50 min
   Histogram: 4 0 0 0 0 0 0 0 0 0
   GhostSurf: 0 ave 0 max 0 min
   Histogram: 4 0 0 0 0 0 0 0 0 0

The first line gives the total CPU run time for the simulation, in
seconds.

The next section gives a breakdown of the CPU timing (in seconds) in
7 categories.  The first four are timings for particles moves, which
includes interaction with surface elements, then particle collisions,
then sorting of particles (required to perform collisions), and
communication of particles between processors.  The Modify section is
time for operations invoked by fixes and computes.  The Output section
is for dump command and statistical output.  The Other category is
typically for load-imbalance, as some MPI tasks wait for others MPI
tasks to complete.  In each category the min,ave,max time across
processors is shown, as well as a variation, and the percentage of
total time.

The next section gives some statistics about the run.  These are total
counts of particle moves, grid cells touched by particles, the number
of particles communicated between processors, collisions of particles
with the global boundary and with surface elements (none in this
problem), as well as collision and reaction statistics.

The next section gives additional statistics, normalized by timestep
or processor count.

The next 2 sections are optional.  The "Gas reaction tallies" section
is only output if the :doc:`react <react>` command is used.  For each
reaction with a non-zero tally, the number of those reactions that
occurred during the run is printed.  The "Surface reaction tallies"
section is only output if the :doc:`surf\_react <surf_react>` command was
used one or more times, to assign reaction models to individual
surface elements or the box boundaries.  For each of the commands, and
each of its reactions with a non-zero tally, the number of those
reactions that occurred during the run is printed.  Note that this is
effectively a summation over all the surface elements and/or box
boundaries the :doc:`surf\_react <surf_react>` command was used to assign
a reaction model to.

The last section is a histogramming across processors of various
per-processor statistics: particle count, owned grid cells, processor,
ghost grid cells which are copies of cells owned by other processors,
and empty cells which are ghost cells without surface information
(only used to pass particles to neighboring processors).

The ave value is the average across all processors.  The max and min
values are for any processor.  The 10-bin histogram shows the
distribution of the value across processors.  The total number of
histogram counts is equal to the number of processors.


.. _sws: https://sparta.github.io
.. _sd: Manual.html
.. _sc: Section_commands.html
