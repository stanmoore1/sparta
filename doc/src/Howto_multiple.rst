.. _howto_3:

Running multiple simulations from one input script
==================================================

This can be done in several ways.  See the documentation for
individual commands for more details on how these examples work.

If "multiple simulations" means continue a previous simulation for
more timesteps, then you simply use the :doc:`run <run>` command
multiple times.  For example, this script


.. parsed-literal::

   read_grid data.grid
   create_particles 1000000
   run 10000
   run 10000
   run 10000
   run 10000
   run 10000

would run 5 successive simulations of the same system for a total of
50,000 timesteps.

If you wish to run totally different simulations, one after the other,
the :doc:`clear <clear>` command can be used in between them to
re-initialize SPARTA.  For example, this script


.. parsed-literal::

   read_grid data.grid
   create_particles 1000000
   run 10000
   clear
   read_grid data.grid2
   create_particles 500000
   run 10000

would run 2 independent simulations, one after the other.

For large numbers of independent simulations, you can use
:doc:`variables <variable>` and the :doc:`next <next>` and
:doc:`jump <jump>` commands to loop over the same input script multiple
times with different settings.  For example, this script, named
in.flow


.. parsed-literal::

   variable d index run1 run2 run3 run4 run5 run6 run7 run8
   shell cd $d
   read_grid data.grid
   create_particles 1000000
   run 10000
   shell cd ..
   clear
   next d
   jump in.flow

would run 8 simulations in different directories, using a data.grid
file in each directory.  The same concept could be used to run the
same system at 8 different gas densities, using a density variable and
storing the output in different log and dump files, for example


.. parsed-literal::

   variable a loop 8
   variable rho index 1.0e18 4.0e18 1.0e19 4.0e19 1.0e20 4.0e20 1.0e21 4.0e21
   log log.$a
   read data.grid
   global nrho ${rho}
   ...
   compute myGrid grid all all n temp
   dump 1 grid all 1000 dump.$a id c_myGrid
   run 100000
   clear
   next rho
   next a
   jump in.flow

All of the above examples work whether you are running on 1 or
multiple processors, but assumed you are running SPARTA on a single
partition of processors.  SPARTA can be run on multiple partitions via
the "-partition" command-line switch as described in :ref:`Section 2.5 <start_7>` of the manual.

In the last 2 examples, if SPARTA were run on 3 partitions, the same
scripts could be used if the "index" and "loop" variables were
replaced with *universe*\ -style variables, as described in the
:doc:`variable <variable>` command.  Also, the "next rho" and "next a"
commands would need to be replaced with a single "next a rho" command.
With these modifications, the 8 simulations of each script would run
on the 3 partitions one after the other until all were finished.
Initially, 3 simulations would be started simultaneously, one on each
partition.  When one finished, that partition would then start the 4th
simulation, and so forth, until all 8 were completed.
