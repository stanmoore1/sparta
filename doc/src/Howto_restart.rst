.. _howto_10:

Restarting a simulation
=======================

There are two ways to continue a long SPARTA simulation.  Multiple
:doc:`run <run>` commands can be used in the same input script.  Each
run will continue from where the previous run left off.  Or binary
restart files can be saved to disk using the :doc:`restart <restart>`
command.  At a later time, these binary files can be read via a
:doc:`read\_restart <read_restart>` command in a new script.

Here is an example of a script that reads a binary restart file and
then issues a new run command to continue where the previous run left
off.  It illustrates what settings must be made in the new script.
Details are discussed in the documentation for the
:doc:`read\_restart <read_restart>` and
:doc:`write\_restart <write_restart>` commands.

Look at the *in.collide* input script provided in the *bench*
directory of the SPARTA distribution to see the original script that
this script is based on.  If that script had the line


.. parsed-literal::

   restart              50 tmp.restart

added to it, it would produce 2 binary restart files (tmp.restart.50
and tmp.restart.100) as it ran for 130 steps, one at step 50, and one
at step 100.

This script could be used to read the first restart file and re-run
the last 80 timesteps:


.. parsed-literal::

   read_restart     tmp.restart.50

   seed             12345
   collide                  vss air ar.vss

   stats                    10
   compute             temp temp
   stats_style      step cpu np nattempt ncoll c_temp

   timestep         7.00E-9
   run              80

Note that the following commands do not need to be repeated because
their settings are included in the restart file: *dimension, global,
boundary, create\_box, create\_grid, species, mixture*.  However these
commands do need to be used, since their settings are not in the
restart file: *seed, collide, compute, fix, stats\_style, timestep*.
The :doc:`read\_restart <read_restart>` doc page gives details.

If you actually use this script to perform a restarted run, you will
notice that the statistics output does not match exactly.  On step 50,
the collision counts are 0 in the restarted run, because the line is
printed before the restarted simulation begins.  The collision counts
in subsequent steps are similar but not identical.  This is because
new random numbers are used for collisions in the restarted run.  This
affects all the randomized operations in a simulation, so in general
you should only expect a restarted run to be statistically similar to
the original run.
