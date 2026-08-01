.. _cmd_1:

SPARTA input script
===================

SPARTA executes by reading commands from a input script (text file),
one line at a time.  When the input script ends, SPARTA exits.  Each
command causes SPARTA to take some action.  It may set an internal
variable, read in a file, or run a simulation.  Most commands have
default settings, which means you only need to use the command if you
wish to change the default.

In many cases, the ordering of commands in an input script is not
important.  However the following rules apply:

(1) SPARTA does not read your entire input script and then perform a
simulation with all the settings.  Rather, the input script is read
one line at a time and each command takes effect when it is read.
Thus this sequence of commands:


.. code-block:: SPARTA

   timestep 0.5 
   run      100 
   run      100

does something different than this sequence:


.. code-block:: SPARTA

   run      100 
   timestep 0.5 
   run      100

In the first case, the specified timestep (0.5 secs) is used for two
simulations of 100 timesteps each.  In the 2nd case, the default
timestep (1.0 sec is used for the 1st 100 step simulation and a 0.5
fmsec timestep is used for the 2nd one.

(2) Some commands are only valid when they follow other commands.  For
example you cannot define the grid overlaying the simulation box until
the box itself has been defined.  Likewise you cannot read in
triangulated surfaces until a grid has been defined to store them.

Many input script errors are detected by SPARTA and an ERROR or
WARNING message is printed.  :doc:`Section 12 <Section_errors>` gives
more information on what errors mean.  The documentation for each
command lists restrictions on how the command can be used.
