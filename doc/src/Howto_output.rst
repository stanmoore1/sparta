.. _howto_4:

Output from SPARTA (stats, dumps, computes, fixes, variables)
=============================================================

There are four basic kinds of SPARTA output:

* :doc:`Statistical output <stats_style>`, which is a list of quantities
  printed every few timesteps to the screen and logfile.
* :doc:`Dump files <dump>`, which contain snapshots of particle, grid
  cell, or surface element quantities and are written at a specified
  frequency.
* Certain fixes can output user-specified quantities directly to files:
  :doc:`fix ave/time <fix_ave_time>` for time averaging, and :doc:`fix print <fix_print>` for single-line output of
  :doc:`variables <variable>`.  Fix print can also output to the
  screen.
* :doc:`Restart files <restart>`.

A simulation prints one set of statistical output and (optionally)
restart files.  It can generate any number of dump files and fix
output files, depending on what :doc:`dump <dump>` and :doc:`fix <fix>`
commands you specify.

As discussed below, SPARTA gives you a variety of ways to determine
what quantities are computed and printed when the statistics, dump, or
fix commands listed above perform output.  Throughout this discussion,
note that users can also add their own computes and fixes to SPARTA
(see :doc:`Section 10 <Section_modify>`) which can generate values that
can then be output with these commands.

The following sub-sections discuss different SPARTA commands related
to output and the kind of data they operate on and produce:

* :ref:`Global/per-particle/per-grid/per-surf/per-tally data <global>`
* :ref:`Scalar/vector/array data <scalar>`
* :ref:`Statistical output <stats>`
* :ref:`Dump file output <dump>`
* :ref:`Fixes that write output files <fixoutput>`
* :ref:`Computes that process output quantities <computeoutput>`
* :ref:`Computes that generate values to output <compute>`
* :ref:`Fixes that generate values to output <fix>`
* :ref:`Variables that generate values to output <variable>`
* :ref:`Summary table of output options and data flow between commands <table>`

.. _global:

Global/per-particle/per-grid/per-surf/pre-tally data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Various output-related commands work with four different styles of
data: global, per particle, per grid, or per surf.  A global datum is
one or more system-wide values, e.g. the temperature of the system.  A
per particle datum is one or more values per particle, e.g. the kinetic
energy of each particle.  A per grid datum is one or more values per
grid cell, e.g. the temperature of the particles in the grid cell.  A
per surf datum is one or more values per surface element, e.g. the
count of particles that collided with the surface element.  A
per-tally datum is one or more values per event, e.g. a particle
colliding or reacting with a surface element.

.. _scalar:

Scalar/vector/array data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Global, per particle, per grid, per surf, and per tally datums can
each come in two kinds: a single scalar value, a vector of values.
Additionally, global quantities can also be a 2d array of values.  The
doc page for a "compute" or "fix" or "variable" that generates data
will specify both the style and kind of data it produces, e.g. a
per-particle vector.  Some computes can produce more than one form of
a single style, e.g. a global scalar and a global vector.

When a quantity is accessed, as in many of the output commands
discussed below, it can be referenced via the following bracket
notation, where ID in this case is the ID of a compute.  The leading
"c\_" would be replaced by "f\_" for a fix, or "v\_" for a variable:

+-------------+--------------------------------------------+
| c\_ID       | entire scalar, vector, or array            |
+-------------+--------------------------------------------+
| c\_ID[I]    | one element of vector, one column of array |
+-------------+--------------------------------------------+
| c\_ID[I][J] | one element of array                       |
+-------------+--------------------------------------------+

In other words, using one bracket reduces the dimension of the data
once (vector -> scalar, array -> vector).  Using two brackets reduces
the dimension twice (array -> scalar).  Thus a command that uses
scalar values as input can typically also process elements of a vector
or array.

.. _stats:

Statistical output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The frequency and format of statistical output is set by the
:doc:`stats <stats>`, :doc:`stats\_style <stats_style>`, and
:doc:`stats\_modify <stats_modify>` commands.  The
:doc:`stats\_style <stats_style>` command also specifies what values are
calculated and written out.  Pre-defined keywords can be specified
(e.g. np, ncoll, etc).  Three additional kinds of keywords can also be
specified (c\_ID, f\_ID, v\_name), where a :doc:`compute <compute>` or
:doc:`fix <fix>` or :doc:`variable <variable>` provides the value to be
output.  In each case, the compute, fix, or variable must generate
global values to be used as an argument of the
:doc:`stats\_style <stats_style>` command.

.. _dump:

Dump file output
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dump file output is specified by the :doc:`dump <dump>` and
:doc:`dump\_modify <dump_modify>` commands.  There are several
pre-defined formats: dump particle, dump grid, dump surf, dump tally,
etc.

Each of these allows specification of what values are output with each
particle, grid cell, or surface element.  Pre-defined attributes can
be specified (e.g. id, x, y, z for particles or id, vol for grid
cells, etc).  Three additional kinds of keywords can also be specified
(c\_ID, f\_ID, v\_name), where a :doc:`compute <compute>` or :doc:`fix <fix>`
or :doc:`variable <variable>` provides the values to be output.  In each
case, the compute, fix, or variable must generate per particle, per
grid, or per surf values for input to the corresponding
:doc:`dump <dump>` command.

.. _fixoutput:

Fixes that write output files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two fixes take various quantities as input and can write output files:
:doc:`fix ave/time <fix_ave_time>` and :doc:`fix print <fix_print>`.

The :doc:`fix ave/time <fix_ave_time>` command enables direct output to
a file and/or time-averaging of global scalars or vectors.  The user
specifies one or more quantities as input.  These can be global
:doc:`compute <compute>` values, global :doc:`fix <fix>` values, or
:doc:`variables <variable>` of any style except the particle style which
does not produce single values.  Since a variable can refer to
keywords used by the :doc:`stats\_style <stats_style>` command (like
particle count), a wide variety of quantities can be time averaged
and/or output in this way.  If the inputs are one or more scalar
values, then the fix generates a global scalar or vector of output.
If the inputs are one or more vector values, then the fix generates a
global vector or array of output.  The time-averaged output of this
fix can also be used as input to other output commands.

The :doc:`fix print <fix_print>` command can generate a line of output
written to the screen and log file or to a separate file, periodically
during a running simulation.  The line can contain one or more
:doc:`variable <variable>` values for any style variable except the
particle style.  As explained above, variables themselves can contain
references to global values generated by :doc:`stats keywords <stats_style>`, :doc:`computes <compute>`, :doc:`fixes <fix>`,
or other :doc:`variables <variable>`.  Thus the :doc:`fix print <fix_print>` command is a means to output a wide variety of
quantities separate from normal statistical or dump file output.

.. _computeoutput:

Computes that process output quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :doc:`compute reduce <compute_reduce>` command takes one or more per
particle or per grid or per surf vector quantities as inputs and
"reduces" them (sum, min, max, ave) to scalar quantities.  These are
produced as output values which can be used as input to other output
commands.

.. _compute:

Computes that generate values to output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every :doc:`compute <compute>` in SPARTA produces either global or per
particle or per grid or per surf values.  The values can be scalars or
vectors or arrays of data.  These values can be output using the other
commands described in this section.  The doc page for each compute
command describes what it produces.  Computes that produce per
particle or per grid or per surf values have the word "particle" or
"grid" or "surf" in their style name.  Computes without those words
produce global values.

.. _fix:

Fixes that generate values to output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some :doc:`fixes <fix>` in SPARTA produces either global or per particle
or per grid or per surf values which can be accessed by other
commands.  The values can be scalars or vectors or arrays of data.
These values can be output using the other commands described in this
section.  The doc page for each fix command tells whether it produces
any output quantities and describes them.

Two fixes of particular interest for output are the :doc:`fix ave/grid <fix_ave_grid>` and :doc:`fix ave/surf <fix_ave_surf>`
commands.

The :doc:`fix ave/grid <fix_ave_grid>` command enables time-averaging of
per grid vectors.  The user specifies one or more quantities as input.
These can be per grid vectors or arrays from :doc:`compute <compute>` or
:doc:`fix <fix>` commands.  If the input is a single vector, then the
fix generates a per grid vector.  If the input is multiple vectors or
array, the fix generates a per grid array.  The time-averaged output
of this fix can also be used as input to other output commands.

The :doc:`fix ave/surf <fix_ave_surf>` command enables time-averaging of
per surf vectors.  The user specifies one or more quantities as input.
These can be per surf vectors or arrays from :doc:`compute <compute>` or
:doc:`fix <fix>` commands.  If the input is a single vector, then the
fix generates a per surf vector.  If the input is multiple vectors or
array, the fix generates a per surf array.  The time-averaged output
of this fix can also be used as input to other output commands.

.. _variable:

Variables that generate values to output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:doc:`Variables <variable>` defined in an input script generate either a
global scalar value or a per particle vector (only particle-style
variables) when it is accessed.  The formulas used to define equal-
and particle-style variables can contain references to the
:doc:`stats\_style <stats_style>` keywords and to global and per particle
data generated by computes, fixes, and other variables.  The values
generated by variables can be output using the other commands
described in this section.

.. _table:

Summary table of output options and data flow between commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This table summarizes the various commands that can be used for
generating output from SPARTA.  Each command produces output data of
some kind and/or writes data to a file.  Most of the commands can take
data from other commands as input.  Thus you can link many of these
commands together in pipeline form, where data produced by one command
is used as input to another command and eventually written to the
screen or to a file.  Note that to hook two commands together the
output and input data types must match, e.g. global/per atom/local
data and scalar/vector/array data.

Also note that, as described above, when a command takes a scalar as
input, that could be an element of a vector or array.  Likewise a
vector input could be a column of an array.

+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| Command                                | Input                                | Output                                                     |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| :doc:`stats\_style <stats_style>`      | global scalars                       | screen, log file                                           |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| :doc:`dump particle <dump>`            | per particle vectors                 | dump file                                                  |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| :doc:`dump grid <dump>`                | per grid vectors                     | dump file                                                  |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| :doc:`dump surf <dump>`                | per surf vectors                     | dump file                                                  |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| :doc:`dump tally <dump>`               | per tally vectors                    | dump file                                                  |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| :doc:`fix print <fix_print>`           | global scalar from variable          | screen, file                                               |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| :doc:`print <print>`                   | global scalar from variable          | screen                                                     |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| :doc:`computes <compute>`              | N/A                                  | global or per particle/grid/surf/tally scalar/vector/array |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| :doc:`fixes <fix>`                     | N/A                                  | global or per particle/grid/surf scalar/vector/array       |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| :doc:`variables <variable>`            | global scalars, per particle vectors | global scalar, per particle vector                         |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| :doc:`compute reduce <compute_reduce>` | per particle/grid/surf vectors       | global scalar/vector                                       |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| :doc:`fix ave/time <fix_ave_time>`     | global scalars/vectors               | global scalar/vector/array, file                           |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| :doc:`fix ave/grid <fix_ave_grid>`     | per grid vectors/arrays              | per grid vector/array                                      |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
| :doc:`fix ave/surf <fix_ave_surf>`     | per surf vectors/arrays              | per surf vector/array                                      |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
|                                        |                                      |                                                            |  |
+----------------------------------------+--------------------------------------+------------------------------------------------------------+--+
