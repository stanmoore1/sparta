.. _err_1:

Common problems
===============

If two SPARTA runs do not produce the same answer on different
machines or different numbers of processors, this is typically not a
bug.  On different machines, there can be numerical round-off in the
computations which causes slight differences in particle trajectories
or the number of particles, which will lead to numerical divergence of
the particle trajectories and averaged statistical quantities within a
few 100s or few 1000s of timesteps.  When running on different numbers
of processors, random numbers are used in different ways, so two
simulations can be immediately different.  However, the statistical
properties (e.g. overall particle temperature or per grid cell
temperature or surface energy flux) for the two runs on different
machines or on different numbers of processors should still be
similar.

A SPARTA simulation typically has two stages, setup and run.  Most
SPARTA errors are detected at setup time; others like running out of
memory may not occur until the middle of a run.

SPARTA tries to flag errors and print informative error messages so
you can fix the problem.  Of course, SPARTA cannot figure out physics
or numerical mistakes, like choosing too big a timestep or specifying
erroneous collision parameters.  If you run into errors that SPARTA
doesn't catch that you think it should flag, please send an email to
the `developers <https://sparta.github.io/authors.html>`_.

If you get an error message about an invalid command in your input
script, you can determine what command is causing the problem by
looking in the log.sparta file, or using the :doc:`echo command <echo>`
in your script or "-echo screen" as a :ref:`command-line argument <start_7>` to see it on the screen.  For a
given command, SPARTA expects certain arguments in a specified order.
If you mess this up, SPARTA will often flag the error, but it may read
a bogus argument and assign a value that is valid, but not what you
wanted.

Generally, SPARTA will print a message to the screen and logfile and
exit gracefully when it encounters a fatal error.  Sometimes it will
print a WARNING to the screen and logfile and continue on; you can
decide if the WARNING is important or not.  A WARNING message that is
generated in the middle of a run is only printed to the screen, not to
the logfile, to avoid cluttering up statistical output.  If SPARTA
crashes or hangs without spitting out an error message first then it
could be a bug (see the :ref:`next section <err_2>`) or one of the following
cases:

SPARTA runs in the available memory a processor allows to be
allocated.  Most reasonable runs are compute limited, not memory
limited, so this shouldn't be a bottleneck on most platforms.  Almost
all large memory allocations in the code are done via C-style malloc's
which will generate an error message if you run out of memory.
Smaller chunks of memory are allocated via C++ "new" statements.  If
you are unlucky, you could run out of memory just when one of these
small requests is made, in which case the code will crash or hang (in
parallel), since SPARTA doesn't trap on those errors.

Illegal arithmetic can cause SPARTA to run slow or crash.  This is
typically due to invalid physics and numerics that your simulation is
computing.  If you see wild statistical values or NaN values in your
SPARTA output, something is wrong with your simulation.  If you
suspect this is happening, it is a good idea to print out statistical
info frequently (e.g. every timestep) via the :doc:`stats <stats>`
command so you can monitor what is happening.  Visualizing the
particle motion is also a good idea to insure your model is behaving
as you expect.

In parallel, one way SPARTA can hang is due to how different MPI
implementations handle buffering of messages.  If the code hangs
without an error message, it may be that you need to specify an MPI
setting or two (usually via an environment variable) to enable
buffering or boost the sizes of messages that can be buffered.
