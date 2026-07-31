.. _start_6:

Running SPARTA
==============

By default, SPARTA runs by reading commands from standard input.  Thus
if you run the SPARTA executable by itself, e.g.


.. parsed-literal::

   spa_g++

it will simply wait, expecting commands from the keyboard.  Typically
you should put commands in an input script and use I/O redirection,
e.g.


.. parsed-literal::

   spa_g++ < in.file

For parallel environments this should also work.  If it does not, use
the '-in' command-line switch, e.g.


.. parsed-literal::

   spa_g++ -in in.file

:doc:`Section 3 <Section_commands>` describes how input scripts are
structured and what commands they contain.

You can test SPARTA on any of the sample inputs provided in the
examples or bench directory.  Input scripts are named in.\* and sample
outputs are named log.\*.name.P where name is a machine and P is the
number of processors it was run on.

Here is how you might run one of the benchmarks on a
Linux box, using mpirun to launch a parallel job:

cd src
make g++
cp spa\_g++ ../bench
cd ../bench
mpirun -np 4 spa\_g++ < in.free

or:


.. parsed-literal::

   cd build
   cmake -DCMAKE_CXX_COMPILER=g++ -DSPARTA_MACHINE=g++ /path/to/sparta/cmake
   cp src/spa_g++ /path/to/bench
   cd /path/to/bench
   mpirun -np 4 spa_g++ < in.free

See `this page <bench_>`_ for timings for this and the other benchmarks on
various platforms.

.. _bench: https://sparta.github.io/bench.html



The screen output from SPARTA is described in the next section.  As it
runs, SPARTA also writes a log.sparta file with the same information.

Note that this sequence of commands copies the SPARTA executable
(spa\_g++) to the directory with the input files.  This may not be
necessary, but some versions of MPI reset the working directory to
where the executable is, rather than leave it as the directory where
you launch mpirun from (if you launch spa\_g++ on its own and not under
mpirun).  If that happens, SPARTA will look for additional input files
and write its output files to the executable directory, rather than
your working directory, which is probably not what you want.

If SPARTA encounters errors in the input script or while running a
simulation it will print an ERROR message and stop or a WARNING
message and continue.  See :doc:`Section 12 <Section_errors>` for a
discussion of the various kinds of errors SPARTA can or can't detect,
a list of all ERROR and WARNING messages, and what to do about them.

SPARTA can run a problem on any number of processors, including a
single processor.  The random numbers used by each processor will be
different so you should only expect statistical consistency if the
same problem is run on different numbers of processors.

SPARTA can run as large a problem as will fit in the physical memory
of one or more processors.  If you run out of memory, you must run on
more processors or setup a smaller problem.
