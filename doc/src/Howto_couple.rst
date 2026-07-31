.. _howto_7:

Coupling SPARTA to other codes
==============================

SPARTA is designed to allow it to be coupled to other codes.  For
example, a continuum finite element (FE) simulation might use SPARTA
grid cell quantities as boundary conditions on FE nodal points,
compute a FE solution, and return continuum flow conditions as
boundary conditions for SPARTA to use.

SPARTA can be coupled to other codes in at least 3 ways.  Each has
advantages and disadvantages, which you'll have to think about in the
context of your application.

(1) Define a new :doc:`fix <fix>` command that calls the other code.  In
this scenario, SPARTA is the driver code.  During its timestepping,
the fix is invoked, and can make library calls to the other code,
which has been linked to SPARTA as a library.  See :doc:`Section 8 <Section_modify>` of the documentation for info on how to add a
new fix to SPARTA.

(2) Define a new SPARTA command that calls the other code.  This is
conceptually similar to method (1), but in this case SPARTA and the
other code are on a more equal footing.  Note that now the other code
is not called during the timestepping of a SPARTA run, but between
runs.  The SPARTA input script can be used to alternate SPARTA runs
with calls to the other code, invoked via the new command.  The
:doc:`run <run>` command facilitates this with its *every* option, which
makes it easy to run a few steps, invoke the command, run a few steps,
invoke the command, etc.

In this scenario, the other code can be called as a library, as in
(1), or it could be a stand-alone code, invoked by a system() call
made by the command (assuming your parallel machine allows one or more
processors to start up another program).  In the latter case the
stand-alone code could communicate with SPARTA thru files that the
command writes and reads.

See :doc:`Section\_modify <Section_modify>` of the documentation for how
to add a new command to SPARTA.

(3) Use SPARTA as a library called by another code.  In this case the
other code is the driver and calls SPARTA as needed.  Or a wrapper
code could link and call both SPARTA and another code as libraries.
Again, the :doc:`run <run>` command has options that allow it to be
invoked with minimal overhead (no setup or clean-up) if you wish to do
multiple short runs, driven by another program.

Examples of driver codes that call SPARTA as a library are included in
the examples/COUPLE directory of the SPARTA distribution; see
examples/COUPLE/README for more details.

.. warning::

   The examples/COUPLE dir has not been added to the
   distribution yet.

:ref:`Section 2.3 <start_3>` of the manual describes how to
build SPARTA as a library.  Once this is done, you can interface with
SPARTA either via C++, C, Fortran, or Python (or any other language
that supports a vanilla C-like interface).  For example, from C++ you
could create one (or more) "instances" of SPARTA, pass it an input
script to process, or execute individual commands, all by invoking the
correct class methods in SPARTA.  From C or Fortran you can make
function calls to do the same things.  See
:doc:`Section\_9 <Section_python>` of the manual for a description of the
Python wrapper provided with SPARTA that operates through the SPARTA
library interface.

The files src/library.cpp and library.h contain the C-style interface
to SPARTA.  See :ref:`Section 6.6 <howto_6>` of the manual for a description
of the interface and how to extend it for your needs.

Note that the sparta\_open() function that creates an instance of
SPARTA takes an MPI communicator as an argument.  This means that
instance of SPARTA will run on the set of processors in the
communicator.  Thus the calling code can run SPARTA on all or a subset
of processors.  For example, a wrapper script might decide to
alternate between SPARTA and another code, allowing them both to run
on all the processors.  Or it might allocate half the processors to
SPARTA and half to the other code and run both codes simultaneously
before syncing them up periodically.  Or it might instantiate multiple
instances of SPARTA to perform different calculations.
