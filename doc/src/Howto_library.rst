.. _howto_6:

Library interface to SPARTA
===========================

As described in :ref:`Section 2.4 <start_4>`, SPARTA can
be built as a library, so that it can be called by another code, used
in a :ref:`coupled manner <howto_7>` with other codes, or
driven through a :doc:`Python interface <Section_python>`.

All of these methodologies use a C-style interface to SPARTA that is
provided in the files src/library.cpp and src/library.h.  The
functions therein have a C-style argument list, but contain C++ code
you could write yourself in a C++ application that was invoking SPARTA
directly.  The C++ code in the functions illustrates how to invoke
internal SPARTA operations.  Note that SPARTA classes are defined
within a SPARTA namespace (SPARTA\_NS) if you use them from another C++
application.

Library.cpp provides these core functions:


.. parsed-literal::

   void sparta_open(int, char \*\*, MPI_Comm, void \*\*);
   void sparta_close(void \*);
   void sparta_file(void \*, char \*);
   char \*sparta_command(void \*, char \*);

The sparta\_open() function is used to initialize SPARTA, passing in a
list of strings as if they were :ref:`command-line arguments <start_7>` when SPARTA is run in
stand-alone mode from the command line, and a MPI communicator for
SPARTA to run under.  It returns a ptr to the SPARTA object that is
created, and which is used in subsequent library calls.  The
sparta\_open() function can be called multiple times, to create
multiple instances of SPARTA.

SPARTA will run on the set of processors in the communicator.  This
means the calling code can run SPARTA on all or a subset of
processors.  For example, a wrapper script might decide to alternate
between SPARTA and another code, allowing them both to run on all the
processors.  Or it might allocate half the processors to SPARTA and
half to the other code and run both codes simultaneously before
syncing them up periodically.  Or it might instantiate multiple
instances of SPARTA to perform different calculations.

The sparta\_close() function is used to shut down an instance of SPARTA
and free all its memory.

The sparta\_file() and sparta\_command() functions are used to pass a
file or string to SPARTA as if it were an input script or single
command in an input script.  Thus the calling code can read or
generate a series of SPARTA commands one line at a time and pass it
thru the library interface to setup a problem and then run it,
interleaving the sparta\_command() calls with other calls to extract
information from SPARTA, perform its own operations, or call another
code's library.

Beyond these, library.cpp provides many additional functions so a
driver program can run SPARTA, read its output, handle errors, and
introspect its build.  Briefly, these include: functions to execute a
multi-line command string (sparta\_commands\_string) and to query or
interrupt a run (sparta\_is\_running, sparta\_force\_timeout); functions to
read thermodynamic output, both live (sparta\_get\_thermo) and from a
cached snapshot of the most recent output that can be read while a run
is in progress (sparta\_last\_thermo); error-handling functions that let
a driver recover from an error instead of aborting the process, when
SPARTA is built with exceptions (sparta\_has\_error,
sparta\_get\_last\_error\_message); data-extraction functions for global
quantities and settings such as box bounds, units, the version, the
git commit and branch, and per-surface counts (sparta\_extract\_global,
sparta\_extract\_setting) and for per-particle, per-grid, per-surface or
global data produced by a compute or fix (sparta\_extract\_compute,
sparta\_extract\_fix, sparta\_extract\_variable); introspection functions
to enumerate the styles compiled into the executable (sparta\_style\_count,
sparta\_style\_name) and the currently defined computes, fixes, dumps,
regions, variables, mixtures and surface collision/reaction models
(sparta\_id\_count, sparta\_id\_name); and build-configuration queries
(sparta\_version, sparta\_config\_has\_package, sparta\_config\_accelerator,
and the sparta\_config\_has\_\*\_support functions for MPI, PNG, JPEG,
FFmpeg and gzip support).  See the library.cpp file and its associated
header file library.h for the full list and exact signatures.

Other functions may be added to the library interface as needed to
allow reading from or writing to internal SPARTA data structures.

The key idea of the library interface is that you can write any
functions you wish to define how your code talks to SPARTA and add
them to src/library.cpp and src/library.h, as well as to the :doc:`Python interface <Section_python>`.  The routines you add can in principle
access or change any SPARTA data you wish.  The examples/COUPLE and
python directories have example C++ and C and Python codes which show
how a driver code can link to SPARTA as a library, run SPARTA on a
subset of processors, grab data from SPARTA, change it, and put it
back into SPARTA.

.. warning::

   The examples/COUPLE dir has not been added to the
   distribution yet.
