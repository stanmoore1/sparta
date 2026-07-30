Utility classes
===============

A handful of small classes underpin the rest of the code. None of them is
interesting on its own, but new code is expected to use them rather than
roll its own equivalent, so they are worth knowing.

Integer sizes
-------------

``src/spatype.h`` defines the integer types SPARTA uses in place of bare
``int``, and selects their widths from a compile-time size variant:

.. list-table::
   :header-rows: 1
   :widths: 14 22 22 22 20

   * - Type
     - ``SPARTA_SMALL``
     - ``SPARTA_BIG`` (default)
     - ``SPARTA_BIGBIG``
     - Used for
   * - ``smallint``
     - ``int``
     - ``int``
     - ``int``
     - per-processor counts
   * - ``cellint``
     - ``uint32_t``
     - ``uint32_t``
     - ``uint64_t``
     - grid cell IDs
   * - ``surfint``
     - ``int``
     - ``int``
     - ``int64_t``
     - surface element IDs
   * - ``bigint``
     - ``int``
     - ``int64_t``
     - ``int64_t``
     - global counts, timesteps

The default build handles more than 2 billion particles (``bigint`` is
64-bit) but limits cell and surface IDs to 32 bits. ``-DSPARTA_BIGBIG``
widens those too, at the cost of memory and bandwidth in the grid
structures. Because a cell ID encodes a path through the grid hierarchy
rather than an index (:doc:`Developer_grid`), deep hierarchies exhaust 32
bits sooner than the cell *count* would suggest -- which is the usual reason
to need ``BIGBIG``.

The rule when writing new code: a quantity that is per-processor is
``int``; a quantity summed over all processors, or a timestep count, is
``bigint``; a cell ID is ``cellint``; a surface ID is ``surfint``. Getting
this wrong produces overflow that only appears at scale.

``src/spatype.h`` also defines ``SPARTA_ALIGN(n)``, which expands to
``alignas(n)`` where the compiler supports it and to nothing otherwise.

Memory
------

``Memory`` (``src/memory.h``) wraps allocation. Its templated ``create``,
``grow`` and ``destroy`` methods handle 1d, 2d and 3d arrays, including
offset variants for arrays that are not zero-based:

.. code-block:: c++

   TYPE *create(TYPE *&array, int n, const char *name)
   TYPE *grow(TYPE *&array, int n, const char *name)
   void destroy(TYPE *array)

Multi-dimensional arrays are allocated as one contiguous block with a
pointer array on top, so ``array[i][j]`` works while the data stays
contiguous -- which matters for both cache behavior and MPI, since the block
can be sent in one call.

The ``name`` argument appears in the error message if the allocation fails,
so it should be the variable's name. ``grow()`` on a null pointer is
equivalent to ``create()``, which is why growth loops do not need to
special-case the first call.

Use these rather than ``new``/``malloc``: they centralize failure handling
and are what ``MemoryKokkos`` overrides in a KOKKOS build
(:doc:`Developer_kokkos`).

Error
-----

``Error`` (``src/error.h``) is how the code stops or complains:

.. code-block:: c++

   void all(const char *, int, const char *);       // all procs, fatal
   void one(const char *, int, const char *);       // one proc, fatal
   void warning(const char *, int, const char *);   // non-fatal
   void message(const char *, int, const char *);   // informational

The first two arguments are always ``FLERR``, the macro from
``src/pointers.h`` that expands to ``__FILE__,__LINE__``:

.. code-block:: c++

   error->all(FLERR,"Illegal compute grid command");

The ``all``/``one`` distinction is a correctness matter, not a style
preference. ``all()`` is collective and must be reached by every processor;
calling it from one processor deadlocks. ``one()`` is for errors only one
processor can detect -- a bad value in its own data -- and aborts the run
without waiting for the others.

Error and warning messages are also documented: the ``ERROR/WARNING``
comment blocks at the bottom of many header files feed
:doc:`Errors_messages`.

Input and variables
-------------------

``Input`` (``src/input.cpp``) reads the input script, handles line
continuation and quoting, performs variable substitution, and dispatches
commands. Its argument-parsing helpers are used throughout the code and
should be preferred to hand-rolled parsing:

* ``numeric()``, ``inumeric()``, ``bnumeric()`` -- convert an argument to
  a double, int or ``bigint``, reporting an error that names the
  offending command if the argument is not a number
* ``bounds()`` -- parse an index range such as ``2*5`` or ``*`` into low and
  high limits
* ``expand_args()`` -- expand wildcard arguments such as ``c_1[*]`` into an
  explicit argument list
* ``count_words()``, ``substitute()`` -- tokenizing and ``$`` substitution

``Variable`` (``src/variable.cpp``) implements the
:doc:`variable <variable>` command: index, loop, world, universe, string,
getenv, equal, particle and grid styles, plus the expression evaluator used
for equal-style formulas. It is the largest single file in ``src``, mostly
because of the operator and function table in the evaluator.

Random numbers
--------------

Two generators are provided, ``RanKnuth`` (``src/random_knuth.h``) and
``RanMars`` (``src/random_mars.h``). Classes that need randomness
own an instance seeded from the :doc:`seed <seed>` command combined with
the processor rank, so different ranks draw different streams.

Two consequences follow, and both surface regularly as apparent bugs:
results depend on processor count, and they depend on the order in which
particles are processed. Neither is a defect. See
:doc:`Developer_parallel` and :doc:`Errors_common`.

The KOKKOS package replaces these with a Kokkos random pool except when
built with ``SPARTA_KOKKOS_EXACT`` -- see :doc:`Developer_kokkos`.

Running without MPI
-------------------

``src/STUBS`` is a stub MPI implementation: ``mpi.h`` plus a ``mpi.c`` that
implements the subset of MPI SPARTA uses for a single processor. Linking
against it produces a serial executable with no MPI dependency, and the
code itself is unchanged -- ``MPI_Allreduce`` on one processor is a copy.

This is what the ``serial`` cmake preset and the ``mpi-stubs`` CI job
build, and it is the easiest way to get SPARTA running somewhere without a
working MPI installation.

Where to go next
----------------

* :doc:`Developer_org` -- where these classes sit in the object model
* :doc:`Developer_testing` -- the build and test matrix
