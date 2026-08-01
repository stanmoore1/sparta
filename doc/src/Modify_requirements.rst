.. _mod_10:

What a contribution must include
================================

A new style is not finished when it compiles. SPARTA's continuous
integration enforces most of what follows, so it is cheaper to know the
list up front than to discover it one failed run at a time.

The source files
----------------

One ``.cpp`` and one ``.h``, named for the class and prefixed by the base
class: a compute is ``compute_foo.cpp`` / ``compute_foo.h`` holding class
``ComputeFoo``. :doc:`Section_modify` and the per-style pages that follow
it describe which methods you must override.

The header registers the style with a macro above the include guard:

.. code-block:: c++

   #ifdef COMPUTE_CLASS

   ComputeStyle(foo,ComputeFoo)

   #else

   #ifndef SPARTA_COMPUTE_FOO_H
   #define SPARTA_COMPUTE_FOO_H

The first argument is the keyword typed in an input script, the second is
the class name. The macro name depends on the category:

.. list-table::
   :header-rows: 1
   :widths: 26 26 48

   * - Category
     - Macro
     - File prefix
   * - collision style
     - ``CollideStyle``
     - ``collide_``
   * - input script command
     - ``CommandStyle``
     - none
   * - compute
     - ``ComputeStyle``
     - ``compute_``
   * - dump
     - ``DumpStyle``
     - ``dump_``
   * - fix
     - ``FixStyle``
     - ``fix_``
   * - reaction style
     - ``ReactStyle``
     - ``react_``
   * - region
     - ``RegionStyle``
     - ``region_``
   * - surface collision model
     - ``SurfCollideStyle``
     - ``surf_collide_``
   * - surface reaction model
     - ``SurfReactStyle``
     - ``surf_react_``

There is no list of styles to edit. ``src/Make.sh`` greps every header for
these macros and generates the ``style_*.h`` files at build time, which is
why they are not in the repository. Dropping the two files into ``src``
and rebuilding is all the registration there is.

A documentation page
--------------------

Every style needs a page in ``doc/src`` named after its source file, and
must be listed in ``Section_commands.rst`` and the relevant
``Commands_*.rst`` table. This is checked, not merely expected:
``doc/utils/check-styles.py`` parses the ``Style()`` macros out of the
headers, compares them against the ``:doc:`` links in those tables, and
exits non-zero on any discrepancy. Adding a style without documenting it
fails the build.

Follow the layout of an existing page for the same category. The
conventional sections are Syntax, Examples, Description, Restrictions,
Related commands, and Default. Input script examples go in a
``.. code-block:: SPARTA`` block so they are highlighted.

An example input script
-----------------------

At least one, under ``examples``, in a new or existing suite directory.
Anything named ``in.*`` there becomes a ctest entry automatically, run at
one and four MPI ranks. Generate reference logs for it and commit those
too; :doc:`Developer_testing` describes the naming convention and why the
rank count is part of it.

Keep it short. The whole suite runs on every pull request, so a test that
takes minutes is a cost everyone pays.

Error and warning messages
--------------------------

Errors raised by your code should be documented in a block at the end of
the header, after the ``#endif``:

.. code-block:: c++

   /* ERROR/WARNING messages:

   E: Compute foo mixture does not exist

   Self-explanatory.

   W: Compute foo ignoring split cells

   Split cells have no flow volume of their own, so they are skipped.

   */

``E:`` is a fatal error, ``W:`` a warning; each is followed by a blank
line and a prose paragraph. ``Self-explanatory.`` is the accepted filler
when the message really does say everything.

Note that ``doc/src/Errors_messages.rst`` is maintained by hand and is not
generated from these blocks, so a new message needs to be added in both
places. Nothing in CI catches a mismatch between the two.

What CI checks
--------------

Opening a pull request runs five jobs, all on Ubuntu with the CMake build:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Job
     - What it does
   * - ``mpi``
     - Real MPI with the FFT, PYTHON and VTK packages, then ctest.
   * - ``mpi-stubs``
     - Serial build against ``src/STUBS``, then ctest. Proves the code
       still builds with no MPI installed.
   * - ``mpi-kokkos-exact``
     - KOKKOS with the Serial backend and ``SPARTA_KOKKOS_EXACT=ON``,
       running the suite with ``-k on -sf kk`` against the same reference
       logs as the non-KOKKOS build.
   * - ``bigbig``
     - Builds with ``-DSPARTA_BIGBIG``. Compile coverage only, no tests.
   * - ``docs``
     - Builds the manual with warnings treated as errors, then runs
       ``make -C doc check``.

Two things about the ``docs`` job catch people out. Sphinx runs with
``-W``, so *any* warning fails the pull request, including a ``:doc:`` or
``:ref:`` link that points at nothing. And ``char_check`` rejects any
non-ASCII character in ``doc/src``: write ``--`` rather than an em dash,
and spell out symbols or use ``:math:`` rather than pasting Unicode.

If you add a KOKKOS version of a style, the ``mpi-kokkos-exact`` job is
the one that matters, because it holds the accelerated code to producing
the *same* numbers as the plain build rather than merely similar ones.
:doc:`Developer_kokkos` explains what that requires of the
implementation.
