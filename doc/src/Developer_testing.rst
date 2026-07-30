Testing
=======

SPARTA's test suite is its example problems. Each example directory holds
an input script and a set of reference log files, and testing consists of
re-running the script and comparing the numbers it prints against the
reference. There are no unit tests; the granularity is a whole simulation.

The machinery is ``tools/testing/regression.py``, wired into ctest by
``cmake/common/test/sparta_test_utils.cmake``.

Reference logs
--------------

The ``examples`` tree holds roughly 40 problem directories and around 200
reference logs. They are named for the date they were blessed, the
configuration, and the problem:

.. parsed-literal::

   examples/spiky/log.11Sep23.mpi_1.spiky
   examples/spiky/log.11Sep23.mpi_4.spiky

The processor count is part of the name because results depend on it.
Random numbers are consumed in a different order when particles are
distributed differently, so a 1-rank and a 4-rank run of the same script
diverge at the level of individual particles
(:doc:`Developer_parallel`). Each is therefore blessed separately, and a
test compares like with like.

Comparison is numerical, not textual. ``regression.py`` parses the stats
table out of both logs and compares columns within a tolerance, so
round-off differences between compilers and platforms do not cause
failures while real regressions do. The script descends from LAMMPS's
regression tooling and still uses the ``log.py`` reader from the Pizza.py
toolkit.

Running the tests
-----------------

Tests are registered with ctest when SPARTA is configured with
``-DSPARTA_ENABLE_TESTING=ON``:

.. parsed-literal::

   mkdir build; cd build
   cmake -C ../cmake/presets/mpi.cmake -DSPARTA_ENABLE_TESTING=ON ../cmake
   make -j4
   ctest --output-on-failure -j4

``sparta_add_test()`` in ``sparta_test_utils.cmake`` creates one ctest
entry per input script per processor count, each running the script in its
own working directory and invoking ``regression.py`` as the test driver.

``SPARTA_SPA_ARGS`` passes extra command-line arguments to every test run,
which is how the KOKKOS configuration is tested without duplicating the
test list.

Re-blessing
-----------

When a change alters results legitimately -- a bug fix, a model change --
the reference logs have to be regenerated. ``tools/testing/rebless.sh``
does this from the build directory:

.. parsed-literal::

   cd /path/to/sparta/build
   /path/to/sparta/tools/testing/rebless.sh [--rerun-failed]

``--rerun-failed`` re-blesses only the tests that failed, which is what you
usually want: re-blessing everything hides unrelated regressions that crept
in at the same time.

Re-blessing is a deliberate act. A reference log is a claim that the
physics is right, and replacing one should be justified in the commit
message that does it.

Continuous integration
----------------------

``.github/workflows/main.yml`` runs four build-and-test configurations on
every pull request:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Job
     - What it covers
   * - ``mpi``
     - Real MPI with the FFT, PYTHON and VTK packages, plus ctest. Uploads
       the example outputs as an artifact.
   * - ``mpi-stubs``
     - The serial build against ``src/STUBS``, proving SPARTA still builds
       and runs with no MPI installation (:doc:`Developer_utils`).
   * - ``mpi-kokkos-exact``
     - The KOKKOS package with the Serial backend and
       ``SPARTA_KOKKOS_EXACT=ON``, running the regression suite with
       ``-k on -sf kk`` against the *same* reference logs as the non-KOKKOS
       build. See :doc:`Developer_kokkos`.
   * - ``bigbig``
     - A build with ``-DSPARTA_BIGBIG`` -- compile coverage for the wide
       integer variant, which is not exercised by the other jobs.

A fifth job, ``docs``, builds this manual with warnings treated as errors
and runs the documentation checks.

The ``mpi-kokkos-exact`` job is the one worth understanding, because it is
what makes the accelerated code trustworthy: without exact reproducibility
there would be no way to check a threaded implementation against a serial
one except statistically, and statistical agreement hides many real bugs.

Adding a test
-------------

Add an input script to a new or existing directory under ``examples``,
verify by inspection that its results are physically right, generate
reference logs for the processor counts you want covered, and register it
with ``sparta_add_test()``. Keep it short -- the suite runs on every pull
request, so a test that takes minutes is a tax on everyone.

Where to go next
----------------

* :doc:`Developer_kokkos` -- why exact reproducibility matters
* :doc:`Developer_parallel` -- why results depend on processor count
* :doc:`Section_start` -- building SPARTA
