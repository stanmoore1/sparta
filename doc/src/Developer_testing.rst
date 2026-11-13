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

The ``examples`` tree holds about 40 problem directories and about 260
reference logs.  They are named for the date they were blessed, the number
of MPI ranks (``mpi_N``), and the problem; the same log serves the serial,
MPI and KOKKOS builds:

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
table out of both logs and compares each column with a max-norm, absolute
tolerance (1e-7 by default), so tiny round-off passes while real
regressions do not.  Problems that are chaotic enough to amplify last-bit
differences between platforms can still fail, and some are left out of the
default list for that reason.  The script descends from LAMMPS's regression
tooling and reads logs with the ``olog`` reader from the Pizza.py toolkit
(``tools/testing/olog.py``).

Running the tests
-----------------

Tests are registered with ctest when SPARTA is configured with
``-DSPARTA_ENABLE_TESTING=ON``:

.. parsed-literal::

   mkdir build; cd build
   cmake -C ../cmake/presets/mpi.cmake -DSPARTA_ENABLE_TESTING=ON ../cmake
   make -j4
   ctest --output-on-failure -j4

``sparta_add_all_tests_to_config()`` in ``sparta_test_utils.cmake`` finds
every ``in.*`` script in the enabled example directories, apart from those
listed in ``SPARTA_DISABLED_TESTS``, and creates one ctest entry per script
for 1 and for 4 MPI ranks (only 1 in a build without MPI), each running in the
directory's copy under ``build/examples/<directory>`` and invoking
``regression.py`` as the test driver.  Tests from the same directory share
that copy.

``SPARTA_SPA_ARGS`` passes extra command-line arguments to every test run,
which is how the KOKKOS configuration is tested without duplicating the
test list.

Re-blessing
-----------

When a change alters results legitimately -- a bug fix, a model change --
the reference logs have to be regenerated. ``tools/testing/rebless.sh``
does this; it must be run from a build directory named ``build``:

.. parsed-literal::

   cd /path/to/sparta/build
   /path/to/sparta/tools/testing/rebless.sh [--rerun-failed]

Without options it reconfigures (with the preset named by
``SPARTA_REBLESS_PRESET``, ``mpi`` by default), rebuilds, and runs the whole
suite with re-blessing on.  ``--rerun-failed`` instead reruns only the tests
that failed last time.  Either way it then promotes *every* new log it finds
under ``build/examples`` and stages the changes with git, so review
``git diff --staged`` before committing: re-blessing more than you meant to
hides unrelated regressions that crept in at the same time.

Re-blessing is a deliberate act. A reference log is a claim that the
physics is right, and replacing one should be justified in the commit
message that does it.

Continuous integration
----------------------

``.github/workflows/main.yml`` runs four build configurations on pull
requests to master, three of which run the test suite:

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
     - Compile only: a KOKKOS build (the ``kokkos_omp`` preset) with
       ``-DSPARTA_BIGBIG``, covering the wide integer variant, which the
       other jobs do not build.  Despite its name, the preset does not
       enable Kokkos' OpenMP backend, so this build uses Serial.

A fifth job, ``doc``, builds this manual with warnings treated as errors on
Python 3.9 and 3.11, then checks its links, checks the command and style
listings against ``src/``, and spell-checks it.

The ``mpi-kokkos-exact`` job is the one worth understanding, because it is
what makes the accelerated code trustworthy: without exact reproducibility
there would be no way to check a threaded implementation against a serial
one except statistically, and statistical agreement hides many real bugs.

Adding a test
-------------

Add an ``in.*`` input script to a directory under ``examples``; a new
directory also has to be added to ``SPARTA_ENABLED_TEST_SUITES`` in
``cmake/common/set/sparta_cmake_defaults.cmake``.  Verify by inspection
that its results are physically right, then commit reference logs for 1
and 4 ranks -- a test with none writes one for you to review, named
``log.archive.<date>.mpi_N.<test>``. Keep it short -- the suite runs on every pull
request, so a test that takes minutes is a tax on everyone.

Where to go next
----------------

* :doc:`Developer_kokkos` -- why exact reproducibility matters
* :doc:`Developer_parallel` -- why results depend on processor count
* :doc:`Section_start` -- building SPARTA
