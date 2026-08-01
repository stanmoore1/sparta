.. _start_5:

Testing SPARTA
==============

SPARTA can be tested by using the CMake build system.

**Basic Testing**

To enable basic testing, use the ``SPARTA_ENABLE_TESTING`` option when
configuring sparta:

.. code-block:: bash

   cmake -C /path/to/sparta/cmake/presets/NAME.cmake \
     -DSPARTA_MACHINE=basic-test-tutorial \
     -DSPARTA_ENABLE_TESTING=ON \
     /path/to/sparta/cmake

Setting ``SPARTA_ENABLE_TESTING`` to ON, adds tests in
``/path/to/sparta/examples/**/in.*`` to be run via ctest. Each ``in.*`` file
corresponds to an individual test. If ``BUILD_MPI`` is ON, tests will be
configured to run with both 1 and 4 mpi ranks. If the binaries are built,
tests can be run via ctest:

.. code-block:: bash

   make
   ctest

This will run all the tests in serial. To run the tests in parallel, use -j:

.. code-block:: bash

   ctest -j4

This will run up to four single rank, single thread per rank ``mpi_1`` tests
in parallel or up to one 4 rank, single thread per rank ``mpi_4`` tests.
ctest has many options including regex filters for running tests that only
match the specified regex. See ``ctest --help`` for more information.

**Adding and Removing tests**

Add more tests by creating one or more input decks in
``/path/to/sparta/examples/SUITE``. Each ``in.*`` file in
``/path/to/sparta/examples/SUITE`` corresponds to an individual test and
will be picked up by the CMake build system if ``SPARTA_ENABLE_TESTING`` is
ON.

To disable tests, remove the ``in.*`` file or remove the ``in.`` prefix from
the ``in.TEST`` file by renaming the file to ``DISABLED.in.TEST``, for
example.

**Advanced Testing**

To enable advanced testing, use the ``SPARTA_DSMC_TESTING_PATH`` option when
configuring sparta:

.. code-block:: bash

   cmake -C /path/to/sparta/cmake/presets/NAME.cmake \
     -DSPARTA_MACHINE=advanced-test-tutorial \
     -DSPARTA_DSMC_TESTING_PATH=/path/to/dsmc_testing \
     /path/to/sparta/cmake

Setting ``SPARTA_DSMC_TESTING_PATH`` to a valid dsmc_testing path adds tests
in ``SPARTA_DSMC_TESTING_PATH`` to be run by
``SPARTA_DSMC_TESTING_PATH/regression.py`` via ctest.

After configuring, build the binaries and run the tests via ctest:

.. code-block:: bash

   make
   ctest

This will run all tests found in ``SPARTA_DSMC_TESTING_PATH/examples`` by
``SPARTA_DSMC_TESTING_PATH/regression.py``. If ``SPARTA_ENABLE_TESTING`` is
ON, all tests found in ``/path/to/sparta/examples`` will configured to run by
``SPARTA_DSMC_TESTING_PATH/regression.py``.

**SPARTA CMake Testing options**

The following options allow the user more control over how the tests are run:

``SPARTA_SPA_ARGS`` can be specified to add additional arguments for the
sparta binaries being run by ctest. This option is only applied if
``SPARTA_ENABLE_TESTING`` or ``SPARTA_DSMC_TESTING_PATH`` are enabled.

``SPARTA_DSMC_TESTING_DRIVER_ARGS`` can be specified to add additional
arguments to the ``SPARTA_DSMC_TESTING_PATH/regression.py`` script.

The ``SPARTA_CTEST_CONFIGS`` option allows the user to run the same set of
binaries with different arguments. ``SPARTA_CTEST_CONFIGS`` lets the user add
additional ctest configurations, separated by ';', that allow
``SPARTA_SPA_ARGS_CONFIG_NAME`` or
``SPARTA_DSMC_TESTING_DRIVER_ARGS_CONFIG_NAME`` to be specified. For example:

.. code-block:: bash

   cmake -C /path/to/sparta/cmake/presets/NAME.cmake \
     -DSPARTA_MACHINE=advanced-test-tutorial \
     -DSPARTA_DSMC_TESTING_PATH=/path/to/dsmc_testing \
     -DSPARTA_CTEST_CONFIGS="PARALLEL;SERIAL" \
     -DSPARTA_SPA_ARGS_SERIAL=spa_serial_args \
     -DSPARTA_SPA_ARGS_PARALLEL=spa_parallel_args \
     -DSPARTA_DSMC_TESTING_DRIVER_ARGS_PARALLEL=driver_parallel_args \
     -DSPARTA_DSMC_TESTING_DRIVER_ARGS_PARALLEL=driver_serial_args \
     /path/to/sparta/cmake

To verify that the binaries are being run with the proper arguments:

.. code-block:: bash

   make
   ctest -C SERIAL -VV
   ctest -C PARALLEL -VV

The ``SPARTA_MULTIBUILD_CONFIGS`` option allows the user to run different
sets of binaries for the same input decks. ``SPARTA_MULTIBUILD_CONFIGS`` lets
the user add additional build configurations, separated by ';', that will
build sparta with the cache file located in
``SPARTA_MULTIBUILD_PRESET_DIR/CONFIG_NAME.cmake``. For example:

.. code-block:: bash

   cmake -DSPARTA_MULTIBUILD_CONFIGS="test_mac;test_mac_mpi" \
         -DSPARTA_MULTIBUILD_PRESET_DIR=/path/to/sparta/cmake/presets/ \
         /path/to/sparta/cmake

This cmake command assumes that
``/path/to/sparta/cmake/presets/{test_mac_mpi,test_mac}.cmake`` exist.

To verify that the correct binaries are being run:

.. code-block:: bash

   make
   ctest -VV
