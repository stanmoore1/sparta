.. _start_5:

Testing SPARTA
==============

SPARTA can be tested by using the CMake build system.

**Basic Testing**

To enable basic testing, use the SPARTA\_ENABLE\_TESTING option when configuring
sparta:

cmake -C /path/to/sparta/cmake/presets/NAME.cmake   -DSPARTA\_MACHINE=basic-test-tutorial   -DSPARTA\_ENABLE\_TESTING=ON   /path/to/sparta/cmake

Setting SPARTA\_ENABLE\_TESTING to ON, adds tests in 
/path/to/sparta/examples/\*\*/in.\* to be run via ctest. Each in.\* file corresponds
to an individual test. If BUILD\_MPI is ON, tests will be configured to run with 
both 1 and 4 mpi ranks. If the binaries are built, tests can be run via ctest:

make
ctest

This will run all the tests in serial. To run the tests in parallel, use -j:

ctest -j4

This will run up to four single rank, single thread per rank mpi\_1 tests in parallel
or up to one 4 rank, single thread per rank mpi\_4 tests. ctest has many options
including regex filters for running tests that only match the specified regex.
See ctest --help for more information.

**Adding and Removing tests**

Add more tests by creating one or more input decks in 
/path/to/sparta/examples/SUITE. Each in.\* file in 
/path/to/sparta/examples/SUITE corresponds to an individual test and
will be picked up by the CMake build system if SPARTA\_ENABLE\_TESTING is ON.

To disable tests, remove the in.\* file or remove the in. prefix from
the in.TEST file by renaming the file to DISABLED.in.TEST, for example.

**Advanced Testing**

To enable advanced testing, use the SPARTA\_DSMC\_TESTING\_PATH option when
configuring sparta:

cmake -C /path/to/sparta/cmake/presets/NAME.cmake   -DSPARTA\_MACHINE=advanced-test-tutorial   -DSPARTA\_DSMC\_TESTING\_PATH=/path/to/dsmc\_testing   /path/to/sparta/cmake

Setting SPARTA\_DSMC\_TESTING\_PATH to a valid dsmc\_testing path adds tests in
SPARTA\_DSMC\_TESTING\_PATH to be run by SPARTA\_DSMC\_TESTING\_PATH/regression.py
via ctest.

After configuring, build the binaries and run the tests via ctest:

make
ctest

This will run all tests found in SPARTA\_DSMC\_TESTING\_PATH/examples by
SPARTA\_DSMC\_TESTING\_PATH/regression.py. If SPARTA\_ENABLE\_TESTING is ON,
all tests found in /path/to/sparta/examples will configured to run by
SPARTA\_DSMC\_TESTING\_PATH/regression.py.

**SPARTA CMake Testing options**

The following options allow the user more control over how the tests are run:

SPARTA\_SPA\_ARGS can be specified to add additional arguments for the sparta 
binaries being run by ctest. This option is only applied if
SPARTA\_ENABLE\_TESTING or SPARTA\_DSMC\_TESTING\_PATH are enabled.

SPARTA\_DSMC\_TESTING\_DRIVER\_ARGS can be specified to add additional arguments to
the SPARTA\_DSMC\_TESTING\_PATH/regression.py script.

The SPARTA\_CTEST\_CONFIGS option allows the user to run the same set of binaries
with different arguments. SPARTA\_CTEST\_CONFIGS lets the user add additional ctest
configurations, seperated by ';', that allow SPARTA\_SPA\_ARGS\_CONFIG\_NAME
or SPARTA\_DSMC\_TESTING\_DRIVER\_ARGS\_CONFIG\_NAME to be specified. For example:

cmake -C /path/to/sparta/cmake/presets/NAME.cmake   -DSPARTA\_MACHINE=advanced-test-tutorial   -DSPARTA\_DSMC\_TESTING\_PATH=/path/to/dsmc\_testing   -DSPARTA\_CTEST\_CONFIGS="PARALLEL;SERIAL"   -DSPARTA\_SPA\_ARGS\_SERIAL=spa\_serial\_args   -DSPARTA\_SPA\_ARGS\_PARALLEL=spa\_parallel\_args   -DSPARTA\_DSMC\_TESTING\_DRIVER\_ARGS\_PARALLEL=driver\_parallel\_args   -DSPARTA\_DSMC\_TESTING\_DRIVER\_ARGS\_PARALLEL=driver\_serial\_args   /path/to/sparta/cmake

To verify that the binaries are being run with the proper arguments:

make
ctest -C SERIAL -VV
ctest -C PARALLEL -VV

The SPARTA\_MULTIBUILD\_CONFIGS option allows the user to run different sets of
binaries for the same input decks. SPARTA\_MULTIBUILD\_CONFIGS lets the user add
additional build configurations, separated by ';', that will build sparta 
with the cache file located in 
`SPARTA\_MULTIBUILD\_PRESET\_DIR/CONFIG\_NAME.cmake`. For example:

cmake -DSPARTA\_MULTIBUILD\_CONFIGS="test\_mac;test\_mac\_mpi"       -DSPARTA\_MULTIBUILD\_PRESET\_DIR=/path/to/sparta/cmake/presets/       /path/to/sparta/cmake

This cmake command assumes that 
/path/to/sparta/cmake/presets/*test\_mac\_mpi,test\_mac*.cmake exist.

To verify that the correct binaries are being run:

make
ctest -VV
