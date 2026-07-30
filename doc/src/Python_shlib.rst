.. _py_1:

Building SPARTA as a shared library
===================================

Instructions on how to build SPARTA as a shared library are given in
:ref:`Section 2.4 <start_4>`.  A shared library is one
that is dynamically loadable, which is what Python requires.  On Linux
this is a library file that ends in ".so", not ".a".

For make, from the src directory, type


.. parsed-literal::

   make mode=shlib foo

For CMake, from the build directory, tyoe


.. parsed-literal::

   cmake -C /path/to/sparta/cmake/presets/foo.cmake -DBUILD_SHARED_LIBS=ON /path/to/sparta/cmake
   make

where foo is the machine target name, such as icc or g++ or serial.
This should create the file libsparta\_foo.so in the src directory, as
well as a soft link libsparta.so, which is what the Python wrapper
will load by default.  Note that if you are building multiple machine
versions of the shared library, the soft link is always set to the
most recently built version.

If this fails, see :ref:`Section 2.3 <start_3>` for more
details, especially if your SPARTA build uses auxiliary libraries like
MPI which may not be built as shared libraries on your system.
