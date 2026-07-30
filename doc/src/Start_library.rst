.. _start_4:

Building SPARTA as a library
============================

SPARTA can be built as either a static or shared library, which can
then be called from another application or a scripting language.  See
:ref:`Section 6.7 <howto_7>` for more info on coupling
SPARTA to other codes.  See :doc:`Section 11 <Section_python>` for more
info on wrapping and running SPARTA from Python.

The CMake build system will produce the library static of dynamic libsparta
library in build/src.

**Static library:**
^^^^^^^^^^^^^^^^^^^

CMake builds sparta as a static library in libsparta.a, by default.

To build SPARTA as a static library (\*.a file on Linux), type


.. parsed-literal::

   make foo mode=lib

where foo is the machine name.  This kind of library is typically used
to statically link a driver application to SPARTA, so that you can
insure all dependencies are satisfied at compile time.  This will use
the ARCHIVE and ARFLAGS settings in src/MAKE/Makefile.foo.  The build
will create the file libsparta\_foo.a which another application can
link to.  It will also create a soft link libsparta.a, which will
point to the most recently built static library.

**Shared library:**
^^^^^^^^^^^^^^^^^^^

To build SPARTA as a shared library (\*.so file on Linux), which can be
dynamically loaded, e.g. from Python, type


.. parsed-literal::

   make foo mode=shlib

or:


.. parsed-literal::

   cmake -C /path/to/sparta/cmake/presets/foo.cmake -DBUILD_SHARED_LIBS=ON /path/to/sparta/cmake
   make

where foo is the machine name.  This kind of library is required when
wrapping SPARTA with Python; see :doc:`Section\_python <Section_python>`
for details.  This will use the SHFLAGS and SHLIBFLAGS settings in
src/MAKE/Makefile.foo and perform the build in the directory
Obj\_shared\_foo.  This is so that each file can be compiled with the
-fPIC flag which is required for inclusion in a shared library.  The
build will create the file libsparta\_foo.so which another application
can link to dyamically.  It will also create a soft link libsparta.so,
which will point to the most recently built shared library.  This is
the file the Python wrapper loads by default.

Note that for a shared library to be usable by a calling program, all
the auxiliary libraries it depends on must also exist as shared
libraries.  This will be the case for libraries included with SPARTA,
such as the dummy MPI library in src/STUBS or any package libraries in
lib/packages, since they are always built as shared libraries using
the -fPIC switch.  However, if a library like MPI or FFTW does not
exist as a shared library, the shared library build will generate an
error.  This means you will need to install a shared library version
of the auxiliary library.  The build instructions for the library
should tell you how to do this.

Here is an example of such errors when the system FFTW or provided
lib/colvars library have not been built as shared libraries:


.. parsed-literal::

   /usr/bin/ld: /usr/local/lib/libfftw3.a(mapflags.o): relocation
   R_X86_64_32 against \`.rodata' can not be used when making a shared
   object; recompile with -fPIC
   /usr/local/lib/libfftw3.a: could not read symbols: Bad value

   /usr/bin/ld: ../../lib/colvars/libcolvars.a(colvarmodule.o):
   relocation R_X86_64_32 against \`__pthread_key_create' can not be used
   when making a shared object; recompile with -fPIC
   ../../lib/colvars/libcolvars.a: error adding symbols: Bad value

As an example, here is how to build and install the `MPICH library <mpich_>`_, a popular open-source version of MPI, distributed by
Argonne National Labs, as a shared library in the default
/usr/local/lib location:

.. _mpich: http://www-unix.mcs.anl.gov/mpi




.. parsed-literal::

   ./configure --enable-shared
   make
   make install

You may need to use "sudo make install" in place of the last line if
you do not have write privileges for /usr/local/lib.  The end result
should be the file /usr/local/lib/libmpich.so.

**Additional requirement for using a shared library:**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The operating system finds shared libraries to load at run-time using
the environment variable LD\_LIBRARY\_PATH.

Using CMake, ensure that CMAKE\_INSTALL\_PREFIX is set properly and then run "make
-j install" or add build/src to LD\_LIBRARY\_PATH in your shell's environment.

Using make, you may wish to copy the file src/libsparta.so or 
src/libsparta\_g++.so (for example) to a place the system can find it 
by default, such as /usr/local/lib, or you may wish to add the SPARTA
src directory to LD\_LIBRARY\_PATH, so that the current version of the 
shared library is always available to programs that use it.

For the csh or tcsh shells, you would add something like this to your
~/.cshrc file:


.. parsed-literal::

   setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/home/sjplimp/sparta/src

**Calling the SPARTA library:**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Either flavor of library (static or shared) allows one or more SPARTA
objects to be instantiated from the calling program.

When used from a C++ program, all of SPARTA is wrapped in a SPARTA\_NS
namespace; you can safely use any of its classes and methods from
within the calling code, as needed.

When used from a C or Fortran program or a scripting language like
Python, the library has a simple function-style interface, provided in
src/library.cpp and src/library.h.

See :ref:`Section\_howto 4.7 <howto_7>` of the manual for
ideas on how to couple SPARTA to other codes via its library
interface.  See :doc:`Section\_python <Section_python>` of the manual for
a description of the Python wrapper provided with SPARTA that operates
through the SPARTA library interface.

The files src/library.cpp and library.h define the C-style API for
using SPARTA as a library.  See :ref:`Section\_howto 4.6 <howto_6>` of the manual for a description of the
interface and how to extend it for your needs.
