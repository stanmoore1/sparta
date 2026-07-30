.. _start_2:

Making SPARTA
=============

This section has the following sub-sections:

* :ref:`Read this first <start_2_1>`
* :ref:`Steps to build a SPARTA executable using make <start_2_2_1>`
* :ref:`Steps to build a SPARTA executable using CMake <start_2_2_2>`
* :ref:`Common errors that can occur when making SPARTA <start_2_3>`
* :ref:`Additional build tips using make <start_2_4_1>`
* :ref:`Additional build tips using CMake <start_2_4_2>`
* :ref:`Building for a Mac <start_2_5>`
* :ref:`Building for Windows <start_2_6>`


----------


.. _start_2_1:

**Read this first:** 

Building SPARTA can be non-trivial.  You may need to edit a makefile,
there are compiler options to consider, additional libraries can be
used (MPI, JPEG).

Please read this section carefully.  If you are not comfortable with
cmake, makefiles, or building codes on a Linux platform, or running an MPI
job on your machine, please find a local expert to help you.

SPARTA requires that the compiler supports C++11. SPARTA will throw an error
if this is not the case. If you are building SPARTA with Kokkos, the compiler
must support C++20.

If you have a build problem that you are convinced is a SPARTA issue
(e.g. the compiler complains about a line of SPARTA source code), then
please send an email to the
`developers <https://sparta.github.io/authors.html>`_.

If you succeed in building SPARTA on a new kind of machine, for which
there isn't a similar Makefile in the src/MAKE directory or .cmake file
in cmake/presets (for KOKKOS builds, a .cmake file in cmake/presets),
send it to the 
`developers <https://sparta.github.io/authors.html>`_ and we'll include it in future SPARTA releases.


----------


.. _start_2_2_1:

**Steps to build a SPARTA executable using make:** 

**Step 0**

The src directory contains the C++ source and header files for SPARTA.
It also contains a top-level Makefile and a MAKE sub-directory with
low-level Makefile.\* files for many machines.  From within the src
directory, type "make" or "gmake".  You should see a list of available
choices.  If one of those is the machine and options you want, you can
type a command like:


.. parsed-literal::

   make g++
   or
   gmake mac

Note that on a multi-core platform you can launch a parallel make, by
using the "-j" switch with the make command, which will build SPARTA
more quickly.

If you get no errors and an executable like spa\_g++ or spa\_mac is
produced, you're done; it's your lucky day.

Note that by default none of the SPARTA optional packages are
installed.  To build SPARTA with optional packages, see :ref:`this section <start_3>` below.

.. warning::

   The optional KOKKOS accelerator package does not
   support Makefiles and must be built using CMake instead (see below).

**Step 1**

If Step 0 did not work, you will need to create a low-level Makefile
for your machine, like Makefile.foo.  Copy an existing
src/MAKE/Makefile.\* as a starting point.  The only portions of the
file you need to edit are the first line, the "compiler/linker
settings" section, and the "SPARTA-specific settings" section.

**Step 2**

Change the first line of src/MAKE/Makefile.foo to list the word "foo"
after the "#", and whatever other options it will set.  This is the
line you will see if you just type "make".

**Step 3**

The "compiler/linker settings" section lists compiler and linker
settings for your C++ compiler, including optimization flags.  You can
use g++, the open-source GNU compiler, which is available on all Linux
systems.  You can also use mpicc which will typically be available if
MPI is installed on your system, though you should check which actual
compiler it wraps.  Vendor compilers often produce faster code.  On
boxes with Intel CPUs, we suggest using the commercial Intel icc
compiler, which can be downloaded from `Intel's compiler site <intel_>`_.

.. _intel: http://www.intel.com/software/products/noncom



If building a C++ code on your machine requires additional libraries,
then you should list them as part of the LIB variable.

The DEPFLAGS setting is what triggers the C++ compiler to create a
dependency list for a source file.  This speeds re-compilation when
source (\*.cpp) or header (\*.h) files are edited.  Some compilers do
not support dependency file creation, or may use a different switch
than -D.  GNU g++ works with -D.  Note that when you build SPARTA for
the first time on a new platform, a long list of \*.d files will be
printed out rapidly.  This is not an error; it is the Makefile doing
its normal creation of dependencies.

**Step 4**

The "system-specific settings" section has several parts.  Note that
if you change any -D setting in this section, you should do a full
re-compile, after typing "make clean", which will describe different
clean options.

The SPA\_INC variable is used to include options that turn on ifdefs
within the SPARTA code.  The options that are currently recognized are:

* -DSPARTA\_GZIP
* -DSPARTA\_JPEG
* -DSPARTA\_PNG
* -DSPARTA\_FFMPEG
* -DSPARTA\_MAP
* -DSPARTA\_UNORDERED\_MAP
* -DSPARTA\_SMALL
* -DSPARTA\_BIG
* -DSPARTA\_BIGBIG
* -DSPARTA\_LONGLONG\_TO\_LONG

The read\_data and dump commands will read/write gzipped files if you
compile with -DSPARTA\_GZIP.  It requires that your Linux support the
"popen" command.

If you use -DSPARTA\_JPEG and/or -DSPARTA\_PNG, the :doc:`dump image <dump>` command will be able to write out JPEG and/or PNG
image files respectively. If not, it will only be able to write out
PPM image files.  For JPEG files, you must also link SPARTA with a
JPEG library, as described below.  For PNG files, you must also link
SPARTA with a PNG library, as described below.

If you use -DSPARTA\_FFMPEG, the :doc:`dump movie <dump_image>` command
will be available to support on-the-fly generation of rendered movies
the need to store intermediate image files.  It requires that your
machines supports the "popen" function in the standard runtime library
and that an FFmpeg executable can be found by SPARTA during the run.

If you use -DSPARTA\_MAP, SPARTA will use the STL map class for hash
tables.  This is less efficient than the unordered map class which is
not yet supported by all C++ compilers.  If you use
-DSPARTA\_UNORDERED\_MAP, SPARTA will use the unordered\_map class for
hash tables and will assume it is part of the STL (e.g. this works for
Clang++).  The default is to use the unordered map class from the
"tri1" extension to the STL which is supported by most compilers.  So
only use either of these options if the build complains that unordered
maps are not recognized.

Use at most one of the -DSPARTA\_SMALL, -DSPARTA\_BIG, -DSPARTA\_BIGBIG
settings.  The default is -DSPARTA\_BIG.  These refer to use of 4-byte
(small) vs 8-byte (big) integers within SPARTA, as described in
src/spatype.h.  The only reason to use the BIGBIG setting is if you
have a regular grid with more than ~2 billion grid cells or a
hierarchical grid with enough levels that grid cell IDs cannot fit in
a 32-bit integer.  In either case, SPARTA will generate an error
message for "Cell ID has too many bits".  See :ref:`Section 4.8 <howto_8>` of the manual for details on how cell
IDs are formatted.  The only reason to use the SMALL setting is if
your machine does not support 64-bit integers.

In all cases, the size of problem that can be run on a per-processor
basis is limited by 4-byte integer storage to about 2 billion
particles per processor (2\^31), which should not normally be a
restriction since such a problem would have a huge per-processor
memory and would run very slowly in terms of CPU secs/timestep.

The -DSPARTA\_LONGLONG\_TO\_LONG setting may be needed if your system or
MPI version does not recognize "long long" data types.  In this case a
"long" data type is likely already 64-bits, in which case this setting
will use that data type.

Using one of the -DFFT\_PACK\_ARRAY, -DFFT\_PACK\_POINTER, and -DFFT\_PACK\_MEMCPY
options can make for faster parallel FFTs on some platforms.  The
-DFFT\_PACK\_ARRAY setting is the default.  See the :doc:`compute fft/grid <compute_fft_grid>` command for info about FFTs.  See Step
6 below for info about building SPARTA with an FFT library.

**Step 5**

The 3 MPI variables are used to specify an MPI library to build SPARTA
with.

If you want SPARTA to run in parallel, you must have an MPI library
installed on your platform.  If you use an MPI-wrapped compiler, such
as "mpicc" to build, you should be able to leave these 3 variables
blank; the MPI wrapper knows where to find the needed files.  If not,
and MPI is installed on your system in the usual place (under
/usr/local), you also may not need to specify these 3 variables.  On
some large parallel machines which use "modules" for their
compile/link environements, you may simply need to include the correct
module in your build environment.  Or the parallel machine may have a
vendor-provided MPI which the compiler has no trouble finding.

Failing this, with these 3 variables you can specify where the mpi.h
file is found (via MPI\_INC), and the MPI library file is found (via
MPI\_PATH), and the name of the library file (via MPI\_LIB).  See
Makefile.serial for an example of how this can be done.

If you are installing MPI yourself, we recommend MPICH 1.2 or 2.0 or
OpenMPI.  MPICH can be downloaded from the `Argonne MPI site <https://www.mpich.org>`_.  OpenMPI can be downloaded from the
`OpenMPI site <http://www.open-mpi.org>`_.  If you are running on a big
parallel platform, your system admins or the vendor should have
already installed a version of MPI, which will be faster than MPICH or
OpenMPI, so find out how to build and link with it.  If you use MPICH
or OpenMPI, you will have to configure and build it for your platform.
The MPI configure script should have compiler options to enable you to
use the same compiler you use for the SPARTA build, which can avoid
problems that can arise when linking SPARTA to the MPI library.

If you just want to run SPARTA on a single processor, you can use the
dummy MPI library provided in src/STUBS, since you don't need a true
MPI library installed on your system.  You will also need to build the
STUBS library for your platform before making SPARTA itself.  From the
src directory, type "make mpi-stubs", or from within the STUBS dir,
type "make" and it should create a libmpi.a suitable for linking to
SPARTA.  If this build fails, you will need to edit the STUBS/Makefile
for your platform.

The file STUBS/mpi.cpp provides a CPU timer function called
MPI\_Wtime() that calls gettimeofday() .  If your system doesn't
support gettimeofday() , you'll need to insert code to call another
timer.  Note that the ANSI-standard function clock() function rolls
over after an hour or so, and is therefore insufficient for timing
long SPARTA simulations.

**Step 6**

The 3 FFT variables allow you to specify an FFT library which SPARTA
uses (for performing 1d FFTs) when built with its FFT package, which
contains commands that invoke FFTs.

SPARTA supports various open-source or vendor-supplied FFT libraries
for this purpose.  If you leave these 3 variables blank, SPARTA will
use the open-source `KISS FFT library <http://kissfft.sf.net>`_, which is
included in the SPARTA distribution.  This library is portable to all
platforms and for typical SPARTA simulations is almost as fast as FFTW
or vendor optimized libraries.  If you are not including the FFT
package in your build, you can also leave the 3 variables blank.

Otherwise, select which kinds of FFTs to use as part of the FFT\_INC
setting by a switch of the form -DFFT\_XXX. 
Available values for XXX
are: MKL or FFTW3.
Selecting -DFFT\_FFTW will use the FFTW3 library.

Similarly a separate FFT library can be specified for KOKKOS package.
By default, SPARTA will use a Kokkos version of the open-source `KISS FFT library <http://kissfft.sf.net>`_, which is included in the SPARTA
distribution. Note that using the KISS FFT library on GPUs may give
suboptimal performance. Other options can be specified using the form
-DFFT\_KOKKOS\_XXX. Available values for XXX when using Kokkos are:
CUFFT, HIPFFT, MKL\_GPU, MKL or FFTW3. When using the Kokkos CUDA
backend, either CUFFT or KISS must be used. When using the Kokkos HIP
backend, either HIPFFT or KISS must be used. When using the Kokkos
SYCL backend, either MKL\_GPU or KISS must be used. When using the
Kokkos OpenMP or Serial backend, either MKL, FFTW3, or KISS must be
used.

The CUFFT option specifies the `cuFFT library <https://developer.nvidia.com/cufft>`_ from NVIDIA. The HIPFFT
option specifies the `rocFFT library <https://rocm.docs.amd.com/projects/rocFFT/en/latest/>`_ from
AMD. The HIPFFT option specifies the `rocFFT library <https://rocm.docs.amd.com/projects/rocFFT/en/latest/>`_ from
AMD. The MKL\_GPU option supports GPU offload of FFTs on Intel GPUs
with oneMKL using the Kokkos SYCL backend.

You may also need to set the FFT\_INC, FFT\_PATH, and FFT\_LIB variables,
so the compiler and linker can find the needed FFT header and library
files.  Note that on some large parallel machines which use "modules"
for their compile/link environements, you may simply need to include
the correct module in your build environment.  Or the parallel machine
may have a vendor-provided FFT library which the compiler has no
trouble finding.

FFTW is a fast, portable library that should also work on any
platform.  You can download it from
`www.fftw.org <http://www.fftw.org>`_. The 3.X versions are supported
as -DFFT\_FFTW3.
Building FFTW for your box should be as simple as ./configure; make.

The FFT\_INC variable also allows for a -DFFT\_SINGLE setting that will
use single-precision FFTs, which can speed-up the calculation,
particularly in parallel or on GPUs.  Fourier transform operations
are somewhat insensitive to floating point truncation
errors and thus do not always need to be performed in double
precision.  Using the -DFFT\_SINGLE setting trades off a little
accuracy for reduced memory use and parallel communication costs for
transposing 3d FFT data.

**Step 7**

The 3 JPG variables allow you to specify a JPEG and/or PNG library
which SPARTA uses when writing out JPEG or PNG files via the :doc:`dump image <dump_image>` command. These can be left blank if you do not
use the -DSPARTA\_JPEG or -DSPARTA\_PNG switches discussed above in Step
4, since in that case JPEG/PNG output will be disabled.

A standard JPEG library usually goes by the name libjpeg.a or
libjpeg.so and has an associated header file jpeglib.h. Whichever JPEG
library you have on your platform, you'll need to set the appropriate
JPG\_INC, JPG\_PATH, and JPG\_LIB variables, so that the compiler and
linker can find it.

A standard PNG library usually goes by the name libpng.a or libpng.so
and has an associated header file png.h. Whichever PNG library you
have on your platform, you'll need to set the appropriate JPG\_INC,
JPG\_PATH, and JPG\_LIB variables, so that the compiler and linker can
find it.

As before, if these header and library files are in the usual place on
your machine, you may not need to set these variables.

**Step 8**

Note that by default none of the SPARTA optional packages are
installed.  To build SPARTA with optional packages, see :ref:`this section <start_3>` below, before proceeding to Step 9.

**Step 9**

That's it.  Once you have a correct Makefile.foo, and you have
pre-built any other needed libraries (e.g. MPI), all you need to do
from the src directory is type one of the following:


.. parsed-literal::

   make foo
   make -j N foo
   gmake foo
   gmake -j N foo

The -j or -j N switches perform a parallel build which can be much
faster, depending on how many cores your compilation machine has.  N
is the number of cores the build runs on.

You should get the executable spa\_foo when the build is complete.


----------


.. _start_2_2_2:

**Steps to build a SPARTA executable using CMake:** 

**Step 0**

Please review https://github.com/sparta/sparta/blob/master/BUILD\_CMAKE.md and ensure that
CMake version 3.12.0 or greater is installed:


.. parsed-literal::

   which cmake
   which cmake3
   cmake --version

On clusters and supercomputers one can use modules to load cmake:


.. parsed-literal::

   module avail cmake
   module load <CMAKE>

On Linux one may use apt, yum, or pacman to install cmake.

On Mac one may use brew or macports to install cmake.

**Step 1**

The cmake directory contains the CMake source files for SPARTA. Create a build
directory and from within the build directory, run cmake:


.. parsed-literal::

   mkdir build
   cd build
   cmake -LH -DSPARTA_MACHINE=tutorial /path/to/sparta/cmake

This will generate the default Makefiles and print the SPARTA CMake options. To
list the generated targets, do:


.. parsed-literal::

   make help

Now you can try to build the SPARTA binaries with:


.. parsed-literal::

   make

If everything works, an executable named spa\_tutorial and a library named
libsparta.a will be produced in build/src.

**Step 2**

If Step 1 did not work, see if you can use any system presets from
/path/to/sparta/cmake/presets. To select a preset:

cd build

# Clear the CMake files
rm -rf CMake\*


.. parsed-literal::

   cmake -C /path/to/sparta/cmake/presets/NAME.cmake -DSPARTA_MACHINE=tutorial /path/to/sparta/cmake
   make

**Step 3**

If Step 2 did not work, look at cmake -LH for a list of SPARTA CMake options and their
meaning, then modify one or more of those options by doing:


.. parsed-literal::

   cd build
   rm -rf CMake\*
   cmake -C /path/to/sparta/cmake/presets/NAME.cmake -D<OPTION_NAME>=<VALUE> /path/to/sparta/cmake
   make

where <OPTION\_NAME> and <VALUE> correspond to valid option value pairs listed by
cmake -LH. For the SPARTA\_DEFAULT\_CXX\_COMPILE\_FLAGS option, see Step 4.

For a full list of CMake option value pairs, see cmake -LAH. The most relevant
CMake options (with example values) for our purposes here are:

-DCMAKE\_C_COMPILER=gcc
-DCMAKE\_CXX\_COMPILER=/usr/local/bin/g++
-DCMAKE\_CXX\_FLAGS=-O3

If your cmake command line is getting too long, consider placing it in a bash
script and escaping newlines. For example:


.. parsed-literal::

   cmake -C /path/to/sparta/cmake/presets/NAME.cmake -D<OPTION_NAME>=<VALUE> /path/to/sparta/cmake

**Step 4**

The SPARTA\_DEFAULT\_CXX\_COMPILE\_FLAGS option passes flags to the compiler when
building object files.  Note that if you change any -D setting in this section,
you should do a full re-compile, after typing "make clean".

The SPARTA\_DEFAULT\_CXX\_COMPILE\_FLAGS option is typically used to include options
that turn on ifdefs within the SPARTA code.  The options that are currently recogized are:

* -DSPARTA\_GZIP
* -DSPARTA\_JPEG
* -DSPARTA\_PNG
* -DSPARTA\_FFMPEG
* -DSPARTA\_MAP
* -DSPARTA\_UNORDERED\_MAP
* -DSPARTA\_SMALL
* -DSPARTA\_BIG
* -DSPARTA\_BIGBIG
* -DSPARTA\_LONGLONG\_TO\_LONG

The read\_data and dump commands will read/write gzipped files if you
compile with -DSPARTA\_GZIP.  It requires that your Linux support the
"popen" command.

If you use -DSPARTA\_JPEG and/or -DSPARTA\_PNG, the :doc:`dump image <dump>` command will be able to write out JPEG and/or PNG
image files respectively. If not, it will only be able to write out
PPM image files.  For JPEG files, you must also link SPARTA with a
JPEG library, as described below.  For PNG files, you must also link
SPARTA with a PNG library, as described below.

If you use -DSPARTA\_FFMPEG, the :doc:`dump movie <dump_image>` command
will be available to support on-the-fly generation of rendered movies
the need to store intermediate image files.  It requires that your
machines supports the "popen" function in the standard runtime library
and that an FFmpeg executable can be found by SPARTA during the run.

If you use -DSPARTA\_MAP, SPARTA will use the STL map class for hash
tables.  This is less efficient than the unordered map class which is
not yet supported by all C++ compilers.  If you use
-DSPARTA\_UNORDERED\_MAP, SPARTA will use the unordered\_map class for
hash tables and will assume it is part of the STL (e.g. this works for
Clang++).  The default is to use the unordered map class from the
"tri1" extension to the STL which is supported by most compilers.  So
only use either of these options if the build complains that unordered
maps are not recognized.

Use at most one of the -DSPARTA\_SMALL, -DSPARTA\_BIG, -DSPARTA\_BIGBIG
settings.  The default is -DSPARTA\_BIG.  These refer to use of 4-byte
(small) vs 8-byte (big) integers within SPARTA, as described in
src/spatype.h.  The only reason to use the BIGBIG setting is if you
have a regular grid with more than ~2 billion grid cells or a
hierarchical grid with enough levels that grid cell IDs cannot fit in
a 32-bit integer.  In either case, SPARTA will generate an error
message for "Cell ID has too many bits".  See :ref:`Section 4.8 <howto_8>` of the manual for details on how cell
IDs are formatted.  The only reason to use the SMALL setting is if
your machine does not support 64-bit integers.

In all cases, the size of problem that can be run on a per-processor
basis is limited by 4-byte integer storage to about 2 billion
particles per processor (2\^31), which should not normally be a
restriction since such a problem would have a huge per-processor
memory and would run very slowly in terms of CPU secs/timestep.

The -DSPARTA\_LONGLONG\_TO\_LONG setting may be needed if your system or
MPI version does not recognize "long long" data types.  In this case a
"long" data type is likely already 64-bits, in which case this setting
will use that data type.

Using one of the -DPACK\_ARRAY, -DPACK\_POINTER, and -DPACK\_MEMCPY
options can make for faster parallel FFTs on some platforms.  The
-DPACK\_ARRAY setting is the default.  See the :doc:`compute fft/grid <compute_fft_grid>` command for info about FFTs.  See STEP
7 below for info about building SPARTA with an FFT library.

**Step 5**

This step is optional. Once you get Steps 3 and 4 working by modifying the
options to the cmake command, try setting the same options in
/path/to/sparta/cmake/presets/NEW.cmake by copying 
/path/to/sparta/cmake/presets/NAME.cmake and modifying the cmake
source code. Note that the CMake cache is sticky and will only evict a 
cached option value pair if you use -D or the FORCE argument to CMake's set
routine.

Now just do:


.. parsed-literal::

   cd build
   rm -rf CMake\*
   cmake -C /path/to/sparta/cmake/presets/NEW.cmake /path/to/sparta/cmake
   make

consider sharing and vetting NEW.cmake by opening a pull request at
https://github.com/sparta/sparta/.

**Step 6**

This step explains how to enable and select MPI in the SPARTA CMake
configuration. There may already be a preset in 
/path/to/sparta/cmake/presets that selects the correct MPI installation.

By default, SPARTA configures with MPI enabled and cmake will print which MPI
was selected. To build serial binaries, use SPARTA's MPI\_STUBS package:


.. parsed-literal::

   cmake -DPKG_MPI_STUBS=ON /path/to/sparta/cmake

You may want a different MPI installation than CMake finds. CMake uses module
files such as FindMPI.cmake to handle wiring in a given installation of a 
library and its headers. If you're on a cluster or supercomputer, use module 
before running cmake so that cmake finds the MPI installation you'd like to
use:

# Show which modules are loaded
module list

# Show which modules are available
module avail


.. parsed-literal::

   module load <MPI>

On Linux one may use apt, yum, or pacman to install MPI.

On Mac one may use brew or macports to install MPI.

Verify that cmake found the correct MPI installation:

cd build
rm -rf CMake\*


.. parsed-literal::

   # cmake should print "Found MPI\*" strings
   cmake **options** /path/to/sparta/cmake

Note that if the preset file you're using enables PKG\_MPI\_STUBS, MPI will not be
searched for unless you explicitly disable PKG\_MPI\_STUBS in the preset file.

If you'd like to use a custom MPI installation or cmake is not locating the MPI
installation you've selected via the module command or package manager, try
export MPI\_ROOT=/path/to/mpi/install before running cmake. Otherwise, please see
https://cmake.org/cmake/help/v3.12/module/FindMPI.html#variables-for-locating-mpi.
Note that this documentation link is for CMake version 3.12.

**Step 7**

When the SPARTA FFT package is enabled with cmake -DPKG\_FFT=ON, you may select
between 3 thiry party libraries (TPLs) for 1d FFTs, which SPARTA uses when
configured with cmake -DFFT=\ *FFTW3,MKL,KISS*\ .

By default SPARTA will use the open-source `KISS FFT library <http://kissfft.sf.net>`_, which is included in the SPARTA distribution.
This library is portable to all platforms and for typical SPARTA simulations is
almost as fast as FFTW or vendor optimized libraries.

Similarly when using the KOKKOS package, you may select between 5 TPLs for FFT
which SPARTA uses when configured with cmake
-DFFT\_KOKKOS=\ *CUFFT,HIPFFT,FFTW3,MKL,KISS*\ . This requires enabling the SPARTA
FFT package which can be selected with cmake -DPKG\_FFT=ON.

By default, SPARTA will use a Kokkos version of the open-source `KISS FFT library <http://kissfft.sf.net>`_, which is included in the SPARTA distribution.
Note that using the KISS FFT library on GPUs may give suboptimal performance.
Other options for -DFFT\_KOKKOS are CUFFT, HIPFFT, MKL or FFTW3. When using the
Kokkos CUDA backend, either CUFFT or KISS must be used. When using the Kokkos
HIP backend, either HIPFFT or KISS must be used. When using the Kokkos OpenMP
or Serial backend, either MKL, FFTW3, or KISS must be used. The CUFFT option
specifies the `cuFFT library <https://developer.nvidia.com/cufft>`_ from NVIDIA.
The HIPFFT option specifies the `rocFFT library <https://rocm.docs.amd.com/projects/rocFFT/en/latest/>`_ from AMD.

You may need to install the FFT TPL you're interested in using. If you're on a
cluster or supercomputer, use module before running cmake so that cmake finds
the FFT installation you'd like to use:

# Show which modules are loaded
module list

# Show which modules are available
module avail


.. parsed-literal::

   module load <FFT>

On Linux one may use apt, yum, or pacman to install FFT.

On Mac one may use brew or macports to install FFT.

Verify that cmake found the correct MPI installation:

cd build
rm -rf CMake\*


.. parsed-literal::

   # cmake should print "Found FFT\*" strings
   cmake **options** /path/to/sparta/cmake

Note that if the preset file you're using enables PKG\_FFT, FFT will not be
searched for unless you explicitly disable PKG\_FFT in the preset file.

If you'd like to use a custom FFT installation or cmake is not locating the FFT
installation you've selected via the module command or package manager, try
export FFT\_ROOT=/path/to/fft/install before running cmake. Otherwise, please
open an issue at https://github.com/sparta/sparta/issues.

**Step 8**

You may select between 2 TPLs, JPEG or PNG, for writing out JPEG or PNG files
via the :doc:`dump image <dump_image>` command. To select a TPL, use:


.. parsed-literal::

   cmake -DBUILD_JPEG=ON /path/to/sparta/cmake

or:


.. parsed-literal::

   cmake -DBUILD_PNG=ON /path/to/sparta/cmake

If you'd like to use a custom jpeg or png installation, please see 
https://cmake.org/cmake/help/v3.12/module/FindJPEG.html or
https://cmake.org/cmake/help/v3.12/module/FindPNG.html. Note that these
documentation links are for CMake version 3.12.

**Step 9**

By default, none of the SPARTA optional packages are installed. To build SPARTA
with optional packages, use:


.. parsed-literal::

   cmake -DPKG_XXX=ON /path/to/sparta/cmake

Where XXX is the package to enable. For a full list of optional packages, see:


.. parsed-literal::

   cmake -LH /path/to/sparta/cmake

**Step 10**

Once you have a correct cmake command line or the NAME.cmake preset file, just
do:


.. parsed-literal::

   cd build
   cmake **OPTIONS** /path/to/sparta/cmake

or:

cd build
cmake -C /path/to/sparta/cmake/presets/NAME.cmake -DSPARTA\_MACHINE=tutorial /path/to/sparta/cmake


.. parsed-literal::

   make -j N

The -j or -j N switches perform a parallel build which can be much faster, 
depending on how many cores your compilation machine has. N is the number of
cores the build runs on.

You should get build/src/spa\_tutorial and build/src/libsparta.a.

**Building with the KOKKOS package:**

The KOKKOS package must be built with CMake (GNU Makefile builds are not
supported for KOKKOS). The presets in cmake/presets/ set up all required
Kokkos options for each supported backend and target platform. Choose the
preset that matches your hardware:

* kokkos\_omp.cmake      - Multi-core CPUs via OpenMP threading
* kokkos\_mpi\_only.cmake - Multi-core CPUs, MPI only (no threading)
* kokkos\_cuda.cmake     - NVIDIA GPUs via CUDA (default: Hopper/H100)
* kokkos\_hip.cmake      - AMD GPUs via HIP (default: MI250X)
* elcapitan\_kokkos.cmake - AMD MI300A APU via HIP (Cray MPICH)
* kokkos\_sycl.cmake     - Intel Ponte Vecchio GPUs via SYCL

Example: build for multi-core CPUs with OpenMP threading:


.. parsed-literal::

   mkdir build
   cd build
   cmake -C /path/to/sparta/cmake/presets/kokkos_omp.cmake /path/to/sparta/cmake
   make -j 4

The executable will be named spa\_kokkos\_omp (the suffix comes from the preset).

Example: build for NVIDIA A100 GPUs (override the default Hopper arch):


.. parsed-literal::

   mkdir build
   cd build
   cmake -C /path/to/sparta/cmake/presets/kokkos_cuda.cmake       -DKokkos_ARCH_HOPPER90=OFF -DKokkos_ARCH_AMPERE80=ON       /path/to/sparta/cmake
   make -j 4

Example: build for NVIDIA V100 GPUs:


.. parsed-literal::

   mkdir build
   cd build
   cmake -C /path/to/sparta/cmake/presets/kokkos_cuda.cmake       -DKokkos_ARCH_HOPPER90=OFF -DKokkos_ARCH_VOLTA70=ON       /path/to/sparta/cmake
   make -j 4

Example: build for AMD MI300X GPUs:


.. parsed-literal::

   mkdir build
   cd build
   cmake -C /path/to/sparta/cmake/presets/kokkos_hip.cmake       -DKokkos_ARCH_VEGA90A=OFF -DKokkos_ARCH_AMD_GFX942=ON       /path/to/sparta/cmake
   make -j 4

After building, run SPARTA with the KOKKOS package using the -k, -sf,
and -pk command-line switches. The -k on g Ng switch specifies Ng GPUs
per node. Examples (assuming a node with 4 GPUs):

For CUDA (NVIDIA) or HIP (AMD) GPUs, 4 MPI tasks using 4 GPUs:


.. parsed-literal::

   mpirun -np 4 spa_kokkos_cuda -k on g 4 -sf kk -in in.collide

For OpenMP on 16-core CPUs, 2 MPI tasks each using 8 threads:


.. parsed-literal::

   mpirun -np 2 spa_kokkos_omp -k on t 8 -sf kk -in in.collide

If you are migrating from the GNU Makefile build system, the Kokkos
KOKKOS\_DEVICES and KOKKOS\_ARCH Makefile variables map to CMake options as
follows:

* KOKKOS\_DEVICES=OpenMP  ->  -DKokkos\_ENABLE\_OPENMP=ON
* KOKKOS\_DEVICES=Cuda    ->  -DKokkos\_ENABLE\_CUDA=ON
* KOKKOS\_ARCH=Volta70    ->  -DKokkos\_ARCH\_VOLTA70=ON
* KOKKOS\_ARCH=Ampere80   ->  -DKokkos\_ARCH\_AMPERE80=ON
* KOKKOS\_ARCH=Hopper90   ->  -DKokkos\_ARCH\_HOPPER90=ON
* KOKKOS\_ARCH=Power9     ->  -DKokkos\_ARCH\_POWER9=ON
* KOKKOS\_ARCH=ARMv8-TX2  ->  -DKokkos\_ARCH\_ARMV8\_THUNDERX2=ON

See :ref:`Section 5.3 <acc_3>` for the full list of
architecture options and detailed instructions for each KOKKOS backend.


----------


.. _start_2_3:

**Errors that can occur when making SPARTA:** 

.. warning::

   If an error occurs when building SPARTA, the compiler
   or linker will state very explicitly what the problem is.  The error
   message should give you a hint as to which of the steps above has
   failed, and what you need to do in order to fix it.  Building a code
   with a Makefile is a very logical process.  The compiler and linker
   need to find the appropriate files and those files need to be
   compatible with SPARTA source files.  When a make fails, there is
   usually a very simple reason, which you or a local expert will need to
   fix.

Here are two non-obvious errors that can occur:

(1) If the make command breaks immediately with errors that indicate
it can't find files with a "\*" in their names, this can be because
your machine's native make doesn't support wildcard expansion in a
makefile.  Try gmake instead of make.  If that doesn't work, try using
a -f switch with your make command to use a pre-generated
Makefile.list which explicitly lists all the needed files, e.g.


.. parsed-literal::

   make makelist
   make -f Makefile.list g++
   gmake -f Makefile.list mac

The first "make" command will create a current Makefile.list with all
the file names in your src dir.  The 2nd "make" command (make or
gmake) will use it to build SPARTA.

(2) If you get an error that says something like 'identifier "atoll"
is undefined', then your machine does not support "long long"
integers.  Try using the -DSPARTA\_LONGLONG\_TO\_LONG setting described
above in Step 4.


----------


.. _start_2_4_1:

**Additional build tips using make:** 

(1) Building SPARTA for multiple platforms.

You can make SPARTA for multiple platforms from the same src
directory.  Each target creates its own object sub-directory called
Obj\_name where it stores the system-specific \*.o files.

(2) Cleaning up.

Typing "make clean-all" or "make clean-foo" will delete \*.o object
files created when SPARTA is built, for either all builds or for a
particular machine.


----------


.. _start_2_4_2:

**Additional build tips using CMake:** 

(1) Building SPARTA for multiple platforms.

It's best to build SPARTA for multiple platforms from different
build directories. However, each target creates its own spa\_TARGET binary and
multiple targets can be built from the same build directory. Note that the \*.o
object files in build/src will reflective of the most recent build
configuration. Also note that if BUILD\_SHARED\_LIBS was enabled,
libsparta will be reflective of the most recent build configuration.

(2) Cleaning up.

Typing "make clean" will delete all binary files for the most recent build
configuration.


----------


.. _start_2_5:

**Building for a Mac:** 

OS X is BSD Unix, so it should just work.  See the Makefile.mac or
cmake/presets/mac.cmake file.


----------


.. _start_2_6:

**Building for Windows:** 

At some point we may provide a pre-built Windows executable
for SPARTA.  Until then you will need to build an executable from 
source files.

One way to do this is install and use cygwin to build SPARTA with a
standard Linux make or CMake, just as you would on any Linux box.

You can also import the \*.cpp and \*.h files into Microsoft Visual
Studio.  If someone does this and wants to provide project files or
other Windows build tips, please send them to the
`developers <https://sparta.github.io/authors.html>`_ and we will include
them in the distribution.
