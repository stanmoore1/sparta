.. _start_3:

Making SPARTA with optional packages
====================================

This section has the following sub-sections:

:ref:`Package basics <start_3_1>`
:ref:`Including/excluding packages with make <start_3_2_1>`
:ref:`Including/excluding packages with CMake <start_3_2_2>`


----------


.. _start_3_1:

**Package basics:** 

The source code for SPARTA is structured as a set of core files which
are always included, plus optional packages.  Packages are groups of
files that enable a specific set of features.  For example, the FFT
package which includes a :doc:`compute fft/grid <compute_fft_grid>`
command and a 2d and 3d FFT library.

For make:
You can see the list of all packages by typing "make package" from
within the src directory of the SPARTA distribution. This also lists
various make commands that can be used to manipulate packages.

For CMake:
You can see the list of all packages by typing "cmake -DSPARTA\_LIST\_PKGS=ON"
from within the build directory.

If you use a command in a SPARTA input script that is part of a
package, you must have built SPARTA with that package, else you will
get an error that the style is invalid or the command is unknown.
Every command's doc page specfies if it is part of a package.


----------


.. _start_3_2_1:

**Including/excluding packages with make:** 

To use (or not use) a package you must include it (or exclude it)
before building SPARTA.  From the src directory, this is typically as
simple as:


.. parsed-literal::

   make yes-fft
   make g++

or


.. parsed-literal::

   make no-fft
   make g++

.. note::

   You should NOT include/exclude packages and build SPARTA in a
   single make command using multiple targets, e.g. make yes-fft g++.
   This is because the make procedure creates a list of source files that
   will be out-of-date for the build if the package configuration changes
   within the same command.

Some packages have individual files that depend on other packages
being included.  SPARTA checks for this and does the right thing.
I.e. individual files are only included if their dependencies are
already included.  Likewise, if a package is excluded, other files
dependent on that package are also excluded.

If you will never run simulations that use the features in a
particular packages, there is no reason to include it in your build.

When you download a SPARTA tarball, no packages are pre-installed in
the src directory.

Packages are included or excluded by typing "make yes-name" or "make
no-name", where "name" is the name of the package in lower-case, e.g.
name = fft for the FFT package.  You can also type "make yes-all", or
"make no-all" to include/exclude all packages.  Type "make package" to
see all of the package-related make options.

.. note::

   Inclusion/exclusion of a package works by simply moving files
   back and forth between the main src directory and sub-directories with
   the package name (e.g. src/FFT or src/KOKKOS), so that the files are
   seen or not seen when SPARTA is built.  After you have included or
   excluded a package, you must re-build SPARTA.

Additional package-related make options exist to help manage SPARTA
files that exist in both the src directory and in package
sub-directories.  You do not normally need to use these commands
unless you are editing SPARTA files.

Typing "make package-update" or "make pu" will overwrite src files
with files from the package sub-directories if the package has been
included.  It should be used after a patch is installed, since patches
only update the files in the package sub-directory, but not the src
files.  Typing "make package-overwrite" will overwrite files in the
package sub-directories with src files.

Typing "make package-status" or "make ps" will show which packages are
currently included. For those that are included, it will list any
files that are different in the src directory and package
sub-directory.  Typing "make package-diff" lists all differences
between these files.  Again, type "make package" to see all of the
package-related make options.

Typing "make package-installed" or "make pi" will show which packages are
currently installed in the src directory.


----------


.. _start_3_2_2:

**Including/excluding packages with CMake:** 

To use (or not use) a package you must include it (or exclude it)
before building SPARTA.  From the build directory, do:


.. parsed-literal::

   cmake -DPKG_FFT=ON /path/to/sparta/cmake
   make -j

or


.. parsed-literal::

   cmake -DPKG_FFT=OFF /path/to/sparta/cmake
   make -j

Some packages have individual files that depend on other packages
being included.  SPARTA checks for this and does the right thing.
I.e. individual files are only included if their dependencies are
already included.  Likewise, if a package is excluded, other files
dependent on that package are also excluded.

If you will never run simulations that use the features in a
particular packages, there is no reason to include it in your build.

When you download a SPARTA tarball, no packages are pre-installed in
the build/src directory.

Packages are included or excluded by typing "cmake -DPKG\_NAME=ON" or 
"cmake -DPKG\_NAME=OFF", where "NAME" is the name of the package in upper-case, 
e.g. name = FFT for the FFT package. You can also type "cmake
-DSPARTA\_ENABLE\_ALL\_PKGS=ON", or "cmake -DSPARTA\_DISABLE\_ALL\_PKGS=ON" to 
include or exclude all packages. Type "cmake -DSPARTA\_LIST\_PKGS=ON" to
see all of the package-related CMake options.

.. note::

   Inclusion or exclusion of a package works by setting CMake boolean
   variables to generate the correct Makefile targets and dependencies. After you
   have included or excluded a package, you must re-build SPARTA.

If a SPARTA package has source code changes, simply run "make" to rebuild SPARTA
with these changes.

Typing "cmake" from the build directory will show which packages are currently
included.
