************
Installation
************

.. index:: installation
.. index:: compilation

SPARTA-GUI is distributed as source code in the ``tools/sparta-gui``
folder of the `SPARTA source distribution
<https://github.com/sparta/sparta>`_ and is compiled from source.  A
short version of the build instructions is also in the
``tools/sparta-gui/README.md`` file; the instructions below are more
detailed but describe the same procedure.

There are two ways to connect SPARTA-GUI to SPARTA:

- **Plugin mode** (the default): SPARTA-GUI loads the SPARTA shared
  library (``libsparta.so`` on Linux, ``libsparta.dylib`` on macOS)
  dynamically at runtime.  The GUI does not need to be recompiled when
  SPARTA is updated, and it can be pointed at different SPARTA builds
  from the *Preferences* dialog.
- **Linked mode**: SPARTA-GUI is linked against the SPARTA library at
  compile time.

.. admonition:: Minimum SPARTA version

   SPARTA-GUI requires features of the SPARTA library interface that
   were added for it (e.g. the timeout mechanism used to stop runs
   cleanly, exception-based error handling, and style queries used for
   auto-completion).  The minimum required SPARTA version is
   **24 Sep 2025**.  SPARTA-GUI will print a suitable error message
   when an incompatible SPARTA library is loaded.

Prerequisites and portability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. index:: prerequisites
.. index:: Qt framework
.. index:: CMake

SPARTA-GUI is programmed in C++ based on the C++17 standard and using
the `Qt GUI framework <https://www.qt.io/development/framework>`_.  Qt
version 6.2 or later is *required*, including the Widgets, Gui,
Network, and Svg modules.  SPARTA-GUI can switch between a "light" and
a "dark" theme according to the settings of the desktop environment.
Building SPARTA-GUI from source requires CMake version 3.20 or later
and a suitable C++ compiler.

On Linux distributions the Qt modules are often packaged separately
from the Qt base libraries.  For example on Ubuntu / Debian the
required development packages are ``qt6-base-dev`` and ``qt6-svg-dev``
(the latter provides the SVG icon engine plugin; without it the
toolbar and menu icons render blank).  On Fedora the corresponding
packages are ``qt6-qtbase-devel`` and ``qt6-qtsvg-devel``.  On macOS,
Qt can be installed with `Homebrew <https://brew.sh/>`_ (``brew
install qt``).

The charts display is drawn by a self-contained native renderer
(:cpp:class:`PlotWidget`) built only on Qt Widgets and ``QPainter``, so
the build does not depend on the Qt Charts or Qt Graphs modules.

Building the SPARTA shared library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. index:: shared library
.. index:: libsparta

SPARTA-GUI runs SPARTA in a thread inside the GUI process, so it needs
SPARTA compiled as a **serial** (non-MPI) **shared** library.  Using
the `CMake build procedure of SPARTA
<https://sparta.github.io/doc/Section_start.html>`_ from the top-level
folder of the SPARTA distribution:

.. code-block:: bash

   cmake -S cmake -B build-lib \
         -D BUILD_SHARED_LIBS=ON \
         -D BUILD_MPI=OFF \
         -D BUILD_PNG=ON -D BUILD_JPEG=ON \
         -D CMAKE_BUILD_TYPE=Release
   cmake --build build-lib

This produces ``build-lib/src/libsparta.so`` (or ``libsparta.dylib`` on
macOS).  Enabling PNG and JPEG support is strongly recommended, since
the snapshot images and the `dump image
<https://sparta.github.io/doc/dump_image.html>`_ output are rendered by
SPARTA itself.  To also enable the `dump movie
<https://sparta.github.io/doc/dump_image.html>`_ command (SPARTA piping
images directly into `FFmpeg <https://ffmpeg.org/>`_), additionally
compile SPARTA with the ``SPARTA_FFMPEG`` define (e.g. by adding
``-DSPARTA_FFMPEG`` to the compiler flags).  To enable the KOKKOS
accelerator package (OpenMP threading, selectable in the SPARTA-GUI
*Preferences*), add ``-D PKG_KOKKOS=ON`` and the appropriate Kokkos
settings as described in the SPARTA manual.

.. admonition:: MPI parallelization
   :class: note

   The design decisions for SPARTA-GUI and how it launches SPARTA
   conflict with parallel runs using MPI, therefore the library must be
   compiled without MPI.  For parallel production runs you have to use
   a regular SPARTA executable compiled with MPI support.  For the use
   cases that SPARTA-GUI has been conceived for (learning SPARTA,
   testing or debugging SPARTA inputs, prototyping new projects), this
   is not a significant limitation.

Building SPARTA-GUI
^^^^^^^^^^^^^^^^^^^

.. index:: compilation; from source
.. index:: CMake configuration
.. index:: plugin mode

SPARTA-GUI plugin version (default)
-----------------------------------

The default configuration compiles SPARTA-GUI with a `plugin loader
<https://github.com/sparta/sparta>`_ that loads the SPARTA shared
library file dynamically at runtime during the start of the GUI.  This
has the advantage that the SPARTA library can be rebuilt from updated
or modified SPARTA sources without having to (re-)compile the GUI.
From the top-level folder of the SPARTA distribution:

.. code-block:: bash

   cmake -S tools/sparta-gui -B build-gui -D CMAKE_BUILD_TYPE=Release
   cmake --build build-gui
   ./build-gui/sparta-gui

If the Qt library is installed as packaged for Linux distributions,
its location is typically auto-detected.  Otherwise, the location of
the Qt installation must be indicated by setting ``-D
Qt6_DIR=/path/to/qt6/lib/cmake/Qt6``, which is a path to a folder
inside the Qt installation that contains the file ``Qt6Config.cmake``.

To build the optional :ref:`interactive VTK 3D viewer <vtk_viewer>`, add
``-D SPARTA_GUI_USE_VTK=ON`` and make a `VTK <https://vtk.org/>`_ library
with development headers available (e.g. ``libvtk9-dev`` on Debian/Ubuntu,
or ``-D VTK_DIR=/path/to/vtk/lib/cmake/vtk-<ver>`` for a custom build).
VTK's Qt integration is *not* required — the viewer renders off-screen —
so a VTK built with or without Qt (and with either Qt5 or Qt6) works.  If
``SPARTA_GUI_USE_VTK=ON`` is set but no suitable VTK is found, the build
still succeeds with the viewer disabled (a status message says so).  The
viewer reads the files written by SPARTA's ``dump particle/vtk`` /
``grid/vtk`` / ``surf/vtk`` styles, which require SPARTA itself to be built
with its VTK package (``-D PKG_VTK=ON`` when building the library).

.. _first_start:

The first start
---------------

.. index:: shared library; first start

SPARTA-GUI looks for the SPARTA shared library by itself: the path it
last used, then the current directory, the dynamic loader's search
path, its own configuration folder, and the usual system library
directories.  Most installations never see anything else.

When none of those turns one up, the application still starts.  Writing,
opening, highlighting, checking and saving an input deck need no
simulator, and a strip above the editor says what is missing and offers
two ways to fix it:

- **Download** fetches the pre-compiled library for this platform from
  the SPARTA webserver into the configuration folder.  It is offered
  only where such a library exists; a build that cannot use one (for
  example an MSVC build, since the pre-compiled libraries are built
  with MinGW) does not show the button.
- **Browse...** picks a library already on this computer.  A file whose
  name does not contain ``libsparta`` gets a question rather than a
  silent refusal -- the name is a heuristic and you may overrule it, but
  whether the file loads is the real test.

Either way the library is adopted immediately: the strip goes away, the
run controls come to life, and the choice is remembered for next time.
Nothing restarts.

The **Run** entries stay greyed out until a library is loaded, which is
what makes the strip's claim visible rather than something to take on
faith.

There are three other ways to say where the library is:

- set the path in the *Preferences* dialog ("Path to SPARTA Shared
  Library File"),
- start the GUI with the ``-p <path>`` / ``--pluginpath <path>``
  command-line flag, or
- set the ``SPARTA_PLUGIN_PATH`` environment variable to the folder
  containing ``libsparta.so``.

The setting is stored persistently.  An empty path ("") as argument to
``-p`` restores the default (auto-detection) behavior; this also lets
you recover in case the configured library file no longer exists.
Changing the library from *Preferences* relaunches the application,
because a library that is already loaded cannot be swapped in place.

.. note::

   Setting ``SPARTA_GUI_FORCE_NO_PLUGIN=1`` in the environment makes
   SPARTA-GUI behave as if no library could be found, whatever is
   installed.  It exists so the first-start experience can be exercised
   on a machine that has a working library.

SPARTA-GUI linked version
-------------------------

It is also possible to link SPARTA-GUI to the SPARTA library directly
at compile time.  This is enabled by setting ``-D
SPARTA_GUI_USE_PLUGIN=OFF`` (default setting is ``ON``).  In this
case, the CMake configuration needs to be told where to find the
SPARTA headers and the SPARTA library:

.. code-block:: bash

   cmake -S tools/sparta-gui -B build-gui \
         -D SPARTA_GUI_USE_PLUGIN=OFF \
         -D SPARTA_SOURCE_DIR=$PWD/src \
         -D SPARTA_LIBRARY=$PWD/build-lib/src/libsparta.so
   cmake --build build-gui

When linked to a shared SPARTA library, it may be necessary to adjust
environment variables so it is found at runtime (``LD_LIBRARY_PATH``
on Linux, ``DYLD_LIBRARY_PATH`` on macOS).

Building this documentation
---------------------------

The HTML documentation you are reading can be built by adding ``-D
BUILD_DOC=ON`` to the SPARTA-GUI CMake configuration (or ``-D
BUILD_DOC_ONLY=ON`` to build just the documentation without the
application).  This creates a Python virtual environment, installs the
required Sphinx packages from ``doc/requirements.txt``, and provides
the build targets ``html``, ``spelling``, ``linkcheck``, and ``pdf``.
When `Doxygen <https://doxygen.nl/>`_ is installed, the API reference
section of the Programmer's Guide is extracted from the C++ sources;
otherwise that page is skipped.

Platform notes
--------------

.. index:: platform notes
.. index:: macOS installation

macOS
"""""

On macOS the convenience script ``tools/sparta-gui/build-macos.sh``
builds both the SPARTA serial shared library and the SPARTA-GUI
application bundle in one step:

.. code-block:: bash

   brew install cmake qt libpng jpeg
   ./tools/sparta-gui/build-macos.sh
   open build-sparta-gui-macos/sparta-gui.app

The script uses Homebrew's Qt and configures the app bundle so that it
finds the freshly built ``libsparta.dylib``.  Building manually with
the CMake commands shown above also works; when building an app bundle
you can create a drag-n-drop disk image with the 'dmg' target
(``cmake --build build-gui --target dmg``).

Linux
"""""

Version 6.2 or later of the Qt library is required.  Those are
provided by, e.g., Ubuntu 22.04LTS or later and all current Fedora
releases.  After installing the Qt development packages listed above,
the standard build procedure applies.  The compiled ``sparta-gui``
binary can be run directly from the build folder or installed with
``cmake --install build-gui``.
