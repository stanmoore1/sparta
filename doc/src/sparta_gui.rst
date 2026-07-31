SPARTA-GUI
==========

SPARTA-GUI is a graphical desktop application for editing, running,
visualizing, and exploring SPARTA simulations.  It is a text editor
customized for SPARTA input scripts which is linked to the SPARTA
library, so it can run SPARTA directly from the contents of the editor
buffer and retrieve and display information from SPARTA while it is
running.  It runs on Linux and macOS and is aimed both at beginners,
who get a single ready-to-use program for most tasks, and at
experienced users for quickly prototyping, debugging, and visualizing
simulation setups.

SPARTA-GUI is a derived work of LAMMPS-GUI by Axel Kohlmeyer, see the
attribution note below.

Features
--------

* Editor: syntax highlighting and reformatting for SPARTA input scripts,
  line numbers, undo/redo, find-and-replace, and context-sensitive
  auto-completion of SPARTA commands and styles.  The lists of available
  compute, fix, dump, region, collide, react, surf_collide, and
  surf_react styles are queried from the SPARTA library, so the
  completions always match the capabilities of the SPARTA version in
  use.  Pressing Ctrl-? (or right-clicking) on a command opens its page
  in the online SPARTA documentation.

* Running: SPARTA runs in a separate thread inside the GUI, started and
  stopped with a mouse click or hotkey.  The screen output is captured
  in an Output window with warnings and errors highlighted, a progress
  bar tracks the current :doc:`run <run>` command, and CPU utilization is
  displayed.  Stopping a run uses SPARTA's timeout mechanism, so the run
  ends cleanly at the next timestep.  Errors from SPARTA are shown in a
  dialog with the offending input line highlighted, and the GUI survives
  input errors, so scripts can be corrected and re-run right away.

* Charts: the columns of the :doc:`stats output <stats_style>` are
  captured during a run and plotted live as line graphs, with optional
  logarithmic axes, smoothing (Savitzky-Golay), curve fitting
  (Levenberg-Marquardt, with user-supplied expressions), custom function
  overlays, and export of the data to CSV, plain text, or YAML files.

* Image viewer: interactive snapshots of the current simulation state
  rendered by SPARTA's :doc:`dump image <dump_image>` facility, with full
  support for its options: particles (selected by mixture, colored and
  sized by type, processor, or per-particle attributes and compute/fix
  output), grid cells rendered as volumes or as x/y/z cut planes
  (colored by processor or per-grid compute/fix output), surface
  elements (colored by a constant color, processor, or per-surf
  compute/fix output), box, sub-box, and axes display, camera controls,
  SSAO depth shading, anti-aliasing, background gradients, adjustable
  lights, and six independent color maps.  A "Copy dump image command"
  action exports the current visualization as a ready-to-use
  :doc:`dump image <dump_image>` command for use in input scripts.

* Slide show: images produced by a :doc:`dump image <dump_image>` command
  in the input are displayed as they are created during a run, with
  playback controls and export of the image sequence to a movie file via
  FFmpeg.

* Input checking: the deck is validated without running SPARTA, flagging
  unknown commands and styles, wrong argument counts, undefined
  variable/compute/fix references and missing files, marked inline in the
  editor.

* Parametric sweeps: a series of runs over a range of index-variable
  values, driven from a docked panel, with the results collected for
  comparison.

* Also: a dialog to set index variables (like the -var command-line
  flag), continuing a finished run by more steps, writing and inspecting
  binary :doc:`restart files <read_restart>`, exporting surfaces and grids
  for ParaView, importing STL surfaces, a run history, quick access to
  the inputs of the bundled examples tree, and preferences including the
  KOKKOS accelerator package and thread count.

Location and building
---------------------

The SPARTA-GUI source code is in the tools/sparta-gui directory of the
SPARTA distribution.  It is written in C++ and requires CMake (3.20 or
later), a C++17 compiler, and the Qt framework version 6.2 or later.

Building is a two-step process: first SPARTA is compiled as a serial
shared library (with CMake, using -D BUILD_SHARED_LIBS=ON and -D
BUILD_MPI=OFF), then the GUI is compiled from the tools/sparta-gui
folder.  By default the GUI loads the SPARTA shared library at runtime
(plugin mode), so SPARTA can be updated or reconfigured without
recompiling the GUI; it can also be linked to the SPARTA library
directly.  On macOS the script tools/sparta-gui/build-macos.sh builds
both the library and the application bundle in one step.

Short build instructions are in the tools/sparta-gui/README.md file.
The full documentation of SPARTA-GUI, including detailed build
instructions for all supported configurations, is a Sphinx manual in
tools/sparta-gui/doc which can be built by adding -D BUILD_DOC=ON to
the SPARTA-GUI CMake configuration.

Attribution and license
-----------------------

SPARTA-GUI is a derived work of `LAMMPS-GUI
<https://github.com/akohlmey/lammps-gui>`_ version 3.0.3 by Axel
Kohlmeyer, adapted for SPARTA.  LAMMPS-GUI is Copyright (c) 2023-2026
Axel Kohlmeyer and is distributed under the GNU General Public License
version 2 (GPL-2.0-or-later).  SPARTA-GUI as a whole, like SPARTA
itself, is distributed under the terms of the GNU General Public
License version 2; see the file tools/sparta-gui/LICENSE.  If you use
SPARTA-GUI in your work, please cite SPARTA and acknowledge LAMMPS-GUI
by Axel Kohlmeyer, on which SPARTA-GUI is based.
