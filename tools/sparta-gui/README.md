# SPARTA-GUI

SPARTA-GUI is a graphical desktop tool to edit, run, visualize, and
explore simulations with the [SPARTA DSMC
code](https://sparta.github.io) — Stochastic PArallel Rarefied-gas
Time-accurate Analyzer.

It provides:

- an editor for SPARTA input scripts with syntax highlighting,
  context-sensitive auto-completion (commands and styles are queried
  from the running SPARTA library), inline warnings, and direct links
  to the online documentation of the command under the cursor,
- running simulations directly from the editor with live capture of
  the screen output, a progress bar, CPU utilization display, and a
  Stop button that interrupts a run cleanly at the next timestep,
- live charts of the `stats` output while the simulation runs,
  including log scales, data export, smoothing, and curve fitting,
- a snapshot Image Viewer built on SPARTA's `dump image` command with
  full support for rendering particles, grid cells, grid cut planes,
  and surface elements, including color maps, SSAO depth shading,
  anti-aliasing, background gradients, and adjustable lighting,
- a slideshow window that shows images as a running `dump image`
  command produces them, and movie export via ffmpeg,
- a variable editor (equivalent of SPARTA's `-var` command line
  flag), preferences with Kokkos accelerator support, and restart
  file inspection.

SPARTA-GUI loads the SPARTA shared library (`libsparta.so` /
`libsparta.dylib`) at runtime via a plugin interface, so the GUI does
not need to be recompiled when SPARTA is updated.

## Provenance and license

SPARTA-GUI is a derived work of
[LAMMPS-GUI](https://github.com/akohlmey/lammps-gui) version 3.0.3
(commit `b6cfbea18a6b46d68de9457ceca9bf09e2946896`),

> Copyright (c) 2023, 2024, 2025, 2026 Axel Kohlmeyer,
> distributed under the GNU General Public License version 2 or later.

The adaptation to SPARTA was performed with a mechanical,
scriptable rename (see `port/rename.map` and `port/rename.sh`)
followed by documented SPARTA-specific changes, so that improvements
from the upstream LAMMPS-GUI project can be ported to SPARTA-GUI with
minimal effort. The recipe for porting upstream updates is described
in `port/UPSTREAM.md`. Files retained from LAMMPS-GUI keep Axel
Kohlmeyer's copyright notice.

SPARTA-GUI as a whole is distributed under the GNU General Public
License version 2 (the license of SPARTA); see the `LICENSE` file.
The bundled third-party components in `thirdparty/` (a subset of the
Lepton expression parser and a range-slider widget) have their own
compatible licenses, documented in their directories.

## Building

Requirements: CMake >= 3.20, a C++17 compiler, and Qt 6.2 or later
(modules: Gui, Widgets, Svg).

First build SPARTA as a serial shared library (see the SPARTA manual
for details):

```sh
cd sparta
cmake -S cmake -B build-lib -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=OFF \
      -D BUILD_PNG=ON -D BUILD_JPEG=ON -D CMAKE_BUILD_TYPE=Release
cmake --build build-lib
```

Then build SPARTA-GUI (plugin mode, the default — the GUI finds
`libsparta.so` at runtime):

```sh
cmake -S tools/sparta-gui -B build-gui -D CMAKE_BUILD_TYPE=Release
cmake --build build-gui
./build-gui/sparta-gui
```

On the first start, point the GUI at the SPARTA shared library you
built (Preferences dialog, "Path to SPARTA Shared Library File"), or
start it with the `SPARTA_PLUGIN_PATH` environment variable set to
the directory containing `libsparta.so`.

Alternatively, link the GUI directly against the SPARTA library at
build time:

```sh
cmake -S tools/sparta-gui -B build-gui -D SPARTA_GUI_USE_PLUGIN=OFF \
      -D SPARTA_SOURCE_DIR=$PWD/src \
      -D SPARTA_LIBRARY=$PWD/build-lib/src/libsparta.so
```

### macOS

On macOS, install the dependencies with Homebrew and use the provided
convenience script, which builds both the SPARTA shared library and
the SPARTA-GUI app bundle:

```sh
brew install cmake qt libpng jpeg
./tools/sparta-gui/build-macos.sh
open build-sparta-gui-macos/sparta-gui.app
```

## Documentation

The SPARTA-GUI manual sources are in `doc/` (Sphinx format) and can
be built with `-D SPARTA_GUI_BUILD_DOC=ON`. A short overview is also
included in the SPARTA manual (`doc/sparta_gui.html` in the SPARTA
distribution).

## Citing

If you use SPARTA-GUI in your work, please cite SPARTA (see
https://sparta.github.io) and acknowledge LAMMPS-GUI by Axel
Kohlmeyer, on which SPARTA-GUI is based.
