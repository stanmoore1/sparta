# Tracking upstream LAMMPS-GUI

SPARTA-GUI is a derived work of
[LAMMPS-GUI](https://github.com/akohlmey/lammps-gui) by Axel
Kohlmeyer. This document records the pinned upstream baseline and the
recipe for porting future upstream changes.

## Pinned upstream baseline

- Repository: https://github.com/akohlmey/lammps-gui
- Version: **3.0.3**
- Commit: **b6cfbea18a6b46d68de9457ceca9bf09e2946896**

The import commit in this repository ("Import LAMMPS-GUI v3.0.3 as
SPARTA-GUI via mechanical rename") is the unmodified output of
`rename.sh` applied to that upstream tree. Every SPARTA-specific
change since then lives in its own commit on top, so
`git log -- tools/sparta-gui/<file>` shows exactly how any file
diverges from upstream.

## File classification

- **Mechanical files** — identical to upstream after applying
  `rename.map`: all chart/plot/fitting sources, logwindow, fileviewer,
  findandreplace, stdcapture, spartarunner, setvariables, slideshow,
  movieimport, imagecache, qaddon, helpers, thirdparty/. These must
  never accumulate manual divergence; upstream patches apply cleanly
  after renaming.
- **Adapted files** — mechanical rename plus documented SPARTA
  changes: spartawrapper.*, plugin/libspartaplugin.*, constants.h,
  spartagui.*, preferences.*, codeeditor.*, imageviewer.*,
  aboutdialog.*, flagwarnings.*, CMakeLists.txt, cmake/.
- **Rewritten files** — derived from upstream but with
  SPARTA-specific content: highlighter.*, dumpimage.*,
  imageviewersettings.cpp, resources/ tables and generator scripts.
- **Dropped files** — upstream tutorials.*, tutorialwizard.*,
  urldownloader.* (LAMMPS tutorial downloads have no SPARTA
  equivalent).

## Porting an upstream update

1. Fetch the upstream diff between the pinned baseline and the new
   release:

   ```sh
   git -C /path/to/lammps-gui diff b6cfbea18a6b46d68de9457ceca9bf09e2946896..vX.Y.Z > upstream.patch
   ```

2. Translate it with the same rename rules used for the original
   import:

   ```sh
   tools/sparta-gui/port/rename.sh --patch < upstream.patch > renamed.patch
   ```

3. Apply it to the SPARTA tree:

   ```sh
   git apply --directory=tools/sparta-gui --reject renamed.patch
   ```

4. Hunks in **mechanical files** should apply cleanly. Hunks in
   **adapted files** may need small manual merges where SPARTA
   changes overlap. Hunks in **rewritten files** and **dropped
   files** need case-by-case review — read the upstream change and
   apply its intent to the SPARTA version (or skip it if it concerns
   LAMMPS-only features).

5. Update the pinned commit at the top of this file, adjust
   `rename.map` if the upstream change introduced new LAMMPS-specific
   names, rebuild, and run the test harness
   (`tools/sparta-gui/test/`) and a GUI smoke test.

## Rename rules

`rename.map` is the single source of truth for the LAMMPS→SPARTA
translation (URLs first, then library constants, then generic
catch-alls; the same rules are applied to file names and file
contents). `rename.sh <tree> <out>` translates a full tree;
`rename.sh --patch` filters a patch on stdin.
