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

## Reviewed upstream releases

Upstream releases are reviewed rather than merged wholesale, since much of the
divergence is in files SPARTA-GUI rewrote. What has been looked at so far:

- **v3.0.4 – v3.0.6** (107 commits, reviewed): four features taken —
  *Extend Run*, *Write Restart File*, the download hardening (cancel, stall
  timeout, keep-the-old-library-until-the-new-one-is-good) and the Set
  Variables/deck synchronisation. Deliberately **not** taken:

  - the introspection-driven syntax engine (`lammpssyntax.*`) and its lint
    checker (`syntaxcheck.*`). SPARTA-GUI already has `inputcheck.*`, which
    covers the same ground — per-line diagnostics with severities, argument
    specs, cross-reference checks, Check Input, inline markers and call tips.
    Theirs is driven by library introspection, ours by the SPARTA docs, which
    are the authority for SPARTA's argument grammar. Adopting theirs would be
    a rewrite for parity.
  - atom-radius and pair-style diameter heuristics, and hybrid sub-styles:
    no SPARTA analogue (particles are sized through mixtures).
  - Windows signing, resources and console capture: not a target platform.
  - macOS Homebrew/MacPorts helper lookup, already present here.

  Still open, and worth doing when SPARTA's renderer next gets attention:
  the `defocus`, `gamma` and `depth` cueing image keywords. Upstream's GUI
  emits them, but SPARTA's `image.cpp` implements none of them, so they need a
  core back-port first — the same shape of work as the `fsaa`/`subbox`/
  `backcolor2`/`lights` back-ports already done.

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

## The SPARTA documentation as an input

Three bundled resources are generated rather than written, two of them from
the SPARTA manual:

| resource | generator | reads |
| --- | --- | --- |
| `resources/help_index.table` | `resources/update-help-index.sh` | `doc/src/*.rst` section titles ending in " command" |
| `resources/command_syntax.{table,json}` | `tools/gen_command_syntax.py` | the `Syntax` section of each `doc/src/*.rst` |
| `resources/image_style.table` | `resources/update-image-styles.sh` | `src/*.cpp` (`per_grid_flag`/`per_surf_flag`) |

Regenerate the first two after any SPARTA doc change that adds, renames or
re-argues a command, and commit the result; `test_resources.cpp` checks that
every entry still resolves. The manual was txt2html until the Sphinx
migration, so both generators parse reST now — see their comments for which
reST constructs stand in for which txt2html markers.

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
