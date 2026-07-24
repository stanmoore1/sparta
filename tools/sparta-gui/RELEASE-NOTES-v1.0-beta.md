# SPARTA-GUI 1.0 beta — release notes

A graphical editor, runner and visualizer for SPARTA. This is a **beta**: it is
feature-complete for what it claims to do and has been exercised end to end, but
it has not yet been used in anger by anyone but its authors. Please report what
breaks.

## What it does

**Write a deck.** Editor with SPARTA syntax highlighting, autocompletion, a
snippet library, selectable colour schemes, find-and-replace, and autosave with
crash recovery. A doc-driven linter validates commands, styles, argument counts
and numeric argument types as you type, and flags references to computes, fixes,
variables and files that do not exist.

**Run it.** Run from the editor buffer or from a file, stop a run in progress,
watch console output and live charts of any stats column, and set index
variables without editing the deck.

**Look at the results.** Snapshot images with the full `dump image` option set,
a slide show over rendered frames, movie export, charts with curve fitting and
post-processing, and statistical analysis (autocorrelation, block averaging,
steady-state detection) that reports a mean with a genuine uncertainty rather
than a naive standard error.

**Geometry and interchange.** Import STL or SPARTA surface files with a
watertight check performed by SPARTA's own `read_surf`, and export grid and
surface data to ParaView.

**Studies.** Parametric sweeps over index variables, ensembles across RNG seeds
reported as mean ± standard error, and an archive of finished runs that can be
reported as HTML/PDF or compared pairwise.

**3D viewer** (where included — see *Platforms* below). An interactive VTK view
with cut planes, iso-surfaces, line and point probes, and a field calculator.

## Workspaces

The window is organised into three workspaces rather than showing every panel at
once, switched from the status bar or with `Ctrl+1/2/3`:

| Workspace | Shows |
|---|---|
| **Setup** | Project files, linter diagnostics |
| **Run** | Console output, variables, live charts |
| **Analyze** | Charts, snapshot images, slide show |

Panels can be rearranged freely and each workspace remembers its own
arrangement. Switching workspaces only changes what is visible — a run's output
survives a round trip. *View → Reset Layout* restores the current workspace.

## Platforms

| Package | 3D viewer |
|---|---|
| Linux tarball | yes |
| macOS disk image | yes |
| Windows installer | **no** |
| Flatpak bundle | **no** |

The 3D viewer needs VTK. It is available for the platforms above where VTK can
be obtained as a package; the Windows build cross-compiles with MinGW and the
Flatpak builds against the KDE runtime, and neither has a VTK to link against
without building it from source. Those two ship without the viewer, the field
post-processing filters and the STL leak visualiser. Everything else is present
on all four. This is expected to be resolved after 1.0.

## Known limitations

- **Sweeps and ensembles run sequentially**, one run at a time, in-process. An
  *N*-point study therefore takes roughly *N* × a single run.
- **Linter depth varies by command.** All 68 catalogued commands are checked for
  unknown names, styles and argument counts; about 18 additionally have their
  keywords validated and about 12 their numeric argument types. Expect it to
  catch obvious mistakes reliably and subtle ones unevenly.
- **The welcome screen's example gallery shows the 20 examples that ship
  thumbnails**, out of 121 decks in the distribution. Use *File → Open Example*
  for the rest.
- **Run archiving is off by default.** Turn it on in *Preferences → General →
  Archive finished runs*; without it the Run History panel and run comparison
  have nothing to work with. The panel says so when it is empty.
- **The ParaView export needs `pvpython`** on the system; the dialog says so and
  points at the setting if it cannot find it.
- **Movie export needs `ffmpeg`** (or ImageMagick); the export explains how to
  install it if neither is present.
- `quit` in a deck terminates the whole application, not just the run — SPARTA's
  `quit` calls `exit()`. The GUI warns before running such a deck.

## Not in this release

**Remote/cluster execution** (submitting decks to Slurm, PBS or Flux over SSH)
is written and works, but its process orchestration cannot be exercised in
continuous integration without a real cluster, so it is held back rather than
shipped untested.

**The visual case-setup canvas** renders only `block` regions and has no
drag-to-resize, so it is held back until the interaction model is finished.

Both remain in development and are expected after 1.0.

## Reporting problems

Please include the SPARTA-GUI version from *About → About SPARTA-GUI*, your
platform, the deck you were running if it can be shared, and what you expected
to happen. Crashes are the highest priority.
