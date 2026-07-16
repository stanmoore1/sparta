# Screenshots for the SPARTA-GUI manual

The screenshots inherited from LAMMPS-GUI showed LAMMPS content and have
been removed.  The corresponding image directives in the `.rst` pages
are still present but commented out with a `TODO screenshot` marker;
once a screenshot below has been captured, place it in this folder
(PNG format) and re-enable the directive in the listed page.

Files kept in this folder:

- `emblem-photos.png` / `inactive-photos.png` - generic busy-indicator
  icons used inline in `visualization.rst` (not screenshots).
- `sparta-gui-colormaps.png` - color map preview, *generated* from
  `src/colormaps.cpp` by running `python3 doc/colormaps_preview.py`
  (requires matplotlib).  Regenerate it whenever the color map table
  changes; do not capture it manually.

## Screenshots still to be captured

Capture with a SPARTA example loaded (e.g. `examples/circle/in.circle`)
at a window size around 800x600, in the light theme unless noted.

| File | Page | Content |
| ---- | ---- | ------- |
| `sparta-gui-screen.png` | overview.rst | Full workspace: Editor plus Output, Charts, and Slide Show windows during a run |
| `sparta-gui-main.png` | basic_usage.rst | Main editor window with an input script loaded (light theme) |
| `sparta-gui-dark.png` | basic_usage.rst | Same as above in the dark theme |
| `sparta-gui-running.png` | basic_usage.rst | Editor during an active run: green current-line highlight, progress bar, %CPU display |
| `sparta-gui-run-error.png` | basic_usage.rst | Error dialog with the offending input line highlighted in red |
| `sparta-gui-log.png` | output.rst | Output window with captured screen/stats output |
| `sparta-gui-chart.png` | output.rst | Charts window plotting a stats column during a run |
| `sparta-gui-post-function.png` | output.rst | Postprocess dialog with a custom function/fit entered |
| `sparta-gui-custom-fit.png` | output.rst | A custom fit overlaid on stats data |
| `sparta-gui-import-data.png` | output.rst | Column-picker dialog from *Plot Data File...* |
| `sparta-gui-variable-info.png` | output.rst | Variables window during a run |
| `sparta-gui-variables.png` | menus.rst | *Set Variables...* dialog |
| `sparta-gui-complete.png` | editor.rst | Completion pop-up offering SPARTA commands/styles |
| `sparta-gui-popup-help.png` | editor.rst | Editor context menu with documentation lookup entries |
| `sparta-gui-popup-view.png` | editor.rst | Read-only file viewer opened from the context menu |
| `sparta-gui-inspect-info.png` | editor.rst | Restart inspection: system info text viewer |
| `sparta-gui-inspect-image.png` | editor.rst | Restart inspection: snapshot image window |
| `sparta-gui-find.png` | dialogs.rst | Find and Replace dialog |
| `sparta-gui-prefs-general.png` | dialogs.rst | Preferences: General Settings tab |
| `sparta-gui-prefs-accel.png` | dialogs.rst | Preferences: Accelerators (KOKKOS) tab |
| `sparta-gui-prefs-image.png` | dialogs.rst | Preferences: Snapshot Image tab |
| `sparta-gui-prefs-editor.png` | dialogs.rst | Preferences: Editor Settings tab |
| `sparta-gui-prefs-charts.png` | dialogs.rst | Preferences: Charts Settings tab |
| `sparta-gui-image.png` | visualization.rst | Image Viewer with particles, grid, and surfaces of a SPARTA example |
| `sparta-gui-image-settings.png` | visualization.rst | Dump Image Settings dialog (e.g. Particles or Grid tab) |
| `sparta-gui-slideshow.png` | visualization.rst | Slide Show window with images of a run |

To find the commented-out directives, search the `.rst` files for
`TODO screenshot`.
