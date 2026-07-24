*****
Menus
*****

.. index:: menus
.. index:: menu bar
.. index:: keyboard shortcuts

The menu bar has entries *File*, *Edit*, *Run*, *Tools*, *View*, and *About*.
Instead of using the mouse to click on them, the individual menus can
also be activated by hitting the `Alt` key together with the
corresponding underlined letter, that is `Alt-F` activates the
*File* menu.  For the corresponding activated sub-menus, the key
corresponding to the underlined letters can be used to select entries
instead of using the mouse.

.. _files:

File
^^^^

.. index:: File menu
.. index:: menus; File
.. index:: Open Example

.. admonition:: The *File* menu offers the usual options:

   - *New Input File* clears the current buffer and resets the file name to ``*unknown*``
   - *Open Input File* opens a dialog to select a new file for editing in the *Editor*
   - *Open Example* offers a submenu with the input scripts from the
     ``examples`` tree of the SPARTA distribution, organized by example
     folder.  Selecting an entry loads that input script and switches
     the working directory to its folder.  The menu is built by
     scanning the ``examples`` folder of the SPARTA distribution the
     loaded SPARTA library was built from.
   - *Save Input File* saves the current file; if the file name is ``*unknown*``
     a dialog will open to select a new file name
   - *Save Input File As* opens a dialog to select a new file name (and folder, if
     desired) and saves the buffer to it.  Writing the buffer to a different folder
     will also switch the current working directory to that folder.
   - *View Text File* opens a dialog to select a file for viewing in a *separate*
     window (read-only) with support for on-the-fly decompression as explained
     above.  If the selected file appears to be an image, a movie, or a binary file,
     a warning is shown instead; use *View Image or Movie File(s)...* for those.
   - *View Image or Movie File(s)...* opens a dialog to select one or more image files
     and shows them together in a standalone :ref:`slide show <slideshow>` window.  This is
     useful for reviewing images created by an external (e.g. large parallel) simulation,
     or for revisiting images from an earlier run without rerunning it.  Image formats
     that Qt cannot read natively are converted on demand with
     `ImageMagick <https://imagemagick.org/>`_ if it is available, and each file is
     converted only once.  Movie files may be selected as well: their frames are
     extracted into individual images with `FFmpeg <https://ffmpeg.org/>`_ after
     confirming a dialog that also selects the frame range and interval, as explained
     under :ref:`Importing movie files <movie_import>`.
   - *Plot Data File...* opens a dialog to select a file with column-oriented numeric
     data and plots it in a standalone :ref:`Charts window <charts>` without running a
     simulation.  See the description below for details.
   - *Inspect Restart File* opens a dialog to select a file.  If that file is a
     `SPARTA restart <https://sparta.github.io/doc/write_restart.html>`_,
     windows with :ref:`information about the file are opened
     <inspect_restart>`.
   - *Quit* exits SPARTA-GUI. If there are unsaved changes, a dialog will
     appear to either cancel the operation, or to save, or to not save the
     modified buffer.

In addition, up to 5 recent file names will be listed after the *Open Input File*
entry that allows re-opening recently opened files.  This list is stored
when quitting and recovered when starting again.

**Plotting external data files.** The *Plot Data File...* entry
(`Ctrl-Shift-P`) opens a dialog to select a file with column-oriented
numeric data and plots it in a standalone :ref:`Charts window <charts>`
without running a simulation.  Supported formats are whitespace-separated
columns (``.dat``), comma-separated values (``.csv``), `YAML
<https://yaml.org/>`_, and `JSON <https://www.json.org/>`_; the format
is recognized from the file name extension or, failing that, from the
content.  After the file is read, a dialog lets you pick which column
provides the x axis and which columns to plot; column names can also be
edited at this point.  All the post-processing and export features
described for the :ref:`Charts window <charts>` are available here as
well.

Edit
^^^^

.. index:: Edit menu
.. index:: menus; Edit
.. index:: Find and Replace

The *Edit* menu offers the usual editor functions like *Undo*, *Redo*,
*Cut*, *Copy*, *Paste*, and a *Find and Replace* dialog (keyboard
shortcut `Ctrl-F`).  It can also open a *Preferences* dialog (keyboard
shortcut `Ctrl-P`) and allows deleting all stored preferences and
settings, so they are reset to their default values.

.. _run_menu:

Run
^^^

.. index:: Run menu
.. index:: menus; Run
.. index:: SPARTA execution
.. index:: SPARTA library interface

The *Run* menu has options to start and stop a SPARTA process.  Rather
than calling the SPARTA executable as a separate executable, SPARTA-GUI
is linked to (or dynamically loads) the SPARTA library and thus can run
SPARTA internally through the `SPARTA library interface
<https://sparta.github.io/doc/Section_howto.html#howto_6>`_ in a
separate thread.

Specifically, a SPARTA instance will be created through the
``SpartaWrapper`` C++ adapter around the SPARTA C library interface.
The buffer contents are then executed line by line through the library
command processor.  Certain commands and features are only available
after a SPARTA instance is created.  As an alternative, it is also
possible to run SPARTA using the contents of the edited file by reading
the file.  This is mainly provided as a fallback option in case the
input uses some feature that is not available when running from a
string buffer.

The SPARTA calculations are run in a concurrent thread so that the GUI
can stay responsive and be updated during the run.  The GUI can retrieve
data from the running SPARTA instance and tell it to stop at the next
timestep.  The *Stop SPARTA* entry will do this by activating the
timeout mechanism of the SPARTA library, which lets the current
timestep complete and then winds down the run cleanly.

The *Check Input* entry (keyboard shortcut `Ctrl-K`) statically validates
the current input deck without running SPARTA and reports unknown
commands or styles, wrong argument counts, undefined variable/compute/fix
references, and missing referenced files.  Problems are marked inline in
the editor and listed in the docked *Diagnostics* window.  See
:ref:`Input Validation <input_validation>` for details.

The *Relaunch SPARTA Instance* entry will destroy the current SPARTA
thread and free its data and then create a new thread with a new SPARTA
instance.  This is usually not needed, since SPARTA-GUI tries to detect
when this is needed and does it automatically.  This is available
in case it missed something and SPARTA behaves in unexpected ways.

The *Set Variables...* entry opens a dialog box where `index style
variables <https://sparta.github.io/doc/variable.html>`_ can be set.
Those variables are passed to the SPARTA instance when it is created
and are thus set *before* a run is started.  This is the equivalent of
the ``-var`` command-line flag of the SPARTA executable.

.. TODO screenshot: capture the Set Variables dialog as
   JPG/sparta-gui-variables.png, then re-enable this figure.
..
.. .. image:: JPG/sparta-gui-variables.png
..    :align: center
..    :scale: 50%

The *Set Variables* dialog will be pre-populated with entries that
are set as index variables in the input and any variables that are
used but not defined, if the built-in parser can detect them.  New
rows for additional variables can be added through the *Add Row*
button and existing rows can be deleted by clicking on the *X* icons
on the right.

The *Continue from Restart...* entry opens a small browser of the `SPARTA
restart <https://sparta.github.io/doc/write_restart.html>`_ files in the
working directory (with their size and modification time).  A selected file
can be inspected (see :ref:`Inspecting restart files <inspect_restart>`) or
used to continue a run: choosing *Insert Continue Commands* inserts a
``read_restart <file>`` line followed by a ``run`` command for the requested
number of additional steps at the editor cursor, ready to review and run.

The *Create Image* entry will send a `dump image
<https://sparta.github.io/doc/dump_image.html>`_ command to the SPARTA
instance, read the resulting file, and show it in an *Image Viewer*
window (see :ref:`Snapshot Image Viewer <snapshot_viewer>`).

View
^^^^

.. _tools_menu:

Tools Menu
^^^^^^^^^^

.. index:: Tools menu
.. index:: menus; Tools

The *Tools* menu collects the operations that work on simulation data but
sit outside the edit-run-look loop that *File* and *Run* cover.

   - *Import Surface (STL / SPARTA)...* opens the :ref:`Import Surface wizard
     <import_surface>` to convert an STL file (ASCII or binary) or open an
     existing SPARTA surface file, transform and preview it, review the
     watertightness diagnostics, optionally generate the implicit-surface
     (ablation) commands, and insert the corresponding ``read_surf`` or
     ``create_isurf`` block at the editor cursor.
   - *Export to ParaView...* opens the :ref:`Export to ParaView dialog
     <export_paraview>` to convert SPARTA surface or grid data to ParaView
     ``.pvd`` format by running the bundled ``surf2paraview.py`` /
     ``grid2paraview.py`` scripts with ParaView's ``pvpython`` and, optionally,
     open the result in ParaView.  ParaView must be installed separately.
   - *Surface Quantities Report...* opens a dialog that integrates a
     per-surface compute or fix over the surface elements of a running
     simulation and reports forces, moments and heat flux.  It needs a live
     simulation with surfaces and a per-surf compute defined; if either is
     missing it says so.

The *Studies* submenu gathers the features that drive the same deck
repeatedly:

   - The *Parametric Sweep...* entry opens the :ref:`Parameter Sweep panel
     <parametric_sweep>`, which runs the current deck repeatedly while varying
     index variables over ranges and tabulates a chosen thermo quantity per run.



.. index:: View menu
.. index:: menus; View
.. index:: window visibility

.. index:: workspaces
.. index:: View menu; workspaces

The *View* menu begins with the three **workspaces**, which are the
main way to change what the window shows.  Rather than displaying every
panel at once and leaving too little room for any of them, each
workspace shows the panels that belong to one task:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Workspace
     - Shortcut
     - Panels
   * - *Setup*
     - ``Ctrl+1``
     - Project Files, Diagnostics
   * - *Run*
     - ``Ctrl+2``
     - Output, Variables, Charts
   * - *Analyze*
     - ``Ctrl+3``
     - Charts, Image, Slide Show, Output

The workspaces are also reachable from the segmented switch in the
status bar.  Panels can be rearranged, added or removed freely and each
workspace remembers its own arrangement, so tailoring *Analyze* does not
disturb *Run*.  Switching workspaces only changes what is visible --
panel contents are never discarded, so the output of a run survives a
round trip through the other workspaces.  *Reset Layout* at the bottom
of the menu restores the current workspace's default arrangement and
leaves the others alone.

Starting a run switches to the *Run* workspace once per session; set
``runmode_autoswitch`` to ``false`` in the configuration to suppress
that.

Below the workspaces, the *View* menu offers to show or hide the
individual windows with log output, charts, slide show, variables, or
snapshot images.  Opening one adds it to the current workspace's
arrangement.  The default settings for their visibility on a run can be
changed in the *Preferences* dialog.

Opening the *Image Window* when no snapshot has been created yet
renders one on demand (equivalent to *Run* -> *Create Image*), so the
window always shows an image rather than an empty pane.  If a snapshot
cannot be created from the current input -- for example because it does
not define a simulation box -- a message explains why instead.

.. index:: Project Files
.. index:: multi-file navigation

The *Project Files Window* entry shows a docked navigator that lists the
files in the working directory of the current input deck.  Files that the
deck references through an ``include`` or ``read_surf``/``read_grid``/
``read_restart``/``read_particles`` command are shown in bold, and
double-clicking any entry opens it in the editor.  Together with the
*Open '<file>' in editor* action in the editor's right-click menu (available
when the cursor is on a file name, e.g. the target of an ``include`` line),
this makes it easy to move between a driver deck and the files it pulls in.

About
^^^^^

.. index:: About menu
.. index:: menus; About
.. index:: documentation; online
.. index:: SPARTA documentation
.. index:: help

The *About* menu finally offers a couple of dialog windows and an
option to launch the SPARTA online documentation in a web browser.  The
*About SPARTA-GUI* entry displays a dialog with a summary of the
configuration settings of the SPARTA library in use and the version
number of SPARTA-GUI itself.  The *Quick Help* displays a dialog with
a minimal description of SPARTA-GUI.  The *SPARTA-GUI Documentation*
entry will open the SPARTA-GUI documentation (this manual) in a web
browser window.  The *SPARTA Manual* entry will open the main page of
the `SPARTA online documentation
<https://sparta.github.io/doc/Manual.html>`_ in a web browser window.
