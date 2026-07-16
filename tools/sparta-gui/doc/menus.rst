*****
Menus
*****

.. index:: menus
.. index:: menu bar
.. index:: keyboard shortcuts

The menu bar has entries *File*, *Edit*, *Run*, *View*, and *About*.
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

The *Create Image* entry will send a `dump image
<https://sparta.github.io/doc/dump_image.html>`_ command to the SPARTA
instance, read the resulting file, and show it in an *Image Viewer*
window (see :ref:`Snapshot Image Viewer <snapshot_viewer>`).

View
^^^^

.. index:: View menu
.. index:: menus; View
.. index:: window visibility

The *View* menu offers to show or hide additional windows with log
output, charts, slide show, variables, or snapshot images.  The
default settings for their visibility can be changed in the
*Preferences* dialog.

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
