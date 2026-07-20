*************
Editor Window
*************

.. _editor:

.. index:: editor window
.. index:: text editor
.. index:: editing features
.. index:: syntax highlighting

The *Editor* window of SPARTA-GUI has most of the usual functionality
that similar programs have: text selection via mouse or with cursor
moves while holding the Shift key, Cut (`Ctrl-X`), Copy (`Ctrl-C`),
Paste (`Ctrl-V`), Undo (`Ctrl-Z`), Redo (`Ctrl-Shift-Z`), Select All
(`Ctrl-A`).  When trying to exit the editor with a modified buffer, a
dialog will pop up asking whether to cancel the exit operation, or to
save or not save the buffer contents to a file.

The syntax highlighting is customized for SPARTA input scripts: SPARTA
commands, styles, numbers, strings, variable references, and comments
are colored differently, and lines that are recognized as invalid are
flagged with an inline warning.

.. index:: color scheme
.. index:: syntax highlighting; color scheme

The palette used for syntax highlighting can be selected in the
*Preferences* dialog (the *Syntax color scheme* drop-down on the
*Editor Settings* tab).  Four schemes are provided:

- **VS Code** (the default) reproduces the familiar *Light+*/*Dark+*
  colors of the Visual Studio Code editor.
- **Solarized** uses the precision palette by Ethan Schoonover, a
  widely adopted cross-editor color standard.
- **One (Atom)** reproduces the *One Light*/*One Dark* colors from the
  Atom editor.
- **Classic (legacy)** is the original SPARTA-GUI palette.

Each scheme sets the editor background and default text color as well as
the individual token colors, and automatically switches between a light
and a dark variant to match the application appearance theme (the
*Solarized*, *VS Code*, and *One* schemes use their own signature
backgrounds; *Classic* keeps the plain theme background).  Comments are
always set apart with an italic, muted color rather than an
attention-grabbing one.  Changing the scheme takes effect immediately,
without restarting SPARTA-GUI.

A SPARTA-GUI logo is shown as a placeholder while the editor is empty;
it disappears as soon as the buffer contains any text.

.. index:: auto-save

The editor has an auto-save mode that can be enabled or disabled in the
*Preferences* dialog.  In auto-save mode, the editor buffer is
automatically saved before running SPARTA or before exiting SPARTA-GUI.

Context Specific Word Completion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. index:: auto-completion
.. index:: word completion
.. index:: code completion

By default, SPARTA-GUI displays a small pop-up frame with possible
choices for SPARTA input script commands or styles after 2 characters of
a word have been typed.

.. TODO screenshot: capture the completion pop-up offering SPARTA
   styles (e.g. after typing "compute 1 grid ...") as
   JPG/sparta-gui-complete.png, then re-enable this figure.
..
.. .. image:: JPG/sparta-gui-complete.png
..    :align: center
..    :scale: 75%

The word can then be completed through selecting an entry by scrolling
up and down with the cursor keys and selecting with the 'Enter' key or
by clicking on the entry with the mouse.  The automatic completion
pop-up can be disabled in the *Preferences* dialog, but the completion
can still be requested manually by either hitting the `Shift+TAB` key or
by right-clicking with the mouse and selecting the option from the
context menu.  Most of the completion information is retrieved from the
active SPARTA instance: the lists of available `compute
<https://sparta.github.io/doc/compute.html>`_, `fix
<https://sparta.github.io/doc/fix.html>`_, `dump
<https://sparta.github.io/doc/dump.html>`_, `region
<https://sparta.github.io/doc/region.html>`_, `collide
<https://sparta.github.io/doc/collide.html>`_, `react
<https://sparta.github.io/doc/react.html>`_, `surf_collide
<https://sparta.github.io/doc/surf_collide.html>`_, and `surf_react
<https://sparta.github.io/doc/surf_react.html>`_ styles as well as the
command names are queried from the loaded SPARTA library, and thus the
completion shows only options that are actually available in the SPARTA
version and configuration in use.  For improved clarity, only the
non-suffix (non-accelerated) versions of styles are shown.

Line Reformatting
^^^^^^^^^^^^^^^^^

.. index:: line reformatting
.. index:: code formatting
.. index:: indentation

The editor supports reformatting lines according to the syntax in order
to have consistently aligned lines.  This primarily means adding
whitespace padding to commands, IDs, and names.  This
reformatting is performed manually by hitting the 'Tab' key.  It is
also possible to have this done automatically when hitting the 'Enter'
key to start a new line.  This feature can be turned on or off in the
*Preferences* dialog for *Editor Settings* with the
"Reformat with 'Enter'" checkbox. The amount of padding for multiple
categories can be adjusted in the same dialog.

Internally this functionality is achieved by splitting the line into
"words" and then putting it back together with padding added where the
context can be detected; otherwise a single space is used between words.

Context Specific Help
^^^^^^^^^^^^^^^^^^^^^

.. index:: context help
.. index:: documentation; inline help
.. index:: documentation; online
.. index:: SPARTA documentation

A unique feature of SPARTA-GUI is the option to look up the SPARTA
documentation for the command in the current line.  This can be done by
either clicking the right mouse button or by using the `Ctrl-?` keyboard
shortcut.  When using the mouse, there are additional entries in the
context menu that open the corresponding documentation page
(``https://sparta.github.io/doc/<command>.html``) in the online SPARTA
manual in a web browser window.  When using the keyboard, the first of
those entries is chosen.

.. TODO screenshot: capture the context menu with the documentation
   lookup entries as JPG/sparta-gui-popup-help.png and the read-only
   file viewer opened from it as JPG/sparta-gui-popup-view.png, then
   re-enable these figures.
..
.. .. |gui-popup1| image:: JPG/sparta-gui-popup-help.png
..    :width: 44%
..
.. .. |gui-popup2| image:: JPG/sparta-gui-popup-view.png
..    :width: 55%
..
..    usage: put "|gui-popup1|  |gui-popup2|" on its own line here

If the word under the cursor is a file, then additionally the context
menu has an entry to open the file in a read-only text viewer window.
If the file is a SPARTA restart file, instead the menu entry offers to
:ref:`inspect the restart <inspect_restart>`.

The text viewer is a convenient way to view the contents of files that
are referenced in the input, for example `surface files
<https://sparta.github.io/doc/read_surf.html>`_ or species files.  The
file viewer also supports on-the-fly decompression of gzipped files
based on the file name suffix.  If the necessary decompression program
is missing or the file cannot be decompressed, the viewer window will
contain a corresponding message.

.. _inspect_restart:

Inspecting a Restart file
^^^^^^^^^^^^^^^^^^^^^^^^^

.. index:: restart file inspection
.. index:: restart files
.. index:: file inspection

When SPARTA-GUI is asked to "Inspect a Restart", it will read the
restart file into a SPARTA instance using the `read_restart command
<https://sparta.github.io/doc/read_restart.html>`_ and then open two
windows.  The first window is a text viewer with a summary of the
system stored in the restart (box dimensions, grid, particle and
surface counts, defined species and mixtures, and so on).  The second
window is a :ref:`Snapshot Image Viewer <snapshot_viewer>` containing a
visualization of the system in the restart.

.. TODO screenshot: capture the two restart inspection windows (info
   text viewer and snapshot image) as JPG/sparta-gui-inspect-info.png
   and JPG/sparta-gui-inspect-image.png, then re-enable these figures.
..
.. .. |inspect1| image:: JPG/sparta-gui-inspect-info.png
..    :width: 40%
..
.. .. |inspect2| image:: JPG/sparta-gui-inspect-image.png
..    :width: 45%
..
..    usage: put "|inspect1|  |inspect2|" on its own line here

.. admonition:: Large Restart Files
   :class: warning

   If the restart file is larger than 250 MBytes, a dialog will ask for
   confirmation before continuing, since large restart files may require
   large amounts of RAM: the entire system must be read into memory.
   Thus restart files for large simulations that have been run on an HPC
   cluster may overload a laptop or local workstation. The *Show
   Details...* button will display a rough estimate of the additional
   memory required.

.. _snippets:

Snippet Library
---------------

.. index:: snippets
.. index:: editor; snippets
.. index:: templates

Beyond single-command auto-completion, the *Edit → Insert Snippet...* dialog
offers a library of ready-made, multi-line SPARTA command blocks for common
tasks — a 3D or 2D flow setup, collisions, reading a surface, ablation
(implicit surfaces), dump image/particle output, time-averaged grid stats,
and more.  Snippets are grouped by category with a live preview; choosing one
(or double-clicking it) inserts the block at the editor cursor.  Some snippets
contain ``${...}`` placeholder tokens (for example a file name or step count)
to fill in after inserting.

.. _autosave_session:

Autosave and Session Restore
----------------------------

.. index:: autosave
.. index:: session restore
.. index:: crash recovery

SPARTA-GUI periodically writes any **unsaved** editor changes to a separate
crash-recovery file (never overwriting your own file).  If the program exits
unexpectedly, the next launch offers to recover that work.  A clean exit
removes the recovery file, and saving your file does too.  The recovery
interval is controlled by the ``autosave_interval`` setting (in seconds; ``0``
disables it).  This is independent of the *Auto-save on 'Run' and 'Quit'*
preference, which writes the buffer back to its own file.

On a clean exit SPARTA-GUI also remembers the last open file and the window
geometry, and restores them on the next launch (unless a file is given on the
command line); set ``restore_session`` to ``false`` to disable reopening the
last file.
