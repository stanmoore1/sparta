*************************
Basic usage of SPARTA-GUI
*************************

.. index:: basic usage
.. index:: getting started
.. index:: main window

.. _command-line-options:

Command-line options
^^^^^^^^^^^^^^^^^^^^

.. index:: command-line options
.. index:: command-line arguments
.. index:: window size
.. index:: visual style

SPARTA-GUI supports the following command-line options:

.. code-block:: bash

   sparta-gui [options] [file]

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-x <width>``, ``--width <width>``
     - Override the editor window width in pixels
   * - ``-y <height>``, ``--height <height>``
     - Override the editor window height in pixels
   * - ``-s <style>``, ``--style <style>``
     - Set the visual style of the application (default: ``Fusion``)
   * - ``-p <path>``, ``--pluginpath <path>``
     - Set the path to the SPARTA shared library (plugin mode only)
   * - ``-c <file>``, ``--chart <file>``
     - Open ``file`` directly in a standalone :ref:`Charts window <charts>`;
       a column-picker dialog is shown first
   * - ``-i <file>``, ``--image <file>``
     - Open ``file`` in the :ref:`slide show viewer <slideshow>`; may be given
       multiple times to load several images at once
   * - ``-t <file>``, ``--text <file>``
     - Open ``file`` in a standalone text viewer
   * - ``-v``, ``--version``
     - Print version information and exit
   * - ``-h``, ``--help``
     - Print usage information and exit

The optional ``file`` argument specifies a SPARTA input script to open
on startup.  If no file is provided, SPARTA-GUI starts with an empty
editor buffer.  Available choices for the visual style depend on the
platform and Qt configuration.  On most platforms, there is also the
option ``Windows`` which is a visual style somewhat resembling
`Windows 95 <https://en.wikipedia.org/wiki/Windows_95>`_.

The ``-c``, ``-i``, and ``-t`` flags open a standalone viewer without
the main editor window; they are mutually exclusive with each other and
with the ``file`` positional argument.

Launching SPARTA-GUI
^^^^^^^^^^^^^^^^^^^^

When SPARTA-GUI starts, it shows the main window, labeled *Editor*, with
either an empty buffer or the contents of the file used as argument.

.. TODO screenshot: capture the main editor window with a SPARTA input
   script loaded, in the light theme (JPG/sparta-gui-main.png) and in
   the dark theme (JPG/sparta-gui-dark.png), then re-enable the figures.
..
.. .. |gui-main1| image:: JPG/sparta-gui-main.png
..    :width: 48%
..
.. .. |gui-main2| image:: JPG/sparta-gui-dark.png
..    :width: 48%
..
.. |gui-main1|  |gui-main2|

There is the typical menu bar at the top, then the main editor buffer,
and a status bar at the bottom.  The input script contents are shown
with line numbers on the left and the input is colored according to
the SPARTA input script syntax.  The status bar shows the status of
SPARTA execution on the left (e.g.  "Ready." when idle) and the current
working directory on the right.  The name of the current file in the
buffer is shown in the window title; the word `*modified*` is added if
the buffer edits have not yet been saved to a file.  The geometry of the
main window is stored when exiting and restored when starting again.

Opening and saving files
^^^^^^^^^^^^^^^^^^^^^^^^

.. index:: opening files
.. index:: saving files
.. index:: file operations

The SPARTA-GUI application can be launched without command-line
arguments and then starts with an empty buffer in the *Editor* window.

If a file argument is given, SPARTA-GUI will use it as the file name for
the *Editor* buffer and read its contents into the buffer, provided a
file of that name exists; otherwise the buffer will be empty, but set up
to save any added content to that file.  Files can also be opened via
the *File* menu, the `Ctrl-O` (`Command-O` on macOS) keyboard shortcut,
or by drag-and-drop of a file from a graphical file manager into the
editor window.  The *Open Example* entry in the *File* menu gives quick
access to the input scripts in the ``examples`` tree of the SPARTA
distribution.  If a file name pattern (SPARTA input scripts are
conventionally named ``in.*``) has been registered with the graphical
environment to launch SPARTA-GUI, an existing input script can be
opened in SPARTA-GUI through double clicking.

Only one file can be edited at a time, so opening a new file with a file
already loaded into the buffer closes that buffer.  If the buffer has
unsaved modifications, you are asked to either cancel the operation,
discard the changes, or save them.  A buffer with modifications can be
saved any time from the *File* menu, by the keyboard shortcut `Ctrl-S`
(`Command-S` on macOS), or by clicking on the *Save* button at the very
left in the status bar.

Running SPARTA
^^^^^^^^^^^^^^

.. index:: running SPARTA
.. index:: SPARTA execution
.. index:: keyboard shortcuts

From within the SPARTA-GUI main window SPARTA can be started either from
the *Run* menu using the *Run SPARTA from Editor Buffer* entry, by the
keyboard shortcut `Ctrl-Enter` (`Command-Enter` on macOS), or by
clicking on the green *Run* button in the status bar.  All of these
operations cause SPARTA to process the entire input script in the
editor buffer, which may contain multiple `run
<https://sparta.github.io/doc/run.html>`_ commands.

SPARTA runs in a separate thread, so the GUI stays responsive and is
able to interact with the running calculation and access data it
produces.  It is important to note that running SPARTA this way uses the
contents of the input buffer for the run (passed line by line to the
command processor of the `SPARTA library interface
<https://sparta.github.io/doc/Section_howto.html#howto_6>`_), and
**not** the original file it was read from.  Thus, if there are unsaved
changes in the buffer, they *will* be used.  As an alternative, it is
also possible to run SPARTA by reading the contents of a file from the
*Run SPARTA from File* menu entry or with `Ctrl-Shift-Enter`.  This
option may be required in some rare cases where the input uses some
functionality that is not compatible with running SPARTA from a string
buffer.  For consistency, any unsaved changes in the buffer must be
either saved to the file or undone before SPARTA can be run from a
file.

The line number of the currently executed command is highlighted in
green in the line number display for the *Editor* window.

.. TODO screenshot: capture the editor during an active run (green
   line-number highlight, progress bar and CPU display in the status
   bar) as JPG/sparta-gui-running.png, then re-enable this figure.
..
.. .. image:: JPG/sparta-gui-running.png
..    :align: center
..    :scale: 75%

While SPARTA is running, the contents of the status bar change.  The
text fields that normally show "Ready." and the current working
directory change into an area showing the CPU utilization in percent.
Next to it is a text indicating that SPARTA is running, which also
indicates the number of active threads (in case the KOKKOS/OpenMP
accelerator was selected in the *Preferences* dialog).  On the right
side, a progress bar is shown that displays the estimated progress for
the current `run <https://sparta.github.io/doc/run.html>`_ command.

.. admonition:: CPU Utilization
   :class: note

   The CPU Utilization should ideally be close to 100% times the number
   of threads.  Since the GUI is running as a separate thread, the CPU
   utilization *may* be higher, for example when the GUI needs to work
   hard to keep up with the output produced by the simulation.  This can
   occur when there is frequent stats output or the simulation runs very
   fast.  In the *Preferences* dialog, the polling interval for updating
   the *Output* and *Charts* windows can be adjusted.  The intervals may
   need to be lowered to avoid missing data between *Chart* data updates
   or to avoid stalling when the stats output is not transferred to the
   *Output* window fast enough.  It is also possible to reduce the
   amount of data by increasing the `stats interval
   <https://sparta.github.io/doc/stats.html>`_.  SPARTA-GUI detects if
   the associated I/O buffer is significantly full, and will print a
   warning *after* the run with suggested adjustments.  The CPU
   utilization can also be lower than expected when some significant
   parts of the code paths in use are not multi-threaded, or when the
   simulation is slowed down by the GUI or other processes also running
   on the host computer and competing with SPARTA-GUI for resources.

If an error occurs (for example when a command is misspelled or a
required file is missing), an error message dialog is shown and the
line of the input which triggered the error is highlighted in red.
SPARTA reports such errors through its exception mechanism, so the GUI
itself keeps running and the input can be corrected and run again
right away.  The state of SPARTA in the status bar is set to "Failed."
instead of "Ready."

.. TODO screenshot: capture the error dialog with the offending input
   line highlighted in red as JPG/sparta-gui-run-error.png, then
   re-enable this figure.
..
.. .. image:: JPG/sparta-gui-run-error.png
..    :align: center
..    :scale: 75%

.. admonition:: Up to three additional windows may open during a run:

   - An :ref:`Output window <logfile>` with the captured screen output from SPARTA
   - A :ref:`Charts window <charts>` with line graphs created from the stats output of the run
   - A :ref:`Slide Show window <slideshow>` with images created by a `dump image command <https://sparta.github.io/doc/dump_image.html>`_
     in the input

More information on those windows and how to adjust their behavior and
contents is given in :doc:`the next pages <output>`.

An active SPARTA run can be stopped cleanly by using either the *Stop
SPARTA* entry in the *Run* menu, the keyboard shortcut `Ctrl-/`
(`Command-/` on macOS), or by clicking on the red button in the status
bar.  This uses the timeout mechanism of the SPARTA library interface:
the running SPARTA process completes the current timestep and then
completes the processing of the buffer while skipping all remaining
timesteps of any run commands.  This way the run is interrupted in a
clean state and, for example, restart files or dump outputs scheduled
at the end of a run are still written.
