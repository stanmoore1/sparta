********
Overview
********

.. index:: overview
.. index:: features

SPARTA-GUI is a graphical text editor customized for editing SPARTA
input scripts.  It uses the `SPARTA library interface
<https://sparta.github.io/doc/Section_howto.html#howto_6>`_ and thus can
run SPARTA directly using the contents of the editor's text buffer.  It
can retrieve and display information from SPARTA while it is running,
display visualizations created with the `dump image command
<https://sparta.github.io/doc/dump_image.html>`_, and is adapted
specifically for editing SPARTA input scripts through syntax
highlighting, text completion, and reformatting, and linking to the
online SPARTA documentation for known SPARTA commands and styles.

SPARTA-GUI aims to support a workflow similar to the traditional
experience of running SPARTA: using a text editor and a command-line
window, launching the SPARTA text-mode executable printing output to
the screen, and post-processing and visualizing SPARTA's output -- but
integrated into a single application.

SPARTA-GUI integrates well with graphical desktop environments where a
filename extension or pattern (SPARTA input scripts are conventionally
named ``in.*``) can be registered with SPARTA-GUI as the application to
launch when double-clicking on such files in a file manager.
SPARTA-GUI will launch and read the file into its buffer.  Input files
can also be dropped into the editor window of the running SPARTA-GUI
application, which will close the current file and open the new file.

SPARTA-GUI makes it easier for beginners to get started running SPARTA,
since you only need to work with a single, ready-to-use program for
most of the tasks.  This saves time and allows users to focus on
learning SPARTA itself, without the need to first learn how to use the
command line or a separate text editor, plotting tool, or visualization
program.  The *Open Example* entry in the *File* menu gives direct
access to the input files of the bundled `SPARTA examples
<https://sparta.github.io/>`_ tree, which are a good starting point for
exploring what SPARTA can do.

While making it easy for beginners to get started with SPARTA, it is
expected that SPARTA-GUI users will eventually transition to workflows
that most experienced SPARTA users employ.  That traditional procedure
is effective for people proficient in using the command line, as it
allows them to use the tools for the individual steps that they are
most comfortable with.  In fact, it is often *required* to adopt this
workflow when running SPARTA simulations on high-performance computing
facilities, since SPARTA-GUI intentionally runs SPARTA without MPI
parallelization.

.. TODO screenshot: capture SPARTA-GUI showing editor, output, chart,
   and slide show windows of a SPARTA example run and save it as
   JPG/sparta-gui-screen.png, then re-enable this figure.
..
.. .. image:: JPG/sparta-gui-screen.png
..    :align: center
..    :scale: 50%

Most features in SPARTA-GUI have been exposed to keyboard shortcuts,
making it also appealing for experienced SPARTA users for prototyping
and testing simulation setups.

.. admonition:: Features

   A detailed discussion and explanation of all features and functionality
   are in the following pages. Here are a few highlights of SPARTA-GUI:

   - Text editor with line numbers and syntax highlighting customized for SPARTA
   - Command completion and indentation support for known commands and styles;
     the completion lists (compute, fix, dump, region, collide, react,
     surf_collide, and surf_react styles) are queried from the loaded
     SPARTA library and thus always match its capabilities
   - Text editor will switch its working directory to the folder of the file in the buffer
   - Indicator for currently executed command
   - Indicator for line that caused an error
   - Progress bar indicates how far a run command has completed and how the CPUs are utilized
   - Context-sensitive help for SPARTA commands via the online documentation
     at https://sparta.github.io/doc/
   - SPARTA is running in a concurrent thread, so the GUI remains responsive
   - SPARTA can be started and stopped with a mouse click or a hotkey;
     stopping interrupts the run cleanly at the next timestep
   - Screen output is captured in an *Output* window
   - Many adjustable settings and preferences are persistent, including the 5 most recent files
   - `Stats output <https://sparta.github.io/doc/stats_style.html>`_ is
     captured and displayed as line graphs in a *Charts* window, with
     optional logarithmic axes, smoothing, curve fitting, and data export
   - Interactive visualization of the current state of particles, grid
     cells, grid cut planes, and surface elements via the `dump image
     <https://sparta.github.io/doc/dump_image.html>`_ facility
   - Capture of images created by `dump image
     <https://sparta.github.io/doc/dump_image.html>`_ in the Slide Show
     window, with export of the image sequence to a movie file
   - Dialog to set variables, similar to the SPARTA command-line flag '-v' / '-var'
   - Support for the KOKKOS accelerator package (OpenMP threads or serial)
   - Inspection of binary restart files created by SPARTA
