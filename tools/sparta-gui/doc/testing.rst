*******
Testing
*******

The ``test`` directory contains some tests for the SPARTA-GUI project
using either the `GoogleTest framework
<https://github.com/google/googletest/>`_ or the `Python unittest
framework <https://docs.python.org/dev/library/unittest.html>`_.

Overview
^^^^^^^^

The test suite uses CMake's CTest front end to select and run the
tests. Tests implemented with GoogleTest are automatically discovered
and can be run individually or as a complete suite for each test
program.  Tests running SPARTA-GUI itself use the "virtual frame buffer"
X server (Xvfb) and are written in Python using the ``unittest``
Python module and the `PyAutoGUI module
<https://pyautogui.readthedocs.io/>`_

Building the Tests
^^^^^^^^^^^^^^^^^^

Tests are built as part of the main project build when using ``-D
ENABLE_TESTING=ON`` during CMake configuration (default setting is
`OFF`).  Due to technical requirements, testing is currently only
enabled for native Linux builds of SPARTA-GUI.  In any other build
environment, the ``-D ENABLE_TESTING=ON`` setting is ignored.

Quick Build
===========

For running the tests, it is not necessary to build the documentation,
so its build can be skipped during configuration.

.. code-block:: bash

   cmake -S . -B build -D SPARTA_GUI_USE_PLUGIN=ON -D BUILD_DOC=OFF -D ENABLE_TESTING=ON
   cmake --build build --parallel 2

Disable Tests
=============

Tests are disabled by default.  If they have been enabled during CMake configuration
they can be disabled at a later point with:

.. code-block:: bash

   cmake -S . -B build -D ENABLE_TESTING=OFF

Running Tests
^^^^^^^^^^^^^

Below are some frequently used command line examples for running tests.
These examples assume that SPARTA-GUI was compiled in the folder ``build``.
CTest can also be invoked from the top of the build tree (``--test-dir
build``); both forms work and select the same set of tests.

Run All Tests
=============

.. code-block:: bash

   ctest --test-dir build/test

Run Tests with Verbose Output
=============================

.. code-block:: bash

   ctest --test-dir build/test -V

List Available Tests
====================

The list of the names of all available tests can be obtained with:

.. code-block:: bash

   ctest --test-dir build/test -N


Run Specific Tests
==================

Individual tests can be selected in different ways.  Most common is the
use of regular expressions to select (``-R``) or exclude (``-E``) tests.
It is also possible to select tests by a range of Test numbers (``-I``)
from the -N test list output. Examples:

.. code-block:: bash

   ctest --test-dir build/test -R HelpersTest
   ctest --test-dir build/test -E Frame
   ctest --test-dir build/test -I 20,25

Current Test Coverage
^^^^^^^^^^^^^^^^^^^^^

The unit tests cover the utility functions, the stdout capture, the log
window warning highlighter, the dump-image command builder, the movie import
and image cache of the Slide Show window, the plot data model with its file
parsers and writers, the chart axis-layout math, and the Qt-free math
toolkit (least squares and smoothing, autocorrelation, curve fitting, the
Levenberg-Marquardt solver, the vendored LeptonMini expression parser, and
the custom-function layer on top of them).  Command-line tests validate
basic executable behavior, and PyAutoGUI-based tests exercise the GUI
itself.  Future expansion will include more GUI component testing and
integration tests.

Test Organization
=================

Tests are organized into three main categories:

1. **Unit Tests**: Using GoogleTest framework to test individual functions
2. **Command-Line Tests**: Using command-line to validate basic executable behavior
3. **GUI Tests**: Tests using the Python unittest framework and
   PyAutoGUI to run SPARTA-GUI inside a virtual frame buffer in a
   "remote controlled fashion".

Unit Tests
==========

test_helpers.cpp
----------------

Comprehensive tests for functions in ``src/helpers.h`` and ``src/helpers.cpp``.
This module contains test cases covering the utility functions used throughout
the application.

**Date Comparison (dateCompare)**
  Tests for the dateCompare function that compares version date strings
  in SPARTA date format (e.g., "22 Jul 2025"):

  - Same dates (returns 0)
  - Different years (returns positive/negative)
  - Different months (returns positive/negative)
  - Different days (returns positive/negative)
  - Full month names vs. abbreviations
  - Invalid date formats
  - Edge cases (year boundaries, month boundaries)
  - December month ordering
  - Both dates invalid
  - Dates with "- Update" suffix

**Line Splitting (splitLine)**
  Tests for the splitLine function that parses command-line style input
  with proper quote handling:

  - Simple whitespace-separated tokens
  - Single-quoted strings
  - Double-quoted strings
  - Escaped quotes within strings
  - Mixed quoting styles
  - Triple-nested quotes
  - Multiple consecutive whitespace characters
  - Empty input
  - Quotes at string boundaries
  - Single word without spaces
  - Hash comment handling
  - Special characters

**Executable Detection (hasExe)**
  Tests for the hasExe function that checks if an executable exists in PATH:

  - Common system commands (sh, ls on Unix; cmd on Windows)
  - Non-existent commands
  - Empty string input
  - bash executable
  - Platform-specific behavior (conditional compilation)

**Theme Detection (isLightTheme)**
  Tests for the isLightTheme function that determines if the current
  Qt theme is light or dark:

  - Boolean return value validation
  - Consistency across calls
  - No crashes on theme query

**Stdout Silencing (silenceStdout/restoreStdout)**
  Tests for the stdout silencing functions that redirect stdout to suppress
  SPARTA library output:

  - Silencing actually suppresses stdout
  - Restoring when not silenced is a no-op
  - Nested silencing and restoring
  - Silencing skipped during active StdCapture
  - Capture restores silenced stdout state
  - Silence and restore preserves stdout

**Directory Purging (purgeDirectory)**
  Tests for the purgeDirectory function that recursively removes directory
  contents:

  - Purging directory with files
  - Non-existent directory (no crash)
  - Empty directory

**Image File Detection (isImageFile)**
  Test for the extension-based check that decides whether a file is loaded
  as an image by the Slide Show window.

**Grayscale Conversion (grayscaleImage)**
  Tests for the helper that renders an image in grayscale, used for the
  "inactive" state of icons:

  - Color is removed while the alpha channel is preserved
  - Contrast is faded towards the midpoint

**Qt Message Silencing (QtMessageSilencer)**
  Tests for the :cpp:class:`QtMessageSilencer` RAII guard that collects Qt
  warning messages instead of printing them to the console:

  - Emitted warnings are collected instead of printed
  - Nothing is collected when nothing is emitted
  - The previous message handler is restored and guards can be nested

**Viewer Window Fit (viewerFitSize)**
  Tests for the pure window-fit computation used by the Image Viewer and
  Slide Show window auto-resize:

  - Content within the screen budget is fitted exactly (content plus frame)
  - An exact fit adds no scroll bar allowance
  - Width or height overflow clamps that axis and adds scroll bar room on
    the other axis
  - Overflow in both directions clamps to the budget
  - The added scroll bar room never exceeds the budget
  - A negative budget clamps to zero

test_stdcapture.cpp
-------------------

Tests for the :cpp:class:`StdCapture` class in
``src/stdcapture.{h,cpp}``, which redirects the C-level stdout file
descriptor through a pipe so that SPARTA library output can be
collected and displayed in the *Output* window.  Test cases cover:

- Capturing simple single-line output
- Capturing multiple lines of output
- Behavior with empty output
- Re-using a single ``StdCapture`` instance for several capture cycles
- ``getChunk()`` returning incremental output while a capture is active
- ``getChunk()`` returning an empty string when not capturing
- Multiple successive ``getChunk()`` calls during one capture
- ``endCapture()`` being a safe no-op when no capture is active
- ``getBufferUse()`` reporting zero before any capture activity
- ``getBufferUse()`` reflecting the size of the most recent chunk
- The original ``stdout`` file descriptor being restored when capture ends

test_flagwarnings.cpp
---------------------

Tests for the :cpp:class:`FlagWarnings` syntax highlighter used in
the *Output* window to flag lines beginning with ``WARNING`` or
``ERROR`` and to update a running count displayed in the summary
label.  Test cases cover:

- Constructor initializes the warning count to zero
- ``WARNING`` and ``ERROR`` lines are correctly detected
- Normal output lines are not flagged
- The summary ``QLabel`` is updated when warnings appear
- Empty documents produce no spurious warnings
- Warnings are detected when they appear at the start of a line
- The running count is correct for multiple warnings

test_dumpimage.cpp
------------------

Tests for the GUI-free dump-image command builder (``src/dumpimage.{h,cpp}``)
that assembles the SPARTA ``dump ... image`` and ``dump_modify`` commands
from a ``DumpImageSettings`` snapshot of the Image Viewer state.  Test
cases cover:

- Basic structure of the generated dump image and ``dump_modify``
  arguments and their deterministic keyword order
- Pruning of all settings that match the SPARTA built-in defaults
- The color-map (``cmap``) emission for the six modes (particle, grid,
  surf, gridx, gridy, gridz), including named, reversed, discrete, and
  sequential maps and the canonical stops of the perceptual maps
- Per-species color and diameter tables (``pcolor``/``pdiam``) reduced
  to the entries that differ from the defaults
- Grid volume rendering vs. grid cut planes, surface element options,
  box/sub-box/outline/axes keywords, background gradient, and lights

test_dumpimagesettings.cpp
--------------------------

Tests for the tabbed dump-image settings dialog
(``src/dumpimagesettingsdialog.{h,cpp}``), the eight-tab editor behind the
Image Viewer's *Settings* button.  Where ``test_dumpimage.cpp`` checks the
commands built from a ``DumpImageSettings`` snapshot, these check the step
before it: that each of the hundred-odd controls reads and writes the field
it is supposed to.

The dialog is a pure function of ``(DumpImageSettings, ImageSettingsEnv)``
-- it holds no simulator and no viewer -- so the tests run offscreen in
well under a second and need neither a display nor the SPARTA library.
Controls are looked up by object name, and each object name is the
``DumpImageSettings`` field it drives.  Test cases cover:

- The tab inventory and the clamping of the requested tab into range
- A full round trip: build from a populated struct, change nothing, read
  back, and get every field unchanged -- which a control wired to the
  wrong field cannot survive
- Field by field, one test per tab: particles, grid, grid planes,
  surfaces, box/axes, camera, quality, and colour maps
- Fields the dialog has no control for (image size, SSAO seed, the movie
  frame rate and bit rate) carried through untouched
- The ``hasAcceptableInput()`` guards: an editor holding invalid text
  keeps the previous value rather than resetting it, and a zero up vector
  leaves all three components alone
- Value sources composed with and without an array subscript, and a
  ``v_`` reference never taking one
- The environment shaping the dialog: a 2D deck disables the Z plane, the
  Z view centre and the up vector; a deck with no surfaces disables the
  Surfaces tab; the cut-plane ranges follow ``boxlo``/``boxhi``; the
  mixture, region and group combos list what the environment says
- Grid volume rendering and the cut planes switching each other off
- Six independent colour-map specs, with the one on screen flushed into
  its slot without switching modes
- The species table, including a colour name Qt cannot parse keeping the
  row's previous RGB
- Degenerate environments: no mixtures, no regions, no species, no
  surfaces, a zero-size box, and a species colour list shorter than the
  species count
- The Help button emitting a page name rather than opening a browser
- Every control having a unique object name and a non-empty accessible
  name, without which the AT-SPI walker and the screenshot sweep cannot
  identify what they just touched

test_sweeppanel.cpp
------------------

Tests for the parametric sweep (``src/sweeppanel.{h,cpp}``): the results
model behind the table, and the panel itself.  The panel is constructible
with neither a main window nor a simulator -- it asks the first only to
detect variables in the deck and the second only to refuse starting on top
of a live run -- so its controls and its spec validator are testable
offscreen.  A small reaper reads and dismisses each ``QMessageBox``, which
is how the validator's refusals become observable.  Test cases cover:

- The results model: headers, rows, reset and clear, and the change
  notifications a view needs
- The panel building its controls with nothing behind it, and detecting
  variables from an absent deck without inventing a row
- Variable rows added and removed, each row getting its own editors, and
  removing with nothing selected removing nothing
- Every refusal the spec validator can produce: no variables, an empty
  value list, a range that is not ``start:stop:step``, a linspace that is
  not ``start:stop:count``, nothing to tabulate, and replicates asked for
  without a seed variable
- Exporting and charting with no results saying so, rather than opening an
  empty file dialog or a blank chart window

test_jsoncolors.cpp
-------------------

Tests for the species colour file (``loadJsonColors()`` /
``saveJsonColors()`` in ``src/imageviewer.cpp``) -- what *Save Colors*
writes and *Load Colors* reads.  It is the one thing the image viewer
persists that a user can hand to someone else, so it carries a header
naming the application and format and a revision number.  Both ends go
through a file dialog, which is why neither had been checked.  Test cases
cover:

- A well-formed file read back with its colours and light intensities
- A round trip: what ``saveJsonColors()`` writes read back by
  ``loadJsonColors()``, header and revision included
- Cancelling either end doing nothing
- Malformed JSON, and JSON that is not an object, refused with the reason
- Someone else's JSON refused rather than misread -- without the header
  check it would come back as an empty colour list and silently reset
  every species
- The right application with the wrong format still refused
- A newer revision refused rather than guessed at
- A directory handed over by the dialog refused as unreadable
- An empty colour list still producing a file with its header, so it can
  be read back at all

test_stlimportwizard.cpp
------------------------

Tests for the surface import wizard (``src/stlimportwizard.{h,cpp}``): the
six tab pages on top of the parsers and command builders that
``test_stlimport.cpp`` already covers.  It needs no simulator -- it checks
its ``SpartaWrapper`` before every use, and with none it falls back to the
preflight watertightness heuristic and its own mesh renderer.  The fixture
writes an ASCII-STL tetrahedron, and the same mesh with one facet removed
for the open case.  Test cases cover:

- An STL loading with every page usable; a source it cannot read leaving
  only the first page and disabling *Insert into editor*, rather than
  generating commands for a mesh it does not have
- The default output being a ``read_surf`` command naming the ``.surf``
  file the wizard will write
- The transform controls reaching the command, and a scale turned back off
  leaving it again
- The invert, clip and transparent options, and the group name trimmed
- All four translation kinds producing a command the builder recognises
- Switching to implicit mode producing ``create_isurf`` and a ``fix ablate``
  instead, its group, fix ID and threshold reaching the commands, and
  switching back restoring ``read_surf``
- The implicit grid resolution recorded in the settings but deliberately
  *not* emitted -- it drives the preview, and emitting it would override
  the grid in the deck the user already has
- Accepting writing the surface file the command names, an existing
  ``.surf`` used where it is rather than rewritten, and cancelling writing
  nothing
- A closed mesh and one with a hole in it reported differently

test_slideshow.cpp
------------------

Tests for the slide show (``src/slideshow.{h,cpp}``), the window that turns
a run's dump images into an animation.  Its sequence bookkeeping needs no
simulator, and the live suites that drive its buttons photograph the result
without checking which image is showing.  Real PNG files, each a distinct
colour, because the window reads their headers to size itself.  Test cases
cover:

- Images collected, a file already in the sequence not added twice (a
  rescan of the run directory offers every image again), and the
  ``contentChanged`` signal reaching the window around it
- Clearing leaving the window empty and reusable, and leaving the files on
  disk
- An image the run has not finished writing still counting as part of the
  sequence
- The active range growing with the sequence, with Stop following the end
  only while it was pinned there -- an explicit Stop is not dragged along
- The Start/Stop boxes being one-based where the indices are not
- Next and previous walking the range, the ends sticky when not looping and
  wrapping when looping (which is the default), and both respecting a
  narrowed range
- Navigating and transforming an empty show, and a single-image sequence
- Play locking the delay while it runs, and rewinding *and drawing* the
  first image of a non-looping range
- The rotate, flip and zoom transforms undoing one another, and the window
  rendering what it holds at mixed image sizes

The slide-show section above also covers the destructive half -- ``deleteImages()``
and ``purgeCache()``, the only place the application removes files from disk.
Those cases check what survived as well as what went: the selected range
deleted and nothing on either side of it, the sequence and the disk staying in
step, declining deleting nothing, the confirmation naming the count and the
bounds, and the state afterwards, where what survives becomes the whole range
again so navigation cannot point past the end.  The conversion cache is
exercised through a file Qt genuinely cannot decode (an SGI written by
ImageMagick), which is the only way anything reaches it.

test_run.cpp
------------

Tests for the run path: ``doRun()``, ``archiveFinishedRun()``,
``continueRestart()`` and ``renderVtkSnapshot()``.  Three of the four need a
simulation actually running, which this suite has -- the same shared
libsparta the window loads anyway, driven through the window's own Run
action on a deck small enough to finish in a moment.  A run is
asynchronous, so every case waits on the ``runFinished`` signal rather than
assuming the run is over when the call returns.  Registered as a single
ctest entry under Xvfb (VTK renders through its own X connection even when
Qt is offscreen) and serialised against the other live-window suites.  Test
cases cover:

- A run finishing and reporting success, its output reaching the log
  window, and its thermo columns reaching the chart
- ``gui_run``, the index variable SPARTA-GUI defines so a deck can name its
  output after the run it came from, readable from the deck and advancing
  between runs -- and each run getting a fresh log window
- An invalid deck reported as a failure rather than a success
- A deck containing ``quit`` asking first, because SPARTA's ``quit`` calls
  ``exit()`` and would take the whole application with it; the word
  appearing inside a comment or another word not triggering it
- A second run on top of a running one refused, and a run ending when
  stopped
- Archiving off by default; an archived run recording the deck, the log,
  the timestamp and the build/host provenance that makes it traceable; a
  failed run archived as failed; a saved deck archived under its own name;
  and each run getting its own entry
- Continuing from a restart file: the working directory's restart files
  listed (and only those -- a bare glob would offer notes that merely
  mention the word), the selected one becoming a ``read_restart`` + ``run``
  pair in the editor with the step count from the dialog, cancelling
  writing nothing, and accepting with nothing selected saying so
- The 3D snapshot refused mid-run, reporting when the library was built
  without the VTK package, and refusing a deck that creates no system box

test_sweeprun.cpp
-----------------

Tests for the parametric sweep *driver* (``SweepController``) against a live
simulator.  ``test_sweeppanel.cpp`` above covers the panel with nothing
behind it; this covers what happens when the button is actually pressed,
which needs a running SPARTA and so had never been exercised.  That gap
mattered more than the line count suggested, because the driver's failure
mode is quiet: a wrong keyword-to-row match in ``readThermo()``, or an
off-by-one in the replicate cursor, yields a table that is entirely
self-consistent and entirely wrong.

Every assertion is therefore against an arithmetic answer.  The deck creates
exactly ``${n}`` particles, so ``Np`` is ``n`` and the table can be checked
digit for digit rather than merely for being present.  Registered as a
single ctest entry under Xvfb and serialised against the other live-window
suites.  Test cases cover:

- One row per combination, in the order the values were given, with the
  right count against the right variable value
- Headers naming the swept variable and each quantity with its reducer
- A cartesian sweep covering every pair with the second variable varying
  fastest, and a zip pairing them instead
- A quantity the run never produced tabulated as ``n/a`` rather than as a
  zero that looks like a measurement, and the same for a reducer with no
  samples to reduce
- All four reducers agreeing on a conserved quantity, over a run long
  enough for the stats poller to collect a series
- Replicates: one row per point with a mean and a standard error, a single
  replicate getting neither, and each replicate genuinely running with a
  distinct seed -- checked by naming the particle count as the seed
  variable, so four replicates read back as a mean of 41.5 and a spread of
  ``sqrt(5/3)/2``
- The progress bar counting every run rather than every sweep point, and
  reporting each one as it starts
- Stopping part way ending the sweep and saying it was stopped
- The window usable again afterwards -- the controller lets go of the run
  signals, so an ordinary run does not add a results row -- and a second
  sweep replacing the first one's table rather than appending to it

test_surfreportlive.cpp
-----------------------

Tests for the per-surface extraction path: ``SpartaWrapper::extractCompute``
and ``extractFix``.  These are the only wrapper calls that hand back a
pointer into SPARTA's own memory, and the surface report is their sole
consumer; ``test_surfreport.cpp`` covers the reduction core, but only ever
against hand-written arrays.  A wrong style constant, a transposed walk or a
stride mismatch would have produced a report full of plausible numbers.

The suite runs one flow past the circle fixture with a per-surf compute and
a fix averaging it, then drives the dialog against that finished state.  Two
checks pin the numbers: one sums a column of the exported CSV and compares
it to the integrated total the report printed, closing the loop through both
the library read and the reduction; the other walks the array independently,
in the test, because the first two halves would agree with each other even
under a transposed read.  Both are guarded against going vacuous -- the
``fx`` column has to contain elements of both signs.  Test cases cover:

- The computes and fixes the run actually defined offered as sources
- Column labels recovered from the deck, directly for a compute and through
  ``c_1[*]`` for the fix that averages it
- One row read per surface element, and the fix and the compute agreeing on
  how many there are
- The report naming the timestep it was taken at
- A source that is not per-surface refused rather than misread, missing
  labels asked for rather than guessed, and export offered only once there
  is something to export

test_vtkscene.cpp
-----------------

Tests for the 3D scene widget (``src/vtkscene.cpp``), built only when
SPARTA-GUI is configured with VTK.  Alongside the layer bookkeeping and the
pixel checks that say the scene is drawing something rather than an empty
grey rectangle, this covers the scene's *probes* -- ``applyLineProbe()``,
``onProbePick()`` and ``applyCutPlane()``.

Those report the only numbers the 3D viewer states as fact.  The filters
underneath are checked in ``test_vtkfilters.cpp``; what was never checked
is the scene's use of them, and a probe that resolves the wrong cell says a
plausible number with no way to notice.  The probe cases therefore run
against a field defined to be the x coordinate, so every sampled value has
an answer that can be written down.  They cover:

- The line spanning the domain and the plotted range matching the field
- The point probe's reading agreeing with the coordinate it says it sampled
  at, and going quiet once the tool is switched off
- A slice becoming its own layer, a plane that misses being reported rather
  than added empty, and cancelling changing nothing
- The refusals: no data loaded, no point field to sample, a cancelled choice

Writing them turned up a defect that made two features useless.
``PlotData::addColumn()`` appends the column name along with the column, so
calling ``setColumnNames()`` as well leaves that many empty columns in
front; ``rowCount()`` reads zero from the first of them and ``loadData()``
discards the whole table.  Both the line probe and the sweep panel's "Chart
Results" did exactly that, so both opened an empty chart every time -- which
reads as an analysis that found nothing rather than as a bug.  Both are now
covered, in this suite and in ``test_sweeprun.cpp`` respectively.

test_stlwizardlive.cpp
----------------------

Tests for the surface import wizard's SPARTA-facing half:
``boxGridCommands()``, ``renderViaSparta()`` and ``runSpartaWatertight()``.
These build the domain every surface-based simulation starts from, and
their failures are quiet -- a box that does not enclose the geometry, or a
grid that is not the resolution the wizard displayed, still runs and still
produces numbers.

So the assertions are against what SPARTA ended up with rather than what
the wizard emitted: the box read back through
``extractGlobal("boxlo"/"boxhi")`` and the cell count from SPARTA's own
output.  The tetrahedron fixture spans exactly 0..1, which makes the padded
box arithmetic, and a variant stretched in z pins the pad to the largest
extent in any axis -- invisible on anything cubic.  Test cases cover:

- The box enclosing the geometry with the right pad, and the pad following
  the longest axis rather than x and y alone
- The grid matching the resolution spin boxes, and following them when they
  change rather than reusing the first render's values
- A 2d surface file getting dimension 2, the standard z slab, and a single
  layer of cells whatever the z box says
- SPARTA accepting a closed surface and rejecting an open one, with the
  verdict saying which -- the authoritative answer, not the preflight's
- The diagnostics pane carrying SPARTA's own output, so a failed render can
  be diagnosed rather than merely noticed
- The render producing a picture with the surface actually in frame, no
  library reported rather than silently rendering nothing, temporary frames
  swept up afterwards, and a good surface still rendering after a bad one

test_paraviewdialog.cpp
-----------------------

Tests for the ParaView export dialog's conversion run (``runConversion()``
and ``onProcessFinished()``).  ``test_paraviewexport.cpp`` covers the pure
part -- which arguments a set of settings turns into; this covers the
orchestration around it.  It is the hand-off to external analysis, so its
failures leave the application looking fine and the data wrong somewhere
else; reporting a non-zero exit as success is the worst of them.

ParaView is not needed.  The tests supply a stub ``pvpython``: a script that
records the directory it ran in and every argument it was given, writes the
output it was asked for, and exits with the code the test chose.  Test cases
cover:

- The arguments handed to the interpreter matching the ones the builder
  produced, and the mode choosing which script runs
- The conversion running in the input file's directory, so relative output
  lands beside it
- The log carrying the tool's own output, and the button going busy and
  coming back
- Success naming what it wrote; a non-zero exit reported as a failure, not
  as done, and not followed by launching ParaView on a file that does not
  exist
- Refusals for a missing input, a missing interpreter and a missing script,
  none of which start anything
- Stale output in both directions: declining leaves the earlier file
  untouched and converts nothing, agreeing clears it before the script trips
  over it
- The tool paths remembered for the next session

test_movielive.cpp
------------------

The movie import against a real movie.  ``test_movieimport.cpp`` parses
ffprobe's JSON from hand-written strings; the half that talks to ffmpeg had
never run -- probing a container that stores no frame count, turning a
selection into a filter expression, and deciding afterwards whether the
extraction worked.

That half is where a silent wrong answer lives: the extracted frames are what
the user then measures, and a stride that is off by one produces a slide show
that looks entirely normal and is not the part of the trajectory that was
asked for.  So the fixture encodes movies whose frames are individually
identifiable -- each a flat colour derived from its number -- and reads the
colours back out of the extracted PNGs.  Test cases cover:

- Size, rate and duration of a real movie, and the packet-counting fallback
  for webm, which stores no frame count
- A text file and a missing file refused with a reason
- The selected range and stride coming back as exactly the right source
  frames, in order, as absolute paths; the whole movie; a single frame
- An impossible range, and an extraction that matched nothing, both reported
  rather than returned empty
- A failure clearing its own output *and* any frames an abandoned earlier
  import left in the directory -- otherwise the failure is reported and the
  slide show still finds a full-looking sequence of somebody else's frames

Recorded in the file header: the upper bound of the range is enforced twice,
by the select filter and again by ``-frames:v``, so breaking either alone
changes nothing observable.  That is belt and braces, not a gap.

test_wrapperload.cpp
--------------------

Loading the SPARTA library, and what the wrapper does when that fails.  Every
other suite uses a library that loads; this one uses the ones that do not: a
file that is not there, one that is not an ELF object, a truncated one, an ELF
header with no body, and a good one loaded on top of an open instance.

In plugin mode every library call goes through a function table, and that
table is absent from application start until the user has chosen a library --
so a call that is unguarded dereferences it to find the function and jumps
through the result.  Writing these found two such calls: ``extractVariable()``
guarded the extract but not the ``free()`` that follows it (itself a table
call), and the two extract paths guarded on the instance handle alone inside a
compound condition that the earlier sweep had not reached.  Both are fixed, and
an audit of every ``SPAFN`` use against its enclosing guard now comes back
clean.  Test cases also cover the truncated-library check that exists so a
corrupt download is refused rather than taken down inside the dynamic linker,
recovery after a rejected library, one instance at a time, and the port's
constant stubs answering the same with and without a simulator.

The file header records that the two safeguards keeping the instance and the
table in step -- ``loadLib()`` closing the instance before releasing the table,
and ``isOpen()`` requiring both -- are redundant with each other: removing
either alone leaves every check here passing.

test_exportimage.cpp
--------------------

Saving a snapshot to a file (``exportImage()``): the only way a rendered frame
leaves the application, used by both viewers and the chart window, and never
run.  Qt writes a handful of formats directly; for anything else the image goes
to a temporary PNG that ImageMagick converts to the name the user typed.  Test
cases cover the formats Qt writes, the converter fallback (an SGI, read back
through the converter to prove the file holds the image that went in), a
cancelled dialog writing nothing and complaining about nothing, no image at all
not even asking, an unwritable destination reported, and the error naming which
viewer was saving -- the same function serves three of them.

Two of the three failure branches around the converter are unreachable on a
Linux box that has ImageMagick, which the file header says rather than leaving
them looking like a gap: ``findExe()`` searches ``/usr/bin`` whatever ``PATH``
says, so "the converter is absent" cannot be produced; and ImageMagick answers
an unrecognised extension by falling back to its own default format and exiting
zero, so "the conversion failed" needs a destination that cannot be written at
all -- where there is nothing left to clean up either.

test_datafileplot.cpp
---------------------

Tests for plotting a data file: ``SpartaGui::plotDataFile()`` and
``ChartWindow::addDataFile()`` -- how a user gets somebody else's numbers onto
a chart, whether that is a reference curve, a previous run's output or a table
from a paper.  The parsers are covered in ``test_plotdata.cpp`` and the column
picker in ``test_plotdatadialog.cpp``; neither end of the path joining them had
been driven, because both sit behind a file dialog and then a second modal.
Nothing checked that the columns the user picked are the ones that reach the
chart -- and a curve from the wrong column is still a curve, labelled from the
same picker.  The fixture writes a table whose columns are separable on sight
(x, 2x, 10x), so the plotted range says which was used.  Test cases cover:

- The default selection becoming one chart per column, and an explicitly
  chosen y column or x column being the one plotted
- Both refusals -- a cancelled file dialog and a cancelled picker -- and an
  empty selection reported rather than opening an empty chart
- A file that is not a table, and a directory, refused with a reason
- The overlay half: a second file drawn onto the chart already shown and
  included in its range, both of its refusals, and two overlays coming out in
  different colours (asked of the pixels, since no accessor exposes a series
  colour)

test_imageviewerinput.cpp
-------------------------

Tests for steering the snapshot with the mouse (``ImageViewer::eventFilter``):
drag to rotate, shift-drag to pan, wheel to zoom.  It is how the viewer is
actually used, and the live screenshot suites can only say the picture changed
-- not that a drag to the right turned the camera to the right, or by how much.

The view state is private, so every check reads it back out of the ``dump
image`` command the viewer emits for the clipboard: the same state the render
uses, and the form the user can paste into a deck.  Settings still at their
default are left out of that command, so the readers fall back to the
documented defaults rather than treating "absent" as "missing".  Test cases
cover:

- A rotation in the direction of the drag, proportional to its length, and
  reversible by dragging back
- A vertical drag changing the elevation without also turning the azimuth
- A two-dimensional view never asking for camera angles, which SPARTA would
  ignore
- Shift-dragging panning instead of rotating, and the centre staying inside
  the box however far it is pushed
- The wheel zooming by a fixed factor per notch, reversibly, within its limits
- A drag that began on another widget, and a move after the button was
  released, both ignored
- Reset undoing every gesture

test_inspect.cpp
----------------

Tests for reading a restart file (``SpartaGui::inspectFile()`` and
``purgeInspectList()``).  This is the one place the application opens somebody's
saved simulation without running it, and it is not a read-only path: it loads
the file into the live SPARTA instance, replacing what was there, and writes a
temporary log beside the user's file.  Test cases cover:

- A file refused by signature rather than by name -- an input deck called
  ``.restart`` must not be half-read into the live instance
- The summary window showing SPARTA's own account of the grid and the particles,
  and naming the file it came from
- The temporary ``.info.log`` not surviving beside the user's file
- The fixed seed inspection supplies, without which rendering a restored state
  that defines a collide style fails
- The window still usable afterwards, since inspection clears the instance
- The purge contract in both directions: windows the user still has open survive
  the next inspection, the ones they closed are collected

Writing it turned up a crash on quit.  The inspect windows were never cleaned
up in ``~SpartaGui``, so Qt destroyed them as children in ``~QWidget`` -- after
the ``sparta`` member was gone -- and the Hide event reaching their event filter
called the simulator through a wrapper that no longer existed.  Removing the fix
crashes this suite, which is how it stays fixed.

test_recovery.cpp
-----------------

Tests for crash recovery: the autosave copy of an unsaved buffer and the
offer to restore it on the next launch.  Its failure mode is invisible until
the session it was meant to survive, and it writes files on a timer next to
the user's own deck, so both directions need checking.  Everything goes
through the real triggers -- the write happens on the autosave timer (set to
one second), the offer happens in the constructor, and the clear happens on
save -- rather than through test-only entry points.  Test cases cover:

- An unsaved buffer autosaved with its text, and a manifest recording where
  it really belongs and when it was written
- The user's own file on disk never touched, however long the editor has
  been left modified
- An unmodified buffer and a whitespace-only one not autosaved at all
- Saving dropping the recovery copy, since the buffer now matches disk
- The offer restoring the buffer, the file it came from, and the modified
  flag -- without which closing would discard recovered work silently
- The offer naming the file and the time, or saying "an unsaved buffer" when
  there is no filename; declining discarding the copy so it is not offered
  again; and nothing asked when there is nothing to recover
- A recovery file whose manifest was lost to a crash between the two writes
  still recovered, and recovered work still being autosaved afterwards
- The new-document guard: save, discard, cancel, and no question at all for
  an unmodified buffer

test_checksum.cpp
-----------------

Tests for the integrity check on downloaded files
(``src/urldownloader.cpp``).  This is the only code that decides whether a
file fetched off the network may be kept, and none of it had ever run.  A
broken integrity check is indistinguishable from a working one from outside:
a parser that never finds the entry, or a comparison that always agrees,
downgrades the check to nothing while still reporting success.

The tests run over ``file://`` URLs, so the real fetch-parse-compare path
runs end to end with no network and no server -- ``QNetworkAccessManager``
treats a local directory as the remote one, ``SHA256SUMS`` and all.  Test
cases cover:

- The local hash pinned against the published SHA-256 of ``""`` and
  ``"abc"``, a one-byte change giving a different hash, and an unreadable
  file having none
- The ``SHA256SUMS`` spellings real tools emit: one space, two spaces, ``*``
  and ``./`` prefixes, uppercase hex, comments, blank and malformed lines
- An entry for another file not accepted as this one's, including one whose
  name merely *ends* with the name being looked up
- A matching hash keeping the file silently; a mismatch refusing it,
  deleting it, and showing the user both hashes
- The fail-open case pinned explicitly: a publisher who ships no
  ``SHA256SUMS`` cannot be checked, so the download is kept -- changing that
  should be a decision rather than an accident

test_chartanalysis.cpp
---------------------

Tests for what ``ChartWindow::postProcess()`` does *after* its dialog is
answered: the seven analyses that read the chart's data, compute something,
and put the answer back on the chart as a fit curve or a reference line.
Splitting the dialog out left this half still needing a chart with data in
it -- which a ``ChartWindow`` loaded from a ``PlotData`` is, with no
simulator involved.  A timer-driven helper fills the dialogs and records
the text of the report that follows.  Test cases cover:

- Refusing a chart with fewer than two points, and doing nothing at all
  with no chart selected
- Autocorrelation opening a window of its own (the abscissa becomes lag, so
  the result cannot share the chart's axes), named after its series and
  running to the requested maximum lag; and a constant series reporting
  that it has none
- Polynomial fit recovering the coefficients of a known line, renaming the
  processed slot, honouring the fit x-range, and falling back to the whole
  series with a warning when the range holds fewer than two points
- A custom function plotted over the data range, and an expression that
  does not parse refused with its reason and no curve drawn
- A custom nonlinear fit recovering the amplitude and rate of a known
  exponential from a deliberately wrong starting guess, taking the label it
  was given, shortening a long one, and naming itself after its expression
  when unlabelled; plus malformed parameters and a fit that cannot converge
- Block-average uncertainty reporting the mean, block-averaged standard
  error, integrated autocorrelation time and effective sample count
- Steady-state detection reporting a burn-in cutoff, the post-burn-in mean
  and how much data survived
- The Birch-Murnaghan fit asking for the atom count, reporting every fitted
  quantity, deriving a₀ = ∛(N × V₀) from its own V₀, and refusing data with
  no minimum in it
- Cancelling the analysis leaving the chart untouched

test_mainwindowfiles.cpp
------------------------

Tests for the main window's File menu and the workers behind it.
``test_mainwindow.cpp`` keeps a list of actions it must not trigger because
each opens a modal nobody can answer, and nearly all of them are the file
actions -- which is why opening, saving, viewing and inspecting a file were
the largest uncovered block in the application.  Two things make them
reachable: the workers take a path directly, and with
``AA_DontUseNativeDialogs`` the ``QFileDialog`` is an ordinary widget that a
timer can hand a filename and accept.  Test cases cover:

- Opening a deck into the editor, making its folder the working directory
  (every relative path in the deck resolves against it), and adding it to
  the recent list
- Opening nothing, opening a file that is not there, and opening over
  unsaved edits asking first
- Writing the buffer, adding a final newline when it lacks one without
  doubling one it has, clearing the modified flag, retitling the window,
  and a round trip through disk being lossless
- A failed write reporting why and leaving the window title and the
  modified flag alone
- Save reusing the name the deck already has without asking, where Save As
  always asks; and cancelling either writing nothing
- Viewing a text file in a read-only window, and cancelling opening nothing
- The image action opening a slide show on what it was given
- Inspecting nothing, and something that is not a restart file being
  refused rather than opening an empty inspection window
- The snippet picker and the About box coming up and going away
- Every file action triggered back to back, with the window still intact

test_chartdialogs.cpp
--------------------

Tests for the chart window's five modal dialogs
(``src/chartdialogs.{h,cpp}``): the chart style editor, the postprocess
analysis picker, the Birch-Murnaghan column setup and its result, and the
reference-line editor.  Each was built on the stack inside a
``ChartWindow`` method and read a live ``ChartViewer`` as it went, so none
could be constructed without a chart with data in it.  Split out, each is a
pure function of the plain struct it is handed, and the target links three
leaf sources -- no chart, no simulator, no display.  Test cases cover:

- **Chart style**: a full round trip of a populated ``ChartStyle``, the raw
  and processed sections proved to be separate controls (both are built
  from the same three helpers, so a copy-paste slip pointing both at the
  raw widgets would otherwise go unnoticed), every display mode and every
  legend placement surviving the trip, an unset colour becoming a visible
  default rather than staying invalid, and the width and marker-size
  ranges enforced
- **Postprocess**: the seven analyses offered in their documented order,
  the default max lag at half the series, the fit range defaulting to the
  data range, the polynomial degree capped by the point count (a degree-n
  polynomial needs n+1 points) and by 8, the block count starting at
  sqrt(N), the parameter spin box relabelled per analysis, and the
  expression, parameter, label and fit-range controls shown for exactly
  the analyses that use them -- cross-checked against
  ``PostProcessSpec::usesFitRange()``
- **Birch-Murnaghan**: the columns it will treat as volume and energy, the
  atom count defaulting to 1 and bounded below, the lattice constant
  a₀ = ∛(N × V₀), and every fitted quantity reported and selectable
- **Reference lines**: a round trip of the lines and the label style, rows
  added and removed with a removed row proved gone from the answer, the
  anchor item texts following the orientation without moving the
  selection, every anchor surviving the trip, an invalid colour becoming
  the default grey, labels trimmed, and the style controls bounded
- **The overlay palette**: adjacent series differing, wrapping rather than
  running out, a negative index still giving a colour, and the palette
  avoiding the raw and processed series colours so an overlaid file does
  not look like part of the chart

test_chartviewer.cpp
-------------------

Tests for the chart window (``src/chartviewer.{h,cpp}``) -- both the dock
panel that plots live thermo output and the standalone plot window the same
class becomes when handed a data file.  Constructed with a null
``SpartaGui`` it needs no simulation, which is exactly the standalone mode
"File > Plot data file" opens.  Test cases cover:

- The live path: charts added and made selectable, data routed to the chart
  whose thermo column it belongs to, data for an unknown column dropped,
  and the window cleared and reused
- The file path: one chart per selected column, a reload replacing rather
  than appending, and the degenerate arguments -- an empty table, no
  Y columns, an X column out of range, a Y column out of range skipped
  rather than plotted, and a single-row table
- Switching charts restoring that chart's Y-axis label while the shared
  title and X-axis label stay put
- All three plotted-series choices, with the Savitzky-Golay window and
  order live only while smoothing and applied to hidden charts too
- The range sliders narrowing the view, and both range slots being
  harmless on an empty window
- The rendered chart checked for ink, not merely for not crashing, and an
  empty chart rendering as well

test_codeeditor.cpp
-------------------

Tests for the input editor (``src/codeeditor.{h,cpp}``), the widget a user
types their deck into.  Its text transforms are what silently damage a deck
if they are wrong, and were previously reachable only through the live GUI
walker, which types into the editor but never checks what came out.  Test
cases cover:

- ``reformatLine()``: a command padded to the command column, runs of
  whitespace collapsed, a lone command left without trailing padding, a
  comment line left exactly as it is, the ID and style of
  ``fix``/``compute``/``dump`` padded to their own columns, the species
  name of a ``mixture`` padded, and the whole transform idempotent so that
  repeated reformatting does not walk the columns sideways
- Reformatting the current line leaving every other line alone
- Commenting and uncommenting, one line and a selection: only the first
  ``#`` removed, leading whitespace skipped to find it without being eaten,
  an uncommented line unchanged, and a mixed selection touching only the
  commented lines
- The completers built by parsing the buffer -- groups (always including
  ``all``, each name listed once), variables (every reference form, with
  the bare ``$x`` form only for single-character names), compute and fix
  IDs, and mixtures -- and all of them surviving an empty buffer
- The error-line highlight and the diagnostic overlay set and cleared,
  including an out-of-range line
- The line-number gutter widening as the line count gains digits
- Pasting through ``paste()``, the route ``Ctrl+V`` takes: text inserted,
  multiple lines kept separate, and an image on the clipboard refused

The same file covers the completion machinery and the context menu.
``runCompletion()`` is a dispatch table -- which of the seventeen completers
applies depends on the position of the word under the cursor and on the
command that starts the line, with the ``c_``/``f_``/``v_`` reference
prefixes applying where the command has no list of its own.  Getting an
entry wrong offers the user the wrong list, which looks like the feature
working.  Test cases cover each word position, every command that has a list
at it, the reference prefixes (upper and lower case), a command's own list
taking precedence at the third word, the directory listing disappearing once
a path separator is typed, and nothing at all on an empty or commented line.
Accepting a completion is driven through the completer's own ``activated``
signal, which is the path the popup takes, including the check that refuses
a completion from another editor's completer.

The context menu is caught as it is shown and its entries read back: comment
entries for the line or the selection, documentation for the command on the
line and -- for a styled command like ``fix ID ave/time`` -- for both the
style and the command, a file name under the cursor offered for viewing (and
for editing unless it is binary), and no run entries at all when there is no
main window behind the editor.


test_fileviewer.cpp
-------------------

Tests for the read-only file viewer (``src/fileviewer.{h,cpp}``), the
shell window the image viewer and slide show share
(``src/viewerwindow.{h,cpp}``), and the two-colour band slider
(``src/rangebandslider.{h,cpp}``).  Test cases cover:

- Plain text shown read-only and unwrapped, positioned at the top rather
  than the end, and titled after its file unless given a title
- A file that cannot be opened reporting the reason in the window, which
  is what a user sees when a run wrote no output
- On-the-fly decompression through ``gzip``, ``bzip2``, ``xz``, ``lzma``
  and ``zstd``, each skipping if that program is not installed; the
  ``lzma`` case is the one entry needing an extra argument
- A file whose suffix is not in the compression table read as plain text,
  and a misnamed ``.gz`` shown as the text it actually is (the ``-f`` flag
  makes the decompressors copy input they cannot decompress straight
  through)
- ``Ctrl+W`` closing the viewer and ``Ctrl+/`` being harmless when there is
  no main window behind it
- The band slider painting an inverted, empty and out-of-range band, and a
  scale whose minimum equals its maximum

test_movieimport.cpp
--------------------

Tests for the parsing and frame-counting helpers of the movie import
(``src/movieimport.{h,cpp}``).  These are free functions, so the tests run
without ``ffprobe`` or ``ffmpeg`` installed.  Test cases cover:

- ``parseFrameRate()``: rational (``30000/1001``) and plain numbers,
  invalid input
- ``selectedFrameCount()``: full range, range with interval, invalid ranges
- ``parseProbeOutput()``: frame count taken from the container, missing
  frame count or frame size, absent video stream, malformed JSON, and
  numeric fields given as JSON numbers instead of strings

The import dialog itself is covered too.  It needs no ``ffmpeg``: the sample
frame it would decode to calibrate its size estimate simply comes back
empty, which is the branch every user without ``ffmpeg`` on their PATH
gets.  Test cases cover:

- Opening preselected on the whole movie, and showing what was probed
- The frame range refusing to invert, in either direction
- The extracted-image count following the range and the interval
- The size reported as "unknown" rather than as a confident zero when
  nothing calibrated the estimate
- The warning raised by a very large extraction, and cleared again when the
  selection comes back under the threshold
- A single-frame movie

test_imagecache.cpp
-------------------

Tests for the :cpp:class:`ImageCache` class (``src/imagecache.{h,cpp}``),
the temporary-directory backed store for converted images and extracted
movie frames owned by the Slide Show window.  Test cases cover:

- Qt-readable formats are loaded directly and never cached
- Exotic formats are converted at most once; changed source files are
  converted again
- Missing files return a null image; unreadable files are reported once
  and not retried until the file changes
- Usage totals track the converted images
- ``forget()`` drops the conversion of a deleted file; purging conversions
  keeps extracted movie frames and failure records
- Cache subdirectories are unique and sanitized
- ``clear()`` and the destructor remove the temporary directory

test_plotdata.cpp
-----------------

Tests for the column-oriented ``PlotData`` model and the parsers and
writers for external data files (``src/plotdata.{h,cpp}``).  Test cases
cover:

- Appending rows and columns to the model
- CSV import with and without a header line
- Whitespace-separated (``.dat``) import with a SPARTA-style header
- YAML data files, including trailing commas, interleaved log
  lines, and a sequence of maps
- JSON import as array-of-rows and object-of-arrays, with error handling
  for unequal columns and malformed input
- Dispatch by file extension and content-based YAML detection in log files
- CSV, ``.dat``, and YAML export round-trips, including YAML quoting rules

test_plotaxismath.cpp
---------------------

Tests for the Qt-free chart axis-layout helpers
(``src/plotaxismath.{h,cpp}``).  Test cases cover:

- ``niceTickInterval()``: 1-2-5-10 interval selection, scaling across
  powers of ten, degenerate ranges, and non-positive tick targets
- ``tickValues()``: even spacing, anchor alignment, snapping zero, endpoint
  inclusion despite floating-point rounding, reversed ranges, custom
  anchors
- ``tickDecimals()``: decimal counts for integer and fractional spacings
- ``formatAxisLabel()``: printf-style integer and floating-point
  specifiers, length-modifier normalization, literal prefix and suffix
  text, and fallback behavior for empty or placeholder-free formats

test_leastsquares.cpp
---------------------

Tests for the dense linear-algebra and smoothing routines
(``src/leastsquares.{h,cpp}``).  Test cases cover:

- Matrix transpose, multiplication, and inversion
- LU linear solve with single and multi-column right-hand sides
- Savitzky-Golay smoothing: moving-average behavior for constant data,
  exact preservation of linear and quadratic data at matching polynomial
  degrees, and noise reduction around a line

test_analysis.cpp
-----------------

Tests for the post-processing analyses (``src/analysis.{h,cpp}``).  Test
cases cover the normalized autocorrelation function: an exact small case,
lag zero being one, empty results for constant or too-short series,
clamping of the maximum lag, and anticorrelation of an alternating series.

test_fitting.cpp
----------------

Tests for the linear-least-squares curve fits (``src/fitting.{h,cpp}``).
Test cases cover recovering known polynomial models and evaluating the
fitted polynomial, recovering a known Birch-Murnaghan equation-of-state
model, and the failure paths for too few data points and non-positive
volumes.

test_levmar.cpp
---------------

Tests for the Levenberg-Marquardt nonlinear least-squares solver
(``src/levmar.{h,cpp}``).  Test cases cover recovering linear,
exponential-decay, and Gaussian models, fitting noisy data, rejecting
underdetermined problems (more parameters than residuals), and reporting a
failing initial model evaluation as an error instead of crashing.

test_lepton.cpp
---------------

Tests for the vendored LeptonMini expression parser
(``thirdparty/lepton_mini``).  Test cases cover expression evaluation,
error handling for invalid expressions, verification of symbolic
derivatives, custom functions, and expression optimization.

test_customfunc.cpp
-------------------

Tests for the custom-function evaluation and fitting layer
(``src/customfunc.{h,cpp}``) built on LeptonMini.  Test cases cover:

- Evaluating user expressions (polynomials, trigonometric functions,
  constants, custom variables) over a sample range
- Skipping non-finite points and clamping the sample count
- Error handling for empty expressions, syntax errors, and undefined
  variables
- Nonlinear custom fits recovering exponential-decay and quadratic models
- Fit-setup validation: variable/parameter clashes, duplicate parameters,
  undeclared symbols, and too few data points

Command-Line Tests
==================

These tests validate the ``sparta-gui`` executable behavior without starting
the full GUI. They run quickly and are useful for CI/CD pipelines.

CommandLine.GetVersion
-----------------------

**Purpose**: Verify version reporting consistency

This test runs::

  sparta-gui --platform offscreen -v

and validates that:

- The executable launches successfully
- Version output includes "SPARTA-GUI (QT6)"
- Version number matches the ``PROJECT_VERSION`` CMake variable
- Process exits cleanly with status 0

**Environment**: ``OMP_NUM_THREADS=1`` to ensure consistent behavior

CommandLine.HasPlugin
----------------------

**Purpose**: Verify build configuration is reflected in help text

This test runs::

  sparta-gui --platform offscreen -h

and validates that help text is consistent with CMake configuration:

- **Plugin Mode** (``SPARTA_GUI_USE_PLUGIN=ON``): Help text includes
  "-p, --pluginpath <path>" option
- **Linked Mode** (``SPARTA_GUI_USE_PLUGIN=OFF``): Help text omits
  plugin path option

**Environment**: ``OMP_NUM_THREADS=1`` to ensure consistent behavior

GUI Tests
=========

These tests validate SPARTA-GUI functionality using PyAutoGUI and Xvfb (virtual
frame buffer). They run the actual GUI application in a headless X server
environment, allowing automated interaction and screenshot capture.

**Important Note**: The argument for the screen number flag ``-n`` for ``xvfb-run``
*must* be different for each test, so that the tests may run in parallel.

Framebuffer.CreateScreenshot (test_shooter.py)
-----------------------------------------------

**Purpose**: Test the screenshot wrapper utility that abstracts different
screenshooter applications

**Test File**: ``test/test_shooter.py``

This test validates the ``shooter`` wrapper script that provides a unified
interface to various Linux screenshot utilities (ImageMagick's ``import``,
``magick import``, ``xfce4-screenshooter``, ``gnome-screenshooter``).

The test runs:

.. code-block:: bash

   xvfb-run -n 11 -s "-screen 0 1024x768x24" -w 1 python test_shooter.py

within a virtual frame buffer and validates:

**ScreenshotChecks.testCreateImage**
  - The ``shooter`` command executes without errors
  - A PNG file is created at the specified path
  - The image dimensions match the virtual frame buffer size (1024x768)
  - The image format is PNG
  - The screenshot captures an all-black screen (expected for empty Xvfb)
  - Specific pixel values at multiple locations are (0,0,0) RGB

**Dependencies**:
  - PyAutoGUI - for screen size detection
  - Pillow (PIL) - for image file validation
  - One of: ImageMagick (``import`` or ``magick``), ``xfce4-screenshooter``,
    or ``gnome-screenshooter``

**Setup/Teardown**:
  - ``setUp()``: Removes leftover ``shot.png`` from previous runs
  - ``tearDown()``: Cleans up ``shot.png`` after test completion

**Environment**: Virtual frame buffer at 1024x768x24, ``PYTHONUNBUFFERED=1``,
``PYTHONDONTWRITEBYTECODE=1``, ``OMP_NUM_THREADS=1``

Framebuffer.CheckSize (test_xvfbsize.py)
-----------------------------------------

**Purpose**: Verify PyAutoGUI functionality and Xvfb screen size configuration

**Test File**: ``test/test_xvfbsize.py``

This test validates that PyAutoGUI can properly interact with the virtual
frame buffer created by Xvfb, which is essential for GUI automation tests.

The test runs:

.. code-block:: bash

  xvfb-run -n 12 -s "-screen 0 1024x768x24" -w 1 python test_xvfbsize.py

within a virtual frame buffer and validates:

**PyAutoGUIChecks.testScreenSize**
  - PyAutoGUI correctly detects the screen dimensions
  - Screen width is 1024 pixels
  - Screen height is 768 pixels

**PyAutoGUIChecks.testMousePosition**
  - PyAutoGUI can detect the mouse cursor position
  - Initial mouse position is at screen center (512, 384)
  - ``pyautogui.moveTo()`` can move cursor to absolute positions
  - ``pyautogui.moveRel()`` can move cursor by relative offsets
  - Position queries return expected coordinates after moves

**Dependencies**:
  - PyAutoGUI - for screen size detection and mouse control

**Environment**: Virtual frame buffer at 1024x768x24, ``PYTHONUNBUFFERED=1``,
``PYTHONDONTWRITEBYTECODE=1``, ``OMP_NUM_THREADS=1``

Framebuffer.GUIEditorChecks (test_gui_edit.py)
-----------------------------------------------

**Purpose**: Test basic SPARTA-GUI editor functionality using automated
GUI interactions

**Test File**: ``test/test_gui_edit.py``

This test validates fundamental editor operations in SPARTA-GUI by launching
the application inside a virtual frame buffer and automating user interactions
with PyAutoGUI. The test uses screenshot comparison to verify visual state.

The test runs:

.. code-block:: bash

   xvfb-run -n 13 -s "-screen 0 1024x768x24" -w 1 python test_gui_edit.py

within a virtual frame buffer and validates:

**GUIEditorChecks.testExitShortcut**
  - SPARTA-GUI launches and displays a white editor background
  - The ``Ctrl-Q`` keyboard shortcut exits the application cleanly
  - The process exits with status 0
  - Screenshots confirm the window was displayed and then closed

**GUIEditorChecks.testExitMenu**
  - The ``Alt-F`` menu shortcut opens the File menu
  - The ``Q`` key selects the Quit entry
  - The application exits cleanly with status 0
  - Screenshots confirm expected visual state

**GUIEditorChecks.testExitModCancelNo**
  - Text can be typed into the editor buffer
  - Exiting with a modified buffer shows a confirmation dialog
  - The ``Cancel`` option returns to the editor without exiting
  - The ``No`` option exits without saving

**Dependencies**:
  - PyAutoGUI - for keyboard and mouse automation
  - Pillow (PIL) - for screenshot validation
  - A supported screenshooter application

**Setup/Teardown**:
  - ``setUp()``: Launches SPARTA-GUI, cleans up leftover test files
  - ``tearDown()``: Terminates the SPARTA-GUI process

**Environment**: Virtual frame buffer at 1024x768x24, ``PYTHONUNBUFFERED=1``,
``PYTHONDONTWRITEBYTECODE=1``, ``OMP_NUM_THREADS=1``,
``SPARTA_GUI=<path to executable>``

Test Fixtures and Utilities
============================

**HelpersTest Fixture**
  Base test fixture that creates a ``QCoreApplication`` instance for tests
  that require Qt functionality. The application is created once per test
  suite and reused across tests for efficiency.  Similar lightweight
  fixtures (``StdCaptureTest``, ``FlagWarningsTest``) are used by the
  matching unit test programs.

**Platform-Specific Testing**
  Tests use conditional compilation (``#ifdef _WIN32``) to adapt to
  platform differences in:

  - Path separators
  - Line endings
  - Available system executables
  - Default shell commands

Future Test Expansion
=====================

Planned additions to the test suite include:

**GUI Component Tests**
  - CodeEditor text manipulation
  - Syntax highlighter accuracy
  - Find/replace functionality
  - Auto-completion behavior

**SPARTA Integration Tests**
  - SpartaWrapper command execution
  - Variable substitution
  - Error handling
  - Output capture

**File I/O Tests**
  - File opening/saving
  - Recent files management
  - Auto-save functionality
  - Restart file inspection

**Preferences Tests**
  - Settings persistence
  - Default value initialization
  - Migration between versions

Adding Tests
^^^^^^^^^^^^

Create a New Test File
======================

1. Create a new test file in the ``test/`` directory (e.g., ``test_newfile.cpp``)
2. Add the test executable to ``test/CMakeLists.txt``:

.. code-block:: cmake

   add_executable(test_newfile
     test_newfile.cpp
     ${CMAKE_SOURCE_DIR}/src/newfile.cpp
   )

   target_include_directories(test_newfile PRIVATE
     ${CMAKE_SOURCE_DIR}/src
   )

   target_link_libraries(test_newfile
     GTest::gtest_main
     Qt6::Widgets
   )

   gtest_discover_tests(test_newfile)


Add Tests to Existing File
==========================

Add new test cases using GoogleTest macros:

.. code-block:: cpp

   TEST_F(HelpersTest, NewTestName)
   {
       // Arrange
       std::string input = "test data";

       // Act
       auto result = function_to_test(input);

       // Assert
       EXPECT_EQ(result, expected_value);
   }

Dependencies
^^^^^^^^^^^^

- **GoogleTest**: Automatically fetched via CMake FetchContent (v1.17.0)
- **Qt6**: Required for Qt-dependent functions (Widgets component)
- **CTest**: Part of CMake, used for test execution

Notes
^^^^^

- Tests that require a Qt application context use a ``HelpersTest`` fixture that creates a ``QCoreApplication`` instance.
- Platform-specific tests (e.g., ``has_exe``) use conditional compilation to test appropriate commands on different operating systems.
- The test suite is designed to be easily extended with additional test files and test cases.
- GoogleTest is fetched automatically during CMake configuration, so no manual installation is required.

CI Integration
^^^^^^^^^^^^^^

The test suite integrates with existing CI workflows:
- Tests run as part of the standard build process when ``ENABLE_TESTING=ON``
- CTest provides standard output for CI systems
- Tests can be disabled for documentation-only builds
