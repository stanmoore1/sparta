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
