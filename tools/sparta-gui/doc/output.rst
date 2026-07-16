************************
Monitoring SPARTA output
************************

.. _logfile:

Output Window
^^^^^^^^^^^^^

.. index:: output window
.. index:: log window
.. index:: screen output

By default, when starting a run, an *Output* window opens that displays
the screen output of the running SPARTA calculation.  This is the text
that would normally be seen in the command-line window, including the
columns of `stats output <https://sparta.github.io/doc/stats_style.html>`_.

.. TODO screenshot: capture the Output window during or after a SPARTA
   run as JPG/sparta-gui-log.png, then re-enable this figure.
..
.. .. image:: JPG/sparta-gui-log.png
..    :align: right
..    :scale: 50%

SPARTA-GUI captures the screen output from SPARTA as it is generated and
updates the *Output* window regularly during a run.  If there are any
warnings or errors in the SPARTA output, they are highlighted by using
bold text colored in red.  There is a small panel at the bottom center
of the *Output* window showing how many warnings and errors were
detected and how many lines the entire output has.  By clicking on the
button on the right with the warning symbol or by using the keyboard
shortcut `Ctrl-N` (`Command-N` on macOS), you can jump to the next
line with a warning or error.

By default, the *Output* window is replaced each time a run is started.
The runs are counted and the run number for the current run is displayed
in the window title.  It is possible to change the behavior of
SPARTA-GUI in the preferences dialog to create a *new* *Output* window
for every run or to not show the current *Output* window.  It is also
possible to show or hide the *current* *Output* window from the *View*
menu.

The text in the *Output* window is read-only and cannot be modified, but
keyboard shortcuts to select and copy all or parts of the text can be
used to transfer text to another program. Also, the keyboard shortcut
`Ctrl-S` (`Command-S` on macOS) is available to save the *Output* buffer to a
file.  The "Select All" and "Copy" functions, as well as a "Save Log to
File" option are also available from a context menu by clicking with the
right mouse button into the *Output* window text area.

Should the *Output* window contain embedded YAML format text (for
example produced by suitable `print
<https://sparta.github.io/doc/print.html>`_ commands in the input), the
keyboard shortcut `Ctrl-Y` (`Command-Y` on macOS) is available to save
only the YAML parts to a file.  This option is also available from a
context menu by clicking with the right mouse button into the *Output*
window text area.

.. _charts:

Charts Window
^^^^^^^^^^^^^

.. index:: charts window
.. index:: plotting
.. index:: stats output
.. index:: data visualization

By default, when starting a run, a *Charts* window opens that displays
plots of the `stats output <https://sparta.github.io/doc/stats_style.html>`_
columns of the SPARTA calculation.

.. TODO screenshot: capture the Charts window with a stats column of a
   SPARTA example run plotted as JPG/sparta-gui-chart.png, then
   re-enable this figure.
..
.. .. image:: JPG/sparta-gui-chart.png
..    :align: right
..    :scale: 33%

.. index:: smoothing
.. index:: Savitzky-Golay filter

The "Data:" drop-down menu on the top right allows selection of the
different columns that are computed and written as stats output to the
output window.  Only one column can be shown at a time.  The plots are
updated regularly with new data as the run progresses, so they can be
used to visually monitor the evolution of available properties.  The
update interval can be set in the *Preferences* dialog.  By default,
the raw data for the selected property is plotted as a blue graph.
From the "Plot:" drop-down menu on the second row (immediately to the
right of the *Chart Style...* and *Postprocess...* quick-access
buttons), you can select whether to plot only the raw data graph, only
a smoothed data graph, or both graphs on top of each other.  The
smoothing process uses a `Savitzky-Golay convolution filter
<https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter>`_.  The
convolution window width (left) and order (right) parameters can be set
in the boxes next to the drop-down menu.  Default settings are 10 and 4
which means that the smoothing window includes 10 points each to the
left and the right of the current data point for a total of 21 points
and a fourth order polynomial is fitted to the data in the window.

Both axes can be switched independently between linear and logarithmic
scaling.  Logarithmic axes are useful for quantities that decay or
grow over many orders of magnitude during a DSMC simulation, for
example while approaching a steady state.

The "Title:" and "Y:" input boxes let you edit the text shown as the
plot title and the y-axis label, respectively.  The text entered in the
"Title:" box is applied to *all* charts, while the "Y:" text changes
only the y-axis label of the currently *selected* plot.  In standalone
plot mode (when plotting an external data file), an additional "X-Axis:"
input box is shown to the right of "Y:" and sets the x-axis label for all
charts.

The window title shows the current run number that this chart window
corresponds to.  Same as for the *Output* window, the chart window is
replaced on each new run, but the behavior can be changed in the
*Preferences* dialog.

.. index:: CSV export
.. index:: YAML export
.. index:: data export

From the *File* menu on the top left, it is possible to save an image
of the currently displayed plot or export the data in either plain text
columns (for use by plotting tools like `gnuplot
<http://www.gnuplot.info/>`_ or `grace
<https://plasma-gate.weizmann.ac.il/Grace/>`_), as CSV data which can be
imported for further processing with Microsoft Excel, `LibreOffice Calc
<https://www.libreoffice.org/>`_, or with Python via `pandas
<https://pandas.pydata.org/>`_, or as YAML which can be imported into
Python with `PyYAML <https://pyyaml.org/>`_ or pandas.

Stats output data from successive run commands in the input script is
combined into a single data set unless the format, number, or names of
output columns are changed with a `stats_style
<https://sparta.github.io/doc/stats_style.html>`_ or a `stats_modify
<https://sparta.github.io/doc/stats_modify.html>`_ command, or the
current time step is reset with `reset_timestep
<https://sparta.github.io/doc/reset_timestep.html>`_.

.. index:: chart style
.. index:: legend

Adjust chart style
------------------

The *Chart Style...* entry in the chart window's *File* menu, or the
chart-style quick-access button at the far left of the second toolbar
row, opens a dialog to change how the data is drawn.  The *Raw data* and
*Processed data* series each have independent settings for the display
style (*Lines*, *Points*, or *Lines and Points*), the color, the line
width, and the point size.  This makes it possible, for example, to show
the raw data as faint points and the smoothed curve as a bold line.  The
*Legend* placement selector in the same dialog adds an in-plot legend
that lists the visible named series; it can be turned *Off* or anchored
to any of the four plot corners (*Top left*, *Top right*, *Bottom
right*, or *Bottom left*).  The selected placement is remembered across
sessions.

.. index:: reference lines

Reference lines
---------------

The *Reference Lines...* entry in the chart window's *File* menu opens a
dialog for adding straight annotation lines that are drawn on *every*
chart in the window.  Each line is either *Vertical* (at a chosen x
value) or *Horizontal* (at a chosen y value) and can carry a text label
and an individually chosen color.  The label position along the line
(*Top*, *Center*, or *Bottom* for vertical lines; *Left*, *Center*, or
*Right* for horizontal lines) is selected per line, while the label font
size, the gap between the label and the line, and whether the labels are
drawn in an opaque box apply to the whole window.  Reference lines are
useful, for example, to mark a target value, the end of the
equilibration phase of a simulation, or a fitted value.

.. index:: post-processing
.. index:: curve fitting
.. index:: autocorrelation
.. index:: polynomial fit
.. index:: custom function
.. index:: custom fit

Post-process data
-----------------

The *Postprocess...* entry in the chart window's *File* menu, or the
quick-access button immediately to the right of the *Chart Style...*
button, runs an analysis on the data of the currently selected property.
The following analyses are available:

- *Autocorrelation* computes the normalized autocorrelation function of
  the selected data up to a chosen maximum lag and shows it in a new
  chart window (the abscissa becomes the lag).  This is useful, for
  example, for estimating correlation times of fluctuating quantities.
- *Polynomial fit* performs a least-squares fit of a polynomial of a
  chosen degree, overlays the fitted curve on the chart, and reports the
  coefficients and the root-mean-square residual.
- *Birch-Murnaghan EOS fit* fits a 4-parameter `Birch-Murnaghan equation
  of state
  <https://en.wikipedia.org/wiki/Birch%E2%80%93Murnaghan_equation_of_state>`_
  to energy-versus-volume data.  This analysis is inherited from
  LAMMPS-GUI and is only meaningful for data with that shape.
- *Custom function* evaluates a user-supplied mathematical expression
  ``f(x)`` over the x range of the data and overlays it as a curve.  The
  expression uses the variable ``x`` for the abscissa and supports the
  usual arithmetic operators and functions (for example
  ``2*x^2 + 3*sin(x)``).
- *Custom fit* performs a nonlinear least-squares fit of a user-supplied
  expression to the data.  In addition to the expression, you provide the
  fit parameters and their initial guesses as ``name=value`` pairs (for
  example ``a=1, b=0.5``) and, optionally, a label for the fitted curve.
  The fit uses a Levenberg-Marquardt algorithm with analytic derivatives
  of the expression; on success the fitted curve is overlaid and the
  fitted parameters, the root-mean-square residual, and the number of
  iterations are reported.  This is convenient, for example, to fit an
  exponential approach to a steady-state value.

The expressions for *Custom function* and *Custom fit* are parsed and
evaluated with a bundled subset of the `Lepton expression parser
<https://simtk.org/projects/lepton>`_.

When any fit or custom-function overlay is active, the "Plot:" drop-down
treats the overlay as the "smoothed" series: selecting "Smoothed" or
"Both" shows the overlay curve, while selecting "Raw" hides it.  This
applies uniformly to all analysis types.

.. TODO screenshot: capture the Postprocess dialog with a custom
   function or custom fit entered as JPG/sparta-gui-post-function.png
   and an example fit overlaid on stats data of a SPARTA run as
   JPG/sparta-gui-custom-fit.png, then re-enable these figures.
..
.. .. figure:: JPG/sparta-gui-post-function.png
..    :align: center
..    :width: 50%
..
..    The *Postprocess* dialog with a *Custom function* expression entered.
..
.. .. figure:: JPG/sparta-gui-custom-fit.png
..    :align: center
..    :width: 55%
..
..    An example post-processing result: a custom fit overlaid on data
..    from a stats output column.

.. index:: plotting external data
.. index:: plot data file

Plot imported data
------------------

The same *Charts* window is also used to plot data from an external file
opened with *File* -> *Plot Data File...* (`Ctrl-Shift-P`, see
:ref:`the File menu <files>`).  The column-picker dialog shown before
the chart opens lets you select which column provides the x axis and
which columns to plot, and also allows renaming columns.  An "X-Axis:"
label field in the first toolbar row (to the right of "Title:" and "Y:")
lets you edit the x-axis label after the chart opens.  All the styling,
export, and post-processing features described above work the same way.
The same column-picker and standalone chart window are also launched
when SPARTA-GUI is invoked from the command line with the
``-c``/``--chart`` flag (see :ref:`command-line options
<command-line-options>`).

.. TODO screenshot: capture the column-picker dialog opened via Plot
   Data File... as JPG/sparta-gui-import-data.png, then re-enable this
   figure.
..
.. .. figure:: JPG/sparta-gui-import-data.png
..    :align: center
..    :width: 45%
..
..    The column-picker dialog shown when opening an external data file
..    with *Plot Data File...*.

The *Preferences* dialog has a *Charts* tab, where you can configure
multiple chart-related settings, like the default title, colors for the
graphs, default choice of the raw / smooth graph selection, whether the
grid for the major and minor ticks is drawn, and the default chart graph
size.

.. admonition:: Slowdown of Simulations from Charts Data Processing
   :class: warning

   Using frequent stats output during long simulations can result in a
   significant slowdown of that simulation since it is accumulating many
   data points for each of the stats columns in the chart window to
   be redrawn with every update.  The updates are consuming additional
   CPU time when smoothing is enabled.  It is thus recommended
   to use a large enough value as argument `N` for the `stats command
   <https://sparta.github.io/doc/stats.html>`_ and to select plotting only
   the "Raw" data in the *Charts* window during such simulations.  It is
   always possible to switch between the different display styles for
   charts during the simulation and after it has finished.

Variable Info
^^^^^^^^^^^^^

.. index:: variable info window
.. index:: variables
.. index:: input script variables

During a run, it may be of interest to monitor the value of input script
`variables <https://sparta.github.io/doc/variable.html>`_, for example
to monitor the progress of loops.  This can be done by enabling the
"Variables Window" in the *View* menu or by using the `Ctrl-Shift-W`
keyboard shortcut.  This shows the currently defined variables and
their values in a separate window.

.. TODO screenshot: capture the Variables window during a run of an
   input with variables as JPG/sparta-gui-variable-info.png, then
   re-enable this figure.
..
.. .. image:: JPG/sparta-gui-variable-info.png
..    :align: right
..    :scale: 50%

Like for the *Output* and *Charts* windows, its content is continuously
updated during a run.  It will show "(none)" if there are no variables
defined.  Note that it is also possible to *set* `index style variables
<https://sparta.github.io/doc/variable.html>`_, that would normally be
set via command-line flags, via the "Set Variables..." dialog from the
*Run* menu.  SPARTA-GUI automatically defines the variable "gui_run"
with the current value of the run counter.  That way it is possible to
automatically record a separate log for each run attempt by using the
command

.. code-block:: SPARTA

   log logfile-${gui_run}.txt

at the beginning of an input file. That would record logs to files
``logfile-1.txt``, ``logfile-2.txt``, and so on for successive runs.
