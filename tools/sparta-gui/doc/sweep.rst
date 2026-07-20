.. _parametric_sweep:

*****************
Parametric Sweeps
*****************

.. index:: parametric sweep
.. index:: DOE
.. index:: design of experiments
.. index:: index variable

A *parametric sweep* runs the current input deck repeatedly while varying one
or more SPARTA `index variables
<https://sparta.github.io/doc/variable.html>`_, and tabulates a chosen
thermodynamic quantity for each parameter combination — the design-of-experiments
(DOE) studies that are routine in production DSMC work.  Open it from *Run →
Parametric Sweep...* (or the *Parametric Sweep* toggle in the *View* menu).

.. TODO screenshot: capture the Parameter Sweep panel with results as
   JPG/sparta-gui-sweep.png

Each combination runs the editor buffer **in-process, one at a time**, reusing
the same run machinery (and the same variable-override mechanism) as a normal
*Run from Editor Buffer*.  Because SPARTA runs sequentially,
the sweep is a queue: a run starts, and when it finishes the next parameter
combination is injected and launched automatically.

Defining the sweep
==================

- **Variables** — each row of the table is one variable to vary.  Click
  *Detect from Deck* to populate the variable names from the ``variable ...
  index`` definitions found in the buffer (only index-style variables can be
  overridden per run, exactly as with the *Set Variables* dialog and the
  ``-var`` command-line flag).  For each variable choose a **Type** and give
  its **Specification**:

  - **List** — explicit values, e.g. ``0.05, 0.1, 0.2``.
  - **Range** — ``start:stop:step`` (inclusive of the endpoint), e.g.
    ``0:1:0.1``.
  - **Linspace** — ``start:stop:count``, e.g. ``0:1:5`` for five evenly-spaced
    points.

- **Combination** — *Cartesian product* runs every combination of the
  variables' values (the last variable varies fastest); *Zip* pairs the values
  index-by-index and requires each variable to have the same number of values.

- **Tabulate** — a comma-separated list of thermo keywords to record (for
  example ``Np`` or a compute output such as ``c_temp``), and a **Reduce**
  choice describing how each run's series is turned into one number: the
  *final value*, or the *minimum*, *maximum*, or *mean* over the run.

Running and results
===================

Press *Run Sweep* to start; it becomes *Stop Sweep* while the sweep is
running, and a progress bar shows the current run out of the total.  The
results table fills in live, one row per combination, with the swept-variable
values followed by the tabulated quantities; a run that fails is kept in the
table and marked.  *Stop Sweep* halts after the current run completes.

The results can be **exported to CSV** or **charted** directly (the first
swept variable on the x-axis and the tabulated quantities as series, using the
same :ref:`Charts <charts>` window as the live thermo plots).

.. note::

   A swept variable must be an *index* variable defined in the deck; a name
   that is not defined that way cannot be overridden per run.  The sweep runs
   locally and sequentially; dispatching each combination to a cluster via
   :ref:`remote execution <remote_execution>` is a planned extension.
