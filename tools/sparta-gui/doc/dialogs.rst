*******
Dialogs
*******

.. index:: dialogs

Find and Replace
----------------

.. index:: Find and Replace
.. index:: dialogs; Find and Replace
.. index:: text search

The *Find and Replace* dialog allows searching for and replacing
text in the *Editor* window.

.. TODO screenshot: capture the Find and Replace dialog as
   JPG/sparta-gui-find.png, then re-enable this figure.
..
.. .. image:: JPG/sparta-gui-find.png
..    :align: right
..    :scale: 33%

The dialog can be opened either from the *Edit* menu or with the
keyboard shortcut `Ctrl-F`. You can enter the text to search for.

.. admonition:: Through three checkboxes the search behavior can be adjusted:

   - If checked, "Match case" does a case-sensitive search; otherwise
     the search is case-insensitive.

   - If checked, "Wrap around" starts searching from the start of the
     document, if there is no match found from the current cursor position
     until the end of the document; otherwise the search will stop.

   - If checked, the "Whole word" setting only finds full word matches
     (white space and special characters are word boundaries).

Clicking on the *Next* button will search for the next occurrence of the
search text and select / highlight it. Clicking on the *Replace* button
will replace an already highlighted search text and find the next one.
If no text is selected, or the selected text does not match the
selection string, then the first click on the *Replace* button will
only search and highlight the next occurrence of the search string.
Clicking on the *Replace All* button will replace all occurrences from
the cursor position to the end of the file; if the *Wrap around* box is
checked, then it will replace **all** occurrences in the **entire**
document.  Clicking on the *Done* button will dismiss the dialog.

------

Preferences
-----------

.. index:: preferences
.. index:: dialogs; Preferences
.. index:: settings
.. index:: configuration

The *Preferences* dialog allows customization of the behavior and
look of SPARTA-GUI.  The settings are grouped and each group is
displayed within a tab.

.. TODO screenshot: capture the five Preferences tabs as
   JPG/sparta-gui-prefs-general.png, JPG/sparta-gui-prefs-accel.png,
   JPG/sparta-gui-prefs-image.png, JPG/sparta-gui-prefs-editor.png,
   and JPG/sparta-gui-prefs-charts.png, then re-enable the figures.
..
.. .. |guiprefs1| image:: JPG/sparta-gui-prefs-general.png
..    :width: 19%
..
.. .. |guiprefs2| image:: JPG/sparta-gui-prefs-accel.png
..    :width: 19%
..
.. .. |guiprefs3| image:: JPG/sparta-gui-prefs-image.png
..    :width: 19%
..
.. .. |guiprefs4| image:: JPG/sparta-gui-prefs-editor.png
..    :width: 19%
..
.. .. |guiprefs5| image:: JPG/sparta-gui-prefs-charts.png
..    :width: 19%
..
.. |guiprefs1|  |guiprefs2|  |guiprefs3|  |guiprefs4|  |guiprefs5|

General Settings
^^^^^^^^^^^^^^^^

.. index:: general settings
.. index:: preferences; general

.. admonition:: The following settings are available in this tab:

   - **Echo input to output buffer:** when checked, all input commands,
     including variable expansions, are echoed to the :ref:`Output window
     <logfile>`. This is equivalent to using ``-echo screen`` on the
     command-line.  There is no log *file* produced by default, since
     SPARTA-GUI uses ``-log none``.
   - **Show Output window by default:** when checked, the screen output of
     a SPARTA run will be collected in an Output window during the run.
   - **Show Charts window by default:** when checked, the stats
     output of a SPARTA run will be collected and displayed in a Charts
     window as line graphs.
   - **Show Slide Show window by default:** when checked, a Slide Show
     window will be shown with images from a dump image command, if
     present, in the SPARTA input.
   - **Replace Output window on new run:** when checked, an existing
     Output window will be replaced on a new SPARTA run; otherwise each
     run will create a new Output window.
   - **Replace Charts window on new run:** when checked, an existing
     Charts window will be replaced on a new SPARTA run; otherwise each
     run will create a new Charts window.
   - **Replace Image window on new render:** when checked, an existing
     Image window will be replaced when a new snapshot image is requested;
     otherwise each command will create a new Image window.
   - **Select Default Font:** Opens a font selection dialog where the type
     and size for the default font (used for everything but the editor and
     log) of the application can be set.
   - **Select Text Font:** Opens a font selection dialog where the type and
     size for the text editor and log font of the application can be set.
   - **Data update interval:** Allows the user to set, in milliseconds,
     the time interval between data updates during a SPARTA run.  The
     default is to update the data (for the Charts and Output windows)
     every 10 milliseconds.  This is good for many cases.  Set this to 100
     milliseconds or more if SPARTA-GUI consumes too many resources during
     a run.  For SPARTA runs that progress *very* fast, however, data may
     be missed; this can be corrected by lowering this interval.  However,
     this will make the GUI use more resources.  This setting may be
     changed to a value between 1 and 1000 milliseconds.
   - **Charts update interval:** Allows the user to set, in milliseconds,
     the time interval between redrawing the plots in the :ref:`Charts
     window <charts>`.  The default is to redraw the plots every 500
     milliseconds.  This is just for the drawing; data collection is
     managed with the previous setting.
   - **Path to SPARTA Shared Library File:** this option is only visible
     when SPARTA-GUI was compiled to load the SPARTA library at runtime
     instead of being linked to it directly (plugin mode, the default).
     Using the *Browse...* button or by changing the text, a different
     shared library file with a different compilation of SPARTA with
     different settings or from a different version can be loaded.
     After changing this setting, SPARTA-GUI needs to be re-launched.

Accelerators
^^^^^^^^^^^^

.. index:: accelerators
.. index:: preferences; accelerators
.. index:: KOKKOS package
.. index:: thread parallelization

This tab enables selection of an accelerator package and modification
of its settings for use when running SPARTA.  SPARTA supports
acceleration through the `KOKKOS package
<https://sparta.github.io/doc/Section_accelerate.html>`_; selecting it
here is equivalent to using the ``-k on -sf kk`` `command-line flags
<https://sparta.github.io/doc/Section_start.html>`_ of the SPARTA
executable.  The setting is only available when the loaded SPARTA
library was compiled with the KOKKOS package included.  The `Number of
threads` field allows setting the number of OpenMP threads to use when
the KOKKOS library was compiled with OpenMP support.  Selecting "None"
runs SPARTA without any accelerator package.

.. _image_preferences:

Snapshot Image
^^^^^^^^^^^^^^

.. index:: snapshot image settings
.. index:: preferences; snapshot image
.. index:: image rendering

This tab allows setting defaults for the snapshot images displayed in
the :ref:`Image Viewer window <snapshot_viewer>`, such as its
dimensions, the zoom factor, and view angles.  The **Antialias** switch
enables `full scene anti-aliasing (FSAA)
<https://en.wikipedia.org/wiki/Spatial_anti-aliasing>`_ which renders
images with double the number of pixels for width and height and then
smoothly scales the image back to the requested size.  This produces
higher quality images with smoother edges at the expense of requiring
more CPU time to render an initial image four times the size.  The **HQ
Image mode** option turns on "Screen Space Ambient Occlusion (SSAO)"
mode when rendering images.  This is also more time consuming, but
produces a more 'spatial' representation of the system with shading by
depth.  The **Shiny Image mode** option will render objects with a
shiny surface when enabled.  Otherwise, the surfaces will be matte.
The **Show Box** option selects whether the simulation box is drawn as
a colored set of sticks.  Furthermore, the diameter of the sticks and
their color can be set.  Similarly, the **Show Axes** option selects
whether a representation of the three system axes will be drawn or not.
In addition, the axes length and diameter can be set as fractions of
the box size.  Finally, there are a couple of text fields to select the
two **Background Colors**.  If the two colors differ, there will be a
vertical background gradient starting with the "Background" color at
the bottom and ending with the "Background2" color at the top.

These settings correspond to the available settings for the SPARTA
`dump image <https://sparta.github.io/doc/dump_image.html>`_ and
`dump_modify <https://sparta.github.io/doc/dump_modify.html>`_
commands.

Editor Settings
^^^^^^^^^^^^^^^

.. index:: editor settings
.. index:: preferences; editor
.. index:: code formatting preferences

This tab allows adjusting settings of the :ref:`editor window <editor>`.
Specifically, the amount of padding to be added to SPARTA commands,
IDs (e.g., for fixes or computes), and names (e.g., for mixtures or
groups).  The value set is the minimum width for the text element and it
can be chosen in the range between 1 and 32.

The three settings which follow enable or disable the automatic
reformatting when hitting the 'Enter' key, the automatic display of
the completion pop-up window, and whether auto-save mode is enabled.
In auto-save mode, the editor buffer is saved before a run or before
exiting SPARTA-GUI.

Charts Settings
^^^^^^^^^^^^^^^

.. index:: charts settings
.. index:: preferences; charts
.. index:: plotting preferences

This tab allows adjusting settings of the :ref:`Charts window <charts>`.
Specifically, one can set the default chart title (if the title contains
'%f' it will be replaced with the name of the current input file), one
can select whether by default the raw data, the smoothed data, or both
will be plotted, one can set the colors for the two lines, the default
smoothing parameters, the default size of the chart graph in pixels, and
whether you want to display major and minor grid lines.
