*************
Visualization
*************

.. _snapshot_viewer:

Snapshot Image Viewer
^^^^^^^^^^^^^^^^^^^^^

.. index:: snapshot viewer
.. index:: image viewer
.. index:: visualization
.. index:: dump image

By selecting the *Create Image* entry in the *Run* menu, or by hitting
the `Ctrl-I` (`Command-I` on macOS) keyboard shortcut, or by clicking on
the "palette" button in the status bar of the :doc:`Editor window
<editor>`, SPARTA-GUI sends a `dump image
<https://sparta.github.io/doc/dump_image.html>`_ command to SPARTA and
reads back the resulting snapshot image with the current state of the
system into an image viewer.  This functionality is *not* available
*during* an ongoing run.  In case SPARTA is not yet initialized,
SPARTA-GUI tries to identify the line with the first `run
<https://sparta.github.io/doc/run.html>`_ command and executes all
commands in the editor up to that line, and then executes a "run 0"
command.  This initializes the system so an image of the initial state
of the system can be rendered.  If there was an error in that process,
a dialog with the error message will appear.

The image is rendered by SPARTA's built-in ray-tracing renderer, the
same code that produces the images of the `dump image
<https://sparta.github.io/doc/dump_image.html>`_ command.  All four
kinds of entities that SPARTA can draw are supported:

- **particles**, selected through a `mixture
  <https://sparta.github.io/doc/mixture.html>`_ and colored and sized by
  particle attributes,
- **grid cells**, rendered as (semi-transparent) volumes colored by
  owning processor or by per-grid compute or fix data,
- **grid cut planes** through the grid perpendicular to the x, y, or z
  axis, colored like grid cells, and
- **surface elements**, colored by a constant color, by owning
  processor, or by per-surf compute or fix data.

Automatic settings
------------------

By default, SPARTA-GUI renders the particles of the mixture "all",
colored by their species type using the default color sequence of the
`dump image <https://sparta.github.io/doc/dump_image.html>`_ command,
with a particle diameter inferred from the cell size of the simulation
grid so that particles are visible but do not dominate the image.  When
the simulation defines surface elements, they are shown as well.  All
of these choices can be changed in the settings dialogs described
below.

-----------

Customizations
--------------

.. |palette| image:: JPG/emblem-photos.png
                     :width: 14px
.. |inactive| image:: JPG/inactive-photos.png
                     :width: 14px

The Image Viewer controls described below support significant
customization of the default visualization through the toolbar buttons,
editable text fields, and additional dialogs.  This covers the full set
of options of the `dump image
<https://sparta.github.io/doc/dump_image.html>`_ and `dump_modify
<https://sparta.github.io/doc/dump_modify.html>`_ commands.  After each
change is applied, SPARTA-GUI will have SPARTA re-create the displayed
image with the updated settings.  The resulting image can be saved or
copied to the clipboard and pasted into a compatible application.

.. admonition:: Delays in updating the image
   :class: note

   Some visualization settings, especially enabling SSAO and FSAA (see
   below) or massively enlarging the image size, can significantly
   increase the time required for SPARTA to render the image.  While in
   most cases the image will be updated in a fraction of a second,
   complex visualizations may take multiple seconds.  While SPARTA is
   rendering an updated image, the small color palette icon in the
   menu bar is colored |palette| and will be grayed out |inactive| when
   rendering is complete.

For further customization or for making the visualization available
when running the simulation with SPARTA directly (e.g. when running in
parallel on a cluster after enlarging the system), you can copy the
current `dump image <https://sparta.github.io/doc/dump_image.html>`_
and `dump_modify <https://sparta.github.io/doc/dump_modify.html>`_
commands to the clipboard with the **Copy dump image command** action,
so they can be pasted into a SPARTA input script in either the included
:doc:`text editor window <editor>` or some other text editor and
adjusted according to the documentation.

The resulting images will be shown automatically in the :ref:`slide show
viewer <slideshow>` when running the simulation with the thus modified
input from SPARTA-GUI.  This is an effective strategy for interactively
composing publication quality visualizations of a DSMC simulation.

.. TODO screenshot: capture the Image Viewer window showing particles,
   grid cells, and surface elements of a SPARTA example (e.g.
   examples/circle) as JPG/sparta-gui-image.png, then re-enable this
   figure.
..
.. .. image:: JPG/sparta-gui-image.png
..    :align: center
..    :width: 70%

-----------

Available controls
------------------

.. index:: image viewer controls

The Image Viewer window consists of three main areas: a menu/toolbar
strip at the top, the rendered image in the center, and a settings panel
on the right side.  Following the general theme of SPARTA-GUI of
extensive keyboard shortcut support, you can select most text fields by
using the `Alt` key and the underlined letter.  For example the *File*
menu is opened with `Alt-F` and its entries can be also selected the
same way or using the cursor keys and `Enter`.  Keyboard shortcuts
starting with `Ctrl` usually work globally inside the image window, that
is even when the corresponding menu item is not visible.

The **menu bar row** has:

- The **File** menu with the following entries:
   - **Save As...** (`Ctrl-S`): Save the rendered image to a file.  The
     file format is inferred from the file name extension.  When the
     `ImageMagick software <https://imagemagick.org/>`_ is installed,
     additional file formats beyond those natively supported by the Qt
     library become available.
   - **Copy Image** (`Ctrl-C`): Copy the rendered image to the clipboard
     for pasting it into another application.  This requires support
     from the receiving application, which is the case for many common
     applications like document editors and web browsers.
   - **Copy dump image command** (`Ctrl-D`): Copy the current `dump image
     <https://sparta.github.io/doc/dump_image.html>`_ and `dump_modify
     <https://sparta.github.io/doc/dump_modify.html>`_ commands to the
     clipboard so they can be pasted into a SPARTA input script.  This
     allows the current visualization settings to be reproduced during a
     simulation run, including in the :ref:`slide show viewer <slideshow>`.
   - **Close** (`Ctrl-W`): Close the Image Viewer window.
   - **Quit** (`Ctrl-Q`): Quit the entire application.
- The **busy indicator**, a small palette icon that is colored |palette|
  while SPARTA is rendering a new image and grayed out |inactive| when
  rendering is complete.
- The **Particle size** text field, where the particle diameter (in
  distance units, corresponding to the *pdiam* keyword) can be adjusted.
- The **Width** spin box where the image width can be set.  It can be
  accessed using the `Alt-W` keyboard shortcut.
- The **Height** spin box, where the image height can be set.  It can
  be accessed using the `Alt-H` keyboard shortcut.

The **toolbar buttons** row below the menu bar provides quick access to
several rendering options and view manipulations.  From left to right
there are:

- **SSAO** (toggle): Enable or disable `Screen Space Ambient Occlusion
  <https://en.wikipedia.org/wiki/Screen_space_ambient_occlusion>`_
  rendering for a more spatial, depth-shaded appearance, at the
  expense of more CPU time.  This corresponds to the *ssao* keyword.
- **Anti-aliasing** (toggle): Render the image at double resolution and
  scale down for smoother edges.  `Full Scene Anti-Aliasing (FSAA)
  <https://en.wikipedia.org/wiki/Spatial_anti-aliasing#Super_sampling_/_full-scene_anti-aliasing>`_
  produces higher quality images at the expense of more CPU time.  This
  corresponds to the *fsaa* keyword.
- **Shininess** (toggle): Switch between shiny and matte surface
  rendering (the *shiny* keyword).
- **Particles** (toggle): Show or hide the particles (the *particle*
  keyword).
- **Grid** (toggle): Show or hide the grid cell volume rendering (the
  *grid* keyword).
- **Surfaces** (toggle): Show or hide the surface elements (the *surf*
  keyword).
- **Box** (toggle): Show or hide the simulation box drawn as colored
  cylinders (the *box* keyword).
- **Axes** (toggle): Show or hide the coordinate axes arrows (the
  *axes* keyword).
- **Zoom in** / **Zoom out**: Adjust the zoom level in 10
  percent increments between 0.1x and 10.0x (the *zoom* keyword).
- **Rotate left** / **Rotate right**: Rotate the view horizontally by
  10 degrees per click (the *phi* angle of the *view* keyword).
- **Rotate up** / **Rotate down**: Rotate the view vertically by
  10 degrees per click (the *theta* angle of the *view* keyword).
  For 2d simulations the view is fixed to look down the z axis and the
  rotation buttons are disabled.
- **Reset**: Reset the view to the default orientation and zoom level.
- **Fit window**: Resize the window so the image is shown at its full
  size, without scroll bars or unused space.  This undoes a manual
  resize of the window; the window is never grown beyond a fraction of
  the screen, so scroll bars remain for very large images.

The default image size, some default image quality settings, and some
colors can be changed in the :doc:`Preferences <dialogs>` dialog
window.

The **settings panel** on the right side of the window provides
additional controls (most are explained in detail below):

- **Mixture**: A drop-down list to select which `mixture
  <https://sparta.github.io/doc/mixture.html>`_ of particles to display
  (default is "all").  Only particles whose species belong to the
  selected mixture are rendered.  The list is retrieved from the
  current SPARTA instance.
- **Global**: Opens the :ref:`Global image settings <global_settings>`
  dialog for fine-grained control of axes, box, sub-box, background,
  quality, view, center, camera, and lighting settings.
- **Particles**: Opens the :ref:`Particle settings <particle_settings>`
  dialog for detailed particle coloring and sizing options.
- **Grid**: Opens the :ref:`Grid settings <grid_settings>` dialog to
  configure the volume rendering of the grid cells and the grid cell
  outlines.
- **Grid planes**: Opens the :ref:`Grid plane settings <plane_settings>`
  dialog to configure cut planes through the grid perpendicular to the
  x, y, and z axes.
- **Surfaces**: Opens the :ref:`Surface settings <surf_settings>` dialog
  to configure the display of the surface elements.
- **Help**: Opens this documentation page for the visualization
  features in SPARTA-GUI in a web browser.

The image is re-rendered after each change to the buttons, text fields
or settings dialogs, and when there are many particles or grid cells to
render and high quality images with anti-aliasing are requested,
re-rendering may take several seconds.  There is no GPU acceleration.

---------------

.. _global_settings:

Global image settings
---------------------

.. index:: image settings
.. index:: global image settings
.. index:: dump_modify

While some persistent default settings for the image output can be
configured in the "Snapshot Image" tab of the :ref:`Preferences dialog
<image_preferences>`, more fine-grained configuration is possible by
opening the "Global image settings" dialog.  However, settings not
stored by the preferences are reset when the image viewer window is
closed.  This dialog is opened by pressing the "Global" button in the
settings panel or by using the `Alt-L` keyboard shortcut.  The settings
in this dialog correspond to options of the SPARTA `dump image
<https://sparta.github.io/doc/dump_image.html>`_ and `dump_modify
<https://sparta.github.io/doc/dump_modify.html>`_ commands.

.. TODO screenshot: capture the Global image settings dialog as
   JPG/sparta-gui-image-global.png, then re-enable this figure.
..
.. .. image:: JPG/sparta-gui-image-global.png
..    :align: center
..    :width: 62%

The dialog is organized into the following sections:

**Axes**
   Controls the display of coordinate axes arrows in the image
   (the *axes* keyword).

   - **Axes** (checkbox): Enable or disable rendering of coordinate axes.
   - **Length**: The length of the axes lines as a fraction of the box
     size.
   - **Diameter**: The diameter of the axes lines as a fraction of the
     box size.

**Box**
   Controls the display of the simulation box (the *box* keyword and
   the *boxcolor* dump_modify keyword).

   - **Box** (checkbox): Enable or disable rendering of the simulation
     box.
   - **Color**: The color used to draw the box edges.  Accepts the
     `named colors <https://sparta.github.io/doc/dump_modify.html>`_
     known to the dump image command.
   - **Diameter**: The diameter of the box edge sticks as fraction
     of the box size.

**Sub-box**
   Controls the display of the per-processor sub-domain boxes (the
   *subbox* keyword and the *subboxcolor* dump_modify keyword).  Since
   SPARTA-GUI runs SPARTA on a single processor, the sub-box coincides
   with the simulation box; the setting is mainly useful when composing
   a dump image command for a parallel run with the **Copy dump image
   command** action.

   - **Subbox** (checkbox): Enable or disable rendering of the sub-domain
     boxes.
   - **Color**: The color used to draw the sub-box edges.
   - **Diameter**: The diameter of the sub-box edge sticks as
     fraction of the box size.

**Background**
   Sets the background color(s) of the rendered image (the *backcolor*
   and *backcolor2* dump_modify keywords).

   - **Bottom color**: The background color at the bottom of the image.
   - **Top color**: The background color at the top of the image.  If
     the two colors differ, a vertical gradient is applied from bottom
     to top.

**Quality**
   Controls rendering quality options.

   - **FSAA** (checkbox): Enable or disable full-scene anti-aliasing
     (the *fsaa* keyword).
   - **SSAO** (checkbox): Enable or disable Screen Space Ambient
     Occlusion for depth-shaded rendering (the *ssao* keyword).
   - **SSAO strength**: The strength of the SSAO effect (range: 0.0 --
     1.0).
   - **Shiny**: The shininess factor for surface rendering (range: 0.0
     -- 1.0, where 0.0 is matte and 1.0 is fully shiny; the *shiny*
     keyword).

**View**
   Adjusts the camera position (the *view*, *center*, *up*, and *zoom*
   keywords).

   - **Theta**: The viewing angle in degrees away from the positive
     z-axis.  Disabled for 2d systems, where SPARTA always looks down
     the z-axis.
   - **Phi**: The azimuthal viewing angle in degrees around the
     z-axis.  Disabled for 2d systems.
   - **Center X / Y / Z**: Fractional coordinates (0.5 = center of the
     box) specifying the center of the view relative to the simulation
     box.
   - **Up X / Y / Z**: The components of the camera's up vector.  The
     vector does not need to be normalized, but it must not be all
     zeros, or the values are ignored.
   - **Zoom**: The zoom factor of the view (range: 0.1 -- 10.0, where
     values larger than 1.0 zoom in).  This is the same setting that
     the zoom in/out buttons of the toolbar change in steps of
     10 percent.

   .. note::

      The *persp* keyword of the dump image command (depth perspective)
      is documented but not yet supported by SPARTA, therefore there is
      no perspective setting in SPARTA-GUI; all images are rendered
      with orthographic projection.

**Lighting**
   Adjusts the intensities of the four light sources used in the
   rendering (the *lights* dump_modify keyword).  Each value is a
   floating point number (range: 0.0 -- 1.0) representing the intensity
   of the respective light source.

   - **Ambient**: The intensity of the uniform, non-directional base
     lighting that illuminates all parts of the scene equally.
   - **Key**: The intensity of the primary, directional light source
     that provides the main illumination and highlights.
   - **Fill**: The intensity of the secondary directional light source
     that fills in the shadows created by the key light.
   - **Back**: The intensity of the tertiary directional light source
     that illuminates the back of the objects, helping to separate
     them from the background.

Press **Apply** to apply the current settings and re-render the image,
or **Cancel** to discard changes.  The **Help** button opens the SPARTA
`dump image <https://sparta.github.io/doc/dump_image.html>`_
documentation.

---------------

.. _particle_settings:

Particle settings
-----------------

.. index:: particle settings
.. index:: particle coloring
.. index:: mixture

This dialog offers detailed customizations for the particle
visualization that are not directly accessible from the main Image
Viewer toolbar.  It is opened by pressing the "Particles" button in the
settings panel or by using the `Alt-P` keyboard shortcut.

.. TODO screenshot: capture the Particle settings dialog as
   JPG/sparta-gui-image-particles.png, then re-enable this figure.
..
.. .. image:: JPG/sparta-gui-image-particles.png
..    :align: center
..    :width: 62%

The dialog contains the following controls:

- **Particles** (checkbox): Enable or disable rendering of particles
  (the *particle* keyword of the dump image command).
- **Color by**: Select the per-particle attribute used for coloring.
  The basic choices are *type* (the species type, with one fixed color
  per type) and *proc* (the owning processor, always a single color in
  SPARTA-GUI runs).  In addition, all per-particle attributes known to
  the `dump particle <https://sparta.github.io/doc/dump.html>`_ command
  can be selected, such as coordinates and velocity components, as well
  as particle-style `variables <https://sparta.github.io/doc/variable.html>`_
  (``v_name``), per-particle `compute
  <https://sparta.github.io/doc/compute.html>`_ results (``c_ID`` or
  ``c_ID[col]``), and per-particle `fix
  <https://sparta.github.io/doc/fix.html>`_ results (``f_ID`` or
  ``f_ID[col]``).  The contents of the list depend on what is available
  in the current simulation.  When a numeric attribute is selected, the
  colors are determined by the particle :ref:`color map <colormaps>`.
- **Per-type colors**: When coloring by *type*, the color assigned to
  each particle type can be customized (the *pcolor* dump_modify
  keyword).  The per-type color rows can be scrolled when the current
  system has many species.  A **Reset** button restores the compiled-in
  default color sequence.
- **Size by**: Select how the particle diameter is determined: a
  constant diameter entered in the text field (the *pdiam* keyword of
  the dump image command, in distance units), the particle *type* with
  an editable per-type diameter table (the *pdiam* dump_modify
  keyword), or a numeric per-particle attribute (the *diameter* choice
  of the dump image command).
- **Map** / **Reverse** / **Min** / **Max**: Configure the particle
  :ref:`color map <colormaps>` used when coloring by a numeric
  attribute.  Use *auto* for **Min** / **Max** to have SPARTA determine
  the range automatically from the visible particles for every image.

Press **Apply** to apply the settings and re-render the image, or
**Cancel** to discard changes.  The **Help** button opens the SPARTA
`dump image <https://sparta.github.io/doc/dump_image.html>`_
documentation.

--------------

.. _grid_settings:

Grid settings
-------------

.. index:: grid visualization
.. index:: grid settings
.. index:: grid groups

This dialog allows enabling and configuring the volume rendering of the
`simulation grid <https://sparta.github.io/doc/create_grid.html>`_.  It
is opened by pressing the "Grid" button in the settings panel or by
using the `Alt-G` keyboard shortcut.

.. TODO screenshot: capture the Grid settings dialog together with an
   image showing grid cells colored by a per-grid compute as
   JPG/sparta-gui-image-grid.png, then re-enable this figure.
..
.. .. image:: JPG/sparta-gui-image-grid.png
..    :align: center
..    :width: 62%

The following settings are available:

- **Grid** (checkbox): Enable or disable rendering of the grid cells as
  semi-transparent volumes (the *grid* keyword of the dump image
  command).
- **Color by**: Select what determines the color of each grid cell:
  *proc* (the owning processor) or any per-grid value produced by a
  `compute <https://sparta.github.io/doc/compute.html>`_ or `fix
  <https://sparta.github.io/doc/fix.html>`_ (``c_ID``, ``c_ID[col]``,
  ``f_ID``, or ``f_ID[col]``), for example a `compute grid
  <https://sparta.github.io/doc/compute_grid.html>`_ that tallies
  density or temperature per cell.  The contents of the list depend on
  what is defined in the current simulation.
- **Grid group**: Restrict the rendering to a `grid group
  <https://sparta.github.io/doc/group.html>`_ (the *gridgroup*
  dump_modify keyword; default is "all").
- **Cell outlines** (checkbox): Draw the outline of each grid cell (the
  *gline* keyword).  The outline diameter as a fraction of the box size
  and the outline color (the *glinecolor* dump_modify keyword) can be
  set in the adjacent fields.
- **Map** / **Reverse** / **Min** / **Max**: Configure the grid
  :ref:`color map <colormaps>` used to translate the selected per-grid
  values into colors.  Use *auto* for **Min** / **Max** to have SPARTA
  determine the range automatically for every image.

Press **Apply** to apply the settings and re-render the image, or
**Cancel** to discard changes.  The **Help** button opens the SPARTA
`dump image <https://sparta.github.io/doc/dump_image.html>`_
documentation.

--------------

.. _plane_settings:

Grid plane settings
-------------------

.. index:: grid cut planes
.. index:: gridx
.. index:: gridy
.. index:: gridz

Rather than rendering the entire grid as volumes, it is often clearer
to show one or more *cut planes* through the grid.  This dialog
configures the *gridx*, *gridy*, and *gridz* keywords of the `dump
image <https://sparta.github.io/doc/dump_image.html>`_ command.  It is
opened by pressing the "Grid planes" button in the settings panel.

.. TODO screenshot: capture the Grid plane settings dialog together
   with an image showing gridx/gridy cut planes as
   JPG/sparta-gui-image-planes.png, then re-enable this figure.
..
.. .. image:: JPG/sparta-gui-image-planes.png
..    :align: center
..    :width: 62%

For each of the three axis directions there is one row of controls:

- **Show** (checkbox): Enable or disable the cut plane perpendicular to
  this axis.
- **Position**: The coordinate along the axis at which the plane cuts
  through the grid (in simulation units, within the box bounds).
- **Color by**: Select what determines the color of the grid cells in
  the plane: *proc* or a per-grid `compute
  <https://sparta.github.io/doc/compute.html>`_ or `fix
  <https://sparta.github.io/doc/fix.html>`_ value, exactly like for the
  :ref:`grid volume rendering <grid_settings>`.
- **Map** / **Reverse** / **Min** / **Max**: Each of the three planes
  has its *own* :ref:`color map <colormaps>` (the *gridx*, *gridy*, and
  *gridz* modes of the *cmap* dump_modify keyword), so, for example, a
  density plane and a temperature plane can use different maps and
  ranges in the same image.

Cut planes are only available for 3d simulations; for 2d simulations
the grid itself is a plane and can be shown with the :ref:`grid volume
rendering <grid_settings>`.

Press **Apply** to apply the settings and re-render the image, or
**Cancel** to discard changes.

--------------

.. _surf_settings:

Surface settings
----------------

.. index:: surface visualization
.. index:: surface settings
.. index:: surf groups

This dialog allows enabling and configuring the visualization of the
`surface elements <https://sparta.github.io/doc/read_surf.html>`_
defined in the simulation.  It is opened by pressing the "Surfaces"
button in the settings panel or by using the `Alt-S` keyboard shortcut.
The controls are disabled when the current simulation defines no
surfaces.

.. TODO screenshot: capture the Surface settings dialog together with
   an image showing surface elements colored by a per-surf compute as
   JPG/sparta-gui-image-surf.png, then re-enable this figure.
..
.. .. image:: JPG/sparta-gui-image-surf.png
..    :align: center
..    :width: 62%

The following settings are available:

- **Surfaces** (checkbox): Enable or disable rendering of the surface
  elements (the *surf* keyword of the dump image command).
- **Color by**: Select what determines the color of each surface
  element: *one* (a single constant color, set with the *scolor*
  dump_modify keyword), *proc* (the owning processor), or any per-surf
  value produced by a `compute
  <https://sparta.github.io/doc/compute.html>`_ or `fix
  <https://sparta.github.io/doc/fix.html>`_ (``c_ID``, ``c_ID[col]``,
  ``f_ID``, or ``f_ID[col]``), for example a `compute surf
  <https://sparta.github.io/doc/compute_surf.html>`_ that tallies
  fluxes on the surface elements.
- **Diameter**: The diameter used to render the line segments of 2d
  surfaces, as a fraction of the shortest box length.
- **Surf group**: Restrict the rendering to a `surf group
  <https://sparta.github.io/doc/group.html>`_ (the *surfgroup*
  dump_modify keyword; default is "all").
- **Element outlines** (checkbox): Draw the outline of each surface
  element (the *sline* keyword).  The outline diameter and the outline
  color (the *slinecolor* dump_modify keyword) can be set in the
  adjacent fields.
- **Map** / **Reverse** / **Min** / **Max**: Configure the surf
  :ref:`color map <colormaps>` used to translate the selected per-surf
  values into colors.

Press **Apply** to apply the settings and re-render the image, or
**Cancel** to discard changes.

------------

.. _colormaps:

Color maps
----------

.. index:: color map
.. index:: reversible color map

Whenever particles, grid cells, grid cut planes, or surface elements
are colored by a numeric value, that value is translated into a color
through a color map (the *cmap* keyword of the `dump_modify
<https://sparta.github.io/doc/dump_modify.html>`_ command).  SPARTA-GUI
maintains **six independent color maps**, one for each of the *cmap*
modes of SPARTA: *particle*, *grid*, *surf*, *gridx*, *gridy*, and
*gridz*.  Each map is configured in the corresponding settings dialog
described above through four controls:

- **Map**: Select the color map.  Currently available continuous color
  maps are: *RWB* (red-white-blue), *PWT* (purple-white-teal), *BWG*
  (blue-white-green), *BGR* (blue-green-red), *Grayscale*
  (black-white), *Viridis*, *Plasma*, *Inferno*, *Magma*, *Cividis*,
  and *Turbo* (from matplotlib), *Teal*, and *Rainbow*.  *Sequential*,
  *Landscape*, and *Basic* are maps with discrete colors.
- **Reverse** (checkbox): Mirror the selected color map so its low and
  high ends are swapped (for example, *RWB* becomes blue-white-red).
- **Min** / **Max**: Set the range of the color map.  Use *auto* to
  have SPARTA determine the range automatically from the data of each
  rendered image or specify an explicit numeric value to pin the range
  (recommended when comparing images, e.g. in the slide show).

The color maps offered by SPARTA-GUI are shown below; the continuous
maps are interpolated between color stops, while *Sequential*,
*Landscape*, and *Basic* use discrete colors.

.. _colormap_preview:

.. figure:: JPG/sparta-gui-colormaps.png
   :align: center
   :width: 60%

   The dump-image color maps offered for coloring particles, grid
   cells, grid cut planes, and surface elements by value.

These color maps are defined in a single table in the C++ source, which
makes them simple to add or modify; see :ref:`add_colormap` in the
Programmer's Guide for step-by-step instructions.  Fully customized
maps beyond these presets can be composed manually with the `dump_modify
cmap <https://sparta.github.io/doc/dump_modify.html>`_ command after
exporting the current settings with **Copy dump image command**.

--------------

.. _slideshow:

Image Slide Show
^^^^^^^^^^^^^^^^

.. index:: slideshow
.. index:: animation
.. index:: image sequence
.. index:: movie export
.. index:: image export

When running a SPARTA input containing a `dump image
<https://sparta.github.io/doc/dump_image.html>`_ command with
SPARTA-GUI, the "Slide Show" window opens to load and display the
images created by SPARTA as they are written.  This is a convenient way
to visually monitor the progress of the simulation.  It also can be
used as an effective way to refine visualizations created with the
:ref:`Snapshot Image Viewer <snapshot_viewer>`.

.. warning::

   When two or more ``dump image`` commands are active at the same time,
   the slide show picks up the images from all of them and displays them
   interleaved in the order they are written.  This is usually not
   intended, but cannot be detected by SPARTA-GUI before the run has
   started and the images have already been mixed.  To avoid it, make
   sure that only one ``dump image`` command is active at any time
   during a run, for example by removing a no longer needed dump with an
   `undump command <https://sparta.github.io/doc/undump.html>`_.

.. admonition:: dump movie
   :class: note

   As an alternative to exporting the slide show images to a movie
   (described below), SPARTA itself supports writing a movie directly
   with the `dump movie <https://sparta.github.io/doc/dump_image.html>`_
   command when the SPARTA library was compiled with ``-D
   SPARTA_FFMPEG=yes``.  In that case the images are piped directly
   into FFmpeg, and the frame rate and bit rate can be set with the
   *framerate* and *bitrate* keywords of the `dump_modify
   <https://sparta.github.io/doc/dump_modify.html>`_ command.

The same window can also display existing image files that were not
created by the current session: select one or more files with *File* ->
*View Image or Movie File(s)...* (see :ref:`the File menu <files>`) to
review images produced by an external (for example large parallel)
simulation, or to revisit images from an earlier run without rerunning
it.  Image formats that Qt cannot read natively are converted on demand
with `ImageMagick <https://imagemagick.org/>`_ if it is available.  Each
such file is converted only once and the converted copy is reused while
the window is open, so displaying it repeatedly neither repeats the
conversion nor repeats any complaint its format may provoke from Qt.  A
file that can be read by neither is reported once on the console and
then skipped.  When the slide show is opened this way, the controls
that act on a running simulation (such as stopping the run or sending
images to the trash) are hidden.

Movie files can be selected in the same dialog; their frames are then
extracted into individual images as described in :ref:`Importing movie
files <movie_import>` below.

From the slide show window the following global keyboard shortcuts are
supported: `Ctrl-W`: close window, `Ctrl-Q`: quit application, `Ctrl-/`:
stop running simulation.  Other keyboard shortcuts are connected to some
of the controls and listed in their documentation below.

.. TODO screenshot: capture the Slide Show window with images of a
   SPARTA example run as JPG/sparta-gui-slideshow.png, then re-enable
   this figure.
..
.. .. image:: JPG/sparta-gui-slideshow.png
..    :align: center

.. _movie_import:

Importing movie files
---------------------

.. index:: movie import

Movie files (``.mp4``, ``.mkv``, ``.webm``, ``.avi``, ``.mov``, and so on,
as well as animated GIF files) can be opened with *File* -> *View Image or
Movie File(s)...* just like image files.  Since the slide show viewer
displays individual images, the frames of a movie must first be
decompressed into a sequence of image files.  This requires the `FFmpeg
<https://ffmpeg.org/>`_ programs ``ffmpeg`` and ``ffprobe``; it is the
inverse of the movie export described below.

When a movie file is selected, a dialog reports its properties and asks
for confirmation before any frames are extracted:

- **First frame** and **Last frame** select the range of the movie to
  extract.
- **Frame interval** thins out that range: an interval of 1 extracts every
  frame, 2 every other frame, and so on.  This is useful to skim a long
  movie without decompressing all of it.
- **Estimated size** is how much temporary disk space the extracted
  images are expected to need.  It is obtained by decoding a single frame
  in the middle of the movie and multiplying its size by the number of
  selected frames, so it is an approximation.  A highlighted warning
  appears when the estimate exceeds one gigabyte, when it would use up
  most of the free space on the volume holding the temporary folder, or
  when more than 1000 images would be extracted.

Because the frames are stored as individual images and not as a
compressed video stream, they usually take up substantially more space
than the movie file itself.  The extracted frames are written to a
temporary folder and are deleted again when the slide show window is
closed.  Below the navigation slider each extracted image is labeled with
the name of the movie and its frame number in it.

Slide show controls
-------------------

.. index:: slideshow controls

There are controls and displays above and below the image.  If you are
uncertain about the function of a specific button, you can place the
cursor on top of it and a descriptive tooltip will appear.

The **toolbar** at the top of the Slide Show window provides the
following controls, organized from left to right:

- **Export to movie** (`Ctrl-E`): Export the active range of images (see
  the **Start** and **Stop** controls below) to a movie file or `animated
  GIF file <https://en.wikipedia.org/wiki/GIF#Animated_GIF>`_.  This requires
  that either the `FFmpeg program <https://ffmpeg.org/>`_ or the
  `ImageMagick software <https://imagemagick.org/>`_ are installed.
  Supported output formats include MP4, MKV, AVI, MPG, MPEG, WEBM, and
  animated GIF.  The file format is determined by the file name
  extension.  Any active image transformations (rotation, mirroring, see
  below) are applied to the exported movie.
- **Save current image** (`Ctrl-S`): Save the currently displayed image
  to a file, including any applied transformations (rotation, mirroring,
  see below).  The file format is inferred from the file name extension.
  When the `ImageMagick software <https://imagemagick.org/>`_ is
  installed, additional file formats beyond those natively supported by
  the `Qt library <https://doc.qt.io/qt-6/qtimageformats-index.html>`_
  become available.
- **Copy to clipboard** (`Ctrl-C`): Copy the current image to the system
  clipboard for pasting it into another application.  This requires
  support from the receiving application, which is the case for many
  common applications like document editors and web browsers.
- **Delete selected images**: Remove the image files in the currently
  selected range (see the **Start** and **Stop** controls below) from
  disk.  A confirmation dialog reports how many files will be removed
  before they are deleted.  With the default range (first to last image)
  this removes the entire sequence.  Since the number of image files can
  be large for long simulations, this provides a safe way to clean up the
  working directory without risk of accidentally deleting other files.
  This will, however, only delete images of the last run.  If that was
  stopped before completion or the output filename has changed, older
  images created by previous runs will not be deleted.  Deleting an image
  also discards its converted copy from the image cache described next.
- **Image cache**: An indicator that is grayed out while the image cache
  is empty and shown in color once it holds anything.  Its tooltip reports
  how many converted images and extracted movie frames are cached and how
  much temporary disk space they occupy.  Pressing it discards the
  converted images after a confirmation; they are converted again the next
  time they are displayed, so nothing is lost but time.  Extracted movie
  frames are never discarded this way, since re-creating them requires
  running FFmpeg over the movie again, and the button is therefore
  disabled when the cache holds nothing but frames.  The entire cache is
  removed when the Slide Show window is closed.
- **Zoom in**: Increase the displayed image size by scaling it up. Every
  click on the button increases the zoom factor by 10 percent.
- **Zoom out**: Decrease the displayed image size by scaling it
  down. Every click on the button decreases the zoom factor by 10
  percent until a minimum zoom factor of 0.1.
- **Rotate clockwise**: Rotate the displayed image 90 degrees clockwise.
- **Rotate counter-clockwise**: Rotate the displayed image 90 degrees
  counter-clockwise.
- **Mirror horizontally**: Flip the displayed image along the vertical
  axis.
- **Mirror vertically**: Flip the displayed image along the horizontal
  axis.
- **Reset image**: Reset the displayed image to the original image. This
  reverts all zoom, rotate, and mirror operations.
- **Fit window**: Resize the window so the displayed image is shown at
  its full size, without scroll bars or unused space.  This undoes a
  manual resize of the window; the window is never grown beyond a
  fraction of the screen, so scroll bars remain for very large images.
- **Stop Simulation** (`Ctrl-/`): Stop a running simulation.

These image transformations are useful when the simulation images need
to be adjusted for presentation purposes.  The same transformations are
also applied when exporting images or movies.

The **playback controls** below the image let you select the displayed
image, restrict the active range, and control the slideshow settings:

- **Play**: Start playing the animation, advancing through the active
  range defined by the **Start** and **Stop** controls.
- **Loop**: Toggle continuous looping of the animation.  When enabled,
  playback wraps around from the last image of the active range back to
  the first.
- **Delay**: Set the delay in milliseconds between frames
  during animation playback.
- **Start**: First image of the active range.  Animation, single
  stepping, movie export, and image deletion are all restricted to
  images at or after this position.  Defaults to the first image.
- **Stop**: Last image of the active range.  Defaults to the last image
  and keeps following the growing sequence while a simulation produces
  new images, unless it has been set to a specific value.

- **First**: Jump to the first image of the active range.
- **Previous**: Step back to the previous image.
- The **slider control** selects a frame in the image sequence by moving
  the slider position.  Its track is colored to indicate the active
  range: positions inside the **Start** to **Stop** range are drawn in
  blue, while the skipped images outside it are drawn in red.
- **Next**: Step forward to the next image.
- **Last**: Jump to the last image of the active range.
