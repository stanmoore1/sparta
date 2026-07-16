########################
SPARTA-GUI Documentation
########################

.. toctree::
   :caption: SPARTA-GUI Documentation

.. only:: html

   .. image:: _images/sparta-gui-banner.png
      :align: center
      :scale: 75%


****************
About SPARTA-GUI
****************

SPARTA-GUI is a graphical text editor with syntax highlighting,
auto-completion, inline help, and indentation support for input scripts
of the `SPARTA Direct Simulation Monte Carlo (DSMC) code
<https://sparta.github.io/>`_.  It is programmed using the `Qt Framework
<https://www.qt.io/>`_ and customized for running, monitoring, and
visualizing SPARTA simulations.  It calls SPARTA directly using the
`SPARTA library interface
<https://sparta.github.io/doc/Section_howto.html#howto_6>`_ instead of
launching an external SPARTA executable.  Therefore it can retrieve and
display information from SPARTA *while it is running* and *immediately*
display visualizations created by a `dump image command
<https://sparta.github.io/doc/dump_image.html>`_ in the input.

The primary motivation for SPARTA-GUI is to make it easy to get started
with SPARTA and to have a consistent behavior across major platforms
like Linux and macOS.  This way one can focus on learning SPARTA itself
and avoid having to spend time learning different tools (for editing
inputs, plotting graphs, visualizing systems) on the different
platforms.

Many of the features in SPARTA-GUI are useful beyond learning SPARTA.
For instance, it can streamline the process of prototyping new
simulation projects or debugging misbehaving simulations.  It also is
extremely useful for creating, debugging, and tweaking visualizations
with the built-in rendering facility of the `dump image command
<https://sparta.github.io/doc/dump_image.html>`_, covering particles,
grid cells, grid cut planes, and surface elements.

***********
Attribution
***********

SPARTA-GUI is a derived work of `LAMMPS-GUI
<https://github.com/akohlmey/lammps-gui>`_ version 3.0.3 by Axel
Kohlmeyer, adapted for the SPARTA DSMC code.  The look and feel of the
application, most of its infrastructure, and large parts of this manual
originate from LAMMPS-GUI; the SPARTA adaptation replaces the
LAMMPS-specific functionality (running molecular dynamics inputs,
atom/bond visualization, thermodynamic output capture) with their SPARTA
equivalents (DSMC input scripts, particle/grid/surface visualization,
stats output capture).

LAMMPS-GUI is Copyright (c) 2023 - 2026 Axel Kohlmeyer.  SPARTA-GUI, as
a whole and like SPARTA itself, is distributed under the terms of the
GNU General Public License version 2.0 (GPL-2.0-or-later for the files
retained from LAMMPS-GUI).

--------

*******************
About this document
*******************

This document contains the documentation of SPARTA-GUI and how to
compile, install, use, configure, and modify it.  Suggestions for new
features and reports of bugs are always welcome.  You can use the `same
channels as for SPARTA itself <https://sparta.github.io/>`_ for that
purpose or submit bug reports or pull requests in the `SPARTA GitHub
repository <https://github.com/sparta/sparta>`_.

------------------

.. raw:: html

   <h2>

This document describes SPARTA-GUI version |version|.

.. raw:: html

   </h2>

------------------

.. raw:: latex

   \clearpage

************
User's Guide
************

.. toctree::
   :caption: Table of Contents
   :maxdepth: 2
   :numbered: 3
   :name: userdoc
   :includehidden:

   installation
   overview
   basic_usage
   output
   visualization
   editor
   menus
   dialogs
   shortcuts

------------------

.. raw:: latex

   \clearpage

******************
Programmer's Guide
******************

This guide provides documentation for developers who want to understand
the internals of SPARTA-GUI or contribute to its development.

.. admonition:: AI Generated Content
   :class: note

   The initial version of the Programmer's Guide section (for the
   LAMMPS-GUI project, from which these pages are derived) was created
   by the `GitHub Copilot Coding Agent
   <https://docs.github.com/en/copilot>`_ and not everything has been
   carefully checked.  It is therefore possible that it contains errors
   where the LLM has misinterpreted the source code.  If you spot any
   such errors or inconsistencies, please submit a bug report issue to
   point them out or -- even better -- submit a pull request with the
   necessary corrections.

.. toctree::
   :caption: Table of Contents
   :maxdepth: 2
   :numbered: 3
   :name: progdoc
   :includehidden:

   introduction
   api_reference
   guidelines
   testing

----------

.. only:: html

   ****************
   Index and Search
   ****************

          * :ref:`genindex`
          * :ref:`search`

   ----------

   .. _webbrowser:
   .. admonition:: Web Browser Compatibility

     This website makes use of advanced features present in "modern" web
     browsers.  This leads to incompatibilities with older web browsers
     and specific vendor browsers (e.g. Internet Explorer on Windows)
     where parts of the pages are not rendered as expected (e.g. the
     layout is broken or mathematical expressions not typeset).

     The following web browser versions have been verified to work as
     expected on Linux, macOS, and Windows where available:

     - Safari version 11.1 and later
     - Firefox version 54 and later
     - Chrome version 54 and later
     - Opera version 41 and later
     - Edge version 80 and later

     Also Android version 7.1 and later and iOS version 11 and later have
     been verified to render this website as expected.
