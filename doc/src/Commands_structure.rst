.. _cmd_3:

Input script structure
======================

This section describes the structure of a typical SPARTA input script.
The "examples" directory in the SPARTA distribution contains sample
input scripts; the corresponding problems are discussed in :doc:`Section 5 <Section_example>`, and animated on the `SPARTA WWW Site <sws_>`_.

A SPARTA input script typically has 4 parts:

1. Initialization
2. Problem definition
3. Settings
4. Run a simulation

The last 2 parts can be repeated as many times as desired.  I.e. run a
simulation, change some settings, run some more, etc.  Each of the 4
parts is now described in more detail.  Remember that almost all the
commands need only be used if a non-default value is desired.

(1) Initialization

Set parameters that need to be defined before the simulation domain,
particles, grid cells, and surfaces are defined.

Relevant commands include :doc:`dimension <dimension>`,
:doc:`units <units>`, and :doc:`seed <seed>`.

(2) Problem definition

These items must be defined before running a SPARTA calculation, and
typically in this order:

* :doc:`create\_box <create_box>` for the simulation box
* :doc:`create\_grid <create_grid>` or :doc:`read\_grid <read_grid>` for grid cells
* :doc:`read\_surf <read_surf>` or :doc:`read\_isurf <read_isurf>` for surfaces
* :doc:`species <species>` for particle species properties
* :doc:`create\_particles <create_particles>` for particles

The first two are required.  Surfaces are optional.  Particles are also
optional in the setup stage, since they can be added as the simulation
runs.

The system can also be load-balanced after the grid and/or particles
are defined in the setup stage using the
:doc:`balance\_grid <balance_grid>` command.  The grid can also be
adapted before or between simulations using the
:doc:`adapt\_grid <adapt_grid>` command.

(3) Settings

Once the problem geometry, grid cells, surfaces, and particles are
defined, a variety of settings can be specified, which include
simulation parameters, output options, etc.

Commands that do this include

:doc:`global <global>`
:doc:`timestep <timestep>`
:doc:`collide <collide>` for a collision model
:doc:`react <react>` for a chemistry model
:doc:`fix <fix>` for boundary conditions, time-averaging, load-balancing, etc
:doc:`compute <compute>` for diagnostic computations
:doc:`stats\_style <stats_style>` for screen output
:doc:`dump <dump>` for snapshots of particle, grid, and surface info
:doc:`dump image <dump>` for on-the-fly images of the simulation
:doc:`dump vtk <dump_vtk>` for native VTK-format snapshots (VTK package)

(4) Run a simulation

A simulation is run using the :doc:`run <run>` command.

.. _sws: https://sparta.github.io
