.. _howto_2:

Axisymmetric simulations
========================

In SPARTA, an axi-symmetric model is a 2d model.  An example input
script is provided in the examples/axisymm directory.

An axi-symmetric problem can be setup using the following commands:

* Set dimension = 2 via the :doc:`dimension <dimension>` command.
* Set the y-dimension lower boundary to "a" via the :doc:`boundary <boundary>` command.
* The y-dimension upper boundary can be anything except "a" or "p" for periodic.
* Use the :doc:`create\_box <create_box>` command to define a 2d simulation box with ylo = 0.0.

If desired, grid cell weighting can be enabled via the :doc:`global weight <global>` command.  The *volume* or *radial* setting can be
used for axi-symmetric models.

Grid cell weighting affects how many particles per grid cell are
created when using the :doc:`create\_particles <create_particles>` and
:doc:`fix emit <fix_emit_face>` command variants.

During a run, it also triggers particle cloning and destruction as
particles move from grid cell to grid cell.  This can be important for
inducing every grid cell to contain roughly the same number of
particles, even if cells are of varying volume, as they often are in
axi-symmetric models.  Note that the effective volume of an
axi-symmetric grid cell is the volume its 2d area sweeps out when
rotated around the y=0 axis of symmetry.
