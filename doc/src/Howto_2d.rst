.. _howto_1:

2d simulations
==============

In SPARTA, as in other DSMC codes, a 2d simulation means that
particles move only in the xy plane, but still have all 3 xyz
components of velocity.  Only the xy components of velocity are used
to advect the particles, so that they stay in the xy plane, but all 3
components are used to compute collision parameters, temperatures,
etc.  Here are the steps to take in an input script to setup a 2d
model.

* Use the :doc:`dimension <dimension>` command to specify a 2d simulation.
* Make the simulation box periodic in z via the :doc:`boundary <boundary>`
  command.  This is the default.
* Using the :doc:`create box <create_box>` command, set the z boundaries
  of the box to values that straddle the z = 0.0 plane.  I.e. zlo < 0.0
  and zhi > 0.0.  Typical values are -0.5 and 0.5, but regardless of the
  actual values, SPARTA computes the "volume" of 2d grid cells as if
  their z-dimension length is 1.0, in whatever :doc:`units <units>` are
  defined.  This volume is used with the :doc:`global nrho <global>`
  setting to calculate numbers of particles to create or insert.  It is
  also used to compute collision frequencies.
* If surfaces are defined via the :doc:`read\_surf <read_surf>` command,
  use 2d objects defined by line segments.

Many of the example input scripts included in the SPARTA distribution
are for 2d models.
