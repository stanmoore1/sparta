.. _howto_9:

Details of surfaces in SPARTA
=============================

A SPARTA simulation can define one or more surface objects, each of
which are read in via the :doc:`read\_surf <read_surf>`.  For 2d
simulations a surface object is a collection of connected line
segments.  For 3d simulations it is a collection of connected
triangles.  The outward normal of lines or triangles, as defined in
the surface file, points into the flow region of the simulation box
which is typically filled with particles.  Depending on the
orientation, surface objects can thus be obstacles that particles flow
around, or they can represent the outer boundary of an irregular
shaped region which particles are inside of.

See the :doc:`read\_surf <read_surf>` doc page for a discussion of these
topics:

* Requirement that a surface object be "watertight", so that particles
  do not enter inside the surface or escape it if used as an outer
  boundary.
* Surface objects (one per file) that contain more than one physical
  object, e.g. two or more spheres in a single file.
* Use of geometric transformations (translation, rotation, scaling,
  inversion) to convert the surface object in a file into different
  forms for use in different simulations.
* Clipping a surface object to the simulation box to effectively use a
  portion of the object in a simulation, e.g. a half sphere instead of a
  full sphere.
* The kinds of surface objects that are illegal, including infinitely
  thin objects, ones with duplicate points, or multiple surface or
  physical objects that touch or overlap.

The :doc:`read\_surf <read_surf>` command assigns an ID to the surface
object in a file.  This can be used to reference the surface elements
in the object in other commands.  For example, every surface object
must have a collision model assigned to it so that particle bounces
off the surface can be computed.  This is done via the
:doc:`surf\_modify <surf_modify>` and :doc:`surf\_collide <surf_collide>`
commands.

As described in the previous :ref:`Section 6.8 <howto_8>`, SPARTA overlays a
grid over the simulation domain to track particles.  Surface elements
are also assigned to grid cells they intersect with, so that
particle/surface collisions can be efficiently computed.  Typically a
grid cell size larger than the surface elements that intersect it may
not desirable since it means flow around the surface object will not
be well resolved.  The size of the smallest surface element in the
system is printed when the surface file is read.  Note that if the
surface object is clipped to the simulation box, small lines or
triangles can result near the box boundary due to the clipping
operation.

The maximum number of surface elements that can intersect a single
child grid cell is set by the :doc:`global surfmax <global>` command.
The default limit is 100.  The actual maximum number in any grid cell
is also printed when the surface file is read.  Values this large or
larger may cause particle moves to become expensive, since each time a
particle moves within that grid cell, possible collisions with all its
overlapping surface elements must be computed.
