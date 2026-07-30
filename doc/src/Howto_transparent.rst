.. _howto_15:

Transparent surface elements
============================

Transparent surfaces are useful for tallying flow statistics.
Particles pass through them unaffected.  However the flux of particles
through those surface elements can be tallied and output.

Transparent surfaces are treated differently than regular surfaces.
They do not need to be watertight.  E.g. you can define a set of line
segments that form a straight (or curved) line in 2d.  Or a set of
triangle that form a plane (or curved surface) in 3d.  You can define
multiple such surfaces, e.g. multiple disjoint planes, and tally flow
statistics through each of them.  To tally or sum the statistics
separately, you may want to assign the triangles in each plane to a
different surface group via the :doc:`read\_surf group <read_surf>` or
:doc:`group surf <group>` commands.

Note that for purposes of collisions, transparent surface elements are
one-sided.  A collision is only tallied for particles passing through
the outward face of the element.  If you want to tally particles
passing through in both directions, then define 2 transparent
surfaces, with opposite orientation.  Again, you may want to put the 2
surfaces in separate groups.

There also should be no restriction on transparent surfaces
intersecting each other or intersecting regular surfaces.  Though
there may be some corner cases we haven't thought about or tested.

These are the relevant commands.  See their doc pages for details:

* :doc:`read\_surf transparent <read_surf>`
* :doc:`surf\_collide transparent <surf_collide>`
* :doc:`compute surf <compute_surf>`

The :doc:`read\_surf <read_surf>` command with its *transparent* keyword
is used to flag all the read-in surface elements as transparent.  This
means they must be in a file separate from regular non-transparent
elements.

The :doc:`surf\_collide <surf_collide>` command must be used with its
*transparent* model and assigned to all transparent surface elements
via the :doc:`surf\_modify <surf_modify>` command.

The :doc:`compute\_surf <compute_surf>` command can be used to tally the
count, mass flux, and energy flux of particles that pass through
transparent surface elements.  These quantities can then be time
averaged via the :doc:`fix ave/surf <fix_ave_surf>` command or output
via the :doc:`dump surf <dump>` command in the usual ways,
as described in :ref:`Section 6.4 <howto_4>`.

The examples/circle/in.circle.transparent script shows how to use
these commands when modeling flow around a 2d circle.  Two additional
transparent line segments are placed in front of the circle to tally
particle count and kinetic energy flux in both directions in front of
the object.  These are defined in the data.plane1 and data.plane2
files.  The resulting tallies are output with the
:doc:`stats\_style <stats_style>` command.  They could also be output
with a :doc:`dump surf <dump>` command for more resolution if the 2
lines were each defined as multiple line segments.
