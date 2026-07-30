.. _howto_14:

Implicit surface ablation
=========================

The implicit surfaces described in the previous section can be used to
perform ablation simulations, where the set of implicit surface
elements evolve over time to model a receding surface.  These are the
relevant commands:

* :doc:`global surfs implicit <global>`
* :doc:`read isurf <read_isurf>`
* :doc:`fix ablate <fix_ablate>`
* :doc:`compute isurf/grid <compute_isurf_grid>`
* :doc:`compute react/isurf/grid <compute_react_isurf_grid>`
* :doc:`fix ave/grid <fix_ave_grid>`
* :doc:`write isurf <write_isurf>`
* :doc:`write\_surf <write_surf>`

The :doc:`read\_isurf <read_isurf>` command takes a binary file as an
argument which contains a pixelated (2d) or voxelated (3d)
representation of the surface (e.g. a porous heat shield material).
It reads the file and assigns the pixel/voxel values to corner points
of a region of the SPARTA grid.

The :doc:`read\_isurf <read_isurf>` command also takes the ID of a :doc:`fix ablate <fix_ablate>` command as an argument.  This fix is invoked
to perform a Marching Squares (2d) or Marching Cubes (3d) algorithm to
convert the corner point values to a set of line segments (2d) or
triangles (3d) each of which is wholly contained in a grid cell.  It
also stores the per grid cell corner point values.

If the *Nevery* argument of the :doc:`fix ablate <fix_ablate>` command
is 0, ablation is never performed, the implicit surfaces are static.
If it is non-zero, an ablation operation is performed every *Nevery*
steps.  A per-grid cell value is used to decrement the corner point
values in each grid cell.  The values can be (1) from a compute such
as :doc:`compute isurf/grid <compute_isurf_grid>` which tallies
statistics about gas particle collisions with surfaces within each
grid cell.  Or :doc:`compute react/isurf/grid <compute_react_isurf_grid>` which tallies the
number of surface reactions that take place.  Or values can be (2)
from a fix such as :doc:`fix ave/grid <fix_ave_grid>` which time
averages these statistics over many timesteps.  Or they can be (3)
generated randomly, which is useful for debugging.

The decrement of grid corner point values is done in a manner that
models recession of the surface elements within in each grid cell.
All the current implicit surface elements are then discarded, and new
ones are generated from the new corner point values via the Marching
Squares or Marching Cubes algorithm.

.. warning::

   Ideally these algorithms should preserve the gas flow
   volume inferred by the previous surfaces and only add to it with the
   new surfaces.  However there are a few cases for the 3d Marching Cubes
   algorithm where the gas flow volume is not strictly preserved.  This
   can trap existing particles inside the new surfaces.  Currently SPARTA
   checks for this condition and deletes the trapped particles.  In the
   future, we plan to modify the standard Marching Cubes algorithm to
   prevent this from happening.  In our testing, the fraction of trapped
   particles in an ablation operation is tiny (around 0.005% or 5 in
   100000).  The number of deleted particles can be monitored as an
   output option by the :doc:`fix ablate <fix_ablate>` command.

The :doc:`write\_isurf <write_isurf>` command can be used to periodically
write out a pixelated/voxelated file of corner point values, in the
same format that the :doc:`read\_isurf <read_isurf>` command reads.  Note
that after ablation, corner point values are typically no longer
integers, but floating point values.  The :doc:`read\_isurf <read_isurf>`
and :doc:`write\_isurf <write_isurf>` commands have options to work with
both kinds of files.  The :doc:`write\_surf <write_surf>` command can
also output implicit surface elements for visualization by tools such
as ParaView which can read SPARTA surface element files after suitable
post-processing.  See the :ref:`Section tools paraview <paraviewtools>` doc page for more details.
