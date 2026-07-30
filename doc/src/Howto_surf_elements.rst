.. _howto_13:

Surface elements: explicit, implicit, distributed
=================================================

SPARTA can work with two kinds of surface elements: explicit and
implicit.  Explicit surfaces are lines (2d) or triangles (3d) defined
in surface data files read by the :doc:`read\_surf <read_surf>` command.
An individual element can be any size; a single surface element can
intersect many grid cells.  Implicit surfaces are lines (2d) or
triangles (3d) defined by grid corner point data files read by the
:doc:`read\_isurf <read_isurf>` command.  The corner point values define
lines or triangles that are wholly contained with single grid cells.

Note that you cannot mix explicit and implicit surfaces in the same
simulation.

The data and attributes of explicit surface elements can be stored in
one of two ways.  The default is for each processor to store a copy of
all the elements.  Memory-wise, this is fine for most models.  The
other option is distributed, where each processor only stores copies
of surface elements assigned to grid cells it owns or has a ghost copy
of.  For models with huge numbers of surface elements, distributing
them will use much less memory per processor.  Note that a surface
element requires about 150 bytes of storage, so storing a million
requires about 150 MBytes.

Implicit surfaces are always stored in a distributed fashion.  Each
processor only stores a copy of surface elements assigned to grid
cells it owns or has a ghost copy of.  Note that 3d implicit surfs are
not yet fully implemented.  Specifically, the
:doc:`read\_isurf <read_isurf>` command will not yet read and create
them.

The :doc:`global surfs <global>` command is used to specify the use of
explicit versus implicit, and distributed versus non-distributed
surface elements.

Unless noted, the following surface-related commands work with either
explicit or implicit surfaces, whether they are distributed or not.
For large data sets, the read and write surf and isurf commands have
options to use multiple files and/or operate in parallel which can
reduce I/O times.

* :doc:`adapt\_grid <adapt_grid>`
* :doc:`compute\_isurf/grid <compute_isurf_grid>`    # for implicit surfs
* :doc:`compute\_surf <compute_surf>`                # for explicit surfs
* :doc:`dump surf <dump>`
* :doc:`dump image <dump_image>`
* :doc:`fix adapt/grid <fix_adapt>`
* :doc:`fix emit/surf <fix_emit_surf>`
* :doc:`group surf <group>`
* :doc:`read\_isurf <read_isurf>`                    # for implicit surfs
* :doc:`read\_surf <read_surf>`                      # for explicit surfs
* :doc:`surf\_modify <surf_modify>`
* :doc:`write\_isurf <write_surf>`                   # for implicit surfs
* :doc:`write\_surf <write_surf>`

These command do not yet support distributed surfaces:

* :doc:`move\_surf <move_surf>`
* :doc:`fix move/surf <fix_move_surf>`
* :doc:`remove\_surf <remove_surf>`
