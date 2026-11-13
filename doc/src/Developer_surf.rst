Surfaces
========

Surface elements are the geometry particles collide with: line segments in
2d, triangles in 3d. The ``Surf`` class (``src/surf.h``, ``src/surf.cpp``)
owns them. Working out which grid cells each element intersects, and what
flow volume is left in a cell that a surface passes through, is done in
``src/grid_surf.cpp`` using the computational geometry in
``src/cut2d.cpp`` and ``src/cut3d.cpp``.

`Details of surfaces in SPARTA <Section_howto.html#howto_9>`_ and `Surface elements: explicit, implicit, distributed <Section_howto.html#howto_13>`_ in the How-to chapter cover surfaces from a user's point of
view.

One element
-----------

The two element types are deliberately parallel:

.. code-block:: c++

   struct Line {
     surfint id;              // unique ID for explicit surf
                              // cell ID for implicit surf
     int type,mask;           // type and group mask
     int isc,isr;             // surface collision / reaction model
                              //   indices, -1 if unassigned
     double p1[3],p2[3];      // end points
                              // rhand rule: Z x (p2-p1) = outward normal
     double norm[3];          // outward normal
     int transparent;         // 1 if surf is transparent
   };

``Tri`` is the same with ``p1``, ``p2``, ``p3`` and the right-hand rule
``(p2-p1) x (p3-p1)``.

Two fields deserve attention. ``isc`` and ``isr`` are indices into the
lists of :doc:`surf_collide <surf_collide>` and
:doc:`surf_react <surf_react>` models -- the *per-element* assignment of
physics, which is why different parts of one geometry can bounce and react
differently. ``transparent`` marks an element that particles pass through;
it exists so that flux through a plane can be tallied without perturbing
the flow (`Transparent surface elements <Section_howto.html#howto_15>`_).

The orientation convention matters when writing geometry code. ``norm`` is
the *outward* normal, fixed by the winding of the points under the
right-hand rule given in the comments above, so for a solid body it points
away from the material and into the flow. Consistent winding across a
surface is what makes "is this point inside" answerable, and is one of the
things :doc:`read_surf <read_surf>` checks on input.

Explicit, implicit, distributed
-------------------------------

The :doc:`global <global>` command's ``surfs`` keyword picks one of three
modes -- explicit, explicit/distributed or implicit, which is always
distributed -- and sets the ``Surf::implicit`` and ``Surf::distributed``
flags accordingly.  The storage layout differs for each, and
``src/surf.h`` documents the three cases directly:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Mode
     - Storage
   * - explicit, all
     - Every processor stores every element. ``nlocal`` = total surface
       count, ``nghost`` = 0, ``nown`` = ``Nsurf/P``, ``mylines``/``mytris``
       unused. Simple, and fine until the surface is large.
   * - explicit, distributed
     - Each processor owns ``Nsurf/P`` elements for bookkeeping purposes,
       but stores the elements touching its owned and ghost grid cells.
       ``nlocal``/``nghost`` count those; ``mylines``/``mytris`` hold the
       ``nown`` elements uniquely owned.
   * - implicit, distributed
     - Elements are generated from grid data and belong to the cell that
       produced them, so ownership follows the grid: ``nlocal`` is the
       elements in owned cells, ``nghost`` those in ghost cells, and
       ``nown`` equals ``nlocal``.

Explicit surfaces are read from a file (:doc:`read_surf <read_surf>`) and
have global IDs. Implicit surfaces are computed from per-grid-cell values
by marching squares in 2d or marching cubes in 3d, by
:doc:`read_isurf <read_isurf>` or :doc:`create_isurf <create_isurf>`, so an
element's ``id`` is normally the ID of the grid cell that generated it --
which is why the ``id`` field is documented as meaning two different
things.

The distributed modes exist because storing an entire large surface on
every processor does not scale. The cost is that any operation over "all
surfaces" becomes a communication problem, which is why so much of
``surf.cpp`` has both a replicated and a distributed implementation.

Assigning surfaces to cells
---------------------------

``Grid::ChildCell`` carries ``nsurf`` and ``csurfs``: the number of surface
elements intersecting the cell and their indices. Building this mapping is
the job of ``src/grid_surf.cpp``, which marks every cell a surface crosses
as ``OVERLAP`` (``ChildInfo::type``); the cut-cell code in ``cut2d.cpp`` and
``cut3d.cpp`` sets those cells' per-corner flags (``ChildInfo::corner``),
and ``Grid::set_inout()`` in ``src/grid.cpp`` then marks the cells no surface
crosses as ``OUTSIDE`` or ``INSIDE``.

The distinction drives everything downstream:

* ``OUTSIDE`` cells are entirely in the flow and need no special handling.
* ``INSIDE`` cells are entirely within a solid body; they hold no particles
  and contribute no volume.
* ``OVERLAP`` cells are cut by the surface. Their flow volume is less than
  their geometric volume, and it must be computed.

Cut and split cells
-------------------

``cut2d.cpp`` and ``cut3d.cpp`` answer two questions about an ``OVERLAP``
cell: what volume of it is flow, and is that volume connected?

The computation clips the cell against the surface elements crossing it and
assembles the resulting polygon (2d) or polyhedron (3d). If the flow region
comes out in more than one disconnected piece -- a thin plate cutting a cell
in two, a corner where two bodies nearly meet -- the cell is *split*, and
each piece becomes a sub cell. ``SplitInfo::csplits`` records which sub cell
each of the cell's surface elements belongs to. See
:doc:`Developer_grid` for how split and sub cells are stored.

This is the most intricate geometry in SPARTA, and the part most sensitive
to degenerate input: surfaces that touch a cell face exactly, elements that
are nearly coplanar, points that coincide to within round-off. Much of the
length of ``cut3d.cpp`` is these cases. If you are changing it, the
``examples`` problems with surfaces are the regression net
(:doc:`Developer_testing`), and `Surface elements: explicit, implicit, distributed <Section_howto.html#howto_13>`_
describes the invariants a valid surface must satisfy -- chiefly that it be
watertight.

Surface collisions during a move
--------------------------------

When ``Update::move()`` is instantiated with ``SURF`` set, advecting a
particle through a cell includes testing its path segment against each of
the cell's ``nsurf`` elements. Only hits on the flow (``OUTSIDE``) side of an element count, and the
element the particle has just bounced off is skipped.  The nearest
intersection within the remaining timestep wins; the particle is placed at
the collision point and handed to the element's ``SurfCollide`` model,
which gives it a new velocity and, if the element has a ``SurfReact`` model
too, calls that itself to transform, delete or replace the particle.  It
then continues with the timestep it has left.

Because the test is per element per cell, the number of surface elements in
a cell is a direct performance concern -- which is one reason grid
adaptation around surfaces matters.

Ablation
--------

:doc:`fix ablate <fix_ablate>` removes material by decrementing the
per-corner values that implicit surfaces are generated from, then
regenerating the surface. That makes the surface time-dependent, and means
the grid-surface mapping and all the cut-cell geometry have to be rebuilt
during a run rather than only at setup. `Implicit surface ablation <Section_howto.html#howto_14>`_ covers the
model.

Where to go next
----------------

* :doc:`Developer_grid` -- split and sub cells
* :doc:`Developer_flow` -- where surface collisions sit in the move
* :doc:`Section_modify` -- adding a surface collision or reaction model
