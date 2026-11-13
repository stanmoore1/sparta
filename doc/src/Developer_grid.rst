The grid
========

The grid is SPARTA's central data structure. Particles live in cells,
collisions happen within cells, surface elements are assigned to the cells
they intersect, and the grid is the unit of parallel decomposition. The
implementation is in ``src/grid.h`` and ``src/grid.cpp``, with
grid-surface interaction split out into ``src/grid_surf.cpp``.

The `grid section of the How-to chapter <Section_howto.html#howto_8>`_ and
the :doc:`create_grid <create_grid>` and :doc:`adapt_grid <adapt_grid>`
commands describe the grid from a user's point of view -- how to define and
adapt one. This page describes how it is stored.

Hierarchical cells
------------------

SPARTA uses a hierarchical Cartesian grid. The simulation box is level 0,
the root. A cell at any level can be subdivided into a fixed
:math:`N_x \times N_y \times N_z` block of children, and those children can
themselves be subdivided. Cells that are not subdivided are *child cells*
and are the ones that hold particles; the subdivided ancestors exist only
to define the hierarchy.

Levels are described by ``ParentLevel`` (``src/grid.h``), which records the
subdivision factors for the level and how many bits of a cell ID it uses:
``nbits`` is the number of bits taken by the ID of a parent at this level --
everything above it in the hierarchy -- and ``newbits`` the number this
level adds to number its children:

.. code-block:: c++

   struct ParentLevel {
     int nbits;      // # of bits to store parent ID at this level
     int newbits;    // extra bits to store children of this parent
     ...
   };

A cell ID is a bit-packed path through the hierarchy rather than an index,
built from the root outwards.  The low bits hold the cell's index among the
children of the root (1 to :math:`N_x N_y N_z`; 0 is the root itself); each
deeper level's child index is shifted left past all of its ancestors' bits,
``childID = (ichild << plevels[level].nbits) | parentID``
(``Grid::id_find_child``, ``src/grid_id.cpp``).  So the parent's ID is the
child's ID masked to its low ``nbits`` bits (``Grid::id_parent_of_child``),
and the child's own index is what is left above them. This is why a cell ID has type
``cellint`` rather than ``int``, and why ``-DSPARTA_BIGBIG`` widens
``cellint`` to 64 bits -- deep hierarchies over large boxes exhaust 32 bits.
See :doc:`Developer_utils` for the size variants.

The consequence worth internalizing: given a cell ID you can compute its
parent, its level, and its position in the box arithmetically, using only
the small per-level ``plevels`` table rather than any per-cell lookup.  The
functions that do it are in ``src/grid_id.cpp``.

Cells on a processor
--------------------

Each processor stores a list of cells in ``Grid::cells``, an array of
``ChildCell``. The array holds both cells this processor owns and *ghost*
cells -- copies of cells owned by neighbors, needed so a particle can be
advected across a processor boundary before being migrated.

.. code-block:: c++

   struct SPARTA_ALIGN(64) ChildCell {
     cellint id;        // ID of child cell
     int level;         // level in the hierarchy, 0 = root
     int proc;          // proc that owns this cell
     int ilocal;        // index of this cell on the owning proc
     cellint neigh[6];  // the 6 face neighbors
     int nmask;         // 3 bits per neigh entry, see below
     double lo[3],hi[3];// opposite corner points
     int nsurf;         // # of surf elements in cell, -1 = empty ghost
     surfint *csurfs;   // indices of those surf elements
     int nsplit;        // 1 unsplit; N>1 split into N; N<=0 sub cell index
     int isplit;        // index into sinfo, -1 if unsplit
   };

``SPARTA_ALIGN(64)`` aligns the struct to 64 bytes only in KOKKOS GPU builds
(``src/spatype.h``); everywhere else it expands to nothing.

Counts are kept separately, and the distinction matters when writing code
that loops over cells:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Counter
     - Meaning
   * - ``nlocal``
     - child cells this processor owns, of all three kinds
   * - ``nghost``
     - ghost child cells stored but owned elsewhere
   * - ``nunsplitlocal``
     - owned unsplit cells
   * - ``nsplitlocal``
     - owned split cells
   * - ``nsublocal``
     - owned sub cells

Two parallel arrays carry more: ``Grid::cinfo`` (``ChildInfo``), for owned
cells only, and ``Grid::sinfo`` (``SplitInfo``), for split cells -- owned
ones first, then ghost ones appended after them.

``ChildInfo`` holds the per-cell state that changes during a run:

.. code-block:: c++

   struct ChildInfo {
     int count;       // # of particles in this cell
     int first;       // index of 1st particle in this cell, -1 if none
     int mask;        // grid group mask
     int type;        // OUTSIDE, INSIDE, OVERLAP, UNKNOWN
     int corner[8];   // corner flags, 4 in 2d / 8 in 3d
     double volume;   // flow volume of the cell
     double weight;   // fnum weighting for this cell
   };

``count`` and ``first`` are the head of the per-cell particle list; see
:doc:`Developer_particle`. ``type`` records the cell's relationship to the
surfaces: OUTSIDE means outside every surface object, i.e. entirely in the
flow; INSIDE means entirely inside a surface object, with no flow; OVERLAP
means a non-transparent surface element crosses it, so it needs cut-cell geometry; UNKNOWN is the
state before it has been classified.

Face neighbors and ``nmask``
----------------------------

``neigh[6]`` holds the six face neighbors in the order XLO, XHI, YLO, YHI,
ZLO, ZHI. What each entry *means* depends on three bits in ``nmask``,
because a neighbor may be a cell this processor owns, a coarser parent
cell, or not known at all:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Value
     - ``neigh`` entry is
   * - 0
     - index of a child neighbor this processor stores
   * - 1
     - index of a parent neighbor in ``pcells``
   * - 2
     - unknown child neighbor
   * - 3
     - as 0, reached through a periodic boundary
   * - 4
     - as 1, reached through a periodic boundary
   * - 5
     - as 2, reached through a periodic boundary
   * - 6
     - a non-periodic boundary, or ZLO/ZHI in 2d

An "unknown" entry is 0 and is not used.  The field is typed ``cellint``
rather than ``int`` because ``Grid::unset_neighbors()`` temporarily replaces
the local indices of child and parent neighbors with global cell IDs --
around a rebalance, while cells move between processors, or a change to
the surfaces -- and
``Grid::reset_neighbors()`` turns them back into local indices.  In the
common case the move loop steps from cell to cell with a single array
lookup.

Split cells and sub cells
-------------------------

A cell that a surface passes through may have its flow volume divided into
several disconnected regions -- imagine a thin plate slicing a cell in two.
SPARTA represents this by *splitting* the cell.

A split cell is a placeholder: it keeps the geometry of the original cell
but holds no particles. Its ``nsplit`` is the number ``N`` of pieces. It
owns ``N`` *sub cells*, each representing one connected flow region, and
those are the cells that actually hold particles. A sub cell's ``nsplit``
is the negative of its index within the parent split cell, and its
``lo``/``hi``/``nsurf``/``csurfs`` are copied from the split cell.

``SplitInfo`` ties the pieces together:

.. code-block:: c++

   struct SplitInfo {
     int icell;      // index of the split cell these sub cells belong to
     int xsub;       // which sub cell xsplit lies in
     double xsplit[3];
     int *csplits;   // which sub cell each of the Nsurf elements belongs to
     int *csubs;     // indices in cells of the Nsplit sub cells
   };

The practical rule when writing new code: a loop over ``nlocal`` sees
unsplit cells, split cells and sub cells all mixed together. If you are
accumulating a per-cell quantity you almost always want to skip split
cells -- they have ``nsplit > 1``, zero particles, and their volume is the
whole cell rather than a flow region -- and process sub cells instead.
The ``nsplit > 1`` tests in ``src/fix_ave_grid.cpp`` and
``src/compute_distsurf_grid.cpp`` show the standard pattern.

Grid and surface together
-------------------------

Deciding which cells a surface intersects, computing cut-cell volumes and
deciding how many sub cells a split cell needs is the job of
``src/grid_surf.cpp`` together with the computational geometry in
``src/cut2d.cpp`` and ``src/cut3d.cpp``; ``Grid::set_inout()`` in
``src/grid.cpp`` then classifies the cells no surface crosses as inside or
outside. That machinery is described in :doc:`Developer_surf`.

Where to go next
----------------

* :doc:`Developer_particle` -- how particles attach to these cells
* :doc:`Developer_surf` -- cut cells and surface geometry
* :doc:`Developer_parallel` -- how cells are assigned to processors
