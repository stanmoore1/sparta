Particles
=========

The ``Particle`` class (``src/particle.h``, ``src/particle.cpp``) owns the
particles on a processor, the species definitions they refer to, and the
machinery that sorts them into grid cells.

:doc:`Howto_particles` covers particles from a user's point of view. This
page describes the storage.

One particle
------------

Particles are stored in a single flat array, ``Particle::particles``, of
``OnePart`` structs:

.. code-block:: c++

   struct SPARTA_ALIGN(16) OnePart {
     int id;            // particle ID
     int ispecies;      // index into the species list
     int icell;         // which local Grid::cells entry it is in
     int flag;          // migration status during a move
     double x[3];       // position
     double v[3];       // velocity
     double erot;       // rotational energy
     double evib;       // vibrational energy
     double dtremain;   // portion of the move timestep still to go
     double weight;     // particle or cell weight, if weighting is on
   };

``nlocal`` is how many are in use and ``maxlocal`` how many the array can
hold; the array is grown geometrically rather than per particle.

Two fields exist only for the duration of a move. ``dtremain`` is how much
of the timestep a particle has left when its move is suspended -- either
because it is being migrated to another processor or because it stopped at
a surface -- and ``flag`` is the ``PKEEP``/``PDONE``/``PENTRY``/... state
described in :doc:`Developer_flow`.

Note what is *not* in the struct: no force, no neighbor list, no per-particle
history. A DSMC particle is a sample of many real molecules and interacts
only through stochastic collisions with others in the same cell, so the
per-particle state is small. This is why SPARTA can carry very large
particle counts.

Species
-------

``OnePart::ispecies`` indexes a list of ``Species`` structs read from a
species file in ``data/``. Species carry molecular mass, rotational and
vibrational degrees of freedom, reference cross sections and the
temperature exponents used by the VSS/VHS collision models. The
:doc:`mixture <mixture>` command groups species for the purposes of
creating particles and computing per-group statistics; a mixture is a set
of species indices plus fractions, not a copy of the species data.

Sorting into cells
------------------

Collisions are computed cell by cell, so before ``Collide::collisions()``
runs the particles must be grouped by cell. SPARTA does this without
reordering the particle array, using a linked list threaded through a
parallel array:

* ``Particle::next`` is an array with one entry per particle: the index of
  the next particle in the same cell, or ``-1``.
* ``Grid::cinfo[icell].first`` is the index of the first particle in that
  cell, or ``-1``.
* ``Grid::cinfo[icell].count`` is how many particles are in the cell.

``Particle::sort()`` rebuilds these each timestep. The idiom for walking
one cell's particles is therefore:

.. code-block:: c++

   int ip = cinfo[icell].first;
   while (ip >= 0) {
     ... particles[ip] ...
     ip = next[ip];
   }

The ``sorted`` flag records whether the lists are currently valid. Anything
that adds, deletes or moves particles invalidates them.

Following a linked list scatters memory accesses, which costs cache misses
in the collision loop. ``Particle::reorder()`` addresses this by physically
rewriting the particle array into cell order, so that each cell's particles
are contiguous. It is invoked from the main loop when the
:doc:`global <global>` command's reorder period is set, and only there --
reordering is a bulk operation that is not worth doing every step.

Compression and rebalancing
---------------------------

Particles that leave the box, are deleted by chemistry, or migrate away
leave holes in the array. ``Particle::compress_*()`` methods close these
up. There are several variants because the right strategy differs: after a
migration the surviving particles are known by flag, while after a
rebalance whole cells' worth of particles move at once and the sorted
lists can be preserved. ``compress_rebalance_sorted()`` is the variant used
when the caller wants the per-cell lists to survive.

Custom attributes
-----------------

SPARTA lets input scripts attach arbitrary named attributes to particles --
and also to grid cells and surface elements -- through the
:doc:`custom <custom>` command. These are stored outside ``OnePart``, in
per-attribute arrays owned by ``Particle``.

The bookkeeping distinguishes four shapes, each with its own count and
index array:

* integer vectors (``ncustom_ivec``, ``icustom_ivec``)
* integer arrays (``ncustom_iarray``, ``icustom_iarray``)
* double vectors (``ncustom_dvec``, ``icustom_dvec``)
* double arrays (``ncustom_darray``, ``icustom_darray``)

``ncustom`` counts all attributes including deleted ones, so the index
arrays are the reliable way to iterate. Any code that grows, compresses,
migrates or restarts particles has to carry the custom arrays along with
the main array -- this is the main reason for the number of near-duplicate
methods in ``particle.cpp``.

:doc:`Howto_custom` describes the user-facing side, and the same mechanism
appears on ``Grid`` and ``Surf`` with the same four-shape structure.

Weighting
---------

When grid-based particle weighting is enabled, each cell has an ``fnum``
weight (``ChildInfo::weight``) and each particle carries the weight it was
created with. Changing cells therefore changes a particle's statistical
significance, which is resolved by cloning or deleting particles at cell
boundaries. ``Particle::pre_weight()`` and ``post_weight()`` bracket the
move to do this; they are no-ops unless ``Grid::cellweightflag`` is set.

Where to go next
----------------

* :doc:`Developer_grid` -- the cells particles are sorted into
* :doc:`Developer_flow` -- where sorting and weighting sit in the timestep
* :doc:`Developer_parallel` -- how particles migrate between processors
