Parallelism
===========

SPARTA is parallelized with MPI by distributing *grid cells* across
processors. Particles follow the cells they are in, and surface elements
are either replicated or distributed as described in
:doc:`Developer_surf`. The relevant files are ``src/comm.cpp``,
``src/irregular.cpp``, ``src/rcb.cpp`` and ``src/balance_grid.cpp``.

What a processor owns
---------------------

A processor owns a subset of the child cells and every particle in them.
It also stores *ghost* copies of some cells it does not own -- enough that a
particle can be advected up to and across a processor boundary before it
has to be handed off. ``Grid::nlocal`` and ``Grid::nghost`` count the two
groups; see :doc:`Developer_grid`.

``ChildCell::proc`` records the owning processor and ``ChildCell::ilocal``
that cell's index on its owner, so a cell in the ghost region carries
enough information to address the real thing.

Assigning cells to processors
-----------------------------

Cells are assigned either at setup (:doc:`create_grid <create_grid>`) or
during a run (:doc:`balance_grid <balance_grid>`,
:doc:`fix balance <fix_balance>`). ``src/balance_grid.cpp`` implements the
strategies, enumerated at the top of the file:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Style
     - Assignment
   * - ``STRIDE``
     - cells dealt out round-robin by ID
   * - ``CLUMP``
     - contiguous runs of IDs per processor
   * - ``BLOCK``
     - a Cartesian block decomposition of the box
   * - ``RANDOM``
     - random assignment
   * - ``PROC``
     - explicit per-cell assignment
   * - ``BISECTION``
     - recursive coordinate bisection (``src/rcb.cpp``)

The choice is not cosmetic. ``STRIDE`` and ``RANDOM`` give excellent load
balance and terrible locality: a processor's cells are scattered through
the box, so almost every cell face is a processor boundary and almost every
particle move crosses one. ``BLOCK`` and ``BISECTION`` give each processor
a compact region, so most moves stay local.

``Grid::clumped`` records which regime the current decomposition is in -- 1
when each processor's cells form a contiguous block, as RCB produces. Some
algorithms are only valid, or only efficient, when the decomposition is
clumped, and check this flag.

Recursive coordinate bisection is the workhorse for adaptive runs: it cuts
the box with a plane, splits processors between the two halves in
proportion to the work on each side, and recurses. The weight it balances
can be cell count, particle count, or measured time -- the ``CELL``,
``PARTICLE`` and ``TIME`` options of :doc:`fix balance <fix_balance>`.
Particle count is usually the right proxy, since collision work scales with
particles rather than cells.

Moving particles: migration
---------------------------

``Comm::migrate_particles()`` is called after every move. Its input is the
list of particles whose move ended -- or was suspended -- in a cell owned by
another processor.

The communication pattern is *irregular*: a processor does not know in
advance which processors will send to it, or how much. ``src/irregular.cpp``
handles this in the standard way -- an initial exchange establishes who is
sending to whom and how many bytes, after which the data itself is sent
with point-to-point calls that both sides are expecting.

Because migration happens inside the move/migrate iteration described in
:doc:`Developer_flow`, a particle can migrate more than once in a single
timestep: a fast particle crossing a corner of the decomposition may pass
through several processors' cells before its ``dtremain`` reaches zero.

Moving cells: rebalancing
-------------------------

``Comm::migrate_cells()`` moves cells themselves, which happens when the
grid is adapted or rebalanced. This is heavier than particle migration
because a cell carries its particles, its surface element list, its custom
attributes and its per-cell state.

``migrate_cells_less_memory()`` is an alternative implementation that
trades speed for peak memory, sending in batches rather than packing
everything at once. Which one runs is a user-facing choice, because for
large problems the packed buffer can be what exhausts memory.

Collective patterns
-------------------

``comm.cpp`` provides two reusable patterns beyond point-to-point
irregular exchange:

* ``Comm::ring()`` passes a buffer around all processors in a ring,
  invoking a callback at each hop. It is the fallback for operations where
  every processor must see every other processor's data and the volume is
  small.
* ``Comm::rendezvous()`` implements the rendezvous algorithm: when
  processor A has data that belongs to processor B but neither knows the
  other, both send to a deterministically chosen intermediary that can pair
  them up. This is how SPARTA handles operations like matching distributed
  surface elements to the cells that need them, without any processor
  holding a global map.

``rendezvous_stats()`` reports the volume moved, which is useful when
diagnosing why a setup phase is slow at scale.

Reproducibility
---------------

Results depend on the processor count. Random numbers are consumed in a
different order when the same particles are distributed differently, so two
runs on different processor counts diverge immediately at the level of
individual particle trajectories. Statistical properties -- temperature,
density, surface flux -- should agree within sampling error, and that is
what the regression tests check. :doc:`Developer_testing` describes how,
and :doc:`Errors_common` explains the same point for users.

Where to go next
----------------

* :doc:`Developer_grid` -- owned versus ghost cells
* :doc:`Developer_kokkos` -- on-node parallelism, which is orthogonal to this
* :doc:`Section_accelerate` -- the user-facing performance guide
