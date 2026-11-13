Flow of control
===============

This page follows one SPARTA run from the input script down to a single
timestep. The relevant files are ``src/input.cpp``, ``src/run.cpp`` and
above all ``src/update.cpp``, which owns the timestepper.

From input script to run
------------------------

``main()`` calls ``Input::file()``, which reads the input script one line
at a time and executes each command as it is read. There is no separate
parse phase: a command takes effect immediately, which is why ordering
matters in an input script.

``Input`` handles a handful of commands itself -- variable substitution,
``if``, ``jump``, ``include``, ``shell`` and so on. Everything else is
dispatched either to an ``Input`` method that handles it or passes it
to the class that owns it (``Input::fix()`` and ``Input::compute()`` hand
theirs to ``Modify``; ``Input::collide_command()`` creates the collision
style itself), or, for
one-shot commands such as ``create_grid``, to a ``Command``
style instantiated from the generated ``style_command.h``, run once, and
destroyed.

The ``run`` command creates a ``Run`` object (``src/run.cpp``) which calls,
in order:

1. ``SPARTA::init()`` -- calls ``init()`` on every subsystem, starting with
   ``Update::init()``, which resolves settings that depend on the whole
   configuration and chooses the move method (below).
2. ``Update::setup()`` -- zero the per-run counters, build the lists of
   tally and time-varying models, invoke ``Modify::setup()`` so fixes and
   computes can initialize, and call ``Output::setup()``, which prints the
   first line of stats output.  ``Collide::setup()`` follows.
3. ``Update::run(nsteps)`` -- the timestep loop.

Choosing the move method
------------------------

The particle move is the hottest routine in the code, so SPARTA does not
branch on dimensionality inside it. Instead ``Update::move()`` is a
template

.. code-block:: c++

   template < int DIM, int SURF, int OPT > void Update::move()

and ``Update::init()`` selects one instantiation and stores it in the
member function pointer ``moveptr``:

.. code-block:: c++

   if (domain->dimension == 3) {
     if (surf->exist)
       moveptr = &Update::move<3,1,0>;
     else {
       if (optmove_flag) moveptr = &Update::move<3,0,1>;
       else moveptr = &Update::move<3,0,0>;
     }
   } else if (domain->axisymmetric) {
     moveptr = &Update::move<1,...>;
   } else if (domain->dimension == 2) {
     moveptr = &Update::move<2,...>;
   }

So ``DIM`` is 2 or 3, or 1 for the axisymmetric case; ``SURF`` is 1 when
surface elements exist; and ``OPT`` selects an optimized path available
only when there are no surfaces. Every combination is compiled separately,
and the dimensionality tests inside the move vanish at compile time. The
cost is compile time and code size; the benefit is that the inner loop
carries no branches it does not need.

The timestep
------------

``Update::run()`` is a loop over ``nsteps``. Stripped of timers and
conditionals, each iteration does this:

1. **Per-step bookkeeping.** ``collide_react_reset()`` zeroes the
   surface collision and reaction tallies, ``tally_set()`` picks which gas,
   surface and boundary tally computes are active this step, and
   ``dynamic_update()`` refreshes the time-varying parameters of surface
   collision models.

2. **Start-of-step fixes.** ``Modify::start_of_step()`` invokes every fix
   that requested this callback -- for example :doc:`fix emit/face
   <fix_emit_face>`, which inserts new particles at a boundary before
   anything moves.

3. **Move.** ``(this->*moveptr)()`` advects every particle. This is the
   bulk of the work, and is described below.

4. **Migrate.** ``Comm::migrate_particles()`` sends particles that ended
   the move owned by another processor to that processor. See
   :doc:`Developer_parallel`.

5. **Sort.** If collisions are enabled, ``Particle::sort()`` bins particles
   into per-cell linked lists, since the collision step works cell by cell.
   ``Particle::reorder()`` optionally rewrites the particle array into
   cell order for better memory locality -- see :doc:`Developer_particle`.

6. **Collide.** ``Collide::collisions()`` performs inter-particle
   collisions and, through the ``React`` class, gas-phase chemistry, one
   grid cell at a time.

7. **End-of-step fixes.** ``Modify::end_of_step()`` invokes fixes that
   accumulate or post-process results -- time averaging, grid adaptation,
   load balancing.

8. **Output.** When the step matches ``Output::next``, ``Output::write()``
   emits stats, dump snapshots and restart files.

Grid-based particle weighting, if enabled, wraps the move and migration
with
``Particle::pre_weight()`` and ``Particle::post_weight()``.

``Timer::stamp()`` is called between phases; the categories it accumulates
(``TIME_MOVE``, ``TIME_COMM``, ``TIME_SORT``, ``TIME_COLLIDE``,
``TIME_MODIFY``, ``TIME_OUTPUT``) are what appear in the timing breakdown
at the end of a run.

Inside the move
---------------

``Update::move()`` is not a simple loop over particles. A particle may cross
several cells in one timestep, may hit surface elements along the way, and
may end up on another processor partway through -- so the move is structured
as *move/migrate iterations*:

* The first iteration processes all of this processor's particles.
* Each particle is advected from cell to cell. For each cell, the code
  computes the fraction of the remaining timestep needed to reach the
  nearest cell face, checks for surface collisions within the cell when
  ``SURF`` is set, and either stops the particle or moves it into the
  neighbor cell and repeats.
* A particle that moves into a ghost cell -- one owned by another
  processor -- keeps moving; if its move ends there it is flagged
  ``PDONE`` and sent to the owner by the migration step in
  ``Update::run()``.
* Only with a finite ghost cutoff (:doc:`global gridcut <global>`) can a
  particle reach a cell this processor knows nothing about.  Its move is
  then suspended with time remaining (``PEXIT``/``PENTRY``), it is migrated
  mid-move, and the loop runs again over the particles just received.
* Those iterations continue while any processor has such particles.

Each particle carries a flag tracking its state through this process. The
values are enumerated at the top of ``src/update.cpp``:

.. code-block:: c++

   enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};

``PKEEP`` is a particle this processor still owns and must move,
``PINSERT`` one just created by a fix, ``PDONE`` one whose move is
complete, ``PDISCARD`` one to be deleted, and ``PENTRY``/``PEXIT`` mark
particles entering or leaving in the middle of a move.

Cell-face crossing is deliberately written to be as cheap as possible,
since the common case is a particle that crosses no face and hits nothing.
Surface collision testing, cell-boundary geometry for split cells, and
the axisymmetric remap of (y, z) back into the radial plane are all handled off the
fast path.

Where to go next
----------------

* :doc:`Developer_grid` -- the cell structures the move walks through
* :doc:`Developer_surf` -- what a surface collision test actually does
* :doc:`Developer_parallel` -- the migration half of move/migrate
