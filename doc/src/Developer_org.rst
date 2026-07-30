Code organization
=================

SPARTA is written in C++ with a deliberately shallow object model. Roughly
300 source and header files sit directly in ``src``, with optional packages
in subdirectories. The class hierarchy is only two or three levels deep in
most places, and the methods that do real computation are written in plain
C-style code operating on plain C-style data -- arrays and structs, not deep
object graphs.

Everything lives in the ``SPARTA_NS`` namespace.

The source tree
---------------

``src`` holds the core of the code. Its subdirectories are optional
packages and build machinery:

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Directory
     - Contents
   * - ``src/KOKKOS``
     - The KOKKOS package: GPU- and thread-capable versions of many styles.
       See :doc:`Developer_kokkos`.
   * - ``src/FFT``
     - The FFT package, used by :doc:`compute fft/grid <compute_fft_grid>`.
   * - ``src/PYTHON``
     - The PYTHON package, which lets input scripts call Python functions.
   * - ``src/VTK``
     - The VTK package, for the ``dump grid/vtk``, ``dump particle/vtk`` and
       ``dump surf/vtk`` styles.
   * - ``src/STUBS``
     - A dummy MPI library, so SPARTA can be built and run serially without
       a real MPI installation. See :doc:`Developer_utils`.
   * - ``src/MAKE``
     - Machine-specific makefiles for the traditional ``make`` build. The
       cmake build in ``cmake/`` does not use these.

Outside ``src``: ``doc`` is this manual, ``examples`` holds the test
problems that double as the regression suite
(:doc:`Developer_testing`), ``tools`` holds pre- and post-processing
scripts, ``lib`` holds optional external libraries, ``python`` holds the
Python wrapper, and ``data`` holds species and reaction parameter files.

The universe object
-------------------

``main()`` in ``src/main.cpp`` does very little. It initializes MPI,
constructs one ``SPARTA`` object, hands it the input script, and destroys
it:

.. code-block:: c++

   SPARTA *sparta = new SPARTA(argc,argv,MPI_COMM_WORLD);
   sparta->input->file();
   delete sparta;

That ``SPARTA`` object -- declared in ``src/sparta.h``, implemented in
``src/sparta.cpp`` -- is the top of the ownership tree. It holds one pointer
to each major subsystem, and those pointers are the spine of the whole
code:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Member
     - Class
     - Responsibility
   * - ``particle``
     - ``Particle``
     - the particles, their species, and per-cell sorting
   * - ``grid``
     - ``Grid``
     - the hierarchical grid of cells
   * - ``surf``
     - ``Surf``
     - surface elements and their intersection with the grid
   * - ``update``
     - ``Update``
     - the timestepper; owns the move loop
   * - ``comm``
     - ``Comm``
     - inter-processor communication and particle migration
   * - ``domain``
     - ``Domain``
     - the simulation box and its boundary conditions
   * - ``collide``
     - ``Collide``
     - inter-particle collisions
   * - ``react``
     - ``React``
     - gas-phase chemistry
   * - ``modify``
     - ``Modify``
     - the lists of active fixes and computes
   * - ``output``
     - ``Output``
     - stats, dumps and restart files
   * - ``input``
     - ``Input``
     - input script parsing and variables
   * - ``memory``, ``error``
     - ``Memory``, ``Error``
     - allocation and error handling (:doc:`Developer_utils`)
   * - ``timer``
     - ``Timer``
     - the timing breakdown printed at the end of a run
   * - ``universe``
     - ``Universe``
     - the set of processors, and partitioning for multi-replica runs
   * - ``python``
     - ``Python``
     - the PYTHON package interface
   * - ``kokkos``, ``memoryKK``
     - ``KokkosSPARTA``, ``MemoryKokkos``
     - present only in a KOKKOS build

Reading ``src/sparta.h`` is the fastest way to get oriented; each member is
annotated with a one-line comment.

The Pointers base class
-----------------------

Almost every class in SPARTA inherits from ``Pointers``
(``src/pointers.h``). ``Pointers`` exists purely to make the universe
object's members reachable without threading them through every
constructor. Its constructor binds a reference to each pointer in the
``SPARTA`` object:

.. code-block:: c++

   Pointers(SPARTA *ptr) :
     sparta(ptr),
     memory(ptr->memory),
     error(ptr->error),
     particle(ptr->particle),
     update(ptr->update),
     ...

The members are declared as reference-to-pointer (``Particle *&particle``),
which matters: it means a derived class sees the *current* value of
``sparta->particle``, not a copy taken at construction time. Subsystems can
therefore be replaced at run time -- which is exactly what a KOKKOS build
does when it swaps in accelerated classes -- without leaving stale pointers
behind in every object that was constructed earlier.

The practical consequence for anyone writing a new class: inherit from
``Pointers``, pass the ``SPARTA *`` up to its constructor, and you can then
write ``particle->nlocal`` or ``grid->cells`` directly.

``src/pointers.h`` also defines a few macros used everywhere, notably
``FLERR`` -- which expands to ``__FILE__,__LINE__`` and is the first
argument to every ``Error`` call.

Styles and the factory mechanism
--------------------------------

Most user-visible functionality is a *style*: a derived class selected by
name from the input script. ``compute grid``, ``fix ave/grid``,
``surf_collide diffuse`` and ``collide vss`` are all styles.

A style header wraps its class declaration in a macro guarded by a
category-specific define:

.. code-block:: c++

   #ifdef COLLIDE_CLASS

   CollideStyle(vss,CollideVSS)

   #else
   ... class definition ...
   #endif

At build time, ``src/Make.sh style`` -- invoked by the makefile, and by
``cmake/make_style.sh`` for the cmake build -- scans the source directory
for these macros and generates a ``style_*.h`` file per category, each one
just a list of ``#include`` lines. Those generated headers are then
included twice by the code that instantiates styles: once to build a table
of names, once to declare the classes. This is why ``style_*.h`` files do
not exist in a clean checkout, and why adding a new style requires only
dropping two files into ``src`` and rebuilding.

The categories in use are ``Collide``, ``Command``, ``Compute``, ``Dump``,
``Fix``, ``React``, ``Region``, ``SurfCollide`` and ``SurfReact``.

:doc:`Section_modify` describes what each category's derived class must
implement, and is the right starting point for adding one.

Where to go next
----------------

* :doc:`Developer_flow` -- what the code actually does each timestep
* :doc:`Developer_grid`, :doc:`Developer_particle`, :doc:`Developer_surf` --
  the three central data structures
* :doc:`Developer_parallel` -- how the work is divided across processors
