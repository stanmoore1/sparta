The KOKKOS package
==================

The KOKKOS package provides versions of SPARTA's hottest styles that run on
GPUs or on multiple threads per MPI rank. It lives in ``src/KOKKOS`` and is
built by enabling ``PKG_KOKKOS``. :doc:`Section_accelerate` covers how to
build and run with it; this page covers how it is put together.

Kokkos parallelism is *on-node* and orthogonal to the MPI decomposition
described in :doc:`Developer_parallel`. A typical run still distributes
grid cells across MPI ranks; within a rank, Kokkos parallelizes the loops
over cells and particles.

The derived-class pattern
-------------------------

Most files in ``src/KOKKOS`` are Kokkos versions of a class in ``src``, and
each derives from the original:

.. code-block:: c++

   class CollideVSSKokkos : public CollideVSS {
   class ComputeGridKokkos : public ComputeGrid, public KokkosBase {

The Kokkos class inherits the setup, option parsing and output plumbing,
and overrides only the methods that do per-particle or per-cell work.
Computes, regions and fixes whose output other Kokkos styles read on the
device also derive from ``KokkosBase`` (``src/KOKKOS/kokkos_base.h``), a
small mix-in supplying device-side per-grid and per-particle output and the
region interface.

The style name gets a ``/kk`` suffix -- ``compute grid/kk``,
``collide vss/kk`` -- registered through the same ``Style()`` macro
mechanism described in :doc:`Developer_org`. The ``-sf kk``
`command-line switch <Section_start.html#start_7>`_ makes SPARTA substitute
the ``/kk`` variant automatically wherever one exists, which is why input
scripts do not normally mention them.

This is also why the accelerated styles are not listed separately in the
command index: they are marked ``(k)`` on their base style's entry.

Data layout and DualView
------------------------

``src/KOKKOS/kokkos_type.h`` defines the type aliases the package is
written in terms of -- the execution space, the memory space, and the array
layout, all derived from Kokkos' defaults so a single source tree compiles
for Serial, OpenMP or CUDA backends.

The central abstraction is Kokkos ``DualView``: a pair of mirrored
allocations, one on the host and one on the device, kept consistent with
explicit ``sync_device()``/``sync_host()`` and
``modify_device()``/``modify_host()`` calls.  The classes that own the
shared data -- ``ParticleKokkos``, ``GridKokkos`` and ``SurfKokkos`` --
wrap these in two methods that other classes call for the data they use:

.. code-block:: c++

   void sync(ExecutionSpace, unsigned int);
   void modify(ExecutionSpace, unsigned int);

(``CollideVSSKokkos`` has the same pair, named ``sync`` and ``modified``.)
The bitmask argument, built from the masks in
``src/KOKKOS/sparta_masks.h``, selects which DualViews are involved --
``PARTICLE_MASK``, ``SPECIES_MASK``, the grid and surface masks -- so a
kernel that touched only the particle array does not force a transfer of
species or grid data.  The mask is per array, not per field: all of a
particle's fields live in one array and move together.
Getting these calls wrong is the classic KOKKOS bug: the code runs and
produces plausible but stale results, because a device kernel read a host
array that had been modified without being synced.

In KOKKOS mode the ``memory`` object is replaced by a ``MemoryKokkos``, a
subclass of ``Memory`` that adds ``create_kokkos()``, ``grow_kokkos()`` and
``destroy_kokkos()`` for Kokkos views; the ``SPARTA`` universe object also
keeps it as ``memoryKK`` (:doc:`Developer_org`).

``copymode``
------------

Kokkos functors are copied by value into kernels. A copied class must not
free the memory the original owns when the copy is destroyed, so SPARTA's
base classes (``Compute``, ``Fix``, ``Grid``, ``Particle``, ``Collide``,
``Comm``, ...) declare a ``copymode`` flag.  A Kokkos class sets it to 1
while launching kernels, back to 0 afterwards, and returns early from its
destructor when it is set.

Reproducing non-Kokkos results
------------------------------

Random number generation is where a threaded implementation necessarily
diverges from a serial one. The Kokkos build normally uses a parallel
random pool:

.. code-block:: c++

   #ifndef SPARTA_KOKKOS_EXACT
     Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
     typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;
   #else
     RandPoolWrap rand_pool;
     typedef RandWrap rand_type;
   #endif

The XorShift64 pool is a different generator from the ``RanKnuth`` the
non-Kokkos code uses, and each thread draws from its own stream, so
trajectories diverge from a non-Kokkos run immediately -- statistically
equivalent, but not comparable step by step.

Building with ``SPARTA_KOKKOS_EXACT=ON`` (CMake; it defines
``SPARTA_KOKKOS_EXACT``) swaps in ``RandPoolWrap``, whose thread 0 uses
SPARTA's ordinary serial ``RanKnuth`` generator, and also makes reductions
and particle emission run in the host's order.  With a single thread -- the
Serial backend -- a Kokkos build then reproduces the non-Kokkos results
*exactly*.  It is slower, gives no such guarantee with more threads, and
cannot be used on GPUs.

That is what makes the KOKKOS package testable against the existing
regression logs, and it is exactly what the ``mpi-kokkos-exact`` CI job
does: it builds the Serial backend with ``SPARTA_KOKKOS_EXACT=ON`` and runs
the standard regression suite with ``-k on -sf kk``, comparing against the
same gold-standard logs the non-Kokkos build is checked against. See
:doc:`Developer_testing`.

Adding a Kokkos version of a style
----------------------------------

The workflow is: copy the nearest existing pair in ``src/KOKKOS``, derive
from your base class (and from ``KokkosBase`` if other Kokkos styles read
your output on the device), move the per-particle or per-cell loop into a
Kokkos functor, replace raw arrays with the ``DualView`` aliases from
``kokkos_type.h``, call the owning classes' ``sync()``/``modify()`` for the
data you read and write, and set ``copymode`` around kernel launches. Then verify with a
``SPARTA_KOKKOS_EXACT`` build against the non-Kokkos logs before doing
anything about performance.

Where to go next
----------------

* :doc:`Section_accelerate` -- building and running with KOKKOS
* :doc:`Developer_testing` -- how the exact-match testing works
* :doc:`Developer_parallel` -- the MPI layer underneath
