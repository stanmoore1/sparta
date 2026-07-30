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

Almost every file in ``src/KOKKOS`` is a Kokkos version of a class in
``src``, following one pattern:

.. code-block:: c++

   class ComputeGridKokkos : public ComputeGrid, public KokkosBase {

The Kokkos class derives from the original, so it inherits the setup,
option parsing and output plumbing, and overrides only the methods that do
per-particle or per-cell work. ``KokkosBase`` (``src/KOKKOS/kokkos_base.h``)
is a small mix-in supplying the interface the rest of the Kokkos code
expects.

The style name gets a ``/kk`` suffix -- ``compute grid/kk``,
``collide vss/kk`` -- registered through the same ``Style()`` macro
mechanism described in :doc:`Developer_org`. The
:doc:`suffix <suffix>` command and the ``-sf kk`` command-line switch make
SPARTA substitute the ``/kk`` variant automatically wherever one exists,
which is why input scripts do not normally mention them.

This is also why the accelerated styles are not listed separately in the
command index: they are marked ``(k)`` on their base style's entry.

Data layout and DualView
------------------------

``src/KOKKOS/kokkos_type.h`` defines the type aliases the package is
written in terms of -- the execution space, the memory space, and the array
layout, all derived from Kokkos' defaults so a single source tree compiles
for Serial, OpenMP or CUDA backends.

The central abstraction is Kokkos ``DualView``: a pair of mirrored
allocations, one on the host and one on the device, with explicit
``sync()`` and ``modified()`` calls to move data between them. Every
Kokkos-enabled class implements those two methods:

.. code-block:: c++

   void sync(ExecutionSpace, unsigned int);
   void modified(ExecutionSpace, unsigned int);

The bitmask argument selects which arrays are involved, so a class that
touched only particle velocities does not force a transfer of positions.
Getting these calls wrong is the classic KOKKOS bug: the code runs and
produces plausible but stale results, because a device kernel read a host
array that had been modified without being synced.

Memory is allocated through ``MemoryKokkos`` rather than ``Memory``, which
is why the ``SPARTA`` universe object carries both (:doc:`Developer_org`).

``copymode``
------------

Kokkos functors are copied by value into kernels. A copied class must not
free the memory the original owns when the copy is destroyed, so SPARTA's
classes carry a ``copymode`` flag; destructors return early when it is set.
The same idiom appears in the non-Kokkos ``Comm`` class. Any new
Kokkos-enabled class needs it.

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

Each thread drawing from its own stream means numbers are consumed in a
different order than the serial code consumes them, so trajectories diverge
immediately -- statistically equivalent, but not comparable run to run.

Building with ``-DSPARTA_KOKKOS_EXACT`` swaps in ``RandPoolWrap``, a
wrapper around SPARTA's ordinary serial generator. It is slower and does
not thread well, but it makes a Kokkos build reproduce the non-Kokkos
results *exactly*.

That is what makes the KOKKOS package testable against the existing
regression logs, and it is exactly what the ``mpi-kokkos-exact`` CI job
does: it builds the Serial backend with ``SPARTA_KOKKOS_EXACT=ON`` and runs
the standard regression suite with ``-k on -sf kk``, comparing against the
same gold-standard logs the non-Kokkos build is checked against. See
:doc:`Developer_testing`.

Adding a Kokkos version of a style
----------------------------------

The workflow is: copy the nearest existing pair in ``src/KOKKOS``, derive
from your base class and ``KokkosBase``, move the per-particle or per-cell
loop into a Kokkos functor, replace raw arrays with the ``DualView``
aliases from ``kokkos_type.h``, implement ``sync()``/``modified()`` for the
arrays you touch, and set ``copymode``. Then verify with a
``SPARTA_KOKKOS_EXACT`` build against the non-Kokkos logs before doing
anything about performance.

Where to go next
----------------

* :doc:`Section_accelerate` -- building and running with KOKKOS
* :doc:`Developer_testing` -- how the exact-match testing works
* :doc:`Developer_parallel` -- the MPI layer underneath
