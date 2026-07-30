Accelerating SPARTA performance
===============================

This section describes various methods for improving SPARTA
performance for different classes of problems running on different
kinds of machines.

Currently the only option is to use the KOKKOS accelerator
packages provided with SPARTA that
contains code optimized for certain kinds of hardware, including
multi-core CPUs and GPUs.

* 5.1 :ref:`Measuring performance <acc_1>`
* 5.2 :ref:`Accelerator packages with optimized styles <acc_2>`
* 5.3 :ref:`KOKKOS package <acc_3>`

The `Benchmark page <https://sparta.github.io/bench.html>`_ of the SPARTA
web site gives performance results for the various accelerator
packages discussed in Section 5.2, for several of the standard SPARTA
benchmark problems, as a function of problem size and number of
compute nodes, on different hardware platforms.


.. toctree::
   :maxdepth: 1

   Speed_measure
   Speed_packages
   Speed_kokkos
