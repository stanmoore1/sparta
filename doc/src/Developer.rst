Developer guide
===============

These pages describe how SPARTA is put together internally: how the source
tree is organized, what happens during a timestep, how the grid, particles
and surfaces are represented, and how work is divided across processors.

They are aimed at someone who wants to modify SPARTA rather than only run
it. If your goal is to add a new compute, fix, collision model or command
without needing to understand the rest of the code, start with
:doc:`Section_modify` instead -- SPARTA's style mechanism is designed so
that a new feature can be written as two self-contained files.

The pages below describe the code as it is, not as a specification. SPARTA
is an active code and the details change; when this text and the source
disagree, the source is right. File and class names are given throughout so
you can go and read the implementation.

.. toctree::
   :maxdepth: 1

   Developer_org
   Developer_flow
   Developer_grid
   Developer_particle
   Developer_surf
   Developer_parallel
   Developer_kokkos
   Developer_utils
   Developer_testing
