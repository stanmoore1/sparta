.. _intro_1:

What is SPARTA
==============

SPARTA is a Direct Simulation Montel Carlo code that models rarefied
gases, using collision, chemistry, and boundary condition models.  It
uses a hierarchical Cartesian grid to track and group particles for 3d
or 2d or axisymmetric models.  Objects emedded in the gas are
represented as triangulated surfaces and cut through grid cells.

For examples of SPARTA simulations, see the `SPARTA WWW Site <sws_>`_.

SPARTA runs efficiently on single-processor desktop or laptop
machines, but is designed for parallel computers.  It will run on any
parallel machine that compiles C++ and supports the `MPI <mpi_>`_
message-passing library.  This includes distributed- or shared-memory
parallel machines as well as commodity clusters.

.. _mpi: http://www-unix.mcs.anl.gov/mpi



SPARTA can model systems with only a few particles up to millions or
billions.  See :doc:`Section 8 <Section_perf>` for information on SPARTA
performance and scalability, or the Benchmarks section of the `SPARTA WWW Site <sws_>`_.

SPARTA is a freely-available open-source code, distributed under the
terms of the `GNU Public License <gnu_>`_, or sometimes by request under
the terms of the `GNU Lesser General Public License (LGPL) <gnu2>`_,
which means you can use or modify the code however you wish.  The only
restrictions imposed by the GPL or LGPL are on how you distribute the
code further.  See :ref:`Section 1.4 <intro_4>` below for a brief discussion
of the open-source philosophy.

.. _gnu: http://www.gnu.org/copyleft/gpl.html



SPARTA is designed to be easy to modify or extend with new
capabilities, such as new collision or chemistry models, boundary
conditions, or diagnostics.  See :doc:`Section 10 <Section_modify>` for
more details.

SPARTA is written in C++ which is used at a hi-level to structure the
code and its options in an object-oriented fashion.  The kernel
computations use simple data structures and C-like code for effciency.
So SPARTA is really written in an object-oriented C style.

SPARTA was developed with internal funding at `Sandia National Laboratories <snl_>`_, a US Department of Energy lab.  See :ref:`Section 1.5 <intro_5>` below for more information on SPARTA funding and
individuals who have contributed to SPARTA.

.. _snl: http://www.sandia.gov

.. _sws: https://sparta.github.io
