.. _mod_11:

Coding style
============

SPARTA's pull request template asks that "the source code follows the
SPARTA formatting guidelines". This page is those guidelines. They are
not enforced automatically; they are what the existing code
does, and the figures below are counts taken from ``src``.

There is no reformatting campaign planned and none is wanted. Match the
file you are editing.

Layout
------

**Two spaces per indent level, spaces only.** Of the 316 files in ``src``,
21 contain a tab and in every case it is aligning a continuation line, not
indenting a block. Do not add more.

**Braces.** A function or method definition puts its opening brace on a
line of its own; a control statement puts it at the end of the line:

.. code-block:: c++

   Collide::Collide(SPARTA *sparta, int, char **arg) : Pointers(sparta)
   {
     if (rotstyle == DISCRETE) {
       Particle::Species *species = particle->species;
       ...
     }
   }

A single-statement body may drop the braces, and commonly shares the line:

.. code-block:: c++

   if (imix < 0) error->all(FLERR,"Collision mixture does not exist");

**No space after a comma** in an argument list. This is near-universal in
the source (2499 occurrences of ``error->all(FLERR,`` against 10 with a
space). Do put spaces around binary operators.

**Methods are separated by a rule**, either plain:

.. code-block:: c++

   /* ---------------------------------------------------------------------- */

or carrying a short description of what follows:

.. code-block:: c++

   /* ----------------------------------------------------------------------
      NTC algorithm
   ------------------------------------------------------------------------- */

Keep lines to about 80 columns.

Naming
------

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Thing
     - Convention
     - Example
   * - class
     - ``CamelCase``
     - ``ComputeGrid``
   * - file
     - ``snake_case`` matching the class
     - ``compute_grid.cpp`` / ``.h``
   * - method
     - lowercase with underscores
     - ``compute_per_grid()``
   * - member variable
     - lowercase, short, no prefix or suffix
     - ``nglocal``, ``groupbit``
   * - boolean member
     - ends in ``flag``
     - ``tvib_flag``, ``kokkos_flag``
   * - include guard
     - ``SPARTA_<FILE>_H``
     - ``SPARTA_COMPUTE_GRID_H``
   * - file-local constant
     - ``#define``, all caps, after the includes
     - ``#define DELTAGRID 1000``

Everything lives in ``namespace SPARTA_NS``, and ``.cpp`` files open with
``using namespace SPARTA_NS;``. Access specifiers such as ``public:`` are
indented one space. Enumerations are anonymous, all caps, and on one line,
with a trailing comment naming any other file that has to be kept in step:

.. code-block:: c++

   enum{NONE,DISCRETE,SMOOTH};       // several files  (NOTE: change order)

Language
--------

The build requires C++11 and does not assume more, except in KOKKOS
builds. Two conventions predate that and are still what the code uses:

* ``NULL``, not ``nullptr`` (1405 occurrences against 44).
* C library headers included in the C form and in quotes --
  ``#include "math.h"``, ``#include "string.h"`` -- rather than
  ``<cmath>`` or ``<cstring>``.

Use the utility classes rather than rolling your own: ``Memory`` for
allocation, ``Error`` for anything fatal or noisy, and ``Input``'s
``numeric()`` / ``inumeric()`` / ``bounds()`` helpers for parsing
arguments. :doc:`Developer_utils` describes them and explains why the
``all()`` versus ``one()`` distinction in ``Error`` is a correctness
matter rather than a stylistic one.

Integer types matter at scale. A per-processor count is ``int``; a
quantity summed over all processors or a timestep count is ``bigint``; a
grid cell ID is ``cellint``; a surface element ID is ``surfint``. Getting
this wrong produces overflow that only shows up on large runs.

File header
-----------

Every new file in ``src`` begins with this block, verbatim:

.. code-block:: c++

   /* ----------------------------------------------------------------------
      SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
      http://sparta.github.io
      Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
      Sandia National Laboratories

      Copyright (2014) Sandia Corporation.  Under the terms of Contract
      DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
      certain rights in this software.  This software is distributed under
      the GNU General Public License.

      See the README file in the top-level SPARTA directory.
   ------------------------------------------------------------------------- */

If you contributed the file, add a line naming yourself and your
institution below the block, as several files in ``src`` already do.

Two design guidelines
---------------------

These are from :doc:`Section_modify` and are worth repeating, because
they decide more reviews than formatting does:

* Consider whether what you want would be better as a pre- or
  post-processing step. Not everything belongs inside the timestep.
* Do not do anything within the timestepping of a run that is not
  parallel. A step that only the root processor can perform, or that
  requires gathering a global quantity onto one rank, will limit every
  simulation that uses your style.
