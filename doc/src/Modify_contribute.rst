.. _mod_9:

Submitting a contribution
=========================

SPARTA is developed in the open at
`github.com/sparta/sparta <https://github.com/sparta/sparta>`_.
Contributions arrive as pull requests against the ``master`` branch, and
that is the preferred route for anything larger than a typo.

The pages before this one describe how to *write* a new style. This page
describes how to get it accepted; :doc:`Modify_requirements` lists what a
contribution has to include, and :doc:`Modify_style` covers the formatting
conventions the source follows.

The short version
-----------------

#. Fork the repository and branch from ``master``.
#. Make the change, following the conventions in
   :doc:`Modify_style`.
#. Add a doc page and at least one example input script
   (:doc:`Modify_requirements`).
#. Check that the tests pass and the manual still builds:

   .. code-block:: bash

      cmake -C ../cmake/presets/mpi.cmake -DSPARTA_ENABLE_TESTING=ON ../cmake
      make -j4
      ctest --output-on-failure -j4
      make -C doc html SPHINXEXTRA="-W --keep-going"
      make -C doc check

#. Open a pull request and fill in the template.

What the pull request should say
--------------------------------

The repository's pull request template asks for four things, and the
review goes faster when they are actually filled in:

*Purpose*
   What the change does and why. If it closes an open issue, write
   ``closes #135`` so GitHub links them.

*Author(s)*
   Name and affiliation of everyone who should be credited. SPARTA has no
   contributor license agreement and asks for no copyright assignment, so
   this is the only record of who wrote the code.

*Backward compatibility*
   Whether any existing input script, restart file or output format
   behaves differently after the change. If reference log files had to be
   re-blessed, say so and say why -- see :doc:`Developer_testing`.

*Implementation notes*
   Anything a reviewer would otherwise have to reverse-engineer: a
   non-obvious algorithm, a deliberate limitation, a dependency added.

Licensing
---------

SPARTA is distributed under the terms of the GNU General Public License,
version 2 (:doc:`Intro_opensource`). By submitting a contribution you are
offering it under those terms. There is nothing to sign: no contributor
license agreement, no developer certificate of origin, no ``Signed-off-by``
line.

Every new source file must carry the standard header block, reproduced in
:doc:`Modify_style`. If you are adapting code from another project, keep
its copyright notice as well and say where it came from -- the regression
driver in ``tools/testing/regression.py`` keeps its LAMMPS header for
exactly this reason.

What happens next
-----------------

Opening a pull request runs the CI described in :doc:`Modify_requirements`.
A green build is a precondition for review, not a substitute for it: a
maintainer still reads the code, and files under ``src/KOKKOS`` and
``lib/kokkos`` automatically request a review from their owner via
``.github/CODEOWNERS``.

Smaller contributions
---------------------

Not everything needs a pull request. Reporting a bug you cannot fix is
useful on its own -- :doc:`Errors_bugs` explains how to make the report
actionable. So is telling us about a paper that used SPARTA, an error in
the manual, or a simulation that produced a picture worth putting on the
web site (:doc:`Intro_citing`).

If you would rather not use GitHub at all, you can send a patch or a pair
of files by email to the
`developers <https://sparta.github.io/authors.html>`_. That is how SPARTA
accepted contributions before it moved to GitHub and it still works,
though a pull request gets the automated checks run for you.
