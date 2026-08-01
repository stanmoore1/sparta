.. _hist_2:

Past versions
=============

Sandia's predecessor to SPARTA is a DSMC code called ICARUS.  It was
developed in the early 1990s by Tim Bartel and `Steve Plimpton <https://sjplimp.github.io>`_.  It was later modified and
extended by Michael Gallis.

ICARUS is a 2d code, written in Fortran, which models the flow
geometry around bodies with a collection of adjoining body-fitted grid
blocks.  The geometry of the grid cells within in a single block is
represented with analytic equations, which allows for fast particle
tracking.

Some details about ICARUS, including simulation snapshots and papers,
are discussed on `this page <https://sjplimp.github.io/dsmc.html>`_

Performance-wise ICARUS scaled quite well on several generations of
parallel machines, and is still used by Sandia researchers today.
ICARUS was export-controlled software, and so was not distributed
widely outside of Sandia.

SPARTA development began in late 2011.  In contrast to ICARUS, it is a
3d code, written in C++, and uses a hierarchical Cartesian grid to
track particles.  Surfaces are embedded in the grid, which cuts and
splits their flow volumes.

The `Authors link <https://sparta.github.io/authors.html>`_ on the SPARTA
web page gives a timeline of features added to the code since it's
initial open-source release.


.. _sws: https://sparta.github.io
.. _sd: Manual.html
.. _sc: Section_commands.html
