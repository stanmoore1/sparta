.. _intro_2:

SPARTA features
===============

This section highlights SPARTA features, with links to specific
commands which give more details.  The :ref:`next section <intro_3>`
illustrates the kinds of grid geometries and surface definitions which
SPARTA supports.

If SPARTA doesn't have your favorite collision model, boundary
condition, or diagnostic, see :doc:`Section 10 <Section_modify>` of the
manual, which describes how it can be added to SPARTA.

General features
----------------

* runs on a single processor or in parallel
* distributed-memory message-passing parallelism (MPI)
* spatial-decomposition of simulation domain for parallelism
* open-source distribution
* highly portable C++
* optional libraries used: MPI
* :doc:`easy to extend <Section_modify>` with new features and functionality
* runs from an :doc:`input script <Section_commands>`
* syntax for defining and using :doc:`variables and formulas <variable>`
* syntax for :doc:`looping over runs <jump>` and breaking out of loops
* run one or :ref:`multiple simulations simultaneously <howto_3>` (in parallel) from one script
* :ref:`build as library <start_4>`, invoke SPARTA thru :ref:`library interface <howto_6>` or provided :doc:`Python wrapper <Section_python>`
* :ref:`couple with other codes <howto_7>`: SPARTA calls other code, other code calls SPARTA, umbrella code calls both

Models
------

* :doc:`3d or 2d <dimension>` or :ref:`2d-axisymmetric <howto_2>` domains
* variety of :doc:`global boundary conditions <boundary>`
* :doc:`create particles <create_particles>` within flow volume
* emit particles from simulation box faces due to :doc:`flow properties <fix_emit_face>`
* emit particles from simulation box faces due to :doc:`profile defined in file <fix_emit_face_file>`
* emit particles from surface elements due to :doc:`normal and flow properties <fix_emit_surf>`
* :ref:`ambipolar <howto_11>` approximation for ionized plasmas

Geometry
--------

* :ref:`Cartesian, heirarchical grids <intro_3>` with multiple levels of local refinement
* :doc:`create grid from input script <create_grid>` or :doc:`read from file <read_grid>`
* embed :triangulated (3d) or line-segmented (2d) surfaces"_#intro\_3 in grid, :doc:`read in from file <read_surf>`

Gas-phase collisions and chemistry
----------------------------------

* collisions between all particles or pairs of species groups within grid cells
* :doc:`collision models: <collide>` VSS (variable soft sphere), VHS (variable hard sphere), HS (hard sphere)
* :doc:`chemistry models: <react>` TCE, QK

Surface collisions and chemistry
--------------------------------

* for surface elements or global simulation box :doc:`boundaries <bound_modify>`
* :doc:`collisions: <surf_collide>` specular or diffuse
* :doc:`reactions <surf_react>`

Performance
-----------

* :doc:`grid cell weighting <global>` of particles
* :doc:`adaptation <adapt_grid>` of the grid cells between runs
* :doc:`on-the-fly adaptation <fix_adapt>` of the grid cells
* :doc:`static <balance_grid>` load-balancing of grid cells or particles
* :doc:`dynamic <fix_balance>` load-balancing of grid cells or particles

Diagnostics
-----------

* :doc:`global boundary statistics <compute_boundary>`
* :doc:`per grid cell statistics <compute_grid>`
* :doc:`per surface element statistics <compute_surf>`
* time-averaging of :doc:`global <fix_ave_time>`, :doc:`grid <fix_ave_grid>`, :doc:`surface <fix_ave_surf>` statistics

Output
------

* :doc:`log file of statistical info <stats_style>`
* :doc:`dump files <dump>` (text or binary) of per particle, per grid cell, per surface element values
* binary :doc:`restart files <restart>`
* on-the-fly :doc:`rendered images and movies <dump_image>` of particles, grid cells, surface elements

Pre- and post-processing
------------------------

* Various pre- and post-processing serial tools are packaged with
  SPARTA; see :doc:`Section 9 <Section_tools>` of the manual.
* Our group has also written and released a separate toolkit called
  `Pizza.py <pizza_>`_ which provides tools for doing setup, analysis,
  plotting, and visualization for SPARTA simulations.  Pizza.py is
  written in `Python <python_>`_ and is available for download from `the Pizza.py WWW site <pizza_>`_.

.. _pizza: https://lammps.github.io/pizza



.. _python: http://www.python.org
