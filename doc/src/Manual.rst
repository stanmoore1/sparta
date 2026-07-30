SPARTA Documentation
====================

24 Sep 2025 version
-------------------

SPARTA stands for Stochastic PArallel Rarefied-gas Time-accurate Analyzer.

SPARTA is a Direct Simulation Monte Carlo (DSMC) simulator designed to run
efficiently on parallel computers.  It was developed at Sandia National
Laboratories, a US Department of Energy facility, with funding from the DOE.
It is an open-source code, distributed freely under the terms of the GNU
Public License (GPL), or sometimes by request under the terms of the GNU
Lesser General Public License (LGPL).

The primary developers of SPARTA are `Steve Plimpton
<https://sjplimp.github.io>`_ and Michael Gallis, who can be contacted at
sjplimp at gmail.com and magalli at sandia.gov.  The `SPARTA WWW Site
<https://sparta.github.io>`_ has more information about the code and its uses.

Version info
------------

The SPARTA "version" is the date when it was released, such as 3 Mar 2014.
SPARTA is updated continuously.  Whenever we fix a bug or add a feature, we
release it immediately, and post a notice on `this page of the WWW site
<https://sparta.github.io/bug.html>`_.  Each dated copy of SPARTA contains all
the features and bug-fixes up to and including that version date.  The version
date is printed to the screen and logfile every time you run SPARTA.  It is
also in the file src/version.h, in the SPARTA directory name created when you
unpack a tarball, and at the top of this page.

* If you browse the HTML doc pages on the SPARTA WWW site, they always
  describe the most current version of SPARTA.
* If you browse the HTML doc pages included in your tarball, they describe
  the version you have.

If you find errors or omissions in this manual, or have suggestions for useful
information to add, please send an email to the developers so we can improve
the SPARTA documentation.

Once you are familiar with SPARTA, you may want to bookmark
:doc:`Section_commands`, which gives quick access to the documentation for
every SPARTA command.

.. toctree::
   :maxdepth: 2
   :numbered: 3
   :caption: User Guide
   :name: userdoc
   :includehidden:

   Section_intro
   Section_start
   Section_commands
   Section_packages
   Section_accelerate
   Section_howto
   Section_example
   Section_perf
   Section_tools
   Section_modify
   Section_python
   Section_errors
   Section_history

.. toctree::
   :maxdepth: 1
   :caption: Command Reference
   :name: reference

   adapt_grid
   balance_grid
   bound_modify
   boundary
   clear
   collide
   collide_modify
   compute
   compute_boundary
   compute_count
   compute_distsurf_grid
   compute_dt_grid
   compute_eflux_grid
   compute_fft_grid
   compute_gas_collision_grid
   compute_gas_collision_tally
   compute_gas_reaction_grid
   compute_gas_reaction_tally
   compute_grid
   compute_isurf_grid
   compute_ke_particle
   compute_lambda_grid
   compute_pflux_grid
   compute_property_grid
   compute_property_surf
   compute_react_boundary
   compute_react_isurf_grid
   compute_react_surf
   compute_reduce
   compute_sonine_grid
   compute_surf
   compute_surf_collision_tally
   compute_surf_reaction_tally
   compute_temp
   compute_thermal_grid
   compute_tvib_grid
   create_box
   create_grid
   create_isurf
   create_particles
   custom
   dimension
   dump
   dump_image
   dump_modify
   dump_vtk
   echo
   fix
   fix_ablate
   fix_adapt
   fix_ambipolar
   fix_ave_grid
   fix_ave_histo
   fix_ave_surf
   fix_ave_time
   fix_balance
   fix_controller
   fix_custom
   fix_dt_reset
   fix_emit_face
   fix_emit_face_file
   fix_emit_surf
   fix_field_grid
   fix_field_particle
   fix_grid_check
   fix_halt
   fix_move_surf
   fix_print
   fix_surf_temp
   fix_temp_global_rescale
   fix_temp_rescale
   fix_vibmode
   global
   group
   if
   include
   jump
   label
   log
   mixture
   move_surf
   next
   package
   partition
   print
   python
   quit
   react
   react_modify
   read_grid
   read_isurf
   read_particles
   read_restart
   read_surf
   region
   remove_surf
   reset_timestep
   restart
   run
   scale_particles
   seed
   shell
   species
   species_modify
   stats
   stats_modify
   stats_style
   suffix
   surf_collide
   surf_modify
   surf_react
   surf_react_adsorb
   timestep
   uncompute
   undump
   unfix
   units
   variable
   write_grid
   write_isurf
   write_restart
   write_surf

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
