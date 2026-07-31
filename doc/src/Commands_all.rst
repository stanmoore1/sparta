.. _cmd_5:

.. _comm:

Individual commands
===================

This section lists all SPARTA commands alphabetically, with a separate
listing below of styles within certain commands.  The :ref:`previous section <cmd_4>` lists many of the same commands, grouped by category.

+-------------------------------------------+-------------------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------+
| :doc:`adapt\_grid <adapt_grid>`           | :doc:`balance\_grid <balance_grid>` | :doc:`boundary <boundary>`            | :doc:`bound\_modify <bound_modify>`     | :doc:`clear <clear>`                    | :doc:`collide <collide>`                    |
+-------------------------------------------+-------------------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------+
| :doc:`collide\_modify <collide_modify>`   | :doc:`compute <compute>`            | :doc:`create\_box <create_box>`       | :doc:`create\_grid <create_grid>`       | :doc:`create\_isurf <create_isurf>`     | :doc:`create\_particles <create_particles>` |
+-------------------------------------------+-------------------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------+
| :doc:`custom <custom>`                    | :doc:`dimension <dimension>`        | :doc:`dump <dump>`                    | :doc:`dump image <dump_image>`          | :doc:`dump\_modify <dump_modify>`       | :doc:`dump movie <dump_image>`              |
+-------------------------------------------+-------------------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------+
| :doc:`dump vtk <dump_vtk>`                | :doc:`echo <echo>`                  | :doc:`fix <fix>`                      | :doc:`global <global>`                  | :doc:`group <group>`                    | :doc:`if <if>`                              |
+-------------------------------------------+-------------------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------+
| :doc:`include <include>`                  | :doc:`jump <jump>`                  | :doc:`label <label>`                  | :doc:`log <log>`                        | :doc:`mixture <mixture>`                | :doc:`move\_surf <move_surf>`               |
+-------------------------------------------+-------------------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------+
| :doc:`next <next>`                        | :doc:`package <package>`            | :doc:`partition <partition>`          | :doc:`print <print>`                    | :doc:`python <python>`                  | :doc:`quit <quit>`                          |
+-------------------------------------------+-------------------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------+
| :doc:`react <react>`                      | :doc:`react\_modify <react_modify>` | :doc:`read\_grid <read_grid>`         | :doc:`read\_isurf <read_isurf>`         | :doc:`read\_particles <read_particles>` | :doc:`read\_restart <read_restart>`         |
+-------------------------------------------+-------------------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------+
| :doc:`read\_surf <read_surf>`             | :doc:`region <region>`              | :doc:`remove\_surf <remove_surf>`     | :doc:`reset\_timestep <reset_timestep>` | :doc:`restart <restart>`                | :doc:`run <run>`                            |
+-------------------------------------------+-------------------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------+
| :doc:`scale\_particles <scale_particles>` | :doc:`seed <seed>`                  | :doc:`shell <shell>`                  | :doc:`species <species>`                | :doc:`species\_modify <species_modify>` | :doc:`stats <stats>`                        |
+-------------------------------------------+-------------------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------+
| :doc:`stats\_modify <stats_modify>`       | :doc:`stats\_style <stats_style>`   | :doc:`suffix <suffix>`                | :doc:`surf\_collide <surf_collide>`     | :doc:`surf\_react <surf_react>`         | :doc:`surf\_modify <surf_modify>`           |
+-------------------------------------------+-------------------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------+
| :doc:`timestep <timestep>`                | :doc:`uncompute <uncompute>`        | :doc:`undump <undump>`                | :doc:`unfix <unfix>`                    | :doc:`units <units>`                    | :doc:`variable <variable>`                  |
+-------------------------------------------+-------------------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------+
| :doc:`write\_grid <write_grid>`           | :doc:`write\_isurf <write_isurf>`   | :doc:`write\_restart <write_restart>` | :doc:`write\_surf <write_surf>`         |                                         |                                             |
+-------------------------------------------+-------------------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------+


----------


Fix styles
----------

See the :doc:`fix <fix>` command for one-line descriptions of each style
or click on the style itself for a full description.  Some of the
styles have accelerated versions, which can be used if SPARTA is built
with the :doc:`appropriate accelerated package <Section_accelerate>`.
This is indicated by additional letters in parenthesis: k = KOKKOS.

+--------------------------------------+--------------------------------------------+--------------------------------------+------------------------------------+------------------------------------------------------+---------------------------------------------+
| :doc:`ablate <fix_ablate>`           | :doc:`adapt (k) <fix_adapt>`               | :doc:`ambipolar (k) <fix_ambipolar>` | :doc:`ave/grid (k) <fix_ave_grid>` | :doc:`ave/histo (k) <fix_ave_histo>`                 | :doc:`ave/histo/weight (k) <fix_ave_histo>` |
+--------------------------------------+--------------------------------------------+--------------------------------------+------------------------------------+------------------------------------------------------+---------------------------------------------+
| :doc:`ave/surf <fix_ave_surf>`       | :doc:`ave/time <fix_ave_time>`             | :doc:`balance (k) <fix_balance>`     | :doc:`controller <fix_controller>` | :doc:`custom <fix_custom>`                           | :doc:`dt/reset (k) <fix_dt_reset>`          |
+--------------------------------------+--------------------------------------------+--------------------------------------+------------------------------------+------------------------------------------------------+---------------------------------------------+
| :doc:`emit/face (k) <fix_emit_face>` | :doc:`emit/face/file <fix_emit_face_file>` | :doc:`emit/surf <fix_emit_surf>`     | :doc:`field/grid <fix_field_grid>` | :doc:`field/particle <fix_field_particle>`           | :doc:`grid/check (k) <fix_grid_check>`      |
+--------------------------------------+--------------------------------------------+--------------------------------------+------------------------------------+------------------------------------------------------+---------------------------------------------+
| :doc:`halt <fix_halt>`               | :doc:`move/surf (k) <fix_move_surf>`       | :doc:`print <fix_print>`             | :doc:`surf/temp <fix_surf_temp>`   | :doc:`temp/global/rescale <fix_temp_global_rescale>` | :doc:`temp/rescale (k) <fix_temp_rescale>`  |
+--------------------------------------+--------------------------------------------+--------------------------------------+------------------------------------+------------------------------------------------------+---------------------------------------------+
| :doc:`vibmode (k) <fix_vibmode>`     |                                            |                                      |                                    |                                                      |                                             |
+--------------------------------------+--------------------------------------------+--------------------------------------+------------------------------------+------------------------------------------------------+---------------------------------------------+


----------


Compute styles
--------------

See the :doc:`compute <compute>` command for one-line descriptions of
each style or click on the style itself for a full description.  Some
of the styles have accelerated versions, which can be used if SPARTA
is built with the :doc:`appropriate accelerated package <Section_accelerate>`.  This is indicated by additional
letters in parenthesis: k = KOKKOS.

+----------------------------------------------------------+----------------------------------------------------------+------------------------------------------------------+--------------------------------------------------------+----------------------------------------------+------------------------------------------------------------+
| :doc:`boundary (k) <compute_boundary>`                   | :doc:`count (k) <compute_count>`                         | :doc:`distsurf/grid (k) <compute_distsurf_grid>`     | :doc:`dt/grid (k) <compute_dt_grid>`                   | :doc:`eflux/grid (k) <compute_eflux_grid>`   | :doc:`fft/grid (k) <compute_fft_grid>`                     |
+----------------------------------------------------------+----------------------------------------------------------+------------------------------------------------------+--------------------------------------------------------+----------------------------------------------+------------------------------------------------------------+
| :doc:`gas/collision/grid <compute_gas_collision_grid>`   | :doc:`gas/collision/tally <compute_gas_collision_tally>` | :doc:`gas/reaction/grid <compute_gas_reaction_grid>` | :doc:`gas/reaction/tally <compute_gas_reaction_tally>` | :doc:`grid (k) <compute_grid>`               | :doc:`isurf/grid <compute_isurf_grid>`                     |
+----------------------------------------------------------+----------------------------------------------------------+------------------------------------------------------+--------------------------------------------------------+----------------------------------------------+------------------------------------------------------------+
| :doc:`ke/particle (k) <compute_ke_particle>`             | :doc:`lambda/grid (k) <compute_lambda_grid>`             | :doc:`pflux/grid (k) <compute_pflux_grid>`           | :doc:`property/grid (k) <compute_property_grid>`       | :doc:`property/surf <compute_property_surf>` | :doc:`react/boundary <compute_react_boundary>`             |
+----------------------------------------------------------+----------------------------------------------------------+------------------------------------------------------+--------------------------------------------------------+----------------------------------------------+------------------------------------------------------------+
| :doc:`react/surf <compute_react_surf>`                   | :doc:`react/isurf/grid <compute_react_isurf_grid>`       | :doc:`reduce <compute_reduce>`                       | :doc:`sonine/grid (k) <compute_sonine_grid>`           | :doc:`surf (k) <compute_surf>`               | :doc:`surf/collision/tally <compute_surf_collision_tally>` |
+----------------------------------------------------------+----------------------------------------------------------+------------------------------------------------------+--------------------------------------------------------+----------------------------------------------+------------------------------------------------------------+
| :doc:`surf/reaction/tally <compute_surf_reaction_tally>` | :doc:`temp (k) <compute_temp>`                           | :doc:`thermal/grid (k) <compute_thermal_grid>`       | :doc:`tvib/grid (k) <compute_tvib_grid>`               |                                              |                                                            |
+----------------------------------------------------------+----------------------------------------------------------+------------------------------------------------------+--------------------------------------------------------+----------------------------------------------+------------------------------------------------------------+


----------


Collide styles
--------------

See the :doc:`collide <collide>` command for details of each style.
Some of the styles have accelerated versions, which can be used if
SPARTA is built with the :doc:`appropriate accelerated package <Section_accelerate>`.  This is indicated by additional
letters in parenthesis: k = KOKKOS.

+--------------------------+
| :doc:`vss (k) <collide>` |
+--------------------------+


----------


Surface collide styles
----------------------

See the :doc:`surf\_collide <surf_collide>` command for details of each
style.  Some of the styles have accelerated versions, which can be
used if SPARTA is built with the :doc:`appropriate accelerated package <Section_accelerate>`.  This is indicated by additional
letters in parenthesis: k = KOKKOS.

+---------------------------------+---------------------------------------+------------------------------------+
| :doc:`adiabatic <surf_collide>` | :doc:`cll <surf_collide>`             | :doc:`diffuse (k) <surf_collide>`  |
+---------------------------------+---------------------------------------+------------------------------------+
| :doc:`impulsive <surf_collide>` | :doc:`piston (k) <surf_collide>`      | :doc:`specular (k) <surf_collide>` |
+---------------------------------+---------------------------------------+------------------------------------+
| :doc:`td <surf_collide>`        | :doc:`transparent (k) <surf_collide>` | :doc:`vanish (k) <surf_collide>`   |
+---------------------------------+---------------------------------------+------------------------------------+


----------


Surface reaction styles
-----------------------

See the :doc:`surf\_react <surf_react>` command for details of each
style. Some of the styles have accelerated versions, which can be
used if SPARTA is built with the :doc:`appropriate accelerated package <Section_accelerate>`.  This is indicated by additional
letters in parenthesis: k = KOKKOS.

+-----------------------------------+--------------------------------+
| :doc:`adsorb <surf_react_adsorb>` | :doc:`global (k) <surf_react>` |
+-----------------------------------+--------------------------------+
| :doc:`prob (k) <surf_react>`      |                                |
+-----------------------------------+--------------------------------+


----------


Dump styles
-----------

See the :doc:`dump <dump>` command for details of each style.  The
*image* and *movie* styles are documented on the
:doc:`dump image <dump_image>` page, and the VTK styles on the
:doc:`dump vtk <dump_vtk>` page.  Some of the styles have accelerated
versions, which can be used if SPARTA is built with the
:doc:`appropriate accelerated package <Section_accelerate>`.  This is
indicated by additional letters in parenthesis: k = KOKKOS.

.. table_from_list::
   :columns: 4

   * :doc:`grid <dump>`
   * :doc:`grid/vtk <dump_vtk>`
   * :doc:`image <dump_image>`
   * :doc:`movie <dump_image>`
   * :doc:`particle <dump>`
   * :doc:`particle/vtk <dump_vtk>`
   * :doc:`surf <dump>`
   * :doc:`surf/vtk <dump_vtk>`
   * :doc:`tally <dump>`


----------


Reaction styles
---------------

See the :doc:`react <react>` command for details of each style.

.. table_from_list::
   :columns: 4

   * :doc:`qk <react>`
   * :doc:`tce <react>`
   * :doc:`tce/qk <react>`


----------


Region styles
-------------

See the :doc:`region <region>` command for details of each style.

.. table_from_list::
   :columns: 4

   * :doc:`block <region>`
   * :doc:`cylinder <region>`
   * :doc:`intersect <region>`
   * :doc:`plane <region>`
   * :doc:`sphere <region>`
   * :doc:`union <region>`


.. _sws: https://sparta.github.io
.. _sd: Manual.html
.. _sc: Section_commands.html
