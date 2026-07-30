.. _howto_17:

Custom per-particle, per-grid, per-surf attributes
==================================================

Particles, grid cells, and surface elements can have custom attributes
which store either single or multiple values per particle, per grid
cell, or per surface element.  If a single value is stored, the
attribute is referred to as a custom per-particle, per-grid, or
per-surf vector.  If multiple values are stored, the attribute is
referred to as a custom per-particle, per-grid, or per-surf array (an
array can have a single column and thus a single value per entity).
Each custom attribute has a name, which allows them to be specified in
input scripts as arguments to various commands.  The values each
attricute stores can be either integer or floating point numbers.

The :doc:`custom <custom>` command can create and set/reset custom
attribute values for all 3 flavors of attributes.  Either by invoking
per-particle, per-grid, or per-surf variables, or by reading a file
with one line of attribute values per particle/grid/surf.  In the case
of per-grid attributes, it can also read a coarse file with values for
coarse grid points.  The attributed values for each grid cell are set
to values of the nearset coarse grid point.  The :doc:`fix custom <fix_custom>` command can do the same thing periodically as
a simulation runs.  :doc:`Dump <dump>` commands can output all 3 flavors
of attributes.

Here are lists of current commands which use custom attributes in
various ways:

**Per-particle custom attributes:**

* :doc:`compute reduce <compute_reduce>` - reduce a per-particle attribute to a scalar value
* :doc:`custom <custom>` - create or set ore remove the values of a per-particle attribute
* :doc:`fix custom <fix_custom>` - reset the values of a per-particle attribute during a simulation
* :doc:`dump particle <dump>` - output per-particle attributes to a dump file
* :doc:`fix ambipolar <fix_ambipolar>` - use a per-particle vector and array for ambipolar quantities
* :doc:`variable <variable>` - use a per-particle attribute in a particle-style variable formula

**Per-grid custom attributes:**

* :doc:`compute reduce <compute_reduce>` - reduce a per-grid attribute to a scalar value
* :doc:`create\_particles <create_particles>` - create particles based on per-grid attributes
* :doc:`custom <custom>` - create or set or remove the values of a per-grid attribute
* :doc:`fix custom <fix_custom>` - reset the values of a per-grid attribute during a simulation
* :doc:`dump grid <dump>` - output per-grid attributes to a dump file
* :doc:`fix ave/grid <fix_ave_grid>` - time-average a per-grid attribute
* :doc:`read\_grid <read_grid>` - define and initialize per-grid attributes
* surf\_react implicit - use per-grid vectors and an array to store chemical state (not yet released in public SPARTA)
* :doc:`variable <variable>` - use a per-grid attribute in a grid-style variable formula
* :doc:`write\_grid <write_grid>` - write per-grid attributes to a grid data file

**Per-surf custom attributes:**

* :doc:`compute reduce <compute_reduce>` - reduce a per-surf attribute to a scalar value
* :doc:`custom <custom>` - create or set or remove the values of a per-surf attribute
* :doc:`fix custom <fix_custom>` - reset the values of a per-surf attribute during a simulation
* :doc:`dump surf <dump>` - output per-surf attributes to a dump file
* :doc:`fix ave/surf <fix_ave_surf>` - time-average a per-surf attribute
* :doc:`fix emit/surf <fix_emit_surf>` - use per-surf attributes to vary particle emission from each surf
* :doc:`fix surf/temp <fix_surf_temp>` - set the temperature of each surf based on gas collisions
* :doc:`read\_surf <read_surf>` - define and initialize per-surf attributes
* :doc:`surf\_collide <surf_collide>` - use a per-surf attribute as temperature for particle/surf collisions
* :doc:`surf\_react adsorb <surf_react_adsorb>` - use per-surf vectors and an array to store chemical state
* :doc:`variable <variable>` - use a per-surf attribute in a surf-style variable formula
* :doc:`write\_surf <write_surf>` - write per-surf attributes to a surf data file

Per-surf custom attributes can be defined for explicit or
explicit/distributed surface elements, as set by the :doc:`global surfs <global>` comand.  But they cannot be used for implicit
surface elements.  Conceptually, implicit surfaces are defined on a
per-grid cell basis, so per-grid custom attributes can be used instead
to define attributes of implicit surfaces.

Note that in some cases the name for a custom attribute is specified
by the user, e.g. for the :doc:`read\_grid <read_grid>` or
:doc:`read\_surf <read_surf>` commands.  In other cases, a command
defines the name for the attributes and documents the name(s) it uses,
e.g. for the :doc:`fix ambipolar <fix_ambipolar>` or :doc:`surf\_react adsorb <surf_react_adsorb>` commands.

Also note that custom attributes can be static or dynamic quantities.
For example, the :doc:`read\_surf <read_surf>` command can be used to
define a *static* temperature for each surface element it reads,
stored as a custom per-surf vector.  By contrast, the :doc:`fix surf/temp <fix_surf_temp>` command can be used to define a
*dynamic* temperature for each surface element which is calculated
once every N steps from the energy flux which colliding particles
impart to each surface element, also stored in a custom per-surf
vector.

In both cases, the custom per-surf temperature can be used by the
:doc:`surf\_collide diffuse <surf_collide>` command to use the current
surface temperature for performing particle/surface element
collisions.  Likewise the :doc:`fix emit/surf <fix_emit_surf>` command
can use the current custom per-surf temperature to alter the emission
properties of each surface elemnt.

Another use of dynamic custom attributes is by the :doc:`fix ambipolar <fix_ambipolar>` and :doc:`surf\_react adsorb <surf_react>`
commands.  The former stores the ambipolar state of each particle in
per-particle attributes.  The latter stores the chemical state of each
surface element in per-surf attributes.  These will vary over the
course of a simulation, and their status can be monitored with the
various output commands listed above.
