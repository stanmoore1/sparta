.. _howto_19:

Details of particles in SPARTA
==============================

Individual simulation particles in SPARTA are conceptually a
collection of physical gas particles of the same molecular species.
If the species is atomic, then one gas particle is an atom.  If the
species is molecular then one gas particle is a molecule.  The number
of gas particles represented by one SPARTA particle is set by the
:doc:`global fnum <global>` command.  If cell weighting is enabled, as
set by the :doc:`global weight <global>` command, it is also affected by
the weight of the cell the particle is currently in.

Each simulation particle stores the following properties:

* ID
* type
* processor that currently owns it
* position (3 components)
* velocity (3 components)
* kinetic, vibrational, and rotation energy

The :doc:`dump particle <dump>` command can output all of these
properties, see its doc page for the associated keywords and a further
description of each property.  Various commands in SPARTA can define
and set additional per-particle properties.  The fix
ambipolar"_fix\_ambipolar.html command is an example.

The ID of each particle is a random integer from 1 to 2\^31 in size,
which is approximately 2 billion possible IDs.  The ID is assigned
when a particle is created.  This is the list of commands or
operations in SPARTA which can currently create particles:

* :doc:`create\_particles <create_particles>`
* fix emit commands: :doc:`fix emit/face <fix_emit_face>`, :doc:`fix emit/face/file <fix_emit_face_file>`, :doc:`fix emit/surf <fix_emit_surf>`
* :doc:`scale\_particles <scale_particles>`
* cell weighting: :doc:`global weight <global>` command
* gas phase reactions: :doc:`react <react>` command
* gas/surface reactions: :doc:`surf\_react <surf_react>` commands

The *ID* for a particle will persist as it moves through the simulation
domain and from processor to processor.  If a reaction occurs (gas or
surface) which creates two particles from one particle, then one of
the two new particles will have the same ID as the original particle.
The second particle will be a assigned a new random ID.  Likewise, if
a particle is cloned (cell weighting or scale\_particles), one of the
new particles will have the same ID as the original particle, the rest
will be assigned new random IDs.

Note that because IDs are assigned to particles randomly, it is
probable that multiple particles will have the same ID, even if a
simulation uses far less than 2 billion particles.  Statistically this
probability is related to the binomial coefficient C(n,k) which is the
number of ways to choose K items from N possible items.  In the case
of SPARTA particles, N = number of possible IDs = 2\^31 = 2 billion.
And K = the number of actual particles in a simulation.

For example, for a 10 million particle simulation, there will be
~23175 pairs of particles with the same ID and ~36 triplets of
particles with the same ID.  The remaining particle IDs will be
unique.

The *type* of a particle is its chemical species.  Internally this is
an integer from 1 to N, where N is the number of defined species.
This is the value output to a :doc:`dump particle <dump>` file if the
*type* keyword is used.  The mapping of integer types to species names
is determined by the :doc:`species <species>` commands used in the input
script. The first species name used is type=1, the next is type=2,
etc.


.. _sws: https://sparta.github.io
.. _sd: Manual.html
.. _sc: Section_commands.html
