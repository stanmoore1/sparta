.. _howto_11:

Using the ambipolar approximation
=================================

The ambipolar approximation is a computationally efficient way to
model low-density plasmas which contain positively-charged ions and
negatively-charged electrons.  In this model, electrons are not free
particles which move independently.  This would require a simulation
with a very small timestep due to electron's small mass and high speed
(1000x that of an ion or neutral particle).

Instead each ambipolar electron is assumed to stay "close" to its
parent ion, so that the plasma gas appears macroscopically neutral.
Each pair of particles thus moves together through the simulation
domain, as if they were a single particle, which is how they are
stored within SPARTA.  This means a normal timestep can be used.

There are two stages during a timestep when the coupled particles are
broken apart and treated as an independent ion and electron.

The first is during gas-phase collisions and chemistry.  The ionized
ambipolar particles in a grid cell are each split into two particles
(ion and electron) and each can participate in two-body collisions
with any other particle in the cell.  Electron/electron collisions are
actually not performed, but are tallied in the overall collision count
(if using a collision mixture with a single group, not when using
multiple groups).  If gas-phase chemistry is turned on, reactions
involving ions and electrons can be specified, which include
dissociation, ionization, exchange, and recombination reactions.  At
the end of the collision/chemistry operations for the grid cell, there
is still a one-to-one pairing between ambipolar ions and electrons.
Each pair is recombined into a single particle.

The second is during collisions with surface (or the boundaries of the
simulation box) if a surface reaction model is defined for the surface
element or boundary.  Just as with gas-phase chemistry, surface
reactions involving ambipolar species can be defined.  For example, an
ambipolar ion/electron pair can re-combine into a neutral species during
the collision.

Here are the SPARTA commands you can use to run a simulation using the
ambipolar approximation.  See the input scripts in examples/ambi for
an example.

Note that you will likely need to use two (or more mixtures) as
arguments to various commands, one which includes the ambipolar
electron species, and one which does not.  Example
:doc:`mixture <mixture>` commands for doing this are shown below.

Use the :doc:`fix ambipolar <fix_ambipolar>` command to specify which
species is the ambipolar electron and what (multiple) species are
ambipolar ions.  This is required for all the other options listed
here to work.  The fix defines two custom per-particle attributes, an
integer vector called "ionambi" which stores a 1 for a particle if it
is an ambipolar ion, and a 0 otherwise.  And a floating-point array
called "velambi" which stores a 3-vector with the velocity of the
associated electron for each ambipolar ion or zeroes otherwise.  Note
that no particles should ever exist in the simulation with a species
matching ambipolar electrons.  Such particles are only generated (and
destroyed) internally, as described above.

Use the :doc:`collide\_modify ambipolar yes <collide_modify>` command if
you want to perform gas-phase collisions using the ambipolar model.
This is not required.  If you do this, you may also want to specify a
mixture for the collide command which has two or more groups.  If this
is the case, the ambipolar electron species must be in a group by
itself.  The other group(s) can contain any combination of ion or
neutral species.  Note that putting the ambipolar electron species in
its own group should improve the efficiency of the code due to the
large disparity in electron versus ion/neutral velocities.

If you want to perform gas-phase chemistry for reactions involving
ambipolar ions and electrons, use the :doc:`react <react>` command with
an input file of reactions that include the ambipolar electron and ion
species defined by the fix ambipolar command.  See the
:doc:`react <react>` command doc page for info the syntax required for
ambipolar reactions.  Their reactants and products must be listed in
specific order.

When creating particles, either by the
:doc:`create\_particles <create_particles>` or :doc:`fix emit <fix_emit_face>`
command variants, do NOT use a mixture that includes the ambipolar
electron species.  If you do this, you will create "free" electrons
which are not coupled to an ambipolar ion.  You can include ambipolar
ions in the mixture.  This will create ambipolar ions along with their
associated electron.  The electron will be assigned a velocity
consistent with its mass and the temperature of the created particles.
You can use the :doc:`mixture copy <mixture>` and :doc:`mixture delete <mixture>` commands to create a mixture that excludes only
the ambipolar electron species, e.g.


.. parsed-literal::

   mixture all copy noElectron
   mixture noElectron delete e

If you want ambipolar ions to re-combine with their electrons when
they collide with surfaces, use the :doc:`surf\_react <surf_react>`
command with an input file of surface reactions that includes
recombination reactions like:


.. parsed-literal::

   N+ + e -> N

See the :doc:`surf\_react <surf_react>` doc page for syntax details.  A
sample surface reaction data file is provided in data/air.surf.  You
assign the surface reaction model to surface or the simulation box
boundaries via the :doc:`surf\_modify <surf_modify>` and
:doc:`bound\_modify <bound_modify>` commands.

For diagnostics and output, you can use the :doc:`compute count <compute_count>` and :doc:`dump particle <dump>` commands.  The
:doc:`compute count <compute_count>` command generate counts of
individual species, entire mixtures, and groups within mixtures.  For
example these commands will include counts of ambipolar ions in
statistical output:


.. parsed-literal::

   compute myCount O+ N+ NO+ e
   stats_style step nsreact nsreactave cpu np c_myCount

Note that the count for species "e" = ambipolar electrons should always
be zero, since those particles only exist during gas and surface
collisions.  The :doc:`stats\_style <stats_style>` *nsreact* and
*nsreactave* keywords print tallies of surface reactions taking place.

The :doc:`dump particle <dump>` command can output the custom particle
attributes defined by the :doc:`fix ambipolar <fix_ambipolar>` command.
E.g. this command


.. parsed-literal::

   dump 1 particle 1000 tmp.dump id type x y z p_ionambi p_velambi[2]

will output the ionambi flag = 1 for ambipolar ions, along with the vy
of their associated ambipolar electrons.

The :ref:`fix ambipolar <fix>` ambipolar.html doc page explains how to
restart ambipolar simulations where the fix is used.
