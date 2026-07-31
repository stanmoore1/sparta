.. _howto_12:

Using multiple vibrational energy levels
========================================

DSMC models for collisions between one or more polyatomic species can
include the effect of multiple discrete vibrational levels, where a
collision transfers vibrational energy not just between the two
particles in aggregate but between the various levels defined for each
particle species.

This kind of model can be enabled in SPARTA using the following
commands:

* :doc:`species ... vibfile ... <species>`
* :doc:`collide\_modify vibrate discrete <collide_modify>`
* :doc:`fix vibmode <fix_vibmode>`
* :doc:`dump particle p\_vibmode <dump>`

The :doc:`species <species>` command with its *vibfile* option allows a
separate file with per-species vibrational information to be read.
See data/air.species.vib for an example of such a file.

Only species with 4,6,8 vibrational degrees of freedom, as defined in
the species file read by the :doc:`species <species>` command, need to
be listed in the *vibfile*\ .  These species have N modes, where N =
degrees of freedom / 2.  For each mode, a vibrational temperature,
relaxation number, and degeneracy is defined in the *vibfile*\ .  These
quantities are used in the energy exchange formulas for each
collision.

The :doc:`collide\_modify vibrate discrete <collide_modify>` command is
used to enable the discrete model.  Other allowed settings are *none*
and *smooth*\ .  The former turns off vibrational energy effects
altogether.  The latter uses a single continuous value to represent
vibrational energy; no per-mode information is used.

The :doc:`fix vibmode <fix_vibmode>` command is used to allocate
per-particle storage for the population of levels appropriate to the
particle's species.  This will be from 1 to 4 values for each species.
Note that this command must be used before particles are created via
the :doc:`create\_particles <create_particles>` command to allow the
level populations for new particles to be set appropriately.  The :doc:`fix vibmode <fix_vibmode>` command doc page has more details.

The :doc:`dump particle <dump>` command can output the custom particle
attributes defined by the :doc:`fix vibmode <fix_vibmode>` command.
E.g. this command


.. parsed-literal::

   dump 1 particle 1000 tmp.dump id type x y z evib p_vibmode[1] p_vibmode[2] p_vibmode[3]

will output for each particle evib = total vibrational energy (summed
across all levels), and the population counts for the first 3
vibrational energy levels.  The vibmode count will be 0 for
vibrational levels that do not exist for particles of a particular
species.

The :doc:`read\_restart <read_restart>` doc page explains how to restart
simulations where a fix like :doc:`fix vibmode <fix_vibmode>` has been
used to store extra per-particle properties.
