.. _mod_2:

Fix styles
==========

:doc:`Fix style commands <fix>` perform operations during the
timestepping loop of a simulation.  They can define methods which are
invoked at different points within the timestep.  They can be used to
insert particles, perform load-balancing, or perform time-averaging of
various quantities.  They can also define and maintain new
per-particle vectors and arrays that define quantities that move with
particles when they migrate from processor to processor or when the
grid is rebalanced or adapated.  They can also produce output of
various kinds, similar to :doc:`compute <compute>` commands.

Here is a brief description of methods to define in a new derived
class.  See fix.h for details.  All of these methods are optional,
except setmask().

+-----------------+-------------------------------------------------------------------+
| setmask         | set flags that determine when the fix is called within a timestep |
+-----------------+-------------------------------------------------------------------+
| init            | initialization before a run                                       |
+-----------------+-------------------------------------------------------------------+
| start\_of\_step | called at beginning of timestep                                   |
+-----------------+-------------------------------------------------------------------+
| end\_of\_step   | called at end of timestep                                         |
+-----------------+-------------------------------------------------------------------+
| add\_particle   | called when a particle is created                                 |
+-----------------+-------------------------------------------------------------------+
| surf\_react     | called when a surface reaction occurs                             |
+-----------------+-------------------------------------------------------------------+
| memory\_usage   | tally memory usage                                                |
+-----------------+-------------------------------------------------------------------+

Flags may also need to be set by a fix to enable specific properties.
See the fix.h header file for one-line descriptions.

Fixes can interact with the Particle class to create new
per-particle vectors and arrays and access and update their
values.  These are the relevant Particle class methods:

+----------------+--------------------------------------------------+
| add\_custom    | add a new custom vector or array                 |
+----------------+--------------------------------------------------+
| find\_custom   | find a previously defined custom vector or array |
+----------------+--------------------------------------------------+
| remove\_custom | remove a custom vector or array                  |
+----------------+--------------------------------------------------+

See the :doc:`fix ambipolar <fix_ambipolar>` for an example of how these
are used.  It define an integer vector called "ionambi" to flag
particles as ambipolar ions, and a floatin-point array called
"velambi" to store the velocity vector for the associated electron.
