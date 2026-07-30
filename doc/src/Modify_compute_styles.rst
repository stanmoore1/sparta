.. _mod_1:

Compute styles
==============

:doc:`Compute style commands <compute>` calculate instantaneous
properties of the simulated system.  They can be global properties, or
per particle or per grid cell or per surface element properties.  The
result can be single value or multiple values (global or per particle
or per grid or per surf).

Here is a brief description of methods to define in a new derived
class.  See compute.h for details.  All of these methods are optional.

+------------------------+-----------------------------------------------------+
| init                   | initialization before a run                         |
+------------------------+-----------------------------------------------------+
| compute\_scalar        | compute a global scalar quantity                    |
+------------------------+-----------------------------------------------------+
| compute\_vector        | compute a global vector of quantities               |
+------------------------+-----------------------------------------------------+
| compute\_per\_particle | compute one or more quantities per particle         |
+------------------------+-----------------------------------------------------+
| compute\_per\_grid     | compute one or more quantities per grid cell        |
+------------------------+-----------------------------------------------------+
| compute\_per\_surf     | compute one or more quantities per surface element  |
+------------------------+-----------------------------------------------------+
| surf\_tally            | call when a particle hits a surface element         |
+------------------------+-----------------------------------------------------+
| boundary\_tally        | call when a particle hits a simulation box boundary |
+------------------------+-----------------------------------------------------+
| memory\_usage          | tally memory usage                                  |
+------------------------+-----------------------------------------------------+

Note that computes with "/particle" in their style name calculate per
particle quantities, with "/grid" in their name calculate per grid
cell quantities, and with "/surf" in their name calculate per surface
element properties.  All others calcuulate global quantities.

Flags may also need to be set by a compute to enable specific
properties.  See the compute.h header file for one-line descriptions.
