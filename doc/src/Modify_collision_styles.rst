.. _mod_4:

Collision styles
================

:doc:`Collision style commands <collide>` define collision models that
calculate interactions between particles in the same grid cell.

Here is a brief description of methods to define in a new derived
class.  See collide.h for details.  All of these methods are required
except init() and modify\_params().

+--------------------+---------------------------------------------------------------------------------------+
| init               | initialization before a run                                                           |
+--------------------+---------------------------------------------------------------------------------------+
| modify\_params     | process style-specific options of the :doc:`collide\_modify <collide_modify>` command |
+--------------------+---------------------------------------------------------------------------------------+
| vremax\_init       | estimate VREmax settings                                                              |
+--------------------+---------------------------------------------------------------------------------------+
| attempt\_collision | compute # of collisions to attempt for entire cell                                    |
+--------------------+---------------------------------------------------------------------------------------+
| attempt\_collision | compute # of collisions to attempt between 2 species groups                           |
+--------------------+---------------------------------------------------------------------------------------+
| test\_collision    | determine if a collision between 2 particles occurs                                   |
+--------------------+---------------------------------------------------------------------------------------+
| setup\_collision   | pre-computation before a 2-particle collision                                         |
+--------------------+---------------------------------------------------------------------------------------+
| perform\_collision | calculate the outcome of a 2-particle collision                                       |
+--------------------+---------------------------------------------------------------------------------------+
