.. _mod_5:

Surface collision styles
========================

:doc:`Surface collision style commands <collide>` define collision
models that calculate interactions between a particle and surface
element.

Here is a brief description of methods to define in a new derived
class.  See surf\_collide.h for details.  All of these methods are
required except dynamic().

+---------+------------------------------------------------------+
| init    | initialization before a run                          |
+---------+------------------------------------------------------+
| collide | perform a particle/surface-element collision         |
+---------+------------------------------------------------------+
| dynamic | allow surface property to change during a simulation |
+---------+------------------------------------------------------+
