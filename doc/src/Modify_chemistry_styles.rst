.. _mod_6:

Chemistry styles
================

Particle/particle chemistry models in SPARTA are specified by
:doc:`reaction style commands <react>` which define lists of possible
reactions and their parameters.

Here is a brief description of methods to define in a new derived
class.  See react.h for details.  The init() method is optional;
the attempt() method is required.

+---------+---------------------------------------------------+
| init    | initialization before a run                       |
+---------+---------------------------------------------------+
| attempt | attempt a chemical reaction between two particles |
+---------+---------------------------------------------------+
