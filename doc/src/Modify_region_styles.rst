.. _mod_3:

Region styles
=============

:doc:`Region style commands <region>` define geometric regions
within the simulation box.  Other commands use regions
to limit their computational scope.

Here is a brief description of methods to define in a new derived
class.  See region.h for details.  The inside() method is required.

inside: determine whether a point is inside/outside the region
