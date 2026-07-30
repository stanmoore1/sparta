.. _mod_7:

Dump styles
===========

:doc:`Dump commands <dump>` output snapshots of simulation data to a
file periodically during a simulation, in a particular file format.
Per particle, per grid cell, or per surface element data can be
output.

Here is a brief description of methods to define in a new derived
class.  See dump.h for details.  The init\_style(), modify\_param(), and
memory\_usage() methods are optional; all the others are required.

+---------------+---------------------------------------------------------------------------------+
| init\_style   | style-specific initialization before a run                                      |
+---------------+---------------------------------------------------------------------------------+
| modify\_param | process style-specific options of the :doc:`dump\_modify <dump_modify>` command |
+---------------+---------------------------------------------------------------------------------+
| write\_header | write the header of a snapshot to a file                                        |
+---------------+---------------------------------------------------------------------------------+
| count         | # of entities this processor will output                                        |
+---------------+---------------------------------------------------------------------------------+
| pack          | pack a processor's data into a buffer                                           |
+---------------+---------------------------------------------------------------------------------+
| write\_data   | write a buffer of data to a file                                                |
+---------------+---------------------------------------------------------------------------------+
| memory\_usage | tally memory usage                                                              |
+---------------+---------------------------------------------------------------------------------+
