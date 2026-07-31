.. _mod_8:

Input script commands
=====================

New commands can be added to SPARTA that will be recognized in input
scripts.  For example, the :doc:`create\_particles <create_particles>`,
:doc:`read\_surf <read_surf>`, and :doc:`run <run>` commands are all
implemented in this fashion.  When such a command is encountered in an
input script, SPARTA simply creates a class with the corresponding
name, invokes the "command" method of the class, and passes it the
arguments from the input script.  The command() method can perform
whatever operations it wishes on SPARTA data structures.

The single method the new class must define is as follows:

+---------+--------------------------------------------------+
| command | operations performed by the input script command |
+---------+--------------------------------------------------+

Of course, the new class can define other methods and variables as
needed.


.. _sws: https://sparta.github.io
.. _sd: Manual.html
.. _sc: Section_commands.html
