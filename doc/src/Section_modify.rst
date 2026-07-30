Modifying & extending SPARTA
============================

This section describes how to extend SPARTA by modifying its source code.


SPARTA is designed in a modular fashion so as to be easy to modify and
extend with new functionality.

In this section, changes and additions users can make are listed along
with minimal instructions.  If you add a new feature to SPARTA and
think it will be of general interest to users, please submit it to the
`developers <https://sparta.github.io/authors.html>`_ for inclusion in
the released version of SPARTA.

The best way to add a new feature is to find a similar feature in
SPARTA and look at the corresponding source and header files to figure
out what it does. You will need some knowledge of C++ to be able to
understand the hi-level structure of SPARTA and its class
organization, but functions (class methods) that do actual
computations are written in vanilla C-style code and operate on simple
C-style data structures (vectors, arrays, structs).

The new features described in this section require you to write a new
C++ derived class. Creating a new class requires 2 files, a source
code file (\*.cpp) and a header file (\*.h).  The derived class must
provide certain methods to work as a new option.  Depending on how
different your new feature is compared to existing features, you can
either derive from the base class itself, or from a derived class that
already exists.  Enabling SPARTA to invoke the new class is as simple
as putting the two source files in the src dir and re-building SPARTA.

The advantage of C++ and its object-orientation is that all the code
and variables needed to define the new feature are in the 2 files you
write, and thus shouldn't make the rest of SPARTA more complex or
cause side-effect bugs.

Here is a concrete example. Suppose you write 2 files collide\_foo.cpp
and collide\_foo.h that define a new class CollideFoo that computes
inter-particle collisions described in the classic 1997 paper by Foo,
et al. If you wish to invoke those potentials in a SPARTA input script
with a command like

collide foo mix-ID params.foo 3.0

then your collide\_foo.h file should be structured as follows:

#ifdef COLLIDE\_CLASS
CollideStyle(foo,CollideFoo)
#else
...
(class definition for CollideFoo)
...
#endif

where "foo" is the style keyword in the collid command, and CollideFoo
is the class name defined in your collide\_foo.cpp and collide\_foo.h
files.

When you re-build SPARTA, your new collision model becomes part of the
executable and can be invoked with a :doc:`collide <collide>` command
like the example above.  Arguments like a mixture ID, params.foo (a
file with collision parameters), and 3.0 can be defined and processed
by your new class.

As illustrated by this example, many kinds of options are referred to
in the SPARTA documentation as the "style" of a particular command.

The instructions below give the header file for the base class that
these styles are derived from.  Public variables in that file are ones
used and set by the derived classes which are also used by the base
class.  Sometimes they are also used by the rest of SPARTA.  Virtual
functions in the base class header file which are set = 0 are ones
that must be defined in the new derived class to give it the
functionality SPARTA expects.  Virtual functions that are not set to 0
are functions that can be optionally defined.

Here are additional guidelines for modifying SPARTA and adding new
functionality:

* Think about whether what you want to do would be better as a pre- or
  post-processing step. Many computations are more easily and more
  quickly done that way.
* Don't do anything within the timestepping of a run that isn't
  parallel.  E.g. don't accumulate a large volume of data on a single
  processor and analyze it.  This runs the risk of seriously degrading
  the parallel efficiency.

  If you have a question about how to compute something or about
  internal SPARTA data structures or algorithms, feel free to send an
  email to the `developers <https://sparta.github.io/authors.html>`_.

* If you add something you think is generally useful, also send an email
  to the `developers <https://sparta.github.io/authors.html>`_ so we can
  consider adding it to the SPARTA distribution.


.. toctree::
   :maxdepth: 1

   Modify_compute_styles
   Modify_fix_styles
   Modify_region_styles
   Modify_collision_styles
   Modify_surf_collide
   Modify_chemistry_styles
   Modify_dump_styles
   Modify_command
