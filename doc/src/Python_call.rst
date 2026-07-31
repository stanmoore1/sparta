.. _py_7:

Calling Python from SPARTA
==========================

There are SPARTA input script commands which can invoke Python code directly.

* :doc:`python <python>`
* :doc:`python-style variables <variable>`
* :doc:`equal-style and grid-style variables with formulas containing Python function wrappers <variable>`

The :doc:`python <python>` command can be used to define and execute a
Python function that you write the code for.  The Python function can
also be assigned to a SPARTA python-style variable via the
:doc:`variable <variable>` command.  Each time the variable is
evaluated, either in the SPARTA input script itself, or by another
SPARTA command that uses the variable, this will trigger the Python
function to be invoked.

The Python function can also be referenced in the formula used to
define an :doc:`equal-style or grid-style variable <variable>`, using
the syntax for a :doc:`Python function wrapper <variable>`.  This make
it easy to pass SPARTA-related arguments to the Python function, as
well as to invoke it whenever the equal- or grid-style variable is
evaluated.  For a grid-style variable it means the Python function can
be invoked once per grid cell, using per-grid properties as arguments
to the function.

The Python code for the function can be included directly in the input
script or in an auxiliary file.  The function can have arguments which
are mapped to SPARTA variables (also defined in the input script) and
it can return a value to a SPARTA variable.  This is thus a mechanism
for your input script to pass information to a piece of Python code,
ask Python to execute the code, and return information to your input
script.

Note that a Python function can be arbitrarily complex.  It can import
other Python modules, instantiate Python classes, call other Python
functions, etc.  The Python code that you provide can contain more
code than the single function.  It can contain other functions or
Python classes, as well as global variables or other mechanisms for
storing state between calls from SPARTA to the function.

The Python function you provide can consist of "pure" Python code that
only performs operations provided by standard Python.  However, the
Python function can also "call back" to SPARTA through its
Python-wrapped library interface, in the manner described above.  This
means it can issue SPARTA input script commands or query and set
internal SPARTA state.  As an example, this can be useful in an input
script to create a more complex loop with branching logic, than can be
created using the simple looping and branching logic enabled by the
:doc:`next <next>` and :doc:`if <if>` commands.

See the :doc:`python <python>` and :doc:`variable <variable>`
command doc pages for more info on using Python from a SPARTA input
script including examples of Python code you can write for both pure
Python operations and callbacks to SPARTA.


.. _sws: https://sparta.github.io
.. _sd: Manual.html
.. _sc: Section_commands.html
