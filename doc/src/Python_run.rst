.. _py_5:

Using SPARTA from Python
========================

The Python interface to SPARTA consists of a Python "sparta" module,
the source code for which is in python/sparta.py, which creates a
"sparta" object, with a set of methods that can be invoked on that
object.  The sample Python code below assumes you have first imported
the "sparta" module in your Python script, as follows:


.. parsed-literal::

   from sparta import sparta

These are the methods defined by the sparta module.  If you look
at the file src/library.cpp you will see that they correspond
one-to-one with calls you can make to the SPARTA library from a C++ or
C or Fortran program.


.. parsed-literal::

   spa = sparta()           # create a SPARTA object using the default libsparta.so library
   spa = sparta("g++")      # create a SPARTA object using the libsparta_g++.so library
   spa = sparta("",list)    # ditto, with command-line args, e.g. list = ["-echo","screen"]
   spa = sparta("g++",list)

   spa.close()              # destroy a SPARTA object

   spa.file(file)           # run an entire input script, file = "in.lj"
   spa.command(cmd)         # invoke a single SPARTA command, cmd = "run 100"

   fnum = spa.extract_global(name,type) # extract a global quantity
                                        # name = "dt", "fnum", etc
                                     # type = 0 = int
                                     #        1 = double

   temp = spa.extract_compute(id,style,type) # extract value(s) from a compute
                                             # id = ID of compute
                                          # style = 0 = global data
                                          #         1 = per particle data
                                          #         2 = per grid cell data
                                          #         3 = per surf element data
                                          # type = 0 = scalar
                                          #        1 = vector
                                          #        2 = array

   var = spa.extract_variable(name,flag)  # extract value(s) from a variable
                                       # name = name of variable
                                       # flag = 0 = equal-style variable
                                       #        1 = particle-style variable


----------


.. warning::

   Currently, the creation of a SPARTA object from within
   sparta.py does not take an MPI communicator as an argument.  There
   should be a way to do this, so that the SPARTA instance runs on a
   subset of processors if desired, but I don't know how to do it from
   Pypar.  So for now, it runs with MPI\_COMM\_WORLD, which is all the
   processors.  If someone figures out how to do this with one or more of
   the Python wrappers for MPI, like Pypar, please let us know and we
   will amend these doc pages.

Note that you can create multiple SPARTA objects in your Python
script, and coordinate and run multiple simulations, e.g.


.. parsed-literal::

   from sparta import sparta
   spa1 = sparta()
   spa2 = sparta()
   spa1.file("in.file1")
   spa2.file("in.file2")

The file() and command() methods allow an input script or single
commands to be invoked.

The extract\_global(), extract\_compute(), and extract\_variable()
methods return values or pointers to data structures internal to
SPARTA.

For extract\_global() see the src/library.cpp file for the list of
valid names.  New names can easily be added.  A double or integer is
returned.  You need to specify the appropriate data type via the type
argument.

For extract\_compute(), the global, per particle, per grid cell, or per
surface element results calculated by the compute can be accessed.
What is returned depends on whether the compute calculates a scalar or
vector or array.  For a scalar, a single double value is returned.  If
the compute or fix calculates a vector or array, a pointer to the
internal SPARTA data is returned, which you can use via normal Python
subscripting.  See :ref:`Section 6.4 <howto_4>` of the
manual for a discussion of global, per particle, per grid, and per
surf data, and of scalar, vector, and array data types.  See the doc
pages for individual :doc:`computes <compute>` for a description of what
they calculate and store.

For extract\_variable(), an :doc:`equal-style or particle-style variable <variable>` is evaluated and its result returned.

For equal-style variables a single double value is returned and the
group argument is ignored.  For particle-style variables, a vector of
doubles is returned, one value per particle, which you can use via
normal Python subscripting.


----------


As noted above, these Python class methods correspond one-to-one with
the functions in the SPARTA library interface in src/library.cpp and
library.h.  This means you can extend the Python wrapper via the
following steps:

* Add a new interface function to src/library.cpp and
  src/library.h.
* Rebuild SPARTA as a shared library.
* Add a wrapper method to python/sparta.py for this interface
  function.
* You should now be able to invoke the new interface function from a
  Python script.  Isn't ctypes amazing?
