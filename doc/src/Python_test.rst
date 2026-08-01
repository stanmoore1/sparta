.. _py_4:

Testing the Python-SPARTA interface
===================================

To test if SPARTA is callable from Python, launch Python interactively
and type:


.. parsed-literal::

   >>> from sparta import sparta
   >>> spa = sparta()

If you get no errors, you're ready to use SPARTA from Python.  If the
2nd command fails, the most common error to see is


.. parsed-literal::

   OSError: Could not load SPARTA dynamic library

which means Python was unable to load the SPARTA shared library.  This
typically occurs if the system can't find the SPARTA shared library or
one of the auxiliary shared libraries it depends on, or if something
about the library is incompatible with your Python.  The error message
should give you an indication of what went wrong.

You can also test the load directly in Python as follows, without
first importing from the sparta.py file:


.. parsed-literal::

   >>> from ctypes import CDLL
   >>> CDLL("libsparta.so")

If an error occurs, carefully go thru the steps in :ref:`Section 2.4 <start_4>` and above about building a shared
library and about insuring Python can find the necessary two files it
needs.

**Test SPARTA and Python in serial:**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run a SPARTA test in serial, type these lines into Python
interactively from the bench directory:


.. parsed-literal::

   >>> from sparta import sparta
   >>> spa = sparta()
   >>> spa.file("in.free")

Or put the same lines in the file test.py and run it as


.. parsed-literal::

   % python test.py

Either way, you should see the results of running the in.free
benchmark on a single processor appear on the screen, the same as if
you had typed something like:


.. parsed-literal::

   spa_g++ < in.free

You can also pass command-line switches, e.g. to set input script
variables, through the Python interface.

Replacing the "spa = sparta()" line above with


.. parsed-literal::

   spa = sparta("",["-v","x","100","-v","y","100","-v","z","100"])

is the same as typing


.. parsed-literal::

   spa_g++ -v x 100 -v y 100 -v z 100 < in.free

from the command line.

**Test SPARTA and Python in parallel:**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run SPARTA in parallel, assuming you have installed the
`Pypar <http://datamining.anu.edu.au/~ole/pypar>`_ package as discussed
above, create a test.py file containing these lines:


.. parsed-literal::

   import pypar
   from sparta import sparta
   spa = sparta()
   spa.file("in.free")
   print "Proc %d out of %d procs has" % (pypar.rank(),pypar.size()),lmp
   pypar.finalize()

You can then run it in parallel as:


.. parsed-literal::

   % mpirun -np 4 python test.py

and you should see the same output as if you had typed


.. parsed-literal::

   % mpirun -np 4 spa_g++ < in.lj

Note that if you leave out the 3 lines from test.py that specify Pypar
commands you will instantiate and run SPARTA independently on each of
the P processors specified in the mpirun command.  In this case you
should get 4 sets of output, each showing that a SPARTA run was made
on a single processor, instead of one set of output showing that
SPARTA ran on 4 processors.  If the 1-processor outputs occur, it
means that Pypar is not working correctly.

Also note that once you import the PyPar module, Pypar initializes MPI
for you, and you can use MPI calls directly in your Python script, as
described in the Pypar documentation.  The last line of your Python
script should be pypar.finalize(), to insure MPI is shut down
correctly.

**Running Python scripts:**
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that any Python script (not just for SPARTA) can be invoked in
one of several ways:


.. parsed-literal::

   % python foo.script
   % python -i foo.script
   % foo.script

The last command requires that the first line of the script be
something like this:


.. parsed-literal::

   #!/usr/local/bin/python 
   #!/usr/local/bin/python -i

where the path points to where you have Python installed, and requires
that you have made the script file executable:


.. parsed-literal::

   % chmod +x foo.script

Without the "-i" flag, Python will exit when the script finishes.
With the "-i" flag, you will be left in the Python interpreter when
the script finishes, so you can type subsequent commands.  As
mentioned above, you can only run Python interactively when running
Python on a single processor, not in parallel.
