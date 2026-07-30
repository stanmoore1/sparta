.. _py_3:

Extending Python with MPI to run in parallel
============================================

If you wish to run SPARTA in parallel from Python, you need to extend
your Python with an interface to MPI.  This also allows you to
make MPI calls directly from Python in your script, if you desire.

There are several Python packages available that purport to wrap MPI
as a library and allow MPI functions to be called from Python.

These include

* `pyMPI <http://pympi.sourceforge.net/>`_
* `maroonmpi <http://code.google.com/p/maroonmpi/>`_
* `mpi4py <http://code.google.com/p/mpi4py/>`_
* `myMPI <http://nbcr.sdsc.edu/forum/viewtopic.php?t=89&sid=c997fefc3933bd66204875b436940f16>`_
* `Pypar <http://code.google.com/p/pypar>`_

All of these except pyMPI work by wrapping the MPI library and
exposing (some portion of) its interface to your Python script.  This
means Python cannot be used interactively in parallel, since they do
not address the issue of interactive input to multiple instances of
Python running on different processors.  The one exception is pyMPI,
which alters the Python interpreter to address this issue, and (I
believe) creates a new alternate executable (in place of "python"
itself) as a result.

In principle any of these Python/MPI packages should work to invoke
SPARTA in parallel and MPI calls themselves from a Python script which
is itself running in parallel.  However, when I downloaded and looked
at a few of them, their documentation was incomplete and I had trouble
with their installation.  It's not clear if some of the packages are
still being actively developed and supported.

The one I recommend, since I have successfully used it with SPARTA, is
Pypar.  Pypar requires the ubiquitous `Numpy package <http://numpy.scipy.org>`_ be installed in your Python.  After
launching python, type


.. parsed-literal::

   import numpy

to see if it is installed.  If not, here is how to install it (version
1.3.0b1 as of April 2009).  Unpack the numpy tarball and from its
top-level directory, type


.. parsed-literal::

   python setup.py build
   sudo python setup.py install

The "sudo" is only needed if required to copy Numpy files into your
Python distribution's site-packages directory.

To install Pypar (version pypar-2.1.4\_94 as of Aug 2012), unpack it
and from its "source" directory, type


.. parsed-literal::

   python setup.py build
   sudo python setup.py install

Again, the "sudo" is only needed if required to copy Pypar files into
your Python distribution's site-packages directory.

If you have successfully installed Pypar, you should be able to run
Python and type


.. parsed-literal::

   import pypar

without error.  You should also be able to run python in parallel
on a simple test script


.. parsed-literal::

   % mpirun -np 4 python test.py

where test.py contains the lines


.. parsed-literal::

   import pypar
   print "Proc %d out of %d procs" % (pypar.rank(),pypar.size())

and see one line of output for each processor you run on.

.. warning::

   To use Pypar and SPARTA in parallel from Python, you
   must insure both are using the same version of MPI.  If you only have
   one MPI installed on your system, this is not an issue, but it can be
   if you have multiple MPIs.  Your SPARTA build is explicit about which
   MPI it is using, since you specify the details in your lo-level
   src/MAKE/Makefile.foo file.  Pypar uses the "mpicc" command to find
   information about the MPI it uses to build against.  And it tries to
   load "libmpi.so" from the LD\_LIBRARY\_PATH.  This may or may not find
   the MPI library that SPARTA is using.  If you have problems running
   both Pypar and SPARTA together, this is an issue you may need to
   address, e.g. by moving other MPI installations so that Pypar finds
   the right one.
