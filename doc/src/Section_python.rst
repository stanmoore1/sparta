Python interface to SPARTA
==========================

This section describes various ways that SPARTA and Python can be used
together.

* 11.1 :ref:`Building SPARTA as a shared library <py_1>`
* 11.2 :ref:`Installing the Python wrapper into Python <py_2>`
* 11.3 :ref:`Extending Python with MPI to run in parallel <py_3>`
* 11.4 :ref:`Testing the Python-SPARTA interface <py_4>`
* 11.5 :ref:`Using SPARTA from Python <py_5>`
* 11.6 :ref:`Example Python scripts that use SPARTA <py_6>`
* 11.7 :ref:`Calling Python from SPARTA <py_7>`

If you are not familiar with `Python <https://www.python.org>`_, it is
a powerful scripting and programming language which can do almost
everything that compiled languages like C, C++, or Fortran can do in
fewer lines of code. It also comes with a large collection of add-on
modules for many purposes (either bundled or easily installed from
Python code repositories).  The major drawback is slower execution
speed of the script code compared to compiled programming languages.
But when the script code is interfaced to optimized compiled code,
performance can be on par with a standalone executable, so long as the
scripting is restricted to high-level operations.  Thus Python is also
convenient to use as a "glue" language to "drive" a program like
SPARTA through its library interface, or to hook multiple pieces of
software together, such as a simulation code and a visualization tool,
or to run a coupled multi-scale or multi-physics model.

The SPARTA distribution includes the file python/sparta.py which wraps
the library interface to SPARTA.  That interface is exposed to Python
either when calling SPARTA from Python or when calling Python from a
SPARTA input script and then calling back to SPARTA from Python code.
It is a C-library interface which is designed to be easy to add
functionality to, thus the Python interface to SPARTA is easy to
extend as well.

The Python wrapper for SPARTA uses the amazing "ctypes" package in
Python, which auto-generates the interface code needed between Python
and a set of C interface routines for a library.  Ctypes is part of
standard Python for versions 2.5 and later.  You can check which
version of Python you have installed, by simply typing "python" at a
shell prompt.

If you create interesting Python scripts that run SPARTA or
interesting Python functions that can be called from a SPARTA input
script, that you think would be generally useful, please post them as
a pull request to our "GitHub site"_
https://github.com/sparta/sparta, and they can be added to the
SPARTA distribution or web page.

Before using SPARTA from a Python script, you need to do two things.
You need to build SPARTA as a dynamic shared library, so it can be
loaded by Python.  And you need to tell Python how to find the library
and the Python wrapper file python/sparta.py.  Both these steps are
discussed next.  If you wish to run SPARTA in parallel from Python,
you also need to extend your Python with MPI.  This is also discussed
below.


.. toctree::
   :maxdepth: 1

   Python_shlib
   Python_parallel
   Python_mpi
   Python_test
   Python_run
   Python_examples
   Python_call
