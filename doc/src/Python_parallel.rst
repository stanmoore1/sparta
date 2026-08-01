.. _py_2:

Installing the Python wrapper into Python
=========================================

For Python to invoke SPARTA, there are 2 files it needs to know about:

* python/sparta.py
* src/libsparta.so

Sparta.py is the Python wrapper on the SPARTA library interface.
Libsparta.so is the shared SPARTA library that Python loads, as
described above.

You can insure Python can find these files in one of two ways:

* set two environment variables
* run the python/install.py script

If you set the paths to these files as environment variables, you only
have to do it once.  For the csh or tcsh shells, add something like
this to your ~/.cshrc file, one line for each of the two files:


.. parsed-literal::

   setenv PYTHONPATH ${PYTHONPATH}:/home/sjplimp/sparta/python
   setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/home/sjplimp/sparta/src

If you use the python/install.py script, you need to invoke it every
time you rebuild SPARTA (as a shared library) or make changes to the
python/sparta.py file.

You can invoke install.py from the python directory as


.. parsed-literal::

   % python install.py [libdir] [pydir]

The optional libdir is where to copy the SPARTA shared library to; the
default is /usr/local/lib.  The optional pydir is where to copy the
sparta.py file to; the default is the site-packages directory of the
version of Python that is running the install script.

Note that libdir must be a location that is in your default
LD\_LIBRARY\_PATH, like /usr/local/lib or /usr/lib.  And pydir must be a
location that Python looks in by default for imported modules, like
its site-packages dir.  If you want to copy these files to
non-standard locations, such as within your own user space, you will
need to set your PYTHONPATH and LD\_LIBRARY\_PATH environment variables
accordingly, as above.

If the install.py script does not allow you to copy files into system
directories, prefix the python command with "sudo".  If you do this,
make sure that the Python that root runs is the same as the Python you
run.  E.g. you may need to do something like


.. parsed-literal::

   % sudo /usr/local/bin/python install.py [libdir] [pydir]

You can also invoke install.py from the make command in the src
directory as


.. parsed-literal::

   % make install-python

In this mode you cannot append optional arguments.  Again, you may
need to prefix this with "sudo".  In this mode you cannot control
which Python is invoked by root.

Note that if you want Python to be able to load different versions of
the SPARTA shared library (see :ref:`this section <py_5>` below), you will
need to manually copy files like libsparta\_g++.so into the appropriate
system directory.  This is not needed if you set the LD\_LIBRARY\_PATH
environment variable as described above.
