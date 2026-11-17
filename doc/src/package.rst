.. index:: package

package command
===============

**Syntax:**


.. parsed-literal::

   package style args

* style = *kokkos*
* args = arguments specific to the style
  
  .. parsed-literal::
  
       kokkos args = keyword value ...
         zero or more keyword/value pairs may be appended
         keywords = comm or react/extra or react/retry or gpu/aware or plist or comm/batch
           comm value = threaded or serial
             threaded = perform pack/unpack using KOKKOS (e.g. on GPU or using OpenMP) (default for GPUs)
             serial = perform communication pack/unpack in non-KOKKOS mode (default for CPUs)
           react/extra = factor
             factor = increase memory used for collisions by this factor (default)
           react/retry = yes or no
             yes = retry collision algorithm until successful
             no = do not retry collision algorithm (default)
           gpu/aware = yes or no
             yes = use GPU-aware MPI (default)
             no = do not use GPU-aware MPI
           plist value = dense or flat or sliced
             dense = store per-cell particle lists in a 2d array (default)
             flat = store per-cell particle lists back to back in a 1d array
             sliced = store per-cell particle lists in a 1d array, interleaved in blocks of 32 cells
           comm/batch = yes or no
             yes = pack all irregular messages in one kernel and send them with MPI_Isend
             no = pack and send irregular messages one at a time (default)



**Examples:**


.. parsed-literal::

   package kokkos comm serial
   package kokkos comm threaded react/retry yes
   package kokkos gpu/aware no
   package kokkos plist flat
   package kokkos comm/batch yes

**Description:**

This command invokes package-specific settings for the KOKKOS
accelerator package available in SPARTA.

If this command is specified in an input script, it must be near the
top of the script, before the simulation box has been created.  This
is because it specifies settings that the accelerator package used in
its initialization, before a simulation is defined.

This command can also be specified from the command-line when
launching SPARTA, using the "-pk" `command-line switch <Section_start.html#start_7>`_.  The syntax is exactly the same as
when used in an input script.

Note that the KOKKOS accelerator package requires the package command
to be specified, if the package is to be used in a simulation (SPARTA
can be built with the accelerator package without using it in a
particular simulation).  However, a default version of the command is
typically invoked by other accelerator settings. For example, the
KOKKOS package requires a "-k on" `command-line switch <Section_start.html#start_7>`_ respectively, which invokes a
"package kokkos" command with default settings.

NOTE: A package command for a particular style can be invoked multiple
times when a simulation is setup, e.g. by the "-k on", "-sf", and
"-pk" `command-line switches <Section_start.html#start_7>`_, and by using
this command in an input script.  Each time it is used all of the
style options are set, either to default values or to specified
settings.  I.e. settings from previous invocations do not persist
across multiple invocations.

See the the `Accelerating SPARTA <Section_accelerate.html#acc_3>`_
section of the manual for more details about using the various
accelerator packages for speeding up SPARTA simulations.


----------


The *kokkos* style invokes settings associated with the use of the
KOKKOS package.

All of the settings are optional keyword/value pairs.  Each has a
default value as listed below.

Chemical reactions (gas or surface) can increase the number of
particles in the simulation, which requires extra memory storage. It
is not possible to resize Kokkos data structures during the reaction
routines, so two workarounds are provided. The default is to use the
*react/extra* keyword, which ensures there is extra memory allocated
to store new particles. For example, if *react/extra* is set to 1.1,
then the memory is over-allocated by 10%. If this space is still not
sufficient to hold new particles, the code will error out and the
simulation must be restarted using a larger value for *react/extra*\ .
Alternatively, if the *react/retry* option is set to *yes*\ , backup
copies of the Kokkos data structures are created. If space is exceeded
during reactions, the Kokkos data structures are restored from backup,
their size is increased, and the collide or move routine is started
over from the beginning. This guarantees that reactions will
eventually succeed without producing an error, but increases memory by
a factor of 2 and also has overhead from making a backup copy of the
data. If the *react/retry* option is set to *yes*\ , the
*react/extra* keyword will be ignored. If reactions are not defined,
both of these options will be ignored.

The *comm* keyword determines whether the host or device performs the
packing and unpacking of data when communicating per-atom data between
processors. The value options are *threaded* or *serial*\ .

The optimal choice for this keyword depends on the hardware used.
When running on CPUs or Xeon Phi, the *serial* option is typically
fastest. When using GPUs, the *threaded* value will typically be
optimal. In this case data can stay on the GPU for many timesteps
without being fully moved between the host and GPU.

The *gpu/aware* keyword chooses whether GPU-aware MPI will be used. When 
this keyword is set to *on*\ , buffers in GPU memory are passed directly 
through MPI send/receive calls. This can reduce overhead of first 
copying the data to the host CPU. However GPU-aware MPI is not supported on 
all systems, which can lead to segmentation faults and would require 
using a value of *off*\ .

The *plist* keyword chooses how the list of particles in each grid
cell is stored. With *dense*\ , the lists are rows of a 2d array, each
as long as the fullest cell plus padding, so memory scales as the
number of grid cells times the largest per-cell particle count. When
particle counts per cell vary widely, as they do after grid adaptation
refines cells near a surface, this can be many times the memory of the
particles themselves, and can exhaust GPU memory. With *flat* and
*sliced*\ , memory instead scales with the number of particles. With
*flat*\ , each cell's list is sized from its own particle count and the
lists are stored back to back. With *sliced*\ , consecutive cells are
grouped in blocks of 32, each block is sized from its fullest cell,
and the lists in a block are interleaved, so that GPU threads working
on neighboring cells access neighboring memory. The per-cell scratch
arrays used by collisions are laid out the same way as the particle
list.

All three settings give the same results; with a single thread on
CPUs they are bitwise identical. On GPUs, *sliced* is
typically as fast as *dense* on uniform grids, using slightly more
memory than *flat*\ . *Flat* can be faster than *sliced* for adapted
grids with near-neighbor collisions (see the
:doc:`collide\_modify <collide_modify>` nearcp option), where each thread
scans many particles in its own cell. When reactions are defined, each
cell's list in the *flat* and *sliced* layouts is padded by the
*react/extra* factor plus a few slots, and the padding is increased
automatically if a cell overflows and *react/retry* is set to *yes*\ .

The *comm/batch* keyword changes how irregular communication, such as
particle migration, sends its messages.  With the default *no*\ , the
messages are sent one at a time: each one is packed by its own kernel,
the device is synchronized, and the message is sent with a blocking
MPI_Send before the next one is packed.  With *yes*\ , all messages are
packed by a single kernel into one buffer, the device is synchronized
once, and every message is then posted with a non-blocking MPI_Isend,
so that their latencies overlap.  For particle migration, the
per-message particle counts are also exchanged only with neighboring
processors using non-blocking calls, which removes the global
MPI_Barrier otherwise done every timestep.  This can reduce communication time
when a processor sends to many other processors, particularly over
networks with higher latency.  It uses extra device memory (and host
memory if *gpu/aware* is *no*\ ) to hold all outgoing messages at once.


----------


**Restrictions:**

This command cannot be used after the simulation box is defined by a
:doc:`create\_box <create_box>` command.

The kk style of this command can only be invoked if SPARTA was built
with the KOKKOS package.  See the `Making SPARTA <Section_start.html#start_3>`_ section for more info.

**Related commands:**

:doc:`suffix <suffix>`, "-pk" `command-line setting <Section_start.html#start_7>`_

**Default:**

For the KOKKOS package, the option defaults are react/extra = 1.1,
react/retry = no, gpu/aware yes, plist = dense, and comm/batch no. For CPUs: comm = serial, and for
GPUs: comm = threaded.  These settings are made automatically by the
required "-k on" `command-line switch <Section_start.html#start_7>`_. You
can change them by using the package kokkos command in your input script
or via the "-pk kokkos" `command-line switch <Section_start.html#start_7>`_.


.. _sws: https://sparta.github.io
.. _sd: Manual.html
.. _sc: Section_commands.html
