.. _start_7:

Command-line options
====================

At run time, SPARTA recognizes several optional command-line switches
which may be used in any order.  Either the full word or a one-or-two
letter abbreviation can be used:

* -e or -echo
* -i or -in
* -h or -help
* -k or -kokkos
* -l or -log
* -p or -partition
* -pk or -package
* -pl or -plog
* -ps or -pscreen
* -sc or -screen
* -sf or -suffix
* -v or -var

For example, spa\_g++ might be launched as follows:


.. parsed-literal::

   mpirun -np 16 spa_g++ -v f tmp.out -l my.log -sc none < in.sphere
   mpirun -np 16 spa_g++ -var f tmp.out -log my.log -screen none < in.sphere

Here are the details on the options:


.. parsed-literal::

   -echo style

Set the style of command echoing.  The style can be *none* or *screen*
or *log* or *both*\ .  Depending on the style, each command read from
the input script will be echoed to the screen and/or logfile.  This
can be useful to figure out which line of your script is causing an
input error.  The default value is *log*\ .  The echo style can also be
set by using the :doc:`echo <echo>` command in the input script itself.


.. parsed-literal::

   -in file

Specify a file to use as an input script.  This is an optional switch
when running SPARTA in one-partition mode.  If it is not specified,
SPARTA reads its input script from stdin - e.g. spa\_g++ < in.run.
This is a required switch when running SPARTA in multi-partition mode,
since multiple processors cannot all read from stdin.


.. parsed-literal::

   -help

Print a detailed report about this executable and immediately exit.
The report includes a banner (the SPARTA version, the git commit and
branch it was built from, the build date, the compiler and C++
standard, and the target operating system and architecture), the build
configuration (serial vs. MPI, the KOKKOS accelerator, and PNG, JPEG,
FFmpeg and gzip support), the list of command-line switches, and the
list of options (fix, compute, collide, etc.) compiled into this
executable.


.. parsed-literal::

   -kokkos on/off keyword/value ...

Explicitly enable or disable KOKKOS support, as provided by the KOKKOS
package.  Even if SPARTA is built with this package, as described
above in :ref:`Section 2.3 <start_3>`, this switch must be set to enable
running with the KOKKOS-enabled styles the package provides.  If the
switch is not set (the default), SPARTA will operate as if the KOKKOS
package were not installed; i.e. you can run standard SPARTA 
for testing or benchmarking purposes.

Additional optional keyword/value pairs can be specified which
determine how Kokkos will use the underlying hardware on your
platform.  These settings apply to each MPI task you launch via the
"mpirun" or "mpiexec" command.  You may choose to run one or more MPI
tasks per physical node.  Note that if you are running on a desktop
machine, you typically have one physical node.  On a cluster or
supercomputer there may be dozens or 1000s of physical nodes.

Either the full word or an abbreviation can be used for the keywords.
Note that the keywords do not use a leading minus sign.  I.e. the
keyword is "t", not "-t".  Also note that each of the keywords has a
default setting.  Example of when to use these options and what
settings to use on different platforms is given in :ref:`Section 5.3 <acc_3>`.

* d or device
* g or gpus
* t or threads
* n or numa


.. parsed-literal::

   device Nd

This option is only relevant if you built SPARTA with a GPU backend
(e.g. Kokkos\_ENABLE\_CUDA=ON), you
have more than one GPU per node, and if you are running with only one
MPI task per node.  The Nd setting is the ID of the GPU on the node to
run on.  By default Nd = 0.  If you have multiple GPUs per node, they
have consecutive IDs numbered as 0,1,2,etc.  This setting allows you
to launch multiple independent jobs on the node, each with a single
MPI task per node, and assign each job to run on a different GPU.


.. parsed-literal::

   gpus Ng Ns

This option is only relevant if you built SPARTA with a GPU backend
(e.g. Kokkos\_ENABLE\_CUDA=ON), you
have more than one GPU per node, and you are running with multiple MPI
tasks per node.  The Ng setting is how many GPUs
you will use per node.  The Ns setting is optional.  If set, it is the ID of a
GPU to skip when assigning MPI tasks to GPUs.  This may be useful if
your desktop system reserves one GPU to drive the screen and the rest
are intended for computational work like running SPARTA.  By default
Ng = 1 and Ns is not set.

Depending on which flavor of MPI you are running, SPARTA will look for
one of these 4 environment variables


.. parsed-literal::

   SLURM_LOCALID (various MPI variants compiled with SLURM support)
   MPT_LRANK (HPE MPI)
   MV2_COMM_WORLD_LOCAL_RANK (Mvapich)
   OMPI_COMM_WORLD_LOCAL_RANK (OpenMPI)

which are initialized by the "srun", "mpirun" or "mpiexec" commands.
The environment variable setting for each MPI rank is used to assign a
unique GPU ID to the MPI task.


.. parsed-literal::

   threads Nt

This option assigns Nt number of threads to each MPI task for
performing work when Kokkos is executing in OpenMP or pthreads mode.
The default is Nt = 1, which essentially runs in MPI-only mode.  If
there are Np MPI tasks per physical node, you generally want Np\*Nt =
the number of physical cores per node, to use your available hardware
optimally. If SPARTA is compiled with a GPU backend (e.g.
Kokkos\_ENABLE\_CUDA=ON),
this setting has no effect.


.. parsed-literal::

   -log file

Specify a log file for SPARTA to write status information to.  In
one-partition mode, if the switch is not used, SPARTA writes to the
file log.sparta.  If this switch is used, SPARTA writes to the
specified file.  In multi-partition mode, if the switch is not used, a
log.sparta file is created with hi-level status information.  Each
partition also writes to a log.sparta.N file where N is the partition
ID.  If the switch is specified in multi-partition mode, the hi-level
logfile is named "file" and each partition also logs information to a
file.N.  For both one-partition and multi-partition mode, if the
specified file is "none", then no log files are created.  Using a
:doc:`log <log>` command in the input script will override this setting.
Option -plog will override the name of the partition log files file.N.


.. parsed-literal::

   -partition 8x2 4 5 ...

Invoke SPARTA in multi-partition mode.  When SPARTA is run on P
processors and this switch is not used, SPARTA runs in one partition,
i.e. all P processors run a single simulation.  If this switch is
used, the P processors are split into separate partitions and each
partition runs its own simulation.  The arguments to the switch
specify the number of processors in each partition.  Arguments of the
form MxN mean M partitions, each with N processors.  Arguments of the
form N mean a single partition with N processors.  The sum of
processors in all partitions must equal P.  Thus the command
"-partition 8x2 4 5" has 10 partitions and runs on a total of 25
processors.  Note that with MPI installed on a machine (e.g. your
desktop), you can run on more (virtual) processors than you have
physical processors.

To run multiple independent simulations from one input script, using
multiple partitions, see :ref:`Section 6.3 <howto_3>` of
the manual.  World- and universe-style variables are useful in this
context.


.. parsed-literal::

   -package style args ....

Invoke the :doc:`package <package>` command with style and args.  The
syntax is the same as if the command appeared at the top of the input
script.  For example "-package kokkos on gpus 2" or "-pk kokkos g 2" is the same as
:doc:`package kokkos g 2 <package>` in the input script.  The possible styles
and args are documented on the :doc:`package <package>` doc page.  This
switch can be used multiple times.

Along with the "-suffix" command-line switch, this is a convenient
mechanism for invoking the KOKKOS accelerator package and its options without
having to edit an input script.


.. parsed-literal::

   -plog file

Specify the base name for the partition log files, so partition N
writes log information to file.N. If file is none, then no partition
log files are created.  This overrides the filename specified in the
-log command-line option.  This option is useful when working with
large numbers of partitions, allowing the partition log files to be
suppressed (-plog none) or placed in a sub-directory (-plog
replica\_files/log.sparta) If this option is not used the log file for
partition N is log.sparta.N or whatever is specified by the -log
command-line option.


.. parsed-literal::

   -pscreen file

Specify the base name for the partition screen file, so partition N
writes screen information to file.N. If file is none, then no
partition screen files are created.  This overrides the filename
specified in the -screen command-line option.  This option is useful
when working with large numbers of partitions, allowing the partition
screen files to be suppressed (-pscreen none) or placed in a
sub-directory (-pscreen replica\_files/screen) If this option is not
used the screen file for partition N is screen.N or whatever is
specified by the -screen command-line option.


.. parsed-literal::

   -screen file

Specify a file for SPARTA to write its screen information to.  In
one-partition mode, if the switch is not used, SPARTA writes to the
screen.  If this switch is used, SPARTA writes to the specified file
instead and you will see no screen output.  In multi-partition mode,
if the switch is not used, hi-level status information is written to
the screen.  Each partition also writes to a screen.N file where N is
the partition ID.  If the switch is specified in multi-partition mode,
the hi-level screen dump is named "file" and each partition also
writes screen information to a file.N.  For both one-partition and
multi-partition mode, if the specified file is "none", then no screen
output is performed. Option -pscreen will override the name of the 
partition screen files file.N.


.. parsed-literal::

   -suffix style args

Use variants of various styles if they exist.  The specified style can
be *kk*\ .  This refers to optional KOKKOS package that SPARTA can be built with, as described
above in :ref:`Section 2.3 <start_3>`.

Along with the "-package" command-line switch, this is a convenient
mechanism for invoking the KOKKOS accelerator package and its options without
having to edit an input script.

As an example, the KOKKOS package provides a :doc:`compute\_style temp <compute_temp>` variant, with style name temp/kk. A variant style
can be specified explicitly in your input script, e.g. compute
temp/kk. If the suffix command is used with the appropriate style,
you do not need to modify your input script.  The specified suffix
(kk) is automatically appended whenever your
input script command creates a new :doc:`fix <fix>`,
:doc:`compute <compute>`, etc.
If the variant version does not exist, the standard version is
created.

For the KOKKOS package, using this command-line switch also invokes
the default KOKKOS settings, as if the command "package kokkos" were
used at the top of your input script.  These settings can be changed
by using the "-package kokkos" command-line switch or the :doc:`package kokkos <package>` command in your script.

The :doc:`suffix <suffix>` command can also be used within an input
script to set a suffix, or to turn off or back on any suffix setting
made via the command line.


.. parsed-literal::

   -var name value1 value2 ...

Specify a variable that will be defined for substitution purposes when
the input script is read.  "Name" is the variable name which can be a
single character (referenced as $x in the input script) or a full
string (referenced as ${abc}).  An :doc:`index-style variable <variable>` will be created and populated with the
subsequent values, e.g. a set of filenames.  Using this command-line
option is equivalent to putting the line "variable name index value1
value2 ..."  at the beginning of the input script.  Defining an index
variable as a command-line argument overrides any setting for the same
index variable in the input script, since index variables cannot be
re-defined.  See the :doc:`variable <variable>` command for more info on
defining index and other kinds of variables and :ref:`Section 3.2 <cmd_2>` for more info on using variables in
input scripts.

.. warning::

   Currently, the command-line parser looks for arguments
   that start with "-" to indicate new switches. Thus you cannot specify
   multiple variable values if any of they start with a "-", e.g. a
   negative numeric value. It is OK if the first value1 starts with a
   "-", since it is automatically skipped.
