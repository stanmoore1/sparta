.. _acc_2:

Packages with optimized styles
==============================

Accelerated versions of various :doc:`collide\_style <collide>`,
:doc:`fixes <fix>`, :doc:`computes <compute>`, and other commands have
been added to SPARTA via the KOKKOS package, which may run faster than
the standard non-accelerated versions.

All of these commands are in the KOKKOS package provided with SPARTA.
An overview of packages is give in :doc:`Section packages <Section_packages>`.

SPARTA currently has acceleration support for different kinds of hardware
via the KOKKOS package: many-core CPUs, NVIDIA GPUs, AMD GPUs, and
Intel GPUs.

Whether you will see speedup for your hardware may depend on the size
problem you are running and what commands (accelerated and
non-accelerated) are invoked by your input script.  While these doc
pages include performance guidelines, there is no substitute for
trying out the KOKKOS package.

Any accelerated style has the same name as the corresponding standard
style, except that a suffix is appended.  Otherwise, the syntax for
the command that uses the style is identical, their functionality is
the same, and the numerical results it produces should also be the
same, except for precision and round-off effects, and differences in
random numbers.

For example, the KOKKOS package provides an accelerated variant of the
Temperature Compute :doc:`compute temp <compute_temp>`, namely :doc:`compute temp/kk <compute_temp>`

To see what accelerate styles are currently available, see :ref:`Section 3.5 <cmd_5>` of the manual.  The doc pages for
individual commands (e.g. :doc:`compute temp <compute_temp>`) also list
any accelerated variants available for that style.

To use an accelerator package in SPARTA, and one or more of the styles
it provides, follow these general steps:

using CMake from a build directory:

+---------------------------------+---------------------------------------------------------------------------------------+
| install the accelerator package | cmake -DPKG\_FFT=ON -DPKG\_KOKKOS=ON, etc                                             |
+---------------------------------+---------------------------------------------------------------------------------------+
| add compile/link flags          | cmake -C /path/to/sparta/cmake/presets/kokkos\_cuda.cmake -DKokkos\_ARCH\_PASCAL60=ON |
+---------------------------------+---------------------------------------------------------------------------------------+
| re-build SPARTA                 | make                                                                                  |
+---------------------------------+---------------------------------------------------------------------------------------+

Then do the following:

+----------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+
| prepare and test a regular SPARTA simulation                                                                               | lmp\_kokkos\_cuda -in in.script; mpirun -np 32 lmp\_kokkos\_cuda -in in.script |
+----------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+
| enable specific accelerator support via '-k on' :ref:`command-line switch <start_7>`,                                      | -k on g 1                                                                      |
+----------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+
| set any needed options for the package via "-pk" :ref:`command-line switch <start_7>` or :doc:`package <package>` command, | only if defaults need to be changed, -pk kokkos react/retry yes                |
+----------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+
| use accelerated styles in your input via "-sf" :ref:`command-line switch <start_7>` or :doc:`suffix <suffix>` command      | lmp\_kokkos\_cuda -in in.script -sf kk                                         |
+----------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+

Note that the first 3 steps can be done as a single command with
suitable make command invocations. This is discussed in :doc:`Section 4 <Section_packages>` of the manual, and its use is illustrated in
the individual accelerator sections.  Typically these steps only need
to be done once, to create an executable that uses one or more
accelerator packages.

The last 4 steps can all be done from the command-line when SPARTA is
launched, without changing your input script, as illustrated in the
individual accelerator sections.  Or you can add
:doc:`package <package>` and :doc:`suffix <suffix>` commands to your input
script.

The `Benchmark page <https://sparta.github.io/bench.html>`_ of the SPARTA
web site gives performance results for the various accelerator
packages for several of the standard SPARTA benchmark problems, as a
function of problem size and number of compute nodes, on different
hardware platforms.

Here is a brief summary of what the KOKKOS package provides.

* Styles with a "kk" suffix are part of the KOKKOS package, and can be
  run using OpenMP on multicore CPUs, on an NVIDIA GPU, on an AMD GPU,
  or on an Intel GPU.  The speed-up depends on a variety of
  factors, as discussed on the KOKKOS accelerator page.


The KOKKOS accelerator package doc page explains:

* what hardware and software the accelerated package requires
* how to build SPARTA with the accelerated package
* how to run with the accelerated package either via command-line switches or modifying the input script
* speed-ups to expect
* guidelines for best performance
* restrictions
