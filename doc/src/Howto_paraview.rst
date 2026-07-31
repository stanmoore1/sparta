.. _howto_16:

Visualizing SPARTA output with ParaView
=======================================

The *sparta/tools/paraview* directory contains two Python programs
that can be used to convert SPARTA surface and grid data to ParaView
*.pvd* format for visualization with ParaView:


.. parsed-literal::

   surf2paraview.py  
   grid2paraview.py

Note that you must have ParaView installed on your system to use these
scripts.  Installation and usage instructions follow.

These tools were written by Tom Otahal (Sandia), who can
be contacted at tjotaha at sandia.gov.

**\*\*Important\*\***

The ParaView *pvpython* interpreter must be used to run these Python scripts.  
Using a standard Python interpreter will not work, since the scripts will
not have access to the required ParaView Python modules and libraries.

**\*\*Important\*\***

(1) Getting Started

Download and install ParaView at `Kitware ParaView <https://www.paraview.org>`_

Binary installers are available for Linux, MacOS, and Windows.
Locate the *pvpython* binary in your ParaView installation.

On Linux:


.. parsed-literal::

   pvpython is in the bin/ directory of the extracted tar.gz file

On MacOS:


.. parsed-literal::

   pvpython is in /Applications/paraview.app/Contents/bin/

On Windows:


.. parsed-literal::

   pvpython is in C:\Program Files (x86)\ParaView 5.6.0\bin

-------------------------------

(2) Using surf2paraview.py

The *surf2paraview.py* program converts 3D SPARTA surface triangulation
files and 2D SPARTA closed polygon files into ParaView *.pvd* format.
Additionally, the program can optionally read one or more SPARTA
surface dump files and associate the calculated results with the
surface geometry over time.

The program has two required arguments:


.. parsed-literal::

   pvpython surf2paraview.py data.mir mir_surf

The first argument is the file name of a SPARTA surf file containing a
3d triangulation of an objects surface, or a 2d enclosed polygon of
line segments.  The second argument is the name of the resulting
ParaView output *.pvd* file.  The above command line will produce a file
called *mir\_surf.pvd* and a directory called *mir\_surf/*.  The *mir\_surf/*
directory contains a ParaView *.vtu* file with geometry information and
is referred to by the *mir\_surf.pvd* file.  Start ParaView and open the
file *mir\_surf.pvd* to visualize the surface.

The program has an optional argument to associate time result data
with the surface elements:


.. parsed-literal::

   pvpython surf2paraview.py data.mir mir_surf -r ../parent/mir/tmp_surf.\*

The *-r* (or *--result*\ ) option is followed by a list of file names with
full or relative paths to SPARTA surf dump files.  The files can be
over different time steps and from different processors at the same
time step. The script will organize the result files so that ParaView
can play a smooth animation over all time steps for the stored
variables in the file.  The example above uses a wild card character in
the file name to gather all of the *tmp\_surf.\** files stored in the
directory.  Wild card characters can only be used in the file name part
of the path and can be given for multiple paths.

.. note::

   SPARTA 2d enclosed polygons will be 2d outlines in ParaView.
   This means that any grid cells inside of the polygon will be visible
   in ParaView.  To obscure the inside of the enclosed polygon, select a
   Delaunay 2D filter from the ParaView menu.


.. parsed-literal::

     Filters->Alphabetical->Delaunay 2D

This will triangulate the interior of the polygon and obscure interior
grid cells from view.

The *-e* (or *--exodus*\ ) option will output the contents of the *\*.pvd* and
output directory in Exodus 2 output format as a single file:


.. parsed-literal::

   pvpython surf2paraview.py data.mir mir_surf -r ../parent/mir/tmp_surf.\* --exodus

This will produce an Exodus 2 file *mir\_surf.ex2*, containing the same content
as *mir\_surf.pvd* and *mir\_surf/*. The *.pvd* format output is not written when
Exodus 2 output is requested.

(3) Using grid2paraview.py

The *grid2paraview.py* program converts a text file description of a 2D
or 3D SPARTA mesh into a ParaView *.pvd* file.  Additionally, the
program can optionally read one or more SPARTA grid dump files and
associate the calculated results with the grid cells over time.

The program has two required arguments:


.. parsed-literal::

   pvpython grid2paraview.py mir.txt mir_grid

The first argument is a text file containing a description of the
SPARTA grid.  The description uses commands found in the SPARTA input
deck.  These commands are *dimension*\ , *create\_box*, and *create\_grid* or
*read\_grid*.  The file can also contain "slice" commands which will
define slice planes through the 3d grid and output 3d data for each
slice plane (crinkle cut).  The file can also contain comment lines
with start with a "#" character.

The dimension and create\_box command have exactly the same syntax as
corresponding SPARTA input script commands.  Both of these commands
must be used.

The grid itself can be defined by either a create\_grid or read\_grid
command, one of which must be used.  The create\_grid command is
similar to the SPARTA input script command with the same name, but it
only allows for use of the "level" keyword.  The other keywords that
specify processor assignments for cells are not allowed.  The
read\_grid command has the same syntax as the corresponding SPARTA
input script command, and reads a SPARTA parent grid file, which can
define a hierarchical grid with multiple levels of refinement.

One or more slice commands are optional.  Each defines a 2d plane
in the following manner


.. parsed-literal::

   slice Nx Ny Nz Px Py Pz

where (Nx,Ny,Nz) is the plane normal (need not be normalized) and
(Px,Py,Pz) is a point on the plane.  Note that the plane can be at any
orientation.  ParaView will perform a good interpolation from the 3d
grid cells to the 2d plane.

Each command will output a *\*.pvd* file with the plane normal encoded in
the *\*.pvd* file-name.

As an example, the *mir.txt* file specified above could contain the
following grid description:


.. parsed-literal::

     dimension           3
     create_box          -15.0 30.0 -20.0 15.0 -20.0 20.0
     create_grid         100 100 100 level 2 \* \* \* 2 2 2
     slice               1 0 0 0.0 0.0 0.0
     slice               0 1 0 0.0 0.0 0.0

The second argument for the *grid2paraview* command gives the name of
the resulting *.pvd* file.  The above command line will produce a file
called *mir\_grid.pvd* and a directory called *mir\_grid/*.  The *mir\_grid/*
directory contains all the ParaView *.vtu* files used to describe the
grid cell geometry.  The *mir\_grid.pvd* references the *mir\_grid/*
directory.  Open *mir\_grid.pvd* with ParaView to view the grid.

The program has an optional argument to associate time result data
with the grid cells:


.. parsed-literal::

   pvpython grid2paraview.py mir.txt mir_grid -r ../parent/mir/tmp_flow.\*

The *-r* (or *--result*\ ) option is followed by a list of file names with
full or relative paths to SPARTA grid dump files. This option operates
like the *-r* option in the *surf2paraview.py* program.

The grid description given in the *\*.txt* file must match the data given
in the grid flow files. The grid flow files must also contain a column
that gives the SPARTA encoded integer id for the cell.

For large grids (greater than 100x100x100), the time to write out the
*.pvd* file and data directory can be lengthy.  For this reason, the
*grid2paraview.py* command has three additional options which can break
the grid into smaller chunks at the top-most level of the grid.  Each
chunk will be written out as a separate *.vtu* file in the named
sub directory the *.pvd* file refers to.

These additional options are:


.. parsed-literal::

   -x (or --xchunk, default 100)
   -y (or --ychunk, default 100)
   -z (or --zchunk, default 100)

The program will launch a separate thread of computation for each grid
chunk.  On workstations with many cores and sufficient memory, using
small chunks (of about 1 million cells each) can greatly speed up
output time. For 2d grids, the *-zc* option is ignored.

.. note::

   On Windows platforms, the grid blocking will always be executed
   serially.  This is due to how the multiprocessing module is
   implemented on Windows, which prohibits multiple instances of *pvpython*
   from starting independently.

(4) pvbatch for Large SPARTA Grids

When SPARTA grid output becomes large, the processing time required for
grid2paraview.py can be long on a single node even with multi-processing.
If more than one compute node is available (HPC environment), grid2paraview.py
can be run with MPI using ParaView's *pvbatch* program. The *pvbatch* program
is normally located in the same directory as *pvpython*\ , along with the mpiexec
program that works with ParaView. In some environments, ParaView may have
been compiled from source with a particular version of MPI, in which case 
the appropriate mpiexec program will need to be used.

From the *mir.txt* example in section (3), to run *grid2paraview.py* using
*pvbatch*\ , use the following command line.


.. parsed-literal::

   mpiexec -np 256 pvbatch -sym grid2paraview.py mir.txt mir_grid -r ../parent/mir/tmp_flow.\*

This command will run grid2paraview.py on 256 MPI ranks and produce the same
outputs as the *pvpython* version. Using 256 MPI ranks will be faster than
multi-processing with threads on a single compute node. Notice the "-sym"
argument to *pvbatch*\ , which tells *pvbatch* to run in symmetric MPI mode.
This argument is required.

(5) Catalyst for Large SPARTA Grids

There is an option in *grid2paraview.py* to execute a ParaView Catalyst Python
script that has been exported from the ParaView GUI. For more details on
Catalyst, please see the Catalyst user guide, located here.

`Kitware ParaView Catalyst in-situ <https://www.paraview.org/in-situ/>`_

The Catalyst script will generate images or data extracts for each time-step.
This will avoid having to run ParaView as a separate step to generate
visualizations. The ideal work-flow is to run the ParaView GUI on a much smaller
grid version to setup the visualization and export the Catalyst script.
Then, run *grid2paraview.py* on the larger SPARTA grid output to generate
images. From the *mir.txt* example, to run *grid2paraview.py* using *pvbatch* and
Catalyst, use the following command line (\ *catalyst.py* was exported from
the ParaView GUI).


.. parsed-literal::

   mpiexec -np 32 pvbatch -sym grid2paraview.py mir.txt mir_grid -r -c catalyst.py ../parent/mir/tmp_flow.\*

This will generate images or data extracts, depending on how *catalyst.py* was
setup in the ParaView GUI. The *grid2paraview.py* script will not generate
ParaView grid geometry when the "-c" option is used. Note that *grid2paraview.py*
will assume that the grid input name is "mir\_grid.pvd" in *catalyst.py*\ , since
"mir\_grid" is given as the output directory.  If these two names do not match,
either edit your catalyst script or change the output directory name on the
command line to match what your script expects. The output directory is not 
created when *-c* option is used.

(6) Post-processing large refined SPARTA output grids

When SPARTA grids contain a large amount of grid refinement concentrated in
small areas of the grid, the tool *grid2paraview.py* tends to run out of memory
because it depends on a static distribution of cells to processors in terms of
grid chunks defined at the top level of the grid. To overcome this memory issue,
two new ParaView tools were developed:


.. parsed-literal::

   sort_sparta_grid_file.py and grid2paraview_cells.py

The program *sort\_sparta\_grid\_file.py* takes as input a SPARTA grid file and uses
the parallel bucket sort algorithm to sort the grid cells into the same number
of files as MPI ranks used to run the program.


.. parsed-literal::

   mpiexec -np 4 pvbatch -sym sort_sparta_grid_file.py data.grid

The program must be run using the ParaView *pvbatch* program with the
*-sym* argument.  The above command line will produce 4 output files
containing SPARTA grid dashed ids of cells located in the same area of
the grid. The output file names are based on the name of the *\*.grid*
file used as input (\ *data.grid* in this case). The output files will
be named as shown below.


.. parsed-literal::

   data_sort_bucket_rank_0.txt
   data_sort_bucket_rank_1.txt
   data_sort_bucket_rank_2.txt
   data_sort_bucket_rank_3.txt

The program *grid2paraview\_cells.py* takes similar inputs as the
*grid2paraview.py* program described in section (3), and produces the
same ParaView VTU file output and PVD file output.


.. parsed-literal::

   mpiexec -np 4 pvbatch -sym grid2paraview_cells.py grid.txt output -rf flow_files.txt --float --variables id f_1[5] f_1[7]

The program must be run using the ParaView *pvbatch* program with the
*-sym* argument.  The above command line will produce an *output.pvd*
file and a directory name output/ containing the ParaView VTU file
data. The *grid.txt* file must contain a *read\_grid* statement with
the path to a SPARTA grid cell output file, and is otherwise the same
as the *grid2paraview.py* version. The option *--float* outputs float
precision numbers to the VTU files to save memory (default is double
precision). The *--variables* option limits the output arrays to the
names given on the command line (default is all variable names found
in the flow files given by the *-rf* or *-r* options).

The *grid2paraview\_cells.py* program will look for
*\*\_sort\_bucket\_rank\_?.txt* files produced by the
sort\_sparta\_grid\_file.py program. The matching will depend on the
number of MPI ranks that *grid2paraview\_cells.py* is run on and the
name of the output directory given to *grid2paraview\_cells.py*. If
matching files are found, these will be used as input on each MPI
rank. If no match is found, *grid2paraview\_cells.py* will run
*sort\_sparta\_grid\_file.py* to produce sorted output files for each
rank. The programs are decoupled in this way to allow faster
*grid2paraview\_cells.py* runs once a set of sorted files has been
generated by *sort\_sparta\_grid\_file.py*.
