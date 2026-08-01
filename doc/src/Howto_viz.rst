.. _howto_5:

Visualizing SPARTA snapshots
============================

The :doc:`dump image <dump_image>` command can be used to do on-the-fly
visualization as a simulation proceeds.  It works by creating a series
of JPG or PNG or PPM files on specified timesteps, as well as movies.
The images can include particles, grid cell quantities, and/or surface
element quantities.  This is not a substitute for using an interactive
visualization package in post-processing mode, but on-the-fly
visualization can be useful for debugging or making a high-quality
image of a particular snapshot of the simulation.

The :doc:`dump <dump>` command can be used to create snapshots of
particle, grid cell, or surface element data as a simulation runs.
These can be post-processed and read in to other visualization
packages.

A Python-based toolkit distributed by our group can read SPARTA
particle dump files with columns of user-specified particle
information, and convert them to various formats or pipe them into
visualization software directly.  See the `Pizza.py WWW site <pizza_>`_
for details.  Specifically, Pizza.py can convert SPARTA particle dump
files into PDB, XYZ, `Ensight <ensight_>`_, and VTK formats.  Pizza.py can
pipe SPARTA dump files directly into the Raster3d and RasMol
visualization programs.  Pizza.py has tools that do interactive 3d
OpenGL visualization and one that creates SVG images of dump file
snapshots.

Additional Pizza.py tools may be added that allow visualization of
surface and grid cell information as output by SPARTA.

.. _pizza: https://lammps.github.io/pizza/



.. _vmd: http://www.ks.uiuc.edu/Research/vmd



.. _ensight: https://www.ansys.com/products/fluids/ansys-ensight
