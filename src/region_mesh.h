/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.github.io
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#ifdef REGION_CLASS

RegionStyle(mesh,RegMesh)

#else

#ifndef SPARTA_REGION_MESH_H
#define SPARTA_REGION_MESH_H

#include "stdio.h"
#include "region.h"

namespace SPARTA_NS {

class RegMesh : public Region {
 public:
  RegMesh(class SPARTA *, int, char **);
  ~RegMesh();
  int inside(double *);

 protected:
  int meshdim;               // 2 if the mesh is lines, 3 if it is triangles
  int nelem;                 // # of lines or triangles in the mesh
  int nvert;                 // # of vertices = meshdim per element
  double **verts;            // nvert x 3 coords, meshdim in a row per element
  double *esize;             // 2x area of each tri, or length of each line
  double origin[3];          // reference point for scale and rotate
  double bblo[3],bbhi[3];    // bounding box around the mesh
  double enclosed;           // volume enclosed by the mesh, area if 2d

  int nbiny,nbinz;           // # of ray-casting bins in y and z
  double invbiny,invbinz;    // inverse bin sizes
  int *binfirst;             // index in binlist of 1st elem in each bin, nbin+1
  int *binlist;              // element indices, grouped by bin

  double epslen;             // tolerance for a point lying on the surface
  double jitter;             // ray offset used to dodge a degenerate hit

  void read_mesh(char *);
  void read_stl_file(char *);
  void read_surf_file(char *);
  void bcast_verts();
  void surf_error(const char *, const char *);
  double surf_numeric(const char *, const char *);
  int surf_inumeric(const char *, const char *);

  void translate(double, double, double);
  void scale(double, double, double);
  void rotate(double, double, double, double);
  void setup();
  void bin_range(int, int &, int &, int &, int &);
  int crossings(double, double, double);
  int crossings_line(double, double, int, int);
  int crossings_tri(double, double, double, int, int);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: Cannot open mesh file: %s

Self-explanatory.

E: Surf file does not contain lines or triangles: %s

The header of the file has neither a lines nor a triangles keyword, so
there is no surface to define the region with.

E: Surf file contains both lines and triangles: %s

A surf file defines either a 2d or a 3d surface, not both.

E: Incorrect point format in surf file: %s

A line of the Points section does not have 3 or 4 values on it.

E: Incorrect line format in surf file: %s
E: Incorrect triangle format in surf file: %s

A line of the Lines or Triangles section does not have a valid number of
values on it.  Per-surface custom attributes are not supported by the
region mesh style.

E: Invalid point index in surf file: %s

A line or triangle refers to a point that is not in the Points section.

E: Expected floating point value in surf file: %s
E: Expected integer value in surf file: %s

A value in the file could not be converted to a number.

E: Unexpected end of surf file: %s

The file is truncated.

E: Region mesh has zero extent

All the vertices in the file are coincident, so the mesh encloses
nothing.

W: Region mesh surface is not closed

The area-weighted normals of the surface elements in the file do not sum
to zero, which means the surface has gaps in it.  A point-in-mesh test
on a surface with gaps gives arbitrary answers.

*/
