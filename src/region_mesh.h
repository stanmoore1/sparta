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

#include "region.h"

namespace SPARTA_NS {

class RegMesh : public Region {
 public:
  RegMesh(class SPARTA *, int, char **);
  ~RegMesh();
  int inside(double *);

 protected:
  int ntri;                  // # of triangles in the mesh
  double **tris;             // ntri x 9 = the 3 vertices of each triangle
  double *tnorm;             // length of the normal = 2x area of each triangle
  double origin[3];          // reference point for scale and rotate
  double bblo[3],bbhi[3];    // bounding box around the mesh
  double volume;             // volume enclosed by the mesh

  int nbiny,nbinz;           // # of ray-casting bins in y and z
  double invbiny,invbinz;    // inverse bin sizes
  int *binfirst;             // index in binlist of 1st tri in each bin, len nbin+1
  int *binlist;              // triangle indices, grouped by bin

  double epslen;             // tolerance for a point lying on a triangle
  double jitter;             // ray offset used to dodge a degenerate hit

  void translate(double, double, double);
  void scale(double, double, double);
  void rotate(double, double, double, double);
  void setup();
  void bin_range(int, int &, int &, int &, int &);
  int crossings(double, double, double);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: Region mesh has zero extent

All the triangles in the STL file are coincident, so the mesh encloses
no volume.

W: Region mesh surface is not watertight

The area-weighted normals of the triangles in the STL file do not sum
to zero, which means the surface has holes in it.  A point-in-mesh test
on a surface with holes gives arbitrary answers.

*/
