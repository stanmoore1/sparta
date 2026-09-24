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

/* ----------------------------------------------------------------------
   Ported from the LAMMPS STLReader class (src/stl_reader.h,
   src/stl_reader.cpp), which is used by the LAMMPS create_atoms mesh
   command.  The LAMMPS TextFileReader and ValueTokenizer classes that the
   original uses have no SPARTA equivalent, so the ASCII parser here reads
   and splits the lines itself, but the grammar it accepts, the text vs
   binary detection, and the error messages are all unchanged.
------------------------------------------------------------------------- */

#ifndef SPARTA_STL_READER_H
#define SPARTA_STL_READER_H

#include "stdio.h"
#include "pointers.h"
#include "spaexception.h"

#include <string>
#include <vector>

namespace SPARTA_NS {

// thrown by the STLReader class for malformed or unreadable files

class STLReaderException : public SpartaException {
 public:
  explicit STLReaderException(const std::string &msg) : SpartaException(msg) {}
};

/* ----------------------------------------------------------------------
   read triangle meshes from files in STL format

   an STL file stores a triangle mesh as a flat list of triangles, each of
   them an outward facing normal vector and the coords of its 3 vertices
   both variants of the format are supported, plain text ("ASCII") and
   binary, and which one a file is, is detected from its contents, not from
   its name

   most of the work is done by the static parse() method, which is
   independent of any SPARTA instance and reports all error conditions by
   throwing an STLReaderException, so the caller decides how to handle them

   read_file() is the alternative for the case where the mesh should be read
   by proc 0 only and then communicated to all the other procs.  It needs a
   class instance, aborts SPARTA when the file cannot be read, and stores
   the vertices in an array owned by the class instance.
------------------------------------------------------------------------- */

class STLReader : protected Pointers {
 public:

  // a single triangle of an STL mesh

  struct Triangle {
    double normal[3];      // outward facing normal vector of the facet
    double vert[3][3];     // x,y,z coords of the 3 vertices
  };

  STLReader(class SPARTA *);
  ~STLReader() override;

  int read_file(const char *, double **&);

  static std::vector<Triangle> parse(const std::string &, std::string * = NULL);

  // not part of the LAMMPS class, for callers that accept more than one
  // mesh format and have to decide which reader to use

  static bool is_stl_file(const std::string &);

 private:
  static std::vector<Triangle> parse_text(const std::string &, std::string &);
  static std::vector<Triangle> parse_binary(FILE *, std::string &);

  int ntris,maxtris;
  double **tris;
};

}

#endif

/* ERROR/WARNING messages:

E: Cannot open STL file %s

Self-explanatory.

E: File %s is not a valid ASCII STL file

The file does not start with the solid keyword and its size does not
match the layout of a binary STL file either.

E: Expected 'facet' or 'endsolid' in STL file, got: %s

The file does not follow the ASCII STL grammar.

E: Error reading vertex %d of facet in STL file

A vertex line inside an outer loop is missing or has too few coords.

E: Unexpected end of binary STL file ...

The file is truncated.

E: STL file %s has no triangles

The file was parsed successfully but contains no facets.

*/
