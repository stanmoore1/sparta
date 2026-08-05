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
   Contributing author: extends the LAMMPS VTK dump concept to SPARTA grid
   cells.  Each owned in-group cell is written as a VTK voxel (3d) or pixel
   (2d) in an unstructured grid, with requested attributes as cell data.
------------------------------------------------------------------------- */

#ifdef DUMP_CLASS

DumpStyle(grid/vtk,DumpGridVTK)

#else

#ifndef SPARTA_DUMP_GRID_VTK_H
#define SPARTA_DUMP_GRID_VTK_H

#include "dump_grid.h"
#include "vtk_writer.h"
#include <stdint.h>
#include <string>
#include <vector>

namespace SPARTA_NS {

class DumpGridVTK : public DumpGrid {
 public:
  DumpGridVTK(class SPARTA *, int, char **);
  virtual ~DumpGridVTK();

 protected:
  int vtk_file_format;       // VTK, VTU, PVTU (see cpp enum)

  int ndata;                 // # of user data columns (= original nfield)
  int ngeom;                 // # of appended geometry columns (6 3d / 4 2d)
  int celltype;              // VTK_VOXEL or VTK_PIXEL
  int ncorner;               // 8 (3d) or 4 (2d) corners per cell

  // one cell-data array per user data column
  struct VTKField { int col; int type; std::string name; };
  std::vector<VTKField> fields;

  int n_calls_;
  char *filecurrent;
  char *parallelfilecurrent;

  VTKWriter::Precision prec; // precision of floating point output
  int warned_precision;      // 1 after warning that single precision is coarse

  // data accumulated across write_data() calls, flushed once the filewriter
  // has received all nclusterprocs contributions.  points holds 3 coords per
  // cell corner; ddata/ldata/sdata hold one vector per entry in fields, in
  // the same order, with one value per cell

  std::vector<double> points;
  std::vector<std::vector<double> > ddata;
  std::vector<std::vector<int64_t> > ldata;
  std::vector<std::vector<std::string> > sdata;

  void init_style();
  void openfile();
  void write_header(bigint);
  void pack();
  void write_data(int, double *);
  int modify_param(int, char **);

  void augment_geometry_fields();
  void setup_vtk_fields();
  void setFileCurrent();
  void buf2arrays(int, double *);
  void reset_vtk_data_containers();

  void write_file(int, double *);
  void write_pvtk();
  std::string pvtk_piece_filename(int);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Dump grid/vtk requires '*' in filename for per-timestep output

VTK writers create one file per snapshot, so the dump file name must
contain a '*' wildcard.

E: Dump grid/vtk does not support .vtp; grid cells are volumetric, use .vtu

Grid cells are written as volumetric VTK voxels/pixels, which require
the unstructured (.vtu) or legacy (.vtk) format.

E: Dump grid/vtk legacy .vtk format does not support '%' in filename; use .vtu

The legacy VTK format is single-file only.  Use .vtu for per-processor
output.

W: Dump grid/vtk in single precision resolves coordinates only to ...

dump_modify double no selected single precision, but the coordinates
are far enough from the origin relative to the size of the box that
single precision no longer resolves one part in a million of it.  Use
dump_modify double yes to write the coordinates as Float64.

*/
