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
   Contributing author: ported from the LAMMPS VTK package (dump_vtk),
   which in turn came from LIGGGHTS (www.liggghts.com).
   Writes SPARTA particle data to VTK files (legacy .vtk, XML .vtp/.vtu,
   parallel .pvtp/.pvtu) for visualization in ParaView/VisIt.
------------------------------------------------------------------------- */

#ifdef DUMP_CLASS

DumpStyle(particle/vtk,DumpParticleVTK)

#else

#ifndef SPARTA_DUMP_PARTICLE_VTK_H
#define SPARTA_DUMP_PARTICLE_VTK_H

#include "dump_particle.h"
#include "vtk_writer.h"
#include <stdint.h>
#include <string>
#include <vector>

namespace SPARTA_NS {

class DumpParticleVTK : public DumpParticle {
 public:
  DumpParticleVTK(class SPARTA *, int, char **);
  virtual ~DumpParticleVTK();

 protected:
  int vtk_file_format;       // VTK, VTP, VTU, PVTP, PVTU (see cpp enum)

  int gx,gy,gz;              // buf column indices of the x,y,z point coords

  // one output data array per entry (point data)
  //   col   = starting buf column
  //   ncomp = 1 (scalar) or 3 (vector)
  //   type  = INT/BIGINT/DOUBLE (see cpp enum, matches Dump vtype codes)
  //   name  = array name in the VTK file

  struct VTKField {
    int col;
    int ncomp;
    int type;
    std::string name;
  };
  std::vector<VTKField> fields;

  int n_calls_;              // # of write_data() calls for current snapshot
  char *filecurrent;         // piece file name for this proc/timestep
  char *parallelfilecurrent; // .pvtp/.pvtu summary file name (rank 0 only)

  VTKWriter::Precision prec; // precision of floating point output
  int warned_precision;      // 1 after warning that single precision is coarse

  // data accumulated across write_data() calls, flushed once the filewriter
  // has received all nclusterprocs contributions.  points holds 3 coords per
  // particle; ddata/ldata hold one vector per entry in fields, in the same
  // order, with ncomp values per particle

  std::vector<double> points;
  std::vector<std::vector<double> > ddata;
  std::vector<std::vector<int64_t> > ldata;

  void init_style();
  void openfile();
  void write_header(bigint);
  void write_data(int, double *);
  int modify_param(int, char **);

  void setup_vtk_fields();
  void setFileCurrent();
  void buf2arrays(int, double *);
  void reset_vtk_data_containers();

  void write_file(int, double *);
  void write_pvtk(int);
  std::string pvtk_piece_filename(int);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Dump particle/vtk requires '*' in filename for per-timestep output

VTK writers create one file per snapshot, so the dump file name must
contain a '*' wildcard.

E: Dump particle/vtk legacy .vtk format does not support '%' in filename; use .vtu or .vtp

The legacy VTK format is single-file only.  Use an XML format (.vtu or
.vtp) for per-processor output.

E: Dump particle/vtk requires x, y, and z attributes

The x, y, z attributes define the VTK point coordinates and must be
included in the attribute list.

E: Dump particle/vtk does not support string attributes

Self-explanatory.

W: Dump particle/vtk in single precision resolves coordinates only to ...

dump_modify double no selected single precision, but the coordinates
are far enough from the origin relative to the size of the box that
single precision no longer resolves one part in a million of it.  Use
dump_modify double yes to write the coordinates as Float64.

*/
