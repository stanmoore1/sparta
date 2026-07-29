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

#ifdef COMPUTE_CLASS

ComputeStyle(vdf/grid,ComputeVDFGrid)

#else

#ifndef SPARTA_COMPUTE_VDF_GRID_H
#define SPARTA_COMPUTE_VDF_GRID_H

#include "compute.h"

namespace SPARTA_NS {

class ComputeVDFGrid : public Compute {
 public:
  ComputeVDFGrid(class SPARTA *, int, char **);
  ~ComputeVDFGrid();
  void init();
  void compute_per_grid();
  void reallocate();
  bigint memory_usage();

 protected:
  int groupbit,imix,nvalue;
  int ngroup;
  int oobstyle;              // IGNORE or CLAMP for out-of-range samples

  int *value;                // keyword for each user requested value
  int *nbin;                 // # of bins for each user requested value
  double *lo,*hi;            // bin range for each user requested value
  double *invdelta;          // nbin/(hi-lo) for each user requested value
  int *binoffset;            // 1st column of each value within a group block

  int nbintotal;             // # of columns per mixture group
  int ntotal;                // total # of columns = ngroup*nbintotal
  int nglocal;               // # of owned grid cells

  int needmass;              // 1 if any value requires the species mass

  int weightflag;            // 1 to tally the particle weight, 0 to tally counts
  int cellweightflag;        // 1 if cell weighting is enabled
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: Compute vdf/grid group ID does not exist

Self-explanatory.

E: Compute vdf/grid mixture ID does not exist

Self-explanatory.

E: Compute vdf/grid Nbin must be > 0

Self-explanatory.

E: Compute vdf/grid bin range must have lo < hi

Self-explanatory.

E: Invalid compute vdf/grid value or optional keyword

Self-explanatory.

E: Invalid compute vdf/grid optional keyword

Self-explanatory.

E: Number of groups in compute vdf/grid mixture has changed

This mixture property cannot be changed after this compute command is
issued.

E: Must use compute vdf/grid/kk if Kokkos is enabled

The host copy of the particle list this compute reads is not guaranteed
to be current in a Kokkos run, so the Kokkos version must be used.  The
-sf kk command-line switch selects it automatically.

*/
