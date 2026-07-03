/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.sandia.gov
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(gas/attempt/grid,ComputeGasAttemptGrid)

#else

#ifndef SPARTA_COMPUTE_GAS_ATTEMPT_GRID_H
#define SPARTA_COMPUTE_GAS_ATTEMPT_GRID_H

#include "compute.h"
#include "grid.h"

namespace SPARTA_NS {

class ComputeGasAttemptGrid : public Compute {
 public:
  ComputeGasAttemptGrid(class SPARTA *, int, char **);
  ~ComputeGasAttemptGrid();
  void init();
  void compute_per_grid();
  void clear();
  void attempt_tally(int, int);
  int attempt_tally_only() {return 1;}
  bigint memory_usage();

 protected:
  int groupbit;

  int nglocal;
  Grid::ChildInfo *cinfo;    // local copy

  void reallocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.

*/
