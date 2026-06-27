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

ComputeStyle(vremax/grid,ComputeVremaxGrid)

#else

#ifndef SPARTA_COMPUTE_VREMAX_GRID_H
#define SPARTA_COMPUTE_VREMAX_GRID_H

#include "compute.h"

namespace SPARTA_NS {

class ComputeVremaxGrid : public Compute {
 public:
  ComputeVremaxGrid(class SPARTA *, int, char **);
  ~ComputeVremaxGrid();
  void init();
  void compute_per_grid();
  bigint memory_usage();

 protected:
  int nglocal;

  void reallocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.

E: Compute vremax/grid requires a collide style be defined

The vremax data is owned by the collision model, so a collide command
must be issued before this compute is used.

*/
