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

ComputeStyle(tambi/grid,ComputeTambiGrid)

#else

#ifndef SPARTA_COMPUTE_TAMBI_GRID_H
#define SPARTA_COMPUTE_TAMBI_GRID_H

#include "compute.h"

namespace SPARTA_NS {

class ComputeTambiGrid : public Compute {
 public:
  ComputeTambiGrid(class SPARTA *, int, char **);
  ~ComputeTambiGrid();
  void init();
  virtual void compute_per_grid();
  virtual int query_tally_grid(int, double **&, int *&);
  virtual void post_process_grid(int, int, double **, int *, double *, int);
  virtual void reallocate();
  bigint memory_usage();

 protected:
  int groupbit;
  int npergroup;             // # of tally quantities (6)
  int ntotal;                // total # of columns in tally array
  int nglocal;               // # of owned grid cells

  int nmap1;                 // # of tally quantities the value uses
  int **map;                 // which tally columns the output value uses
  double **tally;            // array of tally quantities, cells by ntotal

  int index_ionambi,index_velambi;   // fix ambipolar custom attributes
  double emass;                      // ambipolar electron mass
  double tprefactor;                 // conversion from KE to temperature
};

}

#endif
#endif
