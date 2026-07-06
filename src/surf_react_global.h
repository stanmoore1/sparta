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

#ifdef SURF_REACT_CLASS

SurfReactStyle(global,SurfReactGlobal)

#else

#ifndef SPARTA_SURF_REACT_GLOBAL_H
#define SPARTA_SURF_REACT_GLOBAL_H

#include "surf_react.h"

namespace SPARTA_NS {

class SurfReactGlobal : public SurfReact {
 public:
  SurfReactGlobal(class SPARTA *, int, char **);
  SurfReactGlobal(class SPARTA *sparta) : SurfReact(sparta) {} // needed for Kokkos
  virtual ~SurfReactGlobal();
  virtual void init();
  int react(Particle::OnePart *&, int, double *, Particle::OnePart *&, int &);
  char *reactionID(int);
  double reaction_coeff(int) {return 0.0;};
  int match_reactant(char *, int) {return 1;}
  int match_product(char *, int) {return 1;}
  virtual void dynamic();

 protected:

  // a probability can be a fixed value (NUMERIC),
  //   an equal-style variable (VAREQUAL) giving one value per timestep, or
  //   a grid-style variable (VARGRID) giving a per-grid-cell value each step

  enum{NUMERIC,VARIABLE,VAREQUAL,VARGRID};

  double prob_create,prob_destroy;     // scalar probabilities (NUMERIC/VAREQUAL)

  int pdelete_mode,pcreate_mode;       // NUMERIC, VAREQUAL, or VARGRID
  char *pdelete_name,*pcreate_name;    // variable names (no v_ prefix)
  int pdelete_var,pcreate_var;         // variable indices, -1 if unused

  double *pdelete_grid,*pcreate_grid;  // per-cell probabilities for VARGRID
  int ngrid;                           // allocated length of grid arrays

  class RanKnuth *random;     // RNG for reaction probabilities

  void parse_prob(char *, int &, char *&, double &);
  void grow_grid();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

*/
