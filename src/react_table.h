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

#ifdef REACT_CLASS

ReactStyle(table,ReactTable)

#else

#ifndef SPARTA_REACT_TABLE_H
#define SPARTA_REACT_TABLE_H

#include "react_bird.h"
#include "particle.h"

namespace SPARTA_NS {

/* ----------------------------------------------------------------------
   react table = tabulated reaction cross sections

   the probability that a selected collision reacts is

     P_react = sigma_react(E) / sigma_total(E)

   where sigma_total is the cross section the collide style actually used
   to select the pair.  this is the same relation the TCE model applies,
   but with sigma_react read from a file rather than derived from an
   Arrhenius fit, so an ab initio or measured reactive cross section can
   be used directly.

   the file format extends the Bird reaction file with a style letter T:

     N2 + O --> N + N + O
     D T <activation energy> <energy release> <file> <keyword>

   the two numbers are in Joules, as elsewhere in a reaction file.  the
   activation energy only gates energetically impossible collisions; the
   threshold behaviour comes from the tabulated cross section itself.
------------------------------------------------------------------------- */

class ReactTable : public ReactBird {
 public:
  ReactTable(class SPARTA *, int, char **);
  ~ReactTable();
  void init();
  int attempt(Particle::OnePart *, Particle::OnePart *, double, double,
              double, double &, int &);

 protected:
  class InterpTable **rtab;   // cross section table per reaction, or NULL
  char **tabfile,**tabkey;    // file and section keyword per reaction
  int maxtab;
  int warnflag;               // 1 once the probability>1 warning has fired

  int read_style(OneReaction *, char *);
  void read_coeffs(OneReaction *, char *, char *);
  void grow_tab(int);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: React table can only be used with a VSS-based collide style

The style needs the total cross section which was used to select the
collision pair, which only the vss and table collide styles provide.

E: React table does not currently support recombination reactions

Self-explanatory.

E: Invalid reaction coefficients in file

A tabulated reaction line must give an activation energy, an energy
release, a file name, and a section keyword.

E: React table requires every reaction to use style T

The style has no way to form a probability for an Arrhenius or Quantum
reaction, so a reaction file for it may not mix styles.

E: React table reaction has no cross section table

A reaction was declared with style T but no table was attached to it.

*/
