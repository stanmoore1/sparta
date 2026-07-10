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

ReactStyle(tce,ReactTCE)

#else

#ifndef SPARTA_REACT_TCE_H
#define SPARTA_REACT_TCE_H

#include "react_bird.h"
#include "particle.h"

namespace SPARTA_NS {

class ReactTCE : public ReactBird {
 public:
  ReactTCE(class SPARTA *, int, char **);
  void init();
  int attempt(Particle::OnePart *, Particle::OnePart *,
              double, double, double, double, double &, int &);

  // detailed-balance backward reaction rates (issue #472)

  double partition_function(int isp, double T);
  double reverse_scale(OneReaction *r);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: React tce can only be used with collide vss

Self-explanatory.

E: Ionization and recombination reactions are not yet implemented

This error conditions will be removed after those reaction styles are
fully implemented.

E: Unknown outcome in reaction

The specified type of the reaction is not encoded in the reaction
style.

E: Reaction ...: temperature exponent ... must be > ..., else the
gamma function is negative or infinite and the reaction probability
is erroneous or NaN

The argument of the gamma function in the TCE reaction probability is
non-positive for this reaction, so the probability is certain to be
erroneous (negative, infinite, or NaN).  The temperature exponent in
the reaction file must be within the printed bound for the energy and
vibrational models in use.

W: Reaction ...: temperature exponent ... must be ...

The temperature exponent of the reaction is outside the exact bounds
of the TCE model for the energy and vibrational models in use, so the
reaction probability will be erroneous.  See the warning message for
the offending bound: the probability must vanish as the collision
energy approaches the activation energy, and must not diverge as the
collision energy approaches infinity.

E: Cannot open reaction file %s

Self-explanatory.

E: Invalid reaction formula in file

Self-explanatory.

E: Invalid reaction type in file

Self-explanatory.

E: Invalid reaction style in file

Self-explanatory.

E: Invalid reaction coefficients in file

Self-explanatory.

*/
