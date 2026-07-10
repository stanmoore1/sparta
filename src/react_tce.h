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
              double, double, double, double &, int &);

  double newtonTvib(int nmode, double Evib,
                      double VibTemp[],
                      double Tvib0,
                      double tol,
                      int nmax);

  double bird_dEvib(int nmode, double Tvib,
                  double VibTemp[]);

  double bird_Evib(int nmode, double Tvib,
                 double VibTemp[],
                 double Evib);
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

E: Reaction probability is NaN: gamma function pole, temperature
exponent is out of bounds

The argument of the gamma function in the TCE reaction probability is
a non-positive integer, so the reaction probability is not a number.
The temperature exponent in the reaction file must be within the
bounds printed as warnings when the reactions are initialized.

W: Reaction probability will be erroneous: non-positive gamma function
argument, temperature exponent is out of bounds

The argument of the gamma function in the TCE reaction probability is
non-positive, so the gamma function is negative or infinite and the
reaction probability is erroneous.  The temperature exponent in the
reaction file must be within the bounds printed as warnings when the
reactions are initialized.

W: Reaction ...: temperature exponent ... must be ...

The temperature exponent of the reaction is outside the exact bounds
of the TCE model for the energy and vibrational models in use, so the
reaction probability will be erroneous.  See the warning message for
the offending bound: the gamma function must be positive and finite,
the probability must vanish as the collision energy approaches the
activation energy, and the probability must not diverge as the
collision energy approaches zero or infinity.

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
