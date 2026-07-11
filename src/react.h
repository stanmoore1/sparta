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

#ifndef SPARTA_REACT_H
#define SPARTA_REACT_H

#include "pointers.h"
#include "particle.h"

namespace SPARTA_NS {

class React : protected Pointers {
 public:
  char *style;
  int nlist;                 // # of reactions read from file

  int recombflag;            // 1 if any recombination reactions defined
  int recombflag_user;       // 0 if user has turned off recomb reactions
  int recomb_species;        // species of 3rd particle in recomb reaction
  int computeChemRates;      // 1 if only computing a TCE rate without
                             // actually doing reaction

  int partialEnergy;         // 1 if using rDOF model, 0 if using all energy
                             // with partial_energy no, the TCE reaction
                             // energy is the TOTAL collision energy
                             // (trans+rot+vib+elec) and the energy factor
                             // of the probability is the microcanonical
                             // average over the reactants' discrete
                             // (vibrational x electronic) ladders at fixed
                             // total energy, with translation, rotation,
                             // and smooth vibration as the continuum:
                             // the equilibrium rate stays on the input
                             // Arrhenius rate for any mix of continuous
                             // and discrete internal modes, and out of
                             // equilibrium the rate responds to the actual
                             // internal-state populations

  double recomb_density;     // num density of particles in collision grid cell
  double recomb_boost;       // rate boost param for recombination reactions
  double recomb_boost_inverse;   // inverse of boost parameter
  Particle::OnePart *recomb_part3;  // ptr to 3rd particle in recomb reaction

  // support for detailed-balance (reverse) reactions, see ReactTCE

  int reverse_active;        // 1 if any reverse reactions need the cell
                             //   temperature at run time
  int reverse_auto;          // 1 to auto-generate a B-style reverse for
                             //   every eligible forward reaction
  char *keq_file;            // file of equilibrium-constant curve fits
                             //   used in place of the internal partition
                             //   functions, or NULL
  double tgas;               // representative cell temperature (K) used to
                             //   evaluate reverse-reaction rates; set per grid
                             //   cell by Collide before the collision loop

  int copy,uncopy,copymode;  // prevent deallocation of
                             //  base class when child copy is destroyed

  React(class SPARTA *, int, char **);
  React(class SPARTA *sparta) : Pointers(sparta) // needed for Kokkos
    { style = NULL; random = NULL; reverse_active = 0; reverse_auto = 0;
      keq_file = NULL; tgas = 0.0; }
  virtual ~React();
  virtual void init() {}
  virtual int recomb_exist(int, int) = 0;
  virtual void ambi_check() = 0;
  virtual int attempt(Particle::OnePart *, Particle::OnePart *,
                      double, double, double, double, double &, int &) = 0;
  virtual char *reactionID(int) = 0;
  virtual double extract_tally(int) = 0;

  void modify_params(int, char **);

 protected:
  class RanKnuth *random;
};

}

#endif
