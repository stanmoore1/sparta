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

#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "react_tce.h"
#include "particle.h"
#include "collide.h"
#include "update.h"
#include "random_knuth.h"
#include "math_const.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

enum{NONE,DISCRETE,SMOOTH};
enum{DISSOCIATION,EXCHANGE,IONIZATION,RECOMBINATION};   // other files

/* ---------------------------------------------------------------------- */

ReactTCE::ReactTCE(SPARTA *sparta, int narg, char **arg) :
  ReactBird(sparta, narg, arg) {}

/* ---------------------------------------------------------------------- */

void ReactTCE::init()
{
  if (!collide || strcmp(collide->style,"vss") != 0)
    error->all(FLERR,"React tce can only be used with collide vss");

  ReactBird::init();

  // error/warn if the temperature exponent of any reaction is out of
  //   bounds for the TCE reaction probability

  check_tce_bounds();

  // reverse exchange reactions are implemented by microcanonical
  // detailed-balance tables, which are built on the total-energy model

  if (partialEnergy)
    for (int i = 0; i < nlist; i++)
      if (rlist[i].active && rlist[i].reverse)
        error->all(FLERR,"Reverse (B-style) reactions require "
                   "react_modify partial_energy no");

  // with partial_energy no, the TCE energy factor is the microcanonical
  // average over the reactants' discrete ladders (SHO vibrational levels
  // when vibrate discrete, electronic states when electronic discrete):
  // build the per-reaction lookup tables; reactions whose reactants carry
  // no discrete ladders fall back to the standard analytic factor, which
  // is the same average with an empty ladder

  if (!partialEnergy) build_micro_tables();
  else free_micro_tables();
}

/* ---------------------------------------------------------------------- */

int ReactTCE::attempt(Particle::OnePart *ip, Particle::OnePart *jp,
                      double pre_etrans, double pre_erot, double pre_evib, double pre_eelec,
                      double &post_etotal, int &kspecies)
{
  double pre_etotal,ecc,e_excess,z;
  OneReaction *r;

  Particle::Species *species = particle->species;
  int isp = ip->ispecies;
  int jsp = jp->ispecies;

  double pre_ave_rotdof = (species[isp].rotdof + species[jsp].rotdof)/2.0;

  int n = reactions[isp][jsp].n;
  if (n == 0) return 0;
  int *list = reactions[isp][jsp].list;

  // probablity to compare to reaction probability

  double react_prob = 0.0;
  double random_prob = random->uniform();

  // loop over possible reactions for these 2 species

  for (int i = 0; i < n; i++) {
    r = &rlist[list[i]];

    // ignore energetically impossible reactions

    pre_etotal = pre_etrans + pre_erot + pre_evib + pre_eelec;

    // two options for total energy in TCE model
    // 0: partialEnergy = true: rDOF model
    // 1: partialEnergy = false: TCE: Rotation + Vibration

    // average DOFs participating in the reaction

    if (partialEnergy) {
       ecc = pre_etrans;
       z = r->coeff[0];
       if (pre_ave_rotdof > 0.1) ecc += pre_erot*z/pre_ave_rotdof;
    } else {
       // total energy model: ecc is the TOTAL collision energy including
       // the pair's electronic energy; z is the continuum internal DOF
       // (rotation, plus vibration when it is a continuous mode); the
       // discrete vibrational and electronic ladders enter the reaction
       // probability through the microcanonical energy-factor tables below
       ecc = pre_etotal;
       z = pre_ave_rotdof;
       if (collide->vibstyle == SMOOTH)
         z += (species[isp].vibdof + species[jsp].vibdof)/2.0;
    }

    // Cover cases where coeff[1].neq.coeff[4]

    if (r->coeff[1]>((-1)*r->coeff[4])) e_excess = ecc - r->coeff[1];
    else e_excess = ecc + r->coeff[4];
    if (e_excess <= 0.0) continue;

    // energy-dependent factor of the TCE probability:
    // the microcanonical (density-of-states weighted) average of the
    // standard factor over the reactants' discrete vibrational and
    // electronic ladders at fixed total collision energy, tabulated per
    // reaction at init; reactions whose reactants carry no discrete
    // ladders use the standard analytic factor
    //   (ecc-Ea)^(eta-1+omega) * (1-Ea/ecc)^(z+1.5-omega)
    // which is the same average with an empty ladder. The average keeps
    // the equilibrium rate on the input Arrhenius rate while the actual
    // internal states of the colliding pair count toward the barrier.

    double efactor;
    if (!partialEnergy && mtab && mtab[list[i]])
      efactor = vib_micro_factor(list[i],ecc);
    else
      efactor = pow(ecc-r->coeff[1],r->coeff[3]-1+r->coeff[5]) *
                pow(1.0-r->coeff[1]/ecc,z+1.5-r->coeff[5]);

    // effective Arrhenius prefactor (issue #472): for a reverse
    // (detailed-balance) RECOMBINATION, scale the seeded forward
    // prefactor by the partition-function ratio and forward temperature
    // exponent evaluated at the local cell temperature React::tgas; if
    // no valid cell temperature is available (fewer than 2 particles in
    // the cell), skip the reaction for this collision.  A reverse
    // EXCHANGE needs no temperature: its energy factor is the
    // microcanonical detailed-balance table built at init.

    double prefactor = r->coeff[2];
    if (r->reverse && (r->type == RECOMBINATION || r->keq_flag)) {
      if (tgas > 0.0) prefactor *= reverse_scale(r);
      else continue;
    }

    // compute probability of reaction
    // gamma function denominator is negative or infinite (erroneous
    //   probability) if the temperature exponent is out of bounds,
    //   checked at init by ReactBird::check_tce_bounds()

    switch (r->type) {
    case DISSOCIATION:
    case IONIZATION:
    case EXCHANGE:
      {
        react_prob += prefactor * tgamma(z+2.5-r->coeff[5]) / tgamma(z+r->coeff[3]+1.5) *
          efactor;
        break;
      }

    case RECOMBINATION:
      {
        // skip if no 3rd particle chosen by Collide::collisions()
        //   this includes effect of boost factor to skip recomb reactions
        // check if this recomb reaction is the same one
        //   that the 3rd particle species maps to, else skip it
        // this effectively skips all recombinations reactions
        //   if selected a 3rd particle species that matches none of them
        // scale probability by boost factor to restore correct stats

        if (recomb_species < 0) continue;
        int *sp2recomb = reactions[isp][jsp].sp2recomb;
        if (sp2recomb[recomb_species] != list[i]) continue;

        react_prob += recomb_boost * recomb_density * prefactor *
          tgamma(z+2.5-r->coeff[5]) / tgamma(z+r->coeff[3]+1.5) *
          efactor;   // extended to general recombination case with non-zero activation energy
        break;
      }

    if (react_prob < 0) error->warning(FLERR,"Negative reaction probability");
    else if (react_prob > 1) error->warning(FLERR,"Reaction probability greater than 1");

    default:
      error->one(FLERR,"Unknown outcome in reaction");
      break;
    }

    // test against random number to see if this reaction occurs
    // if it does, reset species of I,J and optional K to product species
    // J particle is destroyed in recombination reaction, set species = -1
    // K particle can be created in a dissociation or ionization reaction,
    //   set its kspecies, parent will create it
    // important NOTE:
    //   does not matter what order I,J reactants are in compared
    //     to order the reactants are listed in the reaction file
    //   for two reasons:
    //   a) list of N possible reactions above includes all reactions
    //      that I,J species are in, regardless of order
    //   b) properties of pre-reaction state are stored in precoln:
    //      computed by setup_collision()
    //      used by perform_collision() after reaction has taken place
    //      precoln only stores combined properties of I,J
    //      nothing that is I-specific or J-specific

    if (react_prob > random_prob) {
      tally_reactions[list[i]]++;

      if (!computeChemRates) {
        ip->ispecies = r->products[0];

        switch (r->type) {
        case DISSOCIATION:
        case IONIZATION:
        case EXCHANGE:
          {
            jp->ispecies = r->products[1];
            break;
          }
        case RECOMBINATION:
          {
            // always destroy 2nd reactant species

            jp->ispecies = -1;
            break;
          }
        }

        if (r->nproduct > 2) kspecies = r->products[2];
        else kspecies = -1;

        post_etotal = pre_etotal + r->coeff[4];

        // return reaction from 1 to N

        return list[i] + 1;
      }
    }
  }

  // no reaction performed

  return 0;
}

/* ----------------------------------------------------------------------
   partition-function ratio that converts the seeded forward Arrhenius
   prefactor into the backward prefactor by detailed balance,
     A_B / A_F = q_reactants,forward / q_products,forward
               = q(reverse products) / q(reverse reactants)
   evaluated at the local cell temperature React::tgas.
   for a dissociation/recombination pair the product and reactant counts
   differ by one, so the ratio carries one net translational factor and
   has units of volume, converting the forward m^3/s prefactor into the
   backward m^6/s recombination prefactor (issue #472)
------------------------------------------------------------------------- */

double ReactTCE::reverse_scale(OneReaction *r)
{
  // external equilibrium-constant curve fit (react_modify keq_file):
  // k_b = k_f/Keq_fit at the cell temperature; the exponential shift
  // reverse_dEa restates the forward barrier relative to the seeded
  // backward barrier so the standard TCE energy factor stays in place

  if (r->keq_flag)
    return pow(tgas,r->reverse_bf) *
      exp(-r->reverse_dEa/(update->boltz*tgas)) /
      keq_eval(r->keq_coeff,tgas);

  // for a B-style recombination A + B -> AB + M, the third body M
  // (products[1]) is a spectator whose partition function appears on both
  // sides of the paired dissociation and cancels: skip it

  int nprod = r->nproduct;
  if (r->type == RECOMBINATION) nprod = 1;

  double num = 1.0, den = 1.0;
  for (int i = 0; i < nprod; i++)
    num *= partition_function(r->products[i],tgas);
  for (int i = 0; i < r->nreactant; i++)
    den *= partition_function(r->reactants[i],tgas);

  // the forward temperature exponent is applied here at the cell
  // temperature (the seeded backward exponent is 0), so the backward TCE
  // form stays integrable for any forward b (see ReactBird::init)

  return num/den * pow(tgas,r->reverse_bf);
}
