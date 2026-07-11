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

  // custom electronic energy of the 3rd particle, needed by the 3-body
  // detailed-balance probability of a reverse recombination

  index_eelec = -1;
  if (collide->elecstyle == DISCRETE)
    index_eelec = particle->find_custom((char *) "eelec");

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

    // a reverse (B-style) recombination without an external Keq fit is
    // fully microcanonical: the 3rd particle's energy counts toward the
    // barrier (its probability is resolved in the total available
    // energy), so the pair-energy threshold below does not apply to it

    int micro3 = 0;
    if (r->reverse && r->type == RECOMBINATION && !r->keq_flag &&
        mtab && mtab[list[i]] && mtab_num[list[i]]) micro3 = 1;

    // Cover cases where coeff[1].neq.coeff[4]

    if (r->coeff[1]>((-1)*r->coeff[4])) e_excess = ecc - r->coeff[1];
    else e_excess = ecc + r->coeff[4];
    if (e_excess <= 0.0 && !micro3) continue;

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

    double efactor = 0.0;
    if (!micro3) {   // a micro3 reaction uses its own 3-body factor below
      if (!partialEnergy && mtab && mtab[list[i]])
        efactor = vib_micro_factor(list[i],ecc);
      else
        efactor = pow(ecc-r->coeff[1],r->coeff[3]-1+r->coeff[5]) *
                  pow(1.0-r->coeff[1]/ecc,z+1.5-r->coeff[5]);
    }

    // effective Arrhenius prefactor (issue #472): a reverse reaction
    // matched to an external Keq curve fit (react_modify keq_file)
    // scales the seeded forward prefactor by k_f/Keq_fit evaluated at
    // the local cell temperature React::tgas; if no valid cell
    // temperature is available (fewer than 2 particles in the cell),
    // skip the reaction for this collision.  All other reverse
    // reactions need no temperature: they are handled by the
    // microcanonical detailed-balance tables built at init.

    double prefactor = r->coeff[2];
    if (r->reverse && r->keq_flag) {
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

        // fully microcanonical reverse recombination (issue #472): the
        // probability is resolved in the total available energy
        //   w = ecc + eps_t + erot3 + evib3 + eelec3
        // where eps_t is the 3rd particle's translational energy
        // relative to the pair's center of mass; dividing the
        // detailed-balance numerator table by the pair's density of
        // states and the 3rd particle's continuum density weights makes
        // the thermal average reproduce k_f(T)/K_eq(T) at every
        // temperature (see ReactBird::build_db3_table); no cell
        // temperature is used

        if (micro3) {
          Particle::OnePart *p3 = recomb_part3;
          int sp3 = p3->ispecies;

          double mi = species[isp].mass;
          double mj = species[jsp].mass;
          double m3 = species[sp3].mass;
          double divisor = 1.0/(mi + mj);
          double *vi = ip->v;
          double *vj = jp->v;
          double *v3 = p3->v;
          double du3 = v3[0] - (mi*vi[0] + mj*vj[0])*divisor;
          double dv3 = v3[1] - (mi*vi[1] + mj*vj[1])*divisor;
          double dw3 = v3[2] - (mi*vi[2] + mj*vj[2])*divisor;
          double mu3 = m3*(mi + mj)/(m3 + mi + mj);
          double eps_t = 0.5*mu3*(du3*du3 + dv3*dv3 + dw3*dw3);

          double eelec3 = 0.0;
          if (index_eelec >= 0 && species[sp3].elecdat)
            eelec3 = particle->edvec[particle->ewhich[index_eelec]]
              [p3 - particle->particles];

          // continuum density weights of the 3rd particle's energies,
          // matching the flat-measure dimension of the table; skip the
          // attempt at the (measure-zero) singular points

          double c3 = sqrt(eps_t);
          int rotdof3 = species[sp3].rotdof;
          if (rotdof3 > 0 && rotdof3 != 2)
            c3 *= pow(p3->erot,0.5*rotdof3-1.0);
          int vibdof3 = species[sp3].vibdof;
          if (collide->vibstyle == SMOOTH && vibdof3 > 0 && vibdof3 != 2)
            c3 *= pow(p3->evib,0.5*vibdof3-1.0);
          if (!(c3 > 0.0)) continue;

          double xpair = vib_micro_factor(list[i],ecc);
          if (!(xpair > 0.0)) continue;

          double w = ecc + eps_t + p3->erot + p3->evib + eelec3;
          react_prob += recomb_boost * recomb_density *
            db3_num_factor(list[i],w) / (xpair*c3);
          break;
        }

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
   prefactor scale of a reverse reaction matched to an external
   equilibrium-constant curve fit (react_modify keq_file):
   k_b = k_f/Keq_fit at the local cell temperature React::tgas; the
   exponential shift reverse_dEa restates the forward barrier relative
   to the seeded backward barrier so the standard TCE energy factor
   stays in place.  reverse reactions without a Keq fit never call
   this: they are handled by the temperature-free microcanonical
   detailed-balance tables (ReactBird::build_db_table and
   build_db3_table)  (issue #472)
------------------------------------------------------------------------- */

double ReactTCE::reverse_scale(OneReaction *r)
{
  return pow(tgas,r->reverse_bf) *
    exp(-r->reverse_dEa/(update->boltz*tgas)) /
    keq_eval(r->keq_coeff,tgas);
}
