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

ReactStyle(tce/kk,ReactTCEKokkos)

#else

#ifndef SPARTA_REACT_TCE_KOKKOS_H
#define SPARTA_REACT_TCE_KOKKOS_H

#include "math.h"
#include "react_bird_kokkos.h"
#include "kokkos_type.h"
#include "update.h"
#include "math_const.h"

namespace SPARTA_NS {

using MathConst::MY_PI;

class ReactTCEKokkos : public ReactBirdKokkos {
 public:
  ReactTCEKokkos(class SPARTA *, int, char **);
  ReactTCEKokkos(class SPARTA* sparta) : ReactBirdKokkos(sparta) {copy = 1;}
  void init();
  int attempt(Particle::OnePart *, Particle::OnePart *,
              double, double, double, double, double &, int &) { return 0; }

/* ---------------------------------------------------------------------- */

enum{NONE,DISCRETE,SMOOTH};
enum{DISSOCIATION,EXCHANGE,IONIZATION,RECOMBINATION};   // other files

KOKKOS_INLINE_FUNCTION
int attempt_kk(Particle::OnePart *ip, Particle::OnePart *jp,
         double pre_etrans, double pre_erot, double pre_evib, double pre_eelec,
         double &post_etotal, int &kspecies,
         int &recomb_species, double &recomb_density, const t_species_1d_const &d_species,
         const double tgas_cell) const
{
  OneReactionKokkos *r;

  const int isp = ip->ispecies;
  const int jsp = jp->ispecies;

  const double pre_ave_rotdof = (d_species[isp].rotdof + d_species[jsp].rotdof)/2.0;

  const int n = d_reactions(isp,jsp).n;
  if (n == 0) return 0;
  auto& d_list = d_reactions(isp,jsp).d_list;

  // probablity to compare to reaction probability

  double react_prob = 0.0;
  rand_type rand_gen = rand_pool.get_state();
  const double random_prob = rand_gen.drand();
  rand_pool.free_state(rand_gen);

  // loop over possible reactions for these 2 species

  for (int i = 0; i < n; i++) {
    r = &d_rlist[d_list[i]];

    // ignore energetically impossible reactions

    const double pre_etotal = pre_etrans + pre_erot + pre_evib + pre_eelec;

    // two options for total energy in TCE model
    // 0: partialEnergy = true: rDOF model
    // 1: partialEnergy = false: TCE: Rotation + Vibration

    // average DOFs participating in the reaction

    double ecc,z;
    double e_excess = 0.0;

    if (partialEnergy) {
      ecc = pre_etrans;
      z = r->d_coeff[0];
      if (pre_ave_rotdof > 0.1)
        ecc += pre_erot*z/pre_ave_rotdof;
    } else {
      // total energy model, matching ReactTCE::attempt: ecc is the TOTAL
      // collision energy including electronic; z is the continuum
      // internal DOF (rotation, plus vibration when it is a continuous
      // mode); the discrete ladders enter through the tables below
      ecc = pre_etotal;
      z = pre_ave_rotdof;
      if (vibstyle == SMOOTH)
        z += (d_species[isp].vibdof + d_species[jsp].vibdof)/2.0;
    }

    // Cover cases where coeff[1].neq.coeff[4]

    if (r->d_coeff[1]>((-1)*r->d_coeff[4])) e_excess = ecc - r->d_coeff[1];
    else e_excess = ecc + r->d_coeff[4];
    if (e_excess <= 0.0) continue;

    // energy-dependent factor of the TCE probability: the tabulated
    // microcanonical average over the reactants' discrete vibrational
    // and electronic ladders, matching ReactBird::vib_micro_factor
    // exactly; reactions without discrete ladders use the standard
    // analytic factor

    double efactor;
    if (!partialEnergy && d_mtab_n[d_list[i]] > 0) {
      const int ir = d_list[i];
      const int n = d_mtab_n[ir];
      const double x = ecc/d_mtab_du[ir];
      const int k = (int) x;
      if (k >= n-1) efactor = d_mtab(ir,n-1);
      else {
        const double f = x - k;
        efactor = (1.0-f)*d_mtab(ir,k) + f*d_mtab(ir,k+1);
      }
    } else
      efactor = pow(ecc-r->d_coeff[1],r->d_coeff[3]-1+r->d_coeff[5]) *
                pow(1.0-r->d_coeff[1]/ecc,z+1.5-r->d_coeff[5]);

    // effective Arrhenius prefactor: for a reverse (detailed-balance)
    // RECOMBINATION, scale the seeded forward prefactor by the
    // partition-function ratio at the cell temperature, matching
    // ReactTCE::attempt; a reverse EXCHANGE needs no temperature (its
    // energy factor is the microcanonical detailed-balance table)

    double prefactor = r->d_coeff[2];
    if (r->reverse && r->type == RECOMBINATION) {
      if (tgas_cell > 0.0) prefactor *= reverse_scale_kk(r,d_species,tgas_cell);
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
        react_prob += prefactor * tgamma(z+2.5-r->d_coeff[5]) / tgamma(z+r->d_coeff[3]+1.5) *
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
        auto& d_sp2recomb = d_reactions(isp,jsp).d_sp2recomb;
        if (d_sp2recomb[recomb_species] != d_list[i]) continue;

        react_prob += recomb_boost * recomb_density * prefactor *
          tgamma(z+2.5-r->d_coeff[5]) / tgamma(z+r->d_coeff[3]+1.5) *
          efactor;   // extended to general recombination case with non-zero activation energy
        break;
      }

      //if (react_prob < 0) error->warning(FLERR,"Negative reaction probability");
      //else if (react_prob > 1) error->warning(FLERR,"Reaction probability greater than 1");

    default:
      //error->one(FLERR,"Unknown outcome in reaction");
      //d_error_flag() = 1;
      Kokkos::abort("ReactTCEKokkos: Unknown outcome in reaction\n");
      break;
    }

    // test against random number to see if this reaction occurs
    // if it does, reset species of I,J and optional K to product species
    // J particle can be destroyed in recombination reaction, set species = -1
    // K particle can be created in a dissociation or ionization reaction,
    //   set its kspecies, parent will create it
    // important NOTE:
    //   does not matter what order I,J reactants are in compared
    //     to order the reactants are listed in the reaction file
    //   for two reasons:
    //   a) list of N possible reactions above includes all reactions
    //      that I,J species are in, regardless of order
    //   b) properties of pre-reaction state, stored in precoln,
    //      as computed by setup_collision(),
    //      and used by perform_collision() after reaction has taken place,
    //      only store combined properties of I,J,
    //      nothing that is I-specific or J-specific

    if (react_prob > random_prob) {
      Kokkos::atomic_inc(&d_tally_reactions[d_list[i]]);
      if (!computeChemRates) {
        ip->ispecies = r->d_products[0];

        switch (r->type) {
        case DISSOCIATION:
        case IONIZATION:
        case EXCHANGE:
          {
            jp->ispecies = r->d_products[1];
            break;
          }
        case RECOMBINATION:
          {
            // always destroy 2nd reactant species

            jp->ispecies = -1;
            break;
          }
        }

        if (r->nproduct > 2) kspecies = r->d_products[2];
        else kspecies = -1;

        post_etotal = pre_etotal + r->d_coeff[4];

        // return reaction from 1 to N

        return d_list[i] + 1;
      }
    }
  }

  // no reaction performed

  return 0;
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   device twins of ReactTCE::partition_function / reverse_scale
   (issue #472 detailed-balance reverse reactions); must match the host
   implementations exactly for SPARTA_KOKKOS_EXACT
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double partition_function_kk(int isp, double T,
                             const t_species_1d_const &d_species) const
{
  const double kb = boltz;
  const double h = 6.62607015e-34;   // Planck constant (J s)

  double qtrans = pow(2.0*MY_PI*d_species[isp].mass*kb*T/(h*h), 1.5);

  double qrot = 1.0;
  if (d_species[isp].rotdof == 2 && d_species[isp].nrottemp >= 1 &&
      d_species[isp].rottemp[0] > 0.0)
    qrot = T / (d_species[isp].rotsymm * d_species[isp].rottemp[0]);
  else if (d_species[isp].rotdof == 3 && d_species[isp].nrottemp == 3 &&
           d_species[isp].rottemp[0] > 0.0 && d_species[isp].rottemp[1] > 0.0 &&
           d_species[isp].rottemp[2] > 0.0)
    qrot = sqrt(MY_PI*T*T*T /
                (d_species[isp].rotsymm*d_species[isp].rotsymm *
                 d_species[isp].rottemp[0]*d_species[isp].rottemp[1]*
                 d_species[isp].rottemp[2]));

  double qvib = 1.0;
  for (int m = 0; m < d_species[isp].nvibmode; m++) {
    if (d_species[isp].vibtemp[m] <= 0.0) continue;
    double x = exp(-d_species[isp].vibtemp[m]/T);
    int g = d_species[isp].vibdegen[m] > 0 ? d_species[isp].vibdegen[m] : 1;
    qvib *= pow(1.0/(1.0-x), g);
  }

  double qelec = 1.0;
  if (d_nelecstates.extent(0) > 0 && d_nelecstates[isp] > 0) {
    qelec = 0.0;
    for (int i = 0; i < d_nelecstates[isp]; i++)
      qelec += d_elecstates(isp,i).degen *
        exp(-d_elecstates(isp,i).temp/T);
  }

  return qtrans*qrot*qvib*qelec;
}

KOKKOS_INLINE_FUNCTION
double reverse_scale_kk(OneReactionKokkos *r,
                        const t_species_1d_const &d_species,
                        double T) const
{
  int nprod = r->nproduct;
  if (r->type == RECOMBINATION) nprod = 1;

  double num = 1.0, den = 1.0;
  for (int i = 0; i < nprod; i++)
    num *= partition_function_kk(r->d_products[i],T,d_species);
  for (int i = 0; i < r->nreactant; i++)
    den *= partition_function_kk(r->d_reactants[i],T,d_species);

  // forward temperature exponent applied at the cell temperature,
  // matching ReactTCE::reverse_scale

  return num/den * pow(T,r->reverse_bf);
}

 protected:
  int vibstyle;
  int elecstyle;
  double boltz;

  // per-species electronic ladders (partition functions of reverse
  // reactions), shared views owned by ParticleKokkos; zero-length views
  // when no electronic data is defined

  DAT::t_int_1d d_nelecstates;
  t_elecstate_2d d_elecstates;

  // per-reaction microcanonical energy-factor tables, device mirror of
  // the ReactBird host tables; n = 0 marks "no table"

  DAT::t_float_2d d_mtab;
  DAT::t_float_1d d_mtab_du;
  DAT::t_int_1d d_mtab_n;

  DAT::tdual_int_scalar k_error_flag;
  DAT::t_int_scalar d_error_flag;
  HAT::t_int_scalar h_error_flag;
};

}

#endif
#endif

/* ERROR/WARNING messages:

*/
