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

namespace SPARTA_NS {

class ReactTCEKokkos : public ReactBirdKokkos {
 public:
  ReactTCEKokkos(class SPARTA *, int, char **);
  ReactTCEKokkos(class SPARTA* sparta) : ReactBirdKokkos(sparta) {copy = 1;}
  void init();
  int attempt(Particle::OnePart *, Particle::OnePart *,
              double, double, double, double &, int &) {return 0;}

/* ---------------------------------------------------------------------- */

// the Newton solve for the vibrational temperature is done in double
//   precision: exp(vibtemp/Tvib) overflows single precision for
//   vibtemp/Tvib > 88, and the absolute tolerance on Tvib is below single
//   precision resolution for Tvib > 1000 K; it is called only when a
//   reaction is attempted

typedef double tce_float;  // KK_DOUBLE: see comment above

KOKKOS_INLINE_FUNCTION
tce_float bird_Evib(const int& nmode, const tce_float& Tvib,
                 const double vibtemp[],  // KK_DOUBLE: Species data is double
                 const tce_float& Evib) const
{
  // Comutes f for Newton's search method outlined in newtonTvib()

  tce_float f = -Evib;
  const tce_float kb = boltz;

  for (int i = 0; i < nmode; i++) {
    const tce_float vti = vibtemp[i];
    f += (((kb*vti)/(Kokkos::exp(vti/Tvib)-1)));
  }

  return f;
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
tce_float bird_dEvib(const int& nmode, const tce_float& Tvib, const double vibtemp[]) const  // KK_DOUBLE: Species data is double
{
  // Comutes df for Newton's search method

  tce_float df = 0.0;
  const tce_float kb = boltz;

  for (int i = 0; i < nmode; i++) {
    const tce_float vti = vibtemp[i];
    const tce_float vti2 = vti * vti;
    const tce_float Tvib2 = Tvib * Tvib;
    const tce_float k1 = vti/Tvib;
    const tce_float ek1 = Kokkos::exp(k1);
    const tce_float k2 = ek1 - static_cast<tce_float>(1.0);
    const tce_float k22 = k2 * k2;
    df += (vti2*kb*ek1)/(Tvib2*k22);
  }

  return df;
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
tce_float newtonTvib(const int &nmode, const tce_float& Evib, const double vibTemp[],  // KK_DOUBLE: Species data is double
               const tce_float &Tvib0,
               const tce_float &tol,
               const int& nmax) const
{
  // Function for converting vibrational energy to vibrational temperature
  // Computes Tvib assuming the vibrational energy levels occupy a simple harmonic oscillator (SHO) spacing
  // Search for Tvib begins at some initial value "Tvib0" until the search reaches a tolerance level "tol"

  // Uses Newton's method to solve for a vibrational temperature given a
  // distribution of vibrational energy levels

  tce_float Tvib_prev;

  // f and df are computed for Newton's search
  tce_float f = bird_Evib(nmode,Tvib0,vibTemp,Evib);
  tce_float df = bird_dEvib(nmode,Tvib0,vibTemp);

  // Update guess for Tvib and compute error
  tce_float Tvib = Tvib0 - (f/df);
  tce_float err = Kokkos::fabs(Tvib-Tvib0);

  int i = 2;

  // Continue to search for Tvib until the error is greater than the tolerance:
  while((err >= tol) && (i <= nmax))
  {
    Tvib_prev = Tvib;

    f = bird_Evib(nmode,Tvib,vibTemp,Evib);
    df = bird_dEvib(nmode,Tvib,vibTemp);

    Tvib = Tvib_prev-(f/df);
    err = Kokkos::fabs(Tvib-Tvib_prev);

    i++;
  }

  return Tvib;
}
/* ---------------------------------------------------------------------- */

enum{NONE,DISCRETE,SMOOTH};
enum{DISSOCIATION,EXCHANGE,IONIZATION,RECOMBINATION};   // other files

KOKKOS_INLINE_FUNCTION
int attempt_kk(OnePartKK *ip, OnePartKK *jp,
         KK_FLOAT pre_etrans, KK_FLOAT pre_erot, KK_FLOAT pre_evib,
         KK_FLOAT &post_etotal, int &kspecies,
         int &recomb_species, double &recomb_density, const t_species_1d_const &d_species) const  // KK_DOUBLE: recombination density
{
  OneReactionKokkos *r;

  const int isp = ip->ispecies;
  const int jsp = jp->ispecies;
  const KK_FLOAT ievib = ip->evib;
  const KK_FLOAT jevib = jp->evib;

  const KK_FLOAT pre_ave_rotdof = (d_species[isp].rotdof + d_species[jsp].rotdof)/static_cast<KK_FLOAT>(2.0);

  const int n = d_reactions(isp,jsp).n;
  if (n == 0) return 0;
  auto& d_list = d_reactions(isp,jsp).d_list;

  // probablity to compare to reaction probability

  KK_FLOAT react_prob = 0.0;
  rand_type rand_gen = rand_pool.get_state();
  const KK_FLOAT random_prob = static_cast<KK_FLOAT>(rand_gen.drand());
  rand_pool.free_state(rand_gen);
  KK_FLOAT zi = 0.0;
  KK_FLOAT zj = 0.0;
  int avei = 0;
  int avej = 0;
  KK_FLOAT iTvib = 0.0;
  KK_FLOAT jTvib = 0.0;

  // loop over possible reactions for these 2 species

  for (int i = 0; i < n; i++) {
    r = &d_rlist[d_list[i]];

    // ignore energetically impossible reactions

    const KK_FLOAT pre_etotal = pre_etrans + pre_erot + pre_evib;

    // two options for total energy in TCE model
    // 0: partialEnergy = true: rDOF model
    // 1: partialEnergy = false: TCE: Rotation + Vibration

    // average DOFs participating in the reaction

    KK_FLOAT ecc,z;
    KK_FLOAT e_excess = 0.0;

    if (partialEnergy) {
      ecc = pre_etrans;
      z = r->d_coeff[0];
      if (pre_ave_rotdof > static_cast<KK_FLOAT>(0.1))
        ecc += pre_erot*z/pre_ave_rotdof;
    } else {
      ecc = pre_etotal;
      z = pre_ave_rotdof;
    }

    // Cover cases where coeff[1].neq.coeff[4]

    if (r->d_coeff[1]>((-1)*r->d_coeff[4])) e_excess = ecc - r->d_coeff[1];
    else e_excess = ecc + r->d_coeff[4];
    if (e_excess <= static_cast<KK_FLOAT>(0.0)) continue;

    if (!partialEnergy) {

       if (vibstyle == SMOOTH) z += (d_species[isp].vibdof + d_species[jsp].vibdof)/static_cast<KK_FLOAT>(2.0);
       else if (vibstyle == DISCRETE) {
            //Instantaneous z for diatomic molecules
            if (d_species[isp].nvibmode == 1) {
                avei = static_cast<int>
                        (ievib / (boltz * d_species[isp].vibtemp[0]));
                if (avei > 0) zi = static_cast<KK_FLOAT>(2.0) * avei * Kokkos::log(static_cast<KK_FLOAT>(1.0) / avei + static_cast<KK_FLOAT>(1.0));
                else zi = 0.0;
            } else if (d_species[isp].nvibmode > 1) {
                if (ievib < static_cast<KK_FLOAT>(1e-26) ) zi = 0.0; //Low Energy Cut-Off to prevent nan solutions to newtonTvib
                //Instantaneous T for polyatomic
                else {
                  iTvib = newtonTvib(d_species[isp].nvibmode,ievib,d_species[isp].vibtemp,3000,1e-4,1000);
                  zi = (2 * ievib)/(boltz * iTvib);
                }
            } else zi = 0.0;

            if (d_species[jsp].nvibmode == 1) {
                avej = static_cast<int>
                        (jevib / (boltz * d_species[jsp].vibtemp[0]));
                if (avej > 0) zj = static_cast<KK_FLOAT>(2.0) * avej * Kokkos::log(static_cast<KK_FLOAT>(1.0) / avej + static_cast<KK_FLOAT>(1.0));
                else zj = 0.0;
            } else if (d_species[jsp].nvibmode > 1) {
                if (jevib < static_cast<KK_FLOAT>(1e-26)) zj = 0.0;
                else {
                  jTvib = newtonTvib(d_species[jsp].nvibmode,jevib,d_species[jsp].vibtemp,3000,1e-4,1000);
                  zj = (2 * jevib)/(boltz * jTvib);
                }
            } else zj = 0.0;

            if (isnan(zi) || isnan(zj) || zi < 0 || zj < 0) Kokkos::abort("Root-Finding Error\n");
            z += static_cast<KK_FLOAT>(0.5) * (zi+zj);
       }
    }

    // compute probability of reaction

    switch (r->type) {
    case DISSOCIATION:
    case IONIZATION:
    case EXCHANGE:
      {
        react_prob += r->d_coeff[2] * Kokkos::tgamma(z+static_cast<KK_FLOAT>(2.5)-r->d_coeff[5]) / MAX(static_cast<KK_FLOAT>(1.0e-6),Kokkos::tgamma(z+r->d_coeff[3]+static_cast<KK_FLOAT>(1.5))) *
          Kokkos::pow(ecc-r->d_coeff[1],r->d_coeff[3]-1+r->d_coeff[5]) *
          Kokkos::pow(static_cast<KK_FLOAT>(1.0)-r->d_coeff[1]/ecc,z+static_cast<KK_FLOAT>(1.5)-r->d_coeff[5]);
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

        react_prob += recomb_boost * recomb_density * r->d_coeff[2] *
          Kokkos::tgamma(z+static_cast<KK_FLOAT>(2.5)-r->d_coeff[5]) / MAX(static_cast<KK_FLOAT>(1.0e-6),Kokkos::tgamma(z+r->d_coeff[3]+static_cast<KK_FLOAT>(1.5))) *
          Kokkos::pow(ecc-r->d_coeff[1],r->d_coeff[3]-1+r->d_coeff[5]) *  // extended to general recombination case with non-zero activation energy
          Kokkos::pow(static_cast<KK_FLOAT>(1.0)-r->d_coeff[1]/ecc,z+static_cast<KK_FLOAT>(1.5)-r->d_coeff[5]);
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

      // computeChemRates: the tally above is the whole point of this pass and
      //   no reaction is performed, but the search still stops at the first
      //   reaction that passes, as ReactTCE::attempt() does.  Without this the
      //   loop keeps going and tallies every later reaction that also passes

      break;
    }
  }

  // no reaction performed

  return 0;
}

/* ---------------------------------------------------------------------- */

 protected:
  int vibstyle;
  double boltz;

  DAT::tdual_int_scalar k_error_flag;
  DAT::t_int_scalar d_error_flag;
  HAT::t_int_scalar h_error_flag;
};

}

#endif
#endif

/* ERROR/WARNING messages:

*/
