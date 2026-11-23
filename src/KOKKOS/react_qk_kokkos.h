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

ReactStyle(qk/kk,ReactQKKokkos)

#else

#ifndef SPARTA_REACT_QK_KOKKOS_H
#define SPARTA_REACT_QK_KOKKOS_H

#include "math.h"
#include "react_bird_kokkos.h"
#include "kokkos_type.h"

namespace SPARTA_NS {

class ReactQKKokkos : public ReactBirdKokkos {
 public:
  ReactQKKokkos(class SPARTA *, int, char **);
  ReactQKKokkos(class SPARTA* sparta) : ReactBirdKokkos(sparta) {copy = 1;}
  void init();
  int attempt(Particle::OnePart *, Particle::OnePart *,
              double, double, double, double &, int &) {return 0;}

  enum{DISSOCIATION,EXCHANGE,IONIZATION,RECOMBINATION};   // other files

/* ----------------------------------------------------------------------
   quantum-kinetic (QK) reaction attempt, device version
   mirrors ReactQK::attempt(); supports DISSOCIATION and EXCHANGE
   recomb args are accepted for a uniform collide dispatch but unused (QK has
   no recombination)
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
int attempt_kk(Particle::OnePart *ip, Particle::OnePart *jp,
         double pre_etrans, double pre_erot, double pre_evib,
         double &post_etotal, int &kspecies,
         int & /*recomb_species*/, double & /*recomb_density*/,
         const t_species_1d_const &d_species) const
{
  const int isp = ip->ispecies;
  const int jsp = jp->ispecies;

  const double omega = d_omega(isp,jsp);

  const int n = d_reactions(isp,jsp).n;
  if (n == 0) return 0;
  auto& d_list = d_reactions(isp,jsp).d_list;

  double react_prob = 0.0;
  rand_type rand_gen = rand_pool.get_state();
  const double random_prob = rand_gen.drand();

  for (int i = 0; i < n; i++) {
    OneReactionKokkos *r = &d_rlist[d_list[i]];

    const double pre_etotal = pre_etrans + pre_erot + pre_evib;

    // ignore energetically impossible reactions
    // post-collision energy would be <= 0 if Ea < -coeff[4] in the input file
    // QK tests the energy of the molecule below, not the TCE partial energy

    if (pre_etotal + r->d_coeff[4] <= 0.0) continue;
    double ecc;

    // QK model uses vibrational state of the molecule
    // use R1 (dissociating species) if it vibrates, else the other reactant
    // R1 may be either I or J since reaction list includes both orders
    // mirrors ReactQK::attempt()

    Particle::OnePart *mp = ip;
    Particle::OnePart *op = jp;
    if (ip->ispecies != r->d_reactants[0]) {
      mp = jp;
      op = ip;
    }
    if (d_species[mp->ispecies].vibtemp[0] <= 0.0) mp = op;
    const int msp = mp->ispecies;
    if (d_species[msp].vibtemp[0] <= 0.0) continue;

    const double inverse_kT = 1.0 / (boltz * d_species[msp].vibtemp[0]);

    int iv = 0,ilevel,maxlev,limlev;

    switch (r->type) {
    case DISSOCIATION:
      {
        ecc = pre_etrans + mp->evib;
        maxlev = static_cast<int> (ecc * inverse_kT);
        limlev = static_cast<int> (fabs(r->d_coeff[1]) * inverse_kT);
        if (maxlev > limlev) react_prob = 1.0;

        // if both particles are the dissociating species, either can dissociate

        else if (isp == jsp) {
          ecc = pre_etrans + op->evib;
          maxlev = static_cast<int> (ecc * inverse_kT);
          if (maxlev > limlev) react_prob = 1.0;
        }
        break;
      }
    case EXCHANGE:
      {
        if (r->d_coeff[4] < 0.0 && d_species[msp].rotdof > 0) {

          // endothermic reaction

          ecc = pre_etrans + mp->evib;
          maxlev = static_cast<int> (ecc * inverse_kT);
          if (ecc > r->d_coeff[1]) {
            // sample into a local prob, not react_prob: react_prob is the
            //   reaction probability this function returns on, and a rejected
            //   draw must not leave a fractional value in it (or carry one
            //   into the next reaction).  Mirrors ReactQK::attempt()
            // Larsen-Borgnakke level, accepted with probability prob
            // reaction occurs if the level reaches the activation energy
            double prob = 0.0;
            do {
              iv = static_cast<int> (rand_gen.drand()*(maxlev+0.99999999));
              double evib = static_cast<double> (iv / inverse_kT);
              if (evib < ecc) prob = pow(1.0-evib/ecc,1.5-omega);
              else prob = 0.0;
            } while (prob < rand_gen.drand());

            ilevel = static_cast<int> (r->d_coeff[1] * inverse_kT);
            if (iv >= ilevel) react_prob = 1.0;
          }

        } else if (r->d_coeff[4] > 0.0 && d_species[msp].rotdof > 0) {

          ecc = pre_etrans + mp->evib;

          // mspec = post-collision molecular species

          int mspec = r->d_products[0];
          if (d_species[mspec].rotdof < 2.0) mspec = r->d_products[1];

          // levels of the product molecule

          ecc += r->d_coeff[4];
          const double inverse_kT_post = 1.0 / (boltz*d_species[mspec].vibtemp[0]);
          maxlev = static_cast<int> (ecc * inverse_kT_post);
          double prob = 0.0;
          do {
            iv = rand_gen.drand()*(maxlev+0.99999999);
            double evib = static_cast<double> (iv / inverse_kT_post);
            if (evib < ecc) prob = pow(1.0-evib/ecc,1.5 - r->d_coeff[6]);
            else prob = 0.0;
          } while (prob < rand_gen.drand());

          ilevel = static_cast<int> (r->d_coeff[4] * inverse_kT_post);
          if (iv >= ilevel) react_prob = 1.0;
        }

        break;
      }
    default:
      Kokkos::abort("ReactQKKokkos: Unknown outcome in reaction\n");
      break;
    }

    if (react_prob > random_prob) {
      Kokkos::atomic_inc(&d_tally_reactions[d_list[i]]);
      ip->ispecies = r->d_products[0];
      jp->ispecies = r->d_products[1];
      post_etotal = pre_etotal + r->d_coeff[4];
      if (r->nproduct > 2) kspecies = r->d_products[2];
      else kspecies = -1;
      rand_pool.free_state(rand_gen);
      return d_list[i] + 1;
    }
  }

  rand_pool.free_state(rand_gen);
  return 0;
}

 protected:
  double boltz;
  DAT::t_float_2d d_omega;     // VSS omega for each species pair
};

}

#endif
#endif
