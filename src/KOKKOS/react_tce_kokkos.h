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
              double, double, double, double, double &, int &) { return 0; }

/* ---------------------------------------------------------------------- */

enum{NONE,DISCRETE,SMOOTH};
enum{DISSOCIATION,EXCHANGE,IONIZATION,RECOMBINATION};   // other files

KOKKOS_INLINE_FUNCTION
int attempt_kk(Particle::OnePart *ip, Particle::OnePart *jp,
         double pre_etrans, double pre_erot, double pre_evib, double pre_eelec,
         double &post_etotal, int &kspecies,
         int &recomb_species, double &recomb_density,
         Particle::OnePart *recomb_part3, const double recomb_p3_eelec,
         const t_species_1d_const &d_species,
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

    // a reverse (B-style) recombination is fully microcanonical, matching
    // ReactTCE::attempt: the 3rd particle's energy counts toward the
    // barrier, so the pair-energy threshold below does not apply to it; an
    // external-Keq recombination uses the same 3-body table times the
    // residual thermal factor applied below

    int micro3 = 0;
    if (r->reverse && r->type == RECOMBINATION &&
        d_mtab_n[d_list[i]] > 0 && d_mtab_num_flag[d_list[i]]) micro3 = 1;

    // Cover cases where coeff[1].neq.coeff[4]

    if (r->d_coeff[1]>((-1)*r->d_coeff[4])) e_excess = ecc - r->d_coeff[1];
    else e_excess = ecc + r->d_coeff[4];
    if (e_excess <= 0.0 && !micro3) continue;

    // energy-dependent factor of the TCE probability: the tabulated
    // microcanonical average over the reactants' discrete vibrational
    // and electronic ladders, matching ReactBird::vib_micro_factor
    // exactly; reactions without discrete ladders use the standard
    // analytic factor

    double efactor = 0.0;
    if (!micro3) {   // a micro3 reaction uses its own 3-body factor below
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
    }

    // residual thermal correction for an external-Keq reverse reaction,
    // matching ReactTCE::attempt: multiply the detailed-balance table by
    // R(T) = Keq_statmech/Keq_ext at the cell temperature so the thermal
    // average matches the external Keq while the shape stays energy-
    // resolved; other reverse reactions need no temperature (keq_resid = 1)

    double keq_resid = 1.0;
    if (r->reverse && r->keq_flag) {
      if (tgas_cell > 0.0) {
        // clamp to the residual fit window (1000-60000 K), matching
        // ReactTCE::attempt, to avoid an uncontrolled Park extrapolation
        const double tr = tgas_cell < 1000.0 ? 1000.0 :
          (tgas_cell > 60000.0 ? 60000.0 : tgas_cell);
        const double z10 = 10000.0/tr;
        keq_resid = exp(r->keq_resid_coeff[0]/z10 + r->keq_resid_coeff[1] +
                        r->keq_resid_coeff[2]*log(z10) +
                        r->keq_resid_coeff[3]*z10 +
                        r->keq_resid_coeff[4]*z10*z10);
      } else continue;
    }
    double prefactor = r->d_coeff[2] * keq_resid;

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

        // fully microcanonical reverse recombination, matching
        // ReactTCE::attempt exactly (see ReactBird::build_db3_table);
        // no cell temperature is used

        if (micro3) {
          Particle::OnePart *p3 = recomb_part3;
          const int sp3 = p3->ispecies;

          const double mi = d_species[isp].mass;
          const double mj = d_species[jsp].mass;
          const double m3 = d_species[sp3].mass;
          const double divisor = 1.0/(mi + mj);
          double *vi = ip->v;
          double *vj = jp->v;
          double *v3 = p3->v;
          const double du3 = v3[0] - (mi*vi[0] + mj*vj[0])*divisor;
          const double dv3 = v3[1] - (mi*vi[1] + mj*vj[1])*divisor;
          const double dw3 = v3[2] - (mi*vi[2] + mj*vj[2])*divisor;
          const double mu3 = m3*(mi + mj)/(m3 + mi + mj);
          const double eps_t = 0.5*mu3*(du3*du3 + dv3*dv3 + dw3*dw3);

          const double eelec3 = recomb_p3_eelec;

          // continuum density weights of the 3rd particle's energies,
          // matching the flat-measure dimension of the table; skip the
          // attempt at the (measure-zero) singular points

          double c3 = sqrt(eps_t);
          const int rotdof3 = d_species[sp3].rotdof;
          if (rotdof3 > 0 && rotdof3 != 2)
            c3 *= pow(p3->erot,0.5*rotdof3-1.0);
          const int vibdof3 = d_species[sp3].vibdof;
          if (vibstyle == SMOOTH && vibdof3 > 0 && vibdof3 != 2)
            c3 *= pow(p3->evib,0.5*vibdof3-1.0);
          if (!(c3 > 0.0)) continue;

          const int ir = d_list[i];
          const int ntab = d_mtab_n[ir];
          const double dutab = d_mtab_du[ir];

          double xpair;
          {
            const double x = ecc/dutab;
            const int k = (int) x;
            if (k >= ntab-1) xpair = d_mtab(ir,ntab-1);
            else {
              const double f = x - k;
              xpair = (1.0-f)*d_mtab(ir,k) + f*d_mtab(ir,k+1);
            }
          }
          if (!(xpair > 0.0)) continue;

          const double w = ecc + eps_t + p3->erot + p3->evib + eelec3;
          double numw;
          {
            const double x = w/dutab;
            const int k = (int) x;
            if (k >= ntab-1) numw = d_mtab_num(ir,ntab-1);
            else {
              const double f = x - k;
              numw = (1.0-f)*d_mtab_num(ir,k) + f*d_mtab_num(ir,k+1);
            }
          }

          react_prob += keq_resid * recomb_boost * recomb_density * numw / (xpair*c3);
          break;
        }

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

    // diagnostic (matches ReactTCE::attempt): warn once if the cumulative
    //   probability saturates above 1, where the rate can be under-counted

    if (react_prob > 1.0) {
      if (Kokkos::atomic_compare_exchange(&d_probwarn(),0,1) == 0)
        Kokkos::printf("WARNING: TCE reaction probability exceeded 1; "
          "reaction rate may be under-counted - reduce the timestep or fnum, "
          "or refine the grid\n");
    }

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

 protected:
  int vibstyle;
  int elecstyle;
  double boltz;

  // per-reaction microcanonical energy-factor tables, device mirrors of
  // the ReactBird host tables; n = 0 marks "no table"; d_mtab_num rows
  // are only nonzero for reverse recombinations (3-body detailed
  // balance, see ReactBird::build_db3_table)

  DAT::t_float_2d d_mtab;
  DAT::t_float_2d d_mtab_num;
  DAT::t_float_1d d_mtab_du;
  DAT::t_int_1d d_mtab_n;
  DAT::t_int_1d d_mtab_num_flag;   // 1 if mtab_num row is a real 3-body
                                   //   detailed-balance table (mirrors the
                                   //   host mtab_num[i] != NULL test)

  DAT::tdual_int_scalar k_error_flag;
  DAT::t_int_scalar d_error_flag;
  HAT::t_int_scalar h_error_flag;

  // one-time device flag: set the first time a collision's cumulative
  //   reaction probability exceeds 1 (rate may be under-counted); mirrors
  //   the CPU ReactTCE::probwarnflag warning
  DAT::t_int_scalar d_probwarn;
};

}

#endif
#endif

/* ERROR/WARNING messages:

*/
