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

ReactStyle(table/kk,ReactTableKokkos)

#else

#ifndef SPARTA_REACT_TABLE_KOKKOS_H
#define SPARTA_REACT_TABLE_KOKKOS_H

#include "react_bird_kokkos.h"
#include "interp_table_kokkos.h"
#include "kokkos_type.h"
#include "particle.h"
#include "update.h"

namespace SPARTA_NS {

/* ----------------------------------------------------------------------
   react table/kk = tabulated reaction cross sections on the KOKKOS package

   the host side mirrors ReactTable, which this cannot inherit: ReactTable
   derives from ReactBird and so does ReactBirdKokkos, whose device arrays
   are needed here.  ReactTCEKokkos has the same relationship to ReactTCE
   and resolves it the same way, by restating the host logic.

   attempt_kk() reproduces ReactTable::attempt() exactly, including the
   order in which random numbers are drawn, so the two builds agree bit for
   bit under SPARTA_KOKKOS_EXACT.
------------------------------------------------------------------------- */

class ReactTableKokkos : public ReactBirdKokkos {
 public:
  ReactTableKokkos(class SPARTA *, int, char **);
  ReactTableKokkos(class SPARTA *sparta) : ReactBirdKokkos(sparta) {copy = 1;}
  ~ReactTableKokkos();
  void init();

  // the host entry point is never used on this path; the device one below is

  int attempt(Particle::OnePart *, Particle::OnePart *,
              double, double, double, double &, int &) {return 0;}

/* ----------------------------------------------------------------------
   attempt a reaction for this collision
   P_react = sigma_react(E) / sigma_total(E), with sigma_total the cross
     section the collide style used to select this pair
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
int attempt_kk(Particle::OnePart *ip, Particle::OnePart *jp,
               double pre_etrans, double pre_erot, double pre_evib,
               double &post_etotal, int &kspecies,
               int &recomb_species, double &recomb_density,
               const t_species_1d_const &d_species,
               double sigma_total) const
{
  const int isp = ip->ispecies;
  const int jsp = jp->ispecies;

  const int n = d_reactions(isp,jsp).n;
  if (n == 0) return 0;
  auto& d_list = d_reactions(isp,jsp).d_list;

  if (sigma_total <= 0.0) return 0;

  const double pre_etotal = pre_etrans + pre_erot + pre_evib;

  // the tables are indexed by relative translational energy, which is what
  //   measured and computed reactive cross sections are tabulated against

  const double mi = d_species[isp].mass;
  const double mj = d_species[jsp].mass;
  const double mr = (isp == jsp) ? mi/2.0 : mi*mj/(mi+mj);

  double react_prob = 0.0;
  rand_type rand_gen = rand_pool.get_state();
  const double random_prob = rand_gen.drand();
  rand_pool.free_state(rand_gen);

  for (int i = 0; i < n; i++) {
    const int m = d_list[i];
    auto r = &d_rlist[m];

    // ignore energetically impossible reactions, the same gate as
    //   ReactTCE::attempt() applies

    double e_excess;
    if (r->d_coeff[1] > -r->d_coeff[4]) e_excess = pre_etotal - r->d_coeff[1];
    else e_excess = pre_etotal + r->d_coeff[4];
    if (e_excess <= 0.0) continue;

    const double ereact = d_tabetot[m] ? pre_etotal : pre_etrans;
    const double sigma_r = rtabdev.evaluate(d_rtabindex[m],2.0*ereact/mr);

    if (r->type == RECOMBINATION_KK) {

      // the 3rd particle only selects which recombination reaction applies

      if (recomb_species < 0) continue;
      auto& d_sp2recomb = d_reactions(isp,jsp).d_sp2recomb;
      if (d_sp2recomb[recomb_species] != m) continue;

      // sigma_r is a cross section per unit third-body number density, so
      //   n3*sigma_r is an area

      react_prob += recomb_boost * recomb_density * sigma_r / sigma_total;

    } else react_prob += sigma_r / sigma_total;

    // sigma_react exceeding sigma_total clips the rate; device code cannot
    //   warn, so record which of the two causes it was and let the host
    //   warn after the kernel, as ReactTable::attempt() does inline

    if (react_prob > 1.0)
      Kokkos::atomic_or(&d_warn(),
                        (r->type == RECOMBINATION_KK) ? WARN_RBOOST : WARN_ENVELOPE);

    if (react_prob > random_prob) {
      Kokkos::atomic_inc(&d_tally_reactions[m]);
      if (!computeChemRates) {
        ip->ispecies = r->d_products[0];

        switch (r->type) {
        case DISSOCIATION_KK:
        case IONIZATION_KK:
        case EXCHANGE_KK:
          {
            jp->ispecies = r->d_products[1];
            break;
          }
        case RECOMBINATION_KK:
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

        return m + 1;
      }
    }
  }

  return 0;
}

 protected:

  // reaction types, restated here because the enum in the .cpp files is
  //   file scope; the order matches every other react file

  enum{DISSOCIATION_KK,EXCHANGE_KK,IONIZATION_KK,RECOMBINATION_KK};

  // bits of d_warn, one per cause of a probability above 1

  enum{WARN_ENVELOPE=1,WARN_RBOOST=2};

  // host side, mirroring ReactTable

  class InterpTable **rtab;   // cross section table per reaction, or NULL
  char **tabfile,**tabkey;
  int *tabetot;
  int maxtab;
  int warnflag;                 // bits of d_warn already warned about

  int read_style(OneReaction *, char *);
  void read_coeffs(OneReaction *, char *, char *);
  void grow_tab(int);

  // device side

  InterpTableKokkos rtabdev;
  DAT::tdual_int_1d k_rtabindex;
  DAT::t_int_1d d_rtabindex;    // index into rtabdev per reaction
  DAT::tdual_int_1d k_tabetot;
  DAT::t_int_1d d_tabetot;

 public:

  // flags the device sets during a collision kernel and the host acts on
  //   afterwards, since device code can neither warn nor abort itself:
  //   d_warn for a reaction cross section above the total one, and the
  //   table's own flag for an evaluation outside a table with EXTRAP error

  DAT::t_int_scalar d_warn;
  HAT::t_int_scalar h_warn;
  void clear_flags() { Kokkos::deep_copy(d_warn,0); rtabdev.clear_error(); }
  void check_flags();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: React table/kk can only be used with a VSS-based collide style

The style needs the total cross section which was used to select the
collision pair, which only the vss and table collide styles provide.

E: React table/kk requires every reaction to use style T

The style has no way to form a probability for an Arrhenius or Quantum
reaction, so a reaction file for it may not mix styles.

E: React table/kk reaction has no cross section table

A reaction was declared with style T but no table was attached to it.

E: Value is outside the tabulated data range

A reaction cross section table whose extrapolation mode is error was
evaluated outside its range.  Device code cannot raise an error itself, so
this is recorded during the collision kernel and raised afterwards.

*/
