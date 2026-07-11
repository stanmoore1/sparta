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
#include "react_tce_kokkos.h"
#include "particle.h"
#include "collide.h"
#include "random_knuth.h"
#include "error.h"

// DEBUG
#include "update.h"

using namespace SPARTA_NS;

enum{NONE,DISCRETE,SMOOTH};                             // several files
enum{DISSOCIATION,EXCHANGE,IONIZATION,RECOMBINATION};   // other files

/* ---------------------------------------------------------------------- */

ReactTCEKokkos::ReactTCEKokkos(SPARTA *sparta, int narg, char **arg) :
  ReactBirdKokkos(sparta, narg, arg) {}

/* ---------------------------------------------------------------------- */

void ReactTCEKokkos::init()
{
  if (!collide || (strcmp(collide->style,"vss") != 0 && strcmp(collide->style,"vss/kk") != 0))
    error->all(FLERR,"React tce can only be used with collide vss");

  ReactBirdKokkos::init();

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

  vibstyle = collide->vibstyle;
  elecstyle = collide->elecstyle;
  boltz = update->boltz;

  // with partial_energy no, build the host microcanonical energy-factor
  // tables (shared ReactBird machinery) and mirror them into flat device
  // views (zero-padded to the widest table); reactions whose reactants
  // carry no discrete ladders keep n = 0 and use the standard analytic
  // factor at runtime; the mtab_num rows of reverse recombinations
  // (3-body detailed balance) are mirrored alongside

  if (!partialEnergy) {
    build_micro_tables();

    int maxn = 0;
    for (int i = 0; i < mtab_nlist; i++) maxn = MAX(maxn,mtab_n[i]);
    d_mtab = DAT::t_float_2d("react:d_mtab",mtab_nlist,MAX(maxn,1));
    d_mtab_num = DAT::t_float_2d("react:d_mtab_num",mtab_nlist,MAX(maxn,1));
    d_mtab_du = DAT::t_float_1d("react:d_mtab_du",mtab_nlist);
    d_mtab_n = DAT::t_int_1d("react:d_mtab_n",mtab_nlist);
    auto h_mtab = Kokkos::create_mirror_view(d_mtab);
    auto h_mtab_num = Kokkos::create_mirror_view(d_mtab_num);
    auto h_du = Kokkos::create_mirror_view(d_mtab_du);
    auto h_n = Kokkos::create_mirror_view(d_mtab_n);
    for (int i = 0; i < mtab_nlist; i++) {
      h_du(i) = mtab_du[i];
      h_n(i) = mtab[i] ? mtab_n[i] : 0;
      if (mtab[i])
        for (int k = 0; k < mtab_n[i]; k++) h_mtab(i,k) = mtab[i][k];
      if (mtab_num[i])
        for (int k = 0; k < mtab_n[i]; k++) h_mtab_num(i,k) = mtab_num[i][k];
    }
    Kokkos::deep_copy(d_mtab,h_mtab);
    Kokkos::deep_copy(d_mtab_num,h_mtab_num);
    Kokkos::deep_copy(d_mtab_du,h_du);
    Kokkos::deep_copy(d_mtab_n,h_n);
  } else free_micro_tables();
}
