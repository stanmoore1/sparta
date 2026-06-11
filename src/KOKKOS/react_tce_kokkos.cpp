/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.sandia.gov
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
#include "particle_kokkos.h"
#include "collide.h"
#include "random_knuth.h"
#include "error.h"

// DEBUG
#include "update.h"

using namespace SPARTA_NS;

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

  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
  d_particles = particle_kk->k_particles.d_view;
  d_nelecstates = particle_kk->d_nelecstates;
  d_elecstates = particle_kk->d_elecstates;
  d_ewhich = particle_kk->k_ewhich.d_view;
  k_eivec = particle_kk->k_eivec;

  vibstyle = collide->vibstyle;
  elecstyle = collide->elecstyle;
  index_elecstate = collide->index_elecstate;
  boltz = update->boltz;
}
