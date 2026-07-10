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
#include "fix_elecmode.h"
#include "update.h"
#include "particle.h"
#include "collide.h"
#include "comm.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "math_const.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

enum{INT,DOUBLE};                      // several files
enum{NONE,DISCRETE,SMOOTH};            // several files

/* ---------------------------------------------------------------------- */

FixElecmode::FixElecmode(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg != 2) error->all(FLERR,"Illegal fix elecmode command");

  flag_update_custom = 1;
  flag_surf_react = 1;

  random = new RanKnuth(update->ranmaster->uniform());
  double seed = update->ranmaster->uniform();
  random->reset(seed,comm->me,100);

  // create per-particle array

  if (!collide)
    error->all(FLERR,"Cannot use fix elecmode without "
               "collide style defined");

  if (collide->elecstyle != DISCRETE)
    error->all(FLERR,"Cannot use fix elecmode without "
               "collide_modify electronic discrete");

  // check if custom attributes already exist, due to restart file
  // else create per-particle attributes

  index_elecstate = particle->find_custom((char *) "elecstate");
  if (index_elecstate < 0)
    index_elecstate = particle->add_custom((char *) "elecstate",INT,0);
  index_eelec = particle->find_custom((char *) "eelec");
  if (index_eelec < 0)
    index_eelec = particle->add_custom((char *) "eelec",DOUBLE,0);
}

/* ---------------------------------------------------------------------- */

FixElecmode::~FixElecmode()
{
  if (copy) return;

  delete random;
  particle->remove_custom(index_elecstate);
  particle->remove_custom(index_eelec);
}

/* ---------------------------------------------------------------------- */

int FixElecmode::setmask()
{
  int mask = 0;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixElecmode::init()
{
  if (collide->elecstyle != DISCRETE)
    error->all(FLERR,"Cannot use fix elecmode without "
               "collide_modify electronic discrete");
}

/* ----------------------------------------------------------------------
   called when a particle with index is created
    or when temperature dependent properties need to be updated
   populate an electronic state and set eelec
------------------------------------------------------------------------- */

void FixElecmode::update_custom(int index, const TempsInfo &temps,
                               double *vstream)
{
  int *elecstate = particle->eivec[particle->ewhich[index_elecstate]];
  double *eelec = particle->edvec[particle->ewhich[index_eelec]];

  int isp = particle->particles[index].ispecies;

  // if no states, just return

  if (particle->species[isp].elecdat == NULL) return;

  elecstate[index] = particle->ielec(isp,temps.elec,random);
  eelec[index] = update->boltz*particle->species[isp].elecdat->states[elecstate[index]].temp;
}

/* ----------------------------------------------------------------------
   called when a surface reaction occurred for particle I
   iorig = original particle before reaction, I,J = particles after reaction
   if the reaction changed the species of particle I, make its
   elecstate/eelec consistent with the NEW species' state table:
   surf_collide styles with a surface temperature (diffuse, cll, td,
   impulsive, specular) resample both via update_custom() before this hook
   runs, making this a no-op for them; styles without one (adiabatic,
   piston) would otherwise carry a stale state index that can read past
   the end of the new species' table in the next collision
   a J particle newly created by the reaction starts with zeroed custom
   data (ground state), which is already consistent
------------------------------------------------------------------------- */

void FixElecmode::surf_react(Particle::OnePart *iorig, int &i, int &j)
{
  if (i < 0) return;

  Particle::OnePart *particles = particle->particles;
  int isp = particles[i].ispecies;
  if (iorig->ispecies == isp) return;

  int *elecstate = particle->eivec[particle->ewhich[index_elecstate]];
  double *eelec = particle->edvec[particle->ewhich[index_eelec]];

  Particle::ElectronicData *edat = particle->species[isp].elecdat;

  if (edat == NULL) {
    elecstate[i] = 0;
    eelec[i] = 0.0;
    return;
  }

  if (elecstate[i] >= edat->nelecstate) elecstate[i] = 0;
  eelec[i] = update->boltz*edat->states[elecstate[i]].temp;
}
