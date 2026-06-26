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
#include "surf_react_global.h"
#include "input.h"
#include "variable.h"
#include "update.h"
#include "comm.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "math_extra.h"
#include "error.h"

using namespace SPARTA_NS;

enum{NUMERIC,VAREQUAL};

/* ---------------------------------------------------------------------- */

SurfReactGlobal::SurfReactGlobal(SPARTA *sparta, int narg, char **arg) :
  SurfReact(sparta, narg, arg)
{
  if (narg != 4) error->all(FLERR,"Illegal surf_react global command");

  // each probability is either a numeric value or an equal-style variable
  // a variable is flagged with a leading "v_" prefix on the argument

  prob_destroy = prob_create = 0.0;
  pdelete_name = pcreate_name = NULL;
  pdelete_var = pcreate_var = -1;

  parse_prob(arg[2],pdelete_mode,pdelete_name,prob_destroy);
  parse_prob(arg[3],pcreate_mode,pcreate_name,prob_create);

  // dynamic if either probability is set by a variable
  // sum of probabilities can only be statically checked when both are numeric

  if (pdelete_mode == VAREQUAL || pcreate_mode == VAREQUAL) dynamicflag = 1;

  if (pdelete_mode == NUMERIC && pcreate_mode == NUMERIC &&
      prob_destroy + prob_create > 1.0)
    error->all(FLERR,"Illegal surf_react global command");

  // setup the reaction tallies

  nsingle = ntotal = 0;

  nlist = 2;
  tally_single = new int[nlist];
  tally_total = new int[nlist];
  tally_single_all = new int[nlist];
  tally_total_all = new int[nlist];

  size_vector = 2 + 2*nlist;

  // initialize RNG

  random = new RanKnuth(update->ranmaster->uniform());
  double seed = update->ranmaster->uniform();
  random->reset(seed,comm->me,100);
}

/* ---------------------------------------------------------------------- */

SurfReactGlobal::~SurfReactGlobal()
{
  if (copy) return;

  delete [] pdelete_name;
  delete [] pcreate_name;
  delete random;
}

/* ----------------------------------------------------------------------
   parse a probability argument as a numeric value or equal-style variable
   sets mode, and either name (for a variable) or value (for a number)
------------------------------------------------------------------------- */

void SurfReactGlobal::parse_prob(char *str, int &mode, char *&name,
                                 double &value)
{
  if (strstr(str,"v_") == str) {
    mode = VAREQUAL;
    int n = strlen(&str[2]) + 1;
    name = new char[n];
    strcpy(name,&str[2]);
  } else {
    mode = NUMERIC;
    value = input->numeric(FLERR,str);
    if (value < 0.0 || value > 1.0)
      error->all(FLERR,"Surf_react global probability must be from 0.0 to 1.0");
  }
}

/* ----------------------------------------------------------------------
   re-find variables and validate they are equal-style
   variable indices can change between runs, so look them up at init
------------------------------------------------------------------------- */

void SurfReactGlobal::init()
{
  SurfReact::init();

  if (pdelete_mode == VAREQUAL) {
    pdelete_var = input->variable->find(pdelete_name);
    if (pdelete_var < 0)
      error->all(FLERR,"Surf_react global pdelete variable name does not exist");
    if (!input->variable->equal_style(pdelete_var))
      error->all(FLERR,"Surf_react global pdelete variable is not equal-style");
  }

  if (pcreate_mode == VAREQUAL) {
    pcreate_var = input->variable->find(pcreate_name);
    if (pcreate_var < 0)
      error->all(FLERR,"Surf_react global pcreate variable name does not exist");
    if (!input->variable->equal_style(pcreate_var))
      error->all(FLERR,"Surf_react global pcreate variable is not equal-style");
  }
}

/* ----------------------------------------------------------------------
   recompute variable-driven probabilities
   called once per timestep by Update before surface collisions
------------------------------------------------------------------------- */

void SurfReactGlobal::dynamic()
{
  if (pdelete_mode == VAREQUAL) {
    prob_destroy = input->variable->compute_equal(pdelete_var);
    if (prob_destroy < 0.0 || prob_destroy > 1.0)
      error->all(FLERR,"Surf_react global pdelete must be from 0.0 to 1.0");
  }

  if (pcreate_mode == VAREQUAL) {
    prob_create = input->variable->compute_equal(pcreate_var);
    if (prob_create < 0.0 || prob_create > 1.0)
      error->all(FLERR,"Surf_react global pcreate must be from 0.0 to 1.0");
  }

  if (prob_destroy + prob_create > 1.0)
    error->all(FLERR,"Surf_react global pdelete + pcreate > 1.0");
}

/* ----------------------------------------------------------------------
   select surface reaction to perform for particle with ptr IP on surface
   return which reaction 1 (destroy), 2 (create), 0 = no reaction
   if create, add particle and return ptr JP
------------------------------------------------------------------------- */

int SurfReactGlobal::react(Particle::OnePart *&ip, int, double *,
                           Particle::OnePart *&jp, int &)
{
  double r = random->uniform();

  // perform destroy reaction

  if (r < prob_destroy) {
    nsingle++;
    tally_single[0]++;
    ip = NULL;
    return 1;
  }

  // perform create reaction
  // clone 1st particle to create 2nd particle
  // if add_particle performs a realloc:
  //   make copy of x,v with new species
  //   rot/vib energies will be reset by SurfCollide
  //   repoint ip to new particles data struct if reallocated

  if (r < prob_destroy+prob_create) {
    nsingle++;
    tally_single[1]++;
    double x[3],v[3];
    int id = MAXSMALLINT*random->uniform();
    memcpy(x,ip->x,3*sizeof(double));
    memcpy(v,ip->v,3*sizeof(double));
    Particle::OnePart *particles = particle->particles;
    int reallocflag =
      particle->add_particle(id,ip->ispecies,ip->icell,x,v,0.0,0.0);
    if (reallocflag) ip = particle->particles + (ip - particles);
    jp = &particle->particles[particle->nlocal-1];
    return 2;
  }

  // no reaction

  return 0;
}

/* ---------------------------------------------------------------------- */

char *SurfReactGlobal::reactionID(int m)
{
  if (m == 0) return (char *) "delete";
  return (char *) "create";
}
