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

#include "mpi.h"
#include "string.h"
#include "compute_temp.h"
#include "update.h"
#include "particle.h"
#include "domain.h"
#include "error.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

ComputeTemp::ComputeTemp(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg != 2) error->all(FLERR,"Illegal compute temp command");

  scalar_flag = 1;
}

/* ---------------------------------------------------------------------- */

double ComputeTemp::compute_scalar()
{
  invoked_scalar = update->ntimestep;

  Particle::Species *species = particle->species;
  Particle::OnePart *particles = particle->particles;
  int nlocal = particle->nlocal;

  double *v;
  double t = 0.0;

  // accumulate the summed particle weight as well, so the temperature is
  // normalized by the effective particle count.  pweight(i) is the
  // per-particle weight relative to fnum: the per-species weight under
  // SWS, the per-particle weight under stochastic (SWPM) weighting, and
  // 1.0 with no weighting (where wsum reduces to the particle count).

  double wsum = 0.0;
  for (int i = 0; i < nlocal; i++) {
    v = particles[i].v;
    double w = particle->pweight(i);
    t += (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) *
      species[particles[i].ispecies].mass * w;
    wsum += w;
  }

  double local[2] = {t,wsum}, all[2];
  MPI_Allreduce(local,all,2,MPI_DOUBLE,MPI_SUM,world);
  scalar = all[0];
  double wglobal = all[1];

  bigint n = particle->nlocal;
  MPI_Allreduce(&n,&particle->nglobal,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
  if (particle->nglobal == 0) return 0.0;

  // normalize with 3 instead of dim since even 2d has 3 velocity components

  double factor = update->mvv2e / (3.0 * wglobal * update->boltz);
  scalar *= factor;
  return scalar;
}
