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
#include "stdlib.h"
#include "string.h"
#include "compute_tambi_grid.h"
#include "particle.h"
#include "grid.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "fix_ambipolar.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

/* ----------------------------------------------------------------------
   per-cell AMBIPOLAR ELECTRON temperature from the velambi custom
   attribute maintained by fix ambipolar: every ambipolar ion carries the
   velocity of its attached electron, initialized at emission and updated
   by the electron's elastic collisions, so the electron velocity
   distribution in a cell defines a kinetic electron temperature
     T_e = m_e <|v_e - <v_e>|^2> / (3 kB)
   This is the collisional-exchange electron temperature of the ambipolar
   approximation: no field heating, electron conduction, or inelastic
   electron-energy losses are represented.
------------------------------------------------------------------------- */

ComputeTambiGrid::ComputeTambiGrid(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute tambi/grid command");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute grid group ID does not exist");
  groupbit = grid->bitmask[igroup];

  per_grid_flag = 1;
  size_per_grid_cols = 0;
  post_process_grid_flag = 1;

  // 6 tally quantities: N, Mass, mVx, mVy, mVz, mV^2

  npergroup = 6;
  ntotal = npergroup;

  memory->create(map,1,npergroup,"tambi/grid:map");
  for (int j = 0; j < npergroup; j++) map[0][j] = j;

  nglocal = 0;
  vector_grid = NULL;
  tally = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeTambiGrid::~ComputeTambiGrid()
{
  if (copymode) return;

  memory->destroy(map);
  memory->destroy(vector_grid);
  memory->destroy(tally);
}

/* ---------------------------------------------------------------------- */

void ComputeTambiGrid::init()
{
  index_ionambi = particle->find_custom((char *) "ionambi");
  index_velambi = particle->find_custom((char *) "velambi");
  if (index_ionambi < 0 || index_velambi < 0)
    error->all(FLERR,"Compute tambi/grid requires fix ambipolar");

  // ambipolar electron species from fix ambipolar

  int ifix;
  for (ifix = 0; ifix < modify->nfix; ifix++)
    if (strcmp(modify->fix[ifix]->style,"ambipolar") == 0) break;
  if (ifix == modify->nfix)
    error->all(FLERR,"Compute tambi/grid requires fix ambipolar");
  FixAmbipolar *afix = (FixAmbipolar *) modify->fix[ifix];
  emass = particle->species[afix->especies].mass;

  tprefactor = update->mvv2e / (3.0*update->boltz);

  reallocate();
}

/* ---------------------------------------------------------------------- */

void ComputeTambiGrid::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;

  Grid::ChildInfo *cinfo = grid->cinfo;
  Particle::OnePart *particles = particle->particles;
  int nlocal = particle->nlocal;

  int *ionambi = particle->eivec[particle->ewhich[index_ionambi]];
  double **velambi = particle->edarray[particle->ewhich[index_velambi]];

  int i,j,k,icell;
  double *v,*vec;

  for (i = 0; i < nglocal; i++)
    for (j = 0; j < ntotal; j++)
      tally[i][j] = 0.0;

  // one electron per ambipolar ion, velocity from velambi, mass = emass

  for (i = 0; i < nlocal; i++) {
    if (!ionambi[i]) continue;
    icell = particles[i].icell;
    if (!(cinfo[icell].mask & groupbit)) continue;

    v = velambi[i];
    vec = tally[icell];
    k = 0;

    vec[k++] += 1.0;
    vec[k++] += emass;
    vec[k++] += emass*v[0];
    vec[k++] += emass*v[1];
    vec[k++] += emass*v[2];
    vec[k++] += emass * (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
  }
}

/* ---------------------------------------------------------------------- */

int ComputeTambiGrid::query_tally_grid(int index, double **&array, int *&cols)
{
  // single-valued per-grid compute: one output (electron temperature)
  // backed by the one map row of 6 tally columns, regardless of index

  array = tally;
  cols = map[0];
  return npergroup;
}

/* ----------------------------------------------------------------------
   thermal temp of the electron cloud with center-of-mass motion removed,
   same normalization algebra as compute thermal/grid
------------------------------------------------------------------------- */

void ComputeTambiGrid::
post_process_grid(int index, int nsample,
                  double **etally, int *emap, double *vec, int nstride)
{
  // single-valued: the one output uses the one map row (map[0]).  The
  // passed index is ignored (callers use 0 for the whole-vector
  // reference); this compute has no per-index columns to select.

  int lo = 0;
  int hi = nglocal;
  int k = 0;

  if (!etally) {
    nsample = 1;
    etally = tally;
    emap = map[0];
    vec = vector_grid;
    nstride = 1;
  }

  double ncount,mass,mvx,mvy,mvz,mvsq;
  double *values;

  int n = emap[0];

  for (int icell = lo; icell < hi; icell++) {
    values = etally[icell];
    ncount = values[n];
    if (ncount <= 1.0) vec[k] = 0.0;
    else {
      mass = values[n+1];
      mvx = values[n+2];
      mvy = values[n+3];
      mvz = values[n+4];
      mvsq = values[n+5];
      vec[k] = mvsq - (mvx*mvx + mvy*mvy + mvz*mvz)/mass;
      vec[k] *= tprefactor/ncount;
    }
    k += nstride;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeTambiGrid::reallocate()
{
  if (grid->nlocal == nglocal) return;

  memory->destroy(vector_grid);
  memory->destroy(tally);
  nglocal = grid->nlocal;
  memory->create(vector_grid,nglocal,"tambi/grid:vector_grid");
  memory->create(tally,nglocal,ntotal,"tambi/grid:tally");
}

/* ---------------------------------------------------------------------- */

bigint ComputeTambiGrid::memory_usage()
{
  bigint bytes = 0;
  bytes += nglocal * sizeof(double);
  bytes += (bigint) ntotal*nglocal * sizeof(double);
  return bytes;
}
