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

#include "compute_vremax_grid.h"
#include "collide.h"
#include "grid.h"
#include "update.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

ComputeVremaxGrid::ComputeVremaxGrid(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg != 2) error->all(FLERR,"Illegal compute vremax/grid command");

  per_grid_flag = 1;
  size_per_grid_cols = 0;

  nglocal = 0;
  vector_grid = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeVremaxGrid::~ComputeVremaxGrid()
{
  if (copy || copymode) return;

  memory->destroy(vector_grid);
}

/* ---------------------------------------------------------------------- */

void ComputeVremaxGrid::init()
{
  if (!collide)
    error->all(FLERR,"Compute vremax/grid requires a collide style be defined");

  reallocate();
}

/* ----------------------------------------------------------------------
   copy current per-cell vremax into the output vector
   vremax is owned by Collide, indexed [icell][igroup][jgroup]
   report the max over all group pairs in the cell
     for the common single-group case this is simply the cell's vremax
   vremax may be NULL before the first run, in which case output zeroes
------------------------------------------------------------------------- */

void ComputeVremaxGrid::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;

  double ***vremax = collide->get_vremax();

  if (!vremax) {
    for (int i = 0; i < nglocal; i++) vector_grid[i] = 0.0;
    return;
  }

  int ngroups = collide->ngroups;

  for (int icell = 0; icell < nglocal; icell++) {
    double vmax = 0.0;
    for (int ig = 0; ig < ngroups; ig++)
      for (int jg = 0; jg < ngroups; jg++)
        vmax = MAX(vmax,vremax[icell][ig][jg]);
    vector_grid[icell] = vmax;
  }
}

/* ----------------------------------------------------------------------
   reallocate data storage if nglocal has changed
   called by init() and whenever grid changes (see Grid::grid_changed())
------------------------------------------------------------------------- */

void ComputeVremaxGrid::reallocate()
{
  if (grid->nlocal == nglocal) return;

  memory->destroy(vector_grid);
  nglocal = grid->nlocal;
  memory->create(vector_grid,nglocal,"vremax/grid:vector_grid");
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

bigint ComputeVremaxGrid::memory_usage()
{
  bigint bytes = 0;
  bytes += nglocal * sizeof(double);    // vector_grid
  return bytes;
}
