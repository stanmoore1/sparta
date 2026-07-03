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

#include "string.h"
#include "compute_gas_attempt_grid.h"
#include "grid.h"
#include "update.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

ComputeGasAttemptGrid::ComputeGasAttemptGrid(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute gas/attempt/grid command");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute gas/attempt/grid group ID does not exist");
  groupbit = grid->bitmask[igroup];

  // setup

  per_grid_flag = 1;
  size_per_grid_cols = 0;

  gas_tally_flag = 1;         // triggers Collide to invoke attempt_tally() per cell
  timeflag = 1;               // tells Collide which timesteps to invoke attempt_tally()

  nglocal = 0;
  vector_grid = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeGasAttemptGrid::~ComputeGasAttemptGrid()
{
  if (copy || copymode) return;

  memory->destroy(vector_grid);
}

/* ---------------------------------------------------------------------- */

void ComputeGasAttemptGrid::init()
{
  reallocate();
}

/* ----------------------------------------------------------------------
   no operations here, since compute results are stored in tally array
   just used by callers to indicate compute was used
   enables prediction of next step when update needs to tally
------------------------------------------------------------------------- */

void ComputeGasAttemptGrid::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;
}

/* ----------------------------------------------------------------------
   called by Update before timesteps which invoke attempt_tally()
---------------------------------------------------------------------- */

void ComputeGasAttemptGrid::clear()
{
  cinfo = grid->cinfo;
  memset(vector_grid,0,nglocal*sizeof(double));
}

/* ----------------------------------------------------------------------
   tally the number of collision attempts made in icell on this timestep
   nattempt = # of collision attempts for one cell or one group pair in cell
     Collide calls this once per cell (single group)
     or once per group pair (multiple groups), so tallies accumulate
------------------------------------------------------------------------- */

void ComputeGasAttemptGrid::attempt_tally(int icell, int nattempt)
{
  // skip if icell not in grid group

  if (!(cinfo[icell].mask & groupbit)) return;

  vector_grid[icell] += nattempt;
}

/* ----------------------------------------------------------------------
   reallocate data storage if nglocal has changed
   called by init() and whenever grid changes
------------------------------------------------------------------------- */

void ComputeGasAttemptGrid::reallocate()
{
  if (grid->nlocal == nglocal) return;

  memory->destroy(vector_grid);
  nglocal = grid->nlocal;
  memory->create(vector_grid,nglocal,"gas/attempt/grid:vector_grid");

  // clear counts b/c may be accessed before tallying is done
  //   e.g. on initial timestep of a new run, e.g. by dump grid
  //   this is different than compute_grid.cpp b/c compute_per_grid() is a no-op
  // also note if load-balancing is done, tallies will be lost
  //   would need to implement (un)pack_grid_one() to avoid this

  memset(vector_grid,0,nglocal*sizeof(double));
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

bigint ComputeGasAttemptGrid::memory_usage()
{
  bigint bytes = 0;
  bytes += nglocal * sizeof(double);    // vector_grid
  return bytes;
}
