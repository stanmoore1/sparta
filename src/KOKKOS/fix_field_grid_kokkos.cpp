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

#include "fix_field_grid_kokkos.h"
#include "grid.h"
#include "memory_kokkos.h"
#include "sparta_masks.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

FixFieldGridKokkos::FixFieldGridKokkos(SPARTA *sparta, int narg, char **arg) :
  FixFieldGrid(sparta, narg, arg)
{
  // the variable evaluation in compute_field() runs on the host, so this fix
  //   is a host fix.  it is not invoked through Modify -- UpdateKokkos calls
  //   compute_field() directly -- so the datamasks only describe the fact
  //   that no particle data is read or written here

  kokkos_flag = 0;
  execution_space = Host;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
}

/* ---------------------------------------------------------------------- */

FixFieldGridKokkos::~FixFieldGridKokkos()
{
}

/* ----------------------------------------------------------------------
   evaluate the per-grid-cell field on the host, then publish it to the
     device view UpdateKokkos::field_per_grid() reads
------------------------------------------------------------------------- */

void FixFieldGridKokkos::compute_field()
{
  FixFieldGrid::compute_field();

  const int nglocal = grid->nlocal;
  if (!nglocal) return;
  const int ncols = size_per_grid_cols;

  if ((int) k_array_grid.extent(0) < nglocal ||
      (int) k_array_grid.extent(1) != ncols)
    MemKK::realloc_kokkos(k_array_grid,"field/grid/kk:array_grid",nglocal,ncols);

  auto h_array_grid = k_array_grid.view_host();
  for (int i = 0; i < nglocal; i++)
    for (int j = 0; j < ncols; j++)
      h_array_grid(i,j) = array_grid[i][j];

  k_array_grid.modify_host();
  k_array_grid.sync_device();

  d_array_grid = k_array_grid.view_device();
}
