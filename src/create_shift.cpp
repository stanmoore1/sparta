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

#include "stdlib.h"
#include "create_shift.h"
#include "domain.h"
#include "error.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

CreateShift::CreateShift(SPARTA *sparta) : Pointers(sparta) {}

/* ----------------------------------------------------------------------
   set shifted-periodic boundary offsets
   args are 3 fractions of the box length in each dimension, range [-1,1]
   a particle crossing a periodic face is remapped to the opposite face and
     its transverse coords are offset by the shift in those dimensions
------------------------------------------------------------------------- */

void CreateShift::command(int narg, char **arg)
{
  if (!domain->box_exist)
    error->all(FLERR,"Cannot create_shift before simulation box is defined");

  if (narg != 3) error->all(FLERR,"Illegal create_shift command");

  double xfrac = atof(arg[0]);
  double yfrac = atof(arg[1]);
  double zfrac = atof(arg[2]);

  if (xfrac < -1.0 || xfrac > 1.0 ||
      yfrac < -1.0 || yfrac > 1.0 ||
      zfrac < -1.0 || zfrac > 1.0)
    error->all(FLERR,"Create_shift fractions must be between -1.0 and 1.0");

  if (domain->dimension == 2 && zfrac != 0.0)
    error->all(FLERR,"Create_shift z fraction must be 0.0 for 2d simulation");

  domain->shift[0] = xfrac * (domain->boxhi[0] - domain->boxlo[0]);
  domain->shift[1] = yfrac * (domain->boxhi[1] - domain->boxlo[1]);
  domain->shift[2] = zfrac * (domain->boxhi[2] - domain->boxlo[2]);

  domain->print_shift("Created ");
}
