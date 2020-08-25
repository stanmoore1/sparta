/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.sandia.gov
   Steve Plimpton, sjplimp@sandia.gov, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#include "math.h"
#include "fix_vibmode_kokkos.h"
#include "update.h"
#include "particle.h"
#include "collide.h"
#include "comm.h"
#include "random_mars.h"
#include "random_park.h"
#include "math_const.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

enum{INT,DOUBLE};                      // several files
enum{NONE,DISCRETE,SMOOTH};            // several files

/* ---------------------------------------------------------------------- */

FixVibmodeKokkos::FixVibmodeKokkos(SPARTA *sparta, int narg, char **arg) :
  FixVibmode(sparta, narg, arg),
  rand_pool(12345 + comm->me
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            )
{

}

/* ---------------------------------------------------------------------- */

FixVibmodeKokkos::~FixVibmodeKokkos()
{
#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.destroy();
#endif
}

/* ---------------------------------------------------------------------- */

//void FixVibmodeKokkos::init()
//{
//}

/* ----------------------------------------------------------------------
   called when a particle with index is created
   populate all vibrational modes and set evib = sum of mode energies
------------------------------------------------------------------------- */

void FixVibmodeKokkos::add_particle(int index, double temp_thermal,
                                    double temp_rot, double temp_vib,
                                    double *vstream)
{
  //sync //////////////////////////
  FixVibmode::add_particle(index, temp_thermal, temp_rot, temp_vib, vstream);
  //modify /////////////////////////
}

