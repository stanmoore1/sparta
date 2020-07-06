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
  Fix(sparta, narg, arg),
  rand_pool(12345 + comm->me
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            ),
{

}

/* ---------------------------------------------------------------------- */

FixVibmodeKokkos::~FixVibmodeKokkos()
{
#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.destroy();
  if (random_backup)
    delete random_backup;
  if (react_random_backup)
    delete react_random_backup;
#endif
}

/* ----------------------------------------------------------------------
   called when a particle with index is created
   populate all vibrational modes and set evib = sum of mode energies
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixVibmodeKokkos::add_particle_kokkos(int index, double temp_thermal, 
                              double temp_rot, double temp_vib, 
                              double *vstream)
{
  DAT::t_int_1d_um& d_vibmode = d_eiarray[d_ewhich[vibmodeindex]].k_view.d_view;

  int isp = d_particles[index].ispecies;
  int nmode = d_species[isp].nvibmode;

  // no modes, just return

  if (nmode == 0) return;

  // single mode, evib already set by Particle::evib()
  // just convert evib back to mode level

  if (nmode == 1) {
    d_vibmode(index,0) = static_cast<int> 
      (d_particles[index].evib / boltz / 
       d_species[isp].vibtemp[0]);
    return;
  }

  // loop over modes and populate each
  // accumlate new total evib

  int ivib;
  double evib = 0.0;

  for (int imode = 0; imode < nmode; imode++) {
    ivib = static_cast<int> (-log(random->uniform()) * temp_vib /
                             d_species[isp].vibtemp[imode]);
    d_vibmode(index,imode) = ivib;
    evib += ivib * boltz * d_species[isp].vibtemp[imode];
  }

  d_particles[index].evib = evib;
}
