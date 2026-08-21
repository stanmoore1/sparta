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

#include "region_intersect_kokkos.h"
#include "domain.h"
#include "particle_kokkos.h"
#include "error.h"
#include "sparta_masks.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

RegIntersectKokkos::RegIntersectKokkos(SPARTA *sparta, int narg, char **arg) :
  RegIntersect(sparta, narg, arg)
{
  kokkos_flag = 1;
  nprim = 0;
}

/* ---------------------------------------------------------------------- */

RegIntersectKokkos::~RegIntersectKokkos()
{
}

/* ----------------------------------------------------------------------
   flatten the sub-regions into one device-resident descriptor array
   each sub-region must be a Kokkos primitive: a nested composite cannot be
     expressed as a flat list under a single op, so reject it by name
------------------------------------------------------------------------- */

int RegIntersectKokkos::flatten_region_kokkos(tdual_region_prim_1d &k_prims_out, int &op)
{
  Region **regions = domain->regions;

  if ((int) k_prims.extent(0) < nregion)
    k_prims = tdual_region_prim_1d("region:prims",nregion);

  tdual_region_prim_1d k_one;
  int sub_op;

  for (int i = 0; i < nregion; i++) {
    Region *r = regions[list[i]];
    KokkosBase *rkk = dynamic_cast<KokkosBase*>(r);
    if (!rkk || !r->kokkos_flag)
      error->all(FLERR,"KOKKOS package does not (yet) support the region style "
                 "used inside region intersect");
    if (rkk->flatten_region_kokkos(k_one,sub_op) != 1 || sub_op != RKK_OP_NONE)
      error->all(FLERR,"KOKKOS package does not (yet) support a nested region "
                 "union or intersect inside region intersect");
    k_prims.view_host()[i] = k_one.view_host()[0];
  }

  k_prims.modify_host();
  k_prims.sync_device();

  nprim = nregion;
  k_prims_out = k_prims;
  op = RKK_OP_INTERSECT;
  return nprim;
}

/* ---------------------------------------------------------------------- */

void RegIntersectKokkos::match_all_kokkos(DAT::tdual_int_1d k_match_in)
{
  int op;
  tdual_region_prim_1d k_prims_local;
  flatten_region_kokkos(k_prims_local,op);

  d_match = k_match_in.view_device();
  ParticleKokkos* particleKK = (ParticleKokkos*) particle;
  particleKK->sync(Device, PARTICLE_MASK);
  d_particles = particleKK->k_particles.view_device();
  const int nlocal = particle->nlocal;

  auto l_prims = k_prims_local.view_device();
  auto l_match = d_match;
  auto l_particles = d_particles;
  const int l_nprim = nprim;
  const int l_op = op;
  const int l_interior = interior;

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,nlocal),
    KOKKOS_LAMBDA(const int &i) {
      const double x = l_particles[i].x[0];
      const double y = l_particles[i].x[1];
      const double z = l_particles[i].x[2];
      l_match[i] = region_match_kk(l_prims,l_nprim,l_op,l_interior,x,y,z);
    });
  copymode = 0;
  k_match_in.modify_device();
}
