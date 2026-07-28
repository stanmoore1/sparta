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

#include "surf_collide_specular_kokkos.h"
#include "fix.h"
#include "modify.h"
#include "error.h"
#include "sparta_masks.h"
#include "surf.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

SurfCollideSpecularKokkos::SurfCollideSpecularKokkos(SPARTA *sparta, int narg, char **arg) :
  SurfCollideSpecular(sparta, narg, arg),
  fix_ambi_kk_copy(sparta),
  fix_vibmode_kk_copy(sparta)
{
  kokkosable = 1;

  // use 1D view for scalars to reduce GPU memory operations

  d_scalars = t_int_2("surf_collide_specular:scalars");
  d_nsingle = Kokkos::subview(d_scalars,0);
  d_nreact_one = Kokkos::subview(d_scalars,1);

  h_scalars = t_host_int_2("surf_collide_specular:scalars_mirror");
  h_nsingle = Kokkos::subview(h_scalars,0);
  h_nreact_one = Kokkos::subview(h_scalars,1);
}

/* ---------------------------------------------------------------------- */

SurfCollideSpecularKokkos::SurfCollideSpecularKokkos(SPARTA *sparta) :
  SurfCollideSpecular(sparta),
  fix_ambi_kk_copy(sparta),
  fix_vibmode_kk_copy(sparta)
{
  copy = 1;
}

/* ---------------------------------------------------------------------- */

void SurfCollideSpecularKokkos::init()
{
  SurfCollideSpecular::init();

  ambi_flag = vibmode_flag = 0;
  if (modify->n_update_custom) {
    for (int ifix = 0; ifix < modify->nfix; ifix++) {
      if (strcmp(modify->fix[ifix]->style,"ambipolar") == 0) {
        ambi_flag = 1;
        FixAmbipolar *afix = (FixAmbipolar *) modify->fix[ifix];
        if (!afix->kokkos_flag)
          error->all(FLERR,"Must use fix ambipolar/kk when Kokkos is enabled");
        afix_kk = (FixAmbipolarKokkos*)afix;
      } else if (strcmp(modify->fix[ifix]->style,"vibmode") == 0) {
        vibmode_flag = 1;
        FixVibmode *vfix = (FixVibmode *) modify->fix[ifix];
        if (!vfix->kokkos_flag)
          error->all(FLERR,"Must use fix vibmode/kk when Kokkos is enabled");
        vfix_kk = (FixVibmodeKokkos*)vfix;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void SurfCollideSpecularKokkos::pre_collide()
{
  if (ambi_flag) {
    afix_kk->pre_update_custom_kokkos();
    fix_ambi_kk_copy.copy(afix_kk);
  }

  if (vibmode_flag) {
    vfix_kk->pre_update_custom_kokkos();
    fix_vibmode_kk_copy.copy(vfix_kk);
  }

  if (surf->nsr > KOKKOS_MAX_TOT_SURF_REACT)
    error->all(FLERR,"Kokkos currently supports a limited number of surface reaction methods");

  for (int n = 0; n < surf->nsr; n++) {
    if (!surf->sr[n]->kokkosable)
      error->all(FLERR,"Must use Kokkos-enabled surface reaction method with Kokkos");
    int type = SurfReactKKVariant::style_index(surf->sr[n]->style);
    if (type < 0)
      error->all(FLERR,"Unknown Kokkos surface reaction method");
    sr_copies[n].ensure(type,sparta);
    sr_copies[n].assign(surf->sr[n]);
    kk_visit(sr_copies[n],[](auto &sr) { sr.pre_react(); });
  }

  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK);
  d_particles = particle_kk->k_particles.view_device();

  Kokkos::deep_copy(d_scalars,0);
}

/* ---------------------------------------------------------------------- */

void SurfCollideSpecularKokkos::post_collide()
{
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  if (ambi_flag || vibmode_flag) particle_kk->modify(Device,CUSTOM_MASK);

  Kokkos::deep_copy(h_scalars,d_scalars);

  int m = surf->find_collide(id);
  auto sc = surf->sc[m]; // can't modify the copy directly, use the original
  sc->nsingle += h_nsingle();
  surf->nreact_one += h_nreact_one();
  d_particles = {};

  // pre_collide() runs on this KKCopy and has each active surf react model
  //  retain a reference to the particle list.  Release it before the next
  //  copy() blits over the member: a blit does not release, so the reference
  //  would be orphaned and its allocation never freed

  for (int n = 0; n < surf->nsr; n++) {
    if (sr_type_list[n] == 0) sr_kk_global_copy[sr_map[n]].obj.post_react();
    else sr_kk_prob_copy[sr_map[n]].obj.post_react();
  }
}

/* ---------------------------------------------------------------------- */

void SurfCollideSpecularKokkos::backup()
{
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  d_particles = particle_kk->k_particles.view_device();

  for (int n = 0; n < surf->nsr; n++)
    kk_visit(sr_copies[n],[](auto &sr) { sr.backup(); });
}

/* ---------------------------------------------------------------------- */

void SurfCollideSpecularKokkos::restore()
{
  for (int n = 0; n < surf->nsr; n++)
    kk_visit(sr_copies[n],[](auto &sr) { sr.restore(); });

  Kokkos::deep_copy(d_scalars,0);
}
