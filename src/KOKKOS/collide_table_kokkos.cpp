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

#include "string.h"
#include "collide_table_kokkos.h"
#include "interp_table.h"
#include "particle.h"
#include "react.h"
#include "comm.h"
#include "error.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

CollideTableKokkos::CollideTableKokkos(SPARTA *sparta, int narg, char **arg) :
  CollideVSSKokkos(sparta, narg, arg, 0)
{
  // the parent read the parameter file with the plain VSS parser, which
  //   skipped the table directives; parse the table arguments and read it
  //   again so the tables are built

  setup_tables(narg,arg);
}

/* ---------------------------------------------------------------------- */

CollideTableKokkos::~CollideTableKokkos()
{
}

/* ---------------------------------------------------------------------- */

void CollideTableKokkos::init()
{
  CollideVSSKokkos::init();

  // CollideVSSKokkos::init() reimplements CollideVSS::init() rather than
  //   calling it, so the table setup which CollideTable::init() does has to
  //   be repeated here

  build_sigeff();
  build_lbratio();

  // refuse what the device path does not implement, rather than silently
  //   producing a different answer from the non-KOKKOS build

  if (nscatter)
    error->all(FLERR,"Collide table/kk does not support scatter tables");
  if (nlbpair)
    error->all(FLERR,"Collide table/kk does not support a tabulated pair "
               "with internal energy");
  if (react)
    error->all(FLERR,"Collide table/kk does not support chemistry");

  for (int m = 0; m < nsigma; m++)
    if (sigma_tab[m]->extrap_lo == TB_ERROR ||
        sigma_tab[m]->extrap_hi == TB_ERROR)
      error->all(FLERR,"Collide table/kk does not support EXTRAP error");
  for (int m = 0; m < nalpha; m++)
    if (alpha_tab[m]->extrap_lo == TB_ERROR ||
        alpha_tab[m]->extrap_hi == TB_ERROR)
      error->all(FLERR,"Collide table/kk does not support EXTRAP error");

  copy_tables_to_device();
}

/* ----------------------------------------------------------------------
   copy the binned tables built on the host into device memory
   the cross section tables come first, then the alpha tables, so one
     TabMeta array and one flat coefficient array serve both
------------------------------------------------------------------------- */

void CollideTableKokkos::copy_tables_to_device()
{
  const int ntot = nsigma + nalpha;
  if (ntot == 0) return;

  // first pass sizes the flat coefficient array

  bigint ncoeff_total = 0;
  for (int m = 0; m < ntot; m++) {
    InterpTable *t = (m < nsigma) ? sigma_tab[m] : alpha_tab[m-nsigma];
    int tabstyle_,ncoeff_,shift_,nbins_;
    int64_t offset_;
    double alo_,plo_,ahi_,phi_,*coeff_;
    t->export_table(tabstyle_,ncoeff_,shift_,offset_,nbins_,
                    alo_,plo_,ahi_,phi_,coeff_);
    ncoeff_total += (bigint) ncoeff_*nbins_;
  }

  k_tabmeta = tdual_tabmeta_1d("collide/table/kk:tabmeta",ntot);
  k_tabcoeff = DAT::tdual_float_1d("collide/table/kk:tabcoeff",ncoeff_total);

  bigint offset_into_coeff = 0;
  for (int m = 0; m < ntot; m++) {
    InterpTable *t = (m < nsigma) ? sigma_tab[m] : alpha_tab[m-nsigma];
    int tabstyle_,ncoeff_,shift_,nbins_;
    int64_t offset_;
    double alo_,plo_,ahi_,phi_,*coeff_;
    t->export_table(tabstyle_,ncoeff_,shift_,offset_,nbins_,
                    alo_,plo_,ahi_,phi_,coeff_);

    TabMeta meta;
    meta.xlo = t->xlo;
    meta.xhi = t->xhi;
    meta.alo = alo_;
    meta.plo = plo_;
    meta.ahi = ahi_;
    meta.phi = phi_;
    meta.offset = offset_;
    meta.coffset = offset_into_coeff;
    meta.shift = shift_;
    meta.nbins = nbins_;
    meta.ncoeff = ncoeff_;

    // tab_evaluate() branches on 0 = lookup, 1 = linear, else spline, which
    //   is the order of the TB_ enum

    meta.tabstyle = tabstyle_;
    k_tabmeta.view_host()(m) = meta;

    const bigint n = (bigint) ncoeff_*nbins_;
    for (bigint k = 0; k < n; k++)
      k_tabcoeff.view_host()(offset_into_coeff+k) = coeff_[k];
    offset_into_coeff += n;
  }

  k_tabmeta.modify_host();
  k_tabmeta.sync_device();
  d_tabmeta = k_tabmeta.view_device();

  k_tabcoeff.modify_host();
  k_tabcoeff.sync_device();
  d_tabcoeff = k_tabcoeff.view_device();

  // per-pair table indices, with the alpha tables shifted past the sigma ones

  const int ns = particle->nspecies;
  k_sigidx = DAT::tdual_int_2d("collide/table/kk:sigidx",ns,ns);
  k_alphaidx = DAT::tdual_int_2d("collide/table/kk:alphaidx",ns,ns);
  for (int i = 0; i < ns; i++)
    for (int j = 0; j < ns; j++) {
      k_sigidx.view_host()(i,j) = sigma_index[i][j];
      const int a = alpha_index[i][j];
      k_alphaidx.view_host()(i,j) = (a < 0) ? -1 : nsigma + a;
    }
  k_sigidx.modify_host();
  k_sigidx.sync_device();
  d_sigidx = k_sigidx.view_device();
  k_alphaidx.modify_host();
  k_alphaidx.sync_device();
  d_alphaidx = k_alphaidx.view_device();

  ntab_kk = nsigma;
  nalphatab_kk = nalpha;
}
