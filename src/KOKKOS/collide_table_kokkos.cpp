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

  copy_tables_to_device();
}

/* ----------------------------------------------------------------------
   copy the binned tables built on the host into device memory
   the cross section tables come first, then the alpha tables, so one
     TabMeta array and one flat coefficient array serve both
------------------------------------------------------------------------- */

void CollideTableKokkos::copy_tables_to_device()
{
  const int ntot = nsigma + nalpha + nscatter;
  if (ntot == 0) return;

  // first pass sizes the flat coefficient array

  bigint ncoeff_total = 0;
  for (int m = 0; m < ntot; m++) {
    InterpTable *t = table_at(m);
    int tabstyle_,ncoeff_,shift_,nbins_;
    int64_t offset_;
    double alo_,plo_,ahi_,phi_,*coeff_;
    t->export_table(tabstyle_,ncoeff_,shift_,offset_,nbins_,
                    alo_,plo_,ahi_,phi_,coeff_);
    ncoeff_total += (bigint) ncoeff_*t->ncol*nbins_;
  }

  k_tabmeta = tdual_tabmeta_1d("collide/table/kk:tabmeta",ntot);
  k_tabcoeff = DAT::tdual_float_1d("collide/table/kk:tabcoeff",ncoeff_total);

  bigint offset_into_coeff = 0;
  for (int m = 0; m < ntot; m++) {
    InterpTable *t = table_at(m);
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
    meta.ncol = t->ncol;
    meta.errlo = (t->extrap_lo == TB_ERROR);
    meta.errhi = (t->extrap_hi == TB_ERROR);

    // tab_evaluate() branches on 0 = lookup, 1 = linear, else spline, which
    //   is the order of the TB_ enum

    meta.tabstyle = tabstyle_;
    k_tabmeta.view_host()(m) = meta;

    const bigint n = (bigint) ncoeff_*t->ncol*nbins_;
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
  k_scatteridx = DAT::tdual_int_2d("collide/table/kk:scatteridx",ns,ns);
  for (int i = 0; i < ns; i++)
    for (int j = 0; j < ns; j++) {
      k_sigidx.view_host()(i,j) = sigma_index[i][j];
      const int a = alpha_index[i][j];
      k_alphaidx.view_host()(i,j) = (a < 0) ? -1 : nsigma + a;
      const int c = scatter_index[i][j];
      k_scatteridx.view_host()(i,j) = (c < 0) ? -1 : nsigma + nalpha + c;
    }
  k_sigidx.modify_host();
  k_sigidx.sync_device();
  d_sigidx = k_sigidx.view_device();
  k_alphaidx.modify_host();
  k_alphaidx.sync_device();
  d_alphaidx = k_alphaidx.view_device();
  k_scatteridx.modify_host();
  k_scatteridx.sync_device();
  d_scatteridx = k_scatteridx.view_device();

  ntab_kk = nsigma;
  nalphatab_kk = nalpha;
  nscattertab_kk = nscatter;

  copy_lb_to_device();
}

/* ----------------------------------------------------------------------
   pick out table M of the concatenated sigma, alpha, scatter lists
------------------------------------------------------------------------- */

InterpTable *CollideTableKokkos::table_at(int m)
{
  if (m < nsigma) return sigma_tab[m];
  if (m < nsigma+nalpha) return alpha_tab[m-nsigma];
  return scatter_tab[m-nsigma-nalpha];
}

/* ----------------------------------------------------------------------
   copy the Larsen-Borgnakke acceptance normalization to the device
------------------------------------------------------------------------- */

void CollideTableKokkos::copy_lb_to_device()
{
  const int ns = particle->nspecies;
  k_lbidx = DAT::tdual_int_2d("collide/table/kk:lbidx",ns,ns);
  for (int i = 0; i < ns; i++)
    for (int j = 0; j < ns; j++)
      k_lbidx.view_host()(i,j) = lb_index ? lb_index[i][j] : -1;
  k_lbidx.modify_host();
  k_lbidx.sync_device();
  d_lbidx = k_lbidx.view_device();

  nlbpair_kk = nlbpair;
  if (nlbpair == 0) return;

  // the grid is the one CollideTable::build_lbratio() laid down

  nlbgrid_kk = nlbgrid;
  lblo_kk = lblo;
  lbinvdelta_kk = lbinvdelta;

  k_lbratio = DAT::tdual_float_2d("collide/table/kk:lbratio",nlbpair,nlbgrid_kk);
  k_lbmax = DAT::tdual_float_2d("collide/table/kk:lbmax",nlbpair,nlbgrid_kk);
  for (int r = 0; r < nlbpair; r++)
    for (int k = 0; k < nlbgrid_kk; k++) {
      k_lbratio.view_host()(r,k) = lbratio[r][k];
      k_lbmax.view_host()(r,k) = lbmax[r][k];
    }
  k_lbratio.modify_host();
  k_lbratio.sync_device();
  d_lbratio = k_lbratio.view_device();
  k_lbmax.modify_host();
  k_lbmax.sync_device();
  d_lbmax = k_lbmax.view_device();

  lbflag = 1;
}

/* ----------------------------------------------------------------------
   run the collision kernel, then raise any error the device recorded
------------------------------------------------------------------------- */

void CollideTableKokkos::collisions()
{
  Kokkos::deep_copy(d_tab_error,0);
  Kokkos::deep_copy(d_lb_cap,0);

  CollideVSSKokkos::collisions();

  Kokkos::deep_copy(h_tab_error,d_tab_error);
  if (h_tab_error())
    error->one(FLERR,"Value is outside the tabulated data range");

  // same one-time warning as CollideVSS::lb_capcheck()

  Kokkos::deep_copy(h_lb_cap,d_lb_cap);
  if (h_lb_cap() && !lbcapflag) {
    lbcapflag = 1;
    if (comm->me == 0)
      error->warning(FLERR,"Larsen-Borgnakke acceptance loop hit its retry "
                     "cap; the tabulated cross section is far from the VSS "
                     "parameters of that pair, so its internal energy "
                     "exchange is biased toward the VSS law.  Fit diam and "
                     "omega to the table");
  }
}
