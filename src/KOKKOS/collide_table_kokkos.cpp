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

  InterpTable **all = new InterpTable*[ntot];
  for (int m = 0; m < ntot; m++) all[m] = table_at(m);
  tabdev.build(all,ntot,"collide/table/kk");
  delete [] all;

  // per-pair table indices into that concatenated list

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

  copy_sigeff_to_device();
  copy_lb_to_device();
}

/* ----------------------------------------------------------------------
   copy the effective cross section vs temperature to the device
   compute lambda/grid reads it there, so the mean free path and collision
     time which drive fix adapt and fix dt/reset follow the table on the
     device exactly as they do on the host
------------------------------------------------------------------------- */

void CollideTableKokkos::copy_sigeff_to_device()
{
  if (!sigeff || nsigma == 0) return;

  k_sigeff = DAT::tdual_float_2d("collide/table/kk:sigeff",nsigma,ntemp);
  for (int m = 0; m < nsigma; m++)
    for (int k = 0; k < ntemp; k++)
      k_sigeff.view_host()(m,k) = sigeff[m][k];
  k_sigeff.modify_host();
  k_sigeff.sync_device();
  d_sigeff = k_sigeff.view_device();

  nsigeff_kk = nsigma;
  ntemp_kk = ntemp;
  sigeff_tlo_kk = tlo;
  sigeff_tinvdelta_kk = tinvdelta;
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
  tabdev.clear_error();
  Kokkos::deep_copy(d_lb_cap,0);
  Kokkos::deep_copy(d_lb_range,0);

  CollideVSSKokkos::collisions();

  if (tabdev.check_error())
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

  // same one-time warning CollideTable::lb_weight() issues inline

  Kokkos::deep_copy(h_lb_range,d_lb_range);
  if (h_lb_range() && !lbwarn) {
    lbwarn = 1;
    error->warning(FLERR,"Collision energy is outside the Larsen-Borgnakke "
                   "normalization grid, internal energy exchange for the "
                   "tabulated pair reverts to the VSS law");
  }
}
