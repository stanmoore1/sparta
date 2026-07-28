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

#ifndef SPARTA_INTERP_TABLE_KOKKOS_H
#define SPARTA_INTERP_TABLE_KOKKOS_H

#include "interp_table.h"
#include "kokkos_type.h"

namespace SPARTA_NS {

/* ----------------------------------------------------------------------
   device-resident copy of a set of InterpTable objects

   holds the binned representation of several tables in one flat
   coefficient array, and evaluates it with exactly the expressions
   InterpTable::evaluate() and InterpTable::interpolate_row() use, so a
   KOKKOS style reproduces its host counterpart bit for bit.

   the collide and react table styles both need this, and neither can
   reach the other's copy, so it lives here rather than in either.

   TabMeta mirrors the private state of InterpTable which those two
   expressions read: the IEEE-754 bit-indexed bin lookup and the power law
   extrapolation on each end.
------------------------------------------------------------------------- */

struct InterpTableKokkos {

  struct TabMeta {
    double xlo,xhi,alo,plo,ahi,phi;
    int64_t offset;             // bin index of the first bin
    int64_t coffset;            // start of this table inside d_coeff
    int shift;                  // right shift mapping the x bits to a bin
    int nbins,ncoeff,ncol,tabstyle;
    int errlo,errhi;            // 1 if EXTRAP error applies on that end
  };

  typedef Kokkos::DualView<TabMeta*,DeviceType::array_layout,DeviceType>
    tdual_meta_1d;
  typedef tdual_meta_1d::t_dev t_meta_1d;

  int ntable;
  tdual_meta_1d k_meta;
  t_meta_1d d_meta;
  DAT::tdual_float_1d k_coeff;
  DAT::t_float_1d d_coeff;

  // set on device when a table with EXTRAP error is read out of range;
  //   device code cannot raise an error, so the host checks this after the
  //   kernel and aborts exactly as InterpTable::evaluate() would have

  DAT::t_int_scalar d_error;
  HAT::t_int_scalar h_error;

  InterpTableKokkos() : ntable(0) {}

  KOKKOS_INLINE_FUNCTION
  int bin(const TabMeta &t, double x) const {
    union { double d; uint64_t u; } v;
    v.d = x;
    int64_t k = (int64_t) (v.u >> t.shift) - t.offset;
    if (k < 0) k = 0;
    else if (k > t.nbins-1) k = t.nbins-1;
    return (int) k;
  }

  KOKKOS_INLINE_FUNCTION
  double evaluate(int m, double x) const {
    const TabMeta &t = d_meta(m);

    if (x <= t.xlo) {
      if (t.errlo) d_error() = 1;
      return t.alo * pow(x,t.plo);
    }
    if (x >= t.xhi) {
      if (t.errhi) d_error() = 1;
      return t.ahi * pow(x,t.phi);
    }

    const int64_t b = t.coffset + (int64_t) t.ncoeff*bin(t,x);
    if (t.tabstyle == 1) return d_coeff(b) + x*d_coeff(b+1);   // linear
    if (t.tabstyle == 0) return d_coeff(b);                    // lookup
    const double u = x - d_coeff(b);                           // spline
    return d_coeff(b+1) +
      u*(d_coeff(b+2) + u*(d_coeff(b+3) + u*d_coeff(b+4)));
  }

  KOKKOS_INLINE_FUNCTION
  double interpolate_row(int m, double x, double u) const {
    const TabMeta &t = d_meta(m);
    const int64_t b = t.coffset + (int64_t) t.ncol*bin(t,x);
    double f = u*t.ncol - 0.5;
    if (f <= 0.0) return d_coeff(b);
    if (f >= t.ncol-1) return d_coeff(b+t.ncol-1);
    const int j = (int) f;
    return d_coeff(b+j) + (f-j)*(d_coeff(b+j+1)-d_coeff(b+j));
  }

  // host: copy N tables, in the given order, into device memory
  // a NULL entry is allowed and left with nbins = 0; nothing indexes it

  void build(class InterpTable **tabs, int n, const char *name) {
    ntable = n;
    d_error = DAT::t_int_scalar(std::string(name)+":error");
    h_error = Kokkos::create_mirror_view(d_error);
    if (n == 0) return;

    bigint ntotal = 0;
    for (int m = 0; m < n; m++) {
      if (!tabs[m]) continue;
      int ts,nc,sh,nb;
      int64_t off;
      double alo,plo,ahi,phi,*co;
      tabs[m]->export_table(ts,nc,sh,off,nb,alo,plo,ahi,phi,co);
      ntotal += (bigint) nc*tabs[m]->ncol*nb;
    }

    k_meta = tdual_meta_1d(std::string(name)+":meta",n);
    k_coeff = DAT::tdual_float_1d(std::string(name)+":coeff",MAX(ntotal,1));

    bigint at = 0;
    for (int m = 0; m < n; m++) {
      TabMeta meta;
      memset(&meta,0,sizeof(TabMeta));
      if (tabs[m]) {
        int ts,nc,sh,nb;
        int64_t off;
        double alo,plo,ahi,phi,*co;
        tabs[m]->export_table(ts,nc,sh,off,nb,alo,plo,ahi,phi,co);
        meta.xlo = tabs[m]->xlo;
        meta.xhi = tabs[m]->xhi;
        meta.alo = alo; meta.plo = plo;
        meta.ahi = ahi; meta.phi = phi;
        meta.offset = off;
        meta.coffset = at;
        meta.shift = sh;
        meta.nbins = nb;
        meta.ncoeff = nc;
        meta.ncol = tabs[m]->ncol;
        meta.tabstyle = ts;
        meta.errlo = (tabs[m]->extrap_lo == TB_ERROR);
        meta.errhi = (tabs[m]->extrap_hi == TB_ERROR);
        const bigint sz = (bigint) nc*tabs[m]->ncol*nb;
        for (bigint k = 0; k < sz; k++) k_coeff.view_host()(at+k) = co[k];
        at += sz;
      }
      k_meta.view_host()(m) = meta;
    }

    k_meta.modify_host();
    k_meta.sync_device();
    d_meta = k_meta.view_device();
    k_coeff.modify_host();
    k_coeff.sync_device();
    d_coeff = k_coeff.view_device();
  }

  void clear_error() { if (ntable) Kokkos::deep_copy(d_error,0); }

  int check_error() {
    if (!ntable) return 0;
    Kokkos::deep_copy(h_error,d_error);
    return h_error();
  }
};

}

#endif
