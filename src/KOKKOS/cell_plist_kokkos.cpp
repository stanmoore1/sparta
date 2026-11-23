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

#include "cell_plist_kokkos.h"
#include "memory_kokkos.h"

using namespace SPARTA_NS;

/* ----------------------------------------------------------------------
   capacity of a flat cell holding count particles
   extra = react/extra style multiplier, pad = fixed slack
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
static int flat_capacity(const int count, const double extra, const int pad)
{
  int cap = count;
  if (extra > 1.0) cap += static_cast<int> ((extra-1.0)*count + 0.999999);
  return cap + pad;
}

/* ---------------------------------------------------------------------- */

void CellPList::realloc_dense(int n, int w)
{
  flat = 0;
  d_flat = DAT::t_int_1d();
  d_offset = DAT::t_int_1d();
  ncell = nchunk = nslot = 0;
  d_dense = DAT::t_int_2d();
  MemKK::realloc_kokkos(d_dense,"particle:plist",n,w);
}

/* ---------------------------------------------------------------------- */

void CellPList::grow_dense(int n, int w)
{
  Kokkos::resize(d_dense,n,w);
}

/* ----------------------------------------------------------------------
   lay out n cells back to back from d_count
   d_flat reallocated only to grow, with 10% headroom, so the steady state
     does not reallocate every timestep
------------------------------------------------------------------------- */

void CellPList::build_flat(DAT::t_int_1d d_count, int n, double extra,
                           int cshift_in)
{
  flat = 1;
  d_dense = DAT::t_int_2d();
  cshift = cshift_in;
  ncell = n;
  const int csize = 1 << cshift;
  nchunk = (n + csize - 1) >> cshift;

  if (int(d_offset.extent(0)) < nchunk+1) {
    d_offset = DAT::t_int_1d();
    MemKK::realloc_kokkos(d_offset,"particle:plist_offset",nchunk+1);
  }

  auto l_offset = d_offset;
  auto l_count = d_count;
  const int l_pad = pad;
  const double l_extra = extra;
  const int l_cshift = cshift;
  const int l_nchunk = nchunk;

  // region of chunk b = C * fullest capacity among its cells
  // scan in 64 bit, so overflow of the int offsets is caught, not wrapped

  bigint total = 0;
  Kokkos::parallel_scan("CellPList::build_flat",
    Kokkos::RangePolicy<DeviceType>(0,nchunk+1),
    KOKKOS_LAMBDA(const int b, bigint &sum, const bool final) {
      if (final) l_offset(b) = static_cast<int> (sum);
      if (b < l_nchunk) {
        const int lo = b << l_cshift;
        const int hi = MIN(lo + (1 << l_cshift),n);
        int w = 0;
        for (int icell = lo; icell < hi; icell++)
          w = MAX(w,flat_capacity(l_count(icell),l_extra,l_pad));
        sum += (bigint) w << l_cshift;
      }
    },total);

  if (total > MAXSMALLINT) {
    nslot = -1;
    return;
  }
  nslot = static_cast<int> (total);

  if (int(d_flat.extent(0)) < nslot) {
    bigint want = (bigint) (1.1*nslot) + 1;
    if (want > MAXSMALLINT) want = MAXSMALLINT;
    d_flat = DAT::t_int_1d();
    MemKK::realloc_kokkos(d_flat,"particle:plist",(int) want);
  }
}

/* ----------------------------------------------------------------------
   re-lay out the same ncell cells with a larger pad, keeping the first
     d_count(i) entries of each cell
------------------------------------------------------------------------- */

void CellPList::regrow_flat(DAT::t_int_1d d_count, double extra, int newpad)
{
  const CellPList old = *this;

  // new arrays, so build_flat() cannot reuse the old ones in place

  d_offset = DAT::t_int_1d();
  d_flat = DAT::t_int_1d();
  pad = newpad;
  build_flat(d_count,old.ncell,extra,old.cshift);
  if (nslot < 0) return;

  const CellPList cur = *this;
  auto l_count = d_count;

  Kokkos::parallel_for("CellPList::regrow_flat",
    Kokkos::RangePolicy<DeviceType>(0,ncell),
    KOKKOS_LAMBDA(const int icell) {
      const int np = l_count(icell);
      for (int i = 0; i < np; i++) cur(icell,i) = old(icell,i);
    });
}

/* ----------------------------------------------------------------------
   copy contents and layout of src into this list, which must have been
     produced by src.clone()
------------------------------------------------------------------------- */

void CellPList::deep_copy_from(const CellPList &src)
{
  flat = src.flat;
  cshift = src.cshift;
  ncell = src.ncell;
  nchunk = src.nchunk;
  nslot = src.nslot;
  pad = src.pad;
  if (!flat) {
    Kokkos::deep_copy(d_dense,src.d_dense);
  } else {
    Kokkos::deep_copy(d_offset,src.d_offset);
    Kokkos::deep_copy(d_flat,src.d_flat);
  }
}

/* ---------------------------------------------------------------------- */

CellPList CellPList::clone(const char *name) const
{
  CellPList c = *this;
  if (!flat) {
    c.d_dense = DAT::t_int_2d(Kokkos::view_alloc(std::string(name),Kokkos::WithoutInitializing),
                              d_dense.extent(0),d_dense.extent(1));
    Kokkos::deep_copy(c.d_dense,d_dense);
  } else {
    c.d_offset = DAT::t_int_1d(Kokkos::view_alloc(std::string(name),Kokkos::WithoutInitializing),
                               d_offset.extent(0));
    c.d_flat = DAT::t_int_1d(Kokkos::view_alloc(std::string(name),Kokkos::WithoutInitializing),
                             d_flat.extent(0));
    Kokkos::deep_copy(c.d_offset,d_offset);
    Kokkos::deep_copy(c.d_flat,d_flat);
  }
  return c;
}

/* ---------------------------------------------------------------------- */

void CellPList::clear()
{
  d_dense = DAT::t_int_2d();
  d_flat = DAT::t_int_1d();
  d_offset = DAT::t_int_1d();
  ncell = nchunk = nslot = 0;
}

/* ---------------------------------------------------------------------- */

double CellPList::memory_usage() const
{
  return d_dense.span()*sizeof(int) + d_flat.span()*sizeof(int) +
    d_offset.span()*sizeof(int);
}
