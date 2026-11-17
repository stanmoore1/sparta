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

#ifndef SPARTA_CELL_PLIST_KOKKOS_H
#define SPARTA_CELL_PLIST_KOKKOS_H

#include "kokkos_type.h"
#include <string>

namespace SPARTA_NS {

/* ----------------------------------------------------------------------
   per-cell particle list, and per-cell scratch shaped like it

   layouts, chosen at run time by "package kokkos plist dense|flat|sliced"

   dense  = (ncell, width) 2d view, width = largest per-cell count plus
     padding.  every cell pays for the fullest one, so memory scales as
     ncell*width, which grid refinement can drive far past the particle count
   flat   = cells grouped in chunks of C consecutive cells, chunk b owns
     [offset(b), offset(b+1)), padded to the fullest cell in the chunk.
     entry i of the cell in lane l of the chunk is at offset(b) + i*C + l,
     so a warp of C threads, one per cell, reads C consecutive addresses.
     memory scales with the particle count
     C = 1 is CSR (plist flat), C = 32 is sliced ELLPACK (plist sliced)

   the layout is uniform over a kernel launch, so the branch on flat does
     not diverge, and all layouts are always compiled
------------------------------------------------------------------------- */

struct CellPList {
  DAT::t_int_2d_lr d_dense; // dense: (ncell, width), LayoutRight as in #683:
                            //   each cell's list is contiguous
  DAT::t_int_1d d_flat;     // flat: nslot entries
  DAT::t_int_1d d_offset;   // flat: nchunk+1 entries
  int flat;                 // 0 = dense, 1 = flat
  int cshift;               // flat: chunk size C = 1 << cshift
  int ncell;                // flat: # of cells laid out by d_offset
  int nchunk;               // flat: # of chunks
  int nslot;                // flat: offset(nchunk)
  int pad;                  // flat: fixed slack per cell, raised on overflow

  CellPList() : flat(0), cshift(0), ncell(0), nchunk(0), nslot(0), pad(0) {}

  KOKKOS_INLINE_FUNCTION
  int& operator()(const int icell, const int i) const {
    if (flat) {
      const int b = icell >> cshift;
      const int lane = icell & ((1 << cshift) - 1);
      return d_flat(size_t(d_offset(b)) + (size_t(i) << cshift) + lane);
    }
    return d_dense(icell,i);
  }

  // # of entries cell icell can hold

  KOKKOS_INLINE_FUNCTION
  int capacity(const int icell) const {
    if (flat) {
      const int b = icell >> cshift;
      return (d_offset(b+1) - d_offset(b)) >> cshift;
    }
    return d_dense.extent(1);
  }

  // dense layout: (re)allocate, contents not preserved
  void realloc_dense(int n, int w);

  // dense layout: grow, contents preserved
  void grow_dense(int n, int w);

  // flat layout: lay out n cells in chunks of 1 << cshift_in cells from
  //   per-cell counts, capacity = count + ceil(count*(extra-1)) + pad,
  //   maximized over the chunk, contents not preserved
  void build_flat(DAT::t_int_1d d_count, int n, double extra, int cshift_in);

  // flat layout: re-lay out with a larger pad, preserving the first
  //   count entries of every cell
  void regrow_flat(DAT::t_int_1d d_count, double extra, int newpad);

  void deep_copy_from(const CellPList &);
  CellPList clone(const char *) const;
  void clear();
  double memory_usage() const;
};

/* ----------------------------------------------------------------------
   per-cell, per-slot scratch with the shape of a CellPList
   dense: (ncell, width + extra) 2d view, exactly as before; Layout is its
     layout, the device default unless a member asks for another
     (d_nn_last_partner is LayoutRight, as in #683)
   flat:  each cell owns capacity(icell) + extra slots, same chunking
   resize_like() reallocates only to grow, contents are not preserved
------------------------------------------------------------------------- */

template <class T, class Layout = DeviceType::array_layout>
struct CellSlots {
  typedef Kokkos::View<T**, Layout, DeviceType> t_dense;
  typedef Kokkos::View<T*, DeviceType> t_flat;

  t_dense d_dense;
  t_flat d_flat;
  DAT::t_int_1d d_offset;
  int flat;
  int cshift;
  int extra;

  CellSlots() : flat(0), cshift(0), extra(0) {}

  // flat: chunk b holds (width_b + extra) slots per lane, so its region
  //   starts extra*C*b slots past the plist offset

  KOKKOS_INLINE_FUNCTION
  T& operator()(const int icell, const int i) const {
    if (flat) {
      const int b = icell >> cshift;
      const int lane = icell & ((1 << cshift) - 1);
      return d_flat(size_t(d_offset(b)) + ((size_t(b)*extra + i) << cshift) + lane);
    }
    return d_dense(icell,i);
  }

  KOKKOS_INLINE_FUNCTION
  int capacity(const int icell) const {
    if (flat) {
      const int b = icell >> cshift;
      return ((d_offset(b+1) - d_offset(b)) >> cshift) + extra;
    }
    return d_dense.extent(1);
  }

  // dense_width >= 0 overrides the dense width, for scratch whose
  //   per-cell bound is tracked separately from the plist (ambipolar elist)

  void resize_like(const CellPList &plist, int n, const char *name,
                   int extra_in = 0, int dense_width = -1) {
    flat = plist.flat;
    if (!flat) {
      d_flat = t_flat();
      d_offset = DAT::t_int_1d();
      extra = 0;
      const int w = (dense_width >= 0) ? dense_width :
        int(plist.d_dense.extent(1)) + extra_in;
      if (int(d_dense.extent(0)) < n || int(d_dense.extent(1)) < w) {
        d_dense = t_dense();
        d_dense = t_dense(Kokkos::view_alloc(std::string(name),Kokkos::WithoutInitializing),n,w);
      }
    } else {
      d_dense = t_dense();
      d_offset = plist.d_offset;
      cshift = plist.cshift;
      extra = extra_in;
      const size_t nn = size_t(plist.nslot) +
        (size_t(plist.nchunk)*extra_in << plist.cshift);
      if (d_flat.extent(0) < nn) {
        d_flat = t_flat();
        d_flat = t_flat(Kokkos::view_alloc(std::string(name),Kokkos::WithoutInitializing),nn);
      }
    }
  }

  void clear() { d_dense = t_dense(); d_flat = t_flat(); d_offset = DAT::t_int_1d(); }

  double memory_usage() const {
    return d_dense.span()*sizeof(T) + d_flat.span()*sizeof(T);
  }
};

/* ----------------------------------------------------------------------
   two ints per slot, indexed (icell, i, 0|1)
------------------------------------------------------------------------- */

struct CellSlots2 {
  CellSlots<int> a,b;

  KOKKOS_INLINE_FUNCTION
  int& operator()(const int icell, const int i, const int c) const {
    return c ? b(icell,i) : a(icell,i);
  }

  void resize_like(const CellPList &plist, int n, const char *name) {
    a.resize_like(plist,n,name);
    b.resize_like(plist,n,name);
  }

  void clear() { a.clear(); b.clear(); }
  double memory_usage() const { return a.memory_usage() + b.memory_usage(); }
};

/* ----------------------------------------------------------------------
   per-cell, per-group, per-slot scratch (collision group lists)
   dense: (ncell, ngroup, width) 3d view, exactly as before
   flat:  chunk b owns ngroup regions of its plist region size each,
     starting at ngroup*offset(b)
------------------------------------------------------------------------- */

struct CellGroupSlots {
  typedef Kokkos::View<int***, DeviceType> t_dense;
  typedef Kokkos::View<int*, DeviceType> t_flat;

  t_dense d_dense;
  t_flat d_flat;
  DAT::t_int_1d d_offset;
  int flat;
  int cshift;
  int ngroup;

  CellGroupSlots() : flat(0), cshift(0), ngroup(0) {}

  // flat: chunk b holds ngroup regions of width_b slots per lane

  KOKKOS_INLINE_FUNCTION
  int& operator()(const int icell, const int igroup, const int k) const {
    if (flat) {
      const int b = icell >> cshift;
      const int lane = icell & ((1 << cshift) - 1);
      const int o = d_offset(b);
      const int w = (d_offset(b+1) - o) >> cshift;
      return d_flat(size_t(ngroup)*o + ((size_t(igroup)*w + k) << cshift) + lane);
    }
    return d_dense(icell,igroup,k);
  }

  void resize_like(const CellPList &plist, int n, int ngroup_in,
                   const char *name) {
    flat = plist.flat;
    ngroup = ngroup_in;
    if (!flat) {
      d_flat = t_flat();
      d_offset = DAT::t_int_1d();
      const int w = plist.d_dense.extent(1);
      if (int(d_dense.extent(0)) < n || int(d_dense.extent(1)) < ngroup ||
          int(d_dense.extent(2)) < w) {
        d_dense = t_dense();
        d_dense = t_dense(Kokkos::view_alloc(std::string(name),Kokkos::WithoutInitializing),
                          n,ngroup,w);
      }
    } else {
      d_dense = t_dense();
      d_offset = plist.d_offset;
      cshift = plist.cshift;
      const size_t nn = size_t(plist.nslot)*ngroup;
      if (d_flat.extent(0) < nn) {
        d_flat = t_flat();
        d_flat = t_flat(Kokkos::view_alloc(std::string(name),Kokkos::WithoutInitializing),nn);
      }
    }
  }

  void clear() { d_dense = t_dense(); d_flat = t_flat(); d_offset = DAT::t_int_1d(); }

  double memory_usage() const {
    return d_dense.span()*sizeof(int) + d_flat.span()*sizeof(int);
  }
};

}

#endif
