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

#ifndef SPARTA_SURF_REACT_KOKKOS_VARIANT_H
#define SPARTA_SURF_REACT_KOKKOS_VARIANT_H

#include <new>
#include <cstring>

#include "kokkos_copy.h"
#include "kokkos_style_list.h"
#include "surf_react_global_kokkos.h"
#include "surf_react_prob_kokkos.h"

// One slot that can hold a copy of any Kokkos surf reaction style; the surf
//  collide equivalent of surf_collide_kokkos_variant.h, and the same rules
//  apply.  See that header for why the union is hand written, why lifetime is
//  managed explicitly, and why the copy constructor matters.
//
// This replaces a pair of fixed-size arrays that was duplicated verbatim into
//  four host classes (surf_collide_{specular,diffuse,piston}_kokkos and
//  compute_surf_kokkos), each with two constructors repeating the same
//  initializer.  The sr_map copy-paste bug fixed earlier on this branch lived
//  in all four copies of that loop.

namespace SPARTA_NS {

using SurfReactKKStyles = KKStyleList<
  SurfReactGlobalKokkos,
  SurfReactProbKokkos>;

class SurfReactKKVariant {
 public:
  static constexpr int nstyles = SurfReactKKStyles::nstyles;

  template <int I>
  using style = typename KKTypeAt<I,SurfReactKKStyles>::type;

  int tag;                      // index into SurfReactKKStyles, -1 if empty

  SurfReactKKVariant() : tag(-1) {}

  SurfReactKKVariant(const SurfReactKKVariant &other) : tag(-1) {
    copy_construct_from(other);
  }

  SurfReactKKVariant &operator=(const SurfReactKKVariant &other) {
    if (this != &other) {
      destroy();
      copy_construct_from(other);
    }
    return *this;
  }

  ~SurfReactKKVariant() { destroy(); }

  template <int I>
  KOKKOS_INLINE_FUNCTION style<I> &get() { return slot<I>().obj; }

  template <int I>
  KOKKOS_INLINE_FUNCTION const style<I> &get() const { return slot<I>().obj; }

  static int style_index(const char *name) {
    for (int i = 0; i < nstyles; i++)
      if (strcmp(name,style_names[i]) == 0) return i;
    return -1;
  }

  void ensure(int t, SPARTA *sparta) {
    if (tag == t) return;
    destroy();
    emplace(t,sparta);
  }

  void assign(SurfReact *orig) { assign_at(orig); }

 private:

  union U {
    U() {}
    ~U() {}
    KKCopy<SurfReactGlobalKokkos> global;
    KKCopy<SurfReactProbKokkos> prob;
  } u;

  static constexpr const char *style_names[nstyles] = {"global","prob"};

  template <int I>
  KOKKOS_INLINE_FUNCTION KKCopy<style<I>> &slot() {
    if constexpr (I == 0) return u.global;
    else return u.prob;
  }

  template <int I>
  KOKKOS_INLINE_FUNCTION const KKCopy<style<I>> &slot() const {
    if constexpr (I == 0) return u.global;
    else return u.prob;
  }

  template <int I = 0>
  void emplace(int t, SPARTA *sparta) {
    if constexpr (I < nstyles) {
      if (t == I) {
        new (&slot<I>()) KKCopy<style<I>>(sparta);
        tag = I;
        return;
      }
      emplace<I+1>(t,sparta);
    }
  }

  template <int I = 0>
  void copy_construct_from(const SurfReactKKVariant &other) {
    if constexpr (I < nstyles) {
      if (other.tag == I) {
        new (&slot<I>()) KKCopy<style<I>>(other.slot<I>());
        tag = I;
        return;
      }
      copy_construct_from<I+1>(other);
    }
  }

  template <int I = 0>
  void destroy() {
    if constexpr (I < nstyles) {
      if (tag == I) {
        slot<I>().~KKCopy<style<I>>();
        tag = -1;
        return;
      }
      destroy<I+1>();
    }
  }

  template <int I = 0>
  void assign_at(SurfReact *orig) {
    if constexpr (I < nstyles) {
      if (tag == I) {
        slot<I>().copy((style<I> *) orig);
        return;
      }
      assign_at<I+1>(orig);
    }
  }

};

}

#endif

/* ERROR/WARNING messages:

*/
