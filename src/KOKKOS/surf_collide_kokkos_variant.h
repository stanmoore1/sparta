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

#ifndef SPARTA_SURF_COLLIDE_KOKKOS_VARIANT_H
#define SPARTA_SURF_COLLIDE_KOKKOS_VARIANT_H

#include <new>
#include <cstring>

#include "kokkos_copy.h"
#include "kokkos_style_list.h"
#include "surf_collide_specular_kokkos.h"
#include "surf_collide_diffuse_kokkos.h"
#include "surf_collide_vanish_kokkos.h"
#include "surf_collide_piston_kokkos.h"
#include "surf_collide_transparent_kokkos.h"

// One slot that can hold a copy of any Kokkos surf collide style.
//
// This replaces one fixed-size array per style.  Those arrays capped every
//  style at KOKKOS_MAX_SURF_COLL_PER_TYPE instances no matter how few of the
//  other styles were in use; a slot that can hold any style turns that into a
//  single budget shared by all of them.
//
// Everything specific to the set of styles lives in the three adjacent blocks
//  below -- the style list, the union members, and the names.  They must agree
//  in order, and get() below fails to compile if the list and the union
//  disagree.  Every other site is generated from the list via kk_visit().
//
// Lifetime is managed by hand because the alternatives are polymorphic and
//  carry reference members, so the union's implicit constructors are deleted.
//  All of that is host-only: Kokkos copy constructs a functor on the host and
//  then blits the finished object to the device, so it never constructs or
//  destroys the device side image.  The copy constructor in particular has to
//  copy construct the *active* alternative, so that the Kokkos Views inside it
//  are reference counted; getting that wrong leaks device memory.

namespace SPARTA_NS {

using SurfCollideKKStyles = KKStyleList<
  SurfCollideSpecularKokkos,
  SurfCollideDiffuseKokkos,
  SurfCollideVanishKokkos,
  SurfCollidePistonKokkos,
  SurfCollideTransparentKokkos>;

class SurfCollideKKVariant {
 public:
  static constexpr int nstyles = SurfCollideKKStyles::nstyles;

  template <int I>
  using style = typename KKTypeAt<I,SurfCollideKKStyles>::type;

  int tag;                      // index into SurfCollideKKStyles, -1 if empty

  SurfCollideKKVariant() : tag(-1) {}

  SurfCollideKKVariant(const SurfCollideKKVariant &other) : tag(-1) {
    copy_construct_from(other);
  }

  SurfCollideKKVariant &operator=(const SurfCollideKKVariant &other) {
    if (this != &other) {
      destroy();
      copy_construct_from(other);
    }
    return *this;
  }

  ~SurfCollideKKVariant() { destroy(); }

  // the style object held by this slot, for kk_visit()

  template <int I>
  KOKKOS_INLINE_FUNCTION style<I> &get() { return slot<I>().obj; }

  template <int I>
  KOKKOS_INLINE_FUNCTION const style<I> &get() const { return slot<I>().obj; }

  // style name -> list index, or -1 if the style has no Kokkos version here

  static int style_index(const char *name) {
    for (int i = 0; i < nstyles; i++)
      if (strcmp(name,style_names[i]) == 0) return i;
    return -1;
  }

  // make this slot hold style t, reusing it if it already does

  void ensure(int t, SPARTA *sparta) {
    if (tag == t) return;
    destroy();
    emplace(t,sparta);
  }

  // refresh this slot from the live surf collide model it mirrors

  void assign(SurfCollide *orig) { assign_at(orig); }

 private:

  union U {
    U() {}
    ~U() {}
    KKCopy<SurfCollideSpecularKokkos> specular;
    KKCopy<SurfCollideDiffuseKokkos> diffuse;
    KKCopy<SurfCollideVanishKokkos> vanish;
    KKCopy<SurfCollidePistonKokkos> piston;
    KKCopy<SurfCollideTransparentKokkos> transparent;
  } u;

  static constexpr const char *style_names[nstyles] = {
    "specular","diffuse","vanish","piston","transparent"
  };

  // the one place the list order is tied to the union members; the declared
  //  return type makes a mismatch a compile error rather than a silent bug

  template <int I>
  KOKKOS_INLINE_FUNCTION KKCopy<style<I>> &slot() {
    if constexpr (I == 0) return u.specular;
    else if constexpr (I == 1) return u.diffuse;
    else if constexpr (I == 2) return u.vanish;
    else if constexpr (I == 3) return u.piston;
    else return u.transparent;
  }

  template <int I>
  KOKKOS_INLINE_FUNCTION const KKCopy<style<I>> &slot() const {
    if constexpr (I == 0) return u.specular;
    else if constexpr (I == 1) return u.diffuse;
    else if constexpr (I == 2) return u.vanish;
    else if constexpr (I == 3) return u.piston;
    else return u.transparent;
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
  void copy_construct_from(const SurfCollideKKVariant &other) {
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
  void assign_at(SurfCollide *orig) {
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
