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

#ifndef SPARTA_KK_STYLE_LIST_H
#define SPARTA_KK_STYLE_LIST_H

#include <type_traits>

// Virtual functions are not portably available in device code, so a class
//  that must dispatch to one of several concrete styles inside a kernel has
//  to switch on a tag instead.  Writing that switch out by hand, once per
//  use, is what this header exists to avoid: declare the styles once as a
//  KKStyleList and generate every site from it.
//
// kk_visit() expands, via if constexpr, to exactly the chain of
//  "if (tag == 0) ... else if (tag == 1) ..." that would otherwise be typed
//  out.  Code generation and numerics are identical; the difference is that
//  adding or reordering a style is a one-line change instead of ten.
//
// The visitor returns void and the caller captures any result by reference,
//  which keeps one form usable for both the device dispatch (which yields a
//  particle) and the host lifecycle calls (which yield nothing).

namespace SPARTA_NS {

template <class... Styles>
struct KKStyleList {
  static constexpr int nstyles = sizeof...(Styles);
};

// KKTypeAt<I,List>::type -- the Ith style in the list

template <int I, class List>
struct KKTypeAt;

template <int I, class Style, class... Rest>
struct KKTypeAt<I,KKStyleList<Style,Rest...>>
  : KKTypeAt<I-1,KKStyleList<Rest...>> {};

template <class Style, class... Rest>
struct KKTypeAt<0,KKStyleList<Style,Rest...>> {
  using type = Style;
};

// call f with the alternative held by v, chosen by v.tag.  v must provide a
//  static nstyles and a template get<I>().  works for const and non-const v,
//  so the same form serves the const device kernel and the host lifecycle
//  calls.  does nothing if v holds no style

template <int I = 0, class Variant, class Functor>
KOKKOS_INLINE_FUNCTION
void kk_visit(Variant &v, Functor &&f)
{
  if constexpr (I < std::decay_t<Variant>::nstyles) {
    if (v.tag == I) {
      f(v.template get<I>());
      return;
    }
    kk_visit<I+1>(v,f);
  }
}

}

#endif

/* ERROR/WARNING messages:

*/
