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

#ifndef SPARTA_KK_COPY_H
#define SPARTA_KK_COPY_H

#include <type_traits>

#include "pointers.h"

// Hold a copy of a SPARTA class as a member, so that it can be captured by
//  value into a Kokkos functor and its KOKKOS_INLINE_FUNCTION methods called
//  from device code.  Virtual functions are not available on the GPU, so the
//  caller keeps one KKCopy per concrete style instead of a base class pointer.
//
// copy() is ordinary copy assignment, so Kokkos View reference counting is
//  correct by construction: View::operator= releases the handle obj held and
//  retains the new one.  obj.copy is then set so the wrapped class' destructor
//  frees nothing -- the original object it was copied from still owns all of
//  it, and obj's own View members release their references normally when obj
//  is destroyed.
//
// This is possible because Pointers defines operator= as a no-op; without it
//  the reference members of Pointers would make every SPARTA class
//  non-assignable, which is what previously forced this class to use memcpy.
//
// Nesting needs no special handling.  A wrapped class that itself contains
//  KKCopy members is copied by its own compiler-generated assignment
//  operator, which recurses into them.

namespace SPARTA_NS {

template <class ClassStyle>
class KKCopy {
 public:
  ClassStyle obj;

  KKCopy(SPARTA *sparta) : obj(construct(sparta)) {}

  void copy(const ClassStyle *orig) {
    obj = *orig;
    obj.copy = 1;
  }

 private:

  // classes whose only SPARTA* constructor is the real, allocating one
  //  provide a KKShallow overload instead; everything else already has a
  //  cheap SPARTA*-only constructor meant for exactly this

  static ClassStyle construct(SPARTA *sparta) {
    if constexpr (std::is_constructible<ClassStyle,SPARTA*,KKShallow>::value)
      return ClassStyle(sparta,KKShallow());
    else
      return ClassStyle(sparta);
  }

};

}

#endif

/* ERROR/WARNING messages:

*/
