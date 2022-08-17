/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.sandia.gov
   Steve Plimpton, sjplimp@sandia.gov, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#ifndef SPARTA_KK_DEVICE_COPY_H
#define SPARTA_KK_DEVICE_COPY_H

#include "kokkos_type.h"

// copy a class pointer from host to device, with "placement new" workaround
//  to enable virtual functions on the GPU

namespace SPARTA_NS {

template <class ClassStyle>
class KKDeviceCopy {
 public:
  ClassStyle* d_ptr = NULL;

  KKDeviceCopy(ClassStyle* ptr)
  {
    CopyFunctor f = CopyFunctor(ptr);
    Kokkos::parallel_for("create_object_on_device",
      Kokkos::RangePolicy<DeviceType>(0,1),f);
    d_ptr = f.d_ptr;
  }

  ~KKDeviceCopy() {}

  struct CopyFunctor {
   public:
    typedef DeviceType device_type;
    ClassStyle* h_ptr = NULL;
    ClassStyle* d_ptr = NULL;

    CopyFunctor(ClassStyle* h_ptr_in)
    {
      h_ptr = h_ptr_in;
      d_ptr = static_cast<ClassStyle*>(Kokkos::kokkos_malloc<DeviceType>(
               "create_object_on_device_impl", sizeof(ClassStyle)));
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (const int &i) const {
      new (d_ptr) ClassStyle(*h_ptr);
      d_ptr->copy = 1;
    }
  };

};

}

#endif

/* ERROR/WARNING messages:

*/
