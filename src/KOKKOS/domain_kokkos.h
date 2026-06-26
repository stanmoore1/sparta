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

#ifndef SPARTA_DOMAIN_KOKKOS_H
#define SPARTA_DOMAIN_KOKKOS_H

#include "domain.h"
#include "kokkos_type.h"

namespace SPARTA_NS {

enum{XLO,XHI,YLO,YHI,ZLO,ZHI,INTERIOR};         // several files
enum{PERIODIC,OUTFLOW,REFLECT,SURFACE,AXISYM};  // several files


class DomainKokkos : public Domain {
 public:

  DomainKokkos(class SPARTA *);
  ~DomainKokkos();

/* ----------------------------------------------------------------------
   particle ip hits global boundary on face in icell
   called by Update::move()
   xnew = final position of particle at end of move
   return boundary type of global boundary
   return reaction = index of reaction (1 to N) that took place, 0 = no reaction
   if needed, update particle x,v,xnew due to collision
------------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int collide_kokkos(Particle::OnePart *&ip, int face, double* lo, double* hi, double *xnew,
                     /*double &dtremain,*/ int &reaction) const
  {
    //jp = NULL;
    reaction = 0;

    switch (bflag[face]) {

      // outflow boundary, particle deleted by caller

      case OUTFLOW:
        return OUTFLOW;

      // periodic boundary
      // set x to be on periodic box face
      // adjust xnew by periodic box length

      case PERIODIC:
      {
        double *x = ip->x;
        int dim = face / 2;            // 0,1,2 = crossing x,y,z dimension
        int side = face % 2;           // 0 = lo face, 1 = hi face

        if (side == 0) {
          x[dim] = boxhi[dim];
          xnew[dim] += prd[dim];
        } else {
          x[dim] = boxlo[dim];
          xnew[dim] -= prd[dim];
        }

        // shifted-periodic offset, see create_shift / Domain::collide()

        double sgn = (side == 0) ? 1.0 : -1.0;
        for (int t = 0; t < 3; t++) {
          if (t == dim) continue;
          double delta = sgn*shift[t];
          if (delta == 0.0) continue;
          if (delta > 0.0) {
            if (x[t] > boxhi[t] - delta) return OUTFLOW;
          } else {
            if (x[t] < boxlo[t] - delta) return OUTFLOW;
          }
          x[t] += delta;
          xnew[t] += delta;
        }

        return PERIODIC;
      }

      // specular reflection boundary
      // adjust xnew and velocity

      case REFLECT:
      {
        double *v = ip->v;
        //double *lo = grid->cells[icell].lo;
        //double *hi = grid->cells[icell].hi;
        int dim = face / 2;

        if (face % 2 == 0) {
          xnew[dim] = lo[dim] + (lo[dim]-xnew[dim]);
          v[dim] = -v[dim];
        } else {
          xnew[dim] = hi[dim] - (xnew[dim]-hi[dim]);
          v[dim] = -v[dim];
        }

        return REFLECT;
      }

    }

    return 0;
  };

/* ----------------------------------------------------------------------
   undo the periodic remapping of particle coords
     performed by a previous call to collide()
   called by Update::move() if unable to move particle to a new cell
     that contains the remapped coords
------------------------------------------------------------------------- */
  KOKKOS_INLINE_FUNCTION
  void uncollide_kokkos(int face, double *x) const
  {
    int dim = face / 2;
    int side = face % 2;

    if (side == 0) x[dim] = boxlo[dim];
    else x[dim] = boxhi[dim];

    // undo any transverse shifted-periodic offset, see Domain::uncollide()

    double sgn = (side == 0) ? 1.0 : -1.0;
    for (int t = 0; t < 3; t++) {
      if (t == dim) continue;
      x[t] -= sgn*shift[t];
    }
  };

};

}

#endif

/* ERROR/WARNING messages:

*/
