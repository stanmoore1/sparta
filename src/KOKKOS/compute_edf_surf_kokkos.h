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

// one class implements both styles, they differ only in the binned quantity

#ifdef COMPUTE_CLASS

ComputeStyle(edf/surf/kk,ComputeEDFSurfKokkos)
ComputeStyle(adf/surf/kk,ComputeEDFSurfKokkos)

#else

#ifndef SPARTA_COMPUTE_EDF_SURF_KOKKOS_H
#define SPARTA_COMPUTE_EDF_SURF_KOKKOS_H

#include "compute_edf_surf.h"
#include "kokkos_type.h"
#include "math_extra_kokkos.h"
#include "math_const.h"

namespace SPARTA_NS {

class ComputeEDFSurfKokkos : public ComputeEDFSurf {
 public:
  ComputeEDFSurfKokkos(class SPARTA *, int, char **);
  ComputeEDFSurfKokkos(class SPARTA *);
  ~ComputeEDFSurfKokkos();
  void init();
  void reallocate();
  void clear();
  int tallyinfo(surfint *&);
  void pre_surf_tally();
  void post_surf_tally();
  void post_process_surf();

  // must match the enums in compute_edf_surf.cpp

  enum{ENERGY,ANGLE};
  enum{INCIDENT,REFLECTED};
  enum{KE,EROT,EVIB,ETOT};
  enum{IGNORE,CLAMP};

/* ----------------------------------------------------------------------
   histogram one collision with surface element isurf
   iorig = particle ip before collision
   ip,jp = particles after collision
   ip = NULL means no particles after collision
   jp = NULL means one particle after collision
   jp != NULL means two particles after collision
   must stay in step with ComputeEDFSurf::surf_tally()
------------------------------------------------------------------------- */

template <int ATOMIC_REDUCTION>
KOKKOS_INLINE_FUNCTION
void surf_tally_kk(double /*dtremain*/, int isurf, int /*icell*/, int reaction,
                   Particle::OnePart *iorig,
                   Particle::OnePart *ip, Particle::OnePart *jp) const
{
  // skip if no original particle and a reaction is taking place
  //   called by SurfReactAdsorb for on-surf reaction
  // FixEmitSurf also calls with no original particle but no reaction

  if (!iorig && reaction) return;

  // skip if isurf not in surface group

  int transparent;
  surfint surfID;
  double *norm;

  if (dim == 2) {
    if (!(d_lines(isurf).mask & groupbit)) return;
    surfID = d_lines[isurf].id;
    transparent = d_lines[isurf].transparent;
    norm = d_lines(isurf).norm;
  } else {
    if (!(d_tris(isurf).mask & groupbit)) return;
    surfID = d_tris[isurf].id;
    transparent = d_tris[isurf].transparent;
    norm = d_tris(isurf).norm;
  }

  // build list of particles to bin
  // incident = the pre-collision particle
  // reflected = the post-collision particle(s), which for a reaction are
  //   the product species, so each is binned in its own mixture group
  // a transparent surf does not alter the particle, so there is nothing
  //   reflected from it

  Particle::OnePart *plist[2];
  int np = 0;

  if (dirstyle == INCIDENT) {
    if (!iorig) return;
    plist[np++] = iorig;
  } else {
    if (transparent) return;
    if (ip) plist[np++] = ip;
    if (jp) plist[np++] = jp;
  }
  if (np == 0) return;

  // mixture group of each particle, skip particles not in the mixture
  // return before claiming a tally slot if nothing will be tallied

  int glist[2];
  int nkeep = 0;

  for (int i = 0; i < np; i++) {
    const int igroup = d_s2g(imix,plist[i]->ispecies);
    if (igroup < 0) continue;
    plist[nkeep] = plist[i];
    glist[nkeep] = igroup;
    nkeep++;
  }
  if (nkeep == 0) return;

  // thread-safe, tally array is indexed by surf and compressed later

  const int itally = isurf;
  d_tally2surf(itally) = surfID;
  d_surf2tally(isurf) = isurf;

  auto v_array_surf_tally = ScatterViewHelper<typename NeedDup<ATOMIC_REDUCTION,DeviceType>::value,decltype(dup_array_surf_tally),decltype(ndup_array_surf_tally)>::get(dup_array_surf_tally,ndup_array_surf_tally);
  auto a_array_surf_tally = v_array_surf_tally.template access<typename AtomicDup<ATOMIC_REDUCTION,DeviceType>::value>();

  for (int i = 0; i < nkeep; i++) {
    Particle::OnePart *p = plist[i];
    double *v = p->v;
    double sample;
    const double wt = useweight ? p->weight : 1.0;

    if (distflag == ENERGY) {
      double ke;
      switch (engstyle) {
      case KE:
        sample = 0.5*mvv2e * d_species(p->ispecies).mass *
          MathExtraKokkos::lensq3(v);
        break;
      case EROT:
        sample = p->erot;
        break;
      case EVIB:
        sample = p->evib;
        break;
      case ETOT:
        ke = 0.5*mvv2e * d_species(p->ispecies).mass *
          MathExtraKokkos::lensq3(v);
        sample = ke + p->erot + p->evib;
        break;
      default:
        sample = 0.0;
        break;
      }

    } else {

      // polar angle from the surface normal, in degrees
      // norm is a unit vector and points outward from the surface,
      //   so an incident particle has v dot norm < 0
      // fabs() makes the angle range 0 to 90 for incident and reflected alike
      // a motionless particle has no direction, so skip it

      const double vmag = sqrt(MathExtraKokkos::lensq3(v));
      if (vmag == 0.0) continue;
      double cosang = fabs(MathExtraKokkos::dot3(v,norm)) / vmag;
      if (cosang > 1.0) cosang = 1.0;
      sample = acos(cosang) * 180.0/MathConst::MY_PI;
    }

    // test the sample against lo/hi directly rather than testing the bin
    //   index, because a cast to int truncates toward zero, so a sample just
    //   below lo would otherwise be indistinguishable from bin 0
    // a sample exactly at hi, or one pushed past the last bin by roundoff,
    //   is folded into the last bin

    int ibin;
    if (sample < lo || sample > hi) {
      if (oobstyle == IGNORE) continue;
      ibin = (sample < lo) ? 0 : nbin-1;
    } else {
      ibin = static_cast<int> ((sample - lo) * invdelta);
      if (ibin >= nbin) ibin = nbin - 1;
    }

    a_array_surf_tally(itally,glist[i]*nbin + ibin) += wt;
  }
}

 private:
  double mvv2e;
  int useweight;
  int compressed;          // 1 once the device tallies have been compressed

  DAT::tdual_float_2d_lr k_array_surf_tally;
  DAT::t_float_2d_lr d_array_surf_tally;  // tally values for local surfs

  int need_dup;
  Kokkos::Experimental::ScatterView<F_FLOAT**, typename DAT::t_float_2d_lr::array_layout,DeviceType,typename Kokkos::Experimental::ScatterSum,typename Kokkos::Experimental::ScatterDuplicated> dup_array_surf_tally;
  Kokkos::Experimental::ScatterView<F_FLOAT**, typename DAT::t_float_2d_lr::array_layout,DeviceType,typename Kokkos::Experimental::ScatterSum,typename Kokkos::Experimental::ScatterNonDuplicated> ndup_array_surf_tally;

  DAT::t_surfint_1d d_tally2surf;     // tally2surf[I] = surf ID of Ith tally
  DAT::tdual_surfint_1d k_tally2surf;
  DAT::t_int_1d d_surf2tally;         // -1 if surf has not been tallied

  t_species_1d d_species;
  DAT::t_int_2d d_s2g;

  t_line_1d d_lines;
  t_tri_1d d_tris;

  void allocate_tally();
  void grow_tally();
};

}

#endif
#endif

/* ERROR/WARNING messages:

*/
