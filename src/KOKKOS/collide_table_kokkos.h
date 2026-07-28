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

#ifdef COLLIDE_CLASS

CollideStyle(table/kk,CollideTableKokkos)

#else

#ifndef SPARTA_COLLIDE_TABLE_KOKKOS_H
#define SPARTA_COLLIDE_TABLE_KOKKOS_H

#include "collide_vss_kokkos.h"
#include "kokkos_type.h"

namespace SPARTA_NS {

/* ----------------------------------------------------------------------
   collide table/kk = tabulated collision model on the KOKKOS package

   the host side of the style is entirely inherited: CollideVSSKokkos
   derives from CollideTable, so reading the parameter file, building the
   binned tables, the effective cross section for compute lambda/grid and
   the vremax estimate are all already in place.  this class only parses
   the table arguments and copies the built tables to the device.

   the device path covers the total cross section and the energy-dependent
   alpha.  the remaining features of the host style raise an error at setup
   rather than being silently skipped, since each changes the answer:

     scatter tables    the inverse cumulative differential cross section
     internal energy   the Larsen-Borgnakke detailed balance correction for
                         a tabulated pair whose species have rotational or
                         vibrational modes
     chemistry         react tce needs the VHS-to-table probability factor,
                         and react table has no kk variant
     EXTRAP error      a table which must abort outside its range cannot,
                         since device code cannot raise an error
------------------------------------------------------------------------- */

class CollideTableKokkos : public CollideVSSKokkos {
 public:
  CollideTableKokkos(class SPARTA *, int, char **);
  ~CollideTableKokkos();
  void init();

 protected:
  void copy_tables_to_device();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Collide table/kk does not support scatter tables

A scatter table samples the deflection angle from an inverse cumulative
differential cross section, which is not ported to the KOKKOS package.
Run this model without the kk suffix.

E: Collide table/kk does not support a tabulated pair with internal energy

The Larsen-Borgnakke detailed balance correction which a tabulated cross
section requires is not ported to the KOKKOS package, and without it an
equilibrium gas drifts.  Run this model without the kk suffix.

E: Collide table/kk does not support chemistry

The reaction probability factor which relates a tabulated total cross
section to the TCE model is not ported to the KOKKOS package, and react
table has no kk variant.  Run this model without the kk suffix.

E: Collide table/kk does not support EXTRAP error

Device code cannot raise an error, so a table which must abort outside its
range cannot be used with the KOKKOS package.  Use another extrapolation
mode, or run without the kk suffix.

*/
