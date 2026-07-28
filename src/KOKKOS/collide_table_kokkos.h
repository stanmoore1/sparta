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

   the device path covers everything the host style does: the total cross
   section, the energy-dependent alpha, the scatter table, the
   Larsen-Borgnakke detailed balance correction and its retry cap, and the
   EXTRAP error policy, which device code cannot raise directly and so is
   recorded in a flag this class checks after each collision kernel.  the
   effective cross section vs temperature goes to the device as well, so
   compute lambda/grid follows the table there as it does on the host.
------------------------------------------------------------------------- */

class CollideTableKokkos : public CollideVSSKokkos {
 public:
  CollideTableKokkos(class SPARTA *, int, char **);
  ~CollideTableKokkos();
  void init();

  void collisions();

 protected:
  void copy_tables_to_device();
  void copy_sigeff_to_device();
  void copy_lb_to_device();
  class InterpTable *table_at(int);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Value is outside the tabulated data range

A table whose extrapolation mode is error was evaluated outside its range.
Device code cannot raise an error itself, so this is recorded during the
collision kernel and raised here afterwards.

*/
