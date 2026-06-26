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

#ifdef COMMAND_CLASS

CommandStyle(create_shift,CreateShift)

#else

#ifndef SPARTA_CREATE_SHIFT_H
#define SPARTA_CREATE_SHIFT_H

#include "pointers.h"

namespace SPARTA_NS {

class CreateShift : protected Pointers {
 public:
  CreateShift(class SPARTA *);
  void command(int, char **);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot create_shift before simulation box is defined

The create_shift command must come after create_box (or read_restart /
read_data) so the box dimensions are known.

E: Illegal create_shift command

Self-explanatory.  create_shift takes exactly 3 arguments.

E: Create_shift fractions must be between -1.0 and 1.0

Each shift fraction scales the corresponding box length, so values
outside this range are not meaningful.

E: Create_shift z fraction must be 0.0 for 2d simulation

A 2d simulation has no extent in z, so its shift fraction must be zero.

*/
