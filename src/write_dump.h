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

CommandStyle(write_dump,WriteDump)

#else

#ifndef SPARTA_WRITE_DUMP_H
#define SPARTA_WRITE_DUMP_H

#include "pointers.h"

namespace SPARTA_NS {

class WriteDump : protected Pointers {
 public:
  WriteDump(class SPARTA *sparta) : Pointers(sparta) {}
  void command(int, char **);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: Unrecognized dump style in write_dump command

The choice of dump style is unknown.

E: Dump style movie cannot be used with write_dump

A movie is a sequence of frames written during a run, so a single
snapshot of it is not meaningful.  Use dump style image instead.

*/
