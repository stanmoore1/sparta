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

CommandStyle(info,Info)

#else

#ifndef SPARTA_INFO_H
#define SPARTA_INFO_H

#include "stdio.h"
#include "pointers.h"

namespace SPARTA_NS {

class Info : protected Pointers {
 public:
  Info(class SPARTA *sparta) : Pointers(sparta) {}
  void command(int, char **);

 private:
  FILE *out;                 // where the report is written

  void config();
  void sysinfo();
  void comminfo();
  void computes();
  void fixes();
  void dumps();
  void variables();
  void regions();
  void groups();
  void species();
  void mixtures();
  void surf_collide();
  void surf_react();
  void meminfo();
  void timeinfo();
  void styles();

  void styles_category(const char *, int, const char **);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: Unknown info command keyword

The info command was given a keyword it does not recognize.  See the
info doc page for the list of valid keywords.

E: Cannot open info file %s

The specified file cannot be opened.  Check that the path and name are
correct.

*/
