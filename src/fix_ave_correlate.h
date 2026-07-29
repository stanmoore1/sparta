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

#ifdef FIX_CLASS

FixStyle(ave/correlate,FixAveCorrelate)

#else

#ifndef SPARTA_FIX_AVE_CORRELATE_H
#define SPARTA_FIX_AVE_CORRELATE_H

#include "stdio.h"
#include "fix.h"

namespace SPARTA_NS {

class FixAveCorrelate : public Fix {
 public:
  FixAveCorrelate(class SPARTA *, int, char **);
  ~FixAveCorrelate();
  int setmask();
  void init();
  void setup();
  void end_of_step();
  double compute_array(int,int);

 private:
  int me,nvalues;
  int nrepeat,nfreq;
  bigint nvalid;
  int *which,*argindex,*value2index;
  char **ids;
  FILE *fp;

  int type,ave,startstep,overwrite;
  double prefactor;
  long filepos;

  int firstindex;      // index in values ring of latest time sample
  int lastindex;       // index in values ring of oldest time sample
  int nsample;         // number of time samples in values ring

  int npair;           // number of correlation pairs to calculate
  double *count;
  double **values,**corr;

  double *save_count;  // saved values at Nfreq for output to file
  double **save_corr;

  void accumulate();
  bigint nextvalid();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: No values in fix ave/correlate command

Self-explanatory.

E: Compute ID for fix ave/correlate does not exist

Self-explanatory.

E: Fix ave/correlate compute does not calculate a scalar

Self-explanatory.

E: Fix ave/correlate compute does not calculate a vector

Self-explanatory.

E: Fix ave/correlate compute vector is accessed out-of-range

The index for the vector is out of bounds.

E: Fix ID for fix ave/correlate does not exist

Self-explanatory.

E: Fix ave/correlate fix does not calculate a scalar

Self-explanatory.

E: Fix ave/correlate fix does not calculate a vector

Self-explanatory.

E: Fix ave/correlate fix vector is accessed out-of-range

The index for the vector is out of bounds.

E: Fix for fix ave/correlate not computed at compatible time

Fixes generate their values on specific timesteps.  Fix ave/correlate
is requesting a value on a non-allowed timestep.

E: Variable name for fix ave/correlate does not exist

Self-explanatory.

E: Fix ave/correlate variable is not equal-style variable

Self-explanatory.

E: Cannot open fix ave/correlate file %s

The specified file cannot be opened.  Check that the path and name are
correct.

E: Error writing out correlation data

An error occurred while writing the correlation data to the output
file.  The disk may be full.

*/
