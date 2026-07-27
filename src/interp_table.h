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

#ifndef SPARTA_INTERP_TABLE_H
#define SPARTA_INTERP_TABLE_H

#include "stdint.h"
#include "pointers.h"

namespace SPARTA_NS {

// interpolation styles, chosen by the parent command
enum{TB_LOOKUP,TB_LINEAR,TB_SPLINE};

// independent variable as it appears in the file
enum{TB_ENERGY,TB_SPEED};

// how the file's independent variable maps to the run-time index
enum{TB_XVR2,TB_XRAW};

// how the file's dependent variable maps to the run-time value
enum{TB_YSIGMA_G,TB_YRAW};

// behavior outside the tabulated range
enum{TB_CONSTANT,TB_POWERLAW,TB_VSS,TB_ERROR};

/* ----------------------------------------------------------------------
   one tabulated function of a single positive variable, read from a
     keyword-delimited section of a data file

   values are stored in bins whose index is read straight out of the
     IEEE-754 exponent and the leading NMANT mantissa bits of x, so a
     lookup is one shift and one subtract with no transcendental call,
     the same trick the LAMMPS bitmapped pair tables use.  bins are
     therefore spaced logarithmically with equal relative width, and a
     table may span many decades without losing resolution at the low end

   the input data is splined in log-log whenever every value is positive,
     which is exact for a power law and correct for the sparse, widely
     spaced data that cross section tables usually contain

   with NCOL > 1 each x holds a row of NCOL values, used for a
     distribution tabulated against a cumulative probability.  a row is
     taken from the containing bin, which is why only TB_LOOKUP applies
     in x for those tables
------------------------------------------------------------------------- */

class InterpTable : protected Pointers {
 public:
  char *file,*keyword;       // provenance, for log and error messages
  int ninput;                // # of x values read from the file
  int ncol;                  // # of y values per x
  int xvar;                  // TB_ENERGY or TB_SPEED, as read
  int extrap_lo,extrap_hi;   // extrapolation mode below/above the range
  int logflag;               // 1 if the input was splined in log-log
  double xlo,xhi;            // run-time x range spanned by the input
  double ymax;               // max input value, in file units
  int nbins;                 // # of bins, covering whole octaves of x

  InterpTable(class SPARTA *);
  ~InterpTable();

  void read(const char *, const char *, int);
  void convert(int, int, double, double, double);
  void build(int, int);
  double check(int);
  void free_input();
  void bcast();

  // evaluate with the full extrapolation policy, for the hot path

  double evaluate(double);

  // evaluate with the bin clamped into range, for callers which may
  //   sit on or beyond an end bin and must not trigger TB_ERROR

  double interpolate(double);

  // row lookup for NCOL > 1: linear interpolation in the row at
  //   cumulative probability u in [0,1)

  double interpolate_row(double, double);

  double bin_lower(int);

 private:
  int tabstyle;              // TB_LOOKUP, TB_LINEAR, or TB_SPLINE
  int ncoeff;                // # of stored coefficients per bin per column
  int nmant;                 // # of mantissa bits used to index a bin
  int shift;                 // right shift which maps x bits to a bin
  int64_t offset;            // bin index of the first bin
  double *coeff;             // ncoeff*ncol values per bin

  // extrapolation outside [xlo,xhi] is a power law, y = a * x^p
  //   constant sigma is p = 1/2 for TB_YSIGMA_G and p = 0 for TB_YRAW,
  //   and the VSS fallback is p = 1-omega

  double alo,plo,ahi,phi;

  double xscale,yscale;      // file units -> SI, used only while reading
  int xmode,ymode;

  // input data and its spline, freed once the binned table is built

  double *xfile;             // ninput values of the run-time x
  double *yfile;             // ninput*ncol values, in SI
  double *xspl,*yspl,*yspl2;

  char *linebuf;             // dynamically grown line buffer, so a row of
  int maxline;               //   many columns is not limited by a fixed size

  char *read_line(FILE *);
  int bin_index(double);
  void input_value(double, double *, double *);
  void spline(double *, double *, int, double, double, double *);
  void splint(double *, double *, double *, int, double, double *, double *);
  void param_extract(char *);
};

}

#endif
