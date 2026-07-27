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

CollideStyle(table,CollideTable)

#else

#ifndef SPARTA_COLLIDE_TABLE_H
#define SPARTA_COLLIDE_TABLE_H

#include "stdint.h"
#include "collide_vss.h"
#include "particle.h"

namespace SPARTA_NS {

class CollideTable : public CollideVSS {
 public:
  CollideTable(class SPARTA *, int, char **);
  virtual ~CollideTable();

  double vremax_init(int, int);
  virtual int test_collision(int, int, int, Particle::OnePart *,
                             Particle::OnePart *);

 protected:

  // one tabulated total cross section, for one species pair
  //
  // the run-time table stores sigma*g as a function of vr^2
  //   sigma*g is the only combination test_collision() needs, and both
  //   vr^2 and the relative translational energy 1/2*m_r*vr^2 index the
  //   same grid, so no sqrt() and no pow() is needed
  //
  // bins are spaced logarithmically, since cross section data spans decades
  //   of collision energy.  the bin index is read straight out of the IEEE-754
  //   exponent and the leading NMANT mantissa bits of vr^2, so there is one
  //   shift and one subtract and no transcendental call, the same trick the
  //   LAMMPS bitmapped pair tables use.  each bin therefore spans a fixed
  //   ratio 2^(1/2^NMANT) in energy, i.e. equal relative resolution at all
  //   energies, and there are 2^NMANT bins per factor of 2 in energy

  struct Table {
    char *file,*keyword;     // provenance, for log and error messages
    int isp,jsp;             // a species pair which uses this table
    int ninput;              // # of (x,sigma) values read from file
    int xvar;                // ENERGY or SPEED, independent var in the file
    int extrap_lo,extrap_hi; // extrapolation mode below/above the table range
    double xscale,yscale;    // file units -> SI, used only while reading
    int logflag;             // 1 if the input data is splined in log-log

    // input data and its spline, freed once the binned table is built
    // sigma, not sigma*g, is splined: it is the smooth quantity the data
    //   describes, so sparse or widely spaced input is reproduced correctly
    // log-log is used whenever every sigma is positive, which makes a power
    //   law exact and matches how cross section data is normally tabulated

    double *xfile;           // ninput values of vr^2, converted from file x
    double *sfile;           // ninput values of sigma in m^2
    double *xspl,*yspl;      // spline abscissa/ordinate, raw or logged
    double *yspl2;           // spline 2nd derivatives

    double vr2lo,vr2hi;      // vr^2 range spanned by the file data
    int nbins;               // # of bins, covering whole octaves of vr^2
    int shift;               // right shift which maps vr^2 bits to a bin
    int64_t offset;          // bin index of the first bin
    double *coeff;           // ncoeff values per bin, see sigma_g()

    // extrapolation outside [vr2lo,vr2hi] is always a power law in vr^2:
    //   sigma*g = a * vr2^p
    // constant sigma is p = 1/2, the VSS fallback is p = 1-omega

    double alo,plo,ahi,phi;

    double sigmax;           // max of sigma over the input values
  };

  int tabstyle;              // LOOKUP, LINEAR, or SPLINE
  int ncoeff;                // # of coefficients stored per bin
  int nmant;                 // # of mantissa bits used to index a bin,
                             //   so 2^nmant bins per factor of 2 in energy
  int ntables;               // # of tables read
  Table *tables;
  int **tabindex;            // index into tables for each species pair,
                             //   -1 = no table, use the analytic VSS form

  double sigma_g(int, double);
  double interp_sigma_g(Table *, double);
  int bin_index(Table *, double);
  double bin_lower(Table *, int);

  void read_param_file(char *);
  int skip_param_line(int, char **);
  void read_table(Table *, char *, char *);
  void param_extract(Table *, char *, char *, char *);
  void convert_table(Table *);
  void compute_table(Table *);
  void check_table(Table *);
  void input_sg(Table *, double, double *, double *);
  void bcast_table(Table *);
  void null_table(Table *);
  void free_table(Table *);

  void spline(double *, double *, int, double, double, double *);
  void splint(double *, double *, double *, int, double, double *, double *);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal collide command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.

E: Unknown table style in collide table

The interpolation style must be lookup, linear, or spline.

E: Illegal number of collide table entries

N must be between 1 and 2^20.

E: Cross section table spans too many bins

The tabulated energy range together with N requires more than 2^24 bins.
Reduce N or narrow the range of the table.

E: Cannot open cross section table file %s

Self-explanatory.

E: Did not find keyword %s in cross section table file %s

The requested section keyword does not appear in the file.

E: Premature end of cross section table file

The file ended before N rows of data were read.

E: Incorrect line format in cross section table file

A data row must have 3 columns: index, x, sigma.

E: Invalid keyword in cross section table parameters

A keyword on the parameter line following the section keyword is not
recognized, or its value is missing.

E: Cross section table parameters did not set N for keyword %s in file %s

The parameter line must include "N <n>".

E: Invalid cross section table length

N on the parameter line must be > 1, and > 3 if the spline style is used.

E: Cross section table values are not increasing

The independent variable in the table must increase monotonically.

E: Cross section table values must be positive

The independent variable must be > 0, since a zero relative energy or
speed corresponds to no collision.

E: Cross section table has a negative cross section

Cross sections must be >= 0.

E: Collision energy is outside the cross section table range

The table specified "extrap error" for this end of its range, and a
collision was attempted outside it.  Widen the table or select a
different extrapolation mode.

W: Cross section table does not reproduce its input values

The binned table does not reproduce the values read from the file to
within 1%.  Increase N or use the spline style.

W: No cross section tables were defined by collide table

The style will behave exactly like collide vss.

*/
