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

#include "collide_vss.h"
#include "particle.h"

namespace SPARTA_NS {

/* ----------------------------------------------------------------------
   collide table = tabulated collision model

   three quantities may be tabulated per species pair, independently:

     table    total cross section sigma_T(E), used to select pairs
     alpha    energy-dependent VSS alpha, used to deflect them
     scatter  cos(chi) as a function of (E, cumulative probability),
                i.e. the inverse cumulative differential cross section

   sigma_T(E) with a constant alpha reproduces one transport cross
     section exactly at all temperatures; adding alpha(E) reproduces both
     the diffusion and viscosity cross sections at every energy; a scatter
     table reproduces the full angular distribution

   anything not tabulated for a pair falls back to the analytic VSS form,
     so a tabulated pair can be introduced into an existing model without
     disturbing the others
------------------------------------------------------------------------- */

class CollideTable : public CollideVSS {
 public:
  CollideTable(class SPARTA *, int, char **);
  virtual ~CollideTable();
  virtual void init();

  double vremax_init(int, int);
  virtual int test_collision(int, int, int, Particle::OnePart *,
                             Particle::OnePart *);
  double sigma_eff(int, int, double);
  int tabulated_pair(int, int);
  double lb_weight(int, int, double, double);

 protected:

  // ctor for a derived style which is not itself a table style
  //
  // CollideVSSKokkos derives from this class rather than from CollideVSS,
  // because the KOKKOS collision kernels are launched as
  // parallel_for(policy,*this), which slices the object to CollideVSSKokkos
  // and so cannot dispatch virtually to a style derived from it.  the table
  // state therefore has to be reachable from CollideVSSKokkos itself.  this
  // ctor leaves every table NULL, and each method below falls back to the
  // analytic VSS form when they are, so collide vss/kk is unaffected.

  CollideTable(class SPARTA *, int, char **, int);

  // parse the table arguments, read the parameter file and broadcast it
  // called by the public ctor here and by CollideTableKokkos

  void setup_tables(int, char **);
  void null_tables();

  int tabstyle;              // TB_LOOKUP, TB_LINEAR, or TB_SPLINE
  int nmant;                 // # of mantissa bits used to index a bin

  int nsigma,nalpha,nscatter;      // # of tables of each kind
  class InterpTable **sigma_tab;
  class InterpTable **alpha_tab;
  class InterpTable **scatter_tab;

  int **sigma_index;         // index into the tables for each species pair,
  int **alpha_index;         //   -1 = none, use the analytic VSS form
  int **scatter_index;

  // effective cross section vs temperature for compute lambda/grid,
  //   built once per tabulated pair from the cross section table

  double **sigeff;
  double tlo,tinvdelta;
  int ntemp;                 // # of temperature points per sigeff row

  // running maximum of sigma_table/sigma_VHS over [0,E], used to normalize
  //   the Larsen-Borgnakke acceptance so it stays a probability.  one row
  //   per species pair that has both a cross section table and an internal
  //   mode to exchange with, on a log energy grid

  int **lb_index;            // row for a pair, -1 = no correction needed
  int nlbpair;
  double **lbratio;          // sigma_table/sigma_VHS on the grid
  double **lbmax;            // running max of lbratio up to each grid point
  double lblo,lbinvdelta;
  int nlbgrid;               // # of grid points, so an accelerator package
                             //   can size its copy without the NLB define
  int lbwarn;                // 1 once the out-of-range warning has fired

  void build_lbratio();

  double scatter_alpha(int, int);
  int scatter_cosX(int, int, double &);

  void read_param_file(char *);
  int skip_param_line(int, char **);
  class InterpTable *add_table(class InterpTable **&, int &, int, int,
                               char *, char *, int, int, int);
  void build_sigeff();
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

E: Unknown table directive in collide table parameter file

A directive line must use the keyword table, alpha, or scatter.

E: Tabulated alpha must be positive

The VSS deflection law uses 1/alpha, so a zero or negative tabulated
alpha would produce an invalid scattering angle.

E: A scatter table must set M > 1 on its parameter line

A scatter section holds cos(chi) at M cumulative probabilities, so M must
be greater than 1.

E: Cross section table is required for a pair with an alpha or scatter table

An alpha or scatter table changes only the deflection angle, so the pair
must also have a total cross section table.

W: No cross section tables were defined by collide table

The style will behave exactly like collide vss.

W: Collision energy is outside the Larsen-Borgnakke normalization grid, internal energy exchange for the tabulated pair reverts to the VSS law

The acceptance test which restores detailed balance for a tabulated cross
section is normalized on an energy grid spanning 1e-9 to 1e6 eV.  A
collision energy above that range cannot be bounded by it, so the exchange
falls back to the uncorrected VSS sampling for those collisions.

W: Tabulated data does not reproduce its input values

The binned table does not reproduce the values read from the file to
within 1%.  Increase N or use the spline style.

*/
