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

#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "interp_table.h"
#include "update.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

#define MAXLINE 8192
#define EV2J 1.602176634e-19        // electron volt in Joules
#define ANG2SQ 1.0e-20              // Angstrom^2 in m^2
#define CM2SQ 1.0e-4                // cm^2 in m^2
#define MAXBIN (1 << 24)            // max bins in one table
#define MAXCHECK 20000              // max bins sampled by the accuracy check

// reinterpret a positive double as its IEEE-754 bit pattern

union DoubleBits { double d; uint64_t u; };

/* ---------------------------------------------------------------------- */

InterpTable::InterpTable(SPARTA *sparta) : Pointers(sparta)
{
  file = keyword = NULL;
  ninput = ncol = nbins = 0;
  xvar = TB_ENERGY;
  extrap_lo = extrap_hi = TB_CONSTANT;
  logflag = 0;
  xlo = xhi = ymin = ymax = 0.0;
  tabstyle = TB_LINEAR;
  ncoeff = nmant = shift = 0;
  offset = 0;
  coeff = NULL;
  alo = plo = ahi = phi = 0.0;
  xscale = yscale = 1.0;
  xmode = TB_XVR2;
  ymode = TB_YSIGMA_G;
  xfile = yfile = xspl = yspl = yspl2 = NULL;
  linebuf = NULL;
  maxline = 0;
}

/* ---------------------------------------------------------------------- */

InterpTable::~InterpTable()
{
  delete [] file;
  delete [] keyword;
  free_input();
  memory->destroy(coeff);
  memory->destroy(linebuf);
}

/* ---------------------------------------------------------------------- */

void InterpTable::free_input()
{
  memory->destroy(xfile);
  memory->destroy(yfile);
  memory->destroy(xspl);
  memory->destroy(yspl);
  memory->destroy(yspl2);
  xfile = yfile = xspl = yspl = yspl2 = NULL;
}

/* ----------------------------------------------------------------------
   read one logical line, growing the buffer as needed
   a row of a multi-column table can be arbitrarily long
------------------------------------------------------------------------- */

char *InterpTable::read_line(FILE *fp)
{
  int n = 0;
  while (1) {
    if (maxline < n + MAXLINE) {
      maxline = n + MAXLINE;
      memory->grow(linebuf,maxline,"interp/table:linebuf");
    }
    if (fgets(&linebuf[n],maxline-n,fp) == NULL) return n ? linebuf : NULL;
    n += strlen(&linebuf[n]);
    if (n && linebuf[n-1] == '\n') return linebuf;
    if (feof(fp)) return linebuf;
  }
}

/* ----------------------------------------------------------------------
   read one keyword-delimited section of a data file
   ncol_expect = expected # of value columns, or 0 to take it from the file
   only invoked by proc 0
------------------------------------------------------------------------- */

void InterpTable::read(const char *fname, const char *kw, int ncol_expect)
{
  delete [] file;
  delete [] keyword;
  file = new char[strlen(fname)+1];
  strcpy(file,fname);
  keyword = new char[strlen(kw)+1];
  strcpy(keyword,kw);

  FILE *fp = fopen(fname,"r");
  if (fp == NULL) {
    char str[256];
    sprintf(str,"Cannot open tabulated data file %s",fname);
    error->one(FLERR,str);
  }

  // scan for a line whose first word matches the keyword

  char *line;
  char copy[256];
  while (1) {
    if ((line = read_line(fp)) == NULL) {
      char str[512];
      sprintf(str,"Did not find keyword %s in tabulated data file %s",kw,fname);
      error->one(FLERR,str);
    }
    int pre = strspn(line," \t\n\r");
    if (pre == (int) strlen(line) || line[pre] == '#') continue;
    strncpy(copy,line,255);
    copy[255] = '\0';
    char *word = strtok(copy," \t\n\r");
    if (word && strcmp(word,kw) == 0) break;
  }

  // parameter line follows the keyword line

  if ((line = read_line(fp)) == NULL)
    error->one(FLERR,"Premature end of tabulated data file");
  // ncol_expect = 0 lets the file's M keyword choose, otherwise it is fixed

  ncol = ncol_expect;
  param_extract(line);
  if (ncol_expect > 0 && ncol != ncol_expect)
    error->one(FLERR,"Wrong number of columns in tabulated data file");

  memory->create(xfile,ninput,"interp/table:xfile");
  memory->create(yfile,ninput*ncol,"interp/table:yfile");

  // ninput rows of: index x y1 [y2 ... yNcol]
  // blank and comment lines are allowed between rows

  int n = 0;
  while (n < ninput) {
    if ((line = read_line(fp)) == NULL)
      error->one(FLERR,"Premature end of tabulated data file");
    int pre = strspn(line," \t\n\r");
    if (pre == (int) strlen(line) || line[pre] == '#') continue;

    strtok(line," \t\n\r");
    char *w = strtok(NULL," \t\n\r");
    if (w == NULL)
      error->one(FLERR,"Incorrect line format in tabulated data file");
    xfile[n] = atof(w);
    for (int c = 0; c < ncol; c++) {
      w = strtok(NULL," \t\n\r");
      if (w == NULL)
        error->one(FLERR,"Incorrect line format in tabulated data file");
      yfile[n*ncol+c] = atof(w);
    }
    n++;
  }

  fclose(fp);
}

/* ----------------------------------------------------------------------
   parse the parameter line which follows a section keyword
   N <n> [M <m>] [X energy|speed] [XUNITS eV|J|K|m/s]
     [YUNITS m^2|cm^2|A^2] [EXTRAP <lo> <hi>]
------------------------------------------------------------------------- */

void InterpTable::param_extract(char *line)
{
  ninput = 0;
  xvar = TB_ENERGY;
  xscale = 0.0;
  yscale = 1.0;
  extrap_lo = extrap_hi = TB_CONSTANT;
  int mgiven = 0;

  char *word = strtok(line," \t\n\r");
  while (word) {
    if (strcmp(word,"N") == 0) {
      word = strtok(NULL," \t\n\r");
      if (!word) error->one(FLERR,"Invalid keyword in tabulated data parameters");
      ninput = atoi(word);

    } else if (strcmp(word,"M") == 0) {
      word = strtok(NULL," \t\n\r");
      if (!word) error->one(FLERR,"Invalid keyword in tabulated data parameters");
      ncol = atoi(word);
      mgiven = 1;

    } else if (strcmp(word,"X") == 0) {
      word = strtok(NULL," \t\n\r");
      if (!word) error->one(FLERR,"Invalid keyword in tabulated data parameters");
      if (strcmp(word,"energy") == 0) xvar = TB_ENERGY;
      else if (strcmp(word,"speed") == 0) xvar = TB_SPEED;
      else error->one(FLERR,"Invalid keyword in tabulated data parameters");

    } else if (strcmp(word,"XUNITS") == 0) {
      word = strtok(NULL," \t\n\r");
      if (!word) error->one(FLERR,"Invalid keyword in tabulated data parameters");
      if (strcmp(word,"eV") == 0) xscale = EV2J;
      else if (strcmp(word,"J") == 0) xscale = 1.0;
      else if (strcmp(word,"K") == 0) xscale = update->boltz;
      else if (strcmp(word,"m/s") == 0) xscale = 1.0;
      else error->one(FLERR,"Invalid keyword in tabulated data parameters");

    } else if (strcmp(word,"YUNITS") == 0) {
      word = strtok(NULL," \t\n\r");
      if (!word) error->one(FLERR,"Invalid keyword in tabulated data parameters");
      if (strcmp(word,"m^2") == 0) yscale = 1.0;
      else if (strcmp(word,"cm^2") == 0) yscale = CM2SQ;
      else if (strcmp(word,"A^2") == 0) yscale = ANG2SQ;
      else error->one(FLERR,"Invalid keyword in tabulated data parameters");

    } else if (strcmp(word,"EXTRAP") == 0) {
      int *mode = &extrap_lo;
      for (int i = 0; i < 2; i++) {
        word = strtok(NULL," \t\n\r");
        if (!word) error->one(FLERR,"Invalid keyword in tabulated data parameters");
        if (strcmp(word,"constant") == 0) *mode = TB_CONSTANT;
        else if (strcmp(word,"powerlaw") == 0) *mode = TB_POWERLAW;
        else if (strcmp(word,"vss") == 0) *mode = TB_VSS;
        else if (strcmp(word,"error") == 0) *mode = TB_ERROR;
        else error->one(FLERR,"Invalid keyword in tabulated data parameters");
        mode = &extrap_hi;
      }

    } else error->one(FLERR,"Invalid keyword in tabulated data parameters");

    word = strtok(NULL," \t\n\r");
  }

  if (!mgiven && ncol < 1) ncol = 1;
  if (ncol < 1)
    error->one(FLERR,"Invalid tabulated data column count");

  // default x units depend on the independent variable

  if (xscale == 0.0) xscale = (xvar == TB_ENERGY) ? EV2J : 1.0;

  if (ninput == 0) {
    char str[512];
    sprintf(str,"Tabulated data parameters did not set N for keyword %s "
            "in file %s",keyword,file);
    error->one(FLERR,str);
  }
  if (ninput < 2)
    error->one(FLERR,"Invalid tabulated data length");
}

/* ----------------------------------------------------------------------
   convert the file data to SI and to the run-time independent variable,
     then spline it and set the extrapolation coefficients
   MR is the pair reduced mass, used only when xmode is TB_XVR2
   VSS_A, VSS_P give the analytic fallback y = a*x^p for TB_VSS
------------------------------------------------------------------------- */

void InterpTable::convert(int xmode_in, int ymode_in, double mr,
                          double vss_a, double vss_p)
{
  xmode = xmode_in;
  ymode = ymode_in;
  int n = ninput;

  for (int i = 0; i < n; i++) {
    double x = xfile[i] * xscale;
    if (xmode == TB_XRAW) xfile[i] = x;
    else if (xvar == TB_ENERGY) xfile[i] = 2.0*x/mr;
    else xfile[i] = x*x;
    for (int c = 0; c < ncol; c++) yfile[i*ncol+c] *= yscale;
  }

  ymin = yfile[0];
  ymax = yfile[0];
  logflag = 1;
  for (int i = 0; i < n*ncol; i++) {
    if (ymode == TB_YSIGMA_G && yfile[i] < 0.0)
      error->one(FLERR,"Tabulated data has a negative cross section");
    if (yfile[i] <= 0.0) logflag = 0;
    ymin = MIN(ymin,yfile[i]);
    ymax = MAX(ymax,yfile[i]);
  }
  for (int i = 1; i < n; i++)
    if (xfile[i] <= xfile[i-1])
      error->one(FLERR,"Tabulated data values are not increasing");
  if (xfile[0] <= 0.0)
    error->one(FLERR,"Tabulated data values must be positive");

  // a multi-column table is looked up per bin, so it needs no spline

  if (ncol > 1) return;

  // spline the input, in log-log unless a value is non-positive
  // end slopes from finite differences, so a straight line in the splined
  //   variables, i.e. a power law in the log-log case, is exact

  memory->create(xspl,n,"interp/table:xspl");
  memory->create(yspl,n,"interp/table:yspl");
  memory->create(yspl2,n,"interp/table:yspl2");

  for (int i = 0; i < n; i++) {
    xspl[i] = logflag ? log(xfile[i]) : xfile[i];
    yspl[i] = logflag ? log(yfile[i]) : yfile[i];
  }
  double ep0 = (yspl[1]-yspl[0]) / (xspl[1]-xspl[0]);
  double epn = (yspl[n-1]-yspl[n-2]) / (xspl[n-1]-xspl[n-2]);
  spline(xspl,yspl,n,ep0,epn,yspl2);

  // extrapolation coefficients, y = a * x^p
  //   TB_YSIGMA_G carries an extra factor of sqrt(x), so a constant
  //     cross section is p = 1/2 rather than p = 0

  double pconst = (ymode == TB_YSIGMA_G) ? 0.5 : 0.0;

  if (extrap_lo == TB_VSS) {
    alo = vss_a;
    plo = vss_p;
  } else if (extrap_lo == TB_POWERLAW && yfile[0] > 0.0 && yfile[1] > 0.0) {
    double q = log(yfile[1]/yfile[0]) / log(xfile[1]/xfile[0]);
    plo = q + pconst;
    alo = yfile[0] / pow(xfile[0],q);
  } else {
    plo = pconst;
    alo = yfile[0];
  }

  if (extrap_hi == TB_VSS) {
    ahi = vss_a;
    phi = vss_p;
  } else if (extrap_hi == TB_POWERLAW && yfile[n-1] > 0.0 && yfile[n-2] > 0.0) {
    double q = log(yfile[n-1]/yfile[n-2]) / log(xfile[n-1]/xfile[n-2]);
    phi = q + pconst;
    ahi = yfile[n-1] / pow(xfile[n-1],q);
  } else {
    phi = pconst;
    ahi = yfile[n-1];
  }
}

/* ----------------------------------------------------------------------
   the run-time value and its derivative at X, from the input spline
------------------------------------------------------------------------- */

void InterpTable::input_value(double x, double *y, double *dy)
{
  double s,ds;

  if (logflag) {

    // yspl = ln(y) vs xspl = ln(x), so y = exp(S) and dy/dx = y*S'/x

    double lny,dlny;
    splint(xspl,yspl,yspl2,ninput,log(x),&lny,&dlny);
    s = exp(lny);
    ds = s*dlny/x;
  } else {
    splint(xspl,yspl,yspl2,ninput,x,&s,&ds);
  }

  if (ymode == TB_YSIGMA_G) {
    double rt = sqrt(x);
    *y = s*rt;
    *dy = ds*rt + 0.5*s/rt;
  } else {
    *y = s;
    *dy = ds;
  }
}

/* ----------------------------------------------------------------------
   build the binned table
   bins cover whole octaves of x, so that a bin index is a shift of the
     x bit pattern, with 2^nmant bins per octave
------------------------------------------------------------------------- */

void InterpTable::build(int tabstyle_in, int nmant_in)
{
  tabstyle = tabstyle_in;
  nmant = nmant_in;
  shift = 52 - nmant;

  if (ncol > 1) tabstyle = TB_LOOKUP;
  if (tabstyle == TB_SPLINE && ninput < 4)
    error->one(FLERR,"Invalid tabulated data length");

  if (tabstyle == TB_LOOKUP) ncoeff = 1;
  else if (tabstyle == TB_LINEAR) ncoeff = 2;
  else ncoeff = 5;

  xlo = xfile[0];
  xhi = xfile[ninput-1];

  // bin 0 starts at the largest power of 2 which is <= xlo
  // the last bin ends at the smallest power of 2 which is > xhi

  DoubleBits v;
  v.d = xlo;
  offset = (int64_t) (v.u >> shift);
  v.d = xhi;
  int64_t last = (int64_t) (v.u >> shift);

  int64_t nb = last - offset + 1;
  if (nb > MAXBIN)
    error->one(FLERR,"Tabulated data spans too many bins");
  nbins = (int) nb;

  memory->create(coeff,(bigint) ncoeff*ncol*nbins,"interp/table:coeff");

  // a multi-column table stores one row per bin, taken at the bin center
  //   by linear interpolation between the two bracketing input rows

  if (ncol > 1) {
    for (int k = 0; k < nbins; k++) {
      double x = 0.5*(bin_lower(k) + bin_lower(k+1));
      x = MAX(x,xlo);
      x = MIN(x,xhi);
      int klo = 0, khi = ninput-1;
      while (khi-klo > 1) {
        int mid = (khi+klo)/2;
        if (xfile[mid] > x) khi = mid; else klo = mid;
      }
      double h = xfile[khi]-xfile[klo];
      double b = (h > 0.0) ? (x-xfile[klo])/h : 0.0;
      for (int c = 0; c < ncol; c++)
        coeff[(bigint) k*ncol+c] =
          (1.0-b)*yfile[klo*ncol+c] + b*yfile[khi*ncol+c];
    }
    return;
  }

  for (int k = 0; k < nbins; k++) {
    double x0 = bin_lower(k);
    double x1 = bin_lower(k+1);
    double *c = &coeff[(bigint) ncoeff*k];
    double y0,y1,d0,d1;

    if (tabstyle == TB_LOOKUP) {
      input_value(0.5*(x0+x1),&y0,&d0);
      c[0] = y0;

    } else if (tabstyle == TB_LINEAR) {
      input_value(x0,&y0,&d0);
      input_value(x1,&y1,&d1);
      c[1] = (y1-y0) / (x1-x0);
      c[0] = y0 - c[1]*x0;

    } else {

      // cubic Hermite in u = x - x0, matching value and slope at both edges
      // the offset keeps the evaluation accurate when x is large

      input_value(x0,&y0,&d0);
      input_value(x1,&y1,&d1);
      double h = x1 - x0;
      c[0] = x0;
      c[1] = y0;
      c[2] = d0;
      c[3] = (3.0*(y1-y0)/h - 2.0*d0 - d1) / h;
      c[4] = (2.0*(y0-y1)/h + d0 + d1) / (h*h);
    }
  }
}

/* ----------------------------------------------------------------------
   largest relative deviation of the binned table from the input data
   sampled at every input point and at bin centers, so that sparse input
     cannot hide a poorly resolved table
------------------------------------------------------------------------- */

double InterpTable::check(int)
{
  if (ncol > 1) return 0.0;

  double maxerr = 0.0;
  double exact,dummy;

  for (int i = 0; i < ninput; i++) {
    input_value(xfile[i],&exact,&dummy);
    if (exact == 0.0) continue;
    maxerr = MAX(maxerr,fabs(interpolate(xfile[i])-exact)/fabs(exact));
  }

  int stride = MAX(1,nbins/MAXCHECK);
  for (int k = 0; k < nbins; k += stride) {
    double x = 0.5*(bin_lower(k) + bin_lower(k+1));
    if (x <= xlo || x >= xhi) continue;
    input_value(x,&exact,&dummy);
    if (exact == 0.0) continue;
    maxerr = MAX(maxerr,fabs(interpolate(x)-exact)/fabs(exact));
  }

  return maxerr;
}

/* ----------------------------------------------------------------------
   bin index of X, clamped into range
   the exponent and leading nmant mantissa bits of x are the bin index,
     so bins are spaced by a fixed ratio
------------------------------------------------------------------------- */

int InterpTable::bin_index(double x)
{
  DoubleBits v;
  v.d = x;
  int64_t k = (int64_t) (v.u >> shift) - offset;
  if (k < 0) k = 0;
  else if (k > nbins-1) k = nbins-1;
  return (int) k;
}

/* ----------------------------------------------------------------------
   lower edge of bin K, the inverse of bin_index()
------------------------------------------------------------------------- */

double InterpTable::bin_lower(int k)
{
  DoubleBits v;
  v.u = ((uint64_t) (offset + k)) << shift;
  return v.d;
}

/* ---------------------------------------------------------------------- */

double InterpTable::interpolate(double x)
{
  const double *c = &coeff[(bigint) ncoeff*bin_index(x)];

  if (tabstyle == TB_LINEAR) return c[0] + x*c[1];
  if (tabstyle == TB_LOOKUP) return c[0];

  double u = x - c[0];
  return c[1] + u*(c[2] + u*(c[3] + u*c[4]));
}

/* ---------------------------------------------------------------------- */

double InterpTable::evaluate(double x)
{
  // outside the tabulated range, extrapolate as a power law
  // constant, a fitted power law, and the VSS fallback are all of the
  //   form a*x^p, so this is one code path

  if (x <= xlo) {
    if (extrap_lo == TB_ERROR)
      error->one(FLERR,"Value is outside the tabulated data range");
    return alo * pow(x,plo);
  }
  if (x >= xhi) {
    if (extrap_hi == TB_ERROR)
      error->one(FLERR,"Value is outside the tabulated data range");
    return ahi * pow(x,phi);
  }

  return interpolate(x);
}

/* ----------------------------------------------------------------------
   as evaluate(), but never raises the TB_ERROR out-of-range error
   the extrapolation coefficients for TB_ERROR are the constant ones, so
     this continues the end value outside the tabulated range
------------------------------------------------------------------------- */

double InterpTable::evaluate_noerror(double x)
{
  if (x <= xlo) return alo * pow(x,plo);
  if (x >= xhi) return ahi * pow(x,phi);
  return interpolate(x);
}

/* ----------------------------------------------------------------------
   sample a row at cumulative probability U in [0,1)
   the row is taken from the bin containing X, with linear interpolation
     between the two bracketing column entries
------------------------------------------------------------------------- */

double InterpTable::interpolate_row(double x, double u)
{
  const double *c = &coeff[(bigint) ncol*bin_index(x)];

  double f = u*ncol - 0.5;
  if (f <= 0.0) return c[0];
  if (f >= ncol-1) return c[ncol-1];
  int j = (int) f;
  return c[j] + (f-j)*(c[j+1]-c[j]);
}

/* ----------------------------------------------------------------------
   cubic spline second derivatives, adapted from Numerical Recipes
   yp1,ypn = first derivatives at the end points, > 0.99e30 for natural
------------------------------------------------------------------------- */

void InterpTable::spline(double *x, double *y, int n,
                         double yp1, double ypn, double *y2)
{
  double *u;
  memory->create(u,n,"interp/table:u");

  if (yp1 > 0.99e30) y2[0] = u[0] = 0.0;
  else {
    y2[0] = -0.5;
    u[0] = (3.0/(x[1]-x[0])) * ((y[1]-y[0]) / (x[1]-x[0]) - yp1);
  }

  for (int i = 1; i < n-1; i++) {
    double sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1]);
    double p = sig*y2[i-1] + 2.0;
    y2[i] = (sig-1.0) / p;
    u[i] = (y[i+1]-y[i]) / (x[i+1]-x[i]) - (y[i]-y[i-1]) / (x[i]-x[i-1]);
    u[i] = (6.0*u[i] / (x[i+1]-x[i-1]) - sig*u[i-1]) / p;
  }

  double qn,un;
  if (ypn > 0.99e30) qn = un = 0.0;
  else {
    qn = 0.5;
    un = (3.0/(x[n-1]-x[n-2])) * (ypn - (y[n-1]-y[n-2]) / (x[n-1]-x[n-2]));
  }
  y2[n-1] = (un - qn*u[n-2]) / (qn*y2[n-2] + 1.0);
  for (int k = n-2; k >= 0; k--) y2[k] = y2[k]*y2[k+1] + u[k];

  memory->destroy(u);
}

/* ----------------------------------------------------------------------
   evaluate a cubic spline and its derivative at X
------------------------------------------------------------------------- */

void InterpTable::splint(double *xa, double *ya, double *y2a, int n,
                         double x, double *y, double *dy)
{
  int klo = 0;
  int khi = n-1;
  while (khi-klo > 1) {
    int k = (khi+klo) / 2;
    if (xa[k] > x) khi = k;
    else klo = k;
  }

  double h = xa[khi]-xa[klo];
  double a = (xa[khi]-x) / h;
  double b = (x-xa[klo]) / h;

  *y = a*ya[klo] + b*ya[khi] +
    ((a*a*a-a)*y2a[klo] + (b*b*b-b)*y2a[khi]) * (h*h)/6.0;
  *dy = (ya[khi]-ya[klo])/h +
    ((3.0*b*b-1.0)*y2a[khi] - (3.0*a*a-1.0)*y2a[klo]) * h/6.0;
}

/* ----------------------------------------------------------------------
   broadcast a built table from proc 0
   only the run-time data is sent, the raw input is already freed
------------------------------------------------------------------------- */

void InterpTable::bcast()
{
  int ibuf[9];
  if (comm->me == 0) {
    ibuf[0] = ninput;     ibuf[1] = ncol;       ibuf[2] = xvar;
    ibuf[3] = extrap_lo;  ibuf[4] = extrap_hi;  ibuf[5] = nbins;
    ibuf[6] = tabstyle;   ibuf[7] = ncoeff;     ibuf[8] = nmant;
  }
  MPI_Bcast(ibuf,9,MPI_INT,0,world);
  if (comm->me) {
    ninput = ibuf[0];     ncol = ibuf[1];       xvar = ibuf[2];
    extrap_lo = ibuf[3];  extrap_hi = ibuf[4];  nbins = ibuf[5];
    tabstyle = ibuf[6];   ncoeff = ibuf[7];     nmant = ibuf[8];
    shift = 52 - nmant;
  }

  MPI_Bcast(&offset,sizeof(int64_t),MPI_BYTE,0,world);

  double dbuf[8];
  if (comm->me == 0) {
    dbuf[0] = xlo;  dbuf[1] = xhi;  dbuf[2] = alo;  dbuf[3] = plo;
    dbuf[4] = ahi;  dbuf[5] = phi;  dbuf[6] = ymax;  dbuf[7] = ymin;
  }
  MPI_Bcast(dbuf,8,MPI_DOUBLE,0,world);
  if (comm->me) {
    xlo = dbuf[0];  xhi = dbuf[1];  alo = dbuf[2];  plo = dbuf[3];
    ahi = dbuf[4];  phi = dbuf[5];  ymax = dbuf[6];  ymin = dbuf[7];
  }

  bigint ntotal = (bigint) ncoeff*ncol*nbins;
  if (comm->me) memory->create(coeff,ntotal,"interp/table:coeff");
  MPI_Bcast(coeff,ntotal,MPI_DOUBLE,0,world);

  // file and keyword strings, kept for error messages

  int n;
  if (comm->me == 0) n = strlen(file) + 1;
  MPI_Bcast(&n,1,MPI_INT,0,world);
  if (comm->me) { delete [] file; file = new char[n]; }
  MPI_Bcast(file,n,MPI_CHAR,0,world);

  if (comm->me == 0) n = strlen(keyword) + 1;
  MPI_Bcast(&n,1,MPI_INT,0,world);
  if (comm->me) { delete [] keyword; keyword = new char[n]; }
  MPI_Bcast(keyword,n,MPI_CHAR,0,world);
}
