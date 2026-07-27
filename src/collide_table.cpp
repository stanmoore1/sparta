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
#include "collide_table.h"
#include "update.h"
#include "particle.h"
#include "mixture.h"
#include "react.h"
#include "comm.h"
#include "random_knuth.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

enum{LOOKUP,LINEAR,SPLINE};                      // interpolation styles
enum{ENERGY,SPEED};                              // table independent variable
enum{EX_CONSTANT,EX_POWERLAW,EX_VSS,EX_ERROR};   // extrapolation modes

#define MAXLINE 1024
#define EPSZERO 1.0e-14
#define EV2J 1.602176634e-19        // electron volt in Joules
#define ANG2SQ 1.0e-20              // Angstrom^2 in m^2
#define CM2SQ 1.0e-4                // cm^2 in m^2
#define TABLE_TOL 0.01              // binned vs input table tolerance
#define MAXMANT 20                  // max mantissa bits used to index a bin
#define MAXBIN (1 << 24)            // max bins in one table
#define MAXCHECK 20000              // max bins sampled by the accuracy check

// reinterpret a positive double as its IEEE-754 bit pattern

union DoubleBits { double d; uint64_t u; };

/* ----------------------------------------------------------------------
   collide table = tabulated total collision cross sections
   syntax: collide table <mixture> <param-file> <interp-style> <N> [relax ...]
   derives from CollideVSS: species pairs without a table use the analytic
     VSS form, and the VSS params in the file are still used by the React
     styles and by compute lambda/grid via extract()
------------------------------------------------------------------------- */

CollideTable::CollideTable(SPARTA *sparta, int narg, char **arg) :
  CollideVSS(sparta, narg, arg, 0)
{
  if (narg < 5) error->all(FLERR,"Illegal collide command");

  if (strcmp(arg[3],"lookup") == 0) {
    tabstyle = LOOKUP;
    ncoeff = 1;
  } else if (strcmp(arg[3],"linear") == 0) {
    tabstyle = LINEAR;
    ncoeff = 2;
  } else if (strcmp(arg[3],"spline") == 0) {
    tabstyle = SPLINE;
    ncoeff = 5;
  } else error->all(FLERR,"Unknown table style in collide table");

  // N = requested bins per factor of 2 in relative energy
  // round up to a power of 2 so a bin index is a shift of the vr^2 bits

  int nrequest = atoi(arg[4]);
  if (nrequest < 1 || nrequest > (1 << MAXMANT))
    error->all(FLERR,"Illegal number of collide table entries");
  nmant = 0;
  while ((1 << nmant) < nrequest) nmant++;

  // remaining args are the standard VSS optional args

  parse_vss_args(5,narg,arg);

  ntables = 0;
  tables = NULL;

  // proc 0 reads the param file, which yields both the VSS params
  //   and the per-pair cross section tables, then broadcasts everything

  allocate_params();

  memory->create(tabindex,nparams,nparams,"collide/table:tabindex");
  for (int i = 0; i < nparams; i++)
    for (int j = 0; j < nparams; j++) tabindex[i][j] = -1;

  if (comm->me == 0) read_param_file(arg[2]);

  MPI_Bcast(params[0],nparams*nparams*sizeof(Params),MPI_BYTE,0,world);
  MPI_Bcast(&tabindex[0][0],nparams*nparams,MPI_INT,0,world);
  MPI_Bcast(&ntables,1,MPI_INT,0,world);

  if (comm->me && ntables) {
    tables = (Table *) memory->smalloc(ntables*sizeof(Table),
                                       "collide/table:tables");
    for (int m = 0; m < ntables; m++) null_table(&tables[m]);
  }
  for (int m = 0; m < ntables; m++) bcast_table(&tables[m]);

  if (ntables == 0)
    error->warning(FLERR,"No cross section tables were defined by collide table");
}

/* ---------------------------------------------------------------------- */

CollideTable::~CollideTable()
{
  if (copymode) return;

  for (int m = 0; m < ntables; m++) free_table(&tables[m]);
  memory->sfree(tables);
  memory->destroy(tabindex);
}

/* ----------------------------------------------------------------------
   estimate a good initial value for vremax for a group pair
   parent computes the VSS estimate and fills prefactor[][],
     both of which are still needed for the non-tabulated pairs
   for tabulated pairs take the max with the tabulated sigma*g evaluated at
     the characteristic relative speed, the same estimate VSS makes with pi*d^2
   taking a max is always safe: too large a vremax costs attempts, never bias
------------------------------------------------------------------------- */

double CollideTable::vremax_init(int igroup, int jgroup)
{
  double vrmgroup = CollideVSS::vremax_init(igroup,jgroup);

  double *vscale = mixture->vscale;
  int *mix2group = mixture->mix2group;
  int nspecies = particle->nspecies;

  for (int isp = 0; isp < nspecies; isp++) {
    if (mix2group[isp] != igroup) continue;
    for (int jsp = 0; jsp < nspecies; jsp++) {
      if (mix2group[jsp] != jgroup) continue;
      int m = tabindex[isp][jsp];
      if (m < 0) continue;

      // interpolate with the bin clamped into the table, so that a table
      //   which does not span the thermal speed cannot abort the setup

      double beta = MAX(vscale[isp],vscale[jsp]);
      double vrm = 2.0 * interp_sigma_g(&tables[m],beta*beta);
      vrmgroup = MAX(vrmgroup,vrm);
    }
  }

  return vrmgroup;
}

/* ----------------------------------------------------------------------
   determine if collision actually occurs
   1 = yes, 0 = no
   update vremax either way
------------------------------------------------------------------------- */

int CollideTable::test_collision(int icell, int igroup, int jgroup,
                                 Particle::OnePart *ip, Particle::OnePart *jp)
{
  int ispecies = ip->ispecies;
  int jspecies = jp->ispecies;
  int m = tabindex[ispecies][jspecies];

  // no table for this pair, use the analytic VSS cross section

  if (m < 0) {
    if (react) react_prob_factor = 1.0;
    return CollideVSS::test_collision(icell,igroup,jgroup,ip,jp);
  }

  double *vi = ip->v;
  double *vj = jp->v;
  double du  = vi[0] - vj[0];
  double dv  = vi[1] - vj[1];
  double dw  = vi[2] - vj[2];
  double vr2 = du*du + dv*dv + dw*dw;

  // prevent a division by zero, and a denormal vr^2 in the bin index

  if (vr2 < EPSZERO) return 0;

  // vre = sigma(vr) * vr, interpolated directly, no pow() and no sqrt()

  double vre = sigma_g(m,vr2);

  vremax[icell][igroup][jgroup] = MAX(vre,vremax[icell][igroup][jgroup]);
  if (vre/vremax[icell][igroup][jgroup] < random->uniform()) return 0;

  // the TCE model derives its reaction probability as sigma_react/sigma_VHS
  // pairs were selected here with sigma_table instead, so hand the React
  //   style the ratio which restores the intended reaction rate

  if (react)
    react_prob_factor = prefactor[ispecies][jspecies] *
      pow(vr2,1.0-params[ispecies][jspecies].omega) / vre;

  precoln.vr2 = vr2;
  return 1;
}

/* ----------------------------------------------------------------------
   sigma*g for table M at relative velocity squared VR2
------------------------------------------------------------------------- */

double CollideTable::sigma_g(int m, double vr2)
{
  Table *tb = &tables[m];

  // outside the tabulated range, extrapolate as a power law in vr^2
  // constant sigma, a fitted power law, and the VSS fallback are all
  //   expressed as a*vr2^p, so this is one code path

  if (vr2 <= tb->vr2lo) {
    if (tb->extrap_lo == EX_ERROR)
      error->one(FLERR,"Collision energy is outside the cross section table range");
    return tb->alo * pow(vr2,tb->plo);
  }
  if (vr2 >= tb->vr2hi) {
    if (tb->extrap_hi == EX_ERROR)
      error->one(FLERR,"Collision energy is outside the cross section table range");
    return tb->ahi * pow(vr2,tb->phi);
  }

  return interp_sigma_g(tb,vr2);
}

/* ----------------------------------------------------------------------
   bin index of VR2 within table TB
   the exponent and leading nmant mantissa bits of vr^2 are the bin index,
     so bins are spaced by a fixed ratio in energy
   caller guarantees vr2 is inside (vr2lo,vr2hi), the clamp only guards
     against a rounding case landing on an end bin
------------------------------------------------------------------------- */

int CollideTable::bin_index(Table *tb, double vr2)
{
  DoubleBits v;
  v.d = vr2;
  int64_t k = (int64_t) (v.u >> tb->shift) - tb->offset;
  if (k < 0) k = 0;
  else if (k > tb->nbins-1) k = tb->nbins-1;
  return (int) k;
}

/* ----------------------------------------------------------------------
   lower edge of bin K, the inverse of bin_index()
------------------------------------------------------------------------- */

double CollideTable::bin_lower(Table *tb, int k)
{
  DoubleBits v;
  v.u = ((uint64_t) (tb->offset + k)) << tb->shift;
  return v.d;
}

/* ----------------------------------------------------------------------
   interpolate the binned table with the index clamped into range
   used away from the hot path, where vr2 may be on or beyond an end bin
------------------------------------------------------------------------- */

double CollideTable::interp_sigma_g(Table *tb, double vr2)
{
  const double *c = &tb->coeff[ncoeff*bin_index(tb,vr2)];

  if (tabstyle == LINEAR) return c[0] + vr2*c[1];
  if (tabstyle == LOOKUP) return c[0];

  // SPLINE: cubic in u = vr2 - c[0], the lower edge of the bin
  // the offset keeps the evaluation accurate when vr2 is large

  double u = vr2 - c[0];
  return c[1] + u*(c[2] + u*(c[3] + u*c[4]));
}

/* ----------------------------------------------------------------------
   read the collide table param file
   it is a VSS param file plus "table" directive lines:
     <species1> <species2> table <filename> <keyword>
   only invoked by proc 0
------------------------------------------------------------------------- */

void CollideTable::read_param_file(char *fname)
{
  // first pass: the VSS params
  // skip_param_line() makes the parent loop ignore the table directives

  CollideVSS::read_param_file(fname);

  // second pass: the table directives
  // params[][].mr and the VSS params are set by now, and both the unit
  //   conversion and the vss extrapolation mode need them

  FILE *fp = fopen(fname,"r");
  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open VSS parameter file %s",fname);
    error->one(FLERR,str);
  }

  char *words[6];
  char line[MAXLINE];

  while (fgets(line,MAXLINE,fp)) {
    int pre = strspn(line," \t\n\r");
    if (pre == strlen(line) || line[pre] == '#') continue;

    int nwords = wordparse(6,line,words);
    if (!skip_param_line(nwords,words)) continue;
    if (nwords < 5)
      error->one(FLERR,"Incorrect line format in VSS parameter file");

    int isp = particle->find_species(words[0]);
    int jsp = particle->find_species(words[1]);

    // silently ignore a directive for species not in this simulation,
    //   consistent with how the VSS param lines are handled

    if (isp < 0 || jsp < 0) continue;

    tables = (Table *)
      memory->srealloc(tables,(ntables+1)*sizeof(Table),"collide/table:tables");
    Table *tb = &tables[ntables];
    null_table(tb);
    tb->isp = isp;
    tb->jsp = jsp;

    read_table(tb,words[3],words[4]);
    convert_table(tb);
    compute_table(tb);
    check_table(tb);

    memory->destroy(tb->xfile);
    memory->destroy(tb->sfile);
    memory->destroy(tb->xspl);
    memory->destroy(tb->yspl);
    memory->destroy(tb->yspl2);
    tb->xfile = tb->sfile = NULL;
    tb->xspl = tb->yspl = tb->yspl2 = NULL;

    tabindex[isp][jsp] = tabindex[jsp][isp] = ntables;
    ntables++;
  }

  fclose(fp);
}

/* ----------------------------------------------------------------------
   return 1 if this param file line is a table directive
   invoked by CollideVSS::read_param_file() so it can skip these lines
------------------------------------------------------------------------- */

int CollideTable::skip_param_line(int nwords, char **words)
{
  if (nwords > 2 && strcmp(words[2],"table") == 0) return 1;
  return 0;
}

/* ----------------------------------------------------------------------
   read one section of a cross section table file
   only invoked by proc 0
------------------------------------------------------------------------- */

void CollideTable::read_table(Table *tb, char *file, char *keyword)
{
  tb->file = new char[strlen(file)+1];
  strcpy(tb->file,file);
  tb->keyword = new char[strlen(keyword)+1];
  strcpy(tb->keyword,keyword);

  FILE *fp = fopen(file,"r");
  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open cross section table file %s",file);
    error->one(FLERR,str);
  }

  // scan for a line whose first word matches the keyword

  char line[MAXLINE],copy[MAXLINE];
  while (1) {
    if (fgets(line,MAXLINE,fp) == NULL) {
      char str[256];
      sprintf(str,"Did not find keyword %s in cross section table file %s",
              keyword,file);
      error->one(FLERR,str);
    }
    int pre = strspn(line," \t\n\r");
    if (pre == strlen(line) || line[pre] == '#') continue;
    strcpy(copy,line);
    char *word = strtok(copy," \t\n\r");
    if (word && strcmp(word,keyword) == 0) break;
  }

  // parameter line follows the keyword line

  if (fgets(line,MAXLINE,fp) == NULL)
    error->one(FLERR,"Premature end of cross section table file");
  param_extract(tb,line,file,keyword);

  memory->create(tb->xfile,tb->ninput,"collide/table:xfile");
  memory->create(tb->sfile,tb->ninput,"collide/table:sfile");

  // ninput rows of: index x sigma
  // blank and comment lines are allowed between rows

  int n = 0;
  while (n < tb->ninput) {
    if (fgets(line,MAXLINE,fp) == NULL)
      error->one(FLERR,"Premature end of cross section table file");
    int pre = strspn(line," \t\n\r");
    if (pre == strlen(line) || line[pre] == '#') continue;

    strtok(line," \t\n\r");
    char *w2 = strtok(NULL," \t\n\r");
    char *w3 = strtok(NULL," \t\n\r");
    if (w2 == NULL || w3 == NULL)
      error->one(FLERR,"Incorrect line format in cross section table file");

    tb->xfile[n] = atof(w2);
    tb->sfile[n] = atof(w3);
    n++;
  }

  fclose(fp);
}

/* ----------------------------------------------------------------------
   parse the parameter line which follows a section keyword
   N <n> [X energy|speed] [XUNITS eV|J|K|m/s] [YUNITS m^2|cm^2|A^2]
     [EXTRAP <lo> <hi>]
------------------------------------------------------------------------- */

void CollideTable::param_extract(Table *tb, char *line,
                                 char *file, char *keyword)
{
  tb->ninput = 0;
  tb->xvar = ENERGY;
  tb->xscale = 0.0;
  tb->yscale = 1.0;
  tb->extrap_lo = tb->extrap_hi = EX_CONSTANT;

  char *word = strtok(line," \t\n\r");
  while (word) {
    if (strcmp(word,"N") == 0) {
      word = strtok(NULL," \t\n\r");
      if (!word) error->one(FLERR,"Invalid keyword in cross section table parameters");
      tb->ninput = atoi(word);

    } else if (strcmp(word,"X") == 0) {
      word = strtok(NULL," \t\n\r");
      if (!word) error->one(FLERR,"Invalid keyword in cross section table parameters");
      if (strcmp(word,"energy") == 0) tb->xvar = ENERGY;
      else if (strcmp(word,"speed") == 0) tb->xvar = SPEED;
      else error->one(FLERR,"Invalid keyword in cross section table parameters");

    } else if (strcmp(word,"XUNITS") == 0) {
      word = strtok(NULL," \t\n\r");
      if (!word) error->one(FLERR,"Invalid keyword in cross section table parameters");
      if (strcmp(word,"eV") == 0) tb->xscale = EV2J;
      else if (strcmp(word,"J") == 0) tb->xscale = 1.0;
      else if (strcmp(word,"K") == 0) tb->xscale = update->boltz;
      else if (strcmp(word,"m/s") == 0) tb->xscale = 1.0;
      else error->one(FLERR,"Invalid keyword in cross section table parameters");

    } else if (strcmp(word,"YUNITS") == 0) {
      word = strtok(NULL," \t\n\r");
      if (!word) error->one(FLERR,"Invalid keyword in cross section table parameters");
      if (strcmp(word,"m^2") == 0) tb->yscale = 1.0;
      else if (strcmp(word,"cm^2") == 0) tb->yscale = CM2SQ;
      else if (strcmp(word,"A^2") == 0) tb->yscale = ANG2SQ;
      else error->one(FLERR,"Invalid keyword in cross section table parameters");

    } else if (strcmp(word,"EXTRAP") == 0) {
      int *mode = &tb->extrap_lo;
      for (int i = 0; i < 2; i++) {
        word = strtok(NULL," \t\n\r");
        if (!word) error->one(FLERR,"Invalid keyword in cross section table parameters");
        if (strcmp(word,"constant") == 0) *mode = EX_CONSTANT;
        else if (strcmp(word,"powerlaw") == 0) *mode = EX_POWERLAW;
        else if (strcmp(word,"vss") == 0) *mode = EX_VSS;
        else if (strcmp(word,"error") == 0) *mode = EX_ERROR;
        else error->one(FLERR,"Invalid keyword in cross section table parameters");
        mode = &tb->extrap_hi;
      }

    } else error->one(FLERR,"Invalid keyword in cross section table parameters");

    word = strtok(NULL," \t\n\r");
  }

  // default x units depend on the independent variable

  if (tb->xscale == 0.0) tb->xscale = (tb->xvar == ENERGY) ? EV2J : 1.0;

  if (tb->ninput == 0) {
    char str[256];
    sprintf(str,"Cross section table parameters did not set N "
            "for keyword %s in file %s",keyword,file);
    error->one(FLERR,str);
  }
  if (tb->ninput < 2 || (tabstyle == SPLINE && tb->ninput < 4))
    error->one(FLERR,"Invalid cross section table length");
}

/* ----------------------------------------------------------------------
   convert the file data to SI and to vr^2:
     xfile: file x (energy or speed, file units) -> vr^2 in m^2/s^2
     sfile: file sigma (file units) -> sigma in m^2
   build the spline of the input data, in log-log where possible
   set the extrapolation coefficients, in the common form a*vr2^p
------------------------------------------------------------------------- */

void CollideTable::convert_table(Table *tb)
{
  int n = tb->ninput;
  int isp = tb->isp;
  int jsp = tb->jsp;
  double mr = params[isp][jsp].mr;

  for (int i = 0; i < n; i++) {
    double x = tb->xfile[i] * tb->xscale;
    if (tb->xvar == ENERGY) tb->xfile[i] = 2.0*x/mr;
    else tb->xfile[i] = x*x;
    tb->sfile[i] *= tb->yscale;
  }

  tb->sigmax = 0.0;
  tb->logflag = 1;
  for (int i = 0; i < n; i++) {
    if (tb->sfile[i] < 0.0)
      error->one(FLERR,"Cross section table has a negative cross section");
    if (tb->sfile[i] == 0.0) tb->logflag = 0;
    tb->sigmax = MAX(tb->sigmax,tb->sfile[i]);
    if (i && tb->xfile[i] <= tb->xfile[i-1])
      error->one(FLERR,"Cross section table values are not increasing");
  }
  if (tb->xfile[0] <= 0.0)
    error->one(FLERR,"Cross section table values must be positive");

  // spline the input data
  // log-log unless a cross section is zero, e.g. below a threshold
  // end slopes from finite differences, so a straight line in the splined
  //   variables, i.e. a power law in the log-log case, is reproduced exactly

  memory->create(tb->xspl,n,"collide/table:xspl");
  memory->create(tb->yspl,n,"collide/table:yspl");
  memory->create(tb->yspl2,n,"collide/table:yspl2");

  for (int i = 0; i < n; i++) {
    tb->xspl[i] = tb->logflag ? log(tb->xfile[i]) : tb->xfile[i];
    tb->yspl[i] = tb->logflag ? log(tb->sfile[i]) : tb->sfile[i];
  }
  double ep0 = (tb->yspl[1]-tb->yspl[0]) / (tb->xspl[1]-tb->xspl[0]);
  double epn = (tb->yspl[n-1]-tb->yspl[n-2]) / (tb->xspl[n-1]-tb->xspl[n-2]);
  spline(tb->xspl,tb->yspl,n,ep0,epn,tb->yspl2);

  // extrapolation coefficients, sigma*g = a * vr2^p
  //   constant sigma       -> p = 1/2
  //   sigma ~ vr2^q fit log-log from the two end points -> p = q+1/2
  //   VSS fallback         -> p = 1-omega, a = the VSS prefactor

  double omega = params[isp][jsp].omega;
  double cxs = MY_PI*params[isp][jsp].diam*params[isp][jsp].diam;
  double vssa = cxs * pow(2.0*update->boltz*params[isp][jsp].tref/mr,omega-0.5) /
    tgamma(2.5-omega);

  if (tb->extrap_lo == EX_VSS) {
    tb->alo = vssa;
    tb->plo = 1.0 - omega;
  } else if (tb->extrap_lo == EX_POWERLAW &&
             tb->sfile[0] > 0.0 && tb->sfile[1] > 0.0) {
    double q = log(tb->sfile[1]/tb->sfile[0]) / log(tb->xfile[1]/tb->xfile[0]);
    tb->plo = q + 0.5;
    tb->alo = tb->sfile[0] / pow(tb->xfile[0],q);
  } else {
    tb->plo = 0.5;
    tb->alo = tb->sfile[0];
  }

  if (tb->extrap_hi == EX_VSS) {
    tb->ahi = vssa;
    tb->phi = 1.0 - omega;
  } else if (tb->extrap_hi == EX_POWERLAW &&
             tb->sfile[n-1] > 0.0 && tb->sfile[n-2] > 0.0) {
    double q = log(tb->sfile[n-1]/tb->sfile[n-2]) /
      log(tb->xfile[n-1]/tb->xfile[n-2]);
    tb->phi = q + 0.5;
    tb->ahi = tb->sfile[n-1] / pow(tb->xfile[n-1],q);
  } else {
    tb->phi = 0.5;
    tb->ahi = tb->sfile[n-1];
  }
}

/* ----------------------------------------------------------------------
   sigma*g and its derivative at VR2, from the input data spline
------------------------------------------------------------------------- */

void CollideTable::input_sg(Table *tb, double vr2, double *y, double *dy)
{
  double s,ds;
  int n = tb->ninput;

  if (tb->logflag) {

    // yspl = ln(sigma) vs xspl = ln(vr2), so
    //   sigma = exp(S) and d(sigma)/d(vr2) = sigma*S'/vr2
    //   d(sigma*g)/d(vr2) = sigma/sqrt(vr2) * (S' + 1/2)

    double lnsig,dlnsig;
    splint(tb->xspl,tb->yspl,tb->yspl2,n,log(vr2),&lnsig,&dlnsig);
    s = exp(lnsig);
    double rt = sqrt(vr2);
    *y = s*rt;
    *dy = s/rt * (dlnsig + 0.5);

  } else {
    splint(tb->xspl,tb->yspl,tb->yspl2,n,vr2,&s,&ds);
    double rt = sqrt(vr2);
    *y = s*rt;
    *dy = ds*rt + 0.5*s/rt;
  }
}

/* ----------------------------------------------------------------------
   build the binned table
   bins cover whole octaves of vr^2, so that a bin index is a shift of the
     vr^2 bit pattern, with 2^nmant bins per octave
------------------------------------------------------------------------- */

void CollideTable::compute_table(Table *tb)
{
  tb->vr2lo = tb->xfile[0];
  tb->vr2hi = tb->xfile[tb->ninput-1];

  // bin 0 starts at the largest power of 2 which is <= vr2lo
  // the last bin ends at the smallest power of 2 which is > vr2hi

  tb->shift = 52 - nmant;

  DoubleBits v;
  v.d = tb->vr2lo;
  tb->offset = (int64_t) (v.u >> tb->shift);
  v.d = tb->vr2hi;
  int64_t last = (int64_t) (v.u >> tb->shift);

  int64_t nbins = last - tb->offset + 1;
  if (nbins > MAXBIN)
    error->one(FLERR,"Cross section table spans too many bins");
  tb->nbins = (int) nbins;

  memory->create(tb->coeff,ncoeff*tb->nbins,"collide/table:coeff");

  for (int k = 0; k < tb->nbins; k++) {
    double x0 = bin_lower(tb,k);
    double x1 = bin_lower(tb,k+1);
    double *c = &tb->coeff[ncoeff*k];
    double y0,y1,d0,d1;

    if (tabstyle == LOOKUP) {
      input_sg(tb,0.5*(x0+x1),&y0,&d0);
      c[0] = y0;

    } else if (tabstyle == LINEAR) {
      input_sg(tb,x0,&y0,&d0);
      input_sg(tb,x1,&y1,&d1);
      c[1] = (y1-y0) / (x1-x0);
      c[0] = y0 - c[1]*x0;

    } else {

      // cubic Hermite in u = x - x0, matching value and slope at both edges

      input_sg(tb,x0,&y0,&d0);
      input_sg(tb,x1,&y1,&d1);
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
   check that the binned table reproduces the input data
   sampled at every input point and at the midpoint of every bin, so that
     sparse input data cannot hide a poorly resolved table
   report the table range, resolution, and worst-case deviation
------------------------------------------------------------------------- */

void CollideTable::check_table(Table *tb)
{
  double maxerr = 0.0;
  double exact,dummy;

  for (int i = 0; i < tb->ninput; i++) {
    input_sg(tb,tb->xfile[i],&exact,&dummy);
    if (exact == 0.0) continue;
    maxerr = MAX(maxerr,fabs(interp_sigma_g(tb,tb->xfile[i])-exact)/exact);
  }

  int stride = MAX(1,tb->nbins/MAXCHECK);
  for (int k = 0; k < tb->nbins; k += stride) {
    double x = 0.5*(bin_lower(tb,k) + bin_lower(tb,k+1));
    if (x <= tb->vr2lo || x >= tb->vr2hi) continue;
    input_sg(tb,x,&exact,&dummy);
    if (exact == 0.0) continue;
    maxerr = MAX(maxerr,fabs(interp_sigma_g(tb,x)-exact)/exact);
  }

  double mr = params[tb->isp][tb->jsp].mr;
  char str[512];
  sprintf(str,"Cross section table %s from %s:\n"
          "  %d values, E = %.4g to %.4g eV, max sigma = %.4g m^2, "
          "%s interpolation\n"
          "  %d bins, %d per factor of 2 in energy, "
          "reproduces input to %.3g%%",
          tb->keyword,tb->file,tb->ninput,
          0.5*mr*tb->vr2lo/EV2J,0.5*mr*tb->vr2hi/EV2J,tb->sigmax,
          tb->logflag ? "log-log" : "linear",
          tb->nbins,1 << nmant,100.0*maxerr);
  if (screen) fprintf(screen,"%s\n",str);
  if (logfile) fprintf(logfile,"%s\n",str);

  if (maxerr > TABLE_TOL)
    error->warning(FLERR,"Cross section table does not reproduce its input values");
}

/* ----------------------------------------------------------------------
   cubic spline second derivatives, adapted from Numerical Recipes
   yp1,ypn = first derivatives at the end points, > 0.99e30 for natural
------------------------------------------------------------------------- */

void CollideTable::spline(double *x, double *y, int n,
                          double yp1, double ypn, double *y2)
{
  double *u;
  memory->create(u,n,"collide/table:u");

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
   X outside [xa[0],xa[n-1]] is evaluated on the nearest segment,
     which callers avoid except by rounding at the ends
------------------------------------------------------------------------- */

void CollideTable::splint(double *xa, double *ya, double *y2a, int n,
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

/* ---------------------------------------------------------------------- */

void CollideTable::null_table(Table *tb)
{
  tb->file = tb->keyword = NULL;
  tb->xfile = tb->sfile = NULL;
  tb->xspl = tb->yspl = tb->yspl2 = NULL;
  tb->coeff = NULL;
  tb->logflag = 0;
  tb->sigmax = 0.0;
}

/* ---------------------------------------------------------------------- */

void CollideTable::free_table(Table *tb)
{
  delete [] tb->file;
  delete [] tb->keyword;
  memory->destroy(tb->xfile);
  memory->destroy(tb->sfile);
  memory->destroy(tb->xspl);
  memory->destroy(tb->yspl);
  memory->destroy(tb->yspl2);
  memory->destroy(tb->coeff);
}

/* ----------------------------------------------------------------------
   broadcast one built table from proc 0
   only the run-time data is sent, the raw file data is already freed
------------------------------------------------------------------------- */

void CollideTable::bcast_table(Table *tb)
{
  int ibuf[7];
  if (comm->me == 0) {
    ibuf[0] = tb->isp;        ibuf[1] = tb->jsp;
    ibuf[2] = tb->ninput;     ibuf[3] = tb->xvar;
    ibuf[4] = tb->extrap_lo;  ibuf[5] = tb->extrap_hi;
    ibuf[6] = tb->nbins;
  }
  MPI_Bcast(ibuf,7,MPI_INT,0,world);
  if (comm->me) {
    tb->isp = ibuf[0];        tb->jsp = ibuf[1];
    tb->ninput = ibuf[2];     tb->xvar = ibuf[3];
    tb->extrap_lo = ibuf[4];  tb->extrap_hi = ibuf[5];
    tb->nbins = ibuf[6];
    tb->shift = 52 - nmant;
  }

  MPI_Bcast(&tb->offset,sizeof(int64_t),MPI_BYTE,0,world);

  double dbuf[6];
  if (comm->me == 0) {
    dbuf[0] = tb->vr2lo;  dbuf[1] = tb->vr2hi;
    dbuf[2] = tb->alo;    dbuf[3] = tb->plo;
    dbuf[4] = tb->ahi;    dbuf[5] = tb->phi;
  }
  MPI_Bcast(dbuf,6,MPI_DOUBLE,0,world);
  if (comm->me) {
    tb->vr2lo = dbuf[0];  tb->vr2hi = dbuf[1];
    tb->alo = dbuf[2];    tb->plo = dbuf[3];
    tb->ahi = dbuf[4];    tb->phi = dbuf[5];
  }

  if (comm->me)
    memory->create(tb->coeff,ncoeff*tb->nbins,"collide/table:coeff");
  MPI_Bcast(tb->coeff,ncoeff*tb->nbins,MPI_DOUBLE,0,world);

  // file and keyword strings, kept for error messages

  int n;
  if (comm->me == 0) n = strlen(tb->file) + 1;
  MPI_Bcast(&n,1,MPI_INT,0,world);
  if (comm->me) tb->file = new char[n];
  MPI_Bcast(tb->file,n,MPI_CHAR,0,world);

  if (comm->me == 0) n = strlen(tb->keyword) + 1;
  MPI_Bcast(&n,1,MPI_INT,0,world);
  if (comm->me) tb->keyword = new char[n];
  MPI_Bcast(tb->keyword,n,MPI_CHAR,0,world);
}
