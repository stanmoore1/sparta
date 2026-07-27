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
#include "interp_table.h"
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

#define MAXLINE 1024
#define EPSZERO 1.0e-14
#define EV2J 1.602176634e-19
#define TABLE_TOL 0.01              // binned vs input table tolerance
#define MAXMANT 20                  // max mantissa bits used to index a bin

// temperature grid for the effective cross section used by lambda/grid

#define NTEMP 241
#define TEMPLO 1.0
#define TEMPHI 1.0e6

// kinds of table directive

enum{KSIGMA,KALPHA,KSCATTER};

/* ----------------------------------------------------------------------
   syntax: collide table <mixture> <param-file> <interp-style> <N> [relax ...]
   derives from CollideVSS: pairs without a table use the analytic VSS
     form, and the VSS params in the file are still used by the React
     styles, by the vss extrapolation mode, and by extract()
------------------------------------------------------------------------- */

CollideTable::CollideTable(SPARTA *sparta, int narg, char **arg) :
  CollideVSS(sparta, narg, arg, 0)
{
  if (narg < 5) error->all(FLERR,"Illegal collide command");

  if (strcmp(arg[3],"lookup") == 0) tabstyle = TB_LOOKUP;
  else if (strcmp(arg[3],"linear") == 0) tabstyle = TB_LINEAR;
  else if (strcmp(arg[3],"spline") == 0) tabstyle = TB_SPLINE;
  else error->all(FLERR,"Unknown table style in collide table");

  // N = requested bins per factor of 2 in relative energy
  // round up to a power of 2 so a bin index is a shift of the x bits

  int nrequest = atoi(arg[4]);
  if (nrequest < 1 || nrequest > (1 << MAXMANT))
    error->all(FLERR,"Illegal number of collide table entries");
  nmant = 0;
  while ((1 << nmant) < nrequest) nmant++;

  // remaining args are the standard VSS optional args

  parse_vss_args(5,narg,arg);

  nsigma = nalpha = nscatter = 0;
  sigma_tab = alpha_tab = scatter_tab = NULL;
  sigeff = NULL;

  // proc 0 reads the param file, which yields both the VSS params
  //   and the per-pair tables, then broadcasts everything

  allocate_params();

  memory->create(sigma_index,nparams,nparams,"collide/table:sigma_index");
  memory->create(alpha_index,nparams,nparams,"collide/table:alpha_index");
  memory->create(scatter_index,nparams,nparams,"collide/table:scatter_index");
  for (int i = 0; i < nparams; i++)
    for (int j = 0; j < nparams; j++)
      sigma_index[i][j] = alpha_index[i][j] = scatter_index[i][j] = -1;

  if (comm->me == 0) read_param_file(arg[2]);

  MPI_Bcast(params[0],nparams*nparams*sizeof(Params),MPI_BYTE,0,world);
  MPI_Bcast(&sigma_index[0][0],nparams*nparams,MPI_INT,0,world);
  MPI_Bcast(&alpha_index[0][0],nparams*nparams,MPI_INT,0,world);
  MPI_Bcast(&scatter_index[0][0],nparams*nparams,MPI_INT,0,world);

  int nbuf[3];
  if (comm->me == 0) {
    nbuf[0] = nsigma; nbuf[1] = nalpha; nbuf[2] = nscatter;
  }
  MPI_Bcast(nbuf,3,MPI_INT,0,world);

  InterpTable ***lists[3] = {&sigma_tab,&alpha_tab,&scatter_tab};
  int *counts[3] = {&nsigma,&nalpha,&nscatter};
  for (int k = 0; k < 3; k++) {
    if (comm->me) {
      *counts[k] = nbuf[k];
      if (nbuf[k]) {
        *lists[k] = new InterpTable*[nbuf[k]];
        for (int m = 0; m < nbuf[k]; m++) (*lists[k])[m] = new InterpTable(sparta);
      }
    }
    for (int m = 0; m < *counts[k]; m++) (*lists[k])[m]->bcast();
  }

  if (nsigma == 0 && comm->me == 0)
    error->warning(FLERR,"No cross section tables were defined by collide table");
}

/* ---------------------------------------------------------------------- */

CollideTable::~CollideTable()
{
  if (copymode) return;

  for (int m = 0; m < nsigma; m++) delete sigma_tab[m];
  for (int m = 0; m < nalpha; m++) delete alpha_tab[m];
  for (int m = 0; m < nscatter; m++) delete scatter_tab[m];
  delete [] sigma_tab;
  delete [] alpha_tab;
  delete [] scatter_tab;

  memory->destroy(sigma_index);
  memory->destroy(alpha_index);
  memory->destroy(scatter_index);
  memory->destroy(sigeff);
}

/* ---------------------------------------------------------------------- */

void CollideTable::init()
{
  CollideVSS::init();
  build_sigeff();
}

/* ----------------------------------------------------------------------
   estimate a good initial value for vremax for a group pair
   parent computes the VSS estimate and fills prefactor[][],
     both of which are still needed for the non-tabulated pairs
   for tabulated pairs take the max with the tabulated sigma*g evaluated
     at the characteristic relative speed, the same estimate VSS makes
     with pi*d^2.  taking a max is always safe: too large a vremax costs
     attempts, never bias
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
      int m = sigma_index[isp][jsp];
      if (m < 0) continue;

      // interpolate with the bin clamped into the table, so that a table
      //   which does not span the thermal speed cannot abort the setup

      double beta = MAX(vscale[isp],vscale[jsp]);
      double vrm = 2.0 * sigma_tab[m]->interpolate(beta*beta);
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
  int m = sigma_index[ispecies][jspecies];

  // no cross section table for this pair, use the analytic VSS form

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

  double vre = sigma_tab[m]->evaluate(vr2);

  vremax[icell][igroup][jgroup] = MAX(vre,vremax[icell][igroup][jgroup]);
  if (vre/vremax[icell][igroup][jgroup] < random->uniform()) return 0;

  // the TCE model derives its reaction probability as sigma_react/sigma_VHS
  // pairs were selected here with sigma_table instead, so hand the React
  //   style the ratio which restores the intended reaction rate
  // react table instead forms sigma_react/sigma_total directly

  if (react) {
    double vr = sqrt(vr2);
    sigma_total = vre/vr;
    react_prob_factor = prefactor[ispecies][jspecies] *
      pow(vr2,1.0-params[ispecies][jspecies].omega) / vre;
  }

  precoln.vr2 = vr2;
  return 1;
}

/* ----------------------------------------------------------------------
   VSS alpha for this pair, from a table when one is defined
------------------------------------------------------------------------- */

double CollideTable::scatter_alpha(int isp, int jsp)
{
  int m = alpha_index[isp][jsp];
  if (m < 0) return params[isp][jsp].alpha;
  return alpha_tab[m]->evaluate(precoln.vr2);
}

/* ----------------------------------------------------------------------
   sample cos(chi) from a tabulated differential cross section
   return 0 if this pair has no scatter table, so the VSS law is used
------------------------------------------------------------------------- */

int CollideTable::scatter_cosX(int isp, int jsp, double &cosX)
{
  int m = scatter_index[isp][jsp];
  if (m < 0) return 0;
  cosX = scatter_tab[m]->interpolate_row(precoln.vr2,random->uniform());
  if (cosX > 1.0) cosX = 1.0;
  else if (cosX < -1.0) cosX = -1.0;
  return 1;
}

/* ----------------------------------------------------------------------
   effective total cross section for species pair I,J at temperature T
   for a tabulated pair this is <sigma g>/<g>, the collision-rate weighted
     average, which reduces exactly to the VHS expression when sigma has
     the VHS form.  interpolated from a table built at init
------------------------------------------------------------------------- */

double CollideTable::sigma_eff(int isp, int jsp, double temp)
{
  int m = sigma_index[isp][jsp];
  if (m < 0 || !sigeff) return Collide::sigma_eff(isp,jsp,temp);
  if (temp <= 0.0) temp = params[isp][jsp].tref;

  double f = (log(temp) - tlo) * tinvdelta;
  if (f <= 0.0) return sigeff[m][0];
  if (f >= NTEMP-1) return sigeff[m][NTEMP-1];
  int k = (int) f;
  return sigeff[m][k] + (f-k)*(sigeff[m][k+1]-sigeff[m][k]);
}

/* ----------------------------------------------------------------------
   tabulate <sigma g>/<g> vs temperature for every cross section table
     sigma_eff(T) = (1/(kT)^2) int_0^inf sigma(E) E exp(-E/kT) dE
   computed identically on every proc from data all procs already hold
------------------------------------------------------------------------- */

void CollideTable::build_sigeff()
{
  if (nsigma == 0) return;
  if (sigeff) return;

  memory->create(sigeff,nsigma,NTEMP,"collide/table:sigeff");
  tlo = log(TEMPLO);
  tinvdelta = (NTEMP-1) / (log(TEMPHI) - tlo);

  // one reduced mass per table, from a pair which uses it

  double *mrtab = new double[nsigma];
  for (int m = 0; m < nsigma; m++) mrtab[m] = 0.0;
  for (int i = 0; i < nparams; i++)
    for (int j = 0; j < nparams; j++) {
      int m = sigma_index[i][j];
      if (m >= 0) mrtab[m] = params[i][j].mr;
    }

  // trapezoid in log E over the range which carries the Maxwellian weight

  const int NQ = 400;
  for (int m = 0; m < nsigma; m++) {
    double mr = mrtab[m];
    for (int k = 0; k < NTEMP; k++) {
      double kT = update->boltz * exp(tlo + k/tinvdelta);
      double e0 = log(1.0e-6*kT);
      double e1 = log(60.0*kT);
      double h = (e1-e0)/(NQ-1);
      double sum = 0.0;
      for (int q = 0; q < NQ; q++) {
        double E = exp(e0 + q*h);
        double vr2 = 2.0*E/mr;
        double sig = sigma_tab[m]->evaluate(vr2) / sqrt(vr2);
        double w = (q == 0 || q == NQ-1) ? 0.5 : 1.0;
        sum += w * sig * E * exp(-E/kT) * E * h;   // dE = E dlnE
      }
      sigeff[m][k] = sum / (kT*kT);
    }
  }

  delete [] mrtab;
}

/* ----------------------------------------------------------------------
   read the collide table param file
   it is a VSS param file plus directive lines:
     <species1> <species2> table|alpha|scatter <filename> <keyword>
   only invoked by proc 0
------------------------------------------------------------------------- */

void CollideTable::read_param_file(char *fname)
{
  // first pass: the VSS params
  // skip_param_line() makes the parent loop ignore the directives

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
    if (pre == (int) strlen(line) || line[pre] == '#') continue;

    int nwords = wordparse(6,line,words);
    if (!skip_param_line(nwords,words)) continue;
    if (nwords < 5)
      error->one(FLERR,"Incorrect line format in VSS parameter file");

    int kind;
    if (strcmp(words[2],"table") == 0) kind = KSIGMA;
    else if (strcmp(words[2],"alpha") == 0) kind = KALPHA;
    else kind = KSCATTER;

    int isp = particle->find_species(words[0]);
    int jsp = particle->find_species(words[1]);

    // silently ignore a directive for species not in this simulation,
    //   consistent with how the VSS param lines are handled

    if (isp < 0 || jsp < 0) continue;

    if (kind == KSIGMA)
      add_table(sigma_tab,nsigma,isp,jsp,words[3],words[4],
                TB_YSIGMA_G,1,KSIGMA);
    else if (kind == KALPHA)
      add_table(alpha_tab,nalpha,isp,jsp,words[3],words[4],
                TB_YRAW,1,KALPHA);
    else
      add_table(scatter_tab,nscatter,isp,jsp,words[3],words[4],
                TB_YRAW,0,KSCATTER);
  }

  fclose(fp);

  // a deflection table only makes sense alongside a cross section table

  for (int i = 0; i < nparams; i++)
    for (int j = 0; j < nparams; j++)
      if (sigma_index[i][j] < 0 &&
          (alpha_index[i][j] >= 0 || scatter_index[i][j] >= 0))
        error->one(FLERR,"Cross section table is required for a pair "
                   "with an alpha or scatter table");
}

/* ----------------------------------------------------------------------
   read, convert, build and report one table, and index it for the pair
------------------------------------------------------------------------- */

InterpTable *CollideTable::add_table(InterpTable **&list, int &n,
                                     int isp, int jsp, char *file,
                                     char *keyword, int ymode, int ncol,
                                     int kind)
{
  InterpTable **newlist = new InterpTable*[n+1];
  for (int m = 0; m < n; m++) newlist[m] = list[m];
  delete [] list;
  list = newlist;
  list[n] = new InterpTable(sparta);
  InterpTable *tb = list[n];

  double mr = params[isp][jsp].mr;
  double omega = params[isp][jsp].omega;
  double cxs = MY_PI*params[isp][jsp].diam*params[isp][jsp].diam;
  double vssa = cxs * pow(2.0*update->boltz*params[isp][jsp].tref/mr,
                          omega-0.5) / tgamma(2.5-omega);

  tb->read(file,keyword,ncol);

  // a scatter table must carry more than one angle per energy, else the
  //   deflection angle would be a constant

  if (kind == KSCATTER && tb->ncol < 2)
    error->one(FLERR,"A scatter table must set M > 1 on its parameter line");

  tb->convert(TB_XVR2,ymode,mr,vssa,1.0-omega);
  tb->build(tabstyle,nmant);
  double maxerr = tb->check(0);
  tb->free_input();

  const char *label = (kind == KSIGMA) ? "cross section" :
    ((kind == KALPHA) ? "alpha" : "scattering");

  char str[512];
  if (kind == KSCATTER)
    sprintf(str,"Tabulated %s %s from %s for %s %s:\n"
            "  %d energies x %d angles, E = %.4g to %.4g eV\n"
            "  %d bins, %d per factor of 2 in energy",
            label,tb->keyword,tb->file,
            particle->species[isp].id,particle->species[jsp].id,
            tb->ninput,tb->ncol,
            0.5*mr*tb->xlo/EV2J,0.5*mr*tb->xhi/EV2J,
            tb->nbins,1 << nmant);
  else
    sprintf(str,"Tabulated %s %s from %s for %s %s:\n"
            "  %d values, E = %.4g to %.4g eV, max %.4g, %s interpolation\n"
            "  %d bins, %d per factor of 2 in energy, "
            "reproduces input to %.3g%%",
            label,tb->keyword,tb->file,
            particle->species[isp].id,particle->species[jsp].id,
            tb->ninput,0.5*mr*tb->xlo/EV2J,0.5*mr*tb->xhi/EV2J,tb->ymax,
            tb->logflag ? "log-log" : "linear",
            tb->nbins,1 << nmant,100.0*maxerr);
  if (screen) fprintf(screen,"%s\n",str);
  if (logfile) fprintf(logfile,"%s\n",str);

  if (maxerr > TABLE_TOL)
    error->warning(FLERR,"Tabulated data does not reproduce its input values");

  int **index = (kind == KSIGMA) ? sigma_index :
    ((kind == KALPHA) ? alpha_index : scatter_index);
  index[isp][jsp] = index[jsp][isp] = n;
  n++;

  return tb;
}

/* ----------------------------------------------------------------------
   return 1 if this param file line is a table directive
   invoked by CollideVSS::read_param_file() so it can skip these lines
------------------------------------------------------------------------- */

int CollideTable::skip_param_line(int nwords, char **words)
{
  if (nwords < 3) return 0;
  if (strcmp(words[2],"table") == 0) return 1;
  if (strcmp(words[2],"alpha") == 0) return 1;
  if (strcmp(words[2],"scatter") == 0) return 1;
  return 0;
}
