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
#include "react_table_kokkos.h"
#include "interp_table.h"
#include "input.h"
#include "update.h"
#include "particle.h"
#include "collide.h"
#include "comm.h"
#include "error.h"

using namespace SPARTA_NS;

enum{DISSOCIATION,EXCHANGE,IONIZATION,RECOMBINATION};   // other files
enum{ARRHENIUS,QUANTUM,TABULATED};                      // other react files

#define DELTATAB 16

/* ---------------------------------------------------------------------- */

ReactTableKokkos::ReactTableKokkos(SPARTA *sparta, int narg, char **arg) :
  ReactBirdKokkos(sparta, narg, arg, 0)
{
  // the per-reaction table storage must exist before the file is read,
  //   since read_coeffs() fills it as each reaction is parsed

  kokkos_flag = 1;
  tabulated_flag = 1;
  d_warn = DAT::t_int_scalar("react/table/kk:warn");
  h_warn = Kokkos::create_mirror_view(d_warn);

  rtab = NULL;
  tabfile = tabkey = NULL;
  tabetot = NULL;
  maxtab = 0;
  warnflag = 0;

  setup_reactions(arg[1]);

  for (int m = 0; m < nlist; m++) {
    if (rlist[m].style != TABULATED) continue;
    if (!tabfile || !tabfile[m])
      error->all(FLERR,"React table/kk reaction has no cross section table");
  }
  grow_tab(nlist);

  // nlist is known now, so the parent's Kokkos allocation can be done

  init_kokkos();
}

/* ---------------------------------------------------------------------- */

ReactTableKokkos::~ReactTableKokkos()
{
  if (copy) return;

  if (rtab) {
    for (int m = 0; m < maxtab; m++) delete rtab[m];
    delete [] rtab;
  }
  if (tabfile) {
    for (int m = 0; m < maxtab; m++) {
      delete [] tabfile[m];
      delete [] tabkey[m];
    }
    delete [] tabfile;
    delete [] tabkey;
  }
  delete [] tabetot;
}

/* ---------------------------------------------------------------------- */

void ReactTableKokkos::init()
{
  if (!collide || !collide->vssflag)
    error->all(FLERR,
               "React table/kk can only be used with a VSS-based collide style");

  ReactBirdKokkos::init();

  // every active reaction must be tabulated, since this style has no other
  //   way to form a probability, and an untabulated one would have no table

  for (int m = 0; m < nlist; m++)
    if (rlist[m].active && rlist[m].style != TABULATED)
      error->all(FLERR,"React table/kk requires every reaction to use style T");

  // build each reaction's cross section table
  // proc 0 reads and builds, then broadcasts, as elsewhere in SPARTA

  Particle::Species *species = particle->species;

  for (int m = 0; m < nlist; m++) {
    OneReaction *r = &rlist[m];
    if (r->style != TABULATED) continue;
    if (!r->active) continue;
    if (rtab[m]) continue;

    int isp = r->reactants[0];
    int jsp = r->reactants[1];
    double mi = species[isp].mass;
    double mj = species[jsp].mass;
    double mr = (isp == jsp) ? mi/2.0 : mi*mj/(mi+mj);

    rtab[m] = new InterpTable(sparta);
    if (comm->me == 0) {
      rtab[m]->read(tabfile[m],tabkey[m],1);
      rtab[m]->convert(TB_XVR2,TB_YRAW,mr,0.0,0.0);
      rtab[m]->build(TB_LINEAR,10);
      rtab[m]->free_input();

      const char *units = (r->type == RECOMBINATION) ? "m^5" : "m^2";

      char str[512];
      sprintf(str,"Reaction cross section %s from %s for %s:\n"
              "  %d values, E = %.4g to %.4g eV, max %.4g %s, %d bins",
              tabkey[m],tabfile[m],r->id,rtab[m]->ninput,
              0.5*mr*rtab[m]->xlo/1.602176634e-19,
              0.5*mr*rtab[m]->xhi/1.602176634e-19,
              rtab[m]->ymax,units,rtab[m]->nbins);
      if (screen) fprintf(screen,"%s\n",str);
      if (logfile) fprintf(logfile,"%s\n",str);
    }
      rtab[m]->bcast();
  }

  // copy the built tables to the device, one entry per reaction so the
  //   device index is just the reaction index

  rtabdev.build(rtab,nlist,"react/table/kk");

  k_rtabindex = DAT::tdual_int_1d("react/table/kk:rtabindex",MAX(nlist,1));
  k_tabetot = DAT::tdual_int_1d("react/table/kk:tabetot",MAX(nlist,1));
  for (int m = 0; m < nlist; m++) {
    k_rtabindex.view_host()(m) = rtab[m] ? m : -1;
    k_tabetot.view_host()(m) = tabetot[m];
  }
  k_rtabindex.modify_host();
  k_rtabindex.sync_device();
  d_rtabindex = k_rtabindex.view_device();
  k_tabetot.modify_host();
  k_tabetot.sync_device();
  d_tabetot = k_tabetot.view_device();
}

/* ----------------------------------------------------------------------
   recognize the tabulated style letter in addition to the Bird styles
------------------------------------------------------------------------- */

int ReactTableKokkos::read_style(OneReaction *r, char *word)
{
  if (word[0] == 'T' || word[0] == 't') {
    r->style = TABULATED;
    return 1;
  }
  return ReactBird::read_style(r,word);
}

/* ----------------------------------------------------------------------
   a tabulated reaction line carries two energies then a file and keyword
------------------------------------------------------------------------- */

void ReactTableKokkos::read_coeffs(OneReaction *r, char *copy1, char *copy2)
{
  if (r->style != TABULATED) {
    ReactBird::read_coeffs(r,copy1,copy2);
    return;
  }

  // keep the Bird coeff layout so the rest of ReactBird still applies:
  //   coeff[1] = activation energy, coeff[4] = energy release

  r->ncoeff = 5;
  for (int i = 0; i < 5; i++) r->coeff[i] = 0.0;

  char *w1 = strtok(NULL," \t\n\r");
  char *w2 = strtok(NULL," \t\n\r");
  char *w3 = strtok(NULL," \t\n\r");
  char *w4 = strtok(NULL," \t\n\r");
  char *w5 = strtok(NULL," \t\n\r");
  if (!w1 || !w2 || !w3 || !w4) {
    print_reaction(copy1,copy2);
    error->all(FLERR,"Invalid reaction coefficients in file");
  }

  r->coeff[1] = input->numeric(FLERR,w1);
  r->coeff[4] = input->numeric(FLERR,w2);

  int m = (int) (r - rlist);
  grow_tab(m+1);
  tabfile[m] = new char[strlen(w3)+1];
  strcpy(tabfile[m],w3);
  tabkey[m] = new char[strlen(w4)+1];
  strcpy(tabkey[m],w4);

  tabetot[m] = 0;
  if (w5) {
    if (strcmp(w5,"etotal") == 0) tabetot[m] = 1;
    else if (strcmp(w5,"etrans") != 0) {
      print_reaction(copy1,copy2);
      error->all(FLERR,"Invalid energy variable for a tabulated reaction");
    }
  }
}

/* ---------------------------------------------------------------------- */

void ReactTableKokkos::grow_tab(int n)
{
  if (n <= maxtab) return;
  int old = maxtab;
  while (maxtab < n) maxtab += DELTATAB;

  InterpTable **nt = new InterpTable*[maxtab];
  char **nf = new char*[maxtab];
  char **nk = new char*[maxtab];
  int *ne = new int[maxtab];
  for (int m = 0; m < old; m++) {
    nt[m] = rtab[m]; nf[m] = tabfile[m]; nk[m] = tabkey[m]; ne[m] = tabetot[m];
  }
  for (int m = old; m < maxtab; m++) {
    nt[m] = NULL; nf[m] = NULL; nk[m] = NULL; ne[m] = 0;
  }
  delete [] rtab;
  delete [] tabfile;
  delete [] tabkey;
  delete [] tabetot;
  rtab = nt; tabfile = nf; tabkey = nk; tabetot = ne;
}

/* ----------------------------------------------------------------------
   raise what the device recorded during the collision kernel: the EXTRAP
     error policy, and the same one-time warning ReactTable::attempt()
     issues on the host
------------------------------------------------------------------------- */

void ReactTableKokkos::check_flags()
{
  if (rtabdev.check_error())
    error->one(FLERR,"Value is outside the tabulated data range");

  // the same two warnings ReactTable::attempt() issues, each once.  the
  //   host issues only the first one it meets, since it warns inline and
  //   then stops testing; here the kernel has already run, so both causes
  //   may be recorded and both are worth reporting

  Kokkos::deep_copy(h_warn,d_warn);
  const int warn = h_warn();

  if ((warn & WARN_RBOOST) && !(warnflag & WARN_RBOOST)) {
    warnflag |= WARN_RBOOST;
    error->warning(FLERR,"Boosted recombination probability exceeds 1, "
                   "recombination rate will be underpredicted; reduce "
                   "the react_modify rboost factor");
  }

  if ((warn & WARN_ENVELOPE) && !(warnflag & WARN_ENVELOPE)) {
    warnflag |= WARN_ENVELOPE;
    error->warning(FLERR,"Reaction cross section exceeds the total cross "
                   "section, reaction rate will be underpredicted; give "
                   "that pair a collide table cross section which "
                   "envelopes the reactive one");
  }
}
