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
#include "react_table.h"
#include "interp_table.h"
#include "input.h"
#include "update.h"
#include "particle.h"
#include "collide.h"
#include "comm.h"
#include "random_knuth.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

enum{DISSOCIATION,EXCHANGE,IONIZATION,RECOMBINATION};   // other files
enum{ARRHENIUS,QUANTUM,TABULATED};                      // other react files

#define DELTATAB 16

/* ---------------------------------------------------------------------- */

ReactTable::ReactTable(SPARTA *sparta, int narg, char **arg) :
  ReactBird(sparta, narg, arg, 0)
{
  // the per-reaction table storage must exist before the file is read,
  //   since read_coeffs() fills it as each reaction is parsed

  rtab = NULL;
  tabfile = tabkey = NULL;
  tabetot = NULL;
  maxtab = 0;
  warnflag = 0;

  setup_reactions(arg[1]);

  for (int m = 0; m < nlist; m++) {
    if (rlist[m].style != TABULATED) continue;
    if (!tabfile || !tabfile[m])
      error->all(FLERR,"React table reaction has no cross section table");
  }
  grow_tab(nlist);
}

/* ---------------------------------------------------------------------- */

ReactTable::~ReactTable()
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

void ReactTable::init()
{
  if (!collide || !collide->vssflag)
    error->all(FLERR,"React table can only be used with a VSS-based collide style");

  ReactBird::init();

  // every active reaction must be tabulated, since this style has no other
  //   way to form a probability, and an untabulated one would have no table

  for (int m = 0; m < nlist; m++)
    if (rlist[m].active && rlist[m].style != TABULATED)
      error->all(FLERR,"React table requires every reaction to use style T");

  // build each reaction's cross section table
  // proc 0 reads and builds, then broadcasts, as elsewhere in SPARTA
  // x is the relative translational energy, converted to vr^2 with the
  //   reduced mass of the reactant pair

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

      // a recombination table is a cross section per unit third-body
      //   number density, so it carries an extra volume

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
}

/* ----------------------------------------------------------------------
   attempt a reaction for this collision
   probability of each reaction is sigma_react(E)/sigma_total, where
     sigma_total is the cross section the collide style used to select
     this pair, so the realized rate is n <sigma_react g> as intended
------------------------------------------------------------------------- */

int ReactTable::attempt(Particle::OnePart *ip, Particle::OnePart *jp,
                        double pre_etrans, double pre_erot, double pre_evib,
                        double &post_etotal, int &kspecies)
{
  int isp = ip->ispecies;
  int jsp = jp->ispecies;

  int n = reactions[isp][jsp].n;
  if (n == 0) return 0;
  int *list = reactions[isp][jsp].list;

  double sigma_t = collide->sigma_total;
  if (sigma_t <= 0.0) return 0;

  double pre_etotal = pre_etrans + pre_erot + pre_evib;

  // the tables are indexed by relative translational energy, which is what
  //   measured and computed reactive cross sections are tabulated against

  Particle::Species *species = particle->species;
  double mi = species[isp].mass;
  double mj = species[jsp].mass;
  double mr = (isp == jsp) ? mi/2.0 : mi*mj/(mi+mj);

  double react_prob = 0.0;
  double random_prob = random->uniform();

  for (int i = 0; i < n; i++) {
    OneReaction *r = &rlist[list[i]];

    // ignore energetically impossible reactions
    // an endothermic reaction whose energy deficit exceeds the activation
    //   energy must be gated on the deficit, else post_etotal below goes
    //   negative and the post-collision velocities become NaN
    // same test as ReactTCE::attempt()

    double e_excess;
    if (r->coeff[1] > -r->coeff[4]) e_excess = pre_etotal - r->coeff[1];
    else e_excess = pre_etotal + r->coeff[4];
    if (e_excess <= 0.0) continue;

    // index the table by the energy this reaction was tabulated against

    double ereact = tabetot[list[i]] ? pre_etotal : pre_etrans;
    double sigma_r = rtab[list[i]]->evaluate(2.0*ereact/mr);

    if (r->type == RECOMBINATION) {

      // Collide::collisions() picked a 3rd particle for this collision, or
      //   -1 if it skipped the selection.  the boost factor makes that
      //   selection rare and the probability is scaled back up by it, so
      //   the statistics are unchanged; same scheme as ReactTCE
      // the 3rd particle only selects which recombination reaction applies,
      //   so skip this one unless it is the reaction that species maps to

      if (recomb_species < 0) continue;
      int *sp2recomb = reactions[isp][jsp].sp2recomb;
      if (sp2recomb[recomb_species] != list[i]) continue;

      // sigma_r is a cross section per unit third-body number density, in
      //   m^5, so that n3*sigma_r is an area and <sigma_r g> is the usual
      //   three-body rate coefficient in m^6/s

      react_prob += recomb_boost * recomb_density * sigma_r / sigma_t;

    } else react_prob += sigma_r / sigma_t;

    // sigma_react exceeding sigma_total means the collide style's total
    //   cross section does not envelope the reactive one, which clips the
    //   rate.  warn once rather than on every collision
    // for recombination the boost factor multiplies the probability, and
    //   its default of 1000 is large enough that it is the usual cause,
    //   so point at it rather than at the cross sections

    if (react_prob > 1.0 && !warnflag) {
      warnflag = 1;
      if (r->type == RECOMBINATION)
        error->warning(FLERR,"Boosted recombination probability exceeds 1, "
                       "recombination rate will be underpredicted; reduce "
                       "the react_modify rboost factor");
      else
        error->warning(FLERR,"Reaction cross section exceeds the total cross "
                       "section, reaction rate will be underpredicted; give "
                       "that pair a collide table cross section which "
                       "envelopes the reactive one");
    }

    if (react_prob > random_prob) {
      tally_reactions[list[i]]++;

      // compute_chem_rates only accumulates the tally, it does not perform
      //   the reaction, so no products or energies are set and 0 is returned

      if (computeChemRates) continue;

      ip->ispecies = r->products[0];

      switch (r->type) {
      case DISSOCIATION:
      case IONIZATION:
      case EXCHANGE:
        {
          jp->ispecies = r->products[1];
          break;
        }
      case RECOMBINATION:
        {
          // the 2nd reactant is always consumed

          jp->ispecies = -1;
          break;
        }
      }

      if (r->nproduct > 2) kspecies = r->products[2];
      else kspecies = -1;

      post_etotal = pre_etotal + r->coeff[4];

      // return reaction from 1 to N

      return list[i] + 1;
    }
  }

  return 0;
}

/* ----------------------------------------------------------------------
   recognize the tabulated style letter in addition to the Bird styles
------------------------------------------------------------------------- */

int ReactTable::read_style(OneReaction *r, char *word)
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

void ReactTable::read_coeffs(OneReaction *r, char *copy1, char *copy2)
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

void ReactTable::grow_tab(int n)
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
