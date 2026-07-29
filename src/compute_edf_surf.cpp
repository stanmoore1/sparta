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
#include "compute_edf_surf.h"
#include "particle.h"
#include "mixture.h"
#include "surf.h"
#include "grid.h"
#include "update.h"
#include "domain.h"
#include "input.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

// which distribution this instance computes

enum{ENERGY,ANGLE};

// which particles to bin

enum{INCIDENT,REFLECTED};

// which energy to bin, for ENERGY only

enum{KE,EROT,EVIB,ETOT};

// out-of-range handling

enum{IGNORE,CLAMP};

#define DELTA 4096

/* ---------------------------------------------------------------------- */

ComputeEDFSurf::ComputeEDFSurf(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  // style name selects the binned quantity

  if (strcmp(style,"adf/surf") == 0) distflag = ANGLE;
  else distflag = ENERGY;

  int iarg;

  if (distflag == ENERGY) {
    if (narg < 7) error->all(FLERR,"Illegal compute edf/surf command");
  } else {
    if (narg < 5) error->all(FLERR,"Illegal compute adf/surf command");
  }

  int igroup = surf->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute edf/surf group ID does not exist");
  groupbit = surf->bitmask[igroup];

  imix = particle->find_mixture(arg[3]);
  if (imix < 0) error->all(FLERR,"Compute edf/surf mixture ID does not exist");
  ngroup = particle->mixture[imix]->ngroup;

  // bin count and range
  // ANGLE range is always 0 to 90 degrees from the surface normal

  nbin = input->inumeric(FLERR,arg[4]);
  if (nbin <= 0) error->all(FLERR,"Compute edf/surf Nbin must be > 0");

  if (distflag == ENERGY) {
    lo = input->numeric(FLERR,arg[5]);
    hi = input->numeric(FLERR,arg[6]);
    if (lo >= hi)
      error->all(FLERR,"Compute edf/surf bin range must have lo < hi");
    iarg = 7;
  } else {
    lo = 0.0;
    hi = 90.0;
    iarg = 5;
  }

  invdelta = nbin / (hi - lo);

  // process optional keywords

  dirstyle = INCIDENT;
  engstyle = KE;
  oobstyle = IGNORE;
  weightflag = 0;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"dir") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Invalid compute edf/surf optional keyword");
      if (strcmp(arg[iarg+1],"incident") == 0) dirstyle = INCIDENT;
      else if (strcmp(arg[iarg+1],"reflected") == 0) dirstyle = REFLECTED;
      else error->all(FLERR,"Invalid compute edf/surf optional keyword");
      iarg += 2;
    } else if (strcmp(arg[iarg],"value") == 0) {
      if (distflag != ENERGY)
        error->all(FLERR,"Compute adf/surf does not support the value keyword");
      if (iarg+2 > narg)
        error->all(FLERR,"Invalid compute edf/surf optional keyword");
      if (strcmp(arg[iarg+1],"ke") == 0) engstyle = KE;
      else if (strcmp(arg[iarg+1],"erot") == 0) engstyle = EROT;
      else if (strcmp(arg[iarg+1],"evib") == 0) engstyle = EVIB;
      else if (strcmp(arg[iarg+1],"etot") == 0) engstyle = ETOT;
      else error->all(FLERR,"Invalid compute edf/surf optional keyword");
      iarg += 2;
    } else if (strcmp(arg[iarg],"oob") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Invalid compute edf/surf optional keyword");
      if (strcmp(arg[iarg+1],"ignore") == 0) oobstyle = IGNORE;
      else if (strcmp(arg[iarg+1],"clamp") == 0) oobstyle = CLAMP;
      else error->all(FLERR,"Invalid compute edf/surf optional keyword");
      iarg += 2;
    } else if (strcmp(arg[iarg],"weight") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Invalid compute edf/surf optional keyword");
      if (strcmp(arg[iarg+1],"no") == 0) weightflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) weightflag = 1;
      else error->all(FLERR,"Invalid compute edf/surf optional keyword");
      iarg += 2;
    } else error->all(FLERR,"Invalid compute edf/surf optional keyword");
  }

  // setup
  // column layout is group major: all bins of group 1, then group 2, etc

  ntotal = ngroup*nbin;

  per_surf_flag = 1;
  size_per_surf_cols = ntotal;

  surf_tally_flag = 1;
  timeflag = 1;

  ntally = maxtally = 0;
  array_surf_tally = NULL;
  tally2surf = NULL;

  maxsurf = 0;
  array_surf = NULL;
  combined = 0;

  hash = new MyHash;

  dim = domain->dimension;
}

/* ---------------------------------------------------------------------- */

ComputeEDFSurf::~ComputeEDFSurf()
{
  if (copy || copymode) return;

  memory->destroy(array_surf_tally);
  memory->destroy(tally2surf);
  memory->destroy(array_surf);
  delete hash;
}

/* ---------------------------------------------------------------------- */

void ComputeEDFSurf::init()
{
  if (!surf->exist)
    error->all(FLERR,"Cannot use compute edf/surf when surfs do not exist");
  if (surf->implicit)
    error->all(FLERR,"Cannot use compute edf/surf with implicit surfs");

  if (ngroup != particle->mixture[imix]->ngroup)
    error->all(FLERR,"Number of groups in compute edf/surf mixture has changed");

  // UpdateKokkos only drives surf tallying through ComputeSurfKokkos, so say so
  //   here rather than let the generic cast failure in UpdateKokkos report it.
  //   Remove once a Kokkos port exists.

  if (sparta->kokkos) {
    char str[128];
    sprintf(str,"Cannot (yet) use compute %s with the KOKKOS package",style);
    error->all(FLERR,str);
  }

  // only consult the per-particle weight if cell weighting is enabled,
  //   else it is not maintained and every sample counts 1.0

  cellweightflag = grid->cellweightflag ? 1 : 0;

  // initialize tally array in case accessed before a tally timestep

  clear();

  combined = 0;
}

/* ----------------------------------------------------------------------
   no operations here, since compute results are stored in tally array
   just used by callers to indicate compute was used
   enables prediction of next step when update needs to tally
------------------------------------------------------------------------- */

void ComputeEDFSurf::compute_per_surf()
{
  invoked_per_surf = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void ComputeEDFSurf::clear()
{
  lines = surf->lines;
  tris = surf->tris;

  // clear hash of tallied surf IDs
  // called by Update at beginning of timesteps surf tallying is done

  hash->clear();
  ntally = 0;
  combined = 0;
}

/* ----------------------------------------------------------------------
   histogram one collision with surface element isurf
   iorig = particle ip before collision
   ip,jp = particles after collision
   ip = NULL means no particles after collision
   jp = NULL means one particle after collision
   jp != NULL means two particles after collision
   tallies are raw sample counts, which is a linear function of the
     tallies, so fix ave/surf can time average the output directly
------------------------------------------------------------------------- */

void ComputeEDFSurf::surf_tally(double /*dtremain*/, int isurf, int /*icell*/,
                                int reaction, Particle::OnePart *iorig,
                                Particle::OnePart *ip, Particle::OnePart *jp)
{
  // skip if no original particle and a reaction is taking place
  //   called by SurfReactAdsorb for on-surf reaction
  // FixEmitSurf also calls with no original particle but no reaction

  if (!iorig && reaction) return;

  // skip if isurf not in surface group

  int transparent;
  surfint surfID;
  double *norm;

  if (dim == 2) {
    if (!(lines[isurf].mask & groupbit)) return;
    surfID = lines[isurf].id;
    transparent = lines[isurf].transparent;
    norm = lines[isurf].norm;
  } else {
    if (!(tris[isurf].mask & groupbit)) return;
    surfID = tris[isurf].id;
    transparent = tris[isurf].transparent;
    norm = tris[isurf].norm;
  }

  // build list of particles to bin
  // incident = the pre-collision particle
  // reflected = the post-collision particle(s), which for a reaction are
  //   the product species, so each is binned in its own mixture group
  // a transparent surf does not alter the particle, so there is nothing
  //   reflected from it

  Particle::OnePart *plist[2];
  int np = 0;

  if (dirstyle == INCIDENT) {
    if (!iorig) return;
    plist[np++] = iorig;
  } else {
    if (transparent) return;
    if (ip) plist[np++] = ip;
    if (jp) plist[np++] = jp;
  }
  if (np == 0) return;

  // mixture group of each particle, skip particles not in the mixture
  // return before touching the hash if nothing will be tallied

  int *s2g = particle->mixture[imix]->species2group;
  int glist[2];
  int nkeep = 0;

  for (int i = 0; i < np; i++) {
    int igroup = s2g[plist[i]->ispecies];
    if (igroup < 0) continue;
    plist[nkeep] = plist[i];
    glist[nkeep] = igroup;
    nkeep++;
  }
  if (nkeep == 0) return;

  // itally = tally index of isurf
  // if 1st particle hitting isurf, add surf ID to hash
  // grow tally list if needed

  int itally;
  double *vec;

  if (hash->find(surfID) != hash->end()) itally = (*hash)[surfID];
  else {
    if (ntally == maxtally) grow_tally();
    itally = ntally;
    (*hash)[surfID] = itally;
    tally2surf[itally] = surfID;
    vec = array_surf_tally[itally];
    for (int i = 0; i < ntotal; i++) vec[i] = 0.0;
    ntally++;
  }

  vec = array_surf_tally[itally];

  double mvv2e = update->mvv2e;
  Particle::Species *species = particle->species;
  int useweight = weightflag && cellweightflag;

  for (int i = 0; i < nkeep; i++) {
    Particle::OnePart *p = plist[i];
    double *v = p->v;
    double sample;
    double wt = useweight ? p->weight : 1.0;

    if (distflag == ENERGY) {
      double ke;
      switch (engstyle) {
      case KE:
        sample = 0.5*mvv2e * species[p->ispecies].mass * MathExtra::lensq3(v);
        break;
      case EROT:
        sample = p->erot;
        break;
      case EVIB:
        sample = p->evib;
        break;
      case ETOT:
        ke = 0.5*mvv2e * species[p->ispecies].mass * MathExtra::lensq3(v);
        sample = ke + p->erot + p->evib;
        break;
      default:
        sample = 0.0;
        break;
      }

    } else {

      // polar angle from the surface normal, in degrees
      // norm is a unit vector and points outward from the surface,
      //   so an incident particle has v dot norm < 0
      // fabs() makes the angle range 0 to 90 for incident and reflected alike
      // a motionless particle has no direction, so skip it

      double vmag = sqrt(MathExtra::lensq3(v));
      if (vmag == 0.0) continue;
      double cosang = fabs(MathExtra::dot3(v,norm)) / vmag;
      if (cosang > 1.0) cosang = 1.0;
      sample = acos(cosang) * 180.0/MY_PI;
    }

    // test the sample against lo/hi directly rather than testing the bin
    //   index, because a cast to int truncates toward zero, so a sample just
    //   below lo would otherwise be indistinguishable from bin 0
    // a sample exactly at hi, or one pushed past the last bin by roundoff,
    //   is folded into the last bin

    int ibin;
    if (sample < lo || sample > hi) {
      if (oobstyle == IGNORE) continue;
      ibin = (sample < lo) ? 0 : nbin-1;
    } else {
      ibin = static_cast<int> ((sample - lo) * invdelta);
      if (ibin >= nbin) ibin = nbin - 1;
    }

    vec[glist[i]*nbin + ibin] += wt;
  }
}

/* ----------------------------------------------------------------------
   return # of tallies and their indices into my local surf list
------------------------------------------------------------------------- */

int ComputeEDFSurf::tallyinfo(surfint *&ptr)
{
  ptr = tally2surf;
  return ntally;
}

/* ----------------------------------------------------------------------
   sum tally values to owning surfs via surf->collate()
------------------------------------------------------------------------- */

void ComputeEDFSurf::post_process_surf()
{
  if (combined) return;
  combined = 1;

  // reallocate array_surf if necessary

  int nown = surf->nown;

  if (nown > maxsurf) {
    memory->destroy(array_surf);
    maxsurf = nown;
    memory->create(array_surf,maxsurf,ntotal,"edf/surf:array_surf");
  }

  // collate entire array of results

  surf->collate_array(ntally,ntotal,tally2surf,array_surf_tally,array_surf);
}

/* ---------------------------------------------------------------------- */

void ComputeEDFSurf::grow_tally()
{
  maxtally += DELTA;
  memory->grow(tally2surf,maxtally,"edf/surf:tally2surf");
  memory->grow(array_surf_tally,maxtally,ntotal,"edf/surf:array_surf_tally");
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

bigint ComputeEDFSurf::memory_usage()
{
  bigint bytes = 0;
  bytes += (bigint) ntotal*maxtally * sizeof(double);   // array_surf_tally
  bytes += (bigint) maxtally * sizeof(surfint);         // tally2surf
  bytes += (bigint) ntotal*maxsurf * sizeof(double);    // array_surf
  return bytes;
}
