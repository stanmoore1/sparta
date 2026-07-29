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
#include "stdlib.h"
#include "string.h"
#include "compute_vdf_grid.h"
#include "particle.h"
#include "mixture.h"
#include "grid.h"
#include "update.h"
#include "input.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

// user keywords

enum{SPEED,VX,VY,VZ,KE,EROT,EVIB};

// out-of-range handling

enum{IGNORE,CLAMP};

/* ---------------------------------------------------------------------- */

ComputeVDFGrid::ComputeVDFGrid(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 8) error->all(FLERR,"Illegal compute vdf/grid command");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute vdf/grid group ID does not exist");
  groupbit = grid->bitmask[igroup];

  imix = particle->find_mixture(arg[3]);
  if (imix < 0)
    error->all(FLERR,"Compute vdf/grid mixture ID does not exist");
  ngroup = particle->mixture[imix]->ngroup;

  // each value is 4 args: name Nbin lo hi
  // upper bound on # of values, actual count set while parsing

  int maxvalue = (narg-4) / 4;
  if (maxvalue == 0) error->all(FLERR,"Illegal compute vdf/grid command");

  value = new int[maxvalue];
  nbin = new int[maxvalue];
  lo = new double[maxvalue];
  hi = new double[maxvalue];
  invdelta = new double[maxvalue];
  binoffset = new int[maxvalue];

  // process input values

  nvalue = 0;
  int iarg = 4;
  while (iarg < narg) {
    int ivalue;
    if (strcmp(arg[iarg],"speed") == 0) ivalue = SPEED;
    else if (strcmp(arg[iarg],"vx") == 0) ivalue = VX;
    else if (strcmp(arg[iarg],"vy") == 0) ivalue = VY;
    else if (strcmp(arg[iarg],"vz") == 0) ivalue = VZ;
    else if (strcmp(arg[iarg],"ke") == 0) ivalue = KE;
    else if (strcmp(arg[iarg],"erot") == 0) ivalue = EROT;
    else if (strcmp(arg[iarg],"evib") == 0) ivalue = EVIB;
    else break;

    if (iarg+4 > narg) error->all(FLERR,"Illegal compute vdf/grid command");

    value[nvalue] = ivalue;
    nbin[nvalue] = input->inumeric(FLERR,arg[iarg+1]);
    lo[nvalue] = input->numeric(FLERR,arg[iarg+2]);
    hi[nvalue] = input->numeric(FLERR,arg[iarg+3]);

    if (nbin[nvalue] <= 0)
      error->all(FLERR,"Compute vdf/grid Nbin must be > 0");
    if (lo[nvalue] >= hi[nvalue])
      error->all(FLERR,"Compute vdf/grid bin range must have lo < hi");

    invdelta[nvalue] = nbin[nvalue] / (hi[nvalue] - lo[nvalue]);
    nvalue++;
    iarg += 4;
  }

  if (nvalue == 0) error->all(FLERR,"Illegal compute vdf/grid command");

  // process optional keywords

  oobstyle = IGNORE;
  weightflag = 0;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"oob") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Invalid compute vdf/grid optional keyword");
      if (strcmp(arg[iarg+1],"ignore") == 0) oobstyle = IGNORE;
      else if (strcmp(arg[iarg+1],"clamp") == 0) oobstyle = CLAMP;
      else error->all(FLERR,"Invalid compute vdf/grid optional keyword");
      iarg += 2;
    } else if (strcmp(arg[iarg],"weight") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Invalid compute vdf/grid optional keyword");
      if (strcmp(arg[iarg+1],"no") == 0) weightflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) weightflag = 1;
      else error->all(FLERR,"Invalid compute vdf/grid optional keyword");
      iarg += 2;
    } else error->all(FLERR,"Invalid compute vdf/grid value or optional keyword");
  }

  // column layout within one mixture group:
  //   all bins of value 1, then all bins of value 2, etc
  // full column layout is group major: group 1 block, group 2 block, etc

  nbintotal = 0;
  for (int m = 0; m < nvalue; m++) {
    binoffset[m] = nbintotal;
    nbintotal += nbin[m];
  }
  ntotal = ngroup*nbintotal;

  // needmass = 1 if any value needs the species mass

  needmass = 0;
  for (int m = 0; m < nvalue; m++)
    if (value[m] == KE) needmass = 1;

  per_grid_flag = 1;
  size_per_grid_cols = ntotal;

  nglocal = 0;
  array_grid = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeVDFGrid::~ComputeVDFGrid()
{
  if (copymode) return;

  delete [] value;
  delete [] nbin;
  delete [] lo;
  delete [] hi;
  delete [] invdelta;
  delete [] binoffset;

  memory->destroy(array_grid);
}

/* ---------------------------------------------------------------------- */

void ComputeVDFGrid::init()
{
  if (ngroup != particle->mixture[imix]->ngroup)
    error->all(FLERR,"Number of groups in compute vdf/grid "
               "mixture has changed");

  // the non-Kokkos version reads the host copy of the particle list, which
  //   Kokkos only syncs before output, not before Modify::end_of_step(), so
  //   invoking it from e.g. fix ave/grid would silently histogram stale data
  // ComputeVDFGridKokkos sets kokkos_flag and does its own device tally

  if (sparta->kokkos && !kokkos_flag)
    error->all(FLERR,"Must use compute vdf/grid/kk if Kokkos is enabled");

  // only consult the per-particle weight if cell weighting is enabled,
  //   else it is not maintained and every sample counts 1.0

  cellweightflag = grid->cellweightflag ? 1 : 0;

  reallocate();
}

/* ----------------------------------------------------------------------
   bin all particles in all owned grid cells in the grid group
   histograms are raw sample counts, which is a linear function of the
     tallies, so fix ave/grid can time average the output directly
------------------------------------------------------------------------- */

void ComputeVDFGrid::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;

  Grid::ChildInfo *cinfo = grid->cinfo;
  Particle::Species *species = particle->species;
  Particle::OnePart *particles = particle->particles;
  int *s2g = particle->mixture[imix]->species2group;
  int nlocal = particle->nlocal;

  double mvv2e = update->mvv2e;

  int i,m,ibin,ispecies,igroup,icell,kbase;
  double mass,sample,wt;
  double *v,*vec;

  int useweight = weightflag && cellweightflag;

  // zero all accumulators

  for (i = 0; i < nglocal; i++) {
    vec = array_grid[i];
    for (m = 0; m < ntotal; m++) vec[m] = 0.0;
  }

  // loop over all particles, skip species not in mixture group

  for (i = 0; i < nlocal; i++) {
    ispecies = particles[i].ispecies;
    igroup = s2g[ispecies];
    if (igroup < 0) continue;
    icell = particles[i].icell;
    if (!(cinfo[icell].mask & groupbit)) continue;

    v = particles[i].v;
    if (needmass) mass = species[ispecies].mass;
    else mass = 0.0;

    // use the cell weight, not Particle::OnePart::weight, which is only
    //   scratch state maintained by Particle::pre_weight() during a move

    wt = useweight ? cinfo[icell].weight : 1.0;

    vec = array_grid[icell];
    kbase = igroup*nbintotal;

    for (m = 0; m < nvalue; m++) {
      sample = 0.0;
      switch (value[m]) {
      case SPEED:
        sample = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        break;
      case VX:
        sample = v[0];
        break;
      case VY:
        sample = v[1];
        break;
      case VZ:
        sample = v[2];
        break;
      case KE:
        sample = 0.5*mvv2e*mass * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        break;
      case EROT:
        sample = particles[i].erot;
        break;
      case EVIB:
        sample = particles[i].evib;
        break;
      }

      // test the sample against lo/hi directly rather than testing the bin
      //   index, because a cast to int truncates toward zero, so a sample
      //   just below lo would otherwise be indistinguishable from bin 0
      // a sample exactly at hi, or one pushed past the last bin by roundoff,
      //   is folded into the last bin

      if (sample < lo[m] || sample > hi[m]) {
        if (oobstyle == IGNORE) continue;
        ibin = (sample < lo[m]) ? 0 : nbin[m]-1;
      } else {
        ibin = static_cast<int> ((sample - lo[m]) * invdelta[m]);
        if (ibin >= nbin[m]) ibin = nbin[m] - 1;
      }

      vec[kbase + binoffset[m] + ibin] += wt;
    }
  }
}

/* ----------------------------------------------------------------------
   reallocate array if nglocal has changed
   called by init() and whenever grid changes
------------------------------------------------------------------------- */

void ComputeVDFGrid::reallocate()
{
  if (grid->nlocal == nglocal) return;

  memory->destroy(array_grid);
  nglocal = grid->nlocal;
  memory->create(array_grid,nglocal,ntotal,"vdf/grid:array_grid");
}

/* ----------------------------------------------------------------------
   memory usage of local grid-based data
------------------------------------------------------------------------- */

bigint ComputeVDFGrid::memory_usage()
{
  bigint bytes = 0;
  bytes += (bigint) ntotal*nglocal * sizeof(double);   // array_grid
  return bytes;
}
