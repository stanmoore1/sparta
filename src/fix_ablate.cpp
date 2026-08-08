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

#include "spatype.h"
#include "stdlib.h"
#include "string.h"
#include "fix_ablate.h"
#include "update.h"
#include "geometry.h"
#include "math_extra.h"
#include "math_const.h"
#include "grid.h"
#include "surf.h"
#include "surf_collide.h"
#include "particle.h"
#include "domain.h"
#include "decrement_lookup_table.h"
#include "comm.h"
#include "modify.h"
#include "compute.h"
#include "fix.h"
#include "output.h"
#include "input.h"
#include "variable.h"
#include "dump.h"
#include "marching_squares.h"
#include "marching_cubes.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

enum{COMPUTE,FIX,VARIABLE,RANDOM,UNIFORM,FLUX};
enum{GRIDVAR,EQUALVAR};
enum{CVALUE,CDELTA,NVERT};
enum{ABLATE,DEPOSIT,BOTH};     // surface recedes (ablate) or grows (deposit)
enum{CORNER,DISTANCE};    // source units: corner point value or length/time
enum{RNORMAL,RVOLUME};    // how a speed becomes a corner point increment
enum{UNKNOWN,OUTSIDE,INSIDE,OVERLAP};   // cell types, same as Grid
enum{KEEP,DISCARD,MIGRATE};   // fate of a particle the new isosurface encloses
enum{PERIODIC,OUTFLOW,REFLECT,SURFACE,AXISYM};   // same as Domain

#define SWEEP_FRAC 0.5    // max front advance per regeneration, in grid cells
#define EPSSURF 1.0e-4    // push off a surf, same as Grid::point_outside_surfs()
#define CLAMP_FRAC 1.10   // fraction of crossed edges that may lose the
                          //   normal projection before it is worth warning
#define BIGDIST 1.0e20   // no cell offered a distance for this corner point
#define NOMEASURE (-1.0)  // edge_displacement() had no crossing to measure from

#define INVOKED_PER_GRID 16
#define DELTAGRID 1024            // must be bigger than split cells per cell
#define DELTASEND 1024
#define EPSILON 1.0e-4            // this is on a scale of 0 to 255

enum{XLO,XHI,YLO,YHI,ZLO,ZHI,INTERIOR};         // same as Domain
enum{NCHILD,NPARENT,NUNKNOWN,NPBCHILD,NPBPARENT,NPBUNKNOWN,NBOUND};  // Update

// remove if fix particles-inside-surfs issue
enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};  // several files

// NOTES
// should I store one value or 8 per cell
// how to preserve svalues and set type of new surfs,
//   maybe need local per-cell type
// after create new surfs, need to assign group, sc, sr
// how to impose sgroup like ReadIsurf does after surfs created
// how to prevent adaptation of any cells in the implicit grid group(s)?
//   just testing for surfs is not enough, since it may have no surfs
// need to update neigh corner points even if neigh cell is not in group?
// do a run-time test for gridcut = 0.0 ?
// worry about array_grid having updated values for sub cells?  line in store()
// can I output both per-grid and global values?

/* ---------------------------------------------------------------------- */

FixAblate::FixAblate(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  MPI_Comm_rank(world,&me);

  if (narg < 6) error->all(FLERR,"Illegal fix ablate command");

  igroup = grid->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Could not find fix ablate group ID");
  groupbit = grid->bitmask[igroup];

  nevery = atoi(arg[3]);
  if (nevery < 0) error->all(FLERR,"Illegal fix ablate command");

  idsource = NULL;
  argindex = 0;         // only the c_/f_ source styles override this

  scale = atof(arg[4]);
  if (scale < 0.0) error->all(FLERR,"Illegal fix ablate command");

  int iarg = 6;

  if ((strncmp(arg[5],"c_",2) == 0) || (strncmp(arg[5],"f_",2) == 0)) {
    if (arg[5][0] == 'c') which = COMPUTE;
    else if (arg[5][0] == 'f') which = FIX;

    int n = strlen(arg[5]);
    char *suffix = new char[n];
    strcpy(suffix,&arg[5][2]);

    char *ptr = strchr(suffix,'[');
    if (ptr) {
      if (suffix[strlen(suffix)-1] != ']')
        error->all(FLERR,"Illegal fix ablate command");
      argindex = atoi(ptr+1);
      *ptr = '\0';
    } else argindex = 0;

    n = strlen(suffix) + 1;
    idsource = new char[n];
    strcpy(idsource,suffix);
    delete [] suffix;

  } else if (strncmp(arg[5],"v_",2) == 0) {
    which = VARIABLE;

    int n = strlen(arg[5]) - 1;
    idsource = new char[n];
    strcpy(idsource,&arg[5][2]);

  } else if (strcmp(arg[5],"flux") == 0) {

    // flux c_ID|f_ID density RHO sticking s1 s2 ...
    //   the source gives the incident mass FLOW onto each cell, one column
    //   per mixture group, and the film grows at
    //     s = sum_g( sticking_g * flow_g ) / (rho * A_cell)
    //   which is a speed, so this source implies units = distance

    iarg++;
    if (narg < 7) error->all(FLERR,"Illegal fix ablate command");
    which = FLUX;

    if (strncmp(arg[6],"c_",2) == 0) fluxwhich = COMPUTE;
    else if (strncmp(arg[6],"f_",2) == 0) fluxwhich = FIX;
    else error->all(FLERR,"Fix ablate flux source must be a compute or fix");

    int n = strlen(arg[6]);
    idsource = new char[n];
    strcpy(idsource,&arg[6][2]);

  } else if (strcmp(arg[5],"random") == 0) {
    iarg++;

    if (narg < 7) error->all(FLERR,"Illegal fix ablate command");
    which = RANDOM;
    maxrandom = atoi(arg[6]);

  } else if (strcmp(arg[5],"uniform") == 0) {
    iarg++; // one additional input
    if (narg < 7) error->all(FLERR,"Illegal fix ablate command");
    which = UNIFORM;
    maxrandom = atoi(arg[6]);

  } else error->all(FLERR,"Illegal fix ablate command");

  // process optional command line args

  process_args(narg-iarg,&arg[iarg]);

  // error check

  if (which == COMPUTE) {
    icompute = modify->find_compute(idsource);
    if (icompute < 0)
      error->all(FLERR,"Compute ID for fix ablate does not exist");
    if (modify->compute[icompute]->per_grid_flag == 0)
      error->all(FLERR,
                 "Fix ablate compute does not calculate per-grid values");
    if (modify->compute[icompute]->post_process_isurf_grid_flag == 0)
      error->all(FLERR,
                 "Fix ablate compute does not calculate isurf per-grid values");
    if (argindex == 0 &&
        modify->compute[icompute]->size_per_grid_cols != 0)
      error->all(FLERR,"Fix ablate compute does not "
                 "calculate per-grid vector");
    if (argindex && modify->compute[icompute]->size_per_grid_cols == 0)
      error->all(FLERR,"Fix ablate compute does not "
                 "calculate per-grid array");
    if (argindex && argindex > modify->compute[icompute]->size_per_grid_cols)
      error->all(FLERR,"Fix ablate compute array is accessed out-of-range");

  } else if (which == FIX) {
    ifix = modify->find_fix(idsource);
    if (ifix < 0)
      error->all(FLERR,"Fix ID for fix ablate does not exist");
    if (modify->fix[ifix]->per_grid_flag == 0)
      error->all(FLERR,"Fix ablate fix does not calculate per-grid values");
    if (argindex == 0 && modify->fix[ifix]->size_per_grid_cols != 0)
      error->all(FLERR,
                 "Fix ablate fix does not calculate per-grid vector");
    if (argindex && modify->fix[ifix]->size_per_grid_cols == 0)
      error->all(FLERR,
                 "Fix ablate fix does not calculate per-grid array");
    if (argindex && argindex > modify->fix[ifix]->size_per_grid_cols)
      error->all(FLERR,"Fix ablate fix array is accessed out-of-range");
    if (nevery % modify->fix[ifix]->per_grid_freq)
      error->all(FLERR,
                 "Fix for fix ablate not computed at compatible time");

  } else if (which == FLUX) {
    if (filmrho <= 0.0)
      error->all(FLERR,"Fix ablate flux source requires the density keyword");
    if (nsticking == 0)
      error->all(FLERR,"Fix ablate flux source requires the sticking keyword");

    // a flux source states a growth speed, so units is not the user's to
    //   choose here

    unitsflag = DISTANCE;

    int ncol;
    if (fluxwhich == COMPUTE) {
      icompute = modify->find_compute(idsource);
      if (icompute < 0)
        error->all(FLERR,"Compute ID for fix ablate does not exist");
      if (modify->compute[icompute]->per_grid_flag == 0)
        error->all(FLERR,"Fix ablate compute does not calculate per-grid values");
      if (modify->compute[icompute]->post_process_isurf_grid_flag == 0)
        error->all(FLERR,"Fix ablate compute does not calculate isurf "
                   "per-grid values");
      ncol = modify->compute[icompute]->size_per_grid_cols;
    } else {
      ifix = modify->find_fix(idsource);
      if (ifix < 0) error->all(FLERR,"Fix ID for fix ablate does not exist");
      if (modify->fix[ifix]->per_grid_flag == 0)
        error->all(FLERR,"Fix ablate fix does not calculate per-grid values");
      ncol = modify->fix[ifix]->size_per_grid_cols;
    }

    if (ncol == 0) ncol = 1;
    if (nsticking != ncol) {
      char str[128];
      sprintf(str,"Fix ablate needs %d sticking coefficients, one per column "
              "of its flux source, but %d were given",ncol,nsticking);
      error->all(FLERR,str);
    }

  } else if (which == VARIABLE) {
    ivariable = input->variable->find(idsource);
    if (ivariable < 0)
      error->all(FLERR,"Could not find fix ablate variable name");

    // an equal-style variable is a single number for the whole surface,
    //   which is what a rate that depends only on time looks like.  It is
    //   applied to every cell in the group; the deposit interface gate keeps
    //   a uniform scalar from sprouting material in open gas

    if (input->variable->grid_style(ivariable)) varstyle = GRIDVAR;
    else if (input->variable->equal_style(ivariable)) varstyle = EQUALVAR;
    else error->all(FLERR,
                    "Fix ablate variable is not grid-style or equal-style");
  }

  // this fix produces a per-grid array and a scalar

  dim = domain->dimension;

  per_grid_flag = 1;
  if (dim == 2) size_per_grid_cols = 4;
  else size_per_grid_cols = 8;
  per_grid_freq = 1;
  gridmigrate = 1;

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 22;   // 0-1 ablation, 2-19 deposition, 20 front speed,
                      //   21 applied corner value change
  global_freq = 1;
  sum_delta = 0.0;
  sum_applied = 0.0;
  ndelete = 0;

  storeflag = multi_val_flag = 0;
  isc_default = isr_default = 0;
  array_grid = cvalues = NULL;
  cvalues_prev = NULL;
  sfront_cell = sfront_normal = NULL;
  cnow = cnext = NULL;
  segpt = segnorm = NULL;
  segspeed = segband = NULL;
  nseg = NULL;
  maxsegcell = segstride = 0;
  depo_stamp = -1;
  front_last = 0.0;
  clampwarn = 0;
  nrebuild = 0;
  smoothed = 0;
  clampsum = 0.0;
  ntotedge = 0;
  for (int i = 0; i < NREDUCE; i++) depo_all[i] = 0.0;
  dlist = NULL;
  maxdlist = 0;
  mvalues = NULL;
  tvalues = NULL;
  ncorner = size_per_grid_cols;

  if(dim == 2) nmultiv = 4;
  else nmultiv = 6;

  // local storage

  ixyz = NULL;
  mcflags = NULL;
  celldelta = NULL;
  cdelta = NULL;
  cdelta_ghost = NULL;
  mdelta = NULL;
  mdelta_ghost = NULL;
  cflag = NULL;
  cflag_ghost = NULL;
  nvert = NULL;
  nvert_ghost = NULL;

  numsend = NULL;
  maxgrid = maxghost = 0;

  proclist = NULL;
  locallist = NULL;
  maxsend = 0;

  sbuf = NULL;
  maxbuf = 0;

  vbuf = NULL;
  maxvar = 0;

  ms = NULL;
  mc = NULL;

  // RNG for random decrements
  // for now, use same RNG on every proc
  // uncomment two lines if want to change that
  // b/c set_delta_random() is decrementing the same no matter who owns a cell

  random = NULL;
  if (which == RANDOM) {
    random = new RanKnuth(update->ranmaster->uniform());
    //double seed = update->ranmaster->uniform();
    //random->reset(seed,comm->me,100);
  }

  // nvalid = next step on which end_of_step does something
  // add nvalid to all computes that store invocation times
  // since don't know a priori which are invoked by this fix
  // once in end_of_step() can set timestep for ones actually invoked

  if (nevery) {
    bigint nvalid = (update->ntimestep/nevery)*nevery + nevery;
    modify->addstep_compute_all(nvalid);
  }
}

/* ---------------------------------------------------------------------- */

FixAblate::~FixAblate()
{
  delete [] idsource;
  memory->destroy(cvalues);
  memory->destroy(cvalues_prev);
  memory->destroy(sfront_cell);
  memory->destroy(sfront_normal);
  memory->destroy(cnow);
  memory->destroy(cnext);
  memory->destroy(segpt);
  memory->destroy(segnorm);
  memory->destroy(segspeed);
  memory->destroy(segband);
  memory->destroy(nseg);
  memory->destroy(dlist);
  memory->destroy(sticking);

  memory->destroy(mvalues);
  memory->destroy(tvalues);

  memory->destroy(ixyz);
  memory->destroy(mcflags);

  memory->destroy(celldelta);
  memory->destroy(cdelta);
  memory->destroy(cdelta_ghost);
  memory->destroy(mdelta);
  memory->destroy(mdelta_ghost);

  memory->destroy(cflag);
  memory->destroy(cflag_ghost);
  memory->destroy(nvert);
  memory->destroy(nvert_ghost);

  memory->destroy(numsend);

  memory->destroy(proclist);
  memory->destroy(locallist);

  memory->destroy(sbuf);
  memory->destroy(vbuf);

  delete ms;
  delete mc;

  delete random;
}

/* ---------------------------------------------------------------------- */

int FixAblate::setmask()
{
  int mask = 0;
  if (nevery) mask |= END_OF_STEP;
  if (nevery && depositflag) mask |= START_OF_STEP;
  return mask;
}

/* ----------------------------------------------------------------------
   store grid corner point and type values in cvalues and tvalues
   then create implicit surfaces
   called by ReadIsurf when corner point grid is read in
------------------------------------------------------------------------- */

void FixAblate::store_corners(int nx_caller, int ny_caller, int nz_caller,
                              double *cornerlo_caller, double *xyzsize_caller,
                              double **cvalues_caller, double ***mvalues_caller,
                              int *tvalues_caller,
                              double thresh_caller, char *sgroupID, int pushflag,
                              double smoothband)
{
  storeflag = 1;
  if(mvalues_caller) {
    multi_val_flag = 1;
    cvalues = NULL; // likely not needed
  } else {
    multi_val_flag = 0;
    mvalues = NULL; // likely not needed
  }

  // a multivalue fix stores corner state in mvalues and leaves array_grid
  //   (= cvalues) NULL, so it cannot expose the per-grid corner array that
  //   per_grid_flag advertises.  Turn the flag off so generic per-grid
  //   consumers (compute reduce, dump grid, fix ave/grid, ...) reject it
  //   cleanly at setup instead of dereferencing NULL array_grid at run time.
  //   The scalar output (f_ID) is unaffected.

  if (multi_val_flag) per_grid_flag = 0;
  else per_grid_flag = 1;

  nx = nx_caller;
  ny = ny_caller;
  nz = nz_caller;
  cornerlo[0] = cornerlo_caller[0];
  cornerlo[1] = cornerlo_caller[1];
  cornerlo[2] = cornerlo_caller[2];
  xyzsize[0] = xyzsize_caller[0];
  xyzsize[1] = xyzsize_caller[1];
  xyzsize[2] = xyzsize_caller[2];
  thresh = thresh_caller;

  tvalues_flag = 0;
  if (tvalues_caller) tvalues_flag = 1;

  if (sgroupID) {
    int sgroup = surf->find_group(sgroupID);
    if (sgroup < 0) sgroup = surf->add_group(sgroupID);
    sgroupbit = surf->bitmask[sgroup];
  } else sgroupbit = 0;

  // allocate per-grid cell data storage

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  nglocal = grid->nlocal;

  grow_percell(0);

  // copy caller values into local values of FixAblate

  for (int icell = 0; icell < nglocal; icell++) {
    for (int m = 0; m < ncorner; m++) {
      if(!multi_val_flag) cvalues[icell][m] = cvalues_caller[icell][m];
      else {
        for (int n = 0; n < nmultiv; n++)
          mvalues[icell][m][n] = mvalues_caller[icell][m][n];
      }
    }
    if (tvalues_flag) tvalues[icell] = tvalues_caller[icell];
  }

  // set all values to either min or max value

  if (minmaxflag) {
    for (int icell = 0; icell < nglocal; icell++) {
      for (int m = 0; m < ncorner; m++) {
        if (!multi_val_flag) {
          if (cvalues[icell][m] < thresh) cvalues[icell][m] = 0.0;
          else cvalues[icell][m] = 255.0;
        } else {
          for (int n = 0; n < nmultiv; n++) {
            if (mvalues[icell][m][n] < thresh) mvalues[icell][m][n] = 0.0;
            else mvalues[icell][m][n] = 255.0;
          }
        }
      }
    }
  }

  // set ix,iy,iz indices from 1 to Nxyz for each of my owned grid cells
  // same logic as ReadIsurf::create_hash()

  for (int i = 0; i < nglocal; i++)
    ixyz[i][0] = ixyz[i][1] = ixyz[i][2] = 0;

  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;

    ixyz[icell][0] =
      static_cast<int> ((cells[icell].lo[0]-cornerlo[0]) / xyzsize[0] + 0.5) + 1;
    ixyz[icell][1] =
      static_cast<int> ((cells[icell].lo[1]-cornerlo[1]) / xyzsize[1] + 0.5) + 1;
    ixyz[icell][2] =
      static_cast<int> ((cells[icell].lo[2]-cornerlo[2]) / xyzsize[2] + 0.5) + 1;
  }

  // push corner pt values with fully external/internal neighbors to 0 or 255
  // adjust individual corner point values too close to threshold

  if (pushflag && !multi_val_flag) push_lohi();

  // check for consistency

  if (multi_val_flag) epsilon_adjust_multiv();
  else epsilon_adjust();

  // create marching squares/cubes classes, now that have group & threshold

  if (dim == 2) ms = new MarchingSquares(sparta,igroup,thresh);
  else mc = new MarchingCubes(sparta,igroup,thresh);

  // set minimum distance between vertex and grid point (mindist) in marching
  if (dim == 2) ms->mindist = mindist;
  else mc->mindist = mindist;

  // optionally replace a binary field with a graded one, before anything is
  //   built from it, so the surface and the cell marking are derived once
  //   from the field the run will actually use

  if (smoothband > 0.0) {
    if (multi_val_flag)
      error->all(FLERR,"Read_isurf smooth does not support multivalue "
                 "corner points");
    distance_transform(smoothband);
  }
  smoothed = (smoothband > 0.0);

  // for deposition, seed the previous-field snapshot with the field as it
  //   stands now, so that the front has not moved until the first increment
  // without this the snapshot is read before it is ever written, and the
  //   refreshed collision geometry of the first interval is derived from
  //   whatever the allocation happened to contain

  if (depositflag)
    for (int icell = 0; icell < nglocal; icell++)
      for (int m = 0; m < ncorner; m++)
        cvalues_prev[icell][m] = cvalues[icell][m];

  // create implicit surfaces

  create_surfs(1);
}

/* ---------------------------------------------------------------------- */

void FixAblate::init()
{
  if (!storeflag)
    error->all(FLERR,"Fix ablate corner point values not stored");

  // deposition prototype currently supports only the base single-value,
  //   single-decrement path (not the multivalue / multiple-decrement modes)

  if (depositflag && (multi_val_flag || multi_dec_flag))
    error->all(FLERR,"Fix ablate mode deposit does not yet support "
               "multivalue or multiple corner-point decrement modes");

  // mode = both reads the source as a SIGNED rate, and a budget of corner
  //   point value has no sign to read: units = corner places its budget one
  //   corner at a time, which is a rule for one direction only

  if (mode == BOTH && unitsflag != DISTANCE)
    error->all(FLERR,"Fix ablate mode both requires units distance");

  // response says how a SPEED becomes a corner point increment, and units
  //   corner never states a speed, so asking for one there is a mistake
  //   rather than a setting with no effect

  if (responseflag != RNORMAL && unitsflag != DISTANCE)
    error->all(FLERR,"Fix ablate response is only meaningful with "
               "units distance, where the source states a speed");

  // the KOKKOS particle move is a separate implementation and does not read
  //   the refreshed collision geometry, so growth is resolved only when the
  //   isosurface is rebuilt.  That is still accounted for -- the salvage and
  //   burial path runs there too -- but it is coarser, so say so rather than
  //   let a Kokkos run quietly differ from the same input without Kokkos

  if (depositflag && sparta->kokkos && me == 0)
    error->warning(FLERR,"Fix ablate mode deposit: the KOKKOS particle move "
                   "does not use the per-step refreshed surface, so surface "
                   "growth is resolved only every Nevery steps; use a smaller "
                   "Nevery, or run without the KOKKOS package");

  // deposition is handled entirely in this fix, when the isosurface is
  //   regenerated, so it needs no support in the move loop and works the
  //   same in 2d, 3d, axisymmetric and Kokkos runs

  if (which == COMPUTE) {
    icompute = modify->find_compute(idsource);
    if (icompute < 0)
      error->all(FLERR,"Compute ID for fix ablate does not exist");
  } else if (which == FIX) {
    ifix = modify->find_fix(idsource);
    if (ifix < 0)
      error->all(FLERR,"Fix ID for fix ablate does not exist");
  } else if (which == FLUX) {
    if (fluxwhich == COMPUTE) {
      icompute = modify->find_compute(idsource);
      if (icompute < 0)
        error->all(FLERR,"Compute ID for fix ablate does not exist");
    } else {
      ifix = modify->find_fix(idsource);
      if (ifix < 0)
        error->all(FLERR,"Fix ID for fix ablate does not exist");
    }
  } else if (which == VARIABLE) {
    ivariable = input->variable->find(idsource);
    if (ivariable < 0)
      error->all(FLERR,"Variable ID for fix ablate does not exist");
  }

  // reallocate per-grid data if necessary

  nglocal = grid->nlocal;
  grow_percell(0);

  // determine default collision/reaction model indices from existing surfaces
  // these values were set by surf_modify and are used during each ablation step
  //   to correctly re-assign models to newly created implicit surfaces
  // implicit surfs are distributed, so allreduce the values to procs
  //   which own no surfs now, since ablation may create surfs in their cells
  // error if surfs are not all assigned to the same models, e.g. by
  //   surf_modify with multiple surf groups, since create_surfs() can only
  //   re-assign one collision/reaction model to all new surfs

  int isc_local = -1;
  int isr_local = -1;
  int flag = 0;

  int nslocal = surf->nlocal;

  if (nslocal > 0) {
    if (dim == 2) {
      Surf::Line *lines = surf->lines;
      isc_local = lines[0].isc;
      isr_local = lines[0].isr;
      for (int i = 1; i < nslocal; i++)
        if (lines[i].isc != isc_local || lines[i].isr != isr_local) flag = 1;
    } else {
      Surf::Tri *tris = surf->tris;
      isc_local = tris[0].isc;
      isr_local = tris[0].isr;
      for (int i = 1; i < nslocal; i++)
        if (tris[i].isc != isc_local || tris[i].isr != isr_local) flag = 1;
    }
  }

  int local_vals[2],global_vals[2];
  local_vals[0] = isc_local;
  local_vals[1] = isr_local;
  MPI_Allreduce(local_vals,global_vals,2,MPI_INT,MPI_MAX,world);
  isc_default = global_vals[0];
  isr_default = global_vals[1];

  if (nslocal > 0 && (isc_local != isc_default || isr_local != isr_default))
    flag = 1;

  int allflag;
  MPI_Allreduce(&flag,&allflag,1,MPI_INT,MPI_MAX,world);
  if (allflag)
    error->all(FLERR,"Fix ablate requires all surfs be assigned "
               "to the same surface collision and reaction models");

  if (isc_default < 0) isc_default = 0;
}

/* ---------------------------------------------------------------------- */

void FixAblate::end_of_step()
{
  // set per-cell delta vector randomly or from compute/fix source

  if (which == RANDOM) set_delta_random();
  else if (which == UNIFORM) set_delta_uniform();
  else set_delta();

  // if the source is in length/time, convert it to a corner value delta
  //   here, using dc = s * |grad c| * interval
  //   sync() gives each corner point the summed contribution of the cells
  //   sharing it, which for a locally uniform field is celldelta itself
  // this applies to a receding surface as much as a growing one, so it is
  //   deliberately outside the DEPOSIT block below
  // set_delta() leaves Nevery out of its prefactor for this units setting,
  //   since the elapsed interval is what is being applied right here

  if (unitsflag == DISTANCE) {
    Grid::ChildCell *cells = grid->cells;
    Grid::ChildInfo *cinfo = grid->cinfo;
    double interval = nevery * update->dt;
    clampsum = 0.0;
    ntotedge = 0;
    for (int icell = 0; icell < nglocal; icell++) {
      if (!(cinfo[icell].mask & groupbit)) continue;
      if (cells[icell].nsplit <= 0) continue;

      // celldelta is a speed; turn the distance it asks for into the corner
      //   point increment that actually delivers it

      // with mode = both each cell picks its own direction, and the two
      //   differ: which end of a crossed edge is pinned at 0 or 255 is not
      //   the same going one way as the other

      int grow = (mode == DEPOSIT) ||
                 (mode == BOTH && celldelta[icell] > 0.0);

      if (responseflag == RVOLUME) {

        // the surface has to sweep a volume of speed x area x time.  Ask the
        //   cell's solid fraction for the corner point shift that delivers
        //   exactly that, which needs no surface normal: the area the volume
        //   is spread over is measured off the surface the cell holds rather
        //   than inferred from a gradient, so a front oblique to the grid is
        //   no harder than one aligned with it.
        // areas here are Cartesian even in an axisymmetric run, since what
        //   comes back out is a displacement, which is a length either way

        double area = cell_area_cart(icell);
        double cellvol = xyzsize[0]*xyzsize[1];
        if (dim == 3) cellvol *= xyzsize[2];
        if (area <= 0.0 || cellvol <= 0.0) {
          celldelta[icell] = 0.0;
          continue;
        }
        double dvfrac = fabs(celldelta[icell]) * interval * area / cellvol;
        double shift = volume_shift(cvalues[icell],dvfrac,grow);

        // ABLATE subtracts what it is given, DEPOSIT adds it, and BOTH reads
        //   the sign, so only BOTH wants a negative number here

        celldelta[icell] = (mode == BOTH && !grow) ? -shift : shift;

      } else {
        double response = front_response(icell,grow);
        if (response > 0.0) celldelta[icell] *= interval / response;
        else celldelta[icell] = 0.0;
      }
    }

    // the warning is about the projection onto the surface normal, which the
    //   volume response does not use

    if (responseflag == RNORMAL) check_oblique();
  }

  // for DEPOSIT, snapshot the corner values so the realized front motion
  //   over this interval can be measured in front_speed() below

  if (depositflag) {
    for (int icell = 0; icell < nglocal; icell++)
      for (int i = 0; i < ncorner; i++)
        cvalues_prev[icell][i] = cvalues[icell][i];
  }

  // snapshot the corner point value total so the change actually applied can
  //   be reported next to the requested budget in sum_delta.  The two differ
  //   by whatever could not be placed: corner points already saturated at 0
  //   or 255 absorb nothing, and for deposition saturation is the normal
  //   endpoint of a filling cell rather than a corner case

  double sum_before = corner_sum_local();

  // various decrement and sync routines depending on:
  // 1) are multivalues used?
  // 2) is the decrement distributed to multiple corner points?

  if (multi_dec_flag) {
    if (multi_val_flag) {
      decrement_multiv_multid_outside();
      sync_multiv_multid_outside();
      decrement_multiv_multid_inside();
      sync_multiv_multid_inside();
    } else {
      decrement_multid_outside();
      sync_multid_outside();
      decrement_multid_inside();
      sync_multid_inside();
    }
  } else {
    if (multi_val_flag) {
      decrement_multiv();
      sync_multiv();
    } else if (depositflag) {
      increment();
      sync();
    } else {
      decrement();
      sync();
    }
  }

  // sync shared corner point values
  // adjust individual corner point values too close to threshold

  if (multi_val_flag) epsilon_adjust_multiv();
  else epsilon_adjust();

  // signed change the field actually took over this interval: positive for
  //   deposition, negative for ablation, either for mode both

  double applied = corner_sum_local() - sum_before;
  MPI_Allreduce(&applied,&sum_applied,1,MPI_DOUBLE,MPI_SUM,world);

  // measure the normal speed of the advancing front over this interval
  // must be done before create_surfs(), which needs it to set per-surf speeds

  if (depositflag) {
    front_speed();
    check_group_boundary();
  }
  // re-create implicit surfs

  create_surfs(0);

  // error if the front would outrun its own collision list in one step
  // csurfs lists each surf only in the cells it currently overlaps, so a
  //   front advancing a full cell per step reaches particles that are never
  //   tested against it, and they would be overtaken silently

  if (depositflag) {
    double small = MIN(xyzsize[0],xyzsize[1]);
    if (dim == 3) small = MIN(small,xyzsize[2]);
    double max_advance = 0.0;
    for (int icell = 0; icell < nglocal; icell++)
      if (sfront_cell[icell]/small > max_advance)
        max_advance = sfront_cell[icell]/small;
    double allmax;
    MPI_Allreduce(&max_advance,&allmax,1,MPI_DOUBLE,MPI_MAX,world);
    if (allmax > SWEEP_FRAC) {
      char str[256];
      sprintf(str,"Fix ablate deposition front advances %g of a grid cell "
              "between isosurface regenerations (max %g); reduce Nevery or "
              "the deposition rate",allmax,SWEEP_FRAC);
      error->all(FLERR,str);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixAblate::create_surfs(int outflag)
{
  // DEBUG
  // store copy of last ablation's per-cell MC flags before a new ablation

  // mcflags is already grown to maxgrid by grow_percell(), so reset it in
  //   place.  it used to be reallocated here and the previous one kept as
  //   mcflags_old, an allocate and a free of 4 ints per grid cell on every
  //   rebuild, where mcflags_old was only ever read by a debug print that
  //   is commented out

  for (int i = 0; i < maxgrid; i++)
    mcflags[i][0] = mcflags[i][1] = mcflags[i][2] = mcflags[i][3] = -1;

  // sort existing particles since may be clearing split cells

  if (!particle->sorted) particle->sort();

  // reassign particles in sub cells to all be in parent split cell

  if (grid->nsplitlocal) {
    Grid::ChildCell *cells = grid->cells;
    for (int icell = 0; icell < nglocal; icell++)
      if (cells[icell].nsplit > 1)
        grid->combine_split_cell_particles(icell,1);
  }

  // call clear_surf before create new surfs, so cell/corner flags are all set

  grid->unset_neighbors();
  grid->remove_ghosts();
  grid->clear_surf();
  surf->clear_implicit();

  // perform Marching Squares/Cubes to create new implicit surfs
  // cvalues = corner point values
  // tvalues = surf type for surfs in each grid cell

  if (dim == 2) ms->invoke(cvalues,mvalues,tvalues);
  else mc->invoke(cvalues,mvalues,tvalues,mcflags);

  // set surf->nsurf and surf->nown

  surf->nown = surf->nlocal;
  bigint nlocal = surf->nlocal;
  MPI_Allreduce(&nlocal,&surf->nsurf,1,MPI_SPARTA_BIGINT,MPI_SUM,world);

  // output extent of implicit surfs, some may be tiny

  if (outflag) {
    if (dim == 2) surf->output_extent(0);
    else surf->output_extent(0);
  }

  // compute normals of new surfs

  if (dim == 2) surf->compute_line_normal(0);
  else surf->compute_tri_normal(0);

  // MC->cleanup() checks for consistent triangles on grid cell faces
  // needs to come after normals are computed
  // it requires neighbor indices and ghost cell info
  // so first acquire ghosts (which will also grab surfs),
  //   then remove ghost surfs and ghost grid cells again

  if (dim == 3) {
    grid->acquire_ghosts(0);
    grid->reset_neighbors();
    mc->cleanup();
    surf->remove_ghosts();
    grid->unset_neighbors();
    grid->remove_ghosts();
  }

  // assign optional surf group to masks of new surfs

  if (sgroupbit) {
    int nsurf = surf->nlocal;
    if (dim == 3) {
      Surf::Tri *tris = surf->tris;
      for (int i = 0; i < nsurf; i++) tris[i].mask |= sgroupbit;
    } else {
      Surf::Line *lines = surf->lines;
      for (int i = 0; i < nsurf; i++) lines[i].mask |= sgroupbit;
    }
  }

  // assign surf collision/reaction models to newly created surfs
  // this assignment can be made in input script via surf_modify
  //   after implicit surfs are created
  // for active ablation, must be re-assigned at every ablation step
  // use isc_default/isr_default which were set during init() by reading
  //   the values surf_modify assigned to existing surfaces

  int nslocal = surf->nlocal;

  if (dim == 2) {
    Surf::Line *lines = surf->lines;
    if (surf->nsc)
      for (int i = 0; i < nslocal; i++)
        lines[i].isc = isc_default;
    if (surf->nsr && isr_default >= 0)
      for (int i = 0; i < nslocal; i++)
        lines[i].isr = isr_default;
  } else {
    Surf::Tri *tris = surf->tris;
    if (surf->nsc)
      for (int i = 0; i < nslocal; i++)
        tris[i].isc = isc_default;
    if (surf->nsr && isr_default >= 0)
      for (int i = 0; i < nslocal; i++)
        tris[i].isr = isr_default;
  }

  // watertight check can be done before surfs are mapped to grid cells

  // it is a check on Marching Squares/Cubes rather than part of building the
  //   surface, and it is not free: on a 48^3 grid it is a tenth of a rebuild,
  //   paid again every Nevery steps for the length of the run.  checkevery
  //   lets a run that has been validated stop paying for it, and gates
  //   Grid::type_check() below on the same schedule

  int docheck = checkevery > 0 && (nrebuild % checkevery) == 0;
  nrebuild++;

  if (docheck) {
    if (dim == 2) surf->check_watertight_2d();
    else surf->check_watertight_3d();
  }

  // if no surfs created, use clear_surf to set all celltypes = OUTSIDE

  if (surf->nsurf == 0) {
    surf->exist = 0;
    grid->clear_surf();
  }

  // -----------------------
  // map surfs to grid cells
  // -----------------------

  // surfs are already assigned to grid cells
  // create split cells due to new surfs

  grid->surf2grid_implicit(1,outflag);

  // re-setup grid ghosts and neighbors

  grid->setup_owned();
  grid->acquire_ghosts();
  grid->reset_neighbors();
  comm->reset_neighbors();

  // flag cells and corners as OUTSIDE or INSIDE
  // Grid::set_inout() discovers this by flood filling the whole grid outward
  //   from the cells that hold surface, iterating with irregular comm until
  //   no proc has anything left to hand across a proc boundary.  For an
  //   implicit surface the answer is already in the corner point field --
  //   a cell is inside where its corner values exceed the threshold -- so
  //   the fill is a rediscovery of what this fix already knows.  Do it
  //   directly instead, and fall back to the sweep when the field cannot
  //   answer (a fix group smaller than the whole grid, or multi-value
  //   corner points)
  // only deposition takes the direct path: it is what pays the per-interval
  //   rebuild cost the path was written for, while ablation runs keep the
  //   fill their reference logs were made with

  if (!depositflag || !set_inout_implicit()) grid->set_inout();

  // type_check() is validation of the marking above, not part of it, and it
  //   costs two MPI_Allreduce plus a sweep of the grid.  it runs on the same
  //   schedule as the watertight check

  if (outflag || docheck) grid->type_check(outflag);

  // reassign particles in a split cell to sub cell owner
  // particles are unsorted afterwards, within new sub cells

  if (grid->nsplitlocal) {
    Grid::ChildCell *cells = grid->cells;
    for (int icell = 0; icell < nglocal; icell++)
      if (cells[icell].nsplit > 1)
        grid->assign_split_cell_particles(icell);
    particle->sorted = 0;
  }

  // map the per-cell front speed onto the new surfs
  // must follow surf2grid_implicit(), which builds the csurfs lists

  // record the pose the refreshed surface advances away from, and drop any
  //   stale refresh so the move loop uses the surface just committed

  if (depositflag) {
    update->front_step0 = update->ntimestep;
    update->nseg = NULL;
    update->nsegcell = 0;
  }

  // notify all classes that store per-grid data that grid has changed

  grid->notify_changed();

  // ------------------------------------------------------------------------
  // DEBUG - should not have to do any of this once marching cubes is perfect
  // only necessary for 3d

  // for ablation the surface recedes, so no particle can be engulfed and
  //   the 2d path skips the inside-surf particle deletion below
  // for deposition the surface grows into the gas, so particles can be
  //   engulfed in 2d as well as 3d and must be removed

  if (dim == 2 && !depositflag) {
    return;
  }

  // DEBUG - if this line is uncommented, code will do delete no particles
  //         eventually this should work

  // if (dim == 3) {
  //   return;
  // }

  // DEBUG - remove all particles
  // if these lines are uncommented, all particles are wiped out

  // particle->nlocal = 0;
  // return;

  // DEBUG - remove only the particles that are inside the surfs
  //         after ablation
  // similar code as in fix grid/check

  Particle::OnePart *particles = particle->particles;
  int pnlocal = particle->nlocal;

  int ncount = 0;
  int nlist = 0;

  // dlist collects every particle leaving this proc's list: the ones buried,
  //   and for deposition the ones pushed across a face into a cell owned by
  //   another proc.  Built in ascending order, as the compaction requires.

  grow_dlist(pnlocal);

  for (int i = 0; i < pnlocal; i++) {
    particles[i].flag = PKEEP;
    int status = resolve_engulfed(i);
    particles = particle->particles;
    if (status == KEEP) continue;
    if (status == MIGRATE) {
      dlist[nlist++] = i;
      continue;
    }
    particles[i].flag = PDISCARD;
    dlist[nlist++] = i;
    ncount++;
  }

  if (!depositflag) {

    // compress out the deleted particles
    // NOTE: if end up keeping this section, need logic for custom particle vectors
    //       see Particle::compress_rebalance()

    int nbytes = sizeof(Particle::OnePart);

    int i = 0;
    while (i < pnlocal) {
      if (particles[i].flag == PDISCARD) {
        memcpy(&particles[i],&particles[pnlocal-1],nbytes);
        pnlocal--;
      } else i++;
    }
    particle->nlocal = pnlocal;

  } else {

    // one call both sends the molecules pushed across a proc boundary and
    //   compresses out the buried ones, exactly as the move loop uses it
    // it also carries any custom particle attributes, which the hand-rolled
    //   compaction above does not

    int nstart = comm->migrate_particles(nlist,dlist);
    particles = particle->particles;

    // now adjudicate what arrived.  The sending proc could not tell whether
    //   its neighbor's cell held gas at that point, so it pushed the molecule
    //   over optimistically; this is the owner deciding.

    nlist = 0;
    grow_dlist(particle->nlocal - nstart);

    Grid::ChildCell *cells = grid->cells;

    for (int i = nstart; i < particle->nlocal; i++) {
      particles[i].flag = PKEEP;

      // a split cell owns no particles, its sub cells do.  The sender could
      //   not resolve that: the ghost cell it saw carried no split info.

      int icell = particles[i].icell;
      if (cells[icell].nsplit > 1) {
        if (dim == 2) particles[i].icell = update->split2d(icell,particles[i].x);
        else particles[i].icell = update->split3d(icell,particles[i].x);
      }

      int status = resolve_arrived(i);
      particles = particle->particles;
      if (status == KEEP) continue;
      particles[i].flag = PDISCARD;
      dlist[nlist++] = i;
      ncount++;
    }

    if (nlist) particle->compress_migrate(nlist,dlist);
  }

  MPI_Allreduce(&ncount,&ndelete,1,MPI_INT,MPI_SUM,world);

  particle->sorted = 0;
}

/* ----------------------------------------------------------------------
   mark every owned cell OUTSIDE or INSIDE straight from the corner point
     field, in place of Grid::set_inout()'s flood fill
   the field is the definition of the surface, so it already carries the
     answer: a corner point above the threshold is inside the material, and
     a cell whose corner points do not straddle the threshold holds no piece
     of the surface and is entirely on one side of it
   this is exact, not an approximation of the fill, and it is local: no
     iteration and no communication, where the fill needs an MPI_Allreduce
     per pass plus an irregular exchange to carry markings across proc
     boundaries
   return 0 if the field cannot answer, so the caller falls back to the fill:
     - a fix group smaller than the whole grid leaves cells with no corner
       point values at all
     - multi-value corner points describe several materials per cell
     - a cell with straddling corner points that marching squares/cubes did
       not make OVERLAP would mean the two disagree, which is a bug rather
       than a case to paper over, but falling back is the safe response
   only called in deposit mode: ablation runs always use the fill
------------------------------------------------------------------------- */

int FixAblate::set_inout_implicit()
{
  if (multi_val_flag || !cvalues) return 0;

  // group 0 is "all", and only then is every cell's corner point value mine
  //   to read.  ReadISurf allows its own group to be a subset of the fix's,
  //   filling the rest with empty space, so the fix's group is exactly the
  //   set of cells the field describes

  if (igroup != 0) return 0;

  // transparent surfs are not a boundary between inside and outside, and
  //   Grid::set_inout() marks the whole grid OUTSIDE when they are all
  //   transparent.  the field knows nothing of that, so leave it to the fill

  if (surf->all_transparent()) return 0;

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  Grid::SplitInfo *sinfo = grid->sinfo;
  int nlocal = grid->nlocal;

  // pass 1: decide, and check the field against marching squares/cubes
  // nothing is written until the check passes on every proc, so a fallback
  //   hands Grid::set_inout() the same state it would have seen

  int disagree = 0;

  for (int icell = 0; icell < nlocal; icell++) {
    if (cells[icell].nsplit <= 0) continue;
    if (cinfo[icell].type != UNKNOWN) continue;
    double *c = cvalues[icell];
    int nin = 0;
    for (int i = 0; i < ncorner; i++)
      if (c[i] > thresh) nin++;
    if (nin && nin < ncorner) disagree++;
  }

  int disagree_any;
  MPI_Allreduce(&disagree,&disagree_any,1,MPI_INT,MPI_SUM,world);
  if (disagree_any) return 0;

  // pass 2: mark
  // a cell marching squares/cubes left alone takes its type from the field
  // an OVERLAP cell keeps whatever Cut2d/Cut3d worked out, except when the
  //   cut left its corner flags UNKNOWN, which happens when the surface only
  //   touches the cell at a point or an edge; then the field supplies them,
  //   per corner rather than all-or-nothing as the fill would

  int dimension = domain->dimension;

  for (int icell = 0; icell < nlocal; icell++) {
    if (cells[icell].nsplit <= 0) continue;
    double *c = cvalues[icell];

    if (cinfo[icell].type == UNKNOWN) {
      cinfo[icell].type = c[0] > thresh ? INSIDE : OUTSIDE;
      continue;
    }

    if (cinfo[icell].type != OVERLAP) continue;
    if (cinfo[icell].corner[0] != UNKNOWN) continue;

    int nin = 0;
    for (int i = 0; i < ncorner; i++) {
      cinfo[icell].corner[i] = c[i] > thresh ? INSIDE : OUTSIDE;
      if (c[i] > thresh) nin++;
    }

    // the cut gave this cell no volume, so supply one when the corner points
    //   agree on which side of the surface the whole cell lies

    if (nin == ncorner) cinfo[icell].volume = 0.0;
    else if (nin == 0) {
      double *lo = cells[icell].lo;
      double *hi = cells[icell].hi;
      if (dimension == 3)
        cinfo[icell].volume = (hi[0]-lo[0]) * (hi[1]-lo[1]) * (hi[2]-lo[2]);
      else if (domain->axisymmetric)
        cinfo[icell].volume =
          MY_PI * (hi[1]*hi[1]-lo[1]*lo[1]) * (hi[0]-lo[0]);
      else
        cinfo[icell].volume = (hi[0]-lo[0]) * (hi[1]-lo[1]);
    }
  }

  // sub cells take type and corner flags from the split cell they belong to,
  //   which is always at a lower index, so one pass suffices
  // zero volume of INSIDE cells in the same pass, so Collide and
  //   FixGridCheck can catch a particle that ended up in one

  int nsplit = grid->nsplitlocal;

  for (int icell = 0; icell < nlocal; icell++) {
    if (nsplit && cells[icell].nsplit <= 0) {
      int splitcell = sinfo[cells[icell].isplit].icell;
      cinfo[icell].type = cinfo[splitcell].type;
      for (int j = 0; j < ncorner; j++)
        cinfo[icell].corner[j] = cinfo[splitcell].corner[j];
    }
    if (cinfo[icell].type == INSIDE) cinfo[icell].volume = 0.0;
  }

  return 1;
}

/* ----------------------------------------------------------------------
   decide what becomes of particle I now that the isosurface was regenerated
   return KEEP    = in the flow, or was put back into it
          DISCARD = buried, already accounted for, caller must discard it
          MIGRATE = pushed into a ghost cell, caller must migrate it
------------------------------------------------------------------------- */

int FixAblate::resolve_engulfed(int i)
{
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  Grid::SplitInfo *sinfo = grid->sinfo;
  Particle::OnePart *particles = particle->particles;

  int splitcell,subcell,pflag;
  double xcell[3];

  int icell = particles[i].icell;

  if (cells[icell].nsurf == 0) {

    // a cell with no surfs is entirely OUTSIDE or entirely INSIDE
    // for deposition a formerly-open cell can become fully INSIDE in one
    //   regeneration, and the surf-straddle test below is skipped for
    //   no-surf cells, so those particles must be handled here
    // there is no surf left in the cell to reflect off, so the molecule
    //   is buried, i.e. incorporated into the film, with full accounting

    if (!depositflag || cinfo[icell].type != INSIDE) return KEEP;

    int salvaged = salvage_to_neighbor(icell,i);
    if (salvaged == 0) {
      salvaged = salvage_to_ghost(icell,i);
      if (salvaged > 0) return MIGRATE;
    }
    if (salvaged > 0) return KEEP;

    // salvaged < 0 means the surface collision model absorbed the
    //   molecule, which has already been accounted for as buried

    if (salvaged == 0) update->bury_particle(&particle->particles[i]);
    return DISCARD;
  }

  double *x = particles[i].x;

  // check that particle is outside surfs
  // if no xcell found, cannot check

  pflag = grid->point_outside_surfs(icell,xcell);
  if (!pflag) return KEEP;
  pflag = grid->outside_surfs(icell,x,xcell);

  // check that particle is in correct split subcell

  if (pflag && cells[icell].nsplit <= 0) {
    splitcell = sinfo[cells[icell].isplit].icell;
    if (dim == 2) subcell = update->split2d(splitcell,x);
    else subcell = update->split3d(splitcell,x);
    if (subcell != icell) pflag = 0;
  }

  if (pflag) return KEEP;

  // for deposition, try to salvage the particle by reflecting it off the
  //   surf that now separates it from the flow, instead of deleting it
  // the move loop advances the front every step and normally reflects a
  //   particle before it can be overtaken, so this is only a safety net
  //   for the residual jump when the isosurface is regenerated
  // reflect against a stationary wall: a growing surface advances by
  //   accretion, so the lattice atoms the molecule strikes are at rest
  // momentum given to the surface is recorded so nothing is unaccounted

  if (!depositflag) return DISCARD;

  int salvaged = salvage_particle(icell,i);
  if (salvaged == 0) salvaged = salvage_to_neighbor(icell,i);
  if (salvaged == 0) {
    salvaged = salvage_to_ghost(icell,i);
    if (salvaged > 0) return MIGRATE;
  }
  if (salvaged > 0) return KEEP;

  // salvage failed too, so account for the molecule as buried, unless a
  //   surface collision model absorbed it during the attempt, in which case
  //   it was already accounted for there

  if (salvaged == 0) update->bury_particle(&particle->particles[i]);
  return DISCARD;
}

/* ----------------------------------------------------------------------
   decide what becomes of particle I, which another proc pushed across a face
     into a cell this proc owns because it had nowhere to put it
   the sender already gave the molecule its one interaction with the closing
     film, so the only question here is whether there is a legal place for it:
     landing in the flow it stays exactly as it is, and landing in solid it is
     moved out to the surface that separates it from the flow, with no second
     collision.
   it is deliberately not offered the neighbor hop or another proc hop.  A
     molecule crosses at most one proc boundary and gets exactly the chances
     it would have had with both cells on one proc -- which is the whole point
     of doing this, so it must not quietly get more.
   return KEEP or DISCARD
------------------------------------------------------------------------- */

int FixAblate::resolve_arrived(int i)
{
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  Particle::OnePart *particles = particle->particles;

  int icell = particles[i].icell;

  if (cells[icell].nsurf == 0) {
    if (cinfo[icell].type != INSIDE) return KEEP;
    update->bury_particle(&particles[i]);
    return DISCARD;
  }

  double xcell[3];
  if (grid->point_outside_surfs(icell,xcell) &&
      grid->outside_surfs(icell,particles[i].x,xcell)) return KEEP;

  if (salvage_particle(icell,i,0) > 0) return KEEP;

  update->bury_particle(&particle->particles[i]);
  return DISCARD;
}

/* ----------------------------------------------------------------------
   insure dlist is long enough for N particles
------------------------------------------------------------------------- */

void FixAblate::grow_dlist(int n)
{
  if (n <= maxdlist) return;
  maxdlist = n;
  memory->destroy(dlist);
  memory->create(dlist,maxdlist,"ablate:dlist");
}

/* ----------------------------------------------------------------------
   set per-cell delta vector randomly
   celldelta = random integer between 0 and maxrandom
   scale = fraction of cells that are decremented
------------------------------------------------------------------------- */

void FixAblate::set_delta_random()
{
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  // enforce same decrement no matter who owns which cells
  // NOTE: could change this at some point, use differnet RNG for each proc

  if (!grid->hashfilled) grid->rehash();
  Grid::MyHash *hash = grid->hash;
  cellint cellID;
  int rn2,icell;
  double rn1;
  for (int i = 0; i < grid->ncell; i++) {
    rn1 = random->uniform();
    rn2 = static_cast<int> (random->uniform()*maxrandom) + 1.0;
    cellID = i+1;
    if (hash->find(cellID) == hash->end()) continue;
    icell = (*hash)[cellID];
    if (icell >= nglocal) continue;     // ghost cell

    if (rn1 > scale) celldelta[icell] = 0.0;
    else celldelta[icell] = rn2;
  }

  // total decrement for output

  double sum = 0.0;
  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;
    sum += celldelta[icell];
  }

  MPI_Allreduce(&sum,&sum_delta,1,MPI_DOUBLE,MPI_SUM,world);
}

/* ----------------------------------------------------------------------
   set per-cell delta vector uniformly
   celldelta = maxrandom
   scale = fraction of cells that are decremented
------------------------------------------------------------------------- */

void FixAblate::set_delta_uniform()
{
  int nin;
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  // enforce same decrement no matter who owns which cells
  // NOTE: could change this at some point to use differnet RNG for each proc

  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;

    // only ablate surfaces with a surface element (not fully inside or outside)

    nin = 0;
    for (int i = 0; i < ncorner; i++) {
      if (multi_val_flag) {
        if (mvalues[icell][i][0] > thresh) nin++;
      } else {
        if (cvalues[icell][i] > thresh) nin++;
      }
    }

    if (nin == 0 || nin == ncorner) celldelta[icell] = 0.0;
    else celldelta[icell] = maxrandom*scale;
  }

  // total decrement for output

  double sum = 0.0;
  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;
    sum += celldelta[icell];
  }

  MPI_Allreduce(&sum,&sum_delta,1,MPI_DOUBLE,MPI_SUM,world);
}

/* ----------------------------------------------------------------------
   set per-cell delta vector from compute/fix/variable source
   celldelta = nevery * scale * source-value
   NOTE: how does this work for split cells? should only do parent split?
------------------------------------------------------------------------- */

void FixAblate::set_delta()
{
  int i;

  // Nevery is here because the source is a per-step quantity applied once per
  //   interval.  With units = distance it is NOT: the source is a rate in
  //   length/time, and end_of_step() multiplies by the elapsed interval when
  //   it converts to a corner value.  Counting Nevery in both places applied
  //   a distance rate Nevery times too strongly.

  double prefactor = (unitsflag == DISTANCE) ? scale : nevery*scale;

  // compute/fix may invoke computes so wrap with clear/add

  modify->clearstep_compute();

  if (which == COMPUTE) {
    Compute *c = modify->compute[icompute];

    if (!(c->invoked_flag & INVOKED_PER_GRID)) {
      c->compute_per_grid();
      c->invoked_flag |= INVOKED_PER_GRID;
    }
    c->post_process_isurf_grid();

    if (argindex == 0) {
      double *cvec = c->vector_grid;
      for (i = 0; i < nglocal; i++)
        celldelta[i] = prefactor * cvec[i];
    } else {
      double **carray = c->array_grid;
      int im1 = argindex - 1;
      for (i = 0; i < nglocal; i++)
        celldelta[i] = prefactor * carray[i][im1];
    }

  } else if (which == FIX) {
    Fix *f = modify->fix[ifix];

    if (argindex == 0) {
      double *fvec = f->vector_grid;
      for (i = 0; i < nglocal; i++) {
        celldelta[i] = prefactor * fvec[i];
      }
    } else {
      double **farray = f->array_grid;
      int im1 = argindex - 1;
      for (i = 0; i < nglocal; i++)
        celldelta[i] = prefactor * farray[i][im1];
    }

  } else if (which == FLUX) {

    // per mixture group incident mass flow onto each cell, weighted by that
    //   group's capture probability, gives the mass the film gains per unit
    //   time.  Divided by the film density and the cell's surface area that
    //   is a growth speed, which the units = distance path then converts.
    // the columns are mass flows, so nothing here needs to know a species
    //   mass: the per species handling comes from the mixture groups of the
    //   source compute

    // a single column comes back as a vector rather than an array, which is
    //   how every per-grid producer in SPARTA reports one value

    double **carray = NULL;
    double *cvec = NULL;
    int ncol;
    if (fluxwhich == COMPUTE) {
      Compute *c = modify->compute[icompute];
      if (!(c->invoked_flag & INVOKED_PER_GRID)) {
        c->compute_per_grid();
        c->invoked_flag |= INVOKED_PER_GRID;
      }
      c->post_process_isurf_grid();
      ncol = c->size_per_grid_cols;
      if (ncol == 0) cvec = c->vector_grid;
      else carray = c->array_grid;
    } else {
      Fix *f = modify->fix[ifix];
      ncol = f->size_per_grid_cols;
      if (ncol == 0) cvec = f->vector_grid;
      else carray = f->array_grid;
    }
    if (!carray && !cvec) return;

    Grid::ChildCell *cells = grid->cells;
    Grid::ChildInfo *cinfo = grid->cinfo;

    for (i = 0; i < nglocal; i++) celldelta[i] = 0.0;

    for (i = 0; i < nglocal; i++) {
      if (!(cinfo[i].mask & groupbit)) continue;
      if (cells[i].nsplit <= 0) continue;

      double area = cell_area(i);
      if (area <= 0.0) continue;

      double mdot = 0.0;
      if (cvec) mdot = sticking[0] * cvec[i];
      else for (int g = 0; g < ncol; g++) mdot += sticking[g] * carray[i][g];
      if (mdot <= 0.0) continue;

      celldelta[i] = scale * mdot / (filmrho * area);
    }

  } else if (which == VARIABLE && varstyle == EQUALVAR) {

    double one = prefactor * input->variable->compute_equal(ivariable);
    for (i = 0; i < nglocal; i++) celldelta[i] = one;

  } else if (which == VARIABLE) {
    if (nglocal > maxvar) {
      maxvar = grid->maxlocal;
      memory->destroy(vbuf);
      memory->create(vbuf,maxvar,"ablate:vbuf");
    }

    input->variable->compute_grid(ivariable,vbuf,1,0);
    for (i = 0; i < nglocal; i++)
      celldelta[i] = prefactor * vbuf[i];
  }

  // NOTE: this does not get invoked on step 100,
  //   b/c needs to also be done in constructor
  //   ditto for fix adapt?
  //   they need nextvalid() methods like fix_ave_time
  //   or do it how output calcs next_stats for next thermo step

  modify->addstep_compute(update->ntimestep + nevery);

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  // deposition only accretes onto material that is already there, so a cell
  //   holding no surface gets nothing
  // ablation does not need this: decrement() drains corner values toward 0, so
  //   a cell that is entirely gas has nothing left to give and is already
  //   self-limiting.  increment() fills toward 255, which is not -- without
  //   the gate a source that is uniform in space, such as a variable, would
  //   raise corner values in open gas far from the surface and eventually
  //   sprout material out of nowhere.
  // this is the same test set_delta_uniform() applies; the compute and fix
  //   sources are naturally zero away from the surface, since the isurf
  //   computes only tally in cells that hold surface elements

  if (depositflag) {
    for (int icell = 0; icell < nglocal; icell++) {
      if (!(cinfo[icell].mask & groupbit)) continue;
      if (cells[icell].nsplit <= 0) continue;

      int nin = 0;
      for (i = 0; i < ncorner; i++)
        if (cvalues[icell][i] > thresh) nin++;
      if (nin == 0 || nin == ncorner) celldelta[icell] = 0.0;
    }
  }

  double sum = 0.0;
  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;
    sum += celldelta[icell];
  }

  MPI_Allreduce(&sum,&sum_delta,1,MPI_DOUBLE,MPI_SUM,world);
}

/* ----------------------------------------------------------------------
   decrement corner points of each owned grid cell
   skip cells not in group, with no surfs, and sub-cells
   algorithm:
     no corner pt value can be < 0.0
     decrement smallest corner pt by full delta
     if cannot, decrement to 0.0, decrement next smallest by remainder, etc
------------------------------------------------------------------------- */

void FixAblate::decrement()
{
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  int i,imin;
  double minvalue,total;
  double *corners;

  // total = full amount to decrement from cell
  // cdelta[icell] = amount to decrement from each corner point of icell

  for (int icell = 0; icell < nglocal; icell++) cflag[icell] = 0.0;

  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;

    for (i = 0; i < ncorner; i++) cdelta[icell][i] = 0.0;

    // see the comment in increment(): a rate in length/time is a uniform
    //   shift of the field, in either direction

    if (unitsflag == DISTANCE) {
      for (i = 0; i < ncorner; i++) cdelta[icell][i] = celldelta[icell];
      cflag[icell] = interface_cell(icell);
      continue;
    }

    total = celldelta[icell];
    corners = cvalues[icell];
    while (total > 0.0) {
      imin = -1;
      minvalue = 256.0;
      for (i = 0; i < ncorner; i++) {
        if (corners[i] > 0.0 && corners[i] < minvalue &&
            cdelta[icell][i] == 0.0) {
          imin = i;
          minvalue = corners[i];
        }
      }
      if (imin == -1) break;

      if (total < corners[imin]) {
        cdelta[icell][imin] += total;
        total = 0.0;
      } else {
        cdelta[icell][imin] = corners[imin];
        total -= corners[imin];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   increment corner points of each owned grid cell (deposition)
   inverse of decrement(): corner point values grow toward 255.0
   skip cells not in group, with no surfs, and sub-cells
   algorithm:
     no corner pt value can be > 255.0
     increment largest sub-255 corner pt by full delta
       (mirror of decrement, which shrinks the smallest positive corner pt)
     if cannot, fill it to 255.0, increment next largest by remainder, etc
------------------------------------------------------------------------- */

void FixAblate::increment()
{
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  int i,imax;
  double maxvalue,total,room;
  double *corners;

  // total = full amount to deposit onto cell
  // cdelta[icell] = amount to add to each corner point of icell

  for (int icell = 0; icell < nglocal; icell++) cflag[icell] = 0.0;

  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;

    for (i = 0; i < ncorner; i++) cdelta[icell][i] = 0.0;

    // a rate in length/time asks the surface to move a set distance, and the
    //   relation end_of_step() used to convert it, dc = s*|grad c|*dt, is the
    //   relation for a field shifted UNIFORMLY by dc.  Concentrating the
    //   whole budget on one corner does not shift the field uniformly, and
    //   the front then moves by only a fraction of what was asked.
    // so hand every corner an equal share.  sync() averages the contributions
    //   of the cells sharing a corner point under these units, so each corner
    //   ends up rising by exactly celldelta.

    if (unitsflag == DISTANCE) {
      for (i = 0; i < ncorner; i++) cdelta[icell][i] = celldelta[icell];
      cflag[icell] = interface_cell(icell);
      continue;
    }

    total = celldelta[icell];
    corners = cvalues[icell];
    while (total > 0.0) {
      imax = -1;
      maxvalue = -1.0;
      for (i = 0; i < ncorner; i++) {
        if (corners[i] < 255.0 && corners[i] > maxvalue &&
            cdelta[icell][i] == 0.0) {
          imax = i;
          maxvalue = corners[i];
        }
      }
      if (imax == -1) break;

      room = 255.0 - corners[imax];
      if (total < room) {
        cdelta[icell][imax] += total;
        total = 0.0;
      } else {
        cdelta[icell][imax] = room;
        total -= room;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   try to push particle I in cell ICELL, which the regenerated isosurface
     now encloses, back out into the flow volume
   ray-trace from the particle toward a known flow-side point to find the
     surf that separates it from the flow, place it just outside that surf,
     and apply the cell's surface collision model
   the wall is treated as stationary: a depositing surface advances by
     accretion, so the local patch of atoms the molecule strikes is at rest,
     and the collision is an ordinary thermal wall interaction
   momentum handed to the surface is accumulated so it is not lost
   surface chemistry is deliberately NOT applied here.  This runs from a fix,
     outside the move loop, and there is nowhere to put a reaction product:
     it cannot be threaded back into a trajectory, and creating one would
     reallocate the particle list underneath the loop that called us.  The
     collision model alone is applied, which is the reflection this path
     exists to perform.
   collideflag = 0 to only move the molecule out to the surface, without a
     collision.  Used for one that arrived from another proc, which already
     had its one interaction with the film before it was sent, and must not
     be given a second one just because it landed in solid on this side.
   return 1 if the particle was salvaged, 0 if it must be buried instead,
     -1 if the collision model consumed the particle, in which case it has
     already been accounted for and the caller must only discard it
------------------------------------------------------------------------- */

int FixAblate::salvage_particle(int icell, int i, int collideflag)
{
  int minsurf;
  double xcell[3],minxc[3];

  if (!grid->point_outside_surfs(icell,xcell)) return 0;
  if (!grid->nearest_surf(icell,particle->particles[i].x,xcell,minsurf,minxc))
    return 0;

  Particle::OnePart *p = &particle->particles[i];
  double *norm;
  int isc;

  if (dim == 3) {
    Surf::Tri *tri = &surf->tris[minsurf];
    norm = tri->norm;
    isc = tri->isc;
  } else {
    Surf::Line *line = &surf->lines[minsurf];
    norm = line->norm;
    isc = line->isc;
  }

  if (isc < 0) return 0;

  // place the particle just outside the surf, along its outward normal
  // EPSSURF displacement matches what Grid::point_outside_surfs() uses,
  //   so the particle passes the outside_surfs() test afterwards

  double eps = EPSSURF * (dim == 3 ? MIN(MIN(xyzsize[0],xyzsize[1]),xyzsize[2])
                                   : MIN(xyzsize[0],xyzsize[1]));

  double xtry[3];
  xtry[0] = minxc[0] + eps*norm[0];
  xtry[1] = minxc[1] + eps*norm[1];
  xtry[2] = (dim == 3) ? minxc[2] + eps*norm[2] : p->x[2];

  // the offset must not carry the particle out of the cell that owns it
  // it can: when the nearest point on the surface is on a cell edge or
  //   corner, the outward normal there generally points out of the cell, and
  //   a particle whose coords are outside its own cell breaks the move loop
  // clamp it back inside, just off the face rather than exactly on it

  double *clo = grid->cells[icell].lo;
  double *chi = grid->cells[icell].hi;
  int clamped = 0;
  for (int k = 0; k < dim; k++) {
    if (xtry[k] < clo[k]) { xtry[k] = clo[k] + eps; clamped = 1; }
    else if (xtry[k] > chi[k]) { xtry[k] = chi[k] - eps; clamped = 1; }
  }

  // clamping moves the particle back toward the surface, so it may no longer
  //   be outside it.  Rather than guess, ask; if it is not, say so and let
  //   the caller bury it, which is accounted for

  if (clamped && !grid->outside_surfs(icell,xtry,xcell)) return 0;

  p->x[0] = xtry[0];
  p->x[1] = xtry[1];
  if (dim == 3) p->x[2] = xtry[2];

  // moving it into the flow was the whole job for a molecule that arrived
  //   from another proc: it was reflected off the film before it was sent

  if (!collideflag) return 1;

  // record the momentum the collision transfers to the surface

  double vold[3];
  vold[0] = p->v[0];
  vold[1] = p->v[1];
  vold[2] = p->v[2];

  // keep a copy of the molecule as it was, so that if the collision model
  //   absorbs it the mass, momentum and energy it carried into the surface
  //   are still the ones that get booked

  Particle::OnePart porig = *p;

  double dtremain = 0.0;
  int reaction = 0;
  surf->sc[isc]->collide(p,dtremain,minsurf,norm,-1,reaction);

  // the collision model absorbed the particle, e.g. surf_collide vanish
  // it went into the surface, so account for it as buried and tell the
  //   caller to discard it rather than try to save it somewhere else

  if (p == NULL) {
    update->bury_particle(&porig);
    return -1;
  }

  tally_reflection(p,vold,porig.erot,porig.evib);

  return 1;
}

/* ----------------------------------------------------------------------
   the film has closed over particle I completely within its own cell ICELL,
     so there is no surface left there to reflect it from
   if a face neighbor still holds gas, the molecule was not sealed in: the
     film squeezed it, and it should be pushed out through that face rather
     than incorporated.  Measurement says this is what most burials are --
     at moderate growth rates every one of them has a neighbor with gas in it,
     and refining the timestep does not reduce their number, because the cause
     is the cell the salvage looks in, not the size of the step.
   reflect off the closing film, taking its normal to be the face the molecule
     leaves through, and reassign the particle to the neighbor cell
   only owned cells are considered: pushing a particle into a ghost cell would
     mean migrating it to another proc from inside a fix.  A molecule whose
     only gas neighbor is off-proc is buried, as it was before.
   as in salvage_particle(), the collision model is applied without surface
     chemistry, since a reaction product has nowhere to go from here
   return 1 if the particle was moved, 0 if it must be buried, -1 if the
     collision model consumed it, already accounted, and it must be discarded
------------------------------------------------------------------------- */

int FixAblate::salvage_to_neighbor(int icell, int i)
{
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  if (isc_default < 0) return 0;

  static const int fdim[6] = {0,0,1,1,2,2};
  int nface = (dim == 2) ? 4 : 6;

  Particle::OnePart *p = &particle->particles[i];

  double eps = EPSSURF * (dim == 3 ?
                          MIN(MIN(xyzsize[0],xyzsize[1]),xyzsize[2]) :
                          MIN(xyzsize[0],xyzsize[1]));

  for (int f = 0; f < nface; f++) {
    if (grid->neigh_decode(cells[icell].nmask,f) != NCHILD) continue;

    int jcell = cells[icell].neigh[f];
    if (jcell < 0 || jcell >= grid->nlocal) continue;
    if (cinfo[jcell].type == INSIDE) continue;

    // step just past the shared face, keeping the other coords, so the
    //   molecule moves as little as possible

    double xtry[3];
    xtry[0] = p->x[0];
    xtry[1] = p->x[1];
    xtry[2] = p->x[2];

    int d = fdim[f];
    if (f % 2 == 0) xtry[d] = cells[jcell].hi[d] - eps;   // neighbor below
    else xtry[d] = cells[jcell].lo[d] + eps;              // neighbor above
    for (int k = 0; k < dim; k++) {
      if (xtry[k] < cells[jcell].lo[k]) xtry[k] = cells[jcell].lo[k] + eps;
      else if (xtry[k] > cells[jcell].hi[k]) xtry[k] = cells[jcell].hi[k] - eps;
    }

    // a split cell owns no particles, its sub cells do

    int kcell = jcell;
    if (cells[jcell].nsplit > 1) {
      if (dim == 2) kcell = update->split2d(jcell,xtry);
      else kcell = update->split3d(jcell,xtry);
      if (kcell < 0 || kcell >= grid->nlocal) continue;
      if (cinfo[kcell].type == INSIDE) continue;
    }
    if (cinfo[kcell].volume == 0.0) continue;

    double vold[3];
    vold[0] = p->v[0];
    vold[1] = p->v[1];
    vold[2] = p->v[2];
    double xold[3];
    xold[0] = p->x[0];
    xold[1] = p->x[1];
    xold[2] = p->x[2];
    double erotold = p->erot;
    double evibold = p->evib;

    p->x[0] = xtry[0];
    p->x[1] = xtry[1];
    if (dim == 3) p->x[2] = xtry[2];

    int done = 0;

    if (cells[kcell].nsurf == 0) {

      // the neighbor is all gas, so the molecule is simply through the face
      // reflect off the film it is leaving, whose local normal here is that
      //   face; the wall is stationary, as everywhere else in deposition

      double norm[3];
      norm[0] = norm[1] = norm[2] = 0.0;
      norm[d] = (f % 2 == 0) ? -1.0 : 1.0;

      Particle::OnePart porig = *p;

      int reaction = 0;
      double dtremain = 0.0;
      surf->sc[isc_default]->collide(p,dtremain,-1,norm,-1,reaction);

      if (p) done = 1;
      else {

        // the collision model absorbed the molecule into the film

        update->bury_particle(&porig);
        return -1;
      }

    } else {

      // the neighbor is cut by the surface, and stepping through the face
      //   usually lands in ITS solid part, since the film is what closed the
      //   molecule in.  Hand it to the ordinary in-cell salvage, which
      //   ray-traces to the flow side of that cell and reflects off the surf
      //   that separates the two -- the right surface and the right normal.

      done = salvage_particle(kcell,i);
      if (done < 0) return -1;
    }

    if (!done) {
      p->x[0] = xold[0];
      p->x[1] = xold[1];
      p->x[2] = xold[2];
      p->v[0] = vold[0];
      p->v[1] = vold[1];
      p->v[2] = vold[2];
      p->erot = erotold;
      p->evib = evibold;
      continue;
    }

    p->icell = kcell;
    cinfo[kcell].count++;
    cinfo[icell].count--;

    // salvage_particle already accounted its own reflection

    if (cells[kcell].nsurf == 0)
      tally_reflection(p,vold,erotold,evibold);

    return 1;
  }

  return 0;
}

/* ----------------------------------------------------------------------
   the film has closed over particle I in cell ICELL and no cell this proc
     owns can take it, but a face neighbor owned by another proc might
   this proc cannot decide that for itself.  The test for being in the flow
     needs the corner flags and the gas-side reference point of the cell, and
     those exist only for owned cells; a ghost cell carries its surfs but not
     whether a given point is inside them.  So push the molecule across the
     face and let the owner adjudicate, which is exactly what create_surfs()
     does with everything that arrives.
   reflect off the closing film first, taking its normal to be the face the
     molecule leaves through, the same as for an owned neighbor.  If the owner
     finds it still enclosed and buries it after all, both events are real and
     both are booked: the reflection recorded the momentum and energy the film
     took, and the burial records what the molecule had left.
   without this a molecule whose only gas neighbor happens to lie on another
     proc is buried, so how many burials a run reports depends on how the grid
     was divided up.  With it the outcome is the same either way.
   return 1 if it was pushed across, with p->icell set to the cell it goes to,
     0 if no neighbor would take it, -1 if the collision model consumed it
------------------------------------------------------------------------- */

int FixAblate::salvage_to_ghost(int icell, int i)
{
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  if (isc_default < 0) return 0;

  static const int fdim[6] = {0,0,1,1,2,2};
  int nface = (dim == 2) ? 4 : 6;
  int ntotal = grid->nlocal + grid->nghost;

  Particle::OnePart *p = &particle->particles[i];

  double eps = EPSSURF * (dim == 3 ?
                          MIN(MIN(xyzsize[0],xyzsize[1]),xyzsize[2]) :
                          MIN(xyzsize[0],xyzsize[1]));

  for (int f = 0; f < nface; f++) {
    if (grid->neigh_decode(cells[icell].nmask,f) != NCHILD) continue;

    int jcell = cells[icell].neigh[f];
    if (jcell < grid->nlocal || jcell >= ntotal) continue;  // owned, or none

    // an "empty" ghost cell, nsurf < 0, is the normal case here: implicit
    //   surf elements never leave the cell that generated them, so a proc is
    //   never sent its neighbors' surfs and every ghost arrives empty.  That
    //   is precisely why the owner has to decide.  Its proc and ilocal are
    //   still valid, which is all migrating the molecule needs -- the move
    //   loop hands particles to empty ghost cells for the same reason.

    // step just past the shared face, keeping the other coords

    double xtry[3];
    xtry[0] = p->x[0];
    xtry[1] = p->x[1];
    xtry[2] = p->x[2];

    int d = fdim[f];
    if (f % 2 == 0) xtry[d] = cells[jcell].hi[d] - eps;   // neighbor below
    else xtry[d] = cells[jcell].lo[d] + eps;              // neighbor above
    for (int k = 0; k < dim; k++) {
      if (xtry[k] < cells[jcell].lo[k]) xtry[k] = cells[jcell].lo[k] + eps;
      else if (xtry[k] > cells[jcell].hi[k]) xtry[k] = cells[jcell].hi[k] - eps;
    }

    // a split cell owns no particles, its sub cells do, but an empty ghost
    //   carries no split info to resolve that with.  Leave it to the owner,
    //   which does it on arrival.

    // reflect off the film it is leaving, whose local normal here is that
    //   face; the wall is stationary, as everywhere else in deposition
    // if the owner turns out to have no room either and buries the molecule
    //   after all, this reflection still happened and both are booked: it
    //   took the change in momentum and energy, and the burial takes what is
    //   left, so together they are the whole of what the molecule carried

    double norm[3];
    norm[0] = norm[1] = norm[2] = 0.0;
    norm[d] = (f % 2 == 0) ? -1.0 : 1.0;

    double vold[3];
    vold[0] = p->v[0];
    vold[1] = p->v[1];
    vold[2] = p->v[2];
    double erotold = p->erot;
    double evibold = p->evib;

    Particle::OnePart porig = *p;

    int reaction = 0;
    double dtremain = 0.0;
    surf->sc[isc_default]->collide(p,dtremain,-1,norm,-1,reaction);

    if (p == NULL) {

      // the collision model absorbed the molecule into the film

      update->bury_particle(&porig);
      return -1;
    }

    p->x[0] = xtry[0];
    p->x[1] = xtry[1];
    if (dim == 3) p->x[2] = xtry[2];
    p->icell = jcell;
    cinfo[icell].count--;

    tally_reflection(p,vold,erotold,evibold);
    update->nfrontmigrate++;

    return 1;
  }

  return 0;
}

/* ----------------------------------------------------------------------
   book a salvage reflection: the momentum and energy particle PTR handed
     to the advancing front, its velocity having been VOLD and its internal
     energies EROTOLD/EVIBOLD before the reflection
   void * to sidestep a fix_ablate.h dependence on particle.h, same as
     Update::bury_particle()
------------------------------------------------------------------------- */

void FixAblate::tally_reflection(void *ptr, double *vold,
                                 double erotold, double evibold)
{
  Particle::OnePart *p = (Particle::OnePart *) ptr;

  double mass = particle->species[p->ispecies].mass;
  double weight = 1.0;
  if (grid->cellweightflag) weight = p->weight;
  double wmass = weight * mass;

  update->nfrontreflect++;
  update->reflect_mom[0] += wmass * (vold[0] - p->v[0]);
  update->reflect_mom[1] += wmass * (vold[1] - p->v[1]);
  update->reflect_mom[2] += wmass * (vold[2] - p->v[2]);
  update->reflect_energy +=
    0.5*wmass*(MathExtra::lensq3(vold) - MathExtra::lensq3(p->v)) +
    weight*(erotold - p->erot) + weight*(evibold - p->evib);
}

/* ----------------------------------------------------------------------
   magnitude of the corner point value gradient in a cell, in value/length
   corner ordering is x fastest, then y, then z
   central differences over the cell extent xyzsize
------------------------------------------------------------------------- */

double FixAblate::grad_mag(int icell)
{
  double *c = cvalues[icell];
  double gx,gy,gz;

  if (dim == 2) {
    gx = 0.5*((c[1]+c[3]) - (c[0]+c[2])) / xyzsize[0];
    gy = 0.5*((c[2]+c[3]) - (c[0]+c[1])) / xyzsize[1];
    gz = 0.0;
  } else {
    gx = 0.25*((c[1]+c[3]+c[5]+c[7]) - (c[0]+c[2]+c[4]+c[6])) / xyzsize[0];
    gy = 0.25*((c[2]+c[3]+c[6]+c[7]) - (c[0]+c[1]+c[4]+c[5])) / xyzsize[1];
    gz = 0.25*((c[4]+c[5]+c[6]+c[7]) - (c[0]+c[1]+c[2]+c[3])) / xyzsize[2];
  }

  return sqrt(gx*gx + gy*gy + gz*gz);
}

/* ----------------------------------------------------------------------
   per-cell normal speed of the advancing deposition front
   the isosurface is the thresh level set of the corner point field, so the
     standard level set advection relation gives the normal speed
       s = (dc/dt) / |grad c|
   dc/dt = realized mean change in the cell's corner values over the interval
     measured from the cvalues_prev snapshot, so it includes the neighbor
     contributions applied by sync() and any clamping at 255
   |grad c| = corner value gradient magnitude, in value/length
   result is a length/time speed, used by the move loop to advance the front
     analytically between the infrequent isosurface regenerations
------------------------------------------------------------------------- */

void FixAblate::front_speed()
{
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  for (int icell = 0; icell < nglocal; icell++) {
    sfront_cell[icell] = 0.0;
    sfront_normal[icell] = NOMEASURE;
  }

  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;
    // sfront_cell feeds the fast-growth guard, which takes a maximum, so an
    //   unmeasurable cell must not look like a large advance: clamp it to 0.
    // sfront_normal feeds the realized-speed diagnostic, which takes a mean,
    //   so it keeps the sentinel and the mean skips those cells.

    double d = edge_displacement(cvalues_prev[icell],cvalues[icell]);
    sfront_cell[icell] = MAX(d,0.0);
    sfront_normal[icell] =
      edge_displacement(cvalues_prev[icell],cvalues[icell],grad_mag(icell),1);
  }
}

/* ----------------------------------------------------------------------
   error if a growing surface has reached the edge of this fix's grid group
   corner point values exist only on the group, so the surface simply stops
     at the group boundary: its elements there have a free end with nothing
     on the other side to meet.  SPARTA's watertight check does catch that,
     but reports it as unmatched points, which gives no hint of the cause.
   an outer face of the group that lies on the simulation box is exempt, since
     a surface is allowed to end on the box and the watertight check says so
   a PERIODIC box face is not exempt: sync() never carries corner point
     values across a periodic boundary, so the film would terminate at the
     face while its periodic image does not exist, and a particle wrapping
     around the boundary would find gas where it just left material
------------------------------------------------------------------------- */

void FixAblate::check_group_boundary()
{
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  double *boxlo = domain->boxlo;
  double *boxhi = domain->boxhi;

  int ncell[3] = {nx,ny,nz};
  int checklo[3],checkhi[3];
  for (int d = 0; d < 3; d++) {
    checklo[d] = (cornerlo[d] != boxlo[d]) ||
      (domain->bflag[2*d] == PERIODIC);
    checkhi[d] = (cornerlo[d] + ncell[d]*xyzsize[d] != boxhi[d]) ||
      (domain->bflag[2*d+1] == PERIODIC);
  }

  // a corner point index has bit d set when the corner is on the cell's
  //   upper face in dimension d, x fastest then y then z

  int flag = 0;

  for (int icell = 0; icell < nglocal && !flag; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;

    for (int d = 0; d < dim; d++) {
      int bit = 1 << d;
      if (checklo[d] && ixyz[icell][d] == 1)
        for (int i = 0; i < ncorner; i++)
          if (!(i & bit) && cvalues[icell][i] >= thresh) flag = 1;
      if (checkhi[d] && ixyz[icell][d] == ncell[d])
        for (int i = 0; i < ncorner; i++)
          if ((i & bit) && cvalues[icell][i] >= thresh) flag = 1;
    }
  }

  int all;
  MPI_Allreduce(&flag,&all,1,MPI_INT,MPI_MAX,world);
  if (all)
    error->all(FLERR,
               "Fix ablate deposition has grown the surface out to the edge "
               "of the fix's grid group, where there are no corner point "
               "values for it to continue into.  The group is fixed when the "
               "implicit surface is created, by read_isurf or create_isurf, "
               "so use one that covers more of the domain, or deposit less "
               "material.  A group reaching a non-periodic simulation box "
               "face is not limited this way, since a surface may end on "
               "the box.  A periodic face is: corner point values are never "
               "carried across a periodic boundary, so a film cannot "
               "continue there");
}

/* ----------------------------------------------------------------------
   how far the isosurface moved between two corner point fields of one cell
   measured by tracking where it crosses the cell edges before and after
   Marching Squares/Cubes place a vertex on an edge by linear interpolation
     to thresh, so the crossing point is exactly reproducible here
   this is deliberately NOT the level set form s = (dc/dt)/|grad c|.  That
     form divides by a gradient which goes to zero wherever the corner field
     is nearly flat, so neighboring cells can get front speeds that differ by
     orders of magnitude.  Since each surf element is advanced along its own
     normal, the advanced front then tears apart at the cell boundary and
     particles stream through the gap without ever being tested.
   an edge is shared by the neighboring cells that meet on it, so a
     displacement measured this way is common to all of them and the advanced
     front stays continuous.  It is also bounded by the cell size by
     construction, so it cannot blow up.
   returns 0.0 if the surface does not cross this cell in the first field
------------------------------------------------------------------------- */

double FixAblate::edge_displacement(double *cold, double *cnew, double gradmag,
                                    int fullflag)
{
  // cell edges as corner index pairs, x fastest then y then z
  // 2d uses the first 4, 3d all 12

  static const int edge[12][2] =
    {{0,1},{2,3},{0,2},{1,3},                     // 2d: 2 x-edges, 2 y-edges
     {4,5},{6,7},{4,6},{5,7},                     // 3d: upper x and y edges
     {0,4},{1,5},{2,6},{3,7}};                    // 3d: z-edges
  static const int edgedim[12] =
    {0,0,1,1, 0,0,1,1, 2,2,2,2};

  int nedge = (dim == 2) ? 4 : 12;

  double sum = 0.0;
  int ncross = 0;

  for (int e = 0; e < nedge; e++) {
    int i0 = edge[e][0];
    int i1 = edge[e][1];

    // an edge crossing slides along the EDGE, which is not the direction the
    //   surface moves unless the two happen to be parallel.  Projecting onto
    //   the surface normal is a factor cos(theta), and for a level set the
    //   cosine between an edge and the normal is just how much of the
    //   gradient lies along that edge:  cos = (|c1-c0|/L) / |grad c|.
    //   So the normal displacement is  dt * |c1-c0| / |grad c|,  and the
    //   edge length cancels out.
    // gradmag <= 0 asks for the raw along-edge magnitude instead, which is
    //   what the fast growth guard wants: there an over-estimate on an
    //   oblique front is the safe direction.

    // the projection can never lengthen the displacement, since cos <= 1, so
    //   the along-edge value is a hard ceiling.  That is not a safety epsilon
    //   but the same geometry: where the cell gradient vanishes -- a saddle,
    //   two opposing pieces of surface in one cell -- there is no single
    //   normal to project onto, and the ceiling falls back to the magnitude.

    double weight = xyzsize[edgedim[e]];
    if (gradmag > 0.0)
      weight = MIN(fabs(cold[i1] - cold[i0]) / gradmag, weight);

    // the isosurface must have crossed this edge in the old field,
    //   else there is no starting point to measure from

    int oldcross = (cold[i0] < thresh) != (cold[i1] < thresh);
    if (!oldcross) continue;

    double dold = cold[i1] - cold[i0];
    if (dold == 0.0) continue;
    double told = (thresh - cold[i0]) / dold;

    int newcross = (cnew[i0] < thresh) != (cnew[i1] < thresh);

    // fullflag wants only edges the surface crossed at both ends of the
    //   interval, i.e. where it stayed inside this cell the whole time and
    //   the displacement measured is the whole of it.  An edge the surface
    //   left partway through contributes only the part of the motion that
    //   happened inside this cell, which is right for a bound on how far the
    //   front went but biases an average of per-cell speeds downward -- and
    //   for a flat front every cell leaves at the same moment, so the bias
    //   lands on every cell at once.

    if (fullflag && !newcross) continue;

    if (newcross) {
      double dnew = cnew[i1] - cnew[i0];
      if (dnew == 0.0) continue;
      double tnew = (thresh - cnew[i0]) / dnew;
      sum += fabs(tnew - told) * weight;
    } else {

      // the surface swept clean past this edge in one step, so it moved at
      //   least from told out to the end it exited through
      // deposition raises the field, so it exits past the corner that was
      //   below thresh; without this branch a very large increment would
      //   leave no crossing to compare against and be reported as no motion

      if (cold[i0] < thresh) sum += told * weight;
      else sum += (1.0 - told) * weight;
    }

    ncross++;
  }

  // no edge of the old field crossed thresh, so there is nothing to measure
  //   from in this cell.  That is NOT a front speed of zero: it happens
  //   whenever the isosurface sits exactly on a plane of corner points, which
  //   a flat front does every time it has advanced by one cell.  Reporting it
  //   as zero made the realized speed collapse to zero on exactly those
  //   steps, and for a flat front that is every cell at once.  Say "no
  //   measurement" instead and let the caller decide; a cell that really did
  //   not move still has crossings and still returns 0.

  if (!ncross) return NOMEASURE;
  return sum/ncross;
}

/* ----------------------------------------------------------------------
   warn once if the corner point field cannot support a rate in length/time
   converting a front speed into a corner point increment needs the direction
     the surface faces, which comes from the cell's corner value gradient.  A
     field that is (nearly) binary carries no direction information where the
     front runs oblique to the grid: every crossed edge falls the full 0 to
     255 whatever way the surface is turned, the gradient the cell reports is
     not the gradient along those edges, and the projection onto the surface
     normal is unavailable -- front_response() clamps instead.
   the front then still moves, and still moves smoothly, but not at the speed
     that was asked for.  Measured on a plane at 45 degrees to a 2d grid, a
     binary field delivers about 1.6 times the requested speed while the same
     plane on a graded field delivers 1.01.  A front normal to the grid is
     exact eithe way, which is why this is easy to miss.
   so say so, once, naming the fraction of the surface affected
------------------------------------------------------------------------- */

void FixAblate::check_oblique()
{
  if (clampwarn) return;

  // the advice this warning carries is "give it a graded field".  Once that
  //   has been done there is nothing further to say: what is left is the
  //   ordinary discretization error of a surface on a grid, and the measure
  //   below does not rank that -- a narrow graded band scores worse on it
  //   than a wide one while being the more accurate of the two.

  if (smoothed) return;

  double one[2],all[2];
  one[0] = clampsum;
  one[1] = 1.0*ntotedge;
  MPI_Allreduce(one,all,2,MPI_DOUBLE,MPI_SUM,world);

  if (all[1] == 0.0) return;
  double excess = all[0]/all[1];
  if (excess <= CLAMP_FRAC) return;

  clampwarn = 1;
  if (me == 0) {
    char str[512];
    sprintf(str,"Fix ablate: this surface faces directions its "
            "corner point values cannot express (mean %.2f against 1.00 "
            "for a field that can), so a rate in length/time is not delivered "
            "accurately: a plane at 45 degrees to the grid comes out about "
            "1.6x too fast.  This is what a binary 0/255 corner point field "
            "looks like wherever the front is not aligned with the grid.  Use "
            "a graded field -- read_isurf with push no on a file carrying "
            "intermediate values, and without minmax yes -- or drive the fix "
            "with units corner instead",
            excess);
    error->warning(FLERR,str);
  }
}

/* ----------------------------------------------------------------------
   1 if icell holds a piece of the isosurface, i.e. its corner point values
     straddle thresh, else 0
   this is the set of cells a rate in length/time is meaningful for, and the
     set sync() averages over
------------------------------------------------------------------------- */

int FixAblate::interface_cell(int icell)
{
  if (!cvalues) return 1;
  int nin = 0;
  for (int i = 0; i < ncorner; i++)
    if (cvalues[icell][i] > thresh) nin++;
  if (nin == 0 || nin == ncorner) return 0;
  return 1;
}

/* ----------------------------------------------------------------------
   total surface area this cell holds, summed over its own elements
   the flux source gives a mass FLOW onto the cell, mass per unit time with
     no area in it, and dividing by this turns it back into a flux.  Doing it
     here rather than letting the compute do it is deliberate: an implicit
     element takes the ID of the cell that generated it, so all the elements
     of a cell share one tally slot, and a compute normalizing by area would
     have divided each contribution by ITS OWN element area and summed the
     results -- a sum of per-element fluxes, not the cell's flux.  Summing the
     areas here and dividing once is the same quantity a single element would
     have given.
   axisymmetric elements are surfaces of revolution, which axi_line_size()
     already accounts for
------------------------------------------------------------------------- */

double FixAblate::cell_area(int icell)
{
  Grid::ChildCell *cells = grid->cells;
  int nsurf = cells[icell].nsurf;
  if (!nsurf) return 0.0;

  surfint *csurfs = cells[icell].csurfs;
  int axisymmetric = domain->axisymmetric;
  double tmp;

  double area = 0.0;
  for (int m = 0; m < nsurf; m++) {
    int isurf = csurfs[m];
    if (dim == 3) area += surf->tri_size(isurf,tmp);
    else if (axisymmetric) area += surf->axi_line_size(isurf);
    else area += surf->line_size(isurf);
  }

  return area;
}

/* ----------------------------------------------------------------------
   replace a binary 0/255 corner point field with a graded one, by giving
     each corner point near the surface a value proportional to its signed
     distance from that surface
   this is what a rate in length/time needs and a segmented image cannot
     give.  On a binary field every crossed cell edge falls the full 0 to 255
     whichever way the surface is turned, so the direction the surface faces
     is not recoverable, and a speed cannot be converted into a corner point
     increment.  Measured on a plane oblique to the grid the front then runs
     at 1/cos of the speed asked for -- sqrt(2) at 45 degrees in 2d, sqrt(3)
     down a cell diagonal in 3d.
   the surface is not moved: the distance is measured to the surface marching
     squares/cubes builds from the field AS READ, so the graded field
     describes the same body, only in a form that says which way it faces.
   BAND is the half width of the graded band in grid cells.  Outside it the
     field stays saturated, which is what keeps the isosurface local and the
     cell in/out marking unchanged.
   each cell can only see its own surface elements -- implicit elements never
     leave the cell that made them, so a neighbour's are not available even as
     ghosts -- so every cell offers a distance for each of its own corner
     points and the corner takes the smallest offered.  That is the same
     2x2x2 stencil sync() gathers over, so the same halo exchange serves.
------------------------------------------------------------------------- */

void FixAblate::distance_transform(double band)
{
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  double hmin = MIN(xyzsize[0],xyzsize[1]);
  if (dim == 3) hmin = MIN(hmin,xyzsize[2]);
  double scale_d = MIN(thresh,255.0-thresh) / (band*hmin);

  double pt2d[4][3],pt3d[36][3];

  // every cell offers a distance for each of its own corner points, BIGDIST
  //   where it has nothing to say

  for (int icell = 0; icell < nglocal; icell++) {
    for (int i = 0; i < ncorner; i++) cdelta[icell][i] = BIGDIST;

    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;
    if (!interface_cell(icell)) continue;

    int ns;
    if (dim == 2)
      ns = ms->cell_surfs(cvalues[icell],NULL,cells[icell].lo,
                          cells[icell].hi,pt2d);
    else
      ns = mc->cell_surfs(cvalues[icell],NULL,cells[icell].lo,
                          cells[icell].hi,pt3d,NULL);
    if (!ns) continue;

    for (int i = 0; i < ncorner; i++) {

      // corner point coords, x fastest then y then z

      double x[3];
      x[0] = (i & 1) ? cells[icell].hi[0] : cells[icell].lo[0];
      x[1] = (i & 2) ? cells[icell].hi[1] : cells[icell].lo[1];
      x[2] = (dim == 3 && (i & 4)) ? cells[icell].hi[2] : cells[icell].lo[2];

      double best = BIGDIST;
      for (int k = 0; k < ns; k++) {
        double n[3],p1[3],d;
        if (dim == 2) {
          for (int m = 0; m < 3; m++) p1[m] = pt2d[2*k][m];
          double e0 = pt2d[2*k+1][0]-p1[0];
          double e1 = pt2d[2*k+1][1]-p1[1];
          double len = sqrt(e0*e0+e1*e1);
          if (len == 0.0) continue;
          n[0] = -e1/len; n[1] = e0/len; n[2] = 0.0;
          d = fabs((x[0]-p1[0])*n[0] + (x[1]-p1[1])*n[1]);
        } else {
          double e1v[3],e2v[3];
          for (int m = 0; m < 3; m++) p1[m] = pt3d[3*k][m];
          MathExtra::sub3(&pt3d[3*k+1][0],p1,e1v);
          MathExtra::sub3(&pt3d[3*k+2][0],p1,e2v);
          MathExtra::cross3(e1v,e2v,n);
          double len = MathExtra::len3(n);
          if (len == 0.0) continue;
          n[0] /= len; n[1] /= len; n[2] /= len;
          d = fabs((x[0]-p1[0])*n[0] + (x[1]-p1[1])*n[1] + (x[2]-p1[2])*n[2]);
        }
        if (d < best) best = d;
      }

      // sign it from the marking the field already carries, rather than from
      //   the element normal, so the graded field can never disagree with the
      //   side of the surface a corner point was on

      if (best < BIGDIST)
        cdelta[icell][i] = (cvalues[icell][i] >= thresh) ? best : -best;
    }
  }

  // a corner point is shared by up to 2^dim cells and takes the smallest
  //   distance any of them offered, which is the nearest piece of surface

  comm_neigh_corners(CDELTA);

  int ix,iy,iz,jx,jy,jz,jcorner,jcell;

  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;

    ix = ixyz[icell][0];
    iy = ixyz[icell][1];
    iz = ixyz[icell][2];

    for (int i = 0; i < ncorner; i++) {
      int ixfirst = (i % 2) - 1;
      int iyfirst = (i/2 % 2) - 1;
      int izfirst = (dim == 2) ? 0 : (i / 4) - 1;

      double best = BIGDIST;
      jcorner = ncorner;

      for (jz = izfirst; jz <= izfirst+1; jz++) {
        for (jy = iyfirst; jy <= iyfirst+1; jy++) {
          for (jx = ixfirst; jx <= ixfirst+1; jx++) {
            jcorner--;
            if (ix+jx < 1 || ix+jx > nx) continue;
            if (iy+jy < 1 || iy+jy > ny) continue;
            if (iz+jz < 1 || iz+jz > nz) continue;
            jcell = walk_to_neigh(icell,jx,jy,jz);
            double cand;
            if (jcell < nglocal) cand = cdelta[jcell][jcorner];
            else cand = cdelta_ghost[jcell-nglocal][jcorner];
            if (fabs(cand) < fabs(best)) best = cand;
          }
        }
      }

      if (best == BIGDIST) continue;
      double v = thresh + scale_d*best;
      cvalues[icell][i] = MAX(0.0,MIN(255.0,v));
    }
  }

  // a value landing exactly on thresh puts a vertex exactly on a grid corner
  //   point, which is what epsilon_adjust() exists to prevent

  epsilon_adjust();
}

/* ----------------------------------------------------------------------
   fraction of a cell that is solid, i.e. on the >= thresh side of the
     isosurface, using exactly the linear edge crossings Marching
     Squares/Cubes place their vertices at, so the number describes the
     surface SPARTA actually builds rather than the field behind it
   2d: walk the cell boundary counter-clockwise, emitting each solid corner
     and each edge crossing, and take the area of the polygon that results
   3d: a trilinear field is bilinear on every z slice, with the slice's four
     corner values interpolated linearly between the bottom and top faces, so
     integrate the 2d area over z.  Gauss-Legendre is exact for the low order
     polynomial the area is in z away from a topology change, and close
     enough through one.
------------------------------------------------------------------------- */

double FixAblate::solid_area_2d(double *c)
{
  // corners counter-clockwise: 0 (lo,lo), 1 (hi,lo), 3 (hi,hi), 2 (lo,hi)

  static const int ord[4] = {0,1,3,2};
  static const double px[4] = {0.0,1.0,1.0,0.0};
  static const double py[4] = {0.0,0.0,1.0,1.0};

  double xs[8],ys[8];
  int n = 0;

  for (int k = 0; k < 4; k++) {
    int k1 = (k+1) % 4;
    int a = ord[k], b = ord[k1];
    int ain = (c[a] >= thresh), bin = (c[b] >= thresh);

    if (ain) { xs[n] = px[k]; ys[n] = py[k]; n++; }

    if (ain != bin) {
      double d = c[b] - c[a];
      double t = (d == 0.0) ? 0.5 : (thresh - c[a]) / d;
      if (t < 0.0) t = 0.0;
      else if (t > 1.0) t = 1.0;
      xs[n] = px[k] + t*(px[k1]-px[k]);
      ys[n] = py[k] + t*(py[k1]-py[k]);
      n++;
    }
  }

  if (n < 3) return 0.0;

  double a2 = 0.0;
  for (int i = 0; i < n; i++) {
    int j = (i+1) % n;
    a2 += xs[i]*ys[j] - xs[j]*ys[i];
  }
  return 0.5*fabs(a2);
}

/* ---------------------------------------------------------------------- */

double FixAblate::solid_fraction(double *c)
{
  if (dim == 2) return solid_area_2d(c);

  // 5 point Gauss-Legendre on [0,1]

  static const double gz[5] =
    {0.046910077030668, 0.230765344947158, 0.500000000000000,
     0.769234655052842, 0.953089922969332};
  static const double gw[5] =
    {0.118463442528095, 0.239314335249683, 0.284444444444444,
     0.239314335249683, 0.118463442528095};

  double cz[4],v = 0.0;
  for (int q = 0; q < 5; q++) {
    double z = gz[q];
    for (int i = 0; i < 4; i++) cz[i] = c[i]*(1.0-z) + c[i+4]*z;
    v += gw[q] * solid_area_2d(cz);
  }
  return v;
}

/* ----------------------------------------------------------------------
   the uniform corner point rise that changes this cell's solid fraction by
     DVFRAC, growing if GROW else receding
   this is the volume form of front_response(), and it is what a depositing
     surface is really being told: the flux delivers a MASS, which over the
     film density is a VOLUME.  Asking for a volume needs no surface normal,
     and that is the whole point -- the normal form has to project an edge
     crossing onto the direction the surface faces, and a corner point field
     that is coarse or binary cannot say what that direction is.  Here the
     surface area the volume is spread over is measured rather than inferred,
     so an oblique front is no harder than a grid-aligned one.
   the solid fraction is monotone in the shift and bounded by 0 and 1, so a
     bisection cannot fail and needs no derivative
------------------------------------------------------------------------- */

double FixAblate::volume_shift(double *c, double dvfrac, int grow)
{
  if (dvfrac <= 0.0) return 0.0;

  double cs[8];
  double v0 = solid_fraction(c);
  double target = grow ? v0 + dvfrac : v0 - dvfrac;
  if (target >= 1.0) target = 1.0;
  else if (target <= 0.0) target = 0.0;

  // 255 always saturates the cell, so it brackets any achievable target

  double lo = 0.0, hi = 255.0;
  for (int it = 0; it < 60; it++) {
    double mid = 0.5*(lo+hi);
    for (int i = 0; i < ncorner; i++) {
      double v = grow ? c[i] + mid : c[i] - mid;
      cs[i] = MAX(0.0,MIN(255.0,v));
    }
    double v = solid_fraction(cs);
    if (grow ? (v < target) : (v > target)) lo = mid;
    else hi = mid;
  }
  return 0.5*(lo+hi);
}

/* ----------------------------------------------------------------------
   surface area this cell holds, NOT revolved for an axisymmetric run
   the volume response spreads a swept volume over the surface to get a
     normal displacement, and a displacement is a length whether or not the
     geometry is a surface of revolution; cell_area() is the revolved one and
     is what an incident flux has to be divided by
------------------------------------------------------------------------- */

double FixAblate::cell_area_cart(int icell)
{
  Grid::ChildCell *cells = grid->cells;
  int nsurf = cells[icell].nsurf;
  if (!nsurf) return 0.0;

  surfint *csurfs = cells[icell].csurfs;
  double tmp;

  double area = 0.0;
  for (int m = 0; m < nsurf; m++) {
    int isurf = csurfs[m];
    if (dim == 3) area += surf->tri_size(isurf,tmp);
    else area += surf->line_size(isurf);
  }
  return area;
}

/* ----------------------------------------------------------------------
   how far the isosurface in this cell moves per unit rise of its corner point
     values, i.e. d(normal displacement)/d(corner delta), evaluated on the
     field as it stands
   inverting this is what turns a rate in length/time into a corner point
     increment, and it replaces the plain level set relation dc = s*|grad c|,
     which is only the unsaturated case of it.
   on an edge the isosurface crosses, with the low value a < thresh < b:
     both ends free   ->  the crossing moves by  D*L/(b-a),  the level set
                          answer, since raising both ends slides the crossing
                          without changing the slope
     b pinned at 255  ->  only a can rise, the slope changes as well, and the
                          crossing moves by  D*L*(255-thresh)/(255-a)^2.
                          A binary 0/255 field is entirely this case, and at
                          a = 0 it is half the level set answer -- which is
                          why asking for a speed used to deliver half of it
   the ablate direction is the mirror image, pinned at 0 instead of 255
   edges are enumerated and weighted exactly as edge_displacement() does, so
     what this predicts is what that measures
------------------------------------------------------------------------- */

double FixAblate::front_response(int icell, int grow)
{
  static const int edge[12][2] =
    {{0,1},{2,3},{0,2},{1,3},
     {4,5},{6,7},{4,6},{5,7},
     {0,4},{1,5},{2,6},{3,7}};
  static const int edgedim[12] =
    {0,0,1,1, 0,0,1,1, 2,2,2,2};

  int nedge = (dim == 2) ? 4 : 12;
  double *c = cvalues[icell];
  double gradmag = grad_mag(icell);

  double sum = 0.0;
  int ncross = 0;

  for (int e = 0; e < nedge; e++) {
    int i0 = edge[e][0];
    int i1 = edge[e][1];
    if ((c[i0] < thresh) == (c[i1] < thresh)) continue;

    double lo = MIN(c[i0],c[i1]);
    double hi = MAX(c[i0],c[i1]);
    double gap = hi - lo;
    if (gap == 0.0) continue;

    // project onto the surface normal, clamped by cos <= 1 as above

    double weight = xyzsize[edgedim[e]];
    if (gradmag > 0.0) {

      // gap/gradmag is the edge length times the cosine between the edge and
      //   the surface normal, so it can never exceed the edge length.  When
      //   it does, the cell's centre gradient and the gradient along this
      //   edge disagree, which is what a corner point field that is nearly
      //   binary looks like where the front runs oblique to the grid.  The
      //   clamp keeps the arithmetic sane but the projection it stands for
      //   is then not available, and a rate in length/time is not delivered
      //   accurately; count it so end_of_step() can say so.

      // how far past 1 the cosine comes out is the size of the disagreement,
      //   not just that there was one: a graded field can be a little over on
      //   a few edges and still deliver the rate, while a binary field is
      //   over by 1/cos everywhere the front runs oblique -- sqrt(2) at 45
      //   degrees in 2d.  Accumulate the mean so the two can be told apart.

      ntotedge++;
      clampsum += MAX(1.0, gap/(gradmag*weight));
      weight = MIN(gap/gradmag,weight);
    }

    double deriv;
    if (grow) {
      if (hi < 255.0) deriv = weight / gap;
      else {
        double room = 255.0 - lo;
        if (room <= 0.0) continue;
        deriv = weight * (255.0 - thresh) / (room*room);
      }
    } else {
      if (lo > 0.0) deriv = weight / gap;
      else {
        if (hi <= 0.0) continue;
        deriv = weight * thresh / (hi*hi);
      }
    }

    sum += deriv;
    ncross++;
  }

  if (!ncross) return 0.0;
  return sum/ncross;
}

/* ----------------------------------------------------------------------
   refresh the collision geometry of a growing surface
   the isosurface, with all of its cut-cell and connectivity work, is rebuilt
     only every Nevery steps, but the surface keeps growing in between.  Here
     the corner point field is advanced in time and the isosurface re-derived
     from it, so the move loop sees the surface where it actually is now.
   working from the FIELD is the essential point.  Displacing the element
     vertices instead is watertight, but only describes the surface while the
     marching squares case is unchanged; it breaks down as soon as corner
     values cross thresh and vertices are created or destroyed.  Re-deriving
     from the field stays correct through those topology changes, since
     marching squares yields a valid watertight surface for whatever field it
     is handed, and neighboring cells share the corner values along their
     common edges so their geometry still meets exactly.
   the field is extrapolated FORWARD from the last regeneration at the rate
     realized over the previous interval.  Forward is what keeps it safe: the
     refreshed surface then always holds at least as much material as the
     committed one, so a particle outside it is outside the committed surface
     too and remains consistent with the cell in/out typing and flow volumes,
     which are deliberately not redone here.
------------------------------------------------------------------------- */

void FixAblate::refresh_surfs()
{
  if (!depositflag || !nevery) return;
  if (update->front_step0 < 0) return;

  // the KOKKOS move loop is a separate implementation and does not read the
  //   refreshed geometry, so a Kokkos run resolves the growth only once per
  //   Nevery, as it did before

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  // elapsed fraction of an interval, at the middle of the step about to run

  double f = (update->ntimestep - update->front_step0 - 0.5) / nevery;
  if (f < 0.0) f = 0.0;

  // do not advance the field past the point where any cell that currently
  //   holds a piece of surface would become entirely solid
  // beyond that the cell has no refreshed surface to offer, and the move loop
  //   would silently fall back to the committed geometry, which by then sits
  //   behind the real surface: particles walk straight through it
  // the limit has to be one number for the whole domain, since neighboring
  //   cells must evaluate the shared corner values identically or their
  //   geometry no longer meets

  double flim = f;
  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;
    if (cells[icell].nsurf == 0) continue;
    for (int i = 0; i < ncorner; i++) {
      double c = cvalues[icell][i];
      double d = c - cvalues_prev[icell][i];
      if (d <= 0.0 || c >= thresh) continue;
      double fclose = (thresh - c) / d;
      if (fclose < flim) flim = fclose;
    }
  }

  double allflim;
  MPI_Allreduce(&flim,&allflim,1,MPI_DOUBLE,MPI_MIN,world);
  if (f > 0.9*allflim) f = 0.9*allflim;
  if (f < 0.0) f = 0.0;

  if (nglocal > maxsegcell) {
    maxsegcell = nglocal;
    // 2 line segments per cell in 2d, up to 12 triangles per cell in 3d
    segstride = (dim == 2) ? 2 : 12;
    memory->grow(nseg,maxsegcell,"ablate:nseg");
    memory->grow(segpt,maxsegcell,segstride*3*dim,"ablate:segpt");
    memory->grow(segnorm,maxsegcell,segstride*3,"ablate:segnorm");
    memory->grow(segspeed,maxsegcell,"ablate:segspeed");
    memory->grow(segband,maxsegcell,"ablate:segband");
    memory->grow(cnow,ncorner,"ablate:cnow");
    memory->grow(cnext,ncorner,"ablate:cnext");
  }

  for (int icell = 0; icell < nglocal; icell++) {
    nseg[icell] = 0;
    segspeed[icell] = 0.0;
    segband[icell] = 0.0;
  }

  // one step of the interpolation, used to measure the front speed below

  double fstep = 1.0/nevery;

  double pt2d[4][3];
  double pt3d[36][3];

  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;
    if (cells[icell].nsurf == 0) continue;

    int moving = 0;
    for (int i = 0; i < ncorner; i++) {
      double d = cvalues[icell][i] - cvalues_prev[icell][i];
      if (d != 0.0) moving = 1;
      double c = cvalues[icell][i] + f*d;
      if (c < 0.0) c = 0.0;
      else if (c > 255.0) c = 255.0;
      cnow[i] = c;
      c = cvalues[icell][i] + (f+fstep)*d;
      if (c < 0.0) c = 0.0;
      else if (c > 255.0) c = 255.0;
      cnext[i] = c;
    }
    if (!moving) continue;

    // how fast the front is advancing along its own normal, measured the
    //   same way the guard measures it: from where the isosurface crosses
    //   the cell edges now and one timestep from now
    // the move loop needs this to place a collision correctly in space AND
    //   in time, and to catch a particle the front overtakes from behind

    segspeed[icell] = MAX(edge_displacement(cnow,cnext,0.0),0.0) / update->dt;

    // and how far it now stands ahead of the committed surface, which is how
    //   far behind it a particle may be found and still be one the front has
    //   overtaken since the rebuild
    // measured between the two fields rather than taken as speed x elapsed
    //   time, since f above is held back whenever advancing it further would
    //   empty a cell of surface, and the two then disagree

    segband[icell] = MAX(edge_displacement(cvalues[icell],cnow,0.0),0.0);

    int ns;
    if (dim == 2)
      ns = ms->cell_surfs(cnow,NULL,cells[icell].lo,cells[icell].hi,pt2d);
    else
      ns = mc->cell_surfs(cnow,NULL,cells[icell].lo,cells[icell].hi,pt3d,NULL);
    if (!ns) continue;
    if (ns > segstride) ns = segstride;

    for (int k = 0; k < ns; k++) {
      double *nm = &segnorm[icell][3*k];

      if (dim == 2) {
        double *p1 = &segpt[icell][6*k];
        double *p2 = &segpt[icell][6*k+3];
        for (int m = 0; m < 3; m++) {
          p1[m] = pt2d[2*k][m];
          p2[m] = pt2d[2*k+1][m];
        }

        // outward normal by the right hand rule Surf uses: Z x (p2-p1)

        nm[0] = -(p2[1]-p1[1]);
        nm[1] = p2[0]-p1[0];
        nm[2] = 0.0;
        double len = sqrt(nm[0]*nm[0] + nm[1]*nm[1]);
        if (len == 0.0) { ns = k; break; }
        nm[0] /= len;
        nm[1] /= len;

      } else {

        // Marching Cubes emits the corner points in the order add_tri() is
        //   given them reversed, so keep the same convention here

        double *p1 = &segpt[icell][9*k];
        double *p2 = &segpt[icell][9*k+3];
        double *p3 = &segpt[icell][9*k+6];
        for (int m = 0; m < 3; m++) {
          p1[m] = pt3d[3*k+2][m];
          p2[m] = pt3d[3*k+1][m];
          p3[m] = pt3d[3*k][m];
        }

        double e1[3],e2[3];
        MathExtra::sub3(p2,p1,e1);
        MathExtra::sub3(p3,p1,e2);
        MathExtra::cross3(e1,e2,nm);
        double len = MathExtra::len3(nm);
        if (len == 0.0) { ns = k; break; }
        nm[0] /= len; nm[1] /= len; nm[2] /= len;
      }
    }

    nseg[icell] = ns;
  }

  update->segpt = segpt;
  update->segnorm = segnorm;
  update->segspeed = segspeed;
  update->segband = segband;
  update->nseg = nseg;
  update->nsegcell = nglocal;
}

/* ---------------------------------------------------------------------- */

void FixAblate::start_of_step()
{
  refresh_surfs();
}

/* ----------------------------------------------------------------------
   sync all copies of corner points values for all owned grid cells
   algorithm:
     comm my cdelta values that are shared by neighbor
     each corner point is shared by N cells, less on borders
     dsum = sum of decrements to that point by all N cells
     newvalue = MAX(oldvalue-dsum,0)
   all N copies of corner pt are set to newvalue
     in numerically consistent manner (same order of operations)
------------------------------------------------------------------------- */

void FixAblate::sync()
{
  int i,ix,iy,iz,jx,jy,jz,ixfirst,iyfirst,izfirst,jcorner;
  int icell,jcell,ncontrib;
  double total;

  comm_neigh_corners(CDELTA);

  // perform update of corner pts for all my owned grid cells
  //   using contributions from all cells that share the corner point
  // insure order of numeric operations will give exact same answer
  //   for all Ncorner duplicates of a corner point (stored by other cells)

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  for (icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;

    ix = ixyz[icell][0];
    iy = ixyz[icell][1];
    iz = ixyz[icell][2];

    // loop over corner points

    for (i = 0; i < ncorner; i++) {

      // ixyz first = offset from icell of lower left cell of 2x2x2 stencil
      //              that shares the Ith corner point

      ixfirst = (i % 2) - 1;
      iyfirst = (i/2 % 2) - 1;
      if (dim == 2) izfirst = 0;
      else izfirst = (i / 4) - 1;

      // loop over 2x2x2 stencil of cells that share the corner point
      // also works for 2d, since izfirst = 0

      total = 0.0;
      ncontrib = 0;
      jcorner = ncorner;

      for (jz = izfirst; jz <= izfirst+1; jz++) {
        for (jy = iyfirst; jy <= iyfirst+1; jy++) {
          for (jx = ixfirst; jx <= ixfirst+1; jx++) {
            jcorner--;

            // check if neighbor cell is within bounds of ablate grid

            if (ix+jx < 1 || ix+jx > nx) continue;
            if (iy+jy < 1 || iy+jy > ny) continue;
            if (iz+jz < 1 || iz+jz > nz) continue;

            // jcell = local index of (jx,jy,jz) neighbor cell of icell

            jcell = walk_to_neigh(icell,jx,jy,jz);

            // update total with one corner point of jcell
            // jcorner descends from ncorner

            if (jcell < nglocal) {
              total += cdelta[jcell][jcorner];
              if (cflag[jcell] != 0.0) ncontrib++;
            } else {
              total += cdelta_ghost[jcell-nglocal][jcorner];
              if (cflag_ghost[jcell-nglocal] != 0.0) ncontrib++;
            }
          }
        }
      }

      // a source in length/time asks the surface to move a set distance, and
      //   the conversion dc = s*|grad c|*dt that produced celldelta is the
      //   relation for a field shifted UNIFORMLY by dc.  So every cell asks
      //   for the same dc at every one of its corner points, and what a
      //   corner point wants is the AVERAGE of what the cells sharing it
      //   asked for, not their sum.
      // the count has to be taken here rather than divided out beforehand,
      //   because a cell only contributes where it holds a piece of surface:
      //   cells that are entirely solid or entirely gas ask for nothing, and
      //   dividing by the geometric 2^dim would then leave the corner short
      //   by exactly the fraction of its neighbours that had nothing to say.
      // what counts is whether the cell HOLDS a piece of surface, which is
      //   what cflag records -- not whether it asked for a non-zero amount.
      //   With a stochastic rate an interface cell that measured no flux
      //   this interval is asking for zero, and treating that as an absent
      //   neighbour would average over the cells that did measure something
      //   and move the front too fast by exactly 1/(fraction non-zero).

      if (unitsflag == DISTANCE && ncontrib) total /= ncontrib;

      // ABLATE: newvalue = MAX(oldvalue-dsum,0)
      // DEPOSIT: newvalue = MIN(oldvalue+dsum,255)

      if (mode == BOTH) {

        // a signed shift, clamped at both ends: the sign of what the cells
        //   sharing this corner asked for is what says which way it goes

        double v = cvalues[icell][i] + total;
        cvalues[icell][i] = MAX(0.0,MIN(255.0,v));
      } else if (mode == DEPOSIT) {
        if (total > 255.0 - cvalues[icell][i]) cvalues[icell][i] = 255.0;
        else cvalues[icell][i] += total;
      } else {
        if (total > cvalues[icell][i]) cvalues[icell][i] = 0.0;
        else cvalues[icell][i] -= total;
      }

    }
  }
}

/* ----------------------------------------------------------------------
   ensure each corner point value is not too close to threshold
   this avoids creating tiny or zero-size surface elements
   corner_inside_min and corner_outside_max are set in store_corners()
     via epsilon method or isosurface stuffing method
------------------------------------------------------------------------- */

void FixAblate::epsilon_adjust()
{
  int i,icell;

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  // a corner value exactly equal to thresh makes Marching Squares/Cubes place
  // a vertex exactly on a grid corner point.  When a surface feature is
  // grid-aligned this makes neighboring cells emit coincident vertices there,
  // producing a non-watertight surface (e.g. create_isurf of a body whose flat
  // face lies on a grid line) and inconsistent inside/outside cell marking.
  // Removing exactly-on-threshold values is a hard numerical requirement and
  // is always enforced.  Deposition reaches this case often, since it drives
  // corner values upward through thresh.  The wider EPSILON band, which also
  // suppresses tiny surface elements, is only applied when the user requests
  // it via mindist > 0 so as not to change existing results.

  for (icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;

    for (i = 0; i < ncorner; i++) {
      if (mindist > 0.0) {
        if (cvalues[icell][i] >= thresh && cvalues[icell][i] < thresh + EPSILON)
          cvalues[icell][i] = thresh - EPSILON;
        else if (cvalues[icell][i] < thresh && cvalues[icell][i] > thresh - EPSILON)
          cvalues[icell][i] = thresh - EPSILON;
      } else if (cvalues[icell][i] == thresh) {
        cvalues[icell][i] = thresh - EPSILON;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   push corner points value to 0 or 255
     if all surrounding neighs are below or above threshold
     do this for all N copies of an affected corner point
   algorithm:
     comm my cdelta values that are shared by neighbor
     each corner point is shared by N cells, less on borders
     dsum = sum of decrements to that point by all N cells
     newvalue = MAX(oldvalue-dsum,0)
   all N copies of corner pt are set to newvalue
     in numerically consistent manner (same order of operations)
------------------------------------------------------------------------- */

void FixAblate::push_lohi()
{
  int i,ix,iy,iz,ixfirst,iyfirst,izfirst,jx,jy,jz;
  int icell,jcell,jcorner,pushflag;

  comm_neigh_corners(CVALUE);

  // perform push of corner pt values for all my owned grid cells
  //   by checking corner pt values of all cells that share same corner pt
  // if all surrounding corner pts are > threshold, push corner pt -> 255
  // if all surrounding corner pts are < threshold, push corner pt -> 0

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  int plo = 0;
  int phi = 0;

  for (icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;

    ix = ixyz[icell][0];
    iy = ixyz[icell][1];
    iz = ixyz[icell][2];

    // loop over corner points

    for (i = 0; i < ncorner; i++) {

      // flag = -1 if corner pt value < threshold, +1 if > threshold

      if (cvalues[icell][i] < thresh) pushflag = -1;
      else if (cvalues[icell][i] > thresh) pushflag = 1;
      else continue;

      // ixyz first = offset from icell of lower left cell of 2x2x2 stencil
      //              that shares the Ith corner point

      ixfirst = (i % 2) - 1;
      iyfirst = (i/2 % 2) - 1;
      if (dim == 2) izfirst = 0;
      else izfirst = (i / 4) - 1;

      // loop over 2x2x2 stencil of cells that share the corner point
      // also works for 2d, since izfirst = 0

      jcorner = ncorner;

      for (jz = izfirst; jz <= izfirst+1; jz++) {
        for (jy = iyfirst; jy <= iyfirst+1; jy++) {
          for (jx = ixfirst; jx <= ixfirst+1; jx++) {
            jcorner--;

            // check if neighbor cell is within bounds of ablate grid

            if (ix+jx < 1 || ix+jx > nx) continue;
            if (iy+jy < 1 || iy+jy > ny) continue;
            if (iz+jz < 1 || iz+jz > nz) continue;

            // jcell = local index of (jx,jy,jz) neighbor cell of icell

            jcell = walk_to_neigh(icell,jx,jy,jz);

            // set pushflag to 0 if jcorner pt of jcell is not
            //   on same side of threshold as icorner or icell

            if (jcell < nglocal) {
              if ((pushflag == -1 && cvalues[jcell][jcorner] > thresh) ||
                  (pushflag == 1 && cvalues[jcell][jcorner] < thresh))
                pushflag = 0;
            } else {
              if ((pushflag == -1 && cdelta_ghost[jcell-nglocal][jcorner] >
                   thresh) ||
                  (pushflag == 1 && cdelta_ghost[jcell-nglocal][jcorner] <
                   thresh))
                pushflag = 0;
            }
          }
        }
      }

      // DEBUG OFF
      if (pushflag == -1) cvalues[icell][i] = 0;
      else if (pushflag == 1) cvalues[icell][i] = 255;

      if (pushflag == -1) plo++;
      else if (pushflag == 1) phi++;
    }
  }

  bigint bplo = plo;
  bigint bphi = phi;
  bigint ploall,phiall;
  MPI_Allreduce(&bplo,&ploall,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
  MPI_Allreduce(&bphi,&phiall,1,MPI_SPARTA_BIGINT,MPI_SUM,world);

  if (me == 0) {
    if (screen)
      fprintf(screen,"  " BIGINT_FORMAT " " BIGINT_FORMAT
              " pushed corner pt values\n",ploall,phiall);
    if (logfile)
      fprintf(logfile,"  " BIGINT_FORMAT " " BIGINT_FORMAT
              " pushed corner pt values\n",ploall,phiall);
  }
}

/* ----------------------------------------------------------------------
   comm my cdelta values that are shared by neighbor cells
   each corner point is shared by N cells, less on borders
   done via irregular comm
------------------------------------------------------------------------- */

void FixAblate::comm_neigh_corners(int which)
{
  int i,j,k,m,n,ix,iy,iz,jx,jy,jz;
  int icell,ifirst,jcell,proc,ilocal;

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  // make list of datums to send to neighbor procs
  // 8 or 26 cells surrounding icell need icell's cdelta info
  // but only if they are owned by a neighbor proc
  // insure icell is only sent once to same neighbor proc
  // also set proclist and locallist for each sent datum

  int nsend = 0;

  for (icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;

    ix = ixyz[icell][0];
    iy = ixyz[icell][1];
    iz = ixyz[icell][2];
    ifirst = nsend;

    // loop over 3x3x3 stencil of neighbor cells centered on icell

    for (jz = -1; jz <= 1; jz++) {
      for (jy = -1; jy <= 1; jy++) {
        for (jx = -1; jx <= 1; jx++) {

          // skip neigh = self

          if (jx == 0 && jy == 0 && jz == 0) continue;

          // check if neighbor cell is within bounds of ablate grid

          if (ix+jx < 1 || ix+jx > nx) continue;
          if (iy+jy < 1 || iy+jy > ny) continue;
          if (iz+jz < 1 || iz+jz > nz) continue;

          // jcell = local index of (jx,jy,jz) neighbor cell of icell

          jcell = walk_to_neigh(icell,jx,jy,jz);

          // add a send list entry of icell to proc != me if haven't already

          proc = cells[jcell].proc;
          if (proc != me) {
            for (j = ifirst; j < nsend; j++)
              if (proc == proclist[j]) break;
            if (j == nsend) {
              if (nsend == maxsend) grow_send();
              proclist[nsend] = proc;
              locallist[nsend++] = cells[icell].id;
            }
          }
        }
      }
    }

    // # of neighbor procs to send icell to

    numsend[icell] = nsend - ifirst;
  }

  // realloc sbuf if necessary
  // ncomm = ilocal + Ncorner values

  // a rate in length/time also needs to know WHICH neighbours hold a piece
  //   of surface, so sync() can tell a neighbour asking for zero from one
  //   with nothing to say.  That is one more double per datum.

  int cflagcomm = (which == CDELTA && unitsflag == DISTANCE);

  int ncomm;
  if (multi_val_flag && which != NVERT) ncomm = 1 + ncorner*nmultiv;
  else ncomm = 1 + ncorner;
  if (cflagcomm) ncomm++;

  if (nsend*ncomm > maxbuf) {
    memory->destroy(sbuf);
    maxbuf = nsend*ncomm;
    memory->create(sbuf,maxbuf,"ablate:sbuf");
  }

  // pack datums to send
  // datum = ilocal of neigh cell on other proc + Ncorner values

  nsend = 0;
  m = 0;

  for (icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;

    n = numsend[icell];
    for (i = 0; i < n; i++) {
      sbuf[m++] = ubuf(locallist[nsend]).d;

      if (which == NVERT) {
        for (j = 0; j < ncorner; j++)
          sbuf[m++] = nvert[icell][j];
      } else {
        if (multi_val_flag) {
          if (which == CDELTA) {
            for (j = 0; j < ncorner; j++)
              for (k = 0; k < nmultiv; k++)
                sbuf[m++] = mdelta[icell][j][k];
          } else if (which == CVALUE) {
            for (j = 0; j < ncorner; j++)
              for (k = 0; k < nmultiv; k++)
               sbuf[m++] = mvalues[icell][j][k];
          }
        } else {
          if (which == CDELTA) {
            for (j = 0; j < ncorner; j++)
              sbuf[m++] = cdelta[icell][j];
            if (cflagcomm) sbuf[m++] = cflag[icell];
          } else if (which == CVALUE) {
            for (j = 0; j < ncorner; j++)
              sbuf[m++] = cvalues[icell][j];
          }
        }
      }

      nsend++;

    }
  }

  // perform irregular neighbor comm
  // Comm class manages rbuf memory

  double *rbuf;
  int nrecv = comm->irregular_uniform_neighs(nsend,proclist,(char *) sbuf,
                                             ncomm*sizeof(double),
                                             (char **) &rbuf);

  // realloc cdelta_ghost if necessary

  if (grid->nghost > maxghost) {
    if (multi_val_flag) {
      memory->destroy(mdelta_ghost);
      maxghost = grid->nghost;
      memory->create(mdelta_ghost,maxghost,ncorner,nmultiv,"ablate:mdelta_ghost");
    } else {
      memory->destroy(cdelta_ghost);
      maxghost = grid->nghost;
      memory->create(cdelta_ghost,maxghost,ncorner,"ablate:cdelta_ghost");
    }

    memory->destroy(cflag_ghost);
    maxghost = grid->nghost;
    memory->create(cflag_ghost,maxghost,"ablate:cflag_ghost");

    memory->destroy(nvert_ghost);
    maxghost = grid->nghost;
    memory->create(nvert_ghost,maxghost,ncorner,"ablate:nvert_ghost");
  }

  // unpack received data into cdelta_ghost = ghost cell corner points

  // NOTE: need to check if hashfilled
  cellint cellID;
  Grid::MyHash *hash = grid->hash;

  m = 0;
  for (i = 0; i < nrecv; i++) {
    cellID = (cellint) ubuf(rbuf[m++]).u;
    ilocal = (*hash)[cellID];
    icell = ilocal - nglocal;
    if (which == NVERT) {
      for (j = 0; j < ncorner; j++)
        nvert_ghost[icell][j] = rbuf[m++];
    } else {
      if (multi_val_flag) {
        for (j = 0; j < ncorner; j++)
          for (k = 0; k < nmultiv; k++)
            mdelta_ghost[icell][j][k] = rbuf[m++];
      } else {
        for (j = 0; j < ncorner; j++)
          cdelta_ghost[icell][j] = rbuf[m++];
        if (cflagcomm) cflag_ghost[icell] = rbuf[m++];
      }
    }

  }
}

/* ----------------------------------------------------------------------
   walk to neighbor of icell, offset by (jx,jy,jz)
   walk first by x, then by y, last by z
   return jcell = local index of neighbor cell
------------------------------------------------------------------------- */

int FixAblate::walk_to_neigh(int icell, int jx, int jy, int jz)
{
  Grid::ChildCell *cells = grid->cells;

  int jcell = icell;

  if (jx < 0) {
    if (grid->neigh_decode(cells[jcell].nmask,XLO) != NCHILD)
      error->one(FLERR,"Fix ablate walk to neighbor cell failed");
    jcell = cells[jcell].neigh[0];
  } else if (jx > 0) {
    if (grid->neigh_decode(cells[jcell].nmask,XHI) != NCHILD)
      error->one(FLERR,"Fix ablate walk to neighbor cell failed");
    jcell = cells[jcell].neigh[1];
  }

  if (jy < 0) {
    if (grid->neigh_decode(cells[jcell].nmask,YLO) != NCHILD)
      error->one(FLERR,"Fix ablate walk to neighbor cell failed");
    jcell = cells[jcell].neigh[2];
  } else if (jy > 0) {
    if (grid->neigh_decode(cells[jcell].nmask,YHI) != NCHILD)
      error->one(FLERR,"Fix ablate walk to neighbor cell failed");
    jcell = cells[jcell].neigh[3];
  }

  if (jz < 0) {
    if (grid->neigh_decode(cells[jcell].nmask,ZLO) != NCHILD)
      error->one(FLERR,"Fix ablate walk to neighbor cell failed");
    jcell = cells[jcell].neigh[4];
  } else if (jz > 0) {
    if (grid->neigh_decode(cells[jcell].nmask,ZHI) != NCHILD)
      error->one(FLERR,"Fix ablate walk to neighbor cell failed");
    jcell = cells[jcell].neigh[5];
  }

  return jcell;
}

/* ----------------------------------------------------------------------
   pack icell values for per-cell arrays into buf
   if icell is a split cell, also pack all sub cell values
   return byte count of amount packed
   if memflag, only return count, do not fill buf
------------------------------------------------------------------------- */

int FixAblate::pack_grid_one(int icell, char *buf, int memflag)
{
  char *ptr = buf;
  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;

  if (!multi_val_flag) {
    if (memflag) memcpy(ptr,cvalues[icell],ncorner*sizeof(double));
    ptr += ncorner*sizeof(double);
  } else {
    for(int j = 0; j < ncorner; j++) {
      if (memflag) memcpy(ptr,mvalues[icell][j],nmultiv*sizeof(double));
      ptr += nmultiv*sizeof(double);
    }
  }

  // the previous-field snapshot has to travel with the cell it belongs to
  // it is what the front speed and the refreshed collision geometry are
  //   measured against, so leaving it behind on a rebalance would pair each
  //   cell's field with some other cell's history

  if (depositflag) {
    if (memflag) memcpy(ptr,cvalues_prev[icell],ncorner*sizeof(double));
    ptr += ncorner*sizeof(double);
  }

  if (tvalues_flag) {
    if (memflag) {
      double *dbuf = (double *) ptr;
      dbuf[0] = tvalues[icell];
    }
    ptr += sizeof(double);
  }

  if (memflag) {
    double *dbuf = (double *) ptr;
    dbuf[0] = ixyz[icell][0];
    dbuf[1] = ixyz[icell][1];
    dbuf[2] = ixyz[icell][2];
  }
  ptr += 3*sizeof(double);

  // DEBUG

  if (memflag) {
    double *dbuf = (double *) ptr;
    dbuf[0] = mcflags[icell][0];
    dbuf[1] = mcflags[icell][1];
    dbuf[2] = mcflags[icell][2];
    dbuf[3] = mcflags[icell][3];
  }
  ptr += 4*sizeof(double);

  if (cells[icell].nsplit > 1) {
    int isplit = cells[icell].isplit;
    int nsplit = cells[icell].nsplit;
    for (int i = 0; i < nsplit; i++) {
      int jcell = sinfo[isplit].csubs[i];

      if (!multi_val_flag) {
        if (memflag) memcpy(ptr,cvalues[jcell],ncorner*sizeof(double));
        ptr += ncorner*sizeof(double);
      } else {
        for(int j = 0; j < ncorner; j++) {
          if (memflag) memcpy(ptr,mvalues[jcell][j],nmultiv*sizeof(double));
          ptr += nmultiv*sizeof(double);
        }
      }

      if (depositflag) {
        if (memflag) memcpy(ptr,cvalues_prev[jcell],ncorner*sizeof(double));
        ptr += ncorner*sizeof(double);
      }

    }
  }

  return ptr-buf;
}

/* ----------------------------------------------------------------------
   unpack icell values for per-cell array from buf
   return byte count of amount unpacked
------------------------------------------------------------------------- */

int FixAblate::unpack_grid_one(int icell, char *buf)
{
  char *ptr = buf;
  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;

  grow_percell(1);

  if (!multi_val_flag) {
    memcpy(cvalues[icell],ptr,ncorner*sizeof(double));
    ptr += ncorner*sizeof(double);
  } else {
    for(int j = 0; j < ncorner; j++) {
      memcpy(mvalues[icell][j],ptr,nmultiv*sizeof(double));
      ptr += nmultiv*sizeof(double);
    }
  }

  if (depositflag) {
    memcpy(cvalues_prev[icell],ptr,ncorner*sizeof(double));
    ptr += ncorner*sizeof(double);
  }

  if (tvalues_flag) {
    double *dbuf = (double *) ptr;
    tvalues[icell] = static_cast<int> (dbuf[0]);
    ptr += sizeof(double);
  }

  double *dbuf = (double *) ptr;
  ixyz[icell][0] = static_cast<int> (dbuf[0]);
  ixyz[icell][1] = static_cast<int> (dbuf[1]);
  ixyz[icell][2] = static_cast<int> (dbuf[2]);
  ptr += 3*sizeof(double);

  dbuf = (double *) ptr;
  mcflags[icell][0] = static_cast<int> (dbuf[0]);
  mcflags[icell][1] = static_cast<int> (dbuf[1]);
  mcflags[icell][2] = static_cast<int> (dbuf[2]);
  mcflags[icell][3] = static_cast<int> (dbuf[3]);
  ptr += 4*sizeof(double);

  nglocal++;

  if (cells[icell].nsplit > 1) {
    int isplit = cells[icell].isplit;
    int nsplit = cells[icell].nsplit;
    grow_percell(nsplit);
    for (int i = 0; i < nsplit; i++) {
      int jcell = sinfo[isplit].csubs[i];

      if (!multi_val_flag) {
        memcpy(cvalues[jcell],ptr,ncorner*sizeof(double));
        ptr += ncorner*sizeof(double);
      } else {
        for(int j = 0; j < ncorner; j++) {
          memcpy(mvalues[jcell][j],ptr,nmultiv*sizeof(double));
          ptr += nmultiv*sizeof(double);
        }
      }

      if (depositflag) {
        memcpy(cvalues_prev[jcell],ptr,ncorner*sizeof(double));
        ptr += ncorner*sizeof(double);
      }

    }
    nglocal += nsplit;
  }

  return ptr-buf;
}

/* ----------------------------------------------------------------------
   copy per-cell info from Icell to Jcell
   called whenever a grid cell is removed from this processor's list
   caller checks that Icell != Jcell
------------------------------------------------------------------------- */

void FixAblate::copy_grid_one(int icell, int jcell)
{
  if (!multi_val_flag) {
    memcpy(cvalues[jcell],cvalues[icell],ncorner*sizeof(double));
  } else {
    for (int j = 0; j < ncorner; j++)
      memcpy(mvalues[jcell][j],mvalues[icell][j],nmultiv*sizeof(double));
  }

  if (depositflag)
    memcpy(cvalues_prev[jcell],cvalues_prev[icell],ncorner*sizeof(double));

  if (tvalues_flag) tvalues[jcell] = tvalues[icell];

  ixyz[jcell][0] = ixyz[icell][0];
  ixyz[jcell][1] = ixyz[icell][1];
  ixyz[jcell][2] = ixyz[icell][2];

  mcflags[jcell][0] = mcflags[icell][0];
  mcflags[jcell][1] = mcflags[icell][1];
  mcflags[jcell][2] = mcflags[icell][2];
  mcflags[jcell][3] = mcflags[icell][3];
}

/* ----------------------------------------------------------------------
   add a grid cell
   called when a grid cell is added to this processor's list
   initialize values to 0.0
------------------------------------------------------------------------- */

void FixAblate::add_grid_one()
{
  grow_percell(1);

  if (!multi_val_flag) {
    for (int i = 0; i < ncorner; i++) cvalues[nglocal][i] = 0.0;
  } else {
    for (int i = 0; i < ncorner; i++)
      for (int j = 0; j < nmultiv; j++)
        mvalues[nglocal][i][j] = 0.0;
  }

  // an empty cell has an empty history, so the front has not moved in it

  if (depositflag)
    for (int i = 0; i < ncorner; i++) cvalues_prev[nglocal][i] = 0.0;

  if (tvalues_flag) tvalues[nglocal] = 0;
  ixyz[nglocal][0] = 0;
  ixyz[nglocal][1] = 0;
  ixyz[nglocal][2] = 0;

  mcflags[nglocal][0] = -1;
  mcflags[nglocal][1] = -1;
  mcflags[nglocal][2] = -1;
  mcflags[nglocal][3] = -1;

  nglocal++;
}

/* ----------------------------------------------------------------------
   reset final grid cell count after grid cell removals
------------------------------------------------------------------------- */

void FixAblate::reset_grid_count(int nlocal)
{
  nglocal = nlocal;
}

/* ----------------------------------------------------------------------
   insure per-cell arrays are allocated long enough for Nnew cells
------------------------------------------------------------------------- */

void FixAblate::grow_percell(int nnew)
{
  if (nglocal+nnew < maxgrid) return;
  if (nnew == 0) maxgrid = nglocal;
  else maxgrid += DELTAGRID;
  if (multi_val_flag) memory->grow(mvalues,maxgrid,ncorner,nmultiv,"ablate:mvalues");
  else memory->grow(cvalues,maxgrid,ncorner,"ablate:cvalues");
  if (depositflag) {
    memory->grow(cvalues_prev,maxgrid,ncorner,"ablate:cvalues_prev");
    memory->grow(sfront_cell,maxgrid,"ablate:sfront_cell");
    memory->grow(sfront_normal,maxgrid,"ablate:sfront_normal");

    // zero the front speeds, since create_surfs() is called once from
    //   store_corners() before front_speed() has ever run

    for (int icell = 0; icell < maxgrid; icell++)
      sfront_cell[icell] = sfront_normal[icell] = 0.0;
  }
  if (tvalues_flag) memory->grow(tvalues,maxgrid,"ablate:tvalues");
  memory->grow(ixyz,maxgrid,3,"ablate:ixyz");
  memory->grow(mcflags,maxgrid,4,"ablate:mcflags");
  memory->grow(celldelta,maxgrid,"ablate:celldelta");
  memory->grow(cflag,maxgrid,"ablate:cflag");

  // zero cflag: comm_neigh_corners() packs it when the units are a distance,
  //   and distance_transform() comms before the first decrement/increment
  //   has ever written it

  for (int icell = 0; icell < maxgrid; icell++) cflag[icell] = 0.0;
  if (multi_val_flag) memory->grow(mdelta,maxgrid,ncorner,nmultiv,"ablate:mdelta");
  else memory->grow(cdelta,maxgrid,ncorner,"ablate:cdelta");
  if (multi_dec_flag) memory->grow(nvert,maxgrid,ncorner,"ablate:nvert");
  memory->grow(numsend,maxgrid,"ablate:numsend");

  array_grid = cvalues;
}

/* ----------------------------------------------------------------------
   reallocate send vectors
------------------------------------------------------------------------- */

void FixAblate::grow_send()
{
  maxsend += DELTASEND;
  memory->grow(proclist,maxsend,"ablate:proclist");
  memory->grow(locallist,maxsend,"ablate:locallist");
}

/* ----------------------------------------------------------------------
   output sum of grid cell corner point values
   assume boundary corner points have value = 0.0
   NOTE: else would have to apply duplication weights to each of 4/8 corner pts
------------------------------------------------------------------------- */

double FixAblate::compute_scalar()
{
  double sum = corner_sum_local();
  double sumall;
  MPI_Allreduce(&sum,&sumall,1,MPI_DOUBLE,MPI_SUM,world);
  return sumall;
}

/* ----------------------------------------------------------------------
   this proc's share of the corner point value total
   each interior corner point is corner 0 of exactly one cell, so summing
     corner 0 and skipping cells on the lo lattice boundaries counts every
     point once, assuming boundary corner points have value = 0.0
------------------------------------------------------------------------- */

double FixAblate::corner_sum_local()
{
  int ix,iy,iz;

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  double sum = 0.0;
  double cavg;
  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;

    ix = ixyz[icell][0];
    iy = ixyz[icell][1];
    iz = ixyz[icell][2];

    if (dim == 2 && (ix == 0 || iy == 0)) continue;
    if (dim == 3 && (ix == 0 || iy == 0 || iz == 0)) continue;

    if (!multi_val_flag) sum += cvalues[icell][0];
    else {
      cavg = 0.0;
      for (int j = 0; j < nmultiv; j++) cavg += mvalues[icell][0][j];
      sum += cavg/nmultiv;
    }

  }

  return sum;
}

/* ----------------------------------------------------------------------
   vector outputs
   1 = last requested decrement/increment budget
   2 = # of deleted inside particles at last ablation
   3-20 = deposition diagnostics, see compute_vector() below and the doc page
   21 = realized front speed over the last interval
   22 = corner point value change actually applied over the last interval
------------------------------------------------------------------------- */

double FixAblate::compute_vector(int i)
{
  if (i == 0) return sum_delta;
  if (i == 1) return 1.0*ndelete;

  // the change the field actually took, next to the requested budget above
  // meaningful in every mode: ablation discards what saturated corner
  //   points cannot absorb the same way deposition does

  if (i == 3+NDEPO) return sum_applied;

  // deposition diagnostics, so the mass and momentum a growing surface
  //   takes out of the gas can be audited against the gas totals
  // the counters accumulate per proc, so they must be summed over all procs
  //   before being output, else only one proc's share is reported
  // reduced once per timestep and cached, since stats invokes this separately
  //   for each requested index; all procs reach it together

  if (!depositflag) return 0.0;

  // the last index is the front speed the surface actually realized over the
  //   last interval, averaged over the cells that hold a front
  // this is the measured answer to what the source was asking for, so a run
  //   driven by a physical rate can be checked against it directly
  // summed as a total and a count so the average is over all procs' cells,
  //   not an average of per-proc averages

  if (depo_stamp != update->ntimestep) {
    double one[NREDUCE];
    one[0] = 1.0*update->nburied;
    one[1] = update->buried_mass;
    one[2] = update->buried_mom[0];
    one[3] = update->buried_mom[1];
    one[4] = update->buried_mom[2];
    one[5] = 1.0*update->nfrontreflect;
    one[6] = update->reflect_mom[0];
    one[7] = update->reflect_mom[1];
    one[8] = update->reflect_mom[2];
    one[9] = update->surf_mom[0];
    one[10] = update->surf_mom[1];
    one[11] = update->surf_mom[2];

    // energy a buried molecule carried into the film, so that the energy
    //   the gas loses to the surface can be audited the same way its mass
    //   and momentum are

    one[12] = update->buried_ke;
    one[13] = update->buried_erot;
    one[14] = update->buried_evib;
    one[15] = update->surf_energy;
    one[16] = update->reflect_energy;
    one[17] = 1.0*update->nfrontmigrate;

    // the last two are not output, they are the sum and count the realized
    //   front speed below is averaged from

    // the average runs over every cell the front passes through, not only the
    //   ones that moved this interval.  With a stochastic rate -- a measured
    //   flux, say -- an interface cell that happened to catch nothing is a
    //   genuine zero of the sample, and dropping it averages over the cells
    //   that did move, reading high by exactly the fraction that did not.

    one[18] = one[19] = 0.0;
    Grid::ChildCell *cells = grid->cells;
    Grid::ChildInfo *cinfo = grid->cinfo;

    for (int icell = 0; icell < nglocal; icell++) {
      if (!(cinfo[icell].mask & groupbit)) continue;
      if (cells[icell].nsplit <= 0) continue;
      // membership is whether the cell held a piece of front at the START of
      //   the interval, which is exactly when its displacement is measurable.
      //   Testing the field as it stands NOW instead dropped two sets of
      //   cells at once every time a flat front reached a plane of corner
      //   points: the ones that had just filled completely, and the ones the
      //   front had just entered.  For a flat front that is every cell there
      //   is, and the realized speed collapsed to zero on those steps.
      // a cell that held a front and did not move still reports 0 and is
      //   still counted, so a stochastic rate is not biased upward.

      if (sfront_normal[icell] == NOMEASURE) continue;
      one[18] += sfront_normal[icell];
      one[19] += 1.0;
    }

    MPI_Allreduce(one,depo_all,NREDUCE,MPI_DOUBLE,MPI_SUM,world);
    depo_stamp = update->ntimestep;
  }

  if (i >= 2 && i < 2+NDEPO) return depo_all[i-2];

  // displacement per regeneration interval, reported as a speed

  if (i == 2+NDEPO) {

    // no cell held a piece of front for the whole interval, so there is no
    //   measurement this time.  A perfectly flat grid-aligned front does this
    //   every time it reaches a plane of corner points: it leaves every cell
    //   it was in at the same moment.  Report the last speed that was
    //   measured rather than a zero the surface never moved at.
    // a front that genuinely stopped is not this case: its cells still hold
    //   it, still measure, and still report zero.

    if (depo_all[NDEPO+1] == 0.0) return front_last;
    front_last = depo_all[NDEPO] / depo_all[NDEPO+1] / (nevery * update->dt);
    return front_last;
  }

  return 0.0;
}

/* ----------------------------------------------------------------------
   process command line args
------------------------------------------------------------------------- */

void FixAblate::process_args(int narg, char **arg)
{
  mindist = 0.0;
  multi_dec_flag = 0;
  minmaxflag = 0;
  mode = ABLATE;
  depositflag = 0;
  unitsflag = CORNER;
  responseflag = RNORMAL;
  checkevery = 1;
  filmrho = 0.0;
  sticking = NULL;
  nsticking = 0;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"mode") == 0)  {
      if (iarg+2 > narg) error->all(FLERR,"Invalid fix ablate command");
      if (strcmp(arg[iarg+1],"ablate") == 0) mode = ABLATE;
      else if (strcmp(arg[iarg+1],"deposit") == 0) mode = DEPOSIT;
      else if (strcmp(arg[iarg+1],"both") == 0) mode = BOTH;
      else error->all(FLERR,"Illegal fix_ablate command");

      // the advancing front machinery is what makes a surface that moves
      //   into the gas safe, and mode = both can do that in any cell, so it
      //   is on for both settings

      depositflag = (mode != ABLATE);
      iarg += 2;
    } else if (strcmp(arg[iarg],"density") == 0)  {

      // mass density of the deposited film, which converts the mass the
      //   surface captures per unit time into a thickness per unit time

      if (iarg+2 > narg) error->all(FLERR,"Invalid fix ablate command");
      filmrho = atof(arg[iarg+1]);
      if (filmrho <= 0.0) error->all(FLERR,"Illegal fix_ablate command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"sticking") == 0)  {

      // one capture probability per column of the flux source, in order.
      //   With the source built on a mixture that has one group per species
      //   that is one coefficient per species, which is where the per species
      //   handling comes from -- fix ablate itself never looks up a mass.
      // an arg is a coefficient only if ALL of it parses as a number, so a
      //   following keyword or a typo like 0.5x ends the list instead of
      //   being read as whatever prefix atof can make of it

      int n = 0;
      char *end;
      while (iarg+1+n < narg) {
        strtod(arg[iarg+1+n],&end);
        if (end == arg[iarg+1+n] || *end != '\0') break;
        n++;
      }
      if (n == 0) error->all(FLERR,"Illegal fix_ablate command");
      nsticking = n;
      memory->destroy(sticking);
      memory->create(sticking,nsticking,"ablate:sticking");
      for (int k = 0; k < n; k++) {
        sticking[k] = atof(arg[iarg+1+k]);
        if (sticking[k] < 0.0 || sticking[k] > 1.0)
          error->all(FLERR,"Fix ablate sticking coefficient must be "
                     "between 0 and 1");
      }
      iarg += 1 + n;
    } else if (strcmp(arg[iarg],"check") == 0)  {

      // how often to validate the surface just built -- watertightness and
      //   grid cell types.  1 = every rebuild, the default and what it has
      //   always done.  0 = never.  N = every Nth.

      if (iarg+2 > narg) error->all(FLERR,"Invalid fix ablate command");
      checkevery = atoi(arg[iarg+1]);
      if (checkevery < 0) error->all(FLERR,"Illegal fix_ablate command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"response") == 0)  {

      // how a rate in length/time becomes a corner point increment.
      // normal: solve for the displacement of the surface along its own
      //   normal, which needs to know which way the surface faces.
      // volume: solve for the volume the surface sweeps, which does not.

      if (iarg+2 > narg) error->all(FLERR,"Invalid fix ablate command");
      if (strcmp(arg[iarg+1],"normal") == 0) responseflag = RNORMAL;
      else if (strcmp(arg[iarg+1],"volume") == 0) responseflag = RVOLUME;
      else error->all(FLERR,"Illegal fix_ablate command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"units") == 0)  {
      if (iarg+2 > narg) error->all(FLERR,"Invalid fix ablate command");
      if (strcmp(arg[iarg+1],"corner") == 0) unitsflag = CORNER;
      else if (strcmp(arg[iarg+1],"distance") == 0) unitsflag = DISTANCE;
      else error->all(FLERR,"Illegal fix_ablate command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"mindist") == 0)  {
      if (iarg+2 > narg) error->all(FLERR,"Invalid read_isurf command");
      mindist = atof(arg[iarg+1]);
      if (mindist < 0.0 || mindist >= 0.5)
        error->all(FLERR,"Fix ablate mindist value must be >= 0.0 and < 0.5");
      mindist = MAX(mindist,EPSILON);
      iarg += 2;
    } else if (strcmp(arg[iarg],"multiple") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Invalid read_isurf command");
      if (strcmp(arg[iarg+1],"no") == 0) multi_dec_flag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) multi_dec_flag = 1;
      else error->all(FLERR,"Illegal fix_ablate command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"minmax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Invalid read_isurf command");
      if (strcmp(arg[iarg+1],"no") == 0) minmaxflag = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) minmaxflag = 1;
      else error->all(FLERR,"Illegal fix_ablate command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix_ablate command");
  }

}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double FixAblate::memory_usage()
{
  double bytes = 0.0;
  if (multi_val_flag) bytes += maxgrid*ncorner*nmultiv * sizeof(double); // mvalues
  else bytes += maxgrid*ncorner * sizeof(double);   // cvalues
  if (tvalues_flag) bytes += maxgrid * sizeof(int);   // tvalues
  bytes += maxgrid*3 * sizeof(int);            // ixyz
  // NOTE: add for mcflags if keep
  bytes += maxgrid * sizeof(double);           // celldelta
  if (multi_val_flag) bytes += maxgrid*ncorner*nmultiv * sizeof(double); // mdelta
  else bytes += maxgrid*ncorner * sizeof(double);   // cdelta
  if (multi_val_flag) bytes += maxgrid*ncorner*nmultiv * sizeof(double); // mdelta_ghost
  else bytes += maxgrid*ncorner * sizeof(double);   // cdelta_ghost
  bytes += 3*maxsend * sizeof(int);            // proclist,locallist,numsend
  bytes += maxbuf * sizeof(double);            // sbuf
  return bytes;
}
