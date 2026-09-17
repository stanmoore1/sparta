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

#include "mpi.h"
#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "ctype.h"
#include <array>
#include <map>
#include <algorithm>
#include "fix_rigid.h"
#include "update.h"
#include "domain.h"
#include "surf.h"
#include "surf_collide.h"
#include "surf_react.h"
#include "surf_react_adsorb.h"
#include "grid.h"
#include "particle.h"
#include "comm.h"
#include "modify.h"
#include "compute.h"
#include "compute_surf.h"
#include "compute_react_surf.h"
#include "fix_emit_surf.h"
#include "input.h"
#include "geometry.h"
#include "cut2d.h"
#include "cut3d.h"
#include "math_extra.h"
#include "math_eigen.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

static constexpr double EPSILON = 1.0e-7;

#define INVOKED_PER_SURF 32
#define MAXLINE 1024
#define EPSSURF 1.0e-4          // same as Grid
#define EPSENCLOSED 1.0e-8      // min enclosed area/volume, relative
#define BIG 1.0e20
#define DELTA_MODIFY 1024

enum{INT,DOUBLE};                      // several files

enum{OUTSIDE,INSIDE,ONSURF2OUT,ONSURF2IN};    // same as Update

// cell types, same as Grid
// renamed to avoid clash with surf collision side enum above

enum{CELLUNKNOWN,CELLOUTSIDE,CELLINSIDE,CELLOVERLAP};

enum{PERIODIC,OUTFLOW,REFLECT,SURFACE,AXISYM};  // same as Domain

enum{CUTCELL,INCREMENTAL};          // remap modes

// reasons incremental_recut() requests a full grid re-map

enum{FALLBACK_NONE,FALLBACK_NOPREV,FALLBACK_SPLIT,FALLBACK_SURFMAX,
     FALLBACK_UNKNOWN};

enum{LINEAR,HERTZ};             // push-off force laws
enum{EULER,RICHARDSON};         // quaternion rotation update schemes
enum{SINGLE,TYPE,CUSTOM};       // body styles

// local box/box overlap test, touching counts as overlap

static inline int box_overlap(double *alo, double *ahi,
                              double *blo, double *bhi)
{
  if (ahi[0] < blo[0] || alo[0] > bhi[0]) return 0;
  if (ahi[1] < blo[1] || alo[1] > bhi[1]) return 0;
  if (ahi[2] < blo[2] || alo[2] > bhi[2]) return 0;
  return 1;
}

/* ---------------------------------------------------------------------- */

FixRigid::FixRigid(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix rigid command");

  scalar_flag = 1;
  global_freq = 1;
  nevery = 1;

  // gridmigrate insures grid_changed() is invoked when grid cells
  // are rebuilt or migrated, so incremental re-cut data can be reset

  gridmigrate = 1;

  if (!surf->exist) error->all(FLERR,"Fix rigid requires surf elements exist");
  kokkosable = 0;
  if (surf->implicit)
    error->all(FLERR,"Fix rigid cannot be used with implicit surfs");

  // all bodies are defined by one fix rigid, so that body-body
  //   contacts and the per-surf body map have a single owner
  // the fix being replaced by a re-definition is already deleted

  for (int ifix = 0; ifix < modify->nfix; ifix++) {
    if (!modify->fix[ifix] || modify->fix[ifix] == this) continue;
    if (fix_rigid_style(modify->fix[ifix]->style))
      error->all(FLERR,"Only one fix rigid command can be defined");
  }

  // in an axisymmetric domain the surf elements are profiles in the
  //   (x,r) half plane which stand for surfaces of revolution about the
  //   x axis, so a rigid body built from them is a body of revolution
  //   and only axial translation and axial spin preserve that symmetry

  axiflag = domain->axisymmetric;

  // for distributed surfs, each proc owns a subset of the surfs;
  // the fix gathers a replicated copy of its (compact) body elements
  //   in gather_body() and maintains local Surf copies of them

  igroup = surf->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Fix rigid surf group ID does not exist");
  groupbit = surf->bitmask[igroup];

  int n = strlen(arg[3]) + 1;
  csurfID = new char[n];
  strcpy(csurfID,arg[3]);

  n = modify->find_compute(csurfID);
  if (n < 0) error->all(FLERR,"Fix rigid compute ID does not exist");

  // bodystyle = how surfs in the group are assigned to bodies

  customname = NULL;
  int iarg = 4;
  if (strcmp(arg[iarg],"single") == 0) {
    bodystyle = SINGLE;
    iarg++;
  } else if (strcmp(arg[iarg],"type") == 0) {
    bodystyle = TYPE;
    iarg++;
  } else if (strcmp(arg[iarg],"custom") == 0) {
    if (iarg+2 > narg) error->all(FLERR,"Illegal fix rigid command");
    bodystyle = CUSTOM;
    n = strlen(arg[iarg+1]) + 1;
    customname = new char[n];
    strcpy(customname,arg[iarg+1]);
    iarg += 2;
  } else error->all(FLERR,"Fix rigid body style not recognized");

  if (iarg >= narg) error->all(FLERR,"Illegal fix rigid command");

  // parse body params

  dim = domain->dimension;
  csurf = NULL;
  infile = NULL;
  slist = NULL;
  displace = NULL;
  bodyneed = NULL;

  bodyflag = 0;
  densityflag = 0;
  density = 0.0;
  forceinfile = 0;

  massone = 0.0;
  for (int j = 0; j < 3; j++)
    xcmone[j] = vcmone[j] = angmomone[j] = 0.0;
  for (int j = 0; j < 6; j++) moione[j] = 0.0;

  if (strcmp(arg[iarg],"body") == 0) {
    if (iarg+22 > narg) error->all(FLERR,"Fix rigid body args not valid");
    bodyflag = 1;
    int massflag,comflag,vcomflag,moiflag,angmomflag;
    massflag = comflag = vcomflag = moiflag = angmomflag = 0;
    int jarg = iarg+1;

    // NVALUE = # of args each keyword consumes, including the keyword
    // a keyword must not read past the 22 args this style is defined to
    //   take, which the check above showed are present

#define BODY_ARGS(nvalue)                                               \
    if (jarg+(nvalue) > iarg+22)                                        \
      error->all(FLERR,"Fix rigid body args not valid");

    while (jarg < iarg+22) {
      if (strcmp(arg[jarg],"mass") == 0) {
        BODY_ARGS(2);
	massflag = 1;
	massone = input->numeric(FLERR,arg[jarg+1]);
	jarg += 2;
      } else if (strcmp(arg[jarg],"com") == 0) {
        BODY_ARGS(4);
	comflag = 1;
	xcmone[0] = input->numeric(FLERR,arg[jarg+1]);
	xcmone[1] = input->numeric(FLERR,arg[jarg+2]);
	xcmone[2] = input->numeric(FLERR,arg[jarg+3]);
	jarg += 4;
      } else if (strcmp(arg[jarg],"moi") == 0) {
        BODY_ARGS(7);
	moiflag = 1;
	moione[0] = input->numeric(FLERR,arg[jarg+1]);
	moione[1] = input->numeric(FLERR,arg[jarg+2]);
	moione[2] = input->numeric(FLERR,arg[jarg+3]);
	moione[3] = input->numeric(FLERR,arg[jarg+4]);
	moione[4] = input->numeric(FLERR,arg[jarg+5]);
	moione[5] = input->numeric(FLERR,arg[jarg+6]);
	jarg += 7;
      } else if (strcmp(arg[jarg],"vcom") == 0) {
        BODY_ARGS(4);
	vcomflag = 1;
	vcmone[0] = input->numeric(FLERR,arg[jarg+1]);
	vcmone[1] = input->numeric(FLERR,arg[jarg+2]);
	vcmone[2] = input->numeric(FLERR,arg[jarg+3]);
	jarg += 4;
      } else if (strcmp(arg[jarg],"angmom") == 0) {
        BODY_ARGS(4);
	angmomflag = 1;
	angmomone[0] = input->numeric(FLERR,arg[jarg+1]);
	angmomone[1] = input->numeric(FLERR,arg[jarg+2]);
	angmomone[2] = input->numeric(FLERR,arg[jarg+3]);
	jarg += 4;
      } else
	error->all(FLERR,"Fix rigid body keyword not recognized");

    }

#undef BODY_ARGS
    if (!massflag || !comflag || !moiflag || !vcomflag || !angmomflag)
      error->all(FLERR,"Fix rigid body args not valid");
    if (massone <= 0.0)
      error->all(FLERR,"Fix rigid body mass must be positive");
    iarg += 22;

  } else if (strcmp(arg[iarg],"infile") == 0) {

    // read in setup_body(), once the bodies are known

    if (iarg+2 > narg) error->all(FLERR,"Fix rigid infile args not valid");
    int n = strlen(arg[iarg+1]) + 1;
    infile = new char[n];
    strcpy(infile,arg[iarg+1]);
    iarg += 2;

  } else if (strcmp(arg[iarg],"density") == 0) {

    // mass, COM, and moi are computed from each body's geometry in
    //   setup_body(); vcom and angmom default to zero and may be
    //   overridden by the optional vcom/angmom keywords below

    if (iarg+2 > narg) error->all(FLERR,"Fix rigid density args not valid");
    densityflag = 1;
    density = input->numeric(FLERR,arg[iarg+1]);
    if (density <= 0.0)
      error->all(FLERR,"Fix rigid body density must be positive");
    iarg += 2;

  } else error->all(FLERR,"Fix rigid define style not recognized");

  // optional args

  outfile = NULL;
  outevery = 0;
  remapmode = INCREMENTAL;
  rotstyle = EULER;
  pushflag = 0;
  pushboundflag = 0;
  pushstyle = LINEAR;
  gammapush = 0.0;
  fext[0] = fext[1] = fext[2] = 0.0;
  int pushstyleflag = 0;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"push") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Fix rigid body args not valid");
      pushflag = 1;
      kpush = input->numeric(FLERR,arg[iarg+1]);
      pushcutoff = input->numeric(FLERR,arg[iarg+2]);
      if (kpush < 0.0 || pushcutoff <= 0.0)
        error->all(FLERR,"Fix rigid body args not valid");
      iarg += 3;
    } else if (strcmp(arg[iarg],"pushbound") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Fix rigid body args not valid");
      if (strcmp(arg[iarg+1],"yes") == 0) pushboundflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) pushboundflag = 0;
      else error->all(FLERR,"Fix rigid body args not valid");
      pushstyleflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"pushstyle") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Fix rigid body args not valid");
      pushstyleflag = 1;
      if (strcmp(arg[iarg+1],"linear") == 0) pushstyle = LINEAR;
      else if (strcmp(arg[iarg+1],"hertz") == 0) pushstyle = HERTZ;
      else error->all(FLERR,"Fix rigid body args not valid");
      iarg += 2;
    } else if (strcmp(arg[iarg],"pushdamp") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Fix rigid body args not valid");
      pushstyleflag = 1;
      gammapush = input->numeric(FLERR,arg[iarg+1]);
      if (gammapush < 0.0)
        error->all(FLERR,"Fix rigid body args not valid");
      iarg += 2;
    } else if (strcmp(arg[iarg],"remap") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Fix rigid body args not valid");
      if (strcmp(arg[iarg+1],"cutcell") == 0) remapmode = CUTCELL;
      else if (strcmp(arg[iarg+1],"incremental") == 0)
        remapmode = INCREMENTAL;
      else error->all(FLERR,"Fix rigid body args not valid");
      iarg += 2;
    } else if (strcmp(arg[iarg],"rotate") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Fix rigid body args not valid");
      if (strcmp(arg[iarg+1],"euler") == 0) rotstyle = EULER;
      else if (strcmp(arg[iarg+1],"richardson") == 0) rotstyle = RICHARDSON;
      else error->all(FLERR,"Fix rigid body args not valid");
      iarg += 2;
    } else if (strcmp(arg[iarg],"force") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Fix rigid body args not valid");
      fext[0] = input->numeric(FLERR,arg[iarg+1]);
      fext[1] = input->numeric(FLERR,arg[iarg+2]);
      fext[2] = input->numeric(FLERR,arg[iarg+3]);
      iarg += 4;
    } else if (strcmp(arg[iarg],"vcom") == 0) {

      // only meaningful for dstyle = density, where vcom defaults to zero
      //   and is not part of the define style's own args

      if (iarg+4 > narg) error->all(FLERR,"Fix rigid body args not valid");
      if (!densityflag)
        error->all(FLERR,"Fix rigid vcom keyword requires density style");
      vcmone[0] = input->numeric(FLERR,arg[iarg+1]);
      vcmone[1] = input->numeric(FLERR,arg[iarg+2]);
      vcmone[2] = input->numeric(FLERR,arg[iarg+3]);
      iarg += 4;
    } else if (strcmp(arg[iarg],"angmom") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Fix rigid body args not valid");
      if (!densityflag)
        error->all(FLERR,"Fix rigid angmom keyword requires density style");
      angmomone[0] = input->numeric(FLERR,arg[iarg+1]);
      angmomone[1] = input->numeric(FLERR,arg[iarg+2]);
      angmomone[2] = input->numeric(FLERR,arg[iarg+3]);
      iarg += 4;
    } else if (strcmp(arg[iarg],"outfile") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Fix rigid body args not valid");
      int n = strlen(arg[iarg+1]) + 1;
      outfile = new char[n];
      strcpy(outfile,arg[iarg+1]);
      outevery = input->inumeric(FLERR,arg[iarg+2]);
      if (outevery <= 0) error->all(FLERR,"Fix rigid body args not valid");
      iarg += 3;
    } else error->all(FLERR,"Fix rigid body args not valid");
  }

  if (pushstyleflag && !pushflag)
    error->all(FLERR,"Fix rigid pushbound, pushstyle, and pushdamp "
               "require push keyword");

  if (dim == 2 && !axiflag && fext[2] != 0.0)
    error->all(FLERR,"Fix rigid z component of force must be zero for 2d");
  if (axiflag && (fext[1] != 0.0 || fext[2] != 0.0))
    error->all(FLERR,"Fix rigid y,z components of force must be zero "
               "for an axisymmetric domain");

  // setup the rigid bodies

  setup_body();

  // global output: a vector for a single body, an array for any count

  if (nbody == 1) {
    vector_flag = 1;
    size_vector = 22;
  }
  array_flag = 1;
  size_array_rows = nbody;
  size_array_cols = 22;

  // irigid = per-surf flags, indexed by local surf index
  // -1 for static surfs, else body index
  // used by Update::build_rigidmap() to detect moving surfs
  // non-distributed only: every proc stores all surfs, length = nlocal
  // for distributed surfs the local surf list changes as the body
  //   moves, so build_rigidmap() maps by global surf ID instead

  irigid = NULL;
  nsurfall = surf->nlocal;

  if (!surf->distributed) {
    int nslocal = surf->nlocal;
    memory->create(irigid,nslocal,"fix_rigid:irigid");
    for (int i = 0; i < nslocal; i++) irigid[i] = -1;
    for (int i = 0; i < nsurf; i++) irigid[slist[i]] = body[i];
  }

  // remap data structs
  // body surfs are cut/split into grid cells by the normal surf
  //   pipeline (done at read_surf time), so no special setup is needed
  //   here beyond the swept-assignment and incremental work buffers

  ndeleted = 0;
  ndeleted_all = 0;
  ndelvalid = -1;

  // elemlo/elemhi are allocated by setup_body() above

  nmodified = maxmodified = 0;
  listschanged = 0;
  splitchanged = 0;
  pending = NULL;
  npending = maxpending = 0;
  insplitrebuild = 0;
  typechanged = 0;
  modified = NULL;
  nsurf_saved = NULL;
  csurfs_saved = NULL;
  cpage = NULL;             // allocated in setup(), needs all bodies' sizes

  pbodyflag = 0;
  noldinside = maxoldinside = 0;
  oldinside = NULL;
  nrcand = maxrcand = 0;
  rcand = NULL;
  newlist = NULL;
  newmap = NULL;
  reclist = NULL;
  maxreclist = 0;
  maxnewlist = 0;
  cut2d = NULL;
  cut3d = NULL;

  pushbinstart = NULL;
  pushbinlist = NULL;
  pushstamp = NULL;
  pushstampcur = 0;
  bodybinstart = NULL;
  bodybinlist = NULL;
  bodycand = NULL;
  maxbodycand = 0;
  copiesappended = 0;
  ftbuf_mine = ftbuf_all = NULL;
  warnfallback = 0;
  warndelete = 0;
  ndelrun = 0;

  swstamp = NULL;
  swhead = NULL;
  maxswcell = 0;
  swcur = 0;
  swcells = NULL;
  nswcell = maxswcells = 0;
  entnext = NULL;
  entelem = NULL;
  nent = maxent = 0;

  // for incremental mode: cutters and work bufs for re-cutting cells

  // work bufs are sized per run in setup(), since global surfmax can be
  //   changed between runs, after this fix is defined

  if (remapmode == INCREMENTAL) {
    if (dim == 2) cut2d = new Cut2d(sparta,axiflag);
    else cut3d = new Cut3d(sparta);
  }
}

/* ---------------------------------------------------------------------- */

FixRigid::~FixRigid()
{
  // a KOKKOS functor copy of this fix (fix rigid/kk passes *this to a
  //   parallel_reduce) shares these pointers with the original and must
  //   not free them when the copy goes out of scope

  if (copy || copymode) return;

  delete [] csurfID;
  delete [] customname;
  delete [] infile;
  delete [] outfile;

  memory->destroy(xcm);
  memory->destroy(vcm);
  memory->destroy(omega);
  memory->destroy(invmass);
  memory->destroy(invinertia);
  memory->destroy(quat);
  memory->destroy(xcmnew);
  memory->destroy(quatnew);
  memory->destroy(xcmmid);
  memory->destroy(massbody);
  memory->destroy(moi);
  memory->destroy(inertia);
  memory->destroy(angmom);
  memory->destroy(ex_space);
  memory->destroy(ey_space);
  memory->destroy(ez_space);
  memory->destroy(fcm);
  memory->destroy(torque);
  memory->destroy(fpush);
  memory->destroy(tqpush);
  memory->destroy(rmaxbody);
  memory->destroy(rminbody);
  memory->destroy(bbodylo);
  memory->destroy(bbodyhi);
  memory->destroy(bboxeps);
  memory->destroy(pbodylo);
  memory->destroy(pbodyhi);
  memory->destroy(fcm_infile);
  memory->destroy(torque_infile);
  memory->destroy(body);
  memory->destroy(bodystart);

  memory->destroy(slist);
  memory->destroy(displace);
  memory->destroy(irigid);
  memory->destroy(elemlo);
  memory->destroy(elemhi);
  memory->destroy(bodypt);
  memory->destroy(bodynorm);
  memory->destroy(sids);
  memory->destroy(bodymask);
  memory->destroy(bodytype);
  memory->destroy(bodytrans);
  memory->destroy(bodyisc);
  memory->destroy(bodyisr);
  memory->destroy(lblist);
  memory->destroy(copy_index);
  memory->destroy(copy_elem);
  memory->destroy(olist_own);
  memory->destroy(olist_elem);
  memory->destroy(modified);
  memory->destroy(nsurf_saved);
  memory->sfree(csurfs_saved);
  delete cpage;

  // registry csurfs lists may still be installed in live grid cells,
  //   e.g. this fix is unfixed between runs after incremental re-cuts
  // copy each such list into grid-owned page storage before freeing it
  // Modify is destroyed before Grid at program teardown, so grid is valid

  copy_registry_to_grid();
  free_registry();
  copy_split_registry_to_grid();
  free_split_registry();
  for (int m = 0; m < maxpending; m++) {
    memory->destroy(pending[m].map);
    memory->destroy(pending[m].vols);
  }
  memory->sfree(pending);
  memory->destroy(oldinside);
  memory->destroy(rcand);
  memory->destroy(newlist);
  memory->destroy(newmap);
  memory->destroy(reclist);
  delete cut2d;
  delete cut3d;

  memory->destroy(pushbinstart);
  memory->destroy(pushbinlist);
  memory->destroy(pushstamp);
  memory->destroy(bodybinstart);
  memory->destroy(bodybinlist);
  memory->destroy(bodycand);
  memory->destroy(bodyneed);
  memory->destroy(ftbuf_mine);
  memory->destroy(ftbuf_all);
  memory->destroy(swstamp);
  memory->destroy(swhead);
  memory->destroy(swcells);
  memory->destroy(entnext);
  memory->destroy(entelem);
}

/* ---------------------------------------------------------------------- */

int FixRigid::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRigid::init()
{
  // check that global rigid flag is set

  if (update->rigidflag == 0)
    error->all(FLERR,"Cannot use fix rigid unless global rigid is set");

  // with the KOKKOS package active, only the rigid/kk variant brackets
  //   its host-side work with the required device transfers

  if (sparta->kokkos && !kokkosable)
    error->all(FLERR,"Must use fix rigid/kk with KOKKOS");

  // the recoil correction of collisions is exact down to a body as
  //   light as one simulation particle; below that the exact result
  //   would carry the particle past the wall within the step, which the
  //   one-step coupling cannot represent, so the correction is inactive

  double mmax = 0.0;
  for (int i = 0; i < particle->nspecies; i++)
    mmax = MAX(mmax,particle->species[i].mass);
  double mmin = massbody[0];
  for (int ibody = 1; ibody < nbody; ibody++)
    mmin = MIN(mmin,massbody[ibody]);
  if (mmin < update->fnum*mmax && comm->me == 0)
    error->warning(FLERR,"Fix rigid body mass is less than the mass of a "
                   "simulation particle, collisions are not corrected "
                   "for body recoil");

  // check that specified compute is valid for use with fix rigid
  // the compute tallies torque about the COM of the body each surf
  //   belongs to, which it reads from this fix (com rigid)

  int n = modify->find_compute(csurfID);
  if (n < 0) error->all(FLERR,"Could not find fix rigid compute ID");
  if (strcmp(modify->compute[n]->style,"surf") != 0 &&
      strcmp(modify->compute[n]->style,"surf/kk") != 0)
    error->all(FLERR,"Fix rigid compute is not style surf");
  csurf = (ComputeSurf *) modify->compute[n];
  if (csurf->per_surf_flag == 0)
    error->all(FLERR,"Fix rigid compute does not compute per-surf info");
  if (csurf->size_per_surf_cols != 6 || !csurf->force_torque_colcheck())
    error->all(FLERR,"Fix rigid compute must tally exactly "
               "fx fy fz tx ty tz for a single group");
  if (!csurf->com_rigid())
    error->all(FLERR,"Fix rigid compute surf must use com rigid");
  if (!csurf->mixture_covers_all_species())
    error->all(FLERR,"Fix rigid compute surf mixture must contain "
               "all species");

  // insure the compute tallies on the first step of the next run
  // end_of_step() extends this to every step of the run

  csurf->addstep(update->ntimestep+1);

  // body surfs cannot be transparent, and may only carry a surf_react
  //   model in which every reaction leaves exactly one particle: the
  //   body's recoil correction maps one incoming particle to one
  //   outgoing one, so a reaction which destroys the particle, produces
  //   a second, or adsorbs it onto the surface has no defined recoil
  //   and would also change the body's mass, which this fix holds fixed
  // all body surfs must be in the surf group tallied by the compute
  // attributes come from the replicated body table, valid for both
  //   non-distributed and distributed surfs

  // surfs cannot change once a fix rigid is defined:
  //   removal invalidates the body element table; a change to the
  //   body group invalidates the body definition
  // surfs appended after the fix was defined are allowed: grow irigid
  //   and flag them static
  // Update::init() clamps its rigidmap scan to nsurfall, so it is
  //   correct even though it runs before this method

  if (surf->count_group(igroup) != ngroupsurf)
    error->all(FLERR,"Fix rigid body surf group was changed "
               "after fix rigid was defined");

  if (!surf->distributed) {
    if (surf->nlocal < nsurfall)
      error->all(FLERR,"Surfs were removed after fix rigid was defined");
    if (surf->nlocal > nsurfall) {
      memory->grow(irigid,surf->nlocal,"fix_rigid:irigid");
      for (int i = nsurfall; i < surf->nlocal; i++) irigid[i] = -1;
      nsurfall = surf->nlocal;
    }
  }

  // the replicated table of body surf attributes was captured when the
  //   fix was defined: refresh it from the current surfs, or with
  //   distributed surfs, where the local copies of body surfs are built
  //   from it, error if the attributes were changed since, e.g. by a
  //   surf_modify command issued after this fix

  check_body_attributes();

  int cbit = csurf->surf_groupbit();

  for (int i = 0; i < nsurf; i++) {
    if (bodytrans[i])
      error->all(FLERR,"Fix rigid body surfs cannot be transparent");
    if (bodyisr[i] >= 0 && !surf->sr[bodyisr[i]]->one_product_only())
      error->all(FLERR,"Fix rigid body surfs can only use a surf_react "
                 "model in which every reaction leaves exactly one "
                 "particle");
    if (!(bodymask[i] & cbit))
      error->all(FLERR,"Fix rigid compute surf group does not include "
                 "all body surfs");
  }

  // the body surfs follow the motion of the body, so their collision
  //   model cannot impose a wall motion of its own

  for (int i = 0; i < nsurf; i++) {
    if (bodyisc[i] < 0) continue;
    SurfCollide *sc = surf->sc[bodyisc[i]];
    if (sc->moving_wall())
      error->all(FLERR,"Fix rigid body surfs cannot use a surf_collide "
                 "model with its own wall motion");
  }

  // the grid may be re-mapped on any step (remap cutcell: every step,
  //   incremental: any step it falls back), which surf_react adsorb
  //   with per-surf state and distributed surfs only allows on its
  //   synchronization steps

  if (surf->distributed)
    for (int i = 0; i < surf->nsr; i++)
      if (strncmp(surf->sr[i]->style,"adsorb",6) == 0) {
        SurfReactAdsorb *sra = (SurfReactAdsorb *) surf->sr[i];
        if (sra->surf_mode() && sra->sync_every() > 1)
          error->all(FLERR,"Fix rigid with distributed surfs requires "
                     "surf_react adsorb nsync = 1");
      }

  // distributed surfs: Update::init_rigid() may have appended local
  //   copies of body surfs before the surface reaction models init'd,
  //   so their per-surf state was sized for the final local+ghost surf
  //   arrays; now that they have init'd, tell them the arrays changed
  //   (as after a load balance) so per-surf state is re-spread
  // done here, since the fix inits after the models

  if (update->rigid_notify_sr) {
    for (int i = 0; i < surf->nsr; i++) surf->sr[i]->grid_changed();
    update->rigid_notify_sr = 0;
  }

  // fix emit/surf builds its per-cell emission tasks from the surf
  //   positions at setup, which do not follow the body motion

  for (int ifix = 0; ifix < modify->nfix; ifix++) {
    if (strncmp(modify->fix[ifix]->style,"emit/surf",9) != 0) continue;
    int ebit = ((FixEmitSurf *) modify->fix[ifix])->surf_groupbit();
    for (int i = 0; i < nsurf; i++)
      if (bodymask[i] & ebit)
        error->all(FLERR,"Fix emit/surf cannot emit from fix rigid "
                   "body surfs");
  }

  // smallest grid cell edge length, for motion-rate warnings

  Grid::ChildCell *cells = grid->cells;
  int nglocal = grid->nlocal;

  double mine = BIG;
  for (int icell = 0; icell < nglocal; icell++) {
    if (cells[icell].nsplit <= 0) continue;
    mine = MIN(mine,cells[icell].hi[0]-cells[icell].lo[0]);
    mine = MIN(mine,cells[icell].hi[1]-cells[icell].lo[1]);
    if (dim == 3) mine = MIN(mine,cells[icell].hi[2]-cells[icell].lo[2]);
  }
  MPI_Allreduce(&mine,&mincellsize,1,MPI_DOUBLE,MPI_MIN,world);

  // re-enable single-shot warnings for this run

  warnrotate = warntranslate = warnexit = warnfallback = 0;
  warndelete = 0;
  ndelrun = 0;

  // fix rigid must be defined before fixes which change the grid,
  // so its end_of_step() restores overlaid grid cells before they run

  int myindex = modify->find_fix(id);
  for (int ifix = 0; ifix < myindex; ifix++)
    if (strncmp(modify->fix[ifix]->style,"balance",7) == 0 ||
        strncmp(modify->fix[ifix]->style,"adapt",5) == 0 ||
        strncmp(modify->fix[ifix]->style,"move/surf",9) == 0)
      error->all(FLERR,"Fix rigid must be defined before fix balance, "
                 "fix adapt, or fix move/surf");

}

/* ----------------------------------------------------------------------
   called at start of each run, after grid and particles are setup
------------------------------------------------------------------------- */

void FixRigid::setup()
{
  // work buffer for the fused force/torque Allreduce over all bodies

  if (!ftbuf_mine) {
    memory->create(ftbuf_mine,6*nbody,"fix_rigid:ftbuf_mine");
    memory->create(ftbuf_all,6*nbody,"fix_rigid:ftbuf_all");
  }

  // page for merged csurfs lists built by swept_assign_all each step
  // a merged list can hold one cell's current surfs plus the swept
  //   surfs of every body
  // maxsurfpercell can change between runs, so (re)allocate

  delete cpage;
  int maxchunk = grid->maxsurfpercell + nsurf;
  cpage = new MyPage<surfint>(maxchunk,MAX(65536,4*maxchunk));
  if (cpage->errorflag)
    error->all(FLERR,"Fix rigid could not allocate collision-list page");

  // bbox around each body's elements at their current positions,
  //   and the bins of bodies by COM the queries below use

  for (int ibody = 0; ibody < nbody; ibody++) body_bbox(ibody,0);
  body_bins();

  // distributed surfs: insure local copies of the body surfs which
  //   can reach this proc's cells exist and rigidmap covers them,
  //   before the first step's swept assignment
  // rigidmap also spans the ghost surfs acquired for this run, and
  //   per-surf computes must size their arrays for any appended copies

  if (surf->distributed) {
    proc_bbox();
    int changed = ensure_local_copies();
    update->build_rigidmap();
    surfs_changed(changed,2);
    if (changed) listschanged = 1;
  }

  // work bufs for incremental re-cutting of one cell, sized for the
  //   current global surfmax, which can change between runs:
  // newlist/newmap = the cell's new surf list and its split map, both
  //   capped at maxsurfpercell by the cut routines
  // reclist = candidate surfs, the cell's current static surfs plus
  //   every element of every body

  if (remapmode == INCREMENTAL) {
    if (grid->maxsurfpercell > maxnewlist) {
      maxnewlist = grid->maxsurfpercell;
      memory->destroy(newlist);
      memory->destroy(newmap);
      memory->create(newlist,maxnewlist,"fix_rigid:newlist");
      memory->create(newmap,maxnewlist,"fix_rigid:newmap");
    }

    int n = grid->maxsurfpercell + nsurf;
    if (n > maxreclist) {
      maxreclist = n;
      memory->destroy(reclist);
      memory->create(reclist,maxreclist,"fix_rigid:reclist");
    }
  }

  // bin static surfs for push-off candidate pruning

  if (pushflag) push_bins();

  // delete any particles inside a body
  // create_particles marks the bodies' cells INSIDE via the surf pipeline
  //   and normally avoids them, but this is a safety net for any that
  //   end up inside, e.g. via an emit region overlapping a body

  if (particle->exist) {
    particles_to_host();
    ndeleted += remove_inside_particles(0);
  }

  // for incremental remap: grid state is now consistent with the
  //   bodies at their current positions

  if (remapmode == INCREMENTAL) {
    for (int ibody = 0; ibody < nbody; ibody++)
      for (int j = 0; j < 3; j++) {
        pbodylo[ibody][j] = bbodylo[ibody][j];
        pbodyhi[ibody][j] = bbodyhi[ibody][j];
      }
    pbodyflag = 1;
  }
}

/* ---------------------------------------------------------------------- */

void FixRigid::start_of_step()
{
  // csurf is set by init(): a fix defined after the last init (e.g.
  //   re-defined before a "run pre no") has no body state to advance

  if (!csurf)
    error->all(FLERR,"Fix rigid was not initialized before the run");

  double dt = update->dt;
  double dthalf = 0.5 * dt;

  for (int ibody = 0; ibody < nbody; ibody++) {
    double *xcm1 = xcm[ibody];
    double *vcm1 = vcm[ibody];
    double *omega1 = omega[ibody];
    double *quat1 = quat[ibody];
    double *xcmnew1 = xcmnew[ibody];
    double *quatnew1 = quatnew[ibody];
    double *angmom1 = angmom[ibody];
    double *fcm1 = fcm[ibody];
    double *torque1 = torque[ibody];

    // body inverse mass and inertia for collision recoil this step,
    //   from the start-of-step axes before they are advanced below

    set_recoil(ibody);

    // time integrate from current position to end-of-step position
    // velocity Verlet: this is the first half kick and the drift,
    //   the second half kick is applied in end_of_step() once the force
    //   and torque of this step are known
    // fcm,torque = from particle collisions and push-off contacts during
    //   the previous step, i.e. the force at the start of this step,
    //   plus the constant external force
    // vcm/angmom/omega are thus half-step values during the step: the
    //   body moves, and particles collide with it, at these velocities,
    //   which is second-order accurate and exact for a constant force
    // xcmnew/quatnew/exyz_space = end-of-step values

    double dtfhalf = 0.5 * dt / massbody[ibody];

    vcm1[0] += dtfhalf * (fcm1[0] + fext[0]);
    vcm1[1] += dtfhalf * (fcm1[1] + fext[1]);
    vcm1[2] += dtfhalf * (fcm1[2] + fext[2]);

    // drift xcm by full step with the half-step velocity
    // store as xcmnew so have start/stop position for this timestep

    xcmnew1[0] = xcm1[0] + dt * vcm1[0];
    xcmnew1[1] = xcm1[1] + dt * vcm1[1];
    xcmnew1[2] = xcm1[2] + dt * vcm1[2];

    // mid-step COM, the time-average of the COM the particles collide
    //   about during the step; compute surf tallies torques about it

    xcmmid[ibody][0] = 0.5 * (xcm1[0] + xcmnew1[0]);
    xcmmid[ibody][1] = 0.5 * (xcm1[1] + xcmnew1[1]);
    xcmmid[ibody][2] = 0.5 * (xcm1[2] + xcmnew1[2]);

    // half kick of angular momentum in spatial frame

    angmom1[0] += dthalf * torque1[0];
    angmom1[1] += dthalf * torque1[1];
    angmom1[2] += dthalf * torque1[2];

    // compute new omega from new angmom, both in spatial frame

    MathExtra::angmom_to_omega(angmom1,ex_space[ibody],ey_space[ibody],
                               ez_space[ibody],inertia[ibody],omega1);

    // for 2d, insure COM stays in plane and rotation is about z axis
    // guards against small numeric drift in principal axes

    if (dim == 2 && !axiflag) {
      xcmnew1[2] = 0.0;
      omega1[0] = 0.0;
      omega1[1] = 0.0;
    }

    // update quaternion by full step using new omega in spatial frame
    // store as quatnew so have start/stop orientation for this timestep
    // rotate euler (default): omega is held constant over the step, so
    //   dq/dt = 1/2 omega q integrates exactly to the rotation by
    //   angle |omega|*dt about omega; the moving-surf collision tests
    //   assume this same rotation, so the end-of-step geometry the
    //   particles were reflected from is exactly the one installed
    // rotate richardson: LAMMPS-style Richardson iteration which
    //   re-evaluates omega at the half step from the (constant over the
    //   step) angular momentum; useful for rotation-dominated bodies

    // axisymmetric: the body can only spin about its own axis, which
    //   maps the surface of revolution onto itself and so moves no
    //   geometry.  integrating the quaternion would instead rotate each
    //   profile point out of the (x,r) plane by omega_x*dt, which is
    //   not the same body.  so the orientation is held at the identity
    //   and only the spin rate in omega[0] is carried forward, where
    //   the mover reads it as the azimuthal wall velocity omega_x * r

    if (axiflag) {
      quatnew1[0] = 1.0;
      quatnew1[1] = quatnew1[2] = quatnew1[3] = 0.0;

    } else if (rotstyle == RICHARDSON) {
      quatnew1[0] = quat1[0];
      quatnew1[1] = quat1[1];
      quatnew1[2] = quat1[2];
      quatnew1[3] = quat1[3];
      MathExtra::richardson(quatnew1,angmom1,omega1,inertia[ibody],dthalf);
    } else {
      double wmag = MathExtra::len3(omega1);
      if (wmag > 0.0) {
        double axis[3],qrot[4];
        axis[0] = omega1[0]/wmag;
        axis[1] = omega1[1]/wmag;
        axis[2] = omega1[2]/wmag;
        MathExtra::axisangle_to_quat(axis,wmag*dt,qrot);
        MathExtra::quatquat(qrot,quat1,quatnew1);
        MathExtra::qnormalize(quatnew1);
      } else {
        quatnew1[0] = quat1[0];
        quatnew1[1] = quat1[1];
        quatnew1[2] = quat1[2];
        quatnew1[3] = quat1[3];
      }
    }
    MathExtra::q_to_exyz(quatnew1,ex_space[ibody],ey_space[ibody],
                         ez_space[ibody]);

    // hard error if the body state has stopped being a finite number
    // the pose computed here is handed to the surf coords, the moving
    //   collision tests, and the cut/split geometry; an Inf or NaN
    //   propagates into all of them and crashes inside the cut instead
    //   of failing cleanly, so stop at the source
    // the state is replicated on every proc, so the test is collective
    // the usual cause is a body mass at or below the mass one
    //   computational particle carries (fnum times the species mass),
    //   which lets a single gas collision accelerate the body without
    //   bound; a push stiffness too large for the timestep does it too

    if (!isfinite(xcmnew1[0]) || !isfinite(xcmnew1[1]) ||
        !isfinite(xcmnew1[2]) ||
        !isfinite(vcm1[0]) || !isfinite(vcm1[1]) || !isfinite(vcm1[2]) ||
        !isfinite(omega1[0]) || !isfinite(omega1[1]) ||
        !isfinite(omega1[2]) ||
        !isfinite(quatnew1[0]) || !isfinite(quatnew1[1]) ||
        !isfinite(quatnew1[2]) || !isfinite(quatnew1[3])) {
      char str[128];
      sprintf(str,"Fix rigid body %d position, velocity, or rotation is "
              "no longer a finite number",ibody+1);
      error->all(FLERR,str);
    }

    // warn once per run if body motion in a single step is too large
    // rotation > 0.1 radian degrades the collision test for particles
    //   hitting rotating surfs: the hit time is exact, but which element
    //   is hit comes from the chord through the mapped path endpoints
    //   (see Geometry::refine_moving_param)
    // max surf pt displacement > smallest grid cell degrades the
    //   accuracy of surf assignment to grid cells for cutcell remapping

    // neither warning applies to the spin of an axisymmetric body: it
    //   maps the surface onto itself, so it displaces no surf point and
    //   the collision test for it is exact at any spin rate

    if (!warnrotate && !axiflag && MathExtra::len3(omega1)*dt > 0.1) {
      warnrotate = 1;
      if (comm->me == 0)
        error->warning(FLERR,"Fix rigid body rotation per timestep exceeds "
                       "0.1 radian, collision accuracy degrades");
    }

    if (!warntranslate) {
      double dispmax = MathExtra::len3(vcm1) * dt;
      if (!axiflag) dispmax += MathExtra::len3(omega1)*rmaxbody[ibody] * dt;
      if (dispmax > mincellsize) {
        warntranslate = 1;
        if (comm->me == 0)
          error->warning(FLERR,"Fix rigid body moves more than a grid cell "
                         "per timestep, cell assignment accuracy degrades");
      }
    }
  }

  // per-element swept bounding boxes of every body for this step

  for (int ibody = 0; ibody < nbody; ibody++) body_bbox(ibody,1);

  // distributed surfs: a body sweeping into this proc's cells for the
  //   first time since the last grid rebuild needs local copies of its
  //   surfs here, before the swept lists and the mover reference them
  // collective: surfs_changed() reduces whether any proc appended

  if (surf->distributed) {
    int changed = ensure_local_copies();
    if (changed) {
      update->build_rigidmap();
      listschanged = 1;
      copiesappended = 1;
    }
    surfs_changed(changed,0);
  }

  // augment collision lists of all cells any body sweeps through during
  //   the step, so particles in the swept paths are tested against the
  //   moving surfs and reflected rather than overtaken and later deleted

  swept_assign_all();
}

/* ---------------------------------------------------------------------- */

void FixRigid::end_of_step()
{
  int i,j,k,ibody;

  // undo the swept collision-list augmentation from start_of_step

  swept_restore();

  // sum per-surf force/torque to each body's fcm/torque
  // read the compute's RAW local tally rows: values are fully
  //   normalized at tally time, and each row's surf ID maps to a
  //   body element via the ID table, so a local sum plus the
  //   single fused Allreduce below is exactly the collated result
  // this avoids Surf::collate_array entirely, whose reduce path is
  //   an Allreduce over ALL global surfs per compute per step, and
  //   avoids any scan over the surf list: cost is O(local tallies)
  // identical for non-distributed and distributed surfs

  for (i = 0; i < 6*nbody; i++) ftbuf_mine[i] = 0.0;

  if (!(csurf->invoked_flag & INVOKED_PER_SURF)) {
    csurf->compute_per_surf();
    csurf->invoked_flag |= INVOKED_PER_SURF;
  }

  surfint *t2s;
  int ntally = csurf->tallyinfo(t2s);
  double **tally = csurf->tally_array();

  for (i = 0; i < ntally; i++) {
    k = body_elem(t2s[i]);
    if (k < 0) continue;
    double *ft = &ftbuf_mine[6*body[k]];
    for (j = 0; j < 6; j++) ft[j] += tally[i][j];
  }

  // insure the compute tallies on the next step

  csurf->addstep(update->ntimestep+1);

  MPI_Allreduce(ftbuf_mine,ftbuf_all,6*nbody,MPI_DOUBLE,MPI_SUM,world);

  for (ibody = 0; ibody < nbody; ibody++) {
    fcm[ibody][0] = ftbuf_all[6*ibody];
    fcm[ibody][1] = ftbuf_all[6*ibody+1];
    fcm[ibody][2] = ftbuf_all[6*ibody+2];
    torque[ibody][0] = ftbuf_all[6*ibody+3];
    torque[ibody][1] = ftbuf_all[6*ibody+4];
    torque[ibody][2] = ftbuf_all[6*ibody+5];
    axi_project(fcm[ibody],torque[ibody]);
  }

  // for incremental remap: record cells interior to the bodies
  //   before their surfs move to their end-of-step positions

  if (remapmode == INCREMENTAL) record_oldinside();

  double z[3],delta[3],delta12[3],delta13[3];
  z[0] = 0.0; z[1] = 0.0; z[2] = 1.0;

  for (ibody = 0; ibody < nbody; ibody++) {
    double *xcm1 = xcm[ibody];
    double *quat1 = quat[ibody];

    // reset xcm/quat to new xcm/quat calculated in start_of_step()

    xcm1[0] = xcmnew[ibody][0];
    xcm1[1] = xcmnew[ibody][1];
    xcm1[2] = xcmnew[ibody][2];

    quat1[0] = quatnew[ibody][0];
    quat1[1] = quatnew[ibody][1];
    quat1[2] = quatnew[ibody][2];
    quat1[3] = quatnew[ibody][3];

    // enforce the body's degrees of freedom on all its properties
    // 2d: in-plane motion and rotation about z.  start_of_step()
    //   enforces it on xcmnew and omega; quat stays a rotation about z
    //   since omega is along z
    // axisymmetric: translation along x and spin about x.  axi_project()
    //   already removed the transverse force and torque, so this only
    //   guards against drift; quat is pinned to the identity in
    //   start_of_step()

    if (axiflag) {
      xcm1[1] = xcm1[2] = 0.0;
      vcm[ibody][1] = vcm[ibody][2] = 0.0;
      fcm[ibody][1] = fcm[ibody][2] = 0.0;
      torque[ibody][1] = torque[ibody][2] = 0.0;
      angmom[ibody][1] = angmom[ibody][2] = 0.0;
      omega[ibody][1] = omega[ibody][2] = 0.0;

    } else if (dim == 2) {
      xcm1[2] = 0.0;
      vcm[ibody][2] = 0.0;
      fcm[ibody][2] = 0.0;
      torque[ibody][0] = 0.0;
      torque[ibody][1] = 0.0;
      angmom[ibody][0] = 0.0;
      angmom[ibody][1] = 0.0;
      omega[ibody][0] = 0.0;
      omega[ibody][1] = 0.0;
    }

    // regenerate the replicated body geometry from the new pose:
    //   corner pts from displace rotated to the space frame + new COM,
    //   normals recomputed from the corner pts
    // then write it into the Surf copies the mover and cut pipeline read
    // matvec() converts displace vector from body frame to space frame

    for (i = bodystart[ibody]; i < bodystart[ibody+1]; i++) {
      for (j = 0; j < dim; j++) {

        // axisymmetric: the body frame never rotates and the COM stays
        //   on the axis, so the pose map is a shift along x and nothing
        //   else.  doing it as a shift rather than through the (identity)
        //   rotation keeps the radial coordinate bitwise unchanged, so a
        //   profile point on the axis stays exactly at r = 0 for the
        //   whole run.  the cut-cell routines and the watertight check
        //   both compare against r = 0 exactly

        if (axiflag) {
          bodypt[i][j][0] = xcm1[0] + displace[i][j][0];
          bodypt[i][j][1] = displace[i][j][1];
          bodypt[i][j][2] = 0.0;
          continue;
        }

        MathExtra::matvec(ex_space[ibody],ey_space[ibody],ez_space[ibody],
                          displace[i][j],delta);
        if (dim == 2) delta[2] = 0.0;
        MathExtra::add3(xcm1,delta,bodypt[i][j]);
      }

      if (dim == 2) {
        MathExtra::sub3(bodypt[i][1],bodypt[i][0],delta);
        MathExtra::cross3(z,delta,bodynorm[i]);
        MathExtra::norm3(bodynorm[i]);
        bodynorm[i][2] = 0.0;
      } else {
        MathExtra::sub3(bodypt[i][1],bodypt[i][0],delta12);
        MathExtra::sub3(bodypt[i][2],bodypt[i][0],delta13);
        MathExtra::cross3(delta12,delta13,bodynorm[i]);
        MathExtra::norm3(bodynorm[i]);
      }
    }
  }

  update_surf_copies();

  // bbox around each body's elements at their new positions,
  //   and the bins of bodies by COM the queries below use

  for (ibody = 0; ibody < nbody; ibody++) body_bbox(ibody,0);
  body_bins();

  // error if a body now extends beyond a periodic boundary,
  //   b/c body coords are not wrapped across periodic boundaries
  // a body is allowed to exit thru non-periodic boundaries
  // test the true body extent, not the eps-inflated bbox

  double *boxlo = domain->boxlo;
  double *boxhi = domain->boxhi;
  int *bflag = domain->bflag;

  for (ibody = 0; ibody < nbody; ibody++) {
    double *blo = bbodylo[ibody];
    double *bhi = bbodyhi[ibody];
    double eps = bboxeps[ibody];

    // warn once per run if a body is entirely outside the simulation
    //   box, b/c it no longer interacts with any particles

    if (!warnexit) {
      if (bhi[0] < boxlo[0] || blo[0] > boxhi[0] ||
          bhi[1] < boxlo[1] || blo[1] > boxhi[1] ||
          (dim == 3 && (bhi[2] < boxlo[2] || blo[2] > boxhi[2]))) {
        warnexit = 1;
        if (comm->me == 0)
          error->warning(FLERR,"Fix rigid body has exited the simulation "
                         "box and no longer interacts with particles");
      }
    }

    int outflag = 0;
    if (bflag[0] == PERIODIC && blo[0]+eps < boxlo[0]) outflag = 1;
    if (bflag[1] == PERIODIC && bhi[0]-eps > boxhi[0]) outflag = 1;
    if (bflag[2] == PERIODIC && blo[1]+eps < boxlo[1]) outflag = 1;
    if (bflag[3] == PERIODIC && bhi[1]-eps > boxhi[1]) outflag = 1;
    if (dim == 3) {
      if (bflag[4] == PERIODIC && blo[2]+eps < boxlo[2]) outflag = 1;
      if (bflag[5] == PERIODIC && bhi[2]-eps > boxhi[2]) outflag = 1;
    }

    if (outflag) {
      char str[128];
      sprintf(str,"Fix rigid body %d moved beyond a periodic boundary",
              ibody+1);
      error->all(FLERR,str);
    }
  }

  // push-off forces for all bodies, after every body has moved to its
  //   end-of-step position; body-body contact forces are applied
  //   equal-and-opposite to both bodies of a contact, so body-body
  //   interactions conserve momentum

  for (ibody = 0; ibody < nbody; ibody++) {
    fpush[ibody][0] = fpush[ibody][1] = fpush[ibody][2] = 0.0;
    tqpush[ibody][0] = tqpush[ibody][1] = tqpush[ibody][2] = 0.0;
  }

  if (pushflag) {
    for (ibody = 0; ibody < nbody; ibody++) push_off(ibody);

    // for distributed surfs the static-contact contributions are
    //   disjoint per-proc partial sums (each proc handles the static
    //   surfs it owns): merge with one Allreduce for all bodies
    // for non-distributed surfs every proc computed identical totals

    if (surf->distributed) {
      for (ibody = 0; ibody < nbody; ibody++) {
        ftbuf_mine[6*ibody]   = fpush[ibody][0];
        ftbuf_mine[6*ibody+1] = fpush[ibody][1];
        ftbuf_mine[6*ibody+2] = fpush[ibody][2];
        ftbuf_mine[6*ibody+3] = tqpush[ibody][0];
        ftbuf_mine[6*ibody+4] = tqpush[ibody][1];
        ftbuf_mine[6*ibody+5] = tqpush[ibody][2];
      }
      MPI_Allreduce(ftbuf_mine,ftbuf_all,6*nbody,MPI_DOUBLE,MPI_SUM,world);
      for (ibody = 0; ibody < nbody; ibody++) {
        fpush[ibody][0] = ftbuf_all[6*ibody];
        fpush[ibody][1] = ftbuf_all[6*ibody+1];
        fpush[ibody][2] = ftbuf_all[6*ibody+2];
        tqpush[ibody][0] = ftbuf_all[6*ibody+3];
        tqpush[ibody][1] = ftbuf_all[6*ibody+4];
        tqpush[ibody][2] = ftbuf_all[6*ibody+5];
      }
    }

    for (ibody = 0; ibody < nbody; ibody++) {
      axi_project(fpush[ibody],tqpush[ibody]);
      fcm[ibody][0] += fpush[ibody][0];
      fcm[ibody][1] += fpush[ibody][1];
      fcm[ibody][2] += fpush[ibody][2];
      torque[ibody][0] += tqpush[ibody][0];
      torque[ibody][1] += tqpush[ibody][1];
      torque[ibody][2] += tqpush[ibody][2];
    }
  }

  // second half kick of velocity Verlet for every body, now that
  //   its end-of-step force and torque are complete: vcm/angmom/omega
  //   become the velocities at the end of the step, synchronized
  //   with xcm/quat, as reported by the fix and written to outfile

  for (ibody = 0; ibody < nbody; ibody++) final_kick(ibody);

  // write body states to the output file every outevery steps, now that
  //   velocities and forces are complete; the file is compatible with
  //   the infile option for run continuation

  if (outfile && update->ntimestep % outevery == 0) write_outfile();

  // re-map body surfs to grid cells: cut/split cells and
  //   INSIDE/OUTSIDE typing from the new body positions; attempt the
  //   cheap incremental re-cut of only the affected cells, else do an
  //   exact full grid re-map
  // incremental_recut() returns a reason code > 0 if a full re-map is
  //   required; all procs must agree, so reduce the max
  // warn once per run when the fallback occurs, since a fallback on
  //   every step silently costs as much as remap cutcell

  int fallback = 1;
  int structural = 0;
  if (remapmode == INCREMENTAL) {
    int mine[3],all[3];
    mine[0] = incremental_recut();
    mine[1] = (npending > 0);
    mine[2] = typechanged;
    MPI_Allreduce(mine,all,3,MPI_INT,MPI_MAX,world);
    fallback = all[0];
    structural = all[1];

    // an incremental re-cut which changed cell markings must be seen
    //   by emit fixes, whose per-cell tasks depend on them; a full
    //   re-map and split_rebuild() both notify them via
    //   Grid::notify_changed()

    if (!fallback && !structural && all[2])
      for (int ifix = 0; ifix < modify->nfix; ifix++)
        if (strncmp(modify->fix[ifix]->style,"emit",4) == 0)
          modify->fix[ifix]->grid_changed();
    typechanged = 0;

    if (fallback && !warnfallback) {
      warnfallback = 1;
      if (comm->me == 0) {
        const char *why;
        if (fallback == FALLBACK_SPLIT)
          why = "a cell in the re-cut region is or would become "
                "a split cell";
        else if (fallback == FALLBACK_SURFMAX)
          why = "a cell would exceed global surfmax";
        else if (fallback == FALLBACK_UNKNOWN)
          why = "the cut of a cell could not decide its inside/outside "
            "marking (body surfs only touching its faces)";
        else why = "no previous body position is known";
        char str[256];
        snprintf(str,sizeof(str),"Fix rigid incremental remap fell back "
                 "to a full grid re-map because %s",why);
        error->warning(FLERR,str);
      }
    }
  }

  // every split cell whose geometry changed holds its particles in its
  //   sub cells, but the flow regions those stand for moved with the
  //   body, so the particles have to be redistributed
  // pull them up into the split cell first, before any cell is
  //   restructured, and let remove_inside_all() put them back into the
  //   right pieces: the same two steps a full re-map takes around
  //   Grid::clear_surf()
  // purely local, and a proc with no split cell does nothing

  if (!fallback && (splitchanged || structural) &&
      particle->exist && grid->nsplitlocal) {
    particles_to_host();
    if (!particle->sorted) particle->sort();
    Grid::ChildCell *cells = grid->cells;
    int nglocal = grid->nlocal;
    for (int icell = 0; icell < nglocal; icell++)
      if (cells[icell].nsplit > 1)
        grid->combine_split_cell_particles(icell,1);
  }

  // a cell which gained or lost sub cells is restructured here, which
  //   is far cheaper than the full re-map it used to force: the surf
  //   lists, volumes and cell types are already correct, so only the
  //   cell list, the ghosts and the neighbor links are rebuilt
  // every proc enters split_rebuild() together, since it communicates

  if (fallback) grid_rebuild();
  else if (structural) split_rebuild();
  npending = 0;

  // remove particles inside any body in one pass over particles,
  //   with split-cell reassignment after a full re-map or after a
  //   split cell was re-cut in place; no reduction here, deletion
  //   counts stay per-proc and are reduced lazily by compute_scalar()

  if (particle->exist) remove_inside_all(fallback || splitchanged || structural);

  // advance the previous-region bookkeeping for incremental remap

  if (remapmode == INCREMENTAL) {
    for (ibody = 0; ibody < nbody; ibody++)
      for (j = 0; j < 3; j++) {
        pbodylo[ibody][j] = bbodylo[ibody][j];
        pbodyhi[ibody][j] = bbodyhi[ibody][j];
      }
    pbodyflag = 1;
  }
}

/* ----------------------------------------------------------------------
   write current attributes of all rigid bodies to output file
   one line per body, format matches what the infile option reads
   moi is written in the space frame for the current body orientation
------------------------------------------------------------------------- */

void FixRigid::write_outfile()
{
  if (comm->me) return;

  FILE *fp = fopen(outfile,"w");
  if (fp == nullptr) error->one(FLERR,"Cannot open fix rigid outfile");

  fprintf(fp,"# rigid body state from fix %s rigid at timestep " BIGINT_FORMAT
          "\n",id,update->ntimestep);

  // fcm/torque are written after the 16 body params, so that a
  //   continuation run resumes with the force and torque which would
  //   have moved the body on the next step

  fprintf(fp,"# ID mtotal xcm ycm zcm ixx iyy izz ixy ixz iyz "
          "vxcm vycm vzcm lx ly lz fx fy fz tx ty tz\n");

  for (int ibody = 0; ibody < nbody; ibody++) {
    double *in = inertia[ibody];
    double *ex = ex_space[ibody];
    double *ey = ey_space[ibody];
    double *ez = ez_space[ibody];

    // reconstruct space-frame moi from principal moments and current axes
    // I_space = sum over K of inertia[K] e_K outer-product e_K

    double ispace[6];
    ispace[0] = in[0]*ex[0]*ex[0] + in[1]*ey[0]*ey[0] + in[2]*ez[0]*ez[0];
    ispace[1] = in[0]*ex[1]*ex[1] + in[1]*ey[1]*ey[1] + in[2]*ez[1]*ez[1];
    ispace[2] = in[0]*ex[2]*ex[2] + in[1]*ey[2]*ey[2] + in[2]*ez[2]*ez[2];
    ispace[3] = in[0]*ex[0]*ex[1] + in[1]*ey[0]*ey[1] + in[2]*ez[0]*ez[1];
    ispace[4] = in[0]*ex[0]*ex[2] + in[1]*ey[0]*ey[2] + in[2]*ez[0]*ez[2];
    ispace[5] = in[0]*ex[1]*ex[2] + in[1]*ey[1]*ey[2] + in[2]*ez[1]*ez[2];

    // 17 significant digits, the fewest which read back as the same double

    fprintf(fp,"%d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g "
            "%.17g %.17g %.17g %.17g %.17g %.17g "
            "%.17g %.17g %.17g %.17g %.17g %.17g\n",
            ibody+1,massbody[ibody],
            xcm[ibody][0],xcm[ibody][1],xcm[ibody][2],
            ispace[0],ispace[1],ispace[2],ispace[3],ispace[4],ispace[5],
            vcm[ibody][0],vcm[ibody][1],vcm[ibody][2],
            angmom[ibody][0],angmom[ibody][1],angmom[ibody][2],
            fcm[ibody][0],fcm[ibody][1],fcm[ibody][2],
            torque[ibody][0],torque[ibody][1],torque[ibody][2]);
  }

  fclose(fp);
}

/* ----------------------------------------------------------------------
   convert one word of the infile to a double
   the word must be a complete floating point number: atof() would
     silently truncate "1.0d-22" to 1.0 and run the body with that mass
   proc 0 only, so error->one
------------------------------------------------------------------------- */

static double infile_numeric(Error *error, const char *word)
{
  if (!word || !word[0])
    error->one(FLERR,"Unexpected end of line in fix rigid infile");

  int n = strlen(word);
  for (int i = 0; i < n; i++) {
    if (isdigit((unsigned char) word[i])) continue;
    if (word[i] == '-' || word[i] == '+' || word[i] == '.') continue;
    if (word[i] == 'e' || word[i] == 'E') continue;
    error->one(FLERR,"Invalid floating point number in fix rigid infile");
  }

  char *end;
  double value = strtod(word,&end);
  if (end == word || *end != '\0')
    error->one(FLERR,"Invalid floating point number in fix rigid infile");
  return value;
}

/* ----------------------------------------------------------------------
   one-time initialization of rigid body attributes from file
   one non-empty, non-comment line per body: body ID, then 16 params,
     optionally followed by the force and torque which act on the body
     during the first step of a continuation run
   called after gather_body(), so nbody is known
------------------------------------------------------------------------- */

void FixRigid::read_infile(char *filename)
{
  int ibody,nwords;
  double *buf;
  memory->create(buf,nbody*22,"fix_rigid:buf");

  // read and convert all lines, only done by proc 0

  if (comm->me == 0) {
    char *start;
    char line[MAXLINE];
    const char *sep = " \t\n\r\f";
    int *seen = new int[nbody];
    for (ibody = 0; ibody < nbody; ibody++) seen[ibody] = 0;

    FILE *fp = fopen(filename,"r");
    if (fp == nullptr)
      error->one(FLERR,"Cannot open fix rigid infile");

    for (int iline = 0; iline < nbody; iline++) {
      while (true) {
        char *eof = fgets(line,MAXLINE,fp);
        if (eof == nullptr)
          error->one(FLERR,"Unexpected end of fix rigid infile");
        start = &line[strspn(line," \t\n\v\f\r")];
        if (*start != '\0' && *start != '#') break;
      }

      // every line has the same word count: ID + 16, or ID + 22

      nwords = input->count_words(line);
      if (nwords != 17 && nwords != 23)
        error->one(FLERR,"Incorrect rigid body format in fix rigid infile");
      if (iline == 0) forceinfile = (nwords == 23);
      else if (forceinfile != (nwords == 23))
        error->one(FLERR,"Incorrect rigid body format in fix rigid infile");

      // body ID, then totalmass, xcm, moi, vcm, angmom, force, torque

      ibody = (int) infile_numeric(error,strtok(line,sep)) - 1;
      if (ibody < 0 || ibody >= nbody || seen[ibody])
        error->one(FLERR,"Invalid body ID in fix rigid infile");
      seen[ibody] = 1;

      double *values = &buf[22*ibody];
      for (int j = 0; j < nwords-1; j++)
        values[j] = infile_numeric(error,strtok(NULL,sep));
      for (int j = nwords-1; j < 22; j++) values[j] = 0.0;
    }

    fclose(fp);
    delete [] seen;
  }

  // broadcast result of file read to all procs

  MPI_Bcast(&forceinfile,1,MPI_INT,0,world);
  MPI_Bcast(buf,nbody*22,MPI_DOUBLE,0,world);

  for (ibody = 0; ibody < nbody; ibody++) {
    double *values = &buf[22*ibody];
    massbody[ibody] = values[0];
    for (int j = 0; j < 3; j++) xcm[ibody][j] = values[1+j];
    for (int j = 0; j < 6; j++) moi[ibody][j] = values[4+j];
    for (int j = 0; j < 3; j++) vcm[ibody][j] = values[10+j];
    for (int j = 0; j < 3; j++) angmom[ibody][j] = values[13+j];
    for (int j = 0; j < 3; j++) fcm_infile[ibody][j] = values[16+j];
    for (int j = 0; j < 3; j++) torque_infile[ibody][j] = values[19+j];
    if (massbody[ibody] <= 0.0)
      error->all(FLERR,"Fix rigid body mass must be positive");
  }

  memory->destroy(buf);
}

/* ----------------------------------------------------------------------
   build the replicated table of body elements: geometry + attributes
   every proc stores every body element, the authoritative source for
     all body computations; bodies are compact so this is small even
     when the full surf collection is distributed
   body ID of a surf in the group = 1 for bodystyle single, its surf
     type for type, its custom vector value for custom; a surf with
     body ID 0 is static, the IDs of all other surfs must be 1 to nbody
   elements are ordered by body ID, then by surf ID within a body
   non-distributed surfs: filled directly from the local surf arrays,
     which hold all surfs on every proc
   distributed surfs: each proc contributes the body elements it owns,
     gathered to all procs and sorted so every proc builds the
     identical table
------------------------------------------------------------------------- */

// per-element record exchanged between procs for distributed surfs

struct BodyElemRecord {
  double pts[9];
  surfint id;
  int bodyid,type,mask,trans,isc,isr;
};

static bool body_record_cmp(const BodyElemRecord &a, const BodyElemRecord &b)
{
  if (a.bodyid != b.bodyid) return a.bodyid < b.bodyid;
  return a.id < b.id;
}

void FixRigid::gather_body()
{
  int i,j,k,n,ibody;

  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;

  // per-surf custom vector of body IDs, for bodystyle custom
  // owned values are the source for distributed surfs, for
  //   non-distributed surfs spread them to the local arrays

  int *custom = NULL;
  int icustom;
  if (bodystyle == CUSTOM) {
    icustom = surf->find_custom(customname);
    if (icustom < 0)
      error->all(FLERR,"Fix rigid custom vector does not exist");
    if (surf->etype[icustom] != INT || surf->esize[icustom] != 0)
      error->all(FLERR,"Fix rigid custom attribute is not an integer vector");
    if (!surf->distributed) {
      if (surf->estatus[icustom] == 0) surf->spread_custom(icustom);
      custom = surf->eivec_local[surf->ewhich[icustom]];
    } else custom = surf->eivec[surf->ewhich[icustom]];
  }

  // ngroupsurf = # of surfs in group, checked by init() for changes

  bigint bnsurf = surf->count_group(igroup);
  if (bnsurf > MAXSMALLINT) error->all(FLERR,"Too many surfs in rigid body");
  ngroupsurf = bnsurf;

  BodyElemRecord *all;
  int nlocal = surf->nlocal;

  if (!surf->distributed) {

    // pack every surf in the group with a non-zero body ID

    int *bodyid = new int[MAX(nlocal,1)];
    nsurf = 0;

    for (i = 0; i < nlocal; i++) {
      int mask = (dim == 2) ? lines[i].mask : tris[i].mask;
      if (!(mask & groupbit)) continue;
      if (bodystyle == SINGLE) bodyid[i] = 1;
      else if (bodystyle == TYPE)
        bodyid[i] = (dim == 2) ? lines[i].type : tris[i].type;
      else bodyid[i] = custom[i];
      if (bodyid[i] < 0) error->all(FLERR,"Fix rigid body ID is negative");
      if (bodyid[i]) nsurf++;
    }

    all = new BodyElemRecord[MAX(nsurf,1)];

    n = 0;
    for (i = 0; i < nlocal; i++) {
      int mask = (dim == 2) ? lines[i].mask : tris[i].mask;
      if (!(mask & groupbit)) continue;
      if (!bodyid[i]) continue;
      BodyElemRecord &r = all[n++];
      r.bodyid = bodyid[i];
      if (dim == 2) {
        r.id = lines[i].id;
        r.type = lines[i].type;
        r.mask = lines[i].mask;
        r.trans = lines[i].transparent;
        r.isc = lines[i].isc;
        r.isr = lines[i].isr;
        memcpy(&r.pts[0],lines[i].p1,3*sizeof(double));
        memcpy(&r.pts[3],lines[i].p2,3*sizeof(double));
      } else {
        r.id = tris[i].id;
        r.type = tris[i].type;
        r.mask = tris[i].mask;
        r.trans = tris[i].transparent;
        r.isc = tris[i].isc;
        r.isr = tris[i].isr;
        memcpy(&r.pts[0],tris[i].p1,3*sizeof(double));
        memcpy(&r.pts[3],tris[i].p2,3*sizeof(double));
        memcpy(&r.pts[6],tris[i].p3,3*sizeof(double));
      }
    }

    delete [] bodyid;

  } else {

    // pack the body elements this proc owns, gather to all procs

    Surf::Line *mylines = surf->mylines;
    Surf::Tri *mytris = surf->mytris;
    int nown = surf->nown;

    int *bodyid = new int[MAX(nown,1)];
    int nmine = 0;

    for (i = 0; i < nown; i++) {
      int mask = (dim == 2) ? mylines[i].mask : mytris[i].mask;
      if (!(mask & groupbit)) continue;
      if (bodystyle == SINGLE) bodyid[i] = 1;
      else if (bodystyle == TYPE)
        bodyid[i] = (dim == 2) ? mylines[i].type : mytris[i].type;
      else bodyid[i] = custom[i];
      if (bodyid[i] < 0) error->one(FLERR,"Fix rigid body ID is negative");
      if (bodyid[i]) nmine++;
    }

    BodyElemRecord *mine = new BodyElemRecord[MAX(nmine,1)];

    n = 0;
    for (i = 0; i < nown; i++) {
      int mask = (dim == 2) ? mylines[i].mask : mytris[i].mask;
      if (!(mask & groupbit)) continue;
      if (!bodyid[i]) continue;
      BodyElemRecord &r = mine[n++];
      r.bodyid = bodyid[i];
      if (dim == 2) {
        r.id = mylines[i].id;
        r.type = mylines[i].type;
        r.mask = mylines[i].mask;
        r.trans = mylines[i].transparent;
        r.isc = mylines[i].isc;
        r.isr = mylines[i].isr;
        memcpy(&r.pts[0],mylines[i].p1,3*sizeof(double));
        memcpy(&r.pts[3],mylines[i].p2,3*sizeof(double));
      } else {
        r.id = mytris[i].id;
        r.type = mytris[i].type;
        r.mask = mytris[i].mask;
        r.trans = mytris[i].transparent;
        r.isc = mytris[i].isc;
        r.isr = mytris[i].isr;
        memcpy(&r.pts[0],mytris[i].p1,3*sizeof(double));
        memcpy(&r.pts[3],mytris[i].p2,3*sizeof(double));
        memcpy(&r.pts[6],mytris[i].p3,3*sizeof(double));
      }
    }

    delete [] bodyid;

    int nprocs = comm->nprocs;
    int *counts = new int[nprocs];
    int *displs = new int[nprocs];
    int nbytes = nmine * (int) sizeof(BodyElemRecord);
    MPI_Allgather(&nbytes,1,MPI_INT,counts,1,MPI_INT,world);
    displs[0] = 0;
    for (i = 1; i < nprocs; i++) displs[i] = displs[i-1] + counts[i-1];
    bigint btotal = (bigint) displs[nprocs-1] + counts[nprocs-1];
    if (btotal > MAXSMALLINT)
      error->all(FLERR,"Too many surfs in rigid body for distributed gather");
    nsurf = btotal / sizeof(BodyElemRecord);

    all = new BodyElemRecord[MAX(nsurf,1)];
    MPI_Allgatherv(mine,nbytes,MPI_BYTE,all,counts,displs,MPI_BYTE,world);

    delete [] mine;
    delete [] counts;
    delete [] displs;
  }

  if (nsurf == 0) error->all(FLERR,"Fix rigid has no body surface elements");

  // sort elements by body ID, then by surf ID, so that the elements of
  //   each body are contiguous and the table is identical on all procs

  std::sort(all,all+nsurf,body_record_cmp);

  // nbody = largest body ID, every body must have at least one element

  nbody = all[nsurf-1].bodyid;

  memory->create(body,nsurf,"fix_rigid:body");
  memory->create(bodystart,nbody+1,"fix_rigid:bodystart");

  for (ibody = 0; ibody <= nbody; ibody++) bodystart[ibody] = 0;
  for (i = 0; i < nsurf; i++) {
    body[i] = all[i].bodyid - 1;
    bodystart[body[i]+1]++;
  }
  for (ibody = 0; ibody < nbody; ibody++) {
    if (bodystart[ibody+1] == 0) {
      char str[128];
      sprintf(str,"Fix rigid body %d has no surface elements",ibody+1);
      error->all(FLERR,str);
    }
    bodystart[ibody+1] += bodystart[ibody];
  }

  // fill the replicated element table from the sorted records

  memory->create(bodypt,nsurf,dim,3,"fix_rigid:bodypt");
  memory->create(bodynorm,nsurf,3,"fix_rigid:bodynorm");
  memory->create(sids,nsurf,"fix_rigid:sids");
  memory->create(bodymask,nsurf,"fix_rigid:bodymask");
  memory->create(bodytype,nsurf,"fix_rigid:bodytype");
  memory->create(bodytrans,nsurf,"fix_rigid:bodytrans");
  memory->create(bodyisc,nsurf,"fix_rigid:bodyisc");
  memory->create(bodyisr,nsurf,"fix_rigid:bodyisr");
  memory->create(lblist,nsurf,"fix_rigid:lblist");

  double d12[3],d13[3];
  double z[3] = {0.0,0.0,1.0};

  for (i = 0; i < nsurf; i++) {
    BodyElemRecord &r = all[i];
    sids[i] = r.id;
    bodymask[i] = r.mask;
    bodytype[i] = r.type;
    bodytrans[i] = r.trans;
    bodyisc[i] = r.isc;
    bodyisr[i] = r.isr;
    memcpy(bodypt[i][0],&r.pts[0],3*sizeof(double));
    memcpy(bodypt[i][1],&r.pts[3],3*sizeof(double));
    if (dim == 3) memcpy(bodypt[i][2],&r.pts[6],3*sizeof(double));

    // recompute the outward normal the same way Surf does

    if (dim == 2) {
      MathExtra::sub3(bodypt[i][1],bodypt[i][0],d12);
      MathExtra::cross3(z,d12,bodynorm[i]);
      MathExtra::norm3(bodynorm[i]);
      bodynorm[i][2] = 0.0;
    } else {
      MathExtra::sub3(bodypt[i][1],bodypt[i][0],d12);
      MathExtra::sub3(bodypt[i][2],bodypt[i][0],d13);
      MathExtra::cross3(d12,d13,bodynorm[i]);
      MathExtra::norm3(bodynorm[i]);
    }
  }

  delete [] all;

  // idmap = global surf ID -> body element index

  idmap.clear();
  for (i = 0; i < nsurf; i++) idmap[sids[i]] = i;

  // non-distributed: slist = local surf index of each element

  slist = NULL;
  if (!surf->distributed) {
    memory->create(slist,nsurf,"fix_rigid:slist");
    for (i = 0; i < nsurf; i++) slist[i] = -1;
    for (i = 0; i < nlocal; i++) {
      surfint id = (dim == 2) ? lines[i].id : tris[i].id;
      k = body_elem(id);
      if (k >= 0) slist[k] = i;
    }
  }

  // lblist = local surf index of each body element on this proc
  // for distributed surfs, ensure_local_copies() fills lblist and the
  //   list of all local copies at setup and after every surf change
  // olist = owned-array index of the body elements this proc owns

  ncopy = maxcopy = 0;
  copy_index = copy_elem = NULL;

  for (i = 0; i < nsurf; i++) lblist[i] = -1;
  int nslocal = surf->nlocal;
  for (i = 0; i < nslocal; i++) {
    surfint id = (dim == 2) ? lines[i].id : tris[i].id;
    k = body_elem(id);
    if (k >= 0) lblist[k] = i;
  }

  nolist = 0;
  olist_own = olist_elem = NULL;
  if (surf->distributed) {
    Surf::Line *mylines = surf->mylines;
    Surf::Tri *mytris = surf->mytris;
    int nown = surf->nown;
    memory->create(olist_own,nsurf,"fix_rigid:olist_own");
    memory->create(olist_elem,nsurf,"fix_rigid:olist_elem");
    for (i = 0; i < nown; i++) {
      surfint id = (dim == 2) ? mylines[i].id : mytris[i].id;
      k = body_elem(id);
      if (k < 0) continue;
      olist_own[nolist] = i;
      olist_elem[nolist] = k;
      nolist++;
    }
  }
}

/* ----------------------------------------------------------------------
   distributed surfs: insure this proc's local (non-ghost) surf arrays
     contain a copy of every element of each body it needs, at its
     current position
   the swept collision lists and the particle mover reference body
     surfs by local index, and a fast body can sweep into cells on a
     proc whose local arrays do not yet hold its surfs
   a proc needs a body if the body's bbox (swept bbox during a step)
     overlaps the bbox of its owned and ghost cells, or if it already
     stores a copy of one of the body's surfs: a copy is kept until the
     next grid rebuild discards it, and a ghost copy is referenced by a
     ghost cell, so it must follow the body like the local copies do
   a body far from this proc's cells thus costs it no surf storage
   copies must be in the local range: owned cells may only reference
     local surfs (Surf::compress_explicit relies on it)
   ghost surfs follow the local range in the same array, so appending a
     local copy while ghosts exist requires re-packing the ghosts and
     re-indexing the csurfs lists of ghost cells; a ghost copy of a
     promoted element is dropped, its references map to the local copy
   the surf hash is empty outside of Grid::acquire_ghosts(), so it
     needs no maintenance here
   called at setup and from grid_changed() after any grid/surf change;
     refreshes lblist and the list of all local copies
   caller is responsible for update->build_rigidmap() and, if surfs
     were appended, surfs_changed()
------------------------------------------------------------------------- */

int FixRigid::ensure_local_copies()
{
  int i,j,k,m;

  if (!surf->distributed) return 0;

  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;
  int nslocal = surf->nlocal;
  int nsghost = surf->nghost;

  // every copy of a body element in the local range

  for (k = 0; k < nsurf; k++) lblist[k] = -1;
  ncopy = 0;

  for (i = 0; i < nslocal; i++) {
    surfint id = (dim == 2) ? lines[i].id : tris[i].id;
    k = body_elem(id);
    if (k < 0) continue;
    if (lblist[k] < 0) lblist[k] = i;
    if (ncopy == maxcopy) {
      maxcopy += DELTA_MODIFY;
      memory->grow(copy_index,maxcopy,"fix_rigid:copy_index");
      memory->grow(copy_elem,maxcopy,"fix_rigid:copy_elem");
    }
    copy_index[ncopy] = i;
    copy_elem[ncopy] = k;
    ncopy++;
  }

  // bodies this proc needs local copies of

  for (i = 0; i < nbody; i++)
    bodyneed[i] = box_overlap(bbodylo[i],bbodyhi[i],proclo,prochi);
  for (k = 0; k < nsurf; k++)
    if (lblist[k] >= 0) bodyneed[body[k]] = 1;
  for (i = nslocal; i < nslocal+nsghost; i++) {
    surfint id = (dim == 2) ? lines[i].id : tris[i].id;
    k = body_elem(id);
    if (k >= 0) bodyneed[body[k]] = 1;
  }

  int nmissing = 0;
  for (k = 0; k < nsurf; k++)
    if (lblist[k] < 0 && bodyneed[body[k]]) nmissing++;
  if (!nmissing) return 0;

  // save the ghost entries, then truncate the ghost range

  Surf::Line *glines = NULL;
  Surf::Tri *gtris = NULL;
  int *gmap = new int[MAX(nsghost,1)];

  if (nsghost) {
    if (dim == 2) {
      glines = new Surf::Line[nsghost];
      memcpy(glines,&lines[nslocal],nsghost*sizeof(Surf::Line));
    } else {
      gtris = new Surf::Tri[nsghost];
      memcpy(gtris,&tris[nslocal],nsghost*sizeof(Surf::Tri));
    }
  }
  surf->remove_ghosts();

  // append a local copy of each missing element of a needed body
  //   from the body table

  for (k = 0; k < nsurf; k++) {
    if (lblist[k] >= 0 || !bodyneed[body[k]]) continue;
    if (dim == 2) {
      Surf::Line line;
      memset(&line,0,sizeof(Surf::Line));
      line.id = sids[k];
      line.type = bodytype[k];
      line.mask = bodymask[k];
      line.transparent = bodytrans[k];
      line.isc = bodyisc[k];
      line.isr = bodyisr[k];
      memcpy(line.p1,bodypt[k][0],3*sizeof(double));
      memcpy(line.p2,bodypt[k][1],3*sizeof(double));
      memcpy(line.norm,bodynorm[k],3*sizeof(double));
      surf->add_line_copy(1,&line);
    } else {
      Surf::Tri tri;
      memset(&tri,0,sizeof(Surf::Tri));
      tri.id = sids[k];
      tri.type = bodytype[k];
      tri.mask = bodymask[k];
      tri.transparent = bodytrans[k];
      tri.isc = bodyisc[k];
      tri.isr = bodyisr[k];
      memcpy(tri.p1,bodypt[k][0],3*sizeof(double));
      memcpy(tri.p2,bodypt[k][1],3*sizeof(double));
      memcpy(tri.p3,bodypt[k][2],3*sizeof(double));
      memcpy(tri.norm,bodynorm[k],3*sizeof(double));
      surf->add_tri_copy(1,&tri);
    }
    lblist[k] = surf->nlocal - 1;
    if (ncopy == maxcopy) {
      maxcopy += DELTA_MODIFY;
      memory->grow(copy_index,maxcopy,"fix_rigid:copy_index");
      memory->grow(copy_elem,maxcopy,"fix_rigid:copy_elem");
    }
    copy_index[ncopy] = lblist[k];
    copy_elem[ncopy] = k;
    ncopy++;
  }

  // re-append the saved ghosts after the enlarged local range
  // gmap = new index of each old ghost, a promoted element's ghost copy
  //   maps to its new local copy

  for (m = 0; m < nsghost; m++) {
    surfint id = (dim == 2) ? glines[m].id : gtris[m].id;
    k = body_elem(id);
    if (k >= 0) {
      gmap[m] = lblist[k];
      continue;
    }
    if (dim == 2) surf->add_line_copy(0,&glines[m]);
    else surf->add_tri_copy(0,&gtris[m]);
    gmap[m] = surf->nlocal + surf->nghost - 1;
  }

  // re-index ghost-range entries in the csurfs lists of ghost cells
  // sub cells share the list of their split cell, so visit each once

  if (nsghost) {
    Grid::ChildCell *cells = grid->cells;
    int nglocal = grid->nlocal;
    int ngtotal = grid->nlocal + grid->nghost;

    for (int icell = nglocal; icell < ngtotal; icell++) {
      if (cells[icell].nsplit <= 0) continue;
      if (cells[icell].nsurf <= 0) continue;
      surfint *csurfs = cells[icell].csurfs;
      int n = cells[icell].nsurf;
      for (j = 0; j < n; j++)
        if (csurfs[j] >= nslocal) csurfs[j] = gmap[csurfs[j]-nslocal];
    }
  }

  delete [] glines;
  delete [] gtris;
  delete [] gmap;
  return 1;
}

/* ----------------------------------------------------------------------
   bounding box of this proc's owned and ghost cells
   a body whose bbox does not overlap it cannot put a surf in any cell
     this proc stores, so this proc needs no local copy of its surfs
   called whenever the cells change, before ensure_local_copies()
------------------------------------------------------------------------- */

void FixRigid::proc_bbox()
{
  Grid::ChildCell *cells = grid->cells;
  int ntotal = grid->nlocal + grid->nghost;

  proclo[0] = proclo[1] = proclo[2] = BIG;
  prochi[0] = prochi[1] = prochi[2] = -BIG;

  for (int icell = 0; icell < ntotal; icell++) {
    if (cells[icell].nsplit <= 0) continue;
    for (int k = 0; k < 3; k++) {
      proclo[k] = MIN(proclo[k],cells[icell].lo[k]);
      prochi[k] = MAX(prochi[k],cells[icell].hi[k]);
    }
  }
}

/* ----------------------------------------------------------------------
   1 if the coords of a surf (p3 = NULL in 2d) are bit-identical to
     body element k
------------------------------------------------------------------------- */

int FixRigid::same_coords(double *p1, double *p2, double *p3, int k)
{
  if (memcmp(p1,bodypt[k][0],3*sizeof(double))) return 0;
  if (memcmp(p2,bodypt[k][1],3*sizeof(double))) return 0;
  if (p3 && memcmp(p3,bodypt[k][2],3*sizeof(double))) return 0;
  return 1;
}

/* ----------------------------------------------------------------------
   the replicated table of body surf attributes (collision model,
     reaction model, transparency, mask) was captured when the fix was
     defined; a later surf_modify or group command may have changed them
   the surfs at the body element indices must also still be the body
     surfs: a re-numbering (a remove_surf of lower-ID surfs, followed by
     a read_surf which restores the count) would invalidate the body
     element table; since IDs are re-compacted, it is detected by the
     surf coords, which are bit-identical to the body table because
     update_surf_copies() writes them from it
   non-distributed: every proc stores every surf, so refresh the table
     from the current surfs; nothing else is derived from the old values
   distributed: the local copies of body surfs on procs which do not
     otherwise store them are built from the table, so a change would
     make the copies inconsistent across procs: error instead
     each proc checks the elements it uniquely owns
   collective: the result is reduced so all procs error together
------------------------------------------------------------------------- */

void FixRigid::check_body_attributes()
{
  int changed = 0;
  int renumbered = 0;

  if (!surf->distributed) {
    Surf::Line *lines = surf->lines;
    Surf::Tri *tris = surf->tris;
    for (int k = 0; k < nsurf; k++) {
      int i = lblist[k];
      if (i < 0) continue;
      if (dim == 2) {
        if (lines[i].id != sids[k] ||
            !same_coords(lines[i].p1,lines[i].p2,NULL,k))
          renumbered = 1;
        bodyisc[k] = lines[i].isc;
        bodyisr[k] = lines[i].isr;
        bodytrans[k] = lines[i].transparent;
        bodymask[k] = lines[i].mask;
      } else {
        if (tris[i].id != sids[k] ||
            !same_coords(tris[i].p1,tris[i].p2,tris[i].p3,k))
          renumbered = 1;
        bodyisc[k] = tris[i].isc;
        bodyisr[k] = tris[i].isr;
        bodytrans[k] = tris[i].transparent;
        bodymask[k] = tris[i].mask;
      }
    }
  } else {
    Surf::Line *mylines = surf->mylines;
    Surf::Tri *mytris = surf->mytris;
    int nown = surf->nown;
    for (int m = 0; m < nolist; m++) {
      int i = olist_own[m];
      int k = olist_elem[m];
      if (i >= nown) {
        renumbered = 1;
        continue;
      }
      if (dim == 2) {
        if (mylines[i].id != sids[k] ||
            !same_coords(mylines[i].p1,mylines[i].p2,NULL,k))
          renumbered = 1;
        if (mylines[i].isc != bodyisc[k] || mylines[i].isr != bodyisr[k] ||
            mylines[i].transparent != bodytrans[k] ||
            mylines[i].mask != bodymask[k]) changed = 1;
      } else {
        if (mytris[i].id != sids[k] ||
            !same_coords(mytris[i].p1,mytris[i].p2,mytris[i].p3,k))
          renumbered = 1;
        if (mytris[i].isc != bodyisc[k] || mytris[i].isr != bodyisr[k] ||
            mytris[i].transparent != bodytrans[k] ||
            mytris[i].mask != bodymask[k]) changed = 1;
      }
    }
  }

  int flags[2],flags_any[2];
  flags[0] = renumbered;
  flags[1] = changed;
  MPI_Allreduce(flags,flags_any,2,MPI_INT,MPI_MAX,world);
  if (flags_any[0])
    error->all(FLERR,"Fix rigid body surfs were renumbered or moved "
               "after the fix was defined");
  if (flags_any[1])
    error->all(FLERR,"Fix rigid body surf attributes were changed after "
               "the fix was defined");
}

/* ----------------------------------------------------------------------
   notify per-surf computes that the local surf arrays changed, so they
     re-size per-surf storage (e.g. ComputeSurf normflux) and refresh
     cached surf pointers
   same action Grid::notify_changed() takes for computes
------------------------------------------------------------------------- */

void FixRigid::surfs_changed(int changed, int stage)
{
  // if any proc appended copies or re-indexed ghosts, the per-surf
  //   state of surface reaction and collision models must follow,
  //   as after a grid change (see Grid::notify_changed())
  // stage = 1 when called from Update::init_rigid() before the models
  //   init, stage = 2 from setup() after they init: flag the change as
  //   of the previous step, which is what SurfCollide::dynamic() tests
  //   in Update::setup() to re-spread its per-surf values over the new
  //   local+ghost surfs
  // a change during a run, after a grid rebuild, is handled by
  //   grid_changed() instead: Grid::notify_changed() itself notifies
  //   the computes and reaction models after the fixes, so only the
  //   flags are reset there (stage 0 here would flag it as of this step)
  // stage = 1: the reaction models have not init'd yet, so their
  //   notification is deferred to init() via Update::rigid_notify_sr
  // collective: every proc takes the same branch

  int changed_any;
  MPI_Allreduce(&changed,&changed_any,1,MPI_INT,MPI_MAX,world);
  if (changed_any) {
    Compute **compute = modify->compute;
    for (int i = 0; i < modify->ncompute; i++)
      if (compute[i]->per_surf_flag) compute[i]->reallocate();
    if (stage == 1) update->rigid_notify_sr = 1;
    else for (int i = 0; i < surf->nsr; i++) surf->sr[i]->grid_changed();
    if (stage) surf->localghost_changed_step = update->ntimestep - 1;
    else surf->localghost_changed_step = update->ntimestep;
    for (int i = 0; i < surf->ncustom; i++) surf->estatus[i] = 0;
  }
}

/* ----------------------------------------------------------------------
   write the current replicated body geometry (bodypt/bodynorm) into
     the Surf storage the particle mover and cut pipeline read:
   non-distributed: the local copies every proc stores (via slist)
   distributed: every local copy on this proc (via the copy list) and
     the owned copies (via olist), so a later re-map redistributes
     current coords
------------------------------------------------------------------------- */

void FixRigid::update_surf_copies()
{
  int i,index;

  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;

  if (!surf->distributed) {
    for (i = 0; i < nsurf; i++) {
      index = slist[i];
      if (dim == 2) {
        memcpy(lines[index].p1,bodypt[i][0],3*sizeof(double));
        memcpy(lines[index].p2,bodypt[i][1],3*sizeof(double));
        memcpy(lines[index].norm,bodynorm[i],3*sizeof(double));
      } else {
        memcpy(tris[index].p1,bodypt[i][0],3*sizeof(double));
        memcpy(tris[index].p2,bodypt[i][1],3*sizeof(double));
        memcpy(tris[index].p3,bodypt[i][2],3*sizeof(double));
        memcpy(tris[index].norm,bodynorm[i],3*sizeof(double));
      }
    }
    return;
  }

  for (int m = 0; m < ncopy; m++) {
    index = copy_index[m];
    i = copy_elem[m];
    if (dim == 2) {
      memcpy(lines[index].p1,bodypt[i][0],3*sizeof(double));
      memcpy(lines[index].p2,bodypt[i][1],3*sizeof(double));
      memcpy(lines[index].norm,bodynorm[i],3*sizeof(double));
    } else {
      memcpy(tris[index].p1,bodypt[i][0],3*sizeof(double));
      memcpy(tris[index].p2,bodypt[i][1],3*sizeof(double));
      memcpy(tris[index].p3,bodypt[i][2],3*sizeof(double));
      memcpy(tris[index].norm,bodynorm[i],3*sizeof(double));
    }
  }

  Surf::Line *mylines = surf->mylines;
  Surf::Tri *mytris = surf->mytris;

  for (int m = 0; m < nolist; m++) {
    int iown = olist_own[m];
    i = olist_elem[m];
    if (dim == 2) {
      memcpy(mylines[iown].p1,bodypt[i][0],3*sizeof(double));
      memcpy(mylines[iown].p2,bodypt[i][1],3*sizeof(double));
      memcpy(mylines[iown].norm,bodynorm[i],3*sizeof(double));
    } else {
      memcpy(mytris[iown].p1,bodypt[i][0],3*sizeof(double));
      memcpy(mytris[iown].p2,bodypt[i][1],3*sizeof(double));
      memcpy(mytris[iown].p3,bodypt[i][2],3*sizeof(double));
      memcpy(mytris[iown].norm,bodynorm[i],3*sizeof(double));
    }
  }
}

/* ----------------------------------------------------------------------
   compute massbody, xcm, and moi from the body geometry, for a body of
     uniform density
   called for dstyle = density, after gather_body() has built the
     replicated element table and check_enclosed() has verified that the
     body encloses a non-zero area/volume with outward normals, so the
     signed measures accumulated below are guaranteed positive

   the volume integrals which define the mass, COM and inertia tensor are
     reduced to sums over the body elements

   3d: each triangle (a,b,c) forms a tetrahedron with the coordinate
     origin, signed by the triangle's winding.  the signed volumes of
     those tets sum to the volume of the body, and their first and second
     moments sum likewise, since the parts of the tets outside the body
     cancel between triangles.  for one tet with the 4th vertex at the
     origin
       6V   = a . (b x c)
       Mij  = (V/20) [ ai aj + bi bj + ci cj + si sj ],  s = a + b + c
     where Mij = integral of xi xj over the tet.  both are exact, so no
     quadrature and no per-element coordinate frame is needed
   2d: the same reduction is Green's theorem on the polygon, whose area,
     centroid and second moments have the standard closed forms.  a 2d
     body is a plate of unit thickness in z, matching how the rest of
     this fix and the surf collision models treat 2d

   the sums are taken about a reference point on the body, not about the
     coordinate origin.  any reference gives the same COM and the same
     COM-relative inertia, but an origin-based sum loses relative
     precision like (R/L)^2 for a body of size L lying a distance R from
     the origin, because the second moments are then O(V R^2) while the
     answer is O(V L^2): a unit body 1e5 away keeps only 5 digits of its
     inertia.  referencing to a point on the body bounds |x| by the body
     diameter and holds the error at round-off for any placement.
   the inertia tensor fix rigid uses is about the COM, so the
     parallel-axis shift is applied at the end, relative to that same
     reference.  the products of inertia carry the minus sign of the
     ixy = -integral x y dm convention
------------------------------------------------------------------------- */

void FixRigid::body_properties(int ibody, double density)
{
  int i,j,k;
  double measure = 0.0;              // volume (3d) or area (2d)
  double first[3];                   // integral of x over the body
  double second[3][3];               // integral of xi xj over the body

  int istart = bodystart[ibody];
  int istop = bodystart[ibody+1];

  for (k = 0; k < 3; k++) first[k] = 0.0;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++) second[i][j] = 0.0;

  // reference all sums to a point on the body (its first vertex), so the
  //   accumulated moments stay O(body size) however far the body sits
  //   from the coordinate origin.  added back into the COM below.

  double ref[3];
  ref[0] = bodypt[istart][0][0];
  ref[1] = bodypt[istart][0][1];
  ref[2] = (dim == 3) ? bodypt[istart][0][2] : 0.0;

  if (dim == 3) {
    double s[3],cr[3],a[3],b[3],c[3];

    for (i = istart; i < istop; i++) {
      MathExtra::sub3(bodypt[i][0],ref,a);
      MathExtra::sub3(bodypt[i][1],ref,b);
      MathExtra::sub3(bodypt[i][2],ref,c);

      MathExtra::cross3(b,c,cr);
      double v6 = MathExtra::dot3(a,cr);      // 6 * signed tet volume
      if (v6 == 0.0) continue;                // degenerate element

      for (k = 0; k < 3; k++) s[k] = a[k] + b[k] + c[k];

      measure += v6;
      for (k = 0; k < 3; k++) first[k] += v6 * s[k];
      for (j = 0; j < 3; j++)
        for (k = 0; k < 3; k++)
          second[j][k] += v6 * (a[j]*a[k] + b[j]*b[k] + c[j]*c[k] +
                                s[j]*s[k]);
    }

    // 6V per element, so V = measure/6
    // first moment: (v6/6) * (s/4) summed = measure-weighted s / 24
    // second moment: (v6/6) * (1/20) of the bracket = bracket / 120

    measure /= 6.0;
    for (k = 0; k < 3; k++) first[k] /= 24.0;
    for (j = 0; j < 3; j++)
      for (k = 0; k < 3; k++) second[j][k] /= 120.0;

  } else {
    double sxx = 0.0, syy = 0.0, sxy = 0.0;

    for (i = istart; i < istop; i++) {
      double x0 = bodypt[i][0][0] - ref[0], y0 = bodypt[i][0][1] - ref[1];
      double x1 = bodypt[i][1][0] - ref[0], y1 = bodypt[i][1][1] - ref[1];
      double cross = x0*y1 - x1*y0;
      if (cross == 0.0 && x0 == x1 && y0 == y1) continue;

      measure += cross;
      first[0] += (x0 + x1) * cross;
      first[1] += (y0 + y1) * cross;
      sxx += (x0*x0 + x0*x1 + x1*x1) * cross;
      syy += (y0*y0 + y0*y1 + y1*y1) * cross;
      sxy += (x0*y1 + 2.0*x0*y0 + 2.0*x1*y1 + x1*y0) * cross;
    }

    // check_enclosed() requires outward normals, which in 2d means a
    //   clockwise traversal and hence a negative signed area; flip the
    //   sign of every accumulated moment so all are positive measures

    measure *= 0.5;
    measure = -measure;
    first[0] = -first[0] / 6.0;
    first[1] = -first[1] / 6.0;
    first[2] = 0.0;
    second[0][0] = -syy / 12.0;      // integral of y^2, used for ixx
    second[1][1] = -sxx / 12.0;      // integral of x^2, used for iyy
    second[0][1] = second[1][0] = -sxy / 24.0;   // integral of x y
  }

  if (measure <= 0.0)
    body_error(ibody,"properties could not be computed");

  double mass = density * measure;
  if (mass <= 0.0) body_error(ibody,"mass must be positive");
  massbody[ibody] = mass;

  double *xcm1 = xcm[ibody];
  xcm1[0] = first[0] / measure + ref[0];
  xcm1[1] = first[1] / measure + ref[1];
  xcm1[2] = (dim == 3) ? first[2] / measure + ref[2] : 0.0;

  // parallel-axis shift is from the reference point to the COM

  double dcm[3];
  dcm[0] = xcm1[0] - ref[0];
  dcm[1] = xcm1[1] - ref[1];
  dcm[2] = xcm1[2] - ref[2];

  // moments of inertia about the COM
  // 3d: ixx = rho * (Myy + Mzz) - M (ycm^2 + zcm^2), etc
  // 2d: a plate, so izz = ixx + iyy and ixz = iyz = 0

  double *moi1 = moi[ibody];
  double rho = mass / measure;

  if (dim == 3) {
    moi1[0] = rho * (second[1][1] + second[2][2]) -
      mass * (dcm[1]*dcm[1] + dcm[2]*dcm[2]);
    moi1[1] = rho * (second[0][0] + second[2][2]) -
      mass * (dcm[0]*dcm[0] + dcm[2]*dcm[2]);
    moi1[2] = rho * (second[0][0] + second[1][1]) -
      mass * (dcm[0]*dcm[0] + dcm[1]*dcm[1]);
    moi1[3] = -rho * second[0][1] + mass * dcm[0]*dcm[1];
    moi1[4] = -rho * second[0][2] + mass * dcm[0]*dcm[2];
    moi1[5] = -rho * second[1][2] + mass * dcm[1]*dcm[2];

  } else {
    moi1[0] = rho * second[0][0] - mass * dcm[1]*dcm[1];
    moi1[1] = rho * second[1][1] - mass * dcm[0]*dcm[0];
    moi1[2] = moi1[0] + moi1[1];
    moi1[3] = -rho * second[0][1] + mass * dcm[0]*dcm[1];
    moi1[4] = 0.0;
    moi1[5] = 0.0;
  }
}

/* ----------------------------------------------------------------------
   compute massbody, xcm, and moi for an axisymmetric body of revolution
   called for dstyle = density in an axisymmetric domain, in place of
     body_properties(), after check_enclosed() has verified that the
     profile encloses a positive volume of revolution with outward
     normals

   the body elements are a profile in the (x,r) half plane traversed so
     that the element normals (-dr,dx)/L point out of the body, i.e. in
     the +x sense over the outer surface.  the solid they generate by
     revolution about the x axis is closed even where the profile itself
     is open, as long as the open ends terminate on the axis: a
     semicircle from (-R,0) to (R,0) generates a sphere with no missing
     area at the poles.  so the surface integrals below need no special
     handling for a body that touches the axis

   every volume integral is reduced to a surface integral by the
     divergence theorem and then done exactly, element by element.  for
     one element, with x(t) = x1 + t dx and r(t) = r1 + t dr on [0,1],
       dA = 2 pi r L dt,  n = (-dr,dx)/L,  so
       int_V div F dV = 2 pi sum_i int_0^1 [-Fx dr + Fr dx] r dt
     which is a polynomial in t, hence exact with no quadrature.  the
     four integrals needed come from
       F = (0, r/2)    -> V          = int dV
       F = (x^2/2, 0)  -> first      = int x dV
       F = (x^3/3, 0)  -> secondxx   = int x^2 dV
       F = (0, r^3/4)  -> secondrr   = int r^2 dV

   as in body_properties(), the x integrals are referenced to a point on
     the body so that an origin-based sum cannot lose relative precision
     like (R/L)^2 for a body far from the origin.  r is measured from the
     axis, which is physical and cannot be shifted

   the COM of a body of revolution lies on the axis, so ycm = zcm = 0 and
     no parallel-axis shift is needed for ixx.  the transverse moments
     use <z^2> = r^2/2 averaged around the ring, giving
       ixx = rho secondrr
       iyy = izz = rho (secondxx - 2 xcm first + xcm^2 V) + ixx/2
     with all products of inertia zero.  only ixx is used by the
     dynamics, since the body may only spin about its own axis; the
     transverse moments are computed so the reported inertia is right
------------------------------------------------------------------------- */

void FixRigid::body_properties_axi(int ibody, double density)
{
  double volume = 0.0;         // int dV
  double first = 0.0;          // int x dV, about ref
  double secondxx = 0.0;       // int x^2 dV, about ref
  double secondrr = 0.0;       // int r^2 dV

  int istart = bodystart[ibody];
  int istop = bodystart[ibody+1];
  double ref = bodypt[istart][0][0];

  for (int i = istart; i < istop; i++) {
    double x1 = bodypt[i][0][0] - ref, r1 = bodypt[i][0][1];
    double x2 = bodypt[i][1][0] - ref, r2 = bodypt[i][1][1];
    double dx = x2 - x1, dr = r2 - r1;

    // int_0^1 r^2 dt, int_0^1 r^4 dt, int_0^1 x^2 r dt, int_0^1 x^3 r dt

    double ir2 = (r1*r1 + r1*r2 + r2*r2) / 3.0;
    double ir4 = r1*r1*r1*r1 + 2.0*r1*r1*r1*dr + 2.0*r1*r1*dr*dr +
      r1*dr*dr*dr + 0.2*dr*dr*dr*dr;
    double ix2r = x1*x1*r1 + 0.5*(x1*x1*dr + 2.0*x1*dx*r1) +
      (2.0*x1*dx*dr + dx*dx*r1)/3.0 + 0.25*dx*dx*dr;
    double ix3r = x1*x1*x1*r1 + 0.5*(x1*x1*x1*dr + 3.0*x1*x1*dx*r1) +
      (3.0*x1*x1*dx*dr + 3.0*x1*dx*dx*r1)/3.0 +
      0.25*(3.0*x1*dx*dx*dr + dx*dx*dx*r1) + 0.2*dx*dx*dx*dr;

    // 2 pi int [-Fx dr + Fr dx] r dt for each F above, so the prefactor
    //   differs per integral: pi for V and first, 2pi/3 for secondxx
    //   (from x^3/3), pi/2 for secondrr (from r^3/4)

    volume += MY_PI * dx * ir2;
    first -= MY_PI * dr * ix2r;
    secondxx -= (2.0/3.0) * MY_PI * dr * ix3r;
    secondrr += 0.5 * MY_PI * dx * ir4;
  }

  if (volume <= 0.0)
    body_error(ibody,"properties could not be computed");

  massbody[ibody] = density * volume;
  if (massbody[ibody] <= 0.0) body_error(ibody,"mass must be positive");

  // the COM is on the axis by symmetry

  double xcmref = first / volume;
  xcm[ibody][0] = xcmref + ref;
  xcm[ibody][1] = 0.0;
  xcm[ibody][2] = 0.0;

  double *moi1 = moi[ibody];
  moi1[0] = density * secondrr;
  moi1[1] = density * (secondxx - 2.0*xcmref*first + xcmref*xcmref*volume) +
    0.5 * moi1[0];
  moi1[2] = moi1[1];
  moi1[3] = moi1[4] = moi1[5] = 0.0;

  if (moi1[0] <= 0.0) body_error(ibody,"properties could not be computed");
}

/* ----------------------------------------------------------------------
   one-time initialization of rigid body attributes
------------------------------------------------------------------------- */

void FixRigid::setup_body()
{
  int ibody,j;

  // build the replicated body element table: geometry + attributes
  // sets nbody

  gather_body();
  allocate_bodies();

  // body params from the define style
  // body: params for a single body from the command
  // infile: params for every body from the file
  // density: mass, COM, and moi are computed from the geometry below,
  //   vcom and angmom from the optional keywords apply to every body

  if (bodyflag) {
    if (nbody > 1)
      error->all(FLERR,"Fix rigid body style requires a single body");
    massbody[0] = massone;
    for (j = 0; j < 3; j++) {
      xcm[0][j] = xcmone[j];
      vcm[0][j] = vcmone[j];
      angmom[0][j] = angmomone[j];
    }
    for (j = 0; j < 6; j++) moi[0][j] = moione[j];
  } else if (infile) read_infile(infile);
  else {
    for (ibody = 0; ibody < nbody; ibody++)
      for (j = 0; j < 3; j++) {
        vcm[ibody][j] = vcmone[j];
        angmom[ibody][j] = angmomone[j];
      }
  }

  for (ibody = 0; ibody < nbody; ibody++) check_body_params(ibody);

  // insure each body's surfs form a closed (watertight) object
  //   which encloses a non-zero area or volume
  // dstyle = density: compute mass, COM, and moi from the geometry,
  //   after the checks have verified the body is closed and its
  //   normals point outward, which the sums rely on
  // then the body frame axes, quaternion, and body-frame geometry

  for (ibody = 0; ibody < nbody; ibody++) {
    check_watertight(ibody);
    check_enclosed(ibody);
    if (densityflag) {
      if (axiflag) body_properties_axi(ibody,density);
      else body_properties(ibody,density);
    }
    setup_body_one(ibody);
    setup_body_displace(ibody);
  }

  // restore the force/torque of the step before a continuation;
  //   a body is moved by them on the first step

  if (forceinfile) {
    for (ibody = 0; ibody < nbody; ibody++) {
      double *f = fcm_infile[ibody];
      double *tq = torque_infile[ibody];
      if (dim == 2 && !axiflag &&
          (f[2] != 0.0 || tq[0] != 0.0 || tq[1] != 0.0))
        error->all(FLERR,"Fix rigid infile force and torque must be "
                   "in-plane for 2d");
      if (axiflag && (f[1] != 0.0 || f[2] != 0.0 ||
                      tq[1] != 0.0 || tq[2] != 0.0))
        error->all(FLERR,"Fix rigid infile force and torque must be axial "
                   "for an axisymmetric domain");
      for (j = 0; j < 3; j++) {
        fcm[ibody][j] = f[j];
        torque[ibody][j] = tq[j];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate and zero the per-body arrays and per-element geometry
   called once by setup_body(), after gather_body() has set nbody/nsurf
------------------------------------------------------------------------- */

void FixRigid::allocate_bodies()
{
  memory->create(xcm,nbody,3,"fix_rigid:xcm");
  memory->create(vcm,nbody,3,"fix_rigid:vcm");
  memory->create(omega,nbody,3,"fix_rigid:omega");
  memory->create(invmass,nbody,"fix_rigid:invmass");
  memory->create(invinertia,nbody,9,"fix_rigid:invinertia");
  memory->create(quat,nbody,4,"fix_rigid:quat");
  memory->create(xcmnew,nbody,3,"fix_rigid:xcmnew");
  memory->create(quatnew,nbody,4,"fix_rigid:quatnew");
  memory->create(xcmmid,nbody,3,"fix_rigid:xcmmid");
  memory->create(massbody,nbody,"fix_rigid:massbody");
  memory->create(moi,nbody,6,"fix_rigid:moi");
  memory->create(inertia,nbody,3,"fix_rigid:inertia");
  memory->create(angmom,nbody,3,"fix_rigid:angmom");
  memory->create(ex_space,nbody,3,"fix_rigid:ex_space");
  memory->create(ey_space,nbody,3,"fix_rigid:ey_space");
  memory->create(ez_space,nbody,3,"fix_rigid:ez_space");
  memory->create(fcm,nbody,3,"fix_rigid:fcm");
  memory->create(torque,nbody,3,"fix_rigid:torque");
  memory->create(fpush,nbody,3,"fix_rigid:fpush");
  memory->create(tqpush,nbody,3,"fix_rigid:tqpush");
  memory->create(rmaxbody,nbody,"fix_rigid:rmaxbody");
  memory->create(rminbody,nbody,"fix_rigid:rminbody");
  memory->create(bbodylo,nbody,3,"fix_rigid:bbodylo");
  memory->create(bbodyhi,nbody,3,"fix_rigid:bbodyhi");
  memory->create(bboxeps,nbody,"fix_rigid:bboxeps");
  memory->create(pbodylo,nbody,3,"fix_rigid:pbodylo");
  memory->create(pbodyhi,nbody,3,"fix_rigid:pbodyhi");
  memory->create(fcm_infile,nbody,3,"fix_rigid:fcm_infile");
  memory->create(torque_infile,nbody,3,"fix_rigid:torque_infile");
  memory->create(bodyneed,nbody,"fix_rigid:bodyneed");

  for (int ibody = 0; ibody < nbody; ibody++) {
    invmass[ibody] = massbody[ibody] = rmaxbody[ibody] = bboxeps[ibody] = 0.0;
    rminbody[ibody] = 0.0;
    for (int j = 0; j < 3; j++)
      xcm[ibody][j] = vcm[ibody][j] = omega[ibody][j] = xcmnew[ibody][j] =
        xcmmid[ibody][j] = inertia[ibody][j] = angmom[ibody][j] =
        ex_space[ibody][j] = ey_space[ibody][j] = ez_space[ibody][j] =
        fcm[ibody][j] = torque[ibody][j] = fpush[ibody][j] =
        tqpush[ibody][j] = bbodylo[ibody][j] = bbodyhi[ibody][j] =
        pbodylo[ibody][j] = pbodyhi[ibody][j] = fcm_infile[ibody][j] =
        torque_infile[ibody][j] = 0.0;
    for (int j = 0; j < 4; j++) quat[ibody][j] = quatnew[ibody][j] = 0.0;
    for (int j = 0; j < 6; j++) moi[ibody][j] = 0.0;
    for (int j = 0; j < 9; j++) invinertia[ibody][j] = 0.0;
  }

  // per-element body-frame corner pts, and bboxes for swept collision
  //   assignment and push-off candidate pruning

  memory->create(displace,nsurf,dim,3,"fix_rigid:displace");
  memory->create(elemlo,nsurf,3,"fix_rigid:elemlo");
  memory->create(elemhi,nsurf,3,"fix_rigid:elemhi");
}

/* ----------------------------------------------------------------------
   error out with a message naming one body
------------------------------------------------------------------------- */

void FixRigid::body_error(int ibody, const char *msg)
{
  char str[256];
  snprintf(str,sizeof(str),"Fix rigid body %d %s",ibody+1,msg);
  error->all(FLERR,str);
}

/* ----------------------------------------------------------------------
   insure the user-settable params of one body are consistent with the
     degrees of freedom of the domain
   2d: in-plane motion.  dstyle = density computes xcm and moi from the
     geometry, which is planar in 2d, so only the settable values are
     checked
   axisymmetric: the body is a body of revolution about the x axis, so
     it can only translate along x and spin about x.  its COM lies on
     the axis, its transverse velocity and angular momentum must be
     zero, and only ixx of the inertia tensor is ever used
------------------------------------------------------------------------- */

void FixRigid::check_body_params(int ibody)
{
  double *x = xcm[ibody];
  double *v = vcm[ibody];
  double *l = angmom[ibody];
  double *m = moi[ibody];

  if (dim == 2 && !axiflag) {
    if (v[2] != 0.0)
      body_error(ibody,"z components of com and vcom must be zero for 2d");
    if (l[0] != 0.0 || l[1] != 0.0)
      body_error(ibody,"x,y components of angmom must be zero for 2d");
    if (!densityflag) {
      if (x[2] != 0.0)
        body_error(ibody,"z components of com and vcom must be zero for 2d");
      if (m[4] != 0.0 || m[5] != 0.0)
        body_error(ibody,"ixz,iyz components of moi must be zero for 2d");
    }
  }

  if (axiflag) {
    if (v[1] != 0.0 || v[2] != 0.0)
      body_error(ibody,"y,z components of vcom must be zero "
                 "for an axisymmetric domain");
    if (l[1] != 0.0 || l[2] != 0.0)
      body_error(ibody,"y,z components of angmom must be zero "
                 "for an axisymmetric domain");
    if (!densityflag) {
      if (x[1] != 0.0 || x[2] != 0.0)
        body_error(ibody,"y,z components of com must be zero "
                   "for an axisymmetric domain");
      if (m[3] != 0.0 || m[4] != 0.0 || m[5] != 0.0)
        body_error(ibody,"products of inertia must be zero "
                   "for an axisymmetric domain");
      if (m[0] <= 0.0)
        body_error(ibody,"ixx of moi must be positive "
                   "for an axisymmetric domain");
    }
  }
}

/* ----------------------------------------------------------------------
   body frame axes and quaternion of one body from its inertia tensor
------------------------------------------------------------------------- */

void FixRigid::setup_body_one(int ibody)
{
  double *moi1 = moi[ibody];
  double *inertia1 = inertia[ibody];
  double *ex = ex_space[ibody];
  double *ey = ey_space[ibody];
  double *ez = ez_space[ibody];

  // axisymmetric: the body frame is the space frame, always
  // a body of revolution has iyy = izz, so the inertia tensor is
  //   degenerate in the transverse plane and a diagonalization would
  //   return an arbitrary rotation about x.  that rotation is harmless
  //   physically, but it would make the body-frame -> space-frame map
  //   in end_of_step() a non-trivial matrix product, and the profile
  //   points which sit exactly on the axis would then pick up a
  //   round-off y of order 1e-17 instead of staying exactly zero.  the
  //   cut-cell routines and the watertight check both compare against
  //   y = 0 exactly, so that drift would silently open the body up
  // pinning the axes to the identity keeps the map an exact copy plus a
  //   shift in x, and costs nothing: the body can only spin about x,
  //   which moves no geometry, so the body frame never rotates anyway

  if (axiflag) {
    ex[0] = 1.0; ex[1] = 0.0; ex[2] = 0.0;
    ey[0] = 0.0; ey[1] = 1.0; ey[2] = 0.0;
    ez[0] = 0.0; ez[1] = 0.0; ez[2] = 1.0;
    inertia1[0] = moi1[0];
    inertia1[1] = moi1[1];
    inertia1[2] = moi1[2];

    if (xcm[ibody][1] != 0.0 || xcm[ibody][2] != 0.0)
      error->all(FLERR,"Fix rigid body COM must lie on the axisymmetric "
                 "axis");
    if (inertia1[0] <= 0.0)
      error->all(FLERR,"Fix rigid moment of inertia about the "
                 "axisymmetric axis must be positive");

    // an element lying flat on the axis sweeps out no area, so it can
    //   neither be hit nor bound any volume, and it would confuse the
    //   profile-closure check above

    for (int i = bodystart[ibody]; i < bodystart[ibody+1]; i++)
      if (bodypt[i][0][1] == 0.0 && bodypt[i][1][1] == 0.0)
        error->all(FLERR,"Fix rigid body surf lies on the axisymmetric "
                   "axis");

    quat[ibody][0] = 1.0;
    quat[ibody][1] = quat[ibody][2] = quat[ibody][3] = 0.0;
    set_recoil(ibody);
    return;
  }

  // tensor = inertia tensor in space frame

  double tensor[3][3],evectors[3][3];

  tensor[0][0] = moi1[0];
  tensor[1][1] = moi1[1];
  tensor[2][2] = moi1[2];
  tensor[1][2] = tensor[2][1] = moi1[5];
  tensor[0][2] = tensor[2][0] = moi1[4];
  tensor[0][1] = tensor[1][0] = moi1[3];

  // diagonalize the inertia tensor to create body frame

  int ierror = MathEigen::jacobi3(tensor,inertia1,evectors,1);
  if (ierror) error->all(FLERR,"Insufficient Jacobi rotations for rigid body");

  ex[0] = evectors[0][0];
  ex[1] = evectors[1][0];
  ex[2] = evectors[2][0];
  ey[0] = evectors[0][1];
  ey[1] = evectors[1][1];
  ey[2] = evectors[2][1];
  ez[0] = evectors[0][2];
  ez[1] = evectors[1][2];
  ez[2] = evectors[2][2];

  // for 2d, insure the principal axis aligned with z is in the 3rd slot
  // the z axis is a principal axis b/c ixz = iyz = 0 is enforced for 2d

  if (dim == 2) {
    if (fabs(ez[2]) < 1.0-EPSILON) {
      if (fabs(ey[2]) > 1.0-EPSILON) {
        std::swap(inertia1[1],inertia1[2]);
        std::swap(ey[0],ez[0]);
        std::swap(ey[1],ez[1]);
        std::swap(ey[2],ez[2]);
      } else if (fabs(ex[2]) > 1.0-EPSILON) {
        std::swap(inertia1[0],inertia1[2]);
        std::swap(ex[0],ez[0]);
        std::swap(ex[1],ez[1]);
        std::swap(ex[2],ez[2]);
      } else
        body_error(ibody,"inertia tensor has no principal axis along z");
    }
  }

  // if any principal moment < scaled EPSILON, set to 0.0

  double max;
  max = MAX(inertia1[0],inertia1[1]);
  max = MAX(max,inertia1[2]);

  if (inertia1[0] < EPSILON*max) inertia1[0] = 0.0;
  if (inertia1[1] < EPSILON*max) inertia1[1] = 0.0;
  if (inertia1[2] < EPSILON*max) inertia1[2] = 0.0;

  // validity checks on principal moments of inertia
  // for 2d only the moment about the z axis matters
  // for 3d all must be positive and satisfy the triangle inequality,
  //   else the moi settings are not those of a physical rigid body
  // jacobi3() sorted the moments in increasing order

  if (dim == 2) {
    if (inertia1[2] <= 0.0)
      body_error(ibody,"moment of inertia about z axis must be positive");
  } else {
    if (inertia1[0] <= 0.0 || inertia1[1] <= 0.0 || inertia1[2] <= 0.0)
      body_error(ibody,"principal moments of inertia must be positive");
    if (inertia1[0] + inertia1[1] < (1.0-EPSILON)*inertia1[2])
      body_error(ibody,"moments of inertia do not satisfy "
                 "the triangle inequality");
  }

  // enforce 3 evectors as a right-handed coordinate system
  // flip 3rd vector if needed

  double cross[3];
  MathExtra::cross3(ex,ey,cross);
  if (MathExtra::dot3(cross,ez) < 0.0) MathExtra::negate3(ez);

  // create initial quaternion

  MathExtra::exyz_to_q(ex,ey,ez,quat[ibody]);

  set_recoil(ibody);
}

/* ----------------------------------------------------------------------
   finish one-time setup of one body, once its body frame axes are known
------------------------------------------------------------------------- */

void FixRigid::setup_body_displace(int ibody)
{
  // set displacement for each end/corner point in each line/tri
  // delta = vector from COM to end/corner point in space frame
  // displace = delta rotated to be in basis of principal axes, i.e. in body frame
  // corner pts come from the replicated body geometry built by gather_body()

  double delta[3];
  double *xcm1 = xcm[ibody];

  for (int i = bodystart[ibody]; i < bodystart[ibody+1]; i++)
    for (int j = 0; j < dim; j++) {
      delta[0] = bodypt[i][j][0] - xcm1[0];
      delta[1] = bodypt[i][j][1] - xcm1[1];
      if (dim == 3) delta[2] = bodypt[i][j][2] - xcm1[2];
      else delta[2] = 0.0;
      MathExtra::transpose_matvec(ex_space[ibody],ey_space[ibody],
                                  ez_space[ibody],delta,&displace[i][j][0]);
    }

  // initial omega, consistent with initial angmom

  MathExtra::angmom_to_omega(angmom[ibody],ex_space[ibody],ey_space[ibody],
                             ez_space[ibody],inertia[ibody],omega[ibody]);

  // rmaxbody = max distance of any body corner pt from the COM

  rmaxbody[ibody] = 0.0;
  for (int i = bodystart[ibody]; i < bodystart[ibody+1]; i++)
    for (int j = 0; j < dim; j++)
      rmaxbody[ibody] = MAX(rmaxbody[ibody],
                            MathExtra::len3(&displace[i][j][0]));

  // rminbody = lower bound on the distance from the COM to any point of
  //   any body element, used by incremental_recut() to reject a cell which
  //   lies wholly inside the body and so can hold no body surf
  // measured to each element's supporting line (2d) or plane (3d), not to
  //   its corner points: the closest point of an element may be interior to
  //   it, and a corner-point minimum would OVER-estimate the distance and
  //   so reject a cell that does hold a surf.  a distance to the supporting
  //   line/plane is always <= the distance to the element itself, which
  //   keeps the test conservative

  double dmin = BIG;
  for (int i = bodystart[ibody]; i < bodystart[ibody+1]; i++) {
    double *a = &displace[i][0][0];
    double *b = &displace[i][1][0];
    double e1[3],e2[3],nrm[3];
    MathExtra::sub3(b,a,e1);
    if (dim == 2) {
      e1[2] = 0.0;
      double len = MathExtra::len3(e1);
      if (len == 0.0) continue;

      // 2d: |a x e1| / |e1| is the distance from the origin (the COM in the
      //   body frame) to the line through a and b

      dmin = MIN(dmin,fabs(a[0]*e1[1] - a[1]*e1[0]) / len);
    } else {
      double *c = &displace[i][2][0];
      MathExtra::sub3(c,a,e2);
      MathExtra::cross3(e1,e2,nrm);
      double len = MathExtra::len3(nrm);
      if (len == 0.0) continue;

      // 3d: |a . n| / |n| is the distance from the origin to the plane of
      //   the triangle

      dmin = MIN(dmin,fabs(MathExtra::dot3(a,nrm)) / len);
    }
  }

  rminbody[ibody] = (dmin == BIG) ? 0.0 : dmin;
}

/* ----------------------------------------------------------------------
   bin static surfs for push-off candidate pruning
   built once per run in setup(); static surfs never move during a run
   each static surf is added to every bin its bbox overlaps (CSR layout);
     a query gathers the bins overlapping the pushcutoff-inflated body
     bbox and dedups multi-bin surfs with a visit stamp
   bin edge lengths are at least pushcutoff, at most 64 bins per dim
------------------------------------------------------------------------- */

void FixRigid::push_bins()
{
  int i,k,m,ibx,iby,ibz;
  int blo[3],bhi[3];

  // non-distributed: bin the static surfs of the local arrays, which
  //   hold all surfs on every proc (identical bins everywhere)
  // distributed: bin the static surfs this proc OWNS; each surf is
  //   binned on exactly one proc, so per-proc push contributions are
  //   disjoint partial sums
  // static = surf not in any rigid body

  int distributed = surf->distributed;

  Surf::Line *lines;
  Surf::Tri *tris;
  int nslocal;
  if (!distributed) {
    lines = surf->lines;
    tris = surf->tris;
    nslocal = surf->nlocal;
  } else {
    lines = surf->mylines;
    tris = surf->mytris;
    nslocal = surf->nown;
  }

  int *rigidmap = update->rigidmap;

  double *boxlo = domain->boxlo;
  double *boxhi = domain->boxhi;

  for (k = 0; k < 3; k++) {
    pushbinlo[k] = boxlo[k];
    double len = boxhi[k] - boxlo[k];
    int n = (int) (len/pushcutoff);
    n = MAX(n,1);
    n = MIN(n,64);
    if (dim == 2 && k == 2) n = 1;
    pushnbin[k] = n;
    pushbininv[k] = n/len;
  }
  int nbins = pushnbin[0]*pushnbin[1]*pushnbin[2];

  memory->destroy(pushbinstart);
  memory->destroy(pushbinlist);
  memory->destroy(pushstamp);
  memory->create(pushbinstart,nbins+1,"fix_rigid:pushbinstart");
  memory->create(pushstamp,nslocal,"fix_rigid:pushstamp");
  for (i = 0; i < nslocal; i++) pushstamp[i] = 0;
  pushstampcur = 0;

  // two passes: count entries per bin, then fill

  double slo[3],shi[3];

  for (int pass = 0; pass < 2; pass++) {
    if (pass == 0)
      for (i = 0; i <= nbins; i++) pushbinstart[i] = 0;

    for (m = 0; m < nslocal; m++) {

      // skip surfs belonging to any rigid body

      if (!distributed) {
        if (rigidmap[m] >= 0) continue;
      } else {
        surfint id = (dim == 2) ? lines[m].id : tris[m].id;
        if (body_elem(id) >= 0) continue;
      }

      if (dim == 2) {
        for (k = 0; k < 2; k++) {
          slo[k] = MIN(lines[m].p1[k],lines[m].p2[k]);
          shi[k] = MAX(lines[m].p1[k],lines[m].p2[k]);
        }
        slo[2] = shi[2] = 0.0;
      } else {
        for (k = 0; k < 3; k++) {
          slo[k] = MIN(tris[m].p1[k],MIN(tris[m].p2[k],tris[m].p3[k]));
          shi[k] = MAX(tris[m].p1[k],MAX(tris[m].p2[k],tris[m].p3[k]));
        }
      }

      for (k = 0; k < 3; k++) {
        blo[k] = (int) ((slo[k]-pushbinlo[k]) * pushbininv[k]);
        bhi[k] = (int) ((shi[k]-pushbinlo[k]) * pushbininv[k]);
        blo[k] = MAX(0,MIN(blo[k],pushnbin[k]-1));
        bhi[k] = MAX(0,MIN(bhi[k],pushnbin[k]-1));
      }

      for (ibz = blo[2]; ibz <= bhi[2]; ibz++)
        for (iby = blo[1]; iby <= bhi[1]; iby++)
          for (ibx = blo[0]; ibx <= bhi[0]; ibx++) {
            int ibin = (ibz*pushnbin[1] + iby)*pushnbin[0] + ibx;
            if (pass == 0) pushbinstart[ibin+1]++;
            else pushbinlist[pushbinstart[ibin]++] = m;
          }
    }

    if (pass == 0) {
      for (i = 0; i < nbins; i++) pushbinstart[i+1] += pushbinstart[i];
      memory->create(pushbinlist,pushbinstart[nbins],
                     "fix_rigid:pushbinlist");
    } else {
      // filling advanced the starts by one bin: shift them back
      for (i = nbins; i > 0; i--) pushbinstart[i] = pushbinstart[i-1];
      pushbinstart[0] = 0;
    }
  }
}

/* ----------------------------------------------------------------------
   bin bodies by COM, so that the bodies near another body, a grid
     cell, or a point can be found without a loop over all bodies
   bin edge lengths are at least the largest body diameter plus the
     push-off cutoff, so a body in contact with another or overlapping a
     query box is never further than one bin away from it
   total # of bins is capped at about twice the # of bodies, so the
     bin arrays stay small however small the bodies are
   rebuilt each step after the bodies move, from their current bboxes
------------------------------------------------------------------------- */

void FixRigid::body_bins()
{
  int i,k,ibin;
  int ib[3];

  double *boxlo = domain->boxlo;
  double *boxhi = domain->boxhi;

  // rmaxall = how far any body's bbox can extend from its COM

  rmaxall = 0.0;
  for (i = 0; i < nbody; i++)
    rmaxall = MAX(rmaxall,rmaxbody[i] + bboxeps[i]);

  double edge = 2.0*rmaxall;
  if (pushflag) edge += pushcutoff;

  double vol = 1.0;
  for (k = 0; k < dim; k++) vol *= boxhi[k] - boxlo[k];
  double scale = pow(2.0*nbody/vol,1.0/dim);

  for (k = 0; k < 3; k++) {
    bodybinlo[k] = boxlo[k];
    double len = boxhi[k] - boxlo[k];
    int n = (int) (len*scale);
    if (edge > 0.0) n = MIN(n,(int) (len/edge));
    n = MAX(n,1);
    if (dim == 2 && k == 2) n = 1;
    bodynbin[k] = n;
    bodybininv[k] = n/len;
  }
  int nbins = bodynbin[0]*bodynbin[1]*bodynbin[2];

  memory->destroy(bodybinstart);
  memory->destroy(bodybinlist);
  memory->create(bodybinstart,nbins+1,"fix_rigid:bodybinstart");
  memory->create(bodybinlist,nbody,"fix_rigid:bodybinlist");

  // two passes: count bodies per bin, then fill
  // a body outside the box is binned in the nearest edge bin

  for (int pass = 0; pass < 2; pass++) {
    if (pass == 0)
      for (i = 0; i <= nbins; i++) bodybinstart[i] = 0;

    for (i = 0; i < nbody; i++) {
      for (k = 0; k < 3; k++) {
        ib[k] = (int) ((xcm[i][k]-bodybinlo[k]) * bodybininv[k]);
        ib[k] = MAX(0,MIN(ib[k],bodynbin[k]-1));
      }
      ibin = (ib[2]*bodynbin[1] + ib[1])*bodynbin[0] + ib[0];
      if (pass == 0) bodybinstart[ibin+1]++;
      else bodybinlist[bodybinstart[ibin]++] = i;
    }

    if (pass == 0) {
      for (i = 0; i < nbins; i++) bodybinstart[i+1] += bodybinstart[i];
    } else {
      // filling advanced the starts by one bin: shift them back
      for (i = nbins; i > 0; i--) bodybinstart[i] = bodybinstart[i-1];
      bodybinstart[0] = 0;
    }
  }
}

/* ----------------------------------------------------------------------
   list = bodies whose bbox overlaps the box lo/hi, return the count
   scans the bins overlapping the box inflated by rmaxall, since a
     body's bbox extends at most that far from the COM it is binned by,
     then tests each body's bbox exactly
   the list is a single buffer, overwritten by the next query
------------------------------------------------------------------------- */

int FixRigid::body_box(double *lo, double *hi, int **list)
{
  int i,k,ibin,ibody;
  int blo[3],bhi[3];

  for (k = 0; k < 3; k++) {
    blo[k] = (int) ((lo[k]-rmaxall-bodybinlo[k]) * bodybininv[k]);
    bhi[k] = (int) ((hi[k]+rmaxall-bodybinlo[k]) * bodybininv[k]);
    blo[k] = MAX(0,MIN(blo[k],bodynbin[k]-1));
    bhi[k] = MAX(0,MIN(bhi[k],bodynbin[k]-1));
  }

  int n = 0;

  for (int ibz = blo[2]; ibz <= bhi[2]; ibz++)
    for (int iby = blo[1]; iby <= bhi[1]; iby++)
      for (int ibx = blo[0]; ibx <= bhi[0]; ibx++) {
        ibin = (ibz*bodynbin[1] + iby)*bodynbin[0] + ibx;
        for (i = bodybinstart[ibin]; i < bodybinstart[ibin+1]; i++) {
          ibody = bodybinlist[i];
          if (!box_overlap(lo,hi,bbodylo[ibody],bbodyhi[ibody])) continue;
          if (n == maxbodycand) {
            maxbodycand += DELTA_MODIFY;
            memory->grow(bodycand,maxbodycand,"fix_rigid:bodycand");
          }
          bodycand[n++] = ibody;
        }
      }

  *list = bodycand;
  return n;
}

/* ----------------------------------------------------------------------
   contact forces between all corner pts of this body and one source
     element with corner pts p1,p2 (p3 for 3d) and outward normal norm
   for each body corner pt within pushcutoff of the element, apply a
     repulsive force directed from the closest point of the element to
     the corner pt (the element outward normal for a face-on contact),
     with overlap delta = pushcutoff - dist:
     linear spring F = kpush * delta, or
     Hertzian contact F = kpush * delta^3/2 (smooth onset, standard
     model for elastic contact of spherical particulates)
   every element within the cutoff contributes, on either side of it:
     each force is the gradient of its own spring potential in the
     distance to the element, so the total is the gradient of the sum
     and the contacts conserve energy exactly
   a corner pt inside a wall thinner than 2*pushcutoff is repelled by
     both faces at once; their potentials add to a barrier whose peak
     is at the near surface, so the wall repels the body rather than
     driving it through, and the body passes only if it arrives with
     more energy than the barrier
   if gammapush > 0, a dashpot term F += gammapush * d(delta)/dt is
     added (the DEM spring-dashpot pair), i.e. minus gammapush times
     the normal separation rate of the corner pt relative to the source
     surface; the total contact force is clamped at zero, so the
     dashpot never produces adhesion as a contact ends
   ibody = the body whose corner pts are tested
   jbody = the rigid body the element belongs to, or -1 if static
   if jbody >= 0, the reaction force -F is applied to it at the same
     contact point, so body-body contacts conserve momentum exactly
------------------------------------------------------------------------- */

void FixRigid::push_contact(int ibody, double *p1, double *p2, double *p3,
                            double *norm, int jbody)
{
  int i,j;
  double dsq,d,scale;
  double **pts;
  double fone[3],rdelta[3],tq[3],cp[3],fdir[3];

  int npoint = dim;     // 2 corner pts per line, 3 per tri
  double cutsq = pushcutoff*pushcutoff;

  double *xcm1 = xcm[ibody];
  double *fpush1 = fpush[ibody];
  double *tqpush1 = tqpush[ibody];

  // bounding box of the source element inflated by the cutoff:
  //   only body elements whose own box overlaps it can be in contact

  double slo[3],shi[3];
  for (int k = 0; k < 3; k++) {
    slo[k] = MIN(p1[k],p2[k]);
    shi[k] = MAX(p1[k],p2[k]);
    if (dim == 3) {
      slo[k] = MIN(slo[k],p3[k]);
      shi[k] = MAX(shi[k],p3[k]);
    }
    slo[k] -= pushcutoff;
    shi[k] += pushcutoff;
  }

  for (i = bodystart[ibody]; i < bodystart[ibody+1]; i++) {
    if (elemlo[i][0] > shi[0] || elemhi[i][0] < slo[0]) continue;
    if (elemlo[i][1] > shi[1] || elemhi[i][1] < slo[1]) continue;
    if (dim == 3 && (elemlo[i][2] > shi[2] || elemhi[i][2] < slo[2]))
      continue;
    pts = bodypt[i];

    for (j = 0; j < npoint; j++) {

      if (dim == 2)
        dsq = Geometry::closest_point_line(pts[j],p1,p2,cp);
      else
        dsq = Geometry::closest_point_tri(pts[j],p1,p2,p3,norm,cp);
      if (dsq >= cutsq) continue;

      d = sqrt(dsq);
      if (pushstyle == LINEAR) scale = kpush * (pushcutoff-d);
      else scale = kpush * (pushcutoff-d) * sqrt(pushcutoff-d);

      // force direction = from the closest point of the element to the
      //   corner pt, the gradient of the spring potential in d: equals
      //   the element normal when the closest feature is the interior,
      //   and stays conservative when it is an edge or vertex, as it is
      //   for most contacts with a faceted curved surface
      // a corner pt on the element (d = 0) is pushed along the normal

      if (d > 0.0) {
        fdir[0] = (pts[j][0]-cp[0]) / d;
        fdir[1] = (pts[j][1]-cp[1]) / d;
        fdir[2] = (pts[j][2]-cp[2]) / d;
      } else {
        fdir[0] = norm[0]; fdir[1] = norm[1]; fdir[2] = norm[2];
      }

      // dashpot: damp by the normal approach rate of the corner pt
      //   relative to the source surface,
      //   which moves if it belongs to another rigid body

      if (gammapush > 0.0) {
        double vpt[3],vsrc[3],rd[3];
        MathExtra::sub3(pts[j],xcm1,rd);
        MathExtra::cross3(omega[ibody],rd,vpt);
        MathExtra::add3(vcm[ibody],vpt,vpt);
        if (jbody >= 0) {
          MathExtra::sub3(pts[j],xcm[jbody],rd);
          MathExtra::cross3(omega[jbody],rd,vsrc);
          MathExtra::add3(vcm[jbody],vsrc,vsrc);
          MathExtra::sub3(vpt,vsrc,vpt);
        }
        scale -= gammapush * MathExtra::dot3(vpt,fdir);
        if (scale < 0.0) scale = 0.0;
      }

      // axisymmetric: the corner pt stands for a ring of radius r and
      //   the source element for another, so the spring law gives a
      //   force per unit length of contact and the total is 2 pi r
      //   times it.  a pt on the axis then feels no push, which is
      //   right: its ring has no circumference.  the force stays in the
      //   (x,r) plane, so a contact exerts no torque about the axis and
      //   cannot spin the body

      if (axiflag) scale *= MY_2PI * pts[j][1];

      fone[0] = scale*fdir[0];
      fone[1] = scale*fdir[1];
      fone[2] = scale*fdir[2];

      fpush1[0] += fone[0];
      fpush1[1] += fone[1];
      fpush1[2] += fone[2];
      MathExtra::sub3(pts[j],xcm1,rdelta);
      MathExtra::cross3(rdelta,fone,tq);
      tqpush1[0] += tq[0];
      tqpush1[1] += tq[1];
      tqpush1[2] += tq[2];

      // equal-and-opposite reaction on the source body,
      //   applied at the same contact point

      if (jbody >= 0) {
        fpush[jbody][0] -= fone[0];
        fpush[jbody][1] -= fone[1];
        fpush[jbody][2] -= fone[2];
        MathExtra::sub3(pts[j],xcm[jbody],rdelta);
        MathExtra::cross3(rdelta,fone,tq);
        tqpush[jbody][0] -= tq[0];
        tqpush[jbody][1] -= tq[1];
        tqpush[jbody][2] -= tq[2];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   push-off forces on body ibody from too-close static surfs, other
     rigid bodies, and (if pushboundflag) non-periodic box boundaries
   called for each body, after all bodies have committed end-of-step
     geometry
   static surf candidates come from the bins built by push_bins();
     other bodies are pruned by a body-body bbox test, then per element
   forces accumulate in fpush/tqpush; the caller adds them into
     fcm/torque for the next step's time integration
   non-distributed surfs: computed identically on every proc, so no
     communication is needed; distributed surfs: per-proc partial sums
     which the caller merges with one Allreduce
   NOTE: a corner pt shared by adjacent body elements contributes once
     per element, and a corner close to several source elements
     interacts with each of them, so kpush is a per-contact stiffness;
     two bodies engage the corner pts of each against the elements of
     the other, about twice the contacts of one body against a static
     surf of the same shape (each set is a distinct geometric contact,
     and dropping either would make the force depend on body order)
------------------------------------------------------------------------- */

void FixRigid::push_off(int ibody)
{
  int i,j,m,e,jbody;
  double d,scale;
  double **pts;
  double fone[3],rdelta[3],tq[3];
  int blo[3],bhi[3];

  double *xcm1 = xcm[ibody];
  double *fpush1 = fpush[ibody];
  double *tqpush1 = tqpush[ibody];

  // static surf sources:
  //   non-distributed: local surf arrays hold all surfs on every proc,
  //     every proc computes the identical full contribution
  //   distributed: each proc's bins hold only the static surfs it OWNS,
  //     so contributions are disjoint partial sums, merged by
  //     end_of_step() with an Allreduce
  // pair and boundary contributions are identical on every proc, so
  //   for distributed surfs only proc 0 computes them before the merge

  int distributed = surf->distributed;
  Surf::Line *lines;
  Surf::Tri *tris;
  if (!distributed) {
    lines = surf->lines;
    tris = surf->tris;
  } else {
    lines = surf->mylines;
    tris = surf->mytris;
  }

  int npoint = dim;     // 2 corner pts per line, 3 per tri

  // cutlo/cuthi = bbox around body inflated by pushcutoff
  // requires body_bbox() was called for current body position

  double cutlo[3],cuthi[3];
  for (j = 0; j < 3; j++) {
    cutlo[j] = bbodylo[ibody][j] - pushcutoff;
    cuthi[j] = bbodyhi[ibody][j] + pushcutoff;
  }

  // static surf candidates: bins overlapping the inflated body bbox
  // stamp dedups surfs binned into more than one of the bins

  for (j = 0; j < 3; j++) {
    blo[j] = (int) ((cutlo[j]-pushbinlo[j]) * pushbininv[j]);
    bhi[j] = (int) ((cuthi[j]-pushbinlo[j]) * pushbininv[j]);
    blo[j] = MAX(0,MIN(blo[j],pushnbin[j]-1));
    bhi[j] = MAX(0,MIN(bhi[j],pushnbin[j]-1));
  }

  pushstampcur++;

  for (int ibz = blo[2]; ibz <= bhi[2]; ibz++)
    for (int iby = blo[1]; iby <= bhi[1]; iby++)
      for (int ibx = blo[0]; ibx <= bhi[0]; ibx++) {
        int ibin = (ibz*pushnbin[1] + iby)*pushnbin[0] + ibx;
        for (i = pushbinstart[ibin]; i < pushbinstart[ibin+1]; i++) {
          m = pushbinlist[i];
          if (pushstamp[m] == pushstampcur) continue;
          pushstamp[m] = pushstampcur;

          if (dim == 2) {
            if (MAX(lines[m].p1[0],lines[m].p2[0]) < cutlo[0]) continue;
            if (MIN(lines[m].p1[0],lines[m].p2[0]) > cuthi[0]) continue;
            if (MAX(lines[m].p1[1],lines[m].p2[1]) < cutlo[1]) continue;
            if (MIN(lines[m].p1[1],lines[m].p2[1]) > cuthi[1]) continue;
            push_contact(ibody,lines[m].p1,lines[m].p2,NULL,
                         lines[m].norm,-1);
          } else {
            if (MAX(tris[m].p1[0],MAX(tris[m].p2[0],tris[m].p3[0])) <
                cutlo[0]) continue;
            if (MIN(tris[m].p1[0],MIN(tris[m].p2[0],tris[m].p3[0])) >
                cuthi[0]) continue;
            if (MAX(tris[m].p1[1],MAX(tris[m].p2[1],tris[m].p3[1])) <
                cutlo[1]) continue;
            if (MIN(tris[m].p1[1],MIN(tris[m].p2[1],tris[m].p3[1])) >
                cuthi[1]) continue;
            if (MAX(tris[m].p1[2],MAX(tris[m].p2[2],tris[m].p3[2])) <
                cutlo[2]) continue;
            if (MIN(tris[m].p1[2],MIN(tris[m].p2[2],tris[m].p3[2])) >
                cuthi[2]) continue;
            push_contact(ibody,tris[m].p1,tris[m].p2,tris[m].p3,
                         tris[m].norm,-1);
          }
        }
      }

  // other rigid bodies: those whose bbox overlaps the inflated body
  //   bbox, from the body bins, then per-element bbox tests using the
  //   current-position element boxes set by body_bbox(0) when each
  //   body committed its end-of-step geometry
  // each contact applies equal-and-opposite forces to both bodies
  // distributed: proc 0 alone computes the pair and boundary
  //   contributions, so that the Allreduce which merges the static
  //   contributions sums them once and in the same order as a
  //   non-distributed run, which computes them on every proc
  // the bins keep this proc-0 work small: a body only tests the bodies
  //   in its neighboring bins

  int mine = 1;
  if (distributed && comm->me) mine = 0;

  if (mine) {
    int *jlist;
    int nj = body_box(cutlo,cuthi,&jlist);

    for (int jj = 0; jj < nj; jj++) {
      jbody = jlist[jj];
      if (jbody == ibody) continue;

      for (e = bodystart[jbody]; e < bodystart[jbody+1]; e++) {
        if (!box_overlap(cutlo,cuthi,elemlo[e],elemhi[e])) continue;
        if (dim == 2)
          push_contact(ibody,bodypt[e][0],bodypt[e][1],NULL,
                       bodynorm[e],jbody);
        else
          push_contact(ibody,bodypt[e][0],bodypt[e][1],bodypt[e][2],
                       bodynorm[e],jbody);
      }
    }
  }

  // spring force from non-periodic simulation box boundaries
  // corner pts from the replicated body geometry

  if (pushboundflag && mine) {
    double *boxlo = domain->boxlo;
    double *boxhi = domain->boxhi;
    int *bflag = domain->bflag;

    int nface = 2*dim;
    double fsign[6] = {1.0,-1.0,1.0,-1.0,1.0,-1.0};

    for (i = bodystart[ibody]; i < bodystart[ibody+1]; i++) {
      pts = bodypt[i];

      for (j = 0; j < npoint; j++) {
        for (int iface = 0; iface < nface; iface++) {
          if (bflag[iface] == PERIODIC) continue;

          // the axisymmetric axis is not a wall: a body of revolution
          //   is expected to reach r = 0, and pushing it off the axis
          //   would break the symmetry it is built on

          if (bflag[iface] == AXISYM) continue;

          int idim = iface/2;
          if (iface % 2 == 0) d = pts[j][idim] - boxlo[idim];
          else d = boxhi[idim] - pts[j][idim];
          if (d >= pushcutoff) continue;
          if (d < 0.0) d = 0.0;      // past the face: max force k*cutoff

          if (pushstyle == LINEAR) scale = kpush * (pushcutoff-d);
          else scale = kpush * (pushcutoff-d) * sqrt(pushcutoff-d);

          // dashpot vs the static boundary, face normal = fsign*e_idim

          if (gammapush > 0.0) {
            double vpt[3],rd[3];
            MathExtra::sub3(pts[j],xcm1,rd);
            MathExtra::cross3(omega[ibody],rd,vpt);
            MathExtra::add3(vcm[ibody],vpt,vpt);
            scale -= gammapush * fsign[iface] * vpt[idim];
            if (scale < 0.0) scale = 0.0;
          }

          scale *= fsign[iface];
          if (axiflag) scale *= MY_2PI * pts[j][1];
          fone[0] = fone[1] = fone[2] = 0.0;
          fone[idim] = scale;

          fpush1[0] += fone[0];
          fpush1[1] += fone[1];
          fpush1[2] += fone[2];
          MathExtra::sub3(pts[j],xcm1,rdelta);
          MathExtra::cross3(rdelta,fone,tq);
          tqpush1[0] += tq[0];
          tqpush1[1] += tq[1];
          tqpush1[2] += tq[2];
        }
      }
    }
  }

  // fpush/tqpush are merged into fcm/torque by end_of_step() after all
  //   bodies' push-off forces, including reactions from other bodies'
  //   contacts, have been accumulated
}

/* ----------------------------------------------------------------------
   full re-map of surfs to grid cells
   same sequence of operations as in FixMoveSurf::end_of_step()
   called every step (cutcell mode) or on incremental-mode fallback,
     after body surfs move, to cut/split cells and set INSIDE/OUTSIDE
     cell typing from the new body positions
------------------------------------------------------------------------- */

void FixRigid::grid_rebuild()
{
  // every surf compute tallying this step must first bring its tallies
  //   to the host, keyed by surf ID: the KOKKOS variants of compute surf
  //   and compute react/surf index their device tallies by local surf
  //   index, which the rebuild of the surf arrays below invalidates
  //   (distributed surfs), and re-size their per-surf tally index when
  //   they re-allocate after the rebuild, which discards device tallies
  //   not yet fetched
  // must precede any change to the local+ghost surf arrays
  // no-op for a compute whose tallies were already fetched, and for the
  //   non-KOKKOS computes

  for (int m = 0; m < update->nsurf_tally; m++) {
    Compute *c = update->slist_active[m];
    surfint *t2s;
    if (strcmp(c->style,"surf") == 0 || strcmp(c->style,"surf/kk") == 0)
      ((ComputeSurf *) c)->tallyinfo(t2s);
    else if (strcmp(c->style,"react/surf") == 0 ||
             strcmp(c->style,"react/surf/kk") == 0)
      ((ComputeReactSurf *) c)->tallyinfo(t2s);
  }

  // sort particles, grid rebuild requires it

  if (particle->exist) {
    particles_to_host();
    particle->sort();
  }

  // assign split cell particles to parent split cell

  grid->unset_neighbors();
  grid->remove_ghosts();

  if (grid->nsplitlocal) {
    Grid::ChildCell *cells = grid->cells;
    int nglocal = grid->nlocal;
    for (int icell = 0; icell < nglocal; icell++)
      if (cells[icell].nsplit > 1)
        grid->combine_split_cell_particles(icell,1);
  }

  // assign surfs to grid cells

  grid->clear_surf();
  grid->surf2grid(1,0);

  // re-setup owned and ghost cell info

  grid->setup_owned();
  grid->acquire_ghosts();
  grid->reset_neighbors();
  comm->reset_neighbors();

  // flag cells and corners as OUTSIDE or INSIDE

  grid->set_inout();
  grid->type_check(0);

  // notify all classes that store per-grid data that grid may have changed
  // invokes grid_changed() of every rigid fix, which re-establishes the
  //   local body-surf copies and the per-surf rigidmap for the rebuilt
  //   local/ghost surf arrays, before per-surf computes re-size

  // as after a load balance: distributed local/ghost surf arrays were
  //   rebuilt, so per-surf custom values must be re-spread

  if (surf->distributed) {
    surf->localghost_changed_step = update->ntimestep;
    for (int i = 0; i < surf->ncustom; i++) surf->estatus[i] = 0;
  }

  grid->notify_changed();
}

/* ----------------------------------------------------------------------
   add every body's surfs to the collision lists (csurfs) of all grid
     cells they sweep through during this step, so particles anywhere in
     a body's swept path are tested against the moving surfs and
     reflected (rather than overtaken by a fast body and deleted)
   called each start_of_step, after the end-of-step pose of every body
     is known; a single pass over the grid cells handles all bodies
   for each overlapped cell a merged list = its current csurfs plus all
     swept surfs not already present is installed; the original is
     saved for swept_restore(); split cells share the merged list with
     their sub cells, where particles reside
   cut-cell volumes are not changed: this augments collision lists only
------------------------------------------------------------------------- */

void FixRigid::swept_assign_all()
{
  int i,j,ibody,icell,isub,ncur,isplit,dup,nmerged;
  surfint *merged,*cur;

  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;
  int ntotal = grid->nlocal + grid->nghost;

  // per-element swept bounding boxes of every body were set by
  //   body_bbox(ibody,1) in start_of_step()

  cpage->reset();
  nmodified = 0;

  // phase 1: gather (cell, swept element) entries per body, visiting
  //   only the candidate cells near each body from the box->cell
  //   index, so cost scales with the bodies' swept regions and not
  //   with the number of cells this proc owns
  // entries for one cell are chained; a per-cell stamp detects the
  //   first touch of a cell this step

  if (ntotal > maxswcell) {
    int oldmax = maxswcell;
    maxswcell = ntotal;
    memory->grow(swstamp,maxswcell,"fix_rigid:swstamp");
    memory->grow(swhead,maxswcell,"fix_rigid:swhead");
    for (i = oldmax; i < maxswcell; i++) swstamp[i] = 0;
  }
  swcur++;
  nswcell = 0;
  nent = 0;

  int ncand,icand;
  int *cand;

  for (ibody = 0; ibody < nbody; ibody++) {
    double *blo = bbodylo[ibody];
    double *bhi = bbodyhi[ibody];
    ncand = update->rigid_cell_box(blo,bhi,&cand);

    for (icand = 0; icand < ncand; icand++) {
      icell = cand[icand];
      if (cells[icell].nsplit <= 0) continue;
      if (cells[icell].nsurf < 0) continue;
      if (!box_overlap(cells[icell].lo,cells[icell].hi,blo,bhi)) continue;

      for (i = bodystart[ibody]; i < bodystart[ibody+1]; i++) {
        if (!box_overlap(cells[icell].lo,cells[icell].hi,
                         elemlo[i],elemhi[i])) continue;

        if (swstamp[icell] != swcur) {
          swstamp[icell] = swcur;
          swhead[icell] = -1;
          if (nswcell == maxswcells) {
            maxswcells += DELTA_MODIFY;
            memory->grow(swcells,maxswcells,"fix_rigid:swcells");
          }
          swcells[nswcell++] = icell;
        }

        // lblist = local surf index of the element on this proc;
        // for distributed surfs grid_changed() keeps it current

        if (nent == maxent) {
          maxent += DELTA_MODIFY;
          memory->grow(entnext,maxent,"fix_rigid:entnext");
          memory->grow(entelem,maxent,"fix_rigid:entelem");
        }
        entelem[nent] = (surfint) lblist[i];
        entnext[nent] = swhead[icell];
        swhead[icell] = nent++;
      }
    }
  }

  // phase 2: for each touched cell, install a merged list =
  //   current csurfs + chained swept elements not already present
  // dedup vs current csurfs skips body surfs the cut pipeline placed
  //   at a body's start-of-step position; chained entries are unique
  //   among themselves (bodies are disjoint, one entry per element)

  for (int ic = 0; ic < nswcell; ic++) {
    icell = swcells[ic];

    ncur = cells[icell].nsurf;
    cur = cells[icell].csurfs;
    merged = cpage->vget();
    if (!merged)
      error->one(FLERR,"Failed to allocate fix rigid swept surf lists");
    for (j = 0; j < ncur; j++) merged[j] = cur[j];
    nmerged = ncur;

    for (int e = swhead[icell]; e >= 0; e = entnext[e]) {
      surfint selem = entelem[e];
      dup = 0;
      for (j = 0; j < ncur; j++)
        if (cur[j] == selem) { dup = 1; break; }
      if (!dup) merged[nmerged++] = selem;
    }

    if (nmerged == ncur) continue;   // all swept surfs already present
    cpage->vgot(nmerged);

    // save cell settings so they can be restored, then override them
    // install the merged list where particles reside: the cell itself
    //   if unsplit, else only its sub cells; the split cell's own list
    //   must keep its original length, since Update::split2d/3d()
    //   index the sinfo csplits array in lockstep with it

    if (nmodified+MAX(cells[icell].nsplit,1) > maxmodified) {
      while (nmodified+MAX(cells[icell].nsplit,1) > maxmodified)
        maxmodified += DELTA_MODIFY;
      memory->grow(modified,maxmodified,"fix_rigid:modified");
      memory->grow(nsurf_saved,maxmodified,"fix_rigid:nsurf_saved");
      csurfs_saved = (surfint **)
        memory->srealloc(csurfs_saved,maxmodified*sizeof(surfint *),
                         "fix_rigid:csurfs_saved");
    }

    if (cells[icell].nsplit == 1) {
      modified[nmodified] = icell;
      nsurf_saved[nmodified] = cells[icell].nsurf;
      csurfs_saved[nmodified] = cells[icell].csurfs;
      nmodified++;
      cells[icell].nsurf = nmerged;
      cells[icell].csurfs = merged;
    } else {
      isplit = cells[icell].isplit;
      for (j = 0; j < cells[icell].nsplit; j++) {
        isub = sinfo[isplit].csubs[j];
        modified[nmodified] = isub;
        nsurf_saved[nmodified] = cells[isub].nsurf;
        csurfs_saved[nmodified] = cells[isub].csurfs;
        nmodified++;
        cells[isub].nsurf = nmerged;
        cells[isub].csurfs = merged;
      }
    }
  }

  // per-cell surf lists changed on the host (read by fix rigid/kk)

  if (nmodified) listschanged = 1;
}

/* ----------------------------------------------------------------------
   restore the csurfs collision lists augmented by swept_assign()
------------------------------------------------------------------------- */

void FixRigid::swept_restore()
{
  Grid::ChildCell *cells = grid->cells;

  if (nmodified) listschanged = 1;
  for (int m = 0; m < nmodified; m++) {
    int icell = modified[m];
    cells[icell].nsurf = nsurf_saved[m];
    cells[icell].csurfs = csurfs_saved[m];
  }
  nmodified = 0;
}

/* ----------------------------------------------------------------------
   grid cells were rebuilt, adapted, or migrated to other procs
   called via Grid::notify_changed(), after the new owned cells and
     ghost cells (and for distributed surfs, the local/ghost surf
     arrays) are in place, and before per-surf computes re-size
   any merged csurfs lists were discarded by the grid rebuild; csurfs
     lists installed by incremental re-cutting were copied into grid
     storage by Grid::compress() or discarded by Grid::clear_surf(),
     except on a proc which migrated no cells and so skipped compress(),
     whose cells still reference them (handled below)
   next re-map re-cuts body surfs into the new grid cells
------------------------------------------------------------------------- */

void FixRigid::grid_changed()
{
  nmodified = 0;
  if (cpage) cpage->reset();

  // lists installed by incremental re-cutting may still be referenced
  //   by owned cells: a proc which migrated no cells skips
  //   Grid::compress() and so keeps its cells' csurfs pointers;
  //   copy those lists into grid storage before freeing them

  // during split_rebuild() the lists this fix installed are still in
  //   the cells and still current, unlike after a full re-map where
  //   Grid::clear_surf() has already thrown them away

  if (!insplitrebuild) {
    copy_registry_to_grid();
    free_registry();
    copy_split_registry_to_grid();
    free_split_registry();
  }
  listschanged = 1;
  update->rigid_bins_clear();

  // distributed surfs: the local surf arrays were rebuilt, so
  //   re-establish this fix's local body-surf copies and the per-surf
  //   rigidmap, which must also span the newly acquired ghost surfs
  // per-surf computes re-size after all fixes are notified

  if (surf->distributed) {
    for (int ibody = 0; ibody < nbody; ibody++) body_bbox(ibody,0);
    proc_bbox();
    int changed = ensure_local_copies();
    update->build_rigidmap();

    // a fix earlier in the notification may already have re-spread
    //   per-surf custom values over the pre-append layout: invalidate
    //   them again so they are re-spread over the final one
    // collective: the flags below gate collective re-spreads (a custom
    //   attribute in FixEmitSurf::init(), SurfCollide::dynamic()), so
    //   every proc must reset them or none; whether copies were
    //   appended differs by proc, so the decision is reduced first

    int changed_any;
    MPI_Allreduce(&changed,&changed_any,1,MPI_INT,MPI_MAX,world);
    if (changed_any) {
      surf->localghost_changed_step = update->ntimestep;
      for (int i = 0; i < surf->ncustom; i++) surf->estatus[i] = 0;
    }
  }
}

/* ----------------------------------------------------------------------
   for incremental remap: record cells interior to any body,
     i.e. INSIDE cells cut by no surf whose center is within a body
   a cell whose only surfs are transparent is not cut, and is typed
     INSIDE/OUTSIDE like a surf-free cell
   called before the body surfs move to their end-of-step positions
------------------------------------------------------------------------- */

void FixRigid::record_oldinside()
{
  double ctr[3];
  int ibody,icell;

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  int nglocal = grid->nlocal;

  noldinside = 0;

  // candidate cells near each body from the box->cell index,
  //   restricted to owned cells
  // bodies are disjoint, so a cell center is interior to at most one

  for (ibody = 0; ibody < nbody; ibody++) {

    // bbox around body at its current (pre-move) position

    body_bbox(ibody,0);
    double *blo = bbodylo[ibody];
    double *bhi = bbodyhi[ibody];

    int *cand;
    int ncand = update->rigid_cell_box(blo,bhi,&cand);

    for (int ic = 0; ic < ncand; ic++) {
      icell = cand[ic];
      if (icell >= nglocal) continue;
      if (cells[icell].nsplit != 1) continue;
      if (cell_cut(icell)) continue;
      if (cinfo[icell].type != CELLINSIDE) continue;
      if (!box_overlap(cells[icell].lo,cells[icell].hi,blo,bhi)) continue;

      ctr[0] = 0.5 * (cells[icell].lo[0] + cells[icell].hi[0]);
      ctr[1] = 0.5 * (cells[icell].lo[1] + cells[icell].hi[1]);
      if (dim == 3) ctr[2] = 0.5 * (cells[icell].lo[2] + cells[icell].hi[2]);
      else ctr[2] = 0.0;
      if (!inside_body(ibody,ctr)) continue;

      if (noldinside == maxoldinside) {
        maxoldinside += DELTA_MODIFY;
        memory->grow(oldinside,maxoldinside,"fix_rigid:oldinside");
      }
      oldinside[noldinside++] = icell;
    }
  }
}

/* ----------------------------------------------------------------------
   a ghost copy of a split cell whose piece count changed
   this proc cannot restructure a cell it does not own, and does not
     need to: the owner records the same change, so split_rebuild()
     runs this step and re-acquires every ghost before the next move
   until then the copy is marked unsplit, so that nothing indexes its
     piece map, whose length no longer matches its surf list
------------------------------------------------------------------------- */

void FixRigid::split_ghost_drop(int icell)
{
  grid->cells[icell].nsplit = 1;
  grid->cells[icell].isplit = -1;
}

/* ----------------------------------------------------------------------
   record that one owned cell's number of flow pieces changes this step
   the cut of the next cell overwrites the work buffers, so the piece
     map and the piece volumes are copied out here; the map is a
     fix-owned array which becomes sinfo[].csplits when it is applied,
     and the sub cell list is allocated now and filled in then
   nsplitnew = 1 means the cell stops being a split cell, and needs no
     map, no sub cell list and no volumes
------------------------------------------------------------------------- */

void FixRigid::split_pending(int icell, int nsplitnew, int n, int *map,
                             int xsub, double *xsplit, double *vols)
{
  if (npending == maxpending) {
    int oldmax = maxpending;
    maxpending += DELTA_MODIFY;
    pending = (PendingSplit *)
      memory->srealloc(pending,maxpending*sizeof(PendingSplit),
                       "fix_rigid:pending");
    for (int m = oldmax; m < maxpending; m++) {
      pending[m].map = NULL;
      pending[m].maxmap = 0;
      pending[m].vols = NULL;
      pending[m].maxvols = 0;
    }
  }

  PendingSplit *p = &pending[npending++];
  p->icell = icell;
  p->nsplitnew = nsplitnew;
  p->nsurf = n;
  p->csplits = NULL;
  p->csubs = NULL;

  if (nsplitnew == 1) return;

  // the arrays the cell will own are allocated in split_rebuild(), not
  //   here: allocating one now would free the array the cell's current
  //   SplitInfo still points at, which split_cell_unset() has yet to read

  if (n > p->maxmap) {
    p->maxmap = n;
    memory->destroy(p->map);
    memory->create(p->map,p->maxmap,"fix_rigid:pendingmap");
  }
  memcpy(p->map,map,n*sizeof(int));

  p->xsub = xsub;
  p->xsplit[0] = xsplit[0];
  p->xsplit[1] = xsplit[1];
  p->xsplit[2] = xsplit[2];

  if (nsplitnew > p->maxvols) {
    p->maxvols = nsplitnew;
    memory->destroy(p->vols);
    memory->create(p->vols,p->maxvols,"fix_rigid:pendingvols");
  }
  memcpy(p->vols,vols,nsplitnew*sizeof(double));
}

/* ----------------------------------------------------------------------
   apply every pending split change, in place of a full grid re-map
   the incremental passes already produced the correct surf list, flow
     volume, cell type and corner marks of every cell, so the two most
     expensive steps of grid_rebuild() are skipped: Grid::clear_surf()
     plus Grid::surf2grid(), which re-maps every surf to every cell, and
     Grid::set_inout(), whose flood fill is an iterative collective
   what is left is the structural part, the same sequence fix adapt uses
     when it refines or coarsens cells during a run: no owned cell may
     be added or removed while ghost cells are stored, so they are
     dropped and re-acquired around the change
   the sub cells of every cell which stops being split are detached
     first and removed in one sweep afterwards, so that the cell indices
     the pending list holds stay valid until every change is applied
------------------------------------------------------------------------- */

void FixRigid::split_rebuild()
{
  int m;

  // Grid::remove_marked_cells() walks the per-cell particle lists
  // the caller has already pulled the particles of every split cell up
  //   into the cell itself, so none is labelled with a sub cell which
  //   is about to vanish

  if (particle->exist) {
    particles_to_host();
    particle->sort();
  }

  // neighbor links become cell IDs, which survive the cells moving below

  grid->unset_neighbors();
  grid->remove_ghosts();

  // detach the old sub cells of every pending cell, then build the new
  //   ones; detaching only marks, so no index moves in between

  for (m = 0; m < npending; m++)
    grid->split_cell_unset(pending[m].icell);

  // nothing points at the old piece map and sub cell list of a pending
  //   cell any more, so they can be replaced now
  // a cell which stops being split gives both of them up

  for (m = 0; m < npending; m++) {
    cellint id = grid->cells[pending[m].icell].id;
    if (pending[m].nsplitnew == 1) {
      split_registry_remove(id);
      continue;
    }
    pending[m].csplits = csplits_alloc(id,pending[m].nsurf);
    memcpy(pending[m].csplits,pending[m].map,pending[m].nsurf*sizeof(int));
    pending[m].csubs = csubs_alloc(id,pending[m].nsplitnew);
  }

  for (m = 0; m < npending; m++) {
    if (pending[m].nsplitnew == 1) continue;
    grid->split_cell_set(pending[m].icell,pending[m].nsplitnew,
                         pending[m].csplits,pending[m].csubs,
                         pending[m].xsub,pending[m].xsplit,pending[m].vols);
  }

  grid->remove_marked_cells();

  // re-establish the owned cell bookkeeping, the ghost cells and the
  //   neighbor links, exactly as fix adapt does after changing cells

  grid->setup_owned();
  grid->acquire_ghosts();
  grid->reset_neighbors();
  comm->reset_neighbors();

  // the box->cell index holds cell indices, which just moved
  // a step which added and removed the same number of cells leaves the
  //   total unchanged, so it cannot detect this for itself

  update->rigid_bins_clear();

  // as after a grid rebuild with distributed surfs

  if (surf->distributed) {
    surf->localghost_changed_step = update->ntimestep;
    for (int i = 0; i < surf->ncustom; i++) surf->estatus[i] = 0;
  }

  // a per-grid compute sized itself for the old cell count; one which
  //   sees the same count again would keep arrays that now refer to
  //   different cells, so they are dropped, as AdaptGrid does
  // insplitrebuild keeps grid_changed() from taking this fix's surf
  //   lists away: unlike a full re-map, they are still installed in the
  //   cells and still current

  Compute **compute = modify->compute;
  for (int i = 0; i < modify->ncompute; i++)
    if (compute[i]->per_grid_flag) {
      compute[i]->reallocate();
      compute[i]->invoked_flag = 0;
    }

  insplitrebuild = 1;
  grid->notify_changed();
  insplitrebuild = 0;
}

/* ----------------------------------------------------------------------
   install a re-cut split of one cell which keeps its piece count
   the cell's surf list and its csplits map must stay the same length,
     since Update::split2d/3d() index them in lockstep, so the map is
     replaced whenever the surf list is
   the sub cells share the split cell's surf list, and an owned cell's
     sub cells take the new per-piece flow volumes; the split cell's own
     volume is the whole cell volume and is left alone, as
     Grid::surf2grid_split() leaves it
   a ghost copy has no ChildInfo, so it takes the piece map only: it
     exists here so this proc's mover can route a particle crossing into
     the cell to the right sub cell, which Comm::migrate_particles()
     then translates to the owner's index via ChildCell::ilocal
------------------------------------------------------------------------- */

void FixRigid::split_update(int icell, int n, int *map, int xsub,
                            double *xsplit, double *vols, int ghostflag)
{
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  Grid::SplitInfo *sinfo = grid->sinfo;

  int isplit = cells[icell].isplit;
  int nsplit = cells[icell].nsplit;

  int *csplits = csplits_alloc(cells[icell].id,n);
  memcpy(csplits,map,n*sizeof(int));

  sinfo[isplit].csplits = csplits;
  sinfo[isplit].xsub = xsub;
  sinfo[isplit].xsplit[0] = xsplit[0];
  sinfo[isplit].xsplit[1] = xsplit[1];
  if (dim == 3) sinfo[isplit].xsplit[2] = xsplit[2];
  else sinfo[isplit].xsplit[2] = 0.0;

  int *csubs = sinfo[isplit].csubs;

  for (int i = 0; i < nsplit; i++) {
    int isub = csubs[i];
    cells[isub].nsurf = cells[icell].nsurf;
    cells[isub].csurfs = cells[icell].csurfs;
    if (!ghostflag) cinfo[isub].volume = vols[i];
  }
}

/* ----------------------------------------------------------------------
   for incremental remap: re-cut only grid cells near the body
   a cell is re-cut if the set of surfs overlapping it changed,
     or if it is overlapped by a body surf (whose geometry moved)
   candidate surfs for a cell = the static surfs already in its list
     (static surfs never move, so the set overlapping a cell is fixed)
     plus every element of every body, so the cost per cell is
     O(surfs in cell + body surfs) and independent of the total surf
     count; only local surf indices are ever referenced, as required
     for distributed surfs
   cells interior to the body at its old or new position are re-typed
     as INSIDE/OUTSIDE via parity tests, all other cells are untouched;
     a cell cut by no surf (surf-free, or overlapped only by transparent
     surfs) is typed this way, as Grid::set_inout() types it
   ghost cell copies of re-cut cells become stale, which is acceptable:
     the ghost cell surf lists the mover consults are re-covered by the
     swept assignment every step, and cell volumes/types of ghost cells
     are not used
   return 0 if done, else a FALLBACK reason code requesting a full grid
     re-map for a structural change:
     a cell would become or stop being a split cell, a cell's surf
     count exceeds maxsurfpercell, or no previous body position is set
------------------------------------------------------------------------- */

int FixRigid::incremental_recut()
{
  int i,n,ncand,icell,ibody,nsplitone,xsub,moving;
  double xsplit[3],ctr[3],rlo[3],rhi[3];
  double *vols;
  double *clo,*chi;

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  int nglocal = grid->nlocal;
  int maxsurfpercell = grid->maxsurfpercell;
  int *rigidmap = update->rigidmap;

  int ncorner = 4;
  if (dim == 3) ncorner = 8;
  int cornerscratch[8];

  // every body must have a previous region
  // R = union over all bodies of the region rlo/rhi each
  //   occupied before and after its move this step
  // collect the owned cells overlapping R from the box->cell index,
  //   one body region at a time, so bodies far apart do not sweep the
  //   cells between them; the re-cut and re-type passes below iterate
  //   only this list

  if (!pbodyflag) return FALLBACK_NOPREV;
  nrcand = 0;
  splitchanged = 0;
  npending = 0;

  for (ibody = 0; ibody < nbody; ibody++) {
    for (i = 0; i < 3; i++) {
      rlo[i] = MIN(pbodylo[ibody][i],bbodylo[ibody][i]);
      rhi[i] = MAX(pbodyhi[ibody][i],bbodyhi[ibody][i]);
    }

    int *cand;
    int ncells = update->rigid_cell_box(rlo,rhi,&cand);

    // owned cells are re-cut; a ghost cell is included only when it is
    //   a split cell, whose sinfo this proc must keep current because
    //   Comm::migrate_particles() resolves a crossing particle's
    //   destination sub cell on the SENDING proc, from the ghost copy
    // the body geometry is replicated, so every proc re-derives the
    //   same split of the same cell without communicating

    for (int ic = 0; ic < ncells; ic++) {
      icell = cand[ic];
      if (cells[icell].nsplit <= 0) continue;
      if (icell >= nglocal && cells[icell].nsplit == 1) continue;
      if (!box_overlap(cells[icell].lo,cells[icell].hi,rlo,rhi)) continue;
      if (nrcand == maxrcand) {
        maxrcand += DELTA_MODIFY;
        memory->grow(rcand,maxrcand,"fix_rigid:rcand");
      }
      rcand[nrcand++] = icell;
    }
  }

  // a cell in the regions of several bodies is listed once

  if (nbody > 1) {
    std::sort(rcand,rcand+nrcand);
    nrcand = std::unique(rcand,rcand+nrcand) - rcand;
  }

  // pass 1: re-cut cells in R whose surf overlap changed
  //   or which are overlapped by a moved body surf (from any body)
  // candidate list keeps the cell's static surfs in their current
  //   order, followed by the body elements, so an unchanged cell
  //   yields an identical list and is skipped

  for (int ic = 0; ic < nrcand; ic++) {
    icell = rcand[ic];

    // a ghost cell in the list is a split cell whose sinfo is refreshed
    //   below; it has no ChildInfo, so the cut writes its corner marks
    //   to scratch and its flow volumes are the owner's business

    int ghostflag = (icell >= nglocal);
    int nsplitold = cells[icell].nsplit;
    int *corner = ghostflag ? cornerscratch : cinfo[icell].corner;

    ncand = 0;
    surfint *cur = cells[icell].csurfs;
    for (i = 0; i < cells[icell].nsurf; i++)
      if (rigidmap[cur[i]] < 0) reclist[ncand++] = cur[i];

    // only bodies whose bounding box overlaps this cell contribute
    //   candidates: bbodylo/bbodyhi bound every element of the body at
    //   its end-of-step position, so surf2grid_list would reject all of
    //   them one at a time.  the body bins find them without a loop
    //   over all bodies, so the cost of a cell is independent of how
    //   many bodies are defined far away from it

    int *blist;
    int nb = body_box(cells[icell].lo,cells[icell].hi,&blist);
    for (int m = 0; m < nb; m++) {
      ibody = blist[m];

      // radial prefilter: body_box() only compared this cell against the
      //   body's bounding BOX, so it still returns a body whose surfs all
      //   lie far from the cell -- the deep interior of the body, and the
      //   corners of its bbox, which for a rounded body is 1-pi/4 of it.
      //   every such surf would then be tested against the cell one at a
      //   time by surf2grid_list() only to be rejected
      // an element of this body lies at a distance in [rminbody,rmaxbody]
      //   of the COM, so if the cell's own distance range from the COM does
      //   not meet that interval, no element of this body can touch the
      //   cell and none need be offered
      // dlo/dhi are the min/max distance from the COM to the cell, computed
      //   per axis and exact for an axis-aligned box

      double dlo2 = 0.0, dhi2 = 0.0;
      double *clo2 = cells[icell].lo;
      double *chi2 = cells[icell].hi;
      for (int k = 0; k < dim; k++) {
        double c = xcm[ibody][k];
        double dlok = 0.0;
        if (c < clo2[k]) dlok = clo2[k] - c;
        else if (c > chi2[k]) dlok = c - chi2[k];
        double dhik = MAX(fabs(c-clo2[k]),fabs(c-chi2[k]));
        dlo2 += dlok*dlok;
        dhi2 += dhik*dhik;
      }

      // rmaxbody/rminbody are body-frame radii about the COM and so are
      //   invariant under the body's rotation; compare squared distances
      // rminbody is a lower bound on the true element distance, so the
      //   inner test only rejects cells that certainly hold no surf

      double rmax = rmaxbody[ibody] + bboxeps[ibody];
      if (dlo2 > rmax*rmax) continue;
      double rmin = rminbody[ibody] - bboxeps[ibody];
      if (rmin > 0.0 && dhi2 < rmin*rmin) continue;

      for (i = bodystart[ibody]; i < bodystart[ibody+1]; i++)
        reclist[ncand++] = lblist[i];
    }

    // new list of surfs overlapping this cell

    if (dim == 2)
      n = cut2d->surf2grid_list(cells[icell].id,
                                cells[icell].lo,cells[icell].hi,
                                ncand,reclist,newlist,maxsurfpercell);
    else
      n = cut3d->surf2grid_list(cells[icell].id,
                                cells[icell].lo,cells[icell].hi,
                                ncand,reclist,newlist,maxsurfpercell);
    if (n > maxsurfpercell) return FALLBACK_SURFMAX;

    // order the list by local surf index, as Grid::surf2grid() does
    //   before cutting, so the cut sees surfs in the same order in both
    //   remap modes and lists of unchanged cells compare equal

    std::sort(newlist,newlist+n);

    // skip cell if surf list is unchanged and contains no moving surf
    // a moving surf belongs to any rigid body (via rigidmap)

    moving = 0;
    for (i = 0; i < n; i++)
      if (rigidmap[newlist[i]] >= 0) {
        moving = 1;
        break;
      }

    if (!moving && n == cells[icell].nsurf) {
      if (n == 0) continue;
      if (memcmp(newlist,cells[icell].csurfs,n*sizeof(surfint)) == 0)
        continue;
    }

    clo = cells[icell].lo;
    chi = cells[icell].hi;
    ctr[0] = 0.5 * (clo[0] + chi[0]);
    ctr[1] = 0.5 * (clo[1] + chi[1]);
    if (dim == 3) ctr[2] = 0.5 * (clo[2] + chi[2]);
    else ctr[2] = 0.0;

    if (n == 0) {

      // a split cell which no longer overlaps any surf gives up its
      //   sub cells, which changes the cell count

      if (nsplitold > 1) {
        if (ghostflag) { split_ghost_drop(icell); continue; }
        split_pending(icell,1,0,NULL,0,NULL,NULL);
      }

      // cell no longer overlaps any surf
      // full flow volume, interior/exterior typing via parity test

      registry_remove(cells[icell].id);
      cells[icell].nsurf = 0;
      cells[icell].csurfs = NULL;
      listschanged = 1;

      cinfo[icell].volume = full_cell_volume(clo,chi);

      if (inside_any_body(ctr)) cinfo[icell].type = CELLINSIDE;
      else cinfo[icell].type = CELLOUTSIDE;
      for (i = 0; i < ncorner; i++)
        cinfo[icell].corner[i] = cinfo[icell].type;
      if (cinfo[icell].type == CELLINSIDE) cinfo[icell].volume = 0.0;
      typechanged = 1;

    } else {

      // install new surf list

      surfint *list =
        (surfint *) memory->smalloc(n*sizeof(surfint),"fix_rigid:recut");
      memcpy(list,newlist,n*sizeof(surfint));
      registry_replace(cells[icell].id,list);
      cells[icell].nsurf = n;
      cells[icell].csurfs = list;
      listschanged = 1;

      // a cell whose surfs are all transparent is not cut by the full
      //   pipeline (Grid::surf2grid_split() skips non-OVERLAP cells):
      //   full flow volume, interior/exterior typing via parity test

      if (!cell_cut(icell)) {
        if (nsplitold > 1) {
          if (ghostflag) { split_ghost_drop(icell); continue; }
          split_pending(icell,1,0,NULL,0,NULL,NULL);
        }
        cinfo[icell].volume = full_cell_volume(clo,chi);
        if (inside_any_body(ctr)) cinfo[icell].type = CELLINSIDE;
        else cinfo[icell].type = CELLOUTSIDE;
        for (i = 0; i < ncorner; i++)
          cinfo[icell].corner[i] = cinfo[icell].type;
        if (cinfo[icell].type == CELLINSIDE) cinfo[icell].volume = 0.0;
        typechanged = 1;
        continue;
      }

      // re-cut the cell

      if (dim == 2)
        nsplitone = cut2d->split(cells[icell].id,
                                 cells[icell].lo,cells[icell].hi,
                                 n,list,vols,newmap,
                                 corner,xsub,xsplit);
      else
        nsplitone = cut3d->split(cells[icell].id,
                                 cells[icell].lo,cells[icell].hi,
                                 n,list,vols,newmap,
                                 corner,xsub,xsplit);

      // the cut leaves the corner marks UNKNOWN when every surf only
      //   touches the cell faces; the full pipeline resolves such cells
      //   by flood fill in Grid::set_inout(), so fall back to it

      if (corner[0] == CELLUNKNOWN) return FALLBACK_UNKNOWN;

      // the number of disconnected flow pieces changed: the cell gains
      //   or loses sub cells, which changes this proc's cell count and
      //   the sub cell indices other procs migrate particles into
      // recorded now and applied by split_rebuild() at the end of the
      //   step, together with every other such cell

      if (nsplitone != nsplitold) {
        if (ghostflag) { split_ghost_drop(icell); continue; }
        split_pending(icell,nsplitone,n,newmap,xsub,xsplit,vols);

        // a split cell's own volume is the whole cell volume

        if (nsplitone > 1) cinfo[icell].volume = full_cell_volume(clo,chi);
        else cinfo[icell].volume = vols[0];
        cinfo[icell].type = CELLOVERLAP;
        typechanged = 1;
        continue;
      }

      // a cell which was and still is split keeps its sub cells and
      //   takes the new piece map and per-piece volumes in place

      if (nsplitone > 1) {
        split_update(icell,n,newmap,xsub,xsplit,vols,ghostflag);
        splitchanged = 1;
        if (ghostflag) continue;

        // a split cell's own volume is the whole cell volume
        //   (Grid::surf2grid_split() likewise leaves it alone)

        cinfo[icell].type = CELLOVERLAP;
        typechanged = 1;
        continue;
      }

      cinfo[icell].volume = vols[0];
      cinfo[icell].type = CELLOVERLAP;
      typechanged = 1;
    }
  }

  // pass 2: cells a body interior moved away from become OUTSIDE
  // only cells which are now cut by no surf and inside no body,
  //   which leaves any static (non-body) INSIDE cells untouched

  for (int m = 0; m < noldinside; m++) {
    icell = oldinside[m];
    if (cell_cut(icell)) continue;

    clo = cells[icell].lo;
    chi = cells[icell].hi;
    ctr[0] = 0.5 * (clo[0] + chi[0]);
    ctr[1] = 0.5 * (clo[1] + chi[1]);
    if (dim == 3) ctr[2] = 0.5 * (clo[2] + chi[2]);
    else ctr[2] = 0.0;
    if (inside_any_body(ctr)) continue;

    cinfo[icell].type = CELLOUTSIDE;
    typechanged = 1;
    cinfo[icell].volume = full_cell_volume(clo,chi);
    for (i = 0; i < ncorner; i++)
      cinfo[icell].corner[i] = CELLOUTSIDE;
  }

  // pass 3: uncut cells a body interior moved over become INSIDE
  // catches cells swept over entirely within one step, which never
  //   overlap a body surf at start- or end-of-step positions
  // R covers the swept corridor since it unions old and new positions
  // guard on type != INSIDE leaves static INSIDE cells untouched

  for (int ic = 0; ic < nrcand; ic++) {
    icell = rcand[ic];
    if (cells[icell].nsplit != 1) continue;
    if (cell_cut(icell)) continue;
    if (cinfo[icell].type == CELLINSIDE) continue;

    clo = cells[icell].lo;
    chi = cells[icell].hi;
    ctr[0] = 0.5 * (clo[0] + chi[0]);
    ctr[1] = 0.5 * (clo[1] + chi[1]);
    if (dim == 3) ctr[2] = 0.5 * (clo[2] + chi[2]);
    else ctr[2] = 0.0;
    if (!inside_any_body(ctr)) continue;

    cinfo[icell].type = CELLINSIDE;
    for (i = 0; i < ncorner; i++)
      cinfo[icell].corner[i] = CELLINSIDE;
    cinfo[icell].volume = 0.0;
    typechanged = 1;
  }

  return FALLBACK_NONE;
}

/* ----------------------------------------------------------------------
   project a force and torque on an axisymmetric body onto the two
     degrees of freedom it has: translation along x and spin about x
   no-op unless the domain is axisymmetric
   this is not a constraint imposed on the physics, it is the azimuthal
     average of it.  a simulated particle in an axisymmetric run stands
     for real molecules at every azimuth around the ring at its radius,
     so the force it exerts must be averaged around that ring before the
     body responds to it.  a radial force at azimuth phi points along
     (cos phi, sin phi) in the space frame and an azimuthal one along
     (-sin phi, cos phi); both average to zero.  what survives is the
     axial force and the torque about the axis, which compute surf
     already tallies correctly as tx = r * f_theta, since the hit point
     is (x,r,0) and the COM is on the axis
   the body's own response is averaged the same way in the collision
     recoil: see the axisymmetric kmat in Geometry::rigid_recoil()
------------------------------------------------------------------------- */

void FixRigid::axi_project(double *f, double *tq)
{
  if (!axiflag) return;
  f[1] = f[2] = 0.0;
  tq[1] = tq[2] = 0.0;
}

/* ----------------------------------------------------------------------
   flow volume of a grid cell which no surf cuts
   matches Grid::set_inout() and Grid::add_child_cell(): an axisymmetric
     cell is an annulus swept about the axis, not a rectangle
------------------------------------------------------------------------- */

double FixRigid::full_cell_volume(double *clo, double *chi)
{
  if (dim == 3)
    return (chi[0]-clo[0]) * (chi[1]-clo[1]) * (chi[2]-clo[2]);
  if (axiflag)
    return MY_PI * (chi[1]*chi[1] - clo[1]*clo[1]) * (chi[0]-clo[0]);
  return (chi[0]-clo[0]) * (chi[1]-clo[1]);
}

/* ----------------------------------------------------------------------
   return 1 if grid cell icell is cut by a surf, else 0
   a cell overlapped only by transparent surfs is not cut:
     Grid::surf2grid_split() leaves it uncut, and Grid::set_inout()
     types it INSIDE/OUTSIDE by flood fill like a surf-free cell,
     so the incremental re-cut must re-type it the same way
------------------------------------------------------------------------- */

int FixRigid::cell_cut(int icell)
{
  Grid::ChildCell *cells = grid->cells;
  int n = cells[icell].nsurf;
  surfint *list = cells[icell].csurfs;

  if (dim == 2) {
    Surf::Line *lines = surf->lines;
    for (int i = 0; i < n; i++)
      if (!lines[list[i]].transparent) return 1;
  } else {
    Surf::Tri *tris = surf->tris;
    for (int i = 0; i < n; i++)
      if (!tris[list[i]].transparent) return 1;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   return 1 if point x is inside any rigid body, else 0
   only bodies whose bbox contains the point are tested, found from the
     body bins: inside_body() casts a ray against every element of a
     body, and a point outside a body's bbox cannot be inside the body,
     so the box test is exact, not a heuristic
   requires body_bbox() and body_bins() were called for every body,
     which end_of_step() does when the bodies commit their new geometry
------------------------------------------------------------------------- */

int FixRigid::inside_any_body(double *x)
{
  int *blist;
  int nb = body_box(x,x,&blist);
  for (int m = 0; m < nb; m++)
    if (inside_body(blist[m],x)) return 1;
  return 0;
}

/* ----------------------------------------------------------------------
   registry of cells whose csurfs lists are allocated by this fix
   grid-owned csurfs lists live in page memory and are never freed
     individually, so lists installed by incremental re-cutting are
     tracked here and freed when replaced or when the grid changes
------------------------------------------------------------------------- */

void FixRigid::registry_replace(cellint id, surfint *list)
{
  std::map<cellint,surfint *>::iterator it = registry.find(id);
  if (it != registry.end()) {
    memory->sfree(it->second);
    it->second = list;
  } else registry[id] = list;
}

void FixRigid::registry_remove(cellint id)
{
  std::map<cellint,surfint *>::iterator it = registry.find(id);
  if (it == registry.end()) return;
  memory->sfree(it->second);
  registry.erase(it);
}

void FixRigid::free_registry()
{
  for (std::map<cellint,surfint *>::iterator it = registry.begin();
       it != registry.end(); ++it)
    memory->sfree(it->second);
  registry.clear();
}

/* ----------------------------------------------------------------------
   copy every registry list still installed in a live grid cell into
     grid-owned page storage, preserving pointer sharing between a
     split cell and its sub cells, so no cell is left pointing at
     memory this fix is about to free
   used by the destructor: a fix can be unfixed between runs after
     incremental re-cuts installed lists in cells
------------------------------------------------------------------------- */

void FixRigid::copy_registry_to_grid()
{
  if (registry.empty()) return;

  std::map<surfint *,surfint *> replaced;
  Grid::ChildCell *cells = grid->cells;
  int ntotal = grid->nlocal + grid->nghost;

  for (std::map<cellint,surfint *>::iterator it = registry.begin();
       it != registry.end(); ++it)
    replaced[it->second] = NULL;

  for (int icell = 0; icell < ntotal; icell++) {
    if (cells[icell].nsurf <= 0) continue;
    std::map<surfint *,surfint *>::iterator it =
      replaced.find(cells[icell].csurfs);
    if (it == replaced.end()) continue;
    if (it->second == NULL) {
      surfint *copy = grid->csurfs->get(cells[icell].nsurf);
      if (!copy)
        error->one(FLERR,"Failed to allocate grid surf list for fix rigid");
      memcpy(copy,cells[icell].csurfs,cells[icell].nsurf*sizeof(surfint));
      it->second = copy;
    }
    cells[icell].csurfs = it->second;
  }
}

/* ----------------------------------------------------------------------
   registry of split-cell arrays allocated by this fix
   sinfo[].csplits and sinfo[].csubs live in the Grid page allocators,
     which never free an individual list, and csplits has one entry per
     surf in the split cell, so a cell re-cut every step as a body moves
     through it would grow the page without bound
   the arrays this fix installs are tracked here instead, keyed by cell
     index, and freed when replaced or when the grid changes
   same problem and same solution as the csurfs registry above
------------------------------------------------------------------------- */

int *FixRigid::csplits_alloc(cellint id, int n)
{
  int *list = (int *) memory->smalloc(n*sizeof(int),"fix_rigid:csplits");
  std::map<cellint,int *>::iterator it = csplitreg.find(id);
  if (it != csplitreg.end()) {
    memory->sfree(it->second);
    it->second = list;
  } else csplitreg[id] = list;
  return list;
}

int *FixRigid::csubs_alloc(cellint id, int n)
{
  int *list = (int *) memory->smalloc(n*sizeof(int),"fix_rigid:csubs");
  std::map<cellint,int *>::iterator it = csubreg.find(id);
  if (it != csubreg.end()) {
    memory->sfree(it->second);
    it->second = list;
  } else csubreg[id] = list;
  return list;
}

void FixRigid::split_registry_remove(cellint id)
{
  std::map<cellint,int *>::iterator it = csplitreg.find(id);
  if (it != csplitreg.end()) {
    memory->sfree(it->second);
    csplitreg.erase(it);
  }
  it = csubreg.find(id);
  if (it != csubreg.end()) {
    memory->sfree(it->second);
    csubreg.erase(it);
  }
}

void FixRigid::free_split_registry()
{
  for (std::map<cellint,int *>::iterator it = csplitreg.begin();
       it != csplitreg.end(); ++it)
    memory->sfree(it->second);
  csplitreg.clear();

  for (std::map<cellint,int *>::iterator it = csubreg.begin();
       it != csubreg.end(); ++it)
    memory->sfree(it->second);
  csubreg.clear();
}

/* ----------------------------------------------------------------------
   copy every registry split array still installed in a live split cell
     into grid-owned page storage, so no sinfo entry is left pointing at
     memory this fix is about to free
   the cell index a list was registered under may no longer refer to the
     same cell, so entries are matched by pointer identity, as
     copy_registry_to_grid() does for the csurfs lists
   one csplits and one csubs array belong to exactly one sinfo entry, so
     there is no pointer sharing to preserve here
------------------------------------------------------------------------- */

void FixRigid::copy_split_registry_to_grid()
{
  if (csplitreg.empty() && csubreg.empty()) return;

  std::map<int *,int> mine;

  for (std::map<cellint,int *>::iterator it = csplitreg.begin();
       it != csplitreg.end(); ++it)
    mine[it->second] = 1;
  for (std::map<cellint,int *>::iterator it = csubreg.begin();
       it != csubreg.end(); ++it)
    mine[it->second] = 1;

  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;
  int nsplitall = grid->nsplitlocal + grid->nsplitghost;

  for (int i = 0; i < nsplitall; i++) {
    int icell = sinfo[i].icell;
    if (cells[icell].nsplit <= 1) continue;
    int nsurf = cells[icell].nsurf;
    int nsplit = cells[icell].nsplit;

    if (nsurf > 0 && mine.find(sinfo[i].csplits) != mine.end()) {
      int *copy = grid->csplits->get(nsurf);
      if (!copy)
        error->one(FLERR,"Failed to allocate grid split list for fix rigid");
      memcpy(copy,sinfo[i].csplits,nsurf*sizeof(int));
      sinfo[i].csplits = copy;
    }

    if (mine.find(sinfo[i].csubs) != mine.end()) {
      int *copy = grid->csubs->get(nsplit);
      if (!copy)
        error->one(FLERR,"Failed to allocate grid sub list for fix rigid");
      memcpy(copy,sinfo[i].csubs,nsplit*sizeof(int));
      sinfo[i].csubs = copy;
    }
  }
}

/* ----------------------------------------------------------------------
   compute per-element bounding boxes (elemlo/elemhi) and the whole-body
     bounding box (bbodylo/bbodyhi) of one body
   sweepflag = 0: boxes bound elements at their current positions
   sweepflag = 1: boxes also bound elements at their end-of-step positions
     (from xcmnew and exyz_space set from quatnew in start_of_step),
     giving the region each element sweeps through during the step
   boxes are inflated by EPSSURF * body extent to avoid round-off misses
------------------------------------------------------------------------- */

void FixRigid::body_bbox(int ibody, int sweepflag)
{
  int i,j,k;
  double **pts;
  double delta[3],ptnew[3];
  double *lo,*hi;

  int npoint = dim;     // 2 points per line, 3 per tri
  int istart = bodystart[ibody];
  int istop = bodystart[ibody+1];
  double *blo = bbodylo[ibody];
  double *bhi = bbodyhi[ibody];

  blo[0] = blo[1] = blo[2] = BIG;
  bhi[0] = bhi[1] = bhi[2] = -BIG;

  for (i = istart; i < istop; i++) {
    pts = bodypt[i];

    lo = elemlo[i];
    hi = elemhi[i];
    lo[0] = lo[1] = lo[2] = BIG;
    hi[0] = hi[1] = hi[2] = -BIG;

    for (j = 0; j < npoint; j++)
      for (k = 0; k < 3; k++) {
        lo[k] = MIN(lo[k],pts[j][k]);
        hi[k] = MAX(hi[k],pts[j][k]);
      }

    if (sweepflag) {
      for (j = 0; j < npoint; j++) {
        if (axiflag) {
          ptnew[0] = xcmnew[ibody][0] + displace[i][j][0];
          ptnew[1] = displace[i][j][1];
          ptnew[2] = 0.0;
        } else {
          MathExtra::matvec(ex_space[ibody],ey_space[ibody],ez_space[ibody],
                            displace[i][j],delta);
          if (dim == 2) delta[2] = 0.0;
          MathExtra::add3(xcmnew[ibody],delta,ptnew);
        }
        for (k = 0; k < 3; k++) {
          lo[k] = MIN(lo[k],ptnew[k]);
          hi[k] = MAX(hi[k],ptnew[k]);
        }
      }
    }

    for (k = 0; k < 3; k++) {
      blo[k] = MIN(blo[k],lo[k]);
      bhi[k] = MAX(bhi[k],hi[k]);
    }
  }

  double eps = EPSSURF * MAX(bhi[0]-blo[0],bhi[1]-blo[1]);
  eps = EPSSURF * MAX(eps/EPSSURF,bhi[2]-blo[2]);
  bboxeps[ibody] = eps;

  // swept boxes bound the chord of each point's motion; the arc of a
  //   rotating point bulges beyond the chord by up to R*(1-cos(a/2))
  //   for rotation angle a and distance R from the axis, so pad by
  //   that (bounded by R*a^2/8) to cover the true swept region
  // an axisymmetric body's spin is about its own axis of revolution, so
  //   it displaces no surf point and the motion is the chord exactly

  if (sweepflag && !axiflag) {
    double angle = MathExtra::len3(omega[ibody]) * update->dt;
    eps += 0.125 * rmaxbody[ibody] * angle * angle;
  }

  for (i = istart; i < istop; i++)
    for (k = 0; k < 3; k++) {
      elemlo[i][k] -= eps;
      elemhi[i][k] += eps;
    }
  for (k = 0; k < 3; k++) {
    blo[k] -= eps;
    bhi[k] += eps;
  }
}

/* ----------------------------------------------------------------------
   determine if point X is inside closed body ibody via a parity test
   count intersections of segment from X to a point outside the body
     with all body elements: odd = inside, even = outside
   segment direction is oblique to coordinate axes to reduce the chance
     of exactly grazing element edges or vertices
   requires body_bbox() was called to set bbodylo/bbodyhi
------------------------------------------------------------------------- */

int FixRigid::inside_body(int ibody, double *x)
{
  int hitflag,side;
  double param;
  double xout[3],xc[3];

  double *blo = bbodylo[ibody];
  double *bhi = bbodyhi[ibody];

  double dmax = MAX(bhi[0]-blo[0],bhi[1]-blo[1]);
  dmax = MAX(dmax,bhi[2]-blo[2]);

  xout[0] = bhi[0] + 0.414159*dmax;
  xout[1] = x[1] + 0.271828*dmax;
  if (dim == 3) xout[2] = x[2] + 0.161803*dmax;
  else xout[2] = 0.0;

  int count = 0;
  for (int i = bodystart[ibody]; i < bodystart[ibody+1]; i++) {
    if (dim == 2)
      hitflag = Geometry::
        line_line_intersect(x,xout,bodypt[i][0],bodypt[i][1],
                            bodynorm[i],xc,param,side);
    else
      hitflag = Geometry::
        line_tri_intersect(x,xout,bodypt[i][0],bodypt[i][1],
                           bodypt[i][2],bodynorm[i],xc,param,side);
    if (hitflag) count++;
  }

  return count % 2;
}

/* ----------------------------------------------------------------------
   remove particles which are inside any body
   also remove all particles in INSIDE cells
   used at setup; each step uses remove_inside_all() instead
   splitflag = 1 if called after a grid rebuild,
     to first reassign particles in split cells to their sub cells
   requires body_bbox() was called to set every body's bbox
   return # of particles deleted by this proc; counts are summed
     across procs lazily by compute_scalar()
------------------------------------------------------------------------- */

bigint FixRigid::remove_inside_particles(int splitflag)
{
  // reassign particles in split cells to sub cell owner
  // requires sorted particles, done by grid_rebuild()

  // the particles are relabeled to their sub cells but not re-listed
  //   under them, so they are no longer sorted; a fix balance later in
  //   this step would otherwise migrate cells with stale particle lists

  if (splitflag && grid->nsplitlocal) {
    Grid::ChildCell *cells = grid->cells;
    int nglocal = grid->nlocal;
    for (int icell = 0; icell < nglocal; icell++)
      if (cells[icell].nsplit > 1)
        grid->assign_split_cell_particles(icell);
    particle->sorted = 0;
  }

  // flag particles inside a body or in INSIDE cells for deletion

  Grid::ChildInfo *cinfo = grid->cinfo;
  Particle::OnePart *particles = particle->particles;
  int nplocal = particle->nlocal;

  int icell;
  int delflag = 0;

  for (int i = 0; i < nplocal; i++) {
    icell = particles[i].icell;
    if (icell < 0) continue;

    if (cinfo[icell].type == CELLINSIDE) {
      particles[i].icell = -1;
      delflag = 1;
      continue;
    }

    if (inside_any_body(particles[i].x)) {
      particles[i].icell = -1;
      delflag = 1;
    }
  }

  // compress out deleted particles

  int nlocal_old = particle->nlocal;
  if (delflag) particle->compress_rebalance();
  return nlocal_old - particle->nlocal;
}

/* ----------------------------------------------------------------------
   remove particles inside any rigid body, in one pass over particles
   called each step after the grid re-map; every body's bbodylo/bbodyhi
     is current (set when each body committed its end-of-step geometry)
   also removes all particles in INSIDE cells
   splitflag = 1 if called after a full grid rebuild,
     to first reassign particles in split cells to their sub cells
   deletions increment the per-proc ndeleted count with no
     communication; compute_scalar() reduces the counts on demand
------------------------------------------------------------------------- */

void FixRigid::remove_inside_all(int splitflag)
{
  double *x;

  // reassign particles in split cells to sub cell owner
  // requires sorted particles, done by grid_rebuild()

  // the particles are relabeled to their sub cells but not re-listed
  //   under them, so they are no longer sorted; a fix balance later in
  //   this step would otherwise migrate cells with stale particle lists

  if (splitflag && grid->nsplitlocal) {
    Grid::ChildCell *cells = grid->cells;
    int nglocal = grid->nlocal;
    for (int icell = 0; icell < nglocal; icell++)
      if (cells[icell].nsplit > 1)
        grid->assign_split_cell_particles(icell);
    particle->sorted = 0;
  }

  // flag particles inside any body or in INSIDE cells for deletion
  // a particle in an INSIDE cell claimed by no body (e.g. inside
  //   static closed geometry) is deleted but not counted

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  Particle::OnePart *particles = particle->particles;
  int nplocal = particle->nlocal;

  int icell;
  int delflag = 0;

  for (int i = 0; i < nplocal; i++) {
    icell = particles[i].icell;
    if (icell < 0) continue;

    x = particles[i].x;
    int inside = (cinfo[icell].type == CELLINSIDE);

    // a surf-free OUTSIDE cell cannot contain a point interior to a body,
    //   since a body boundary crossing the cell would put a surf in it

    if (!inside && cinfo[icell].type == CELLOUTSIDE &&
        cells[icell].nsurf == 0) continue;

    int inbody = inside_any_body(x);
    if (!inbody && !inside) continue;

    particles[i].icell = -1;
    delflag = 1;
    if (inbody) {
      ndeleted++;
      ndelrun++;
    }
  }

  // compress out deleted particles, once for all bodies

  if (delflag) particle->compress_rebalance();

  end_of_run_delete_warning();
}

/* ----------------------------------------------------------------------
   warn once per run if any particle was deleted inside a body
   split out of remove_inside_all() so the KOKKOS override, which runs the
     deletion pass as a device kernel, reports it the same way
------------------------------------------------------------------------- */

void FixRigid::end_of_run_delete_warning()
{
  // warn once per run if any particle was deleted after the setup pass
  // with swept collision coverage a particle in the body's path is
  //   reflected, so this should not happen; if it does, either the body
  //   moves so far in one step that it jumps past a particle, or its
  //   surfs coincide with grid cell boundaries, which makes the cut
  //   cells degenerate until the body moves off the alignment
  // checked once, on the last step of the run, to avoid a collective
  //   on every step

  if (!warndelete && update->ntimestep == update->laststep) {
    bigint all;
    MPI_Allreduce(&ndelrun,&all,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
    if (all && comm->me == 0) {
      char str[256];
      snprintf(str,sizeof(str),BIGINT_FORMAT " particles were deleted inside "
               "a rigid body during this run.  A body may be moving too "
               "far per timestep, or its surfs may lie exactly on grid cell "
               "boundaries",all);
      error->warning(FLERR,str);
    }

    // re-arm the once-per-run warnings for a following run,
    //   which may skip init() (run ... pre no)

    ndelrun = 0;
    warnrotate = warntranslate = warnexit = warnfallback = 0;
  }
}

/* ----------------------------------------------------------------------
   second half kick of velocity Verlet with the end-of-step force/torque
   omega is recomputed from angmom with the end-of-step axes
------------------------------------------------------------------------- */

void FixRigid::final_kick(int ibody)
{
  double dt = update->dt;
  double dtfhalf = 0.5 * dt / massbody[ibody];
  double dthalf = 0.5 * dt;

  double *vcm1 = vcm[ibody];
  double *fcm1 = fcm[ibody];
  double *angmom1 = angmom[ibody];
  double *torque1 = torque[ibody];
  double *omega1 = omega[ibody];

  vcm1[0] += dtfhalf * (fcm1[0] + fext[0]);
  vcm1[1] += dtfhalf * (fcm1[1] + fext[1]);
  vcm1[2] += dtfhalf * (fcm1[2] + fext[2]);

  angmom1[0] += dthalf * torque1[0];
  angmom1[1] += dthalf * torque1[1];
  angmom1[2] += dthalf * torque1[2];

  MathExtra::angmom_to_omega(angmom1,ex_space[ibody],ey_space[ibody],
                             ez_space[ibody],inertia[ibody],omega1);

  // guard against numeric drift out of the degrees of freedom the body
  //   has: rotation about z only in 2d, translation along x and spin
  //   about x only in an axisymmetric domain

  if (axiflag) {
    vcm1[1] = vcm1[2] = 0.0;
    angmom1[1] = angmom1[2] = 0.0;
    omega1[1] = omega1[2] = 0.0;
  } else if (dim == 2) {
    vcm1[2] = 0.0;
    angmom1[0] = 0.0;
    angmom1[1] = 0.0;
    omega1[0] = 0.0;
    omega1[1] = 0.0;
  }
}

/* ----------------------------------------------------------------------
   set invmass and the space-frame inverse inertia tensor of one body
     from its current principal axes and moments, used by the particle
     mover to correct collisions for the recoil of the finite-mass body
   Iinv = sum over K of (1/inertia[K]) e_K outer-product e_K
   for 2d only rotation about z is possible, so Iinv has only a zz
     component = 1/Izz, which keeps the in-plane response decoupled
------------------------------------------------------------------------- */

void FixRigid::set_recoil(int ibody)
{
  int i,j,k;
  double *e[3] = {ex_space[ibody],ey_space[ibody],ez_space[ibody]};
  double *inertia1 = inertia[ibody];
  double *invi = invinertia[ibody];

  invmass[ibody] = 1.0 / massbody[ibody];
  for (k = 0; k < 9; k++) invi[k] = 0.0;

  // axisymmetric: the body can only spin about x, and the axes are
  //   pinned to the identity, so only 1/ixx is ever needed.  it is
  //   stored in slot 0, where Geometry::rigid_recoil() reads it for the
  //   axisymmetric kmat

  if (axiflag) {
    invi[0] = 1.0 / inertia1[0];

  } else if (dim == 2) {
    double izz = 0.0;
    for (k = 0; k < 3; k++) izz += inertia1[k]*e[k][2]*e[k][2];
    invi[8] = 1.0 / izz;
  } else {
    for (k = 0; k < 3; k++)
      for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
          invi[3*i+j] += e[k][i]*e[k][j] / inertia1[k];
  }
}

/* ----------------------------------------------------------------------
   check that the body surfs form one or more closed (watertight) objects
   2d: each point must appear exactly as often as the 1st endpoint of a
     line as it does as the 2nd endpoint of a line
   3d: each edge must be traversed the same number of times in each
     direction by the tris that share it
   axisymmetric: a profile end point lying exactly on the axis needs no
     match, since the surface of revolution it generates is closed there
     (a semicircle generates a sphere, whose poles are not holes).  this
     is the same exception the Surf class makes for a surf end point on
     the box surface, of which the axis y = 0 is one face.
     no further check is needed once the other end points all match:
     every line contributes +1 at one point and -1 at another, so the
     counts sum to zero over all points, and if the only non-zero ones
     are on the axis then every open chain of the profile has both of
     its ends there.  such a chain generates a closed surface, so the
     body is closed
   matching of points is on exact floating point values, the same as
     the watertight checks applied to all surfs by the Surf class
   all procs store all surfs, so the check is identical on every proc
------------------------------------------------------------------------- */

void FixRigid::check_watertight(int ibody)
{
  int unmatched = 0;
  int istart = bodystart[ibody];
  int istop = bodystart[ibody+1];

  if (dim == 2) {
    std::map<std::array<double,2>,int> count;
    std::array<double,2> key;

    for (int i = istart; i < istop; i++) {
      key[0] = bodypt[i][0][0]; key[1] = bodypt[i][0][1];
      count[key]++;
      key[0] = bodypt[i][1][0]; key[1] = bodypt[i][1][1];
      count[key]--;
    }

    for (std::map<std::array<double,2>,int>::iterator it = count.begin();
         it != count.end(); ++it) {
      if (it->second == 0) continue;
      if (axiflag && it->first[1] == 0.0) continue;
      unmatched++;
    }

  } else {
    std::map<std::array<double,6>,int> count;
    std::array<double,6> key;
    double *pts[4];
    double *a,*b;
    int dir;

    for (int i = istart; i < istop; i++) {
      pts[0] = bodypt[i][0]; pts[1] = bodypt[i][1];
      pts[2] = bodypt[i][2]; pts[3] = bodypt[i][0];

      for (int j = 0; j < 3; j++) {
        a = pts[j];
        b = pts[j+1];

        // store edge with endpoints in canonical order
        // count is +1 if traversed in that order, -1 if reversed

        dir = 1;
        if (b[0] < a[0] ||
            (b[0] == a[0] &&
             (b[1] < a[1] || (b[1] == a[1] && b[2] < a[2])))) {
          std::swap(a,b);
          dir = -1;
        }

        key[0] = a[0]; key[1] = a[1]; key[2] = a[2];
        key[3] = b[0]; key[4] = b[1]; key[5] = b[2];
        count[key] += dir;
      }
    }

    for (std::map<std::array<double,6>,int>::iterator it = count.begin();
         it != count.end(); ++it)
      if (it->second != 0) unmatched++;
  }

  if (unmatched) {
    char str[128];
    if (dim == 2)
      sprintf(str,"is not watertight: %d unmatched points",unmatched);
    else
      sprintf(str,"is not watertight: %d unmatched edges",unmatched);
    body_error(ibody,str);
  }
}

/* ----------------------------------------------------------------------
   check that the body surfs enclose a non-zero area (2d) or volume (3d)
   a zero-thickness body, e.g. a line or tri traversed once in each
     direction, passes the watertight check but has no interior:
     the cut-cell routines cannot mark cells inside/outside it
   measure = signed area via the shoelace sum over the lines, or
     signed volume via the divergence theorem over the tris,
     each computed relative to the centroid of the body points
     so that round-off is set by the body extent, not its position
   axisymmetric: the measure is instead the volume of revolution the
     profile generates, pi sum (r1^2 + r1 r2 + r2^2) (x2-x1) / 3, which
     is the same divergence-theorem reduction and is valid for a profile
     whose free ends terminate on the axis, since the surface of
     revolution is closed there.  the planar area the profile encloses
     is meaningless for such a body, and is zero for a body which is a
     thin shell of revolution, so the volume is the right measure
   the sign is tested too: the normals must point outward (see below)
   all procs store all surfs, so the check is identical on every proc
------------------------------------------------------------------------- */

void FixRigid::check_enclosed(int ibody)
{
  int i,j,k;
  double c[3],a[3],b[3],d[3],e[3];

  int npoint = dim;
  int istart = bodystart[ibody];
  int istop = bodystart[ibody+1];
  double lo[3],hi[3];
  lo[0] = lo[1] = lo[2] = BIG;
  hi[0] = hi[1] = hi[2] = -BIG;
  c[0] = c[1] = c[2] = 0.0;

  for (i = istart; i < istop; i++)
    for (j = 0; j < npoint; j++)
      for (k = 0; k < 3; k++) {
        c[k] += bodypt[i][j][k];
        lo[k] = MIN(lo[k],bodypt[i][j][k]);
        hi[k] = MAX(hi[k],bodypt[i][j][k]);
      }
  for (k = 0; k < 3; k++) c[k] /= (istop-istart)*npoint;

  double extent = MAX(hi[0]-lo[0],hi[1]-lo[1]);
  if (dim == 3) extent = MAX(extent,hi[2]-lo[2]);

  double measure = 0.0;

  if (axiflag) {
    for (i = istart; i < istop; i++) {
      double r1 = bodypt[i][0][1];
      double r2 = bodypt[i][1][1];
      measure += MY_PI3 * (r1*r1 + r1*r2 + r2*r2) *
        (bodypt[i][1][0] - bodypt[i][0][0]);
    }
  } else if (dim == 2) {
    for (i = istart; i < istop; i++) {
      MathExtra::sub3(bodypt[i][0],c,a);
      MathExtra::sub3(bodypt[i][1],c,b);
      measure += a[0]*b[1] - a[1]*b[0];
    }
    measure *= 0.5;
  } else {
    for (i = istart; i < istop; i++) {
      MathExtra::sub3(bodypt[i][0],c,a);
      MathExtra::sub3(bodypt[i][1],c,b);
      MathExtra::sub3(bodypt[i][2],c,d);
      MathExtra::cross3(b,d,e);
      measure += MathExtra::dot3(a,e);
    }
    measure /= 6.0;
  }

  // scale = extent^dim, a body thinner than EPSENCLOSED of its extent
  //   has no usable interior on any grid that could resolve it

  double scale = extent*extent;
  if (dim == 3 || axiflag) scale *= extent;

  if (fabs(measure) <= EPSENCLOSED*scale) {
    if (dim == 2 && !axiflag) body_error(ibody,"encloses zero area");
    else body_error(ibody,"encloses zero volume");
  }

  // the normals must point outward, so the body is an object with the
  //   gas outside it: the point-in-body parity test ignores the normal
  //   direction, so a body traversed the other way round (a container,
  //   with the gas inside) would have its interior and exterior swapped
  //   relative to the cut-cell typing and lose every particle inside it
  // sign of the enclosed measure: 2d lines with outward normals run
  //   clockwise (the normal is to the left of p1 -> p2), 3d triangles
  //   with outward normals run counter-clockwise seen from outside
  // axisymmetric: the measure is the volume between the profile and the
  //   axis, not the area the profile encloses in the plane, and an
  //   outward normal (-dr,dx) points away from the axis where dx > 0,
  //   so an outward-facing profile runs in +x and gives a positive
  //   volume, the opposite sign convention from the planar 2d case

  int inward = 0;
  if (axiflag) {
    if (measure < 0.0) inward = 1;
  } else {
    if (dim == 2 && measure > 0.0) inward = 1;
    if (dim == 3 && measure < 0.0) inward = 1;
  }
  if (inward) body_error(ibody,"surf normals point inward");
}

/* ----------------------------------------------------------------------
   memory usage of the replicated body tables and per-step work arrays
------------------------------------------------------------------------- */

double FixRigid::memory_usage()
{
  double bytes = 0.0;

  // replicated body: geometry, per-element tables, element index maps

  bytes += (double) nsurf * dim * 3 * sizeof(double);     // bodypt
  bytes += (double) nsurf * 3 * sizeof(double);           // bodynorm
  bytes += (double) nsurf * dim * 3 * sizeof(double);     // displace
  bytes += (double) nsurf * 6 * sizeof(double);           // elemlo/elemhi
  bytes += (double) nsurf * (sizeof(surfint) + 6*sizeof(int));  // tables
  bytes += (double) nsurf * 2 * sizeof(int);              // slist,lblist
  bytes += (double) nsurf * 2 * sizeof(int);              // olist_own/elem
  bytes += (double) maxcopy * 2 * sizeof(int);            // copy_index/elem
  bytes += (double) nsurf * (sizeof(surfint) + sizeof(int) +
                             3*sizeof(void *));            // idmap, approx

  // per local surf and per local cell

  bytes += (double) nsurfall * sizeof(int);               // irigid
  if (pushstamp) {
    int nbins = pushnbin[0]*pushnbin[1]*pushnbin[2];
    bytes += (double) (nbins+1) * sizeof(int);            // pushbinstart
    bytes += (double) pushbinstart[nbins] * sizeof(int);  // pushbinlist
    bytes += (double) surf->nlocal * sizeof(int);         // pushstamp
  }
  bytes += (double) maxswcell * 2 * sizeof(int);          // swstamp,swhead
  bytes += (double) maxswcells * sizeof(int);             // swcells
  bytes += (double) maxent * (sizeof(int) + sizeof(surfint)); // entries
  bytes += (double) maxoldinside * sizeof(int);           // oldinside
  bytes += (double) maxrcand * sizeof(int);               // rcand

  // re-cut work bufs, restore lists, swept lists page, registry

  bytes += (double) maxmodified * (2*sizeof(int) + sizeof(surfint *));
  bytes += (double) maxreclist * sizeof(surfint);
  bytes += (double) 2 * maxnewlist * sizeof(int);
  if (cpage) bytes += (double) cpage->size();
  bytes += (double) registry.size() *
    (sizeof(std::pair<int,surfint *>) + grid->maxsurfpercell*sizeof(surfint));
  bytes += (double) csplitreg.size() *
    (sizeof(std::pair<int,int *>) + grid->maxsurfpercell*sizeof(int));
  bytes += (double) csubreg.size() *
    (sizeof(std::pair<int,int *>) + grid->maxsplitpercell*sizeof(int));

  // per-body arrays, including the ftbuf work buffers

  bytes += (double) nbody * 90 * sizeof(double);
  bytes += (double) (nbody+1) * sizeof(int);              // bodystart
  bytes += (double) nsurf * sizeof(int);                  // body
  bytes += (double) nbody * sizeof(int);                  // bodyneed
  if (bodybinstart) {
    int nbins = bodynbin[0]*bodynbin[1]*bodynbin[2];
    bytes += (double) (nbins+1+nbody) * sizeof(int);      // body bins
  }
  bytes += (double) maxbodycand * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   return cummulative count of particles deleted inside the moving bodies
------------------------------------------------------------------------- */

double FixRigid::compute_scalar()
{
  // ndeleted is a per-proc count, summed across procs on demand
  //   rather than with a collective every step
  // cached per timestep: repeated outputs on one step reduce once
  // like all fix scalar outputs, must be accessed on all procs

  if (ndelvalid != update->ntimestep) {
    MPI_Allreduce(&ndeleted,&ndeleted_all,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
    ndelvalid = update->ntimestep;
  }
  return (double) ndeleted_all;
}

/* ----------------------------------------------------------------------
   return properties of a single rigid body, the vector is only defined
     when nbody = 1
------------------------------------------------------------------------- */

double FixRigid::compute_vector(int index)
{
  return compute_array(0,index);
}

/* ----------------------------------------------------------------------
   return properties of rigid body I, 22 columns
------------------------------------------------------------------------- */

double FixRigid::compute_array(int i, int index)
{
  if (index < 3) return xcm[i][index];
  if (index < 6) return vcm[i][index-3];
  if (index < 9) return fcm[i][index-6];
  if (index < 12) return torque[i][index-9];
  if (index < 15) return omega[i][index-12];
  if (index < 19) return quat[i][index-15];
  if (index < 22) return fpush[i][index-19];

  return 0.0;
}
