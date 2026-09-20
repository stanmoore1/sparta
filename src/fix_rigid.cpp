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
#include <algorithm>
#include "fix_rigid.h"
#include "rigid_remap.h"
#include "rigid_contact.h"
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
#include "math_extra.h"
#include "math_eigen.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

static constexpr double EPSILON = 1.0e-7;

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
  if (narg < 5) error->all(FLERR,"Illegal fix rigid command");

  scalar_flag = 1;
  global_freq = 1;
  nevery = 1;

  // gridmigrate insures grid_changed() is invoked when grid cells
  // are rebuilt or migrated, so incremental re-cut data can be reset

  gridmigrate = 1;

  // ghost sub cells route migrating particles to their split cell, since
  //   the sub cells of the owner are restructured in place without
  //   informing the ghost copies (see Grid::subroute)

  grid->subroute = 1;

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

  // bodystyle = how surfs in the group are assigned to bodies

  int n;
  customname = NULL;
  int iarg = 3;
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
  initflag = 0;
  posesplit = 0;
  infile = NULL;
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
  pushstyle = RigidContact::LINEAR;
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
      if (strcmp(arg[iarg+1],"linear") == 0)
        pushstyle = RigidContact::LINEAR;
      else if (strcmp(arg[iarg+1],"hertz") == 0)
        pushstyle = RigidContact::HERTZ;
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

  // per-surf maps are built by init_surfs() at the start of each run,
  //   once the local+ghost surf arrays of the run are final

  surfbody = surfelem = NULL;
  maxsurfmap = 0;
  nsurfall = surf->nlocal;

  // remap data structs
  // body surfs are cut/split into grid cells by the normal surf
  //   pipeline (done at read_surf time), so no special setup is needed
  //   here beyond the swept-assignment and incremental work buffers

  ndeleted = 0;
  ndeleted_all = 0;
  ndelvalid = -1;

  // elemlo/elemhi are allocated by setup_body() above
  // the re-map of the body surfs to grid cells, now that nbody is known

  remap = new RigidRemap(sparta,this);

  contact = NULL;
  if (pushflag)
    contact = new RigidContact(sparta,this,pushstyle,kpush,pushcutoff,
                               gammapush,pushboundflag);

  bodybinstart = NULL;
  bodybinlist = NULL;
  bodycand = NULL;
  maxbodycand = 0;
  copiesappended = 0;
  ntally = maxtally = 0;
  tally2elem = elem2tally = NULL;
  ftally = NULL;
  nfactor_inverse = 0.0;
  weightflag = 0;
  ftbuf_mine = ftbuf_all = NULL;
  warnfallback = 0;
  warndelete = 0;
  ndelrun = 0;
}

/* ---------------------------------------------------------------------- */

FixRigid::~FixRigid()
{
  // a KOKKOS functor copy of this fix (fix rigid/kk passes *this to a
  //   parallel_reduce) shares these pointers with the original and must
  //   not free them when the copy goes out of scope

  if (copy || copymode) return;

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
  memory->destroy(fcm_infile);
  memory->destroy(torque_infile);
  memory->destroy(body);
  memory->destroy(bodystart);

  memory->destroy(displace);
  memory->destroy(surfbody);
  memory->destroy(surfelem);
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
  delete remap;

  delete contact;
  memory->destroy(bodybinstart);
  memory->destroy(bodybinlist);
  memory->destroy(bodycand);
  memory->destroy(bodyneed);
  memory->destroy(tally2elem);
  memory->destroy(elem2tally);
  memory->destroy(ftally);
  memory->destroy(ftbuf_mine);
  memory->destroy(ftbuf_all);
}

/* ---------------------------------------------------------------------- */

int FixRigid::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;
  mask |= END_OF_STEP;
  mask |= POST_RUN;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRigid::init()
{
  // check that global rigid flag is set

  if (update->rigidflag == 0)
    error->all(FLERR,"Cannot use fix rigid unless global rigid is set");

  // ghost sub cells acquired before this fix was defined still route
  //   to the owner's sub cells

  grid->route_ghost_subcells();

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

  // the mover tallies the momentum each collision gives a body surf,
  //   surf_tally() converts it to a force with the same factor as
  //   compute surf, and a particle's cell weight scales its mass

  double nfactor = update->dt/update->fnum;
  nfactor_inverse = 1.0/nfactor;
  weightflag = grid->cellweightflag;

  // body surfs cannot be transparent, and may only carry a surf_react
  //   model in which every reaction leaves exactly one particle: the
  //   body's recoil correction maps one incoming particle to one
  //   outgoing one, so a reaction which destroys the particle, produces
  //   a second, or adsorbs it onto the surface has no defined recoil
  //   and would also change the body's mass, which this fix holds fixed
  // attributes come from the replicated body table, valid for both
  //   non-distributed and distributed surfs

  // surfs cannot change once a fix rigid is defined:
  //   removal invalidates the body element table; a change to the
  //   body group invalidates the body definition
  // surfs appended after the fix was defined are allowed, and static

  if (surf->count_group(igroup) != ngroupsurf)
    error->all(FLERR,"Fix rigid body surf group was changed "
               "after fix rigid was defined");

  if (!surf->distributed && surf->nlocal < nsurfall)
    error->all(FLERR,"Surfs were removed after fix rigid was defined");

  // the replicated table of body surf attributes was captured when the
  //   fix was defined: refresh it from the current surfs, or with
  //   distributed surfs, where the local copies of body surfs are built
  //   from it, error if the attributes were changed since, e.g. by a
  //   surf_modify command issued after this fix

  check_body_attributes();

  for (int i = 0; i < nsurf; i++) {
    if (bodytrans[i])
      error->all(FLERR,"Fix rigid body surfs cannot be transparent");
    if (bodyisr[i] >= 0 && !surf->sr[bodyisr[i]]->one_product_only())
      error->all(FLERR,"Fix rigid body surfs can only use a surf_react "
                 "model in which every reaction leaves exactly one "
                 "particle");
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
  nstep_run = nstep_inplace = nstep_rebuild = nstep_fallback = 0;
  timeflag = (getenv("SPARTA_RIGID_TIMING") != NULL);
  for (int i = 0; i < T_NSTAGE; i++) stagetime[i] = 0.0;
  remap->ncand_run = remap->nlist_run = remap->ncut_run = 0;

  // fix rigid must be defined before fixes which change the grid,
  // so its end_of_step() restores overlaid grid cells before they run

  int myindex = modify->find_fix(id);
  for (int ifix = 0; ifix < myindex; ifix++)
    if (strncmp(modify->fix[ifix]->style,"balance",7) == 0 ||
        strncmp(modify->fix[ifix]->style,"adapt",5) == 0 ||
        strncmp(modify->fix[ifix]->style,"move/surf",9) == 0)
      error->all(FLERR,"Fix rigid must be defined before fix balance, "
                 "fix adapt, or fix move/surf");

  initflag = 1;
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
    surf_maps();
    surfs_changed(changed,2);
    if (changed) remap->listschanged = 1;
  }

  // work bufs of the re-map, sized for this run's global surfmax

  remap->setup();

  // bin static surfs for push-off candidate pruning

  if (contact) contact->setup();

  // delete any particles inside a body
  // create_particles marks the bodies' cells INSIDE via the surf pipeline
  //   and normally avoids them, but this is a safety net for any that
  //   end up inside, e.g. via an emit region overlapping a body

  if (particle->exist) {
    particles_to_host();
    ndeleted += remove_inside_particles(0);
  }

}

/* ---------------------------------------------------------------------- */

void FixRigid::start_of_step()
{
  // a fix defined after the last init (e.g. re-defined before a
  //   "run pre no") has no body state to advance

  if (!initflag)
    error->all(FLERR,"Fix rigid was not initialized before the run");

  stage_begin();
  initial_integrate();
  swept_boxes();
  stage_end(T_INTEGRATE);

  // distributed surfs: a body sweeping into this proc's cells for the
  //   first time since the last grid rebuild needs local copies of its
  //   surfs here, before the swept lists and the mover reference them
  // collective: surfs_changed() reduces whether any proc appended

  if (surf->distributed) {
    stage_begin();
    int changed = ensure_local_copies();
    if (changed) {
      surf_maps();
      remap->listschanged = 1;
      copiesappended = 1;
    }
    surfs_changed(changed,0);
    stage_end(T_COPIES);
  }

  // augment collision lists of all cells any body sweeps through during
  //   the step, so particles in the swept paths are tested against the
  //   moving surfs and reflected rather than overtaken and later deleted

  stage_begin();
  remap->collision_lists();
  stage_end(T_COLLIDELIST);

  // the mover tallies the force/torque of this step's collisions

  clear_tally();
}

/* ---------------------------------------------------------------------- */

void FixRigid::end_of_step()
{
  // undo the swept collision-list augmentation from start_of_step

  stage_begin();
  remap->reset_collision_lists();
  stage_end(T_COLLIDELIST);

  // force and torque on each body from this step's collisions

  stage_begin();
  sum_forces();
  stage_end(T_FORCES);

  // for incremental remap: record cells interior to the bodies
  //   before their surfs move to their end-of-step positions

  stage_begin();
  if (remapmode == INCREMENTAL) remap->refresh();

  // move every body to its end-of-step pose and regenerate its geometry

  set_xv();
  check_bounds();
  stage_end(T_SETXV);

  // push-off contacts and the second half kick

  stage_begin();
  final_integrate();
  stage_end(T_CONTACT);

  // write body states to the output file every outevery steps, now that
  //   velocities and forces are complete; the file is compatible with
  //   the infile option for run continuation

  if (outfile && update->ntimestep % outevery == 0) write_outfile();

  // re-map the body surfs to the grid cells

  remap_grid();
}

/* ----------------------------------------------------------------------
   per-element swept bounding boxes of every body for this step, from
     the start-of-step geometry and the end-of-step pose
------------------------------------------------------------------------- */

void FixRigid::swept_boxes()
{
  posesplit = 1;
  for (int ibody = 0; ibody < nbody; ibody++) body_bbox(ibody,1);
}

/* ----------------------------------------------------------------------
   first half of velocity Verlet for every body: half kick of vcm and
     angmom with the force/torque of the previous step, drift of the COM
     and orientation by a full step to the end-of-step pose
   xcm/quat stay at the start-of-step pose (the grid holds that geometry)
     and xcmnew/quatnew/ex,ey,ez_space are the end-of-step values the
     moving-surf collision tests interpolate between
------------------------------------------------------------------------- */

void FixRigid::initial_integrate()
{
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
}

/* ----------------------------------------------------------------------
   commit the end-of-step pose of every body and regenerate its geometry
     from it: corner pts from displace rotated to the space frame + COM,
     normals from the corner pts, then the Surf copies the mover and cut
     pipeline read, the current bboxes and the bins of bodies by COM
------------------------------------------------------------------------- */

void FixRigid::set_xv()
{
  set_pose();
  for (int ibody = 0; ibody < nbody; ibody++) body_geometry(ibody);
  update_surf_copies();
  body_bins();
}

/* ----------------------------------------------------------------------
   move every body to the end-of-step pose initial_integrate() computed:
     xcm/quat take xcmnew/quatnew, and the body's degrees of freedom are
     enforced on all its properties
   2d: in-plane motion and rotation about z.  start_of_step() enforces it
     on xcmnew and omega; quat stays a rotation about z since omega is
     along z
   axisymmetric: translation along x and spin about x.  axi_project()
     already removed the transverse force and torque, so this only
     guards against drift; quat is pinned to the identity in
     start_of_step()
------------------------------------------------------------------------- */

void FixRigid::set_pose()
{
  posesplit = 0;

  for (int ibody = 0; ibody < nbody; ibody++) {
    double *xcm1 = xcm[ibody];
    double *quat1 = quat[ibody];

    xcm1[0] = xcmnew[ibody][0];
    xcm1[1] = xcmnew[ibody][1];
    xcm1[2] = xcmnew[ibody][2];

    quat1[0] = quatnew[ibody][0];
    quat1[1] = quatnew[ibody][1];
    quat1[2] = quatnew[ibody][2];
    quat1[3] = quatnew[ibody][3];

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
  }
}

/* ----------------------------------------------------------------------
   regenerate the replicated geometry of one body from its current pose:
     corner pts from displace rotated to the space frame + the COM,
     normals recomputed from the corner pts, then the per-element boxes
     and the body bbox
   matvec() converts a displace vector from body frame to space frame
   the same arithmetic as the device kernel of fix rigid/kk, so a body
     regenerated here for a host consumer matches the device geometry
------------------------------------------------------------------------- */

void FixRigid::body_geometry(int ibody)
{
  int i,j;
  double z[3],delta[3],delta12[3],delta13[3];
  double exq[3],eyq[3],ezq[3];
  z[0] = 0.0; z[1] = 0.0; z[2] = 1.0;

  double *xcm1 = xcm[ibody];

  // between initial_integrate() and set_pose() the stored frame is the
  //   end-of-step one while xcm is still the start-of-step COM: the
  //   start-of-step frame is that of quat, the same values ex/ey/ez_space
  //   held when the geometry was last generated from it

  double *ex = ex_space[ibody];
  double *ey = ey_space[ibody];
  double *ez = ez_space[ibody];
  if (posesplit) {
    MathExtra::q_to_exyz(quat[ibody],exq,eyq,ezq);
    ex = exq; ey = eyq; ez = ezq;
  }

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

      MathExtra::matvec(ex,ey,ez,displace[i][j],delta);
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

  body_bbox(ibody,0);
}

/* ----------------------------------------------------------------------
   a body may exit the box through a non-periodic boundary, but body
     coords are not wrapped, so one which reaches a periodic boundary
     is an error; warn once if a body no longer touches the box at all
------------------------------------------------------------------------- */

void FixRigid::check_bounds()
{
  int ibody;
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

}

/* ----------------------------------------------------------------------
   second half of velocity Verlet for every body, once the force and
     torque of the step are complete: the push-off contacts of the
     bodies at their end-of-step poses are added to the collision
     force/torque, then vcm/angmom/omega get the second half kick
------------------------------------------------------------------------- */

void FixRigid::final_integrate()
{
  int ibody;

  for (ibody = 0; ibody < nbody; ibody++) {
    fpush[ibody][0] = fpush[ibody][1] = fpush[ibody][2] = 0.0;
    tqpush[ibody][0] = tqpush[ibody][1] = tqpush[ibody][2] = 0.0;
  }

  if (contact) {
    contact->compute(fpush,tqpush);

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

}

/* ----------------------------------------------------------------------
   re-map the body surfs to the grid cells from their new positions:
     the incremental re-cut of only the cells near the bodies, or the
     full re-map when it declines or remap cutcell is set, then delete
     the particles the bodies swept over
------------------------------------------------------------------------- */

void FixRigid::remap_grid()
{
  int fallback = 1;
  int structural = 0;
  int rebuild = 0;
  stage_begin();
  if (remapmode == INCREMENTAL) {
    int mine[4],all[4];
    mine[0] = remap->recut();
    mine[1] = (remap->npending > 0);
    mine[2] = remap->typechanged;
    mine[3] = remap->rebuild_needed();
    MPI_Allreduce(mine,all,4,MPI_INT,MPI_MAX,world);
    fallback = all[0];
    structural = all[1];
    rebuild = all[3];

    // an incremental re-cut which changed cell markings or cells must
    //   be seen by emit fixes, whose per-cell tasks depend on them; a
    //   full re-map and the collective rebuild both notify them via
    //   Grid::notify_changed()

    if (!fallback && !rebuild && (all[2] || structural)) {
      refresh_host_surfs();
      for (int ifix = 0; ifix < modify->nfix; ifix++)
        if (strncmp(modify->fix[ifix]->style,"emit",4) == 0)
          modify->fix[ifix]->grid_changed();
    }
    remap->typechanged = 0;

    if (fallback && !warnfallback) {
      warnfallback = 1;
      if (comm->me == 0)
        error->warning(FLERR,"Fix rigid incremental remap fell back to a "
                       "full grid re-map because a cell would exceed "
                       "global surfmax");
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

  if (!fallback && (remap->splitchanged || structural) &&
      particle->exist && grid->nsplitlocal)
    combine_split_all();

  // a cell which gained or lost sub cells is restructured in place,
  //   which is far cheaper than the full re-map it used to force: the
  //   surf lists, volumes and cell types are already correct, so only
  //   the sub cells change; every proc enters together, since it
  //   reduces the cell counts, or rebuilds the ghosts when it must

  nstep_run++;
  if (fallback) nstep_fallback++;
  else if (structural && rebuild) nstep_rebuild++;
  else if (structural) nstep_inplace++;
  stage_end(T_RECUT);

  stage_begin();
  if (fallback) grid_rebuild();
  else if (structural) {
    if (particle->exist) sort_for_split_rebuild();
    remap->apply_pending(rebuild);
    relabel_moved_cells();
  }
  remap->npending = 0;
  stage_end(T_APPLY);

  // remove particles inside any body in one pass over particles,
  //   with split-cell reassignment after a full re-map or after a
  //   split cell was re-cut in place; no reduction here, deletion
  //   counts stay per-proc and are reduced lazily by compute_scalar()

  stage_begin();
  if (particle->exist)
    remove_inside_all(fallback || remap->splitchanged || structural);
  stage_end(T_REMOVE);
}

/* ----------------------------------------------------------------------
   per-run summary of the re-map paths taken, for performance work
------------------------------------------------------------------------- */

void FixRigid::post_run()
{
  if (!getenv("SPARTA_RIGID_TIMING")) return;

  bigint mine[4],all[4];
  mine[0] = nstep_run;
  mine[1] = nstep_inplace;
  mine[2] = nstep_rebuild;
  mine[3] = nstep_fallback;
  MPI_Allreduce(mine,all,4,MPI_SPARTA_BIGINT,MPI_MAX,world);

  if (comm->me == 0) {
    char str[256];
    sprintf(str,"Fix rigid re-map: " BIGINT_FORMAT " steps, "
            BIGINT_FORMAT " in place, " BIGINT_FORMAT " ghost rebuilds, "
            BIGINT_FORMAT " full re-maps (max over procs)\n",
            all[0],all[1],all[2],all[3]);
    if (screen) fprintf(screen,"%s",str);
    if (logfile) fprintf(logfile,"%s",str);
  }

  // per-stage times, max over procs

  const char *names[T_NSTAGE] =
    {"integrate+bbox","surf copies","collision lists","tally",
     "sum forces","set_xv+bounds","contacts+kick","recut",
     "  recut: surf lists","  recut: cuts","  recut: retyping",
     "apply split changes","remove inside"};
  double tmax[T_NSTAGE];
  MPI_Allreduce(stagetime,tmax,T_NSTAGE,MPI_DOUBLE,MPI_MAX,world);
  if (comm->me == 0) {
    for (int i = 0; i < T_NSTAGE; i++) {
      if (i == T_TALLY) continue;
      char str[128];
      sprintf(str,"Fix rigid time: %-20s %10.4f s\n",names[i],tmax[i]);
      if (screen) fprintf(screen,"%s",str);
      if (logfile) fprintf(logfile,"%s",str);
    }
  }

  // per-run counts of the re-cut, max over procs

  bigint cmine[3],call[3];
  cmine[0] = remap->ncand_run;
  cmine[1] = remap->nlist_run;
  cmine[2] = remap->ncut_run;
  MPI_Allreduce(cmine,call,3,MPI_SPARTA_BIGINT,MPI_MAX,world);
  if (comm->me == 0) {
    char str[256];
    sprintf(str,"Fix rigid re-cut: " BIGINT_FORMAT " candidate cells, "
            BIGINT_FORMAT " lists changed, " BIGINT_FORMAT
            " cells cut (max over procs)\n",call[0],call[1],call[2]);
    if (screen) fprintf(screen,"%s",str);
    if (logfile) fprintf(logfile,"%s",str);
  }
}

/* ----------------------------------------------------------------------
   reset the per-element force/torque tallies, before the mover runs
------------------------------------------------------------------------- */

void FixRigid::clear_tally()
{
  if (!elem2tally) {
    memory->create(elem2tally,nsurf,"fix_rigid:elem2tally");
    for (int k = 0; k < nsurf; k++) elem2tally[k] = -1;
  }
  for (int i = 0; i < ntally; i++) elem2tally[tally2elem[i]] = -1;
  ntally = 0;
}

/* ---------------------------------------------------------------------- */

void FixRigid::grow_tally()
{
  maxtally += DELTA_MODIFY;
  memory->grow(tally2elem,maxtally,"fix_rigid:tally2elem");
  memory->grow(ftally,maxtally,6,"fix_rigid:ftally");
}

/* ----------------------------------------------------------------------
   tally the force and torque one particle collision exerts on body
     surf isurf, called by the particle mover after the collision
   iorig = the particle before the collision, ip/jp = the one or two
     particles after it, ip = NULL if none
   the momentum the particle(s) lost, times fnum/dt, is the force on
     the surf; the torque is about the mid-step COM of the body, with
     the lever arm to the hit point, the same expressions as compute
     surf with com rigid, so the two agree bit for bit
------------------------------------------------------------------------- */

void FixRigid::surf_tally(int isurf, Particle::OnePart *iorig,
                          Particle::OnePart *ip, Particle::OnePart *jp)
{
  int k = surfelem[isurf];
  int itally = elem2tally[k];
  if (itally < 0) {
    if (ntally == maxtally) grow_tally();
    itally = ntally++;
    elem2tally[k] = itally;
    tally2elem[itally] = k;
    for (int j = 0; j < 6; j++) ftally[itally][j] = 0.0;
  }

  Particle::Species *species = particle->species;
  double weight = 1.0;
  if (weightflag) weight = iorig->weight;
  double origmass = species[iorig->ispecies].mass * weight;

  double pdelta[3],rdelta[3],torque[3];
  pdelta[0] = pdelta[1] = pdelta[2] = 0.0;
  MathExtra::axpy3(-origmass,iorig->v,pdelta);
  if (ip) MathExtra::axpy3(species[ip->ispecies].mass * weight,ip->v,pdelta);
  if (jp) MathExtra::axpy3(species[jp->ispecies].mass * weight,jp->v,pdelta);

  double *xcollide = ip ? ip->x : iorig->x;
  MathExtra::sub3(xcollide,xcmmid[body[k]],rdelta);
  MathExtra::cross3(rdelta,pdelta,torque);

  double *ft = ftally[itally];
  ft[0] -= pdelta[0] * nfactor_inverse;
  ft[1] -= pdelta[1] * nfactor_inverse;
  ft[2] -= pdelta[2] * nfactor_inverse;
  ft[3] -= torque[0] * nfactor_inverse;
  ft[4] -= torque[1] * nfactor_inverse;
  ft[5] -= torque[2] * nfactor_inverse;
}

/* ----------------------------------------------------------------------
   force and torque on every body from this step's collisions
   sum_tallies() forms each proc's per-body sums of its own tallies,
     then one Allreduce over all bodies merges them: O(tallies) local
     work and a single collective sized by the bodies, not the surfs
------------------------------------------------------------------------- */

void FixRigid::sum_forces()
{
  sum_tallies();

  MPI_Allreduce(ftbuf_mine,ftbuf_all,6*nbody,MPI_DOUBLE,MPI_SUM,world);

  for (int ibody = 0; ibody < nbody; ibody++) {
    fcm[ibody][0] = ftbuf_all[6*ibody];
    fcm[ibody][1] = ftbuf_all[6*ibody+1];
    fcm[ibody][2] = ftbuf_all[6*ibody+2];
    torque[ibody][0] = ftbuf_all[6*ibody+3];
    torque[ibody][1] = ftbuf_all[6*ibody+4];
    torque[ibody][2] = ftbuf_all[6*ibody+5];
    axi_project(fcm[ibody],torque[ibody]);
  }
}

/* ----------------------------------------------------------------------
   per-body sums of this proc's tallies into ftbuf_mine, element by
     element in element order, the order the device sums them in
------------------------------------------------------------------------- */

void FixRigid::sum_tallies()
{
  int itally;

  for (int i = 0; i < 6*nbody; i++) ftbuf_mine[i] = 0.0;
  if (!ntally) return;

  for (int ibody = 0; ibody < nbody; ibody++) {
    double *ft = &ftbuf_mine[6*ibody];
    for (int k = bodystart[ibody]; k < bodystart[ibody+1]; k++) {
      itally = elem2tally[k];
      if (itally < 0) continue;
      for (int j = 0; j < 6; j++) ft[j] += ftally[itally][j];
    }
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

// point or edge record of the watertight check: 6 coords and a count
// the comparison is on exact floating point values, the same as the
//   watertight checks applied to all surfs by the Surf class

struct EdgeRecord {
  double x[6];
  int count;
};

static int compare_edge_records(const void *a, const void *b)
{
  const double *x = ((const EdgeRecord *) a)->x;
  const double *y = ((const EdgeRecord *) b)->x;
  for (int k = 0; k < 6; k++) {
    if (x[k] < y[k]) return -1;
    if (x[k] > y[k]) return 1;
  }
  return 0;
}

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

  // lblist = local surf index of each body element on this proc, and
  //   the list of every local copy of a body element
  // non-distributed: every proc stores every surf, one copy per element
  // distributed: ensure_local_copies() refreshes both at setup and after
  //   every surf change
  // olist = owned-array index of the body elements this proc owns

  ncopy = maxcopy = 0;
  copy_index = copy_elem = NULL;
  scan_copies();

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
   lblist = local surf index of each body element on this proc, -1 if
     it stores none; copy_index/copy_elem = every local copy of a body
     element and the element it copies
   non-distributed surfs: every proc stores every surf, one copy each
   distributed surfs: the surf comm may leave several copies, and
     ensure_local_copies() appends the ones a proc lacks; every copy is
     tracked so all are kept at the body's current position
------------------------------------------------------------------------- */

void FixRigid::scan_copies()
{
  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;
  int nslocal = surf->nlocal;

  for (int k = 0; k < nsurf; k++) lblist[k] = -1;
  ncopy = 0;

  for (int i = 0; i < nslocal; i++) {
    surfint id = (dim == 2) ? lines[i].id : tris[i].id;
    int k = body_elem(id);
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
}

/* ----------------------------------------------------------------------
   per-surf maps over the local+ghost surfs, read by the particle mover
     to dispatch the moving-surf collision tests and by the force sum:
     surfbody = body a surf belongs to, surfelem = its body element,
     both -1 for a static surf
   rebuilt whenever the surf arrays change, at init and after
     distributed copies are appended
------------------------------------------------------------------------- */

void FixRigid::surf_maps()
{
  int i,k;

  int n = surf->nlocal + surf->nghost;
  if (n > maxsurfmap) {
    maxsurfmap = n;
    memory->destroy(surfbody);
    memory->destroy(surfelem);
    memory->create(surfbody,maxsurfmap,"fix_rigid:surfbody");
    memory->create(surfelem,maxsurfmap,"fix_rigid:surfelem");
  }
  for (i = 0; i < n; i++) surfbody[i] = surfelem[i] = -1;

  // non-distributed: every element has exactly one local copy
  // distributed: ghost surfs count too, the mover tests them

  if (!surf->distributed) {
    for (k = 0; k < nsurf; k++) {
      surfbody[lblist[k]] = body[k];
      surfelem[lblist[k]] = k;
    }
  } else {
    Surf::Line *lines = surf->lines;
    Surf::Tri *tris = surf->tris;
    for (i = 0; i < n; i++) {
      surfint id = (dim == 2) ? lines[i].id : tris[i].id;
      k = body_elem(id);
      if (k < 0) continue;
      surfbody[i] = body[k];
      surfelem[i] = k;
    }
  }

  update->rigid_maps_changed(this);
}

/* ----------------------------------------------------------------------
   per-run setup of the body surfs, called by Update::init_rigid()
     before the surface collision and reaction models init, so that
     their per-surf state is sized for the final local+ghost surf arrays
------------------------------------------------------------------------- */

void FixRigid::init_surfs()
{
  if (surf->distributed) {
    int changed = ensure_local_copies();
    surf_maps();
    surfs_changed(changed,1);
  } else surf_maps();
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
   called at setup and from grid_changed() after any grid/surf change;
     refreshes lblist and the list of all local copies
   caller is responsible for surf_maps() and, if surfs were appended,
     surfs_changed()
   returns 1 if surfs were appended, else 0
------------------------------------------------------------------------- */

int FixRigid::ensure_local_copies()
{
  int i,k;

  if (!surf->distributed) return 0;

  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;
  int nslocal = surf->nlocal;
  int nsghost = surf->nghost;

  scan_copies();

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

  // build the missing copies from the body table and append them to
  //   the local range; Surf re-packs the ghosts after them and maps each
  //   old ghost index to its new one, so the ghost cells' cut lists can
  //   follow (a ghost copy of an appended element maps to that copy)

  int *gmap;
  memory->create(gmap,MAX(nsghost,1),"fix_rigid:gmap");

  if (dim == 2) {
    Surf::Line *copies = new Surf::Line[nmissing];
    int n = 0;
    for (k = 0; k < nsurf; k++) {
      if (lblist[k] >= 0 || !bodyneed[body[k]]) continue;
      Surf::Line *line = &copies[n++];
      memset(line,0,sizeof(Surf::Line));
      line->id = sids[k];
      line->type = bodytype[k];
      line->mask = bodymask[k];
      line->transparent = bodytrans[k];
      line->isc = bodyisc[k];
      line->isr = bodyisr[k];
      host_geometry(body[k]);
      memcpy(line->p1,bodypt[k][0],3*sizeof(double));
      memcpy(line->p2,bodypt[k][1],3*sizeof(double));
      memcpy(line->norm,bodynorm[k],3*sizeof(double));
    }
    surf->add_local_copies(nmissing,copies,NULL,gmap);
    delete [] copies;
  } else {
    Surf::Tri *copies = new Surf::Tri[nmissing];
    int n = 0;
    for (k = 0; k < nsurf; k++) {
      if (lblist[k] >= 0 || !bodyneed[body[k]]) continue;
      Surf::Tri *tri = &copies[n++];
      memset(tri,0,sizeof(Surf::Tri));
      tri->id = sids[k];
      tri->type = bodytype[k];
      tri->mask = bodymask[k];
      tri->transparent = bodytrans[k];
      tri->isc = bodyisc[k];
      tri->isr = bodyisr[k];
      host_geometry(body[k]);
      memcpy(tri->p1,bodypt[k][0],3*sizeof(double));
      memcpy(tri->p2,bodypt[k][1],3*sizeof(double));
      memcpy(tri->p3,bodypt[k][2],3*sizeof(double));
      memcpy(tri->norm,bodynorm[k],3*sizeof(double));
    }
    surf->add_local_copies(nmissing,NULL,copies,gmap);
    delete [] copies;
  }

  if (nsghost) grid->reindex_ghost_surfs(nslocal,gmap);
  memory->destroy(gmap);

  scan_copies();
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
      host_geometry(body[k]);
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
      host_geometry(body[k]);
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
     the Surf storage the particle mover and cut pipeline read: every
     local copy on this proc, and with distributed surfs the owned
     copies too, so a later re-map redistributes current coords
------------------------------------------------------------------------- */

void FixRigid::update_surf_copies()
{
  int i,index;

  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;

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

  if (!surf->distributed) return;

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
        fcm_infile[ibody][j] = torque_infile[ibody][j] = 0.0;
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
  remap->grid_changed();

  // distributed surfs: the local surf arrays were rebuilt, so
  //   re-establish this fix's local body-surf copies and the per-surf
  //   rigidmap, which must also span the newly acquired ghost surfs
  // per-surf computes re-size after all fixes are notified

  if (surf->distributed) {
    for (int ibody = 0; ibody < nbody; ibody++) body_bbox(ibody,0);
    proc_bbox();
    int changed = ensure_local_copies();
    surf_maps();

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
  host_geometry(ibody);
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

void FixRigid::sort_for_split_rebuild()
{
  particles_to_host();
  if (!particle->sorted) particle->sort();
}

/* ---------------------------------------------------------------------- */

void FixRigid::combine_split_all()
{
  particles_to_host();
  if (!particle->sorted) particle->sort();

  Grid::ChildCell *cells = grid->cells;
  int nglocal = grid->nlocal;
  for (int icell = 0; icell < nglocal; icell++)
    if (cells[icell].nsplit > 1)
      grid->combine_split_cell_particles(icell,1);
}

/* ---------------------------------------------------------------------- */

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
  int i,j,n,m;
  int unmatched = 0;
  int istart = bodystart[ibody];
  int istop = bodystart[ibody+1];
  int nelem = istop - istart;

  if (dim == 2) {

    // one record per line end point: its coords and +1/-1 for the
    //   1st/2nd end point; sorted, equal points are adjacent and their
    //   counts must sum to zero

    n = 2*nelem;
    EdgeRecord *pts = new EdgeRecord[n];
    m = 0;
    for (i = istart; i < istop; i++) {
      pts[m].x[0] = bodypt[i][0][0]; pts[m].x[1] = bodypt[i][0][1];
      pts[m].x[2] = pts[m].x[3] = pts[m].x[4] = pts[m].x[5] = 0.0;
      pts[m++].count = 1;
      pts[m].x[0] = bodypt[i][1][0]; pts[m].x[1] = bodypt[i][1][1];
      pts[m].x[2] = pts[m].x[3] = pts[m].x[4] = pts[m].x[5] = 0.0;
      pts[m++].count = -1;
    }
    qsort(pts,n,sizeof(EdgeRecord),compare_edge_records);

    i = 0;
    while (i < n) {
      int count = 0;
      j = i;
      while (j < n && compare_edge_records(&pts[i],&pts[j]) == 0)
        count += pts[j++].count;
      if (count && !(axiflag && pts[i].x[1] == 0.0)) unmatched++;
      i = j;
    }
    delete [] pts;

  } else {

    // one record per tri edge: end points in canonical order and +1/-1
    //   for an edge traversed in that order or reversed; sorted, equal
    //   edges are adjacent and their counts must sum to zero

    n = 3*nelem;
    EdgeRecord *edges = new EdgeRecord[n];
    double *pts[4];
    double *a,*b;
    m = 0;

    for (i = istart; i < istop; i++) {
      pts[0] = bodypt[i][0]; pts[1] = bodypt[i][1];
      pts[2] = bodypt[i][2]; pts[3] = bodypt[i][0];

      for (j = 0; j < 3; j++) {
        a = pts[j];
        b = pts[j+1];
        int dir = 1;
        if (b[0] < a[0] ||
            (b[0] == a[0] &&
             (b[1] < a[1] || (b[1] == a[1] && b[2] < a[2])))) {
          double *tmp = a; a = b; b = tmp;
          dir = -1;
        }
        edges[m].x[0] = a[0]; edges[m].x[1] = a[1]; edges[m].x[2] = a[2];
        edges[m].x[3] = b[0]; edges[m].x[4] = b[1]; edges[m].x[5] = b[2];
        edges[m++].count = dir;
      }
    }
    qsort(edges,n,sizeof(EdgeRecord),compare_edge_records);

    i = 0;
    while (i < n) {
      int count = 0;
      j = i;
      while (j < n && compare_edge_records(&edges[i],&edges[j]) == 0)
        count += edges[j++].count;
      if (count) unmatched++;
      i = j;
    }
    delete [] edges;
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
  bytes += (double) nsurf * sizeof(int);                  // lblist
  bytes += (double) nsurf * 2 * sizeof(int);              // olist_own/elem
  bytes += (double) maxcopy * 2 * sizeof(int);            // copy_index/elem
  bytes += (double) nsurf * (sizeof(surfint) + sizeof(int) +
                             3*sizeof(void *));            // idmap, approx
  bytes += (double) maxsurfmap * 2 * sizeof(int);         // surfbody/elem
  bytes += (double) nsurf * sizeof(int);                  // elem2tally
  bytes += (double) maxtally * (sizeof(int) + 6*sizeof(double));  // rows
  if (contact) bytes += contact->memory_usage();
  bytes += remap->memory_usage();

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
