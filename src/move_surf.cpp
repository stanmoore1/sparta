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

#include "stdio.h"
#include "string.h"
#include "move_surf.h"
#include "surf.h"
#include "grid.h"
#include "comm.h"
#include "update.h"
#include "domain.h"
#include "input.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathExtra;
using namespace MathConst;

enum{READFILE,TRANSLATE,ROTATE};
enum{UNKNOWN,OUTSIDE,INSIDE,OVERLAP};           // several files

#define MAXLINE 256

/* ---------------------------------------------------------------------- */

MoveSurf::MoveSurf(SPARTA *sparta) : Pointers(sparta)
{
  me = comm->me;
  nprocs = comm->nprocs;

  // pselect = 1 if point is moved, else 0

  if (domain->dimension == 2)
    memory->create(pselect,2*surf->nsurf,"move_surf:pselect");
  else
    memory->create(pselect,3*surf->nsurf,"move_surf:pselect");

  file = NULL;
  entry = NULL;
  fp = NULL;

  // file style bookkeeping

  readflag = 0;
  nread = 0;
  oldcoord = newcoord = NULL;
  fhash = NULL;
}

/* ---------------------------------------------------------------------- */

MoveSurf::~MoveSurf()
{
  memory->destroy(pselect);
  delete [] file;
  delete [] entry;
  if (fp) fclose(fp);

  memory->destroy(oldcoord);
  memory->destroy(newcoord);
  delete fhash;
}

/* ---------------------------------------------------------------------- */

void MoveSurf::command(int narg, char **arg)
{
  if (!surf->exist)
    error->all(FLERR,"Cannot move_surf with no surf elements defined");

  if (surf->distributed)
    error->all(FLERR,
               "Cannot yet use move_surf with distributed surf elements");

  if (narg < 2) error->all(FLERR,"Illegal move_surf command");

  // process command-line args

  int igroup = surf->find_group(arg[0]);
  if (igroup < 0) error->all(FLERR,"Move_surf group ID does not exist");
  groupbit = surf->bitmask[igroup];

  process_args(narg-1,&arg[1]);
  mode = 0;

  // perform surface move

  if (me == 0) {
    if (screen) fprintf(screen,"Moving surfs ...\n");
    if (logfile) fprintf(logfile,"Moving surfs ...\n");
  }

  MPI_Barrier(world);
  double time1 = MPI_Wtime();

  dim = domain->dimension;

  // sort particles

  if (particle->exist) particle->sort();

  MPI_Barrier(world);
  double time2 = MPI_Wtime();

  // move line/tri points via chosen action by full amount

  if (dim == 2) move_lines(1.0,surf->lines);
  else move_tris(1.0,surf->tris);

  // remake list of surf elements I own
  // assign split cell particles to parent split cell
  // assign surfs to grid cells

  grid->unset_neighbors();
  grid->remove_ghosts();

  if (particle->exist && grid->nsplitlocal) {
    Grid::ChildCell *cells = grid->cells;
    int nglocal = grid->nlocal;
    for (int icell = 0; icell < nglocal; icell++)
      if (cells[icell].nsplit > 1)
        grid->combine_split_cell_particles(icell,1);
  }

  grid->clear_surf();
  grid->surf2grid(1);

  if (dim == 2) surf->check_point_near_surf_2d();
  else surf->check_point_near_surf_3d();

  if (dim == 2) surf->check_watertight_2d();
  else surf->check_watertight_3d();

  MPI_Barrier(world);
  double time3 = MPI_Wtime();

  // re-setup owned and ghost cell info

  grid->setup_owned();
  grid->acquire_ghosts();
  grid->reset_neighbors();
  comm->reset_neighbors();

  MPI_Barrier(world);
  double time4 = MPI_Wtime();

  // flag cells and corners as OUTSIDE or INSIDE
  // reallocate per grid cell arrays in per grid computes
  //   local grid cell counts could have changed due to split cell changes

  grid->set_inout();
  grid->type_check();

  // DEBUG
  //grid->debug();

  MPI_Barrier(world);
  double time5 = MPI_Wtime();

  // remove particles as needed due to surface move

  bigint ndeleted;
  if (particle->exist) ndeleted = remove_particles();

  MPI_Barrier(world);
  double time6 = MPI_Wtime();

  double time_total = time6-time1;

  if (comm->me == 0) {
    if (screen) {
      if (particle->exist)
        fprintf(screen,"  " BIGINT_FORMAT " deleted particles\n",ndeleted);
      fprintf(screen,"  CPU time = %g secs\n",time_total);
      fprintf(screen,"  sort/surf2grid/ghost/inout/particle percent = "
              "%g %g %g %g %g\n",
              100.0*(time2-time1)/time_total,100.0*(time3-time2)/time_total,
              100.0*(time4-time3)/time_total,100.0*(time5-time4)/time_total,
              100.0*(time6-time5)/time_total);
    }
    if (logfile) {
      if (particle->exist)
        fprintf(logfile,"  " BIGINT_FORMAT " deleted particles\n",ndeleted);
      fprintf(logfile,"  CPU time = %g secs\n",time_total);
      fprintf(logfile,"  sort/surf2grid/ghost/inout/particle percent = "
              "%g %g %g %g %g\n",
              100.0*(time2-time1)/time_total,100.0*(time3-time2)/time_total,
              100.0*(time4-time3)/time_total,100.0*(time5-time4)/time_total,
              100.0*(time6-time5)/time_total);
    }
  }
}

/* ----------------------------------------------------------------------
   process command args for both move_surf and fix move/surf
------------------------------------------------------------------------- */

void MoveSurf::process_args(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR,"Illegal move surf command");

  int iarg = 0;
  if (strcmp(arg[0],"file") == 0) {
    if (narg < 3) error->all(FLERR,"Illegal move surf command");
    action = READFILE;
    int n = strlen(arg[1]) + 1;
    file = new char[n];
    strcpy(file,arg[1]);
    n = strlen(arg[2]) + 1;
    entry = new char[n];
    strcpy(entry,arg[2]);
    iarg = 3;
  } else if (strcmp(arg[0],"trans") == 0) {
    if (narg < 4) error->all(FLERR,"Illegal move surf command");
    action = TRANSLATE;
    delta[0] = input->numeric(FLERR,arg[1]);
    delta[1] = input->numeric(FLERR,arg[2]);
    delta[2] = input->numeric(FLERR,arg[3]);
    if (domain->dimension == 2 && delta[2] != 0.0)
      error->all(FLERR,"Invalid move surf translation for 2d simulation");
    iarg = 4;
  } else if (strcmp(arg[0],"rotate") == 0) {
    if (narg < 8) error->all(FLERR,"Illegal move surf command");
    action = ROTATE;
    theta = input->numeric(FLERR,arg[1]);
    rvec[0] = input->numeric(FLERR,arg[2]);
    rvec[1] = input->numeric(FLERR,arg[3]);
    rvec[2] = input->numeric(FLERR,arg[4]);
    origin[0] = input->numeric(FLERR,arg[5]);
    origin[1] = input->numeric(FLERR,arg[6]);
    origin[2] = input->numeric(FLERR,arg[7]);
    if (domain->dimension == 2 && (rvec[0] != 0.0 || rvec[1] != 0.0))
      error->all(FLERR,"Invalid move surf rotation for 2d simulation");
    if (rvec[0] == 0.0 && rvec[1] == 0.0 && rvec[2] == 0.0)
      error->all(FLERR,"Invalid move surf rotation");
    theta *= MY_PI/180.0;
    MathExtra::norm3(rvec);
    iarg = 8;
  } else error->all(FLERR,"Illegal move surf command");

  // optional args

  connectflag = 0;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"connect") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal move surf command");
      if (strcmp(arg[iarg+1],"yes") == 0) connectflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) connectflag = 0;
      iarg += 2;
    } else error->all(FLERR,"Illegal move surf command");
  }
}

/* ----------------------------------------------------------------------
   move points in lines via specified action
   each method sets pselect = 1 for moved points
   fraction = portion of full distance points should move
------------------------------------------------------------------------- */

void MoveSurf::move_lines(double fraction, Surf::Line *origlines)
{
  if (connectflag && groupbit != 1) connect_2d_pre();

  if (action == READFILE) move_file_2d(fraction,origlines);
  else if (action == TRANSLATE) translate_2d(fraction,origlines);
  else if (action == ROTATE) rotate_2d(fraction,origlines);

  if (connectflag && groupbit != 1) connect_2d_post();

  surf->compute_line_normal(0);

  // check that all points are still inside simulation box

  surf->check_point_inside(0);
}

/* ----------------------------------------------------------------------
   move points in triangles via specified action
   each method sets pselect = 1 for moved points
   fraction = portion of full distance points should move
------------------------------------------------------------------------- */

void MoveSurf::move_tris(double fraction, Surf::Tri *origtris)
{
  if (connectflag && groupbit != 1) connect_3d_pre();

  if (action == READFILE) move_file_3d(fraction,origtris);
  else if (action == TRANSLATE) translate_3d(fraction,origtris);
  else if (action == ROTATE) rotate_3d(fraction,origtris);

  if (connectflag && groupbit != 1) connect_3d_post();

  surf->compute_tri_normal(0);

  // check that all points are still inside simulation box

  surf->check_point_inside(0);
}

/* ----------------------------------------------------------------------
   read named entry of old/new point coords from file
   caches results and builds coord hash so file is read only once
   file format, one entry:
     entryID
     N
     xold yold [zold] xnew ynew [znew]   (N such lines; z only in 3d)
   old coords are the matching key; new coords are the target positions
------------------------------------------------------------------------- */

void MoveSurf::readfile()
{
  // only read/cache the file entry once
  // subsequent calls (e.g. from fix move/surf) reuse cached coords + hash

  if (readflag) return;
  readflag = 1;

  char line[MAXLINE];
  char *word,*eof;

  // proc 0 opens file and scans to the requested entry

  if (me == 0) {
    fp = fopen(file,"r");
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open move surf file %s",file);
      error->one(FLERR,str);
    }

    while (1) {
      eof = fgets(line,MAXLINE,fp);
      if (eof == NULL) error->one(FLERR,"Did not find entry in move surf file");
      if (strspn(line," \t\n\r") == strlen(line)) continue;  // blank line
      if (line[0] == '#') continue;                          // comment
      word = strtok(line," \t\n\r");
      if (strcmp(word,entry) != 0) continue;                 // wrong entry
      eof = fgets(line,MAXLINE,fp);                           // count line
      if (eof == NULL) error->one(FLERR,"Incomplete entry in move surf file");
      word = strtok(line," \t\n\r");
      nread = input->inumeric(FLERR,word);
      break;
    }
  }

  // allocate coord arrays for nread points on all procs

  MPI_Bcast(&nread,1,MPI_INT,0,world);
  memory->create(oldcoord,nread,3,"move_surf:oldcoord");
  memory->create(newcoord,nread,3,"move_surf:newcoord");

  // proc 0 reads the nread old/new coord pairs

  int dimension = domain->dimension;

  if (me == 0) {
    for (int i = 0; i < nread; i++) {
      eof = fgets(line,MAXLINE,fp);
      if (eof == NULL) error->one(FLERR,"Incomplete entry in move surf file");
      oldcoord[i][0] = input->numeric(FLERR,strtok(line," \t\n\r"));
      oldcoord[i][1] = input->numeric(FLERR,strtok(NULL," \t\n\r"));
      if (dimension == 3)
        oldcoord[i][2] = input->numeric(FLERR,strtok(NULL," \t\n\r"));
      else oldcoord[i][2] = 0.0;
      newcoord[i][0] = input->numeric(FLERR,strtok(NULL," \t\n\r"));
      newcoord[i][1] = input->numeric(FLERR,strtok(NULL," \t\n\r"));
      if (dimension == 3)
        newcoord[i][2] = input->numeric(FLERR,strtok(NULL," \t\n\r"));
      else newcoord[i][2] = 0.0;
    }
    fclose(fp);
    fp = NULL;
  }

  // broadcast coords to all procs

  MPI_Bcast(&oldcoord[0][0],3*nread,MPI_DOUBLE,0,world);
  MPI_Bcast(&newcoord[0][0],3*nread,MPI_DOUBLE,0,world);

  // build hash from original coord -> index into old/new coord arrays
  // later duplicate coords in the entry simply overwrite earlier ones

  fhash = new MyHash();
  OnePoint3d key;
  for (int i = 0; i < nread; i++) {
    key.pt[0] = oldcoord[i][0];
    key.pt[1] = oldcoord[i][1];
    key.pt[2] = oldcoord[i][2];
    (*fhash)[key] = i;
  }
}

/* ----------------------------------------------------------------------
   move points in lines to file target coords via coordinate match, 2d
   a vertex is moved if its original coord matches a point in the entry
   fraction = portion of old->new distance to move
------------------------------------------------------------------------- */

void MoveSurf::move_file_2d(double fraction, Surf::Line *origlines)
{
  double *p1,*p2,*op1,*op2;
  OnePoint3d key;
  MyHash::iterator it;

  readfile();

  Surf::Line *lines = surf->lines;
  int nsurf = surf->nsurf;

  for (int i = 0; i < 2*nsurf; i++) pselect[i] = 0;

  for (int i = 0; i < nsurf; i++) {
    if (!(lines[i].mask & groupbit)) continue;
    p1 = lines[i].p1;
    p2 = lines[i].p2;
    op1 = origlines[i].p1;
    op2 = origlines[i].p2;

    key.pt[0] = op1[0]; key.pt[1] = op1[1]; key.pt[2] = 0.0;
    it = fhash->find(key);
    if (it != fhash->end()) {
      int m = it->second;
      p1[0] = op1[0] + fraction * (newcoord[m][0]-op1[0]);
      p1[1] = op1[1] + fraction * (newcoord[m][1]-op1[1]);
      pselect[2*i] = 1;
    }

    key.pt[0] = op2[0]; key.pt[1] = op2[1]; key.pt[2] = 0.0;
    it = fhash->find(key);
    if (it != fhash->end()) {
      int m = it->second;
      p2[0] = op2[0] + fraction * (newcoord[m][0]-op2[0]);
      p2[1] = op2[1] + fraction * (newcoord[m][1]-op2[1]);
      pselect[2*i+1] = 1;
    }
  }
}

/* ----------------------------------------------------------------------
   move points in triangles to file target coords via coordinate match, 3d
   a vertex is moved if its original coord matches a point in the entry
   fraction = portion of old->new distance to move
------------------------------------------------------------------------- */

void MoveSurf::move_file_3d(double fraction, Surf::Tri *origtris)
{
  double *p1,*p2,*p3,*op1,*op2,*op3;
  OnePoint3d key;
  MyHash::iterator it;

  readfile();

  Surf::Tri *tris = surf->tris;
  int nsurf = surf->nsurf;

  for (int i = 0; i < 3*nsurf; i++) pselect[i] = 0;

  for (int i = 0; i < nsurf; i++) {
    if (!(tris[i].mask & groupbit)) continue;
    p1 = tris[i].p1;
    p2 = tris[i].p2;
    p3 = tris[i].p3;
    op1 = origtris[i].p1;
    op2 = origtris[i].p2;
    op3 = origtris[i].p3;

    key.pt[0] = op1[0]; key.pt[1] = op1[1]; key.pt[2] = op1[2];
    it = fhash->find(key);
    if (it != fhash->end()) {
      int m = it->second;
      p1[0] = op1[0] + fraction * (newcoord[m][0]-op1[0]);
      p1[1] = op1[1] + fraction * (newcoord[m][1]-op1[1]);
      p1[2] = op1[2] + fraction * (newcoord[m][2]-op1[2]);
      pselect[3*i] = 1;
    }

    key.pt[0] = op2[0]; key.pt[1] = op2[1]; key.pt[2] = op2[2];
    it = fhash->find(key);
    if (it != fhash->end()) {
      int m = it->second;
      p2[0] = op2[0] + fraction * (newcoord[m][0]-op2[0]);
      p2[1] = op2[1] + fraction * (newcoord[m][1]-op2[1]);
      p2[2] = op2[2] + fraction * (newcoord[m][2]-op2[2]);
      pselect[3*i+1] = 1;
    }

    key.pt[0] = op3[0]; key.pt[1] = op3[1]; key.pt[2] = op3[2];
    it = fhash->find(key);
    if (it != fhash->end()) {
      int m = it->second;
      p3[0] = op3[0] + fraction * (newcoord[m][0]-op3[0]);
      p3[1] = op3[1] + fraction * (newcoord[m][1]-op3[1]);
      p3[2] = op3[2] + fraction * (newcoord[m][2]-op3[2]);
      pselect[3*i+2] = 1;
    }
  }
}

/* ----------------------------------------------------------------------
   translate surf points in 2d
------------------------------------------------------------------------- */

void MoveSurf::translate_2d(double fraction, Surf::Line *origlines)
{
  double *p1,*p2,*op1,*op2;

  Surf::Line *lines = surf->lines;
  int nsurf = surf->nsurf;

  for (int i = 0; i < 2*nsurf; i++) pselect[i] = 0;

  double dx = fraction * delta[0];
  double dy = fraction * delta[1];

  for (int i = 0; i < nsurf; i++) {
    if (!(lines[i].mask & groupbit)) continue;
    p1 = lines[i].p1;
    p2 = lines[i].p2;
    op1 = origlines[i].p1;
    op2 = origlines[i].p2;

    p1[0] = op1[0] + dx;
    p1[1] = op1[1] + dy;
    pselect[2*i] = 1;

    p2[0] = op2[0] + dx;
    p2[1] = op2[1] + dy;
    pselect[2*i+1] = 1;
  }
}

/* ----------------------------------------------------------------------
   translate surf points in 3d
------------------------------------------------------------------------- */

void MoveSurf::translate_3d(double fraction, Surf::Tri *origtris)
{
  double *p1,*p2,*p3,*op1,*op2,*op3;

  Surf::Tri *tris = surf->tris;
  int nsurf = surf->nsurf;

  for (int i = 0; i < 3*nsurf; i++) pselect[i] = 0;

  double dx = fraction * delta[0];
  double dy = fraction * delta[1];
  double dz = fraction * delta[2];

  for (int i = 0; i < nsurf; i++) {
    if (!(tris[i].mask & groupbit)) continue;
    p1 = tris[i].p1;
    p2 = tris[i].p2;
    p3 = tris[i].p3;
    op1 = origtris[i].p1;
    op2 = origtris[i].p2;
    op3 = origtris[i].p3;

    p1[0] = op1[0] + dx;
    p1[1] = op1[1] + dy;
    p1[2] = op1[2] + dz;
    pselect[3*i] = 1;

    p2[0] = op2[0] + dx;
    p2[1] = op2[1] + dy;
    p2[2] = op2[2] + dz;
    pselect[3*i+1] = 1;

    p3[0] = op3[0] + dx;
    p3[1] = op3[1] + dy;
    p3[2] = op3[2] + dz;
    pselect[3*i+2] = 1;
  }
}

/* ----------------------------------------------------------------------
   rotate surf points in 2d
------------------------------------------------------------------------- */

void MoveSurf::rotate_2d(double fraction, Surf::Line *origlines)
{
  double *p1,*p2,*op1,*op2;
  double q[4],d[3],dnew[3];
  double rotmat[3][3];

  Surf::Line *lines = surf->lines;
  int nsurf = surf->nsurf;

  for (int i = 0; i < 2*nsurf; i++) pselect[i] = 0;

  double angle = fraction * theta;
  MathExtra::axisangle_to_quat(rvec,angle,q);
  MathExtra::quat_to_mat(q,rotmat);

  for (int i = 0; i < nsurf; i++) {
    if (!(lines[i].mask & groupbit)) continue;
    p1 = lines[i].p1;
    p2 = lines[i].p2;
    op1 = origlines[i].p1;
    op2 = origlines[i].p2;

    d[0] = op1[0] - origin[0];
    d[1] = op1[1] - origin[1];
    d[2] = op1[2] - origin[2];
    MathExtra::matvec(rotmat,d,dnew);
    p1[0] = dnew[0] + origin[0];
    p1[1] = dnew[1] + origin[1];
    pselect[2*i] = 1;

    d[0] = op2[0] - origin[0];
    d[1] = op2[1] - origin[1];
    d[2] = op2[2] - origin[2];
    MathExtra::matvec(rotmat,d,dnew);
    p2[0] = dnew[0] + origin[0];
    p2[1] = dnew[1] + origin[1];
    pselect[2*i+1] = 1;
  }
}

/* ----------------------------------------------------------------------
   rotate surf points in 3d
------------------------------------------------------------------------- */

void MoveSurf::rotate_3d(double fraction, Surf::Tri *origtris)
{
  double *p1,*p2,*p3,*op1,*op2,*op3;
  double q[4],d[3],dnew[3];
  double rotmat[3][3];

  Surf::Tri *tris = surf->tris;
  int nsurf = surf->nsurf;

  for (int i = 0; i < 3*nsurf; i++) pselect[i] = 0;

  double angle = fraction * theta;
  MathExtra::axisangle_to_quat(rvec,angle,q);
  MathExtra::quat_to_mat(q,rotmat);

  for (int i = 0; i < nsurf; i++) {
    if (!(tris[i].mask & groupbit)) continue;
    p1 = tris[i].p1;
    p2 = tris[i].p2;
    p3 = tris[i].p3;
    op1 = origtris[i].p1;
    op2 = origtris[i].p2;
    op3 = origtris[i].p3;

    d[0] = op1[0] - origin[0];
    d[1] = op1[1] - origin[1];
    d[2] = op1[2] - origin[2];
    MathExtra::matvec(rotmat,d,dnew);
    p1[0] = dnew[0] + origin[0];
    p1[1] = dnew[1] + origin[1];
    p1[2] = dnew[2] + origin[2];
    pselect[3*i] = 1;

    d[0] = op2[0] - origin[0];
    d[1] = op2[1] - origin[1];
    d[2] = op2[2] - origin[2];
    MathExtra::matvec(rotmat,d,dnew);
    p2[0] = dnew[0] + origin[0];
    p2[1] = dnew[1] + origin[1];
    p2[2] = dnew[2] + origin[2];
    pselect[3*i+1] = 1;

    d[0] = op3[0] - origin[0];
    d[1] = op3[1] - origin[1];
    d[2] = op3[2] - origin[2];
    MathExtra::matvec(rotmat,d,dnew);
    p3[0] = dnew[0] + origin[0];
    p3[1] = dnew[1] + origin[1];
    p3[2] = dnew[2] + origin[2];
    pselect[3*i+2] = 1;
  }
}

/* ----------------------------------------------------------------------
   add points in moved lines to hash
------------------------------------------------------------------------- */

void MoveSurf::connect_2d_pre()
{
  // hash for end points of moved lines
  // key = end point
  // value = global index (0 to 2*Nline-1) of the point
  // NOTE: could prealloc hash to correct size here

  hash = new MyHash();

  // add moved points to hash

  double *p1,*p2;
  OnePoint3d key;

  Surf::Line *lines = surf->lines;
  int nsurf = surf->nsurf;

  for (int i = 0; i < nsurf; i++) {
    if (!(lines[i].mask & groupbit)) continue;
    p1 = lines[i].p1;
    p2 = lines[i].p2;
    key.pt[0] = p1[0]; key.pt[1] = p1[1]; key.pt[2] = 0.0;
    if (hash->find(key) == hash->end()) (*hash)[key] = 2*i+0;
    key.pt[0] = p2[0]; key.pt[1] = p2[1]; key.pt[2] = 0.0;
    if (hash->find(key) == hash->end()) (*hash)[key] = 2*i+1;
  }
}

/* ----------------------------------------------------------------------
   move points in lines connected to line points that were moved
------------------------------------------------------------------------- */

void MoveSurf::connect_2d_post()
{
  // check if non-moved points are in hash
  // if so, set their coords to matching point
  // set pselect for newly moved points so remove_particles() will work

  int m,value,j,jwhich;
  double *p[2],*q;
  OnePoint3d key;

  Surf::Line *lines = surf->lines;
  int nsurf = surf->nsurf;

  for (int i = 0; i < nsurf; i++) {
    if (lines[i].mask & groupbit) continue;
    p[0] = lines[i].p1;
    p[1] = lines[i].p2;

    for (m = 0; m < 2; m++) {
      key.pt[0] = p[m][0]; key.pt[1] = p[m][1]; key.pt[2] = 0.0;
      if (hash->find(key) != hash->end()) {
        value = (*hash)[key];
        j = value/2;
        jwhich = value % 2;
        if (jwhich == 0) q = lines[j].p1;
        else q = lines[j].p2;
        p[m][0] = q[0];
        p[m][1] = q[1];
        if (m == 0) pselect[2*i] = 1;
        else pselect[2*i+1] = 1;
      }
    }
  }

  // free the hash

  delete hash;
}

/* ----------------------------------------------------------------------
   add points in moved triangles to hash
------------------------------------------------------------------------- */

void MoveSurf::connect_3d_pre()
{
  // hash for corner points of moved triangles
  // key = corner point
  // value = global index (0 to 3*Ntri-1) of the point
  // NOTE: could prealloc hash to correct size here

  hash = new MyHash();

  // add moved points to hash

  double *p1,*p2,*p3;
  OnePoint3d key;

  Surf::Tri *tris = surf->tris;
  int nsurf = surf->nsurf;

  for (int i = 0; i < nsurf; i++) {
    if (!(tris[i].mask & groupbit)) continue;
    p1 = tris[i].p1;
    p2 = tris[i].p2;
    p3 = tris[i].p3;
    key.pt[0] = p1[0]; key.pt[1] = p1[1]; key.pt[2] = p1[2];
    if (hash->find(key) == hash->end()) (*hash)[key] = 3*i+0;
    key.pt[0] = p2[0]; key.pt[1] = p2[1]; key.pt[2] = p2[2];
    if (hash->find(key) == hash->end()) (*hash)[key] = 3*i+1;
    key.pt[0] = p3[0]; key.pt[1] = p3[1]; key.pt[2] = p3[2];
    if (hash->find(key) == hash->end()) (*hash)[key] = 3*i+2;
  }
}

/* ----------------------------------------------------------------------
   move points in tris connected to tri points that were moved
------------------------------------------------------------------------- */

void MoveSurf::connect_3d_post()
{
  // check if non-moved points are in hash
  // if so, set their coords to matching point
  // set pselect for newly moved points so remove_particles() will work

  int m,value,j,jwhich;
  double *p[3],*q;
  OnePoint3d key;

  Surf::Tri *tris = surf->tris;
  int nsurf = surf->nsurf;

  for (int i = 0; i < nsurf; i++) {
    if (tris[i].mask & groupbit) continue;
    p[0] = tris[i].p1;
    p[1] = tris[i].p2;
    p[2] = tris[i].p3;

    for (m = 0; m < 3; m++) {
      key.pt[0] = p[m][0]; key.pt[1] = p[m][1]; key.pt[2] = p[m][2];
      if (hash->find(key) != hash->end()) {
        value = (*hash)[key];
        j = value/3;
        jwhich = value % 3;
        if (jwhich == 0) q = tris[j].p1;
        else if (jwhich == 1) q = tris[j].p2;
        else q = tris[j].p3;
        p[m][0] = q[0];
        p[m][1] = q[1];
        p[m][2] = q[2];
        if (m == 0) pselect[3*i] = 1;
        else if (m == 1) pselect[3*i+1] = 1;
        else pselect[3*i+2] = 1;
      }
    }
  }

  // free the hash

  delete hash;
}

/* ----------------------------------------------------------------------
   remove particles in any cell that is now INSIDE or contains moved surfs
   surfs that moved determined by pselect for any of its points
   reassign particles in split cells to sub cell owner
   compress particles if any flagged for deletion
   NOTE: doc this logic better
------------------------------------------------------------------------- */

bigint MoveSurf::remove_particles()
{
  int isurf,nsurf;
  surfint *csurfs;

  dim = domain->dimension;
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  int nglocal = grid->nlocal;
  int delflag = 0;

  for (int icell = 0; icell < nglocal; icell++) {

    // cell is inside surfs
    // remove particles in case it wasn't before

    if (cinfo[icell].type == INSIDE) {
      if (cinfo[icell].count) delflag = 1;
      particle->remove_all_from_cell(cinfo[icell].first);
      cinfo[icell].count = 0;
      cinfo[icell].first = -1;
      continue;
    }

    // cell has surfs or is split
    // if m < nsurf, loop over csurfs did not finish
    // which means cell contains a moved surf, so delete all its particles

    if (cells[icell].nsurf && cells[icell].nsplit >= 1) {
      nsurf = cells[icell].nsurf;
      csurfs = cells[icell].csurfs;

      int m;
      if (dim == 2) {
        for (m = 0; m < nsurf; m++) {
          isurf = csurfs[m];
          if (pselect[2*isurf]) break;
          if (pselect[2*isurf+1]) break;
        }
      } else {
        for (m = 0; m < nsurf; m++) {
          isurf = csurfs[m];
          if (pselect[3*isurf]) break;
          if (pselect[3*isurf+1]) break;
          if (pselect[3*isurf+2]) break;
        }
      }

      if (m < nsurf) {
        if (cinfo[icell].count) delflag = 1;
        particle->remove_all_from_cell(cinfo[icell].first);
        cinfo[icell].count = 0;
        cinfo[icell].first = -1;
      }
    }

    if (cells[icell].nsplit > 1)
      grid->assign_split_cell_particles(icell);
  }

  int nlocal_old = particle->nlocal;
  if (delflag) particle->compress_rebalance();
  bigint delta = nlocal_old - particle->nlocal;
  bigint ndeleted;
  MPI_Allreduce(&delta,&ndeleted,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
  return ndeleted;
}
