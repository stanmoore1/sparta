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

#include "string.h"
#include <algorithm>
#include "grid.h"
#include "domain.h"
#include "update.h"
#include "comm.h"
#include "particle.h"
#include "modify.h"
#include "collide.h"
#include "surf.h"
#include "cut2d.h"
#include "cut3d.h"
#include "irregular.h"
#include "geometry.h"
#include "math_const.h"
#include "hashlittle.h"
#include "my_page.h"
#include "math_extra.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

// prototype for non-class function

int compare_surfIDs(const void *, const void *);

#define BIG 1.0e20
#define CHUNK 16
#define EPSSURF 1.0e-4
#define DELTA_SEND 16384
#define DELTA_MOVED 128
#define DELTA_JOURNAL 4096

enum{UNKNOWN,OUTSIDE,INSIDE,OVERLAP};         // several files
enum{NCHILD,NPARENT,NUNKNOWN,NPBCHILD,NPBPARENT,NPBUNKNOWN,NBOUND};  // Grid
enum{PERAUTO,PERCELL,PERSURF};                // several files
enum{SOUTSIDE,SINSIDE,ONSURF2OUT,ONSURF2IN};  // several files (changed 2 words)

// operations for surfaces in grid cells

/* ----------------------------------------------------------------------
   map surf elements to grid cells for explicit surfs (distributed or not)
   via one of two algorithms
   cell_alg = original, loop over my cells, check all surfs within bbox
   surf_alg = Jan19, loop over N/P surfs, find small set of cells each overlaps,
              perform rendezvous comm to convert cells per surf to surfs per cell
   new_alg = Nov20, no more parent cells, rendezvous alg at each level of grid
   new2_alg = Mar21, modified new alg, use recursive tree to speed overlap finding
     this one is now surf_alg
   for distributed surfs, have to use surf alg
   PERAUTO option chooses based on total nsurfs vs nprocs
   see info on subflag, outflag options with surf2grid_split()
   called from ReadSurf, MoveSurf, RemoveSurf, ReadRestart, and FixMoveSurf
------------------------------------------------------------------------- */

void Grid::surf2grid(int subflag, int outflag)
{
  if (surf->distributed) {
    surf2grid_surf_algorithm(outflag);
  } else if (surfgrid_algorithm == PERAUTO) {
    if (comm->nprocs > surf->nsurf) surf2grid_cell_algorithm(outflag);
    else {
      surf2grid_surf_algorithm(outflag);
    }
  } else if (surfgrid_algorithm == PERCELL) {
    surf2grid_cell_algorithm(outflag);
  } else if (surfgrid_algorithm == PERSURF) {
    surf2grid_surf_algorithm(outflag);
  }

  // now have nsurf,csurfs list of local surfs that overlap each cell
  // compute cut volume and split info for each cell

  surf2grid_split(subflag,outflag);
}

/* ----------------------------------------------------------------------
   compute split cells for implicit surfs
   surfs per cell already created
   called from ReadISurf and FixAblate
------------------------------------------------------------------------- */

void Grid::surf2grid_implicit(int subflag, int outflag)
{
  int dim = domain->dimension;
  if (dim == 3 && !cut3d) cut3d = new Cut3d(sparta);
  else if (dim == 2 && !cut2d) cut2d = new Cut2d(sparta,domain->axisymmetric);

  tmap = tcomm1 = tcomm2 = tcomm3 = 0.0;

  if (outflag) surf2grid_stats();
  surf2grid_split(subflag,outflag);
}

/* ----------------------------------------------------------------------
   map surf elements into a single grid cell = icell
   flag = 0 for grid refinement, 1 for grid coarsening
   in cells: set nsurf, csurfs, nsplit, isplit
   in cinfo: set type, corner, volume
   initialize sinfo as needed
   called from AdaptGrid
------------------------------------------------------------------------- */

void Grid::surf2grid_one(int flag, int icell, int iparent, int nsurf_caller,
                         Cut3d *cut3d, Cut2d *cut2d)
{
  int nsurf,isub,xsub,nsplitone;
  int *iptr;
  surfint *sptr;
  double xsplit[3];
  double *vols;

  int dim = domain->dimension;

  // identify surfs in new cell only for grid refinement

  if (flag == 0) {
    sptr = csurfs->vget();
    if (dim == 3)
      nsurf = cut3d->surf2grid_list(cells[icell].id,
                                    cells[icell].lo,cells[icell].hi,
                                    cells[iparent].nsurf,cells[iparent].csurfs,
                                    sptr,maxsurfpercell);
    else
      nsurf = cut2d->surf2grid_list(cells[icell].id,
                                    cells[icell].lo,cells[icell].hi,
                                    cells[iparent].nsurf,cells[iparent].csurfs,
                                    sptr,maxsurfpercell);

    if (nsurf == 0) return;
    if (nsurf > maxsurfpercell) {
      printf("Surfs in one refined cell = %d\n",nsurf);
      error->one(FLERR,"Too many surfs in one refined cell - set global surfmax");
    }

    cinfo[icell].type = OVERLAP;
    cells[icell].nsurf = nsurf;
    cells[icell].csurfs = sptr;
    csurfs->vgot(nsurf);

  } else nsurf = nsurf_caller;

  // split check done for both refinement and coarsening

  int *surfmap = csplits->vget();
  ChildCell *c = &cells[icell];

  if (dim == 3)
    nsplitone = cut3d->split(c->id,c->lo,c->hi,c->nsurf,c->csurfs,
                             vols,surfmap,cinfo[icell].corner,
                             xsub,xsplit);
  else
    nsplitone = cut2d->split(c->id,c->lo,c->hi,c->nsurf,c->csurfs,
                             vols,surfmap,cinfo[icell].corner,
                             xsub,xsplit);

  if (nsplitone == 1) {
    if (cinfo[icell].corner[0] != UNKNOWN)
      cinfo[icell].volume = vols[0];

  } else {
    c->nsplit = nsplitone;
    nunsplitlocal--;

    c->isplit = grid->nsplitlocal;
    add_split_cell(1);
    SplitInfo *s = &sinfo[nsplitlocal-1];
    s->icell = icell;
    s->csplits = surfmap;
    s->xsub = xsub;
    s->xsplit[0] = xsplit[0];
    s->xsplit[1] = xsplit[1];
    if (dim == 3) s->xsplit[2] = xsplit[2];
    else s->xsplit[2] = 0.0;

    iptr = s->csubs = csubs->vget();

    // add nsplitone sub cells
    // collide and fixes also need to add cells

    for (int i = 0; i < nsplitone; i++) {
      isub = nlocal;
      add_sub_cell(icell,1);
      if (collide) collide->add_grid_one();
      if (modify->n_pergrid) modify->add_grid_one();
      cells[isub].nsplit = -i;
      cinfo[isub].volume = vols[i];
      iptr[i] = isub;
    }

    csplits->vgot(nsurf);
    csubs->vgot(nsplitone);
  }
}

/* ----------------------------------------------------------------------
   find surfs that overlap owned grid cells, only for non-distributed surfs
   algorithm: for each of my cells, check all surfs
   in cells: set nsurf, csurfs
   in cinfo: set type=OVERLAP for cells with surfs
------------------------------------------------------------------------- */

void Grid::surf2grid_cell_algorithm(int outflag)
{
  int i,nsurf,nontrans;
  double t1,t2;
  surfint *ptr;
  double *lo,*hi;

  int dim = domain->dimension;
  if (dim == 3 && !cut3d) cut3d = new Cut3d(sparta);
  else if (dim == 2 && !cut2d) cut2d = new Cut2d(sparta,domain->axisymmetric);

  if (outflag) {
    MPI_Barrier(world);
    t1 = MPI_Wtime();
  }

  surf->bbox_all();

  double *slo = surf->bblo;
  double *shi = surf->bbhi;

  // compute overlap of all surfs with each cell I own
  // info stored in nsurf,csurfs
  // skip if nsplit <= 0 b/c split cells could exist if restarting

  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;

  int max = 0;

  for (int icell = 0; icell < nlocal; icell++) {
    if (cells[icell].nsplit <= 0) continue;

    // skip grid cell if outside bounding box of all surfs

    lo = cells[icell].lo;
    hi = cells[icell].hi;
    if (!box_overlap(lo,hi,slo,shi)) continue;

    // cut2d/3d surf2grid finds intersection of all surfs with a single grid cell

    ptr = csurfs->vget();

    if (dim == 3)
      nsurf = cut3d->surf2grid(cells[icell].id,cells[icell].lo,cells[icell].hi,
                               ptr,maxsurfpercell);
    else
      nsurf = cut2d->surf2grid(cells[icell].id,cells[icell].lo,cells[icell].hi,
                               ptr,maxsurfpercell);

    if (nsurf > maxsurfpercell) {
      max = MAX(max,nsurf);
      csurfs->vgot(0);
    } else if (nsurf) {
      csurfs->vgot(nsurf);
      cells[icell].nsurf = nsurf;
      cells[icell].csurfs = ptr;

      // only mark cell as OVERLAP if has a non-transparent surf element

      nontrans = 0;

      if (dim == 2) {

        for (i = 0; i < nsurf; i++) {
          if (!lines[ptr[i]].transparent) {
            nontrans = 1;
            break;
          }
        }
      } else {
        for (i = 0; i < nsurf; i++) {
          if (!tris[ptr[i]].transparent) {
            nontrans = 1;
            break;
          }
        }
      }

      if (nontrans) cinfo[icell].type = OVERLAP;
    }
  }

  // error if surf count exceeds maxsurfpercell in any cell

  int maxall;
  MPI_Allreduce(&max,&maxall,1,MPI_INT,MPI_MAX,world);
  if (maxall) {
    if (me == 0) printf("Max surfs in any cell = %d\n",maxall);
    error->all(FLERR,"Too many surfs in one cell - set global surfmax");
  }

  // timing info

  if (outflag) {
    MPI_Barrier(world);
    t2 = MPI_Wtime();
    tmap = t2-t1;
    tcomm1 = tcomm2 = tcomm3 = tcomm4 = 0.0;
  }

  if (outflag) surf2grid_stats();
}

/* ----------------------------------------------------------------------
   find surfs that overlap owned grid cells
     for non-distributed or distributed explicit surfs
   algorithm:
     each proc responsible for subset of surfs
     loop over levels of hierarchical grid
     conceptual uniform grid at that level overlayed on bounding box for all surfs
     partition uniform grid via RCB into sub-grids per proc
     irregular comm of 2 sets of data to the RCB procs
       (a) surfs I own with a bbox that overlaps any procs RCB sub-grid
       (b) child cells I own at level to owning proc in RCB decomp
     each proc can then identify surf/grid intersections for its RCB grid cells
       done recursively by dropping each surf down conceptual tree of parent cells
       until reach child cells that exist at level
     irregular comm of surf/grid intersection pairs back to procs that own grid cells
   in cells: set nsurf, csurfs
   in cinfo: set type=OVERLAP for cells with surfs
------------------------------------------------------------------------- */

void Grid::surf2grid_surf_algorithm(int outflag)
{
  int i,n,icell,isurf;
  cellint childID,parentID;
  double t1,t2,t3,t4,t5;
  Irregular *irregular;

  double *boxlo,*boxhi;             // corner points of entire simulation box
  double *allsurflo,*allsurfhi;     // bounding box for all surfs in system
  int unilo[3],unihi[3];            // indices of corner cells of uniform grid
                                    //   at level which encompasses allsurf_lo/hi
  int myunilo[3],myunihi[3];        // my RCB portion of unilo/hi, inclusive
  double slo[3],shi[3];             // bounding box around one surf
  int sunilo[3],sunihi[3];          // indices of corner cells of uniform grid
                                    //   at level which encompasses a single surf
  int glo[3],ghi[3];                // corner indices of a grid box
  double bblo[3],bbhi[3];           // corners of a bounding box
  double rcblo[3],rcbhi[3];         // corners of my RCB box
  GridTree *gtree;                  // tree of RCB cuts for partitioning uniform grid

  // data structs for communication

  struct Send2 {
    cellint childID;
    int proc,icell;
  };

  struct Send3 {
    surfint surfID;
    int icell;
  };

  struct RCBlohi {
    double lo[3],hi[3];             // corner pts for RCB child cells
  };

  int me = comm->me;
  int nprocs = comm->nprocs;
  int distributed = surf->distributed;

  int dim = domain->dimension;
  if (dim == 3 && !cut3d) cut3d = new Cut3d(sparta);
  else if (dim == 2 && !cut2d) cut2d = new Cut2d(sparta,domain->axisymmetric);

  boxlo = domain->boxlo;
  boxhi = domain->boxhi;

  surf->bbox_all();                 // bounding box for all surfs
  allsurflo = surf->bblo;
  allsurfhi = surf->bbhi;

  int *plist;
  memory->create(plist,nprocs,"surf2grid:plist");
  gtree = (GridTree *) memory->smalloc(nprocs*sizeof(GridTree),"surf2grid:gtree");

  // data structs for 3 rendezvous comms

  int *proclist1 = NULL;
  int *proclist2 = NULL;
  int *proclist3 = NULL;
  char *sbuf1 = NULL;
  Send2 *sbuf2 = NULL;
  Send3 *sbuf3 = NULL;
  int maxsend1 = 0;
  int maxsend2 = 0;
  int maxsend3 = 0;
  int **pairs = NULL;
  int maxpair = 0;

  // which set of Lines or Tris to process, distributed or not

  Surf::Line *lines;
  Surf::Tri *tris;
  int nsurf,istart,istop,idelta,nbytes_surf;

  if (distributed) {
    lines = surf->mylines;
    tris = surf->mytris;
    nsurf = surf->nown;
    istart = 0;
    istop = nsurf;
    idelta = 1;
  } else {
    lines = surf->lines;
    tris = surf->tris;
    int ntotal = surf->nsurf;
    nsurf = ntotal / nprocs;
    if (me < ntotal % nprocs) nsurf++;
    istart = comm->me;
    istop = ntotal;
    idelta = nprocs;
  }

  if (dim == 2) nbytes_surf = sizeof(Surf::Line);
  else nbytes_surf = sizeof(Surf::Tri);

  // loop over levels of grid
  // at each iteration, operate only on child cells that exist at that level

  tmap = tcomm1 = tcomm2 = tcomm3 = 0.0;

  int minlevel = set_minlevel();

  for (int level = minlevel; level <= maxlevel; level++) {

    if (outflag) {
      MPI_Barrier(world);
      t1 = MPI_Wtime();
    }

    // compute extent of uniform grid at level which overlaps surf bbox
    // unilo/hi = inclusive range of grid box of overlapping grid box

    id_find_child_uniform_level(level,0,boxlo,boxhi,allsurflo,
                                unilo[0],unilo[1],unilo[2]);
    id_find_child_uniform_level(level,1,boxlo,boxhi,allsurfhi,
                                unihi[0],unihi[1],unihi[2]);

    // compute a recursive decomp (RCB) of the uniform grid box
    // gtree = tree of RCB cuts, cuts are along grid planes
    // myunilo/hi = inclusive range of my portion of grid box
    // rcblo/hi = corner points of my RCB box

    partition_grid(0,nprocs-1,unilo[0],unihi[0],unilo[1],unihi[1],
                   unilo[2],unihi[2],gtree);
    myunilo[0] = unilo[0]; myunihi[0] = unihi[0];
    myunilo[1] = unilo[1]; myunihi[1] = unihi[1];
    myunilo[2] = unilo[2]; myunihi[2] = unihi[2];
    mybox(me,0,nprocs-1,myunilo[0],myunihi[0],myunilo[1],
          myunihi[1],myunilo[2],myunihi[2],gtree);

    childID = id_uniform_level(level,myunilo[0],myunilo[1],myunilo[2]);
    id_lohi(childID,level,boxlo,boxhi,rcblo,bbhi);
    childID = id_uniform_level(level,myunihi[0],myunihi[1],myunihi[2]);
    id_lohi(childID,level,boxlo,boxhi,bblo,rcbhi);

    // first irregular comm
    // loop over my surfs:
    //   compute single surf bbox as a brick of uniform grid cells at this level
    //   drop bbox down RCB tree to identify set of RCB procs the surf overlaps
    // send copy of surf geometry to RCB procs
    // nrecv1 = # of surfs I have copy of in RCB decomp
    // NOTE: this comm might be faster in Rvous mode?

    int nsend = 0;

    for (isurf = istart; isurf < istop; isurf += idelta) {
      if (dim == 2) surf->bbox_one(&lines[isurf],slo,shi);
      else surf->bbox_one(&tris[isurf],slo,shi);
      id_find_child_uniform_level(level,0,boxlo,boxhi,slo,
                                  sunilo[0],sunilo[1],sunilo[2]);
      id_find_child_uniform_level(level,1,boxlo,boxhi,shi,
                                  sunihi[0],sunihi[1],sunihi[2]);

      // drop trimmed surf box on RCB tree
      // return list of procs whose RCB subbox it overlaps

      int np = 0;
      box_drop(sunilo,sunihi,0,nprocs-1,gtree,np,plist);
      if (!np) continue;

      for (i = 0; i < np; i++) {
        if (nsend == maxsend1) {
          maxsend1 += DELTA_SEND;
          memory->grow(proclist1,maxsend1,"surf2grid:proclist1");
          if (dim == 2)
            sbuf1 = (char *) memory->srealloc(sbuf1,maxsend1*sizeof(Surf::Line),
                                              "surf2grid:sbuf1");
          else
            sbuf1 = (char *) memory->srealloc(sbuf1,maxsend1*sizeof(Surf::Tri),
                                              "surf2grid:sbuf1");
        }
        proclist1[nsend] = plist[i];
        if (dim == 2)
          memcpy(&sbuf1[(bigint) nsend*nbytes_surf],&lines[isurf],nbytes_surf);
        else memcpy(&sbuf1[(bigint) nsend*nbytes_surf],&tris[isurf],nbytes_surf);
        nsend++;
      }
    }

    irregular = new Irregular(sparta);
    int nrecv1 = irregular->create_data_uniform(nsend,proclist1,1);
    char *rbuf1 = (char *) memory->smalloc((bigint)nrecv1*nbytes_surf,"surf2grid:rbuf");
    irregular->exchange_uniform(sbuf1,nbytes_surf,rbuf1);
    delete irregular;

    if (outflag) {
      MPI_Barrier(world);
      t2 = MPI_Wtime();
      tcomm1 += t2-t1;
    }

    // second irregular comm
    // identify which RCB proc owns each of my child cells at this level
    // send childID and my proc ID to RCB procs
    // nrecv2 = # of child cells I have copy of in RCB decomp

    int cx,cy,cz;
    double ctr[3];

    nsend = 0;

    for (icell = 0; icell < nlocal; icell++) {
      if (cells[icell].level != level) continue;
      if (cells[icell].nsplit <= 0) continue;

      ctr[0] = 0.5 * (cells[icell].lo[0] + cells[icell].hi[0]);
      ctr[1] = 0.5 * (cells[icell].lo[1] + cells[icell].hi[1]);
      ctr[2] = 0.5 * (cells[icell].lo[2] + cells[icell].hi[2]);
      id_find_child_uniform_level(level,0,boxlo,boxhi,ctr,cx,cy,cz);

      // glo/hi = single cell grid box

      glo[0] = cx; ghi[0] = cx;
      glo[1] = cy; ghi[1] = cy;
      glo[2] = cz; ghi[2] = cz;

      int np = 0;
      box_drop(glo,ghi,0,nprocs-1,gtree,np,plist);
      if (np != 1) error->one(FLERR,"Box drop of grid cell failed");

      if (nsend == maxsend2) {
        maxsend2 += DELTA_SEND;
        memory->grow(proclist2,maxsend2,"surf2grid:proclist2");
        sbuf2 = (Send2 *) memory->srealloc(sbuf2,maxsend2*sizeof(Send2),
                                          "surf2grid:sbuf2");
      }

      proclist2[nsend] = plist[0];
      sbuf2[nsend].childID = cells[icell].id;
      sbuf2[nsend].proc = me;
      sbuf2[nsend].icell = icell;
      nsend++;
    }

    irregular = new Irregular(sparta);
    int nrecv2 = irregular->create_data_uniform(nsend,proclist2,1);
    Send2 *rbuf2 = (Send2 *) memory->smalloc((bigint)nrecv2*sizeof(Send2),"surf2grid:rbuf2");
    irregular->exchange_uniform((char *) sbuf2,sizeof(Send2),(char *) rbuf2);
    delete irregular;

    if (outflag) {
      MPI_Barrier(world);
      t3 = MPI_Wtime();
      tcomm2 += t3-t2;
    }

    // chash = hash with cell IDs I own in RCB decomp
    //   key = childID, value = index in my RCB list of child cells
    // phash = hash with all parent cell IDs of child cells
    //   key = parentID, value not used
    //   b/c RCB grid box is compact, size of phash should be small
    // rcblohi = lo/hi extents of each child cells

    MyHash *chash = new MyHash();
    MyHash *phash = new MyHash();
    RCBlohi *rcblohi =
      (RCBlohi *) memory->smalloc(nrecv2*sizeof(RCBlohi),"surf2grid:rcblohi");

    for (i = 0; i < nrecv2; i++) {
      childID = rbuf2[i].childID;
      (*chash)[childID] = i;
      id_lohi(childID,level,boxlo,boxhi,rcblohi[i].lo,rcblohi[i].hi);

      for (int ilevel = level; ilevel > 0; ilevel--) {
        parentID = id_parent_of_child(childID,ilevel);
        if (phash->find(parentID) != phash->end()) break;
        (*phash)[parentID] = 0;
        childID = parentID;
      }
    }

    // in RCB decomp, compute intersections between:
    //   my RCB child cells (only those that exist) and
    //   set of RCB surfs that overlap my RCB grid box
    // append results one by one to pairs = surf/grid intersections
    // loop over surfs:
    //   check if surf actually intersects with RCB box, else skip it
    //     could just be the bbox of surf overlaps with RCB box
    //   bblo/hi = intersection of surf bbox with RCB box
    //   recurse2d/3d starts with bblo/hi within parentID = 0 (sim box)
    //     will find every child cell in chash that surf intersects with
    //     checks for actual intersection are via cut2d/cut3d
    //   build list of pairs, one pair = surf index, cell index
    //     both are indices into received RCB surf/cells data

    Surf::Line *rcblines;
    Surf::Tri *rcbtris;
    if (dim == 2) rcblines = (Surf::Line *) rbuf1;
    else rcbtris = (Surf::Tri *) rbuf1;

    int npair = 0;
    int overlap;

    if (dim == 2) {
      for (i = 0; i < nrecv1; i++) {

        // skip surf if it does not intersect my RCB box

        overlap = cut2d->surf2grid_one(rcblines[i].p1,rcblines[i].p2,rcblo,rcbhi);
        if (!overlap) continue;

        // slo/hi = bbox around one surf

        surf->bbox_one(&rcblines[i],slo,shi);

        // bblo/hi = overlap of surf bbox with my RCB box

        bblo[0] = MAX(slo[0],rcblo[0]);
        bblo[1] = MAX(slo[1],rcblo[1]);
        bbhi[0] = MIN(shi[0],rcbhi[0]);
        bbhi[1] = MIN(shi[1],rcbhi[1]);
        bblo[2] = 0.0;
        bbhi[2] = 0.0;

        // find all my RCB child cells this surf intersects

        recurse2d(0,0,boxlo,boxhi,i,&rcblines[i],bblo,bbhi,
                  npair,maxpair,pairs,chash,phash);
      }

    } else {
      for (i = 0; i < nrecv1; i++) {

        // skip surf if it does not intersect my RCB box

        overlap = cut3d->surf2grid_one(rcbtris[i].p1,rcbtris[i].p2,rcbtris[i].p3,
                                       rcblo,rcbhi);
        if (!overlap) continue;

        // slo/hi = bbox around one surf

        surf->bbox_one(&rcbtris[i],slo,shi);

        // bblo/hi = overlap of surf bbox with my RCB box

        bblo[0] = MAX(slo[0],rcblo[0]);
        bblo[1] = MAX(slo[1],rcblo[1]);
        bblo[2] = MAX(slo[2],rcblo[2]);
        bbhi[0] = MIN(shi[0],rcbhi[0]);
        bbhi[1] = MIN(shi[1],rcbhi[1]);
        bbhi[2] = MIN(shi[2],rcbhi[2]);

        // find all my RCB child cells this surf intersects

        recurse3d(0,0,boxlo,boxhi,i,&rcbtris[i],bblo,bbhi,
                  npair,maxpair,pairs,chash,phash);
      }
    }

    if (outflag) {
      MPI_Barrier(world);
      t4 = MPI_Wtime();
      tmap += t4-t3;
    }

    // third irregular comm
    // send each surf/grid intersection pair back to proc that owns grid cell

    int surfindex,cellindex;

    nsend = 0;

    for (i = 0; i < npair; i++) {
      if (nsend == maxsend3) {
        maxsend3 += DELTA_SEND;
        memory->grow(proclist3,maxsend3,"surf2grigd:proclist3");
        sbuf3 = (Send3 *) memory->srealloc(sbuf3,maxsend3*sizeof(Send3),
                                          "surf2grid:sbuf3");
      }

      surfindex = pairs[i][0];
      cellindex = pairs[i][1];
      proclist3[i] = rbuf2[cellindex].proc;
      if (dim == 2) sbuf3[i].surfID = rcblines[surfindex].id;
      else sbuf3[i].surfID = rcbtris[surfindex].id;
      sbuf3[i].icell = rbuf2[cellindex].icell;
      nsend++;
    }

    irregular = new Irregular(sparta);
    int nrecv3 = irregular->create_data_uniform(nsend,proclist3,1);
    Send3 *rbuf3 = (Send3 *) memory->smalloc((bigint)nrecv3*sizeof(Send3),
                                             "surf2grid:rbuf3");
    irregular->exchange_uniform((char *) sbuf3,sizeof(Send3),(char *) rbuf3);
    delete irregular;

    // process received cell/surf pairs back in simulation decomposition
    // set nsurf and csurfs for each cell (only child cells at this level)
    // 1st pass: count surfs in each cell, then allocate csurfs in each cell
    // 2nd pass: fill each cell's csurf list

    for (i = 0; i < nrecv3; i++) {
      icell = rbuf3[i].icell;
      cells[icell].nsurf++;
    }

    // skip sub cells since may exist in a restart

    for (icell = 0; icell < nlocal; icell++) {
      if (cells[icell].level != level) continue;
      if (cells[icell].nsplit <= 0) continue;
      nsurf = cells[icell].nsurf;
      if (nsurf) {
        if (nsurf > maxsurfpercell)
          error->one(FLERR,"Too many surfs in one cell - set global surfmax");
        cells[icell].csurfs = csurfs->get(nsurf);
        cells[icell].nsurf = 0;
      }
    }

    for (i = 0; i < nrecv3; i++) {
      icell = rbuf3[i].icell;
      nsurf = cells[icell].nsurf;
      cells[icell].csurfs[nsurf] = rbuf3[i].surfID;
      cells[icell].nsurf++;
    }

    if (outflag) {
      MPI_Barrier(world);
      t5 = MPI_Wtime();
      tcomm3 += t5-t4;
    }

    // clean up for this level iteration

    memory->sfree(rbuf1);
    memory->sfree(rbuf2);
    memory->sfree(rbuf3);
    memory->sfree(rcblohi);
    delete chash;
    delete phash;
  }

  if (outflag) {
    MPI_Barrier(world);
    t1 = MPI_Wtime();
  }

  // clean up after all iterations

  memory->destroy(proclist1);
  memory->destroy(proclist2);
  memory->destroy(proclist3);
  memory->sfree(sbuf1);
  memory->sfree(sbuf2);
  memory->sfree(sbuf3);
  memory->destroy(pairs);
  memory->destroy(plist);
  memory->sfree(gtree);

  // non-distributed surfs:
  // each cell's csurf list currently stores surf IDs
  // convert them indices into global list stored by each proc
  // shash used to store IDs of entire global list

  if (!distributed) {
    lines = surf->lines;
    tris = surf->tris;
    int nslocal = surf->nlocal;

    MySurfHash shash;
    surfint *list;

    if (dim == 2) {
      for (i = 0; i < nslocal; i++)
        shash[lines[i].id] = i;
    } else {
      for (i = 0; i < nslocal; i++)
        shash[tris[i].id] = i;
    }

    for (icell = 0; icell < nlocal; icell++) {
      if (!cells[icell].nsurf) continue;
      if (cells[icell].nsplit <= 0) continue;

      list = cells[icell].csurfs;
      n = cells[icell].nsurf;

      for (i = 0; i < n; i++)
        list[i] = shash[list[i]];
    }
  }

  // distributed surfs:
  // rendezvous operation to obtain nlocal surfs for each proc
  //   these are the surfs that intersect child cells this proc owns
  // each grid cell requests a surf from proc that owns surf in mylines/mytris
  //   use shash to only do this once per surf
  // receive the surf and store in nlocal lines/tris

  if (distributed) {

    // ncount = # of unique surfs I need for my owned grid cells
    // store IDs of those surfs in shash

    MySurfHash shash;
    MyIterator it;
    surfint *list;
    int ncount = 0;

    for (icell = 0; icell < nlocal; icell++) {
      if (!cells[icell].nsurf) continue;
      if (cells[icell].nsplit <= 0) continue;

      list = cells[icell].csurfs;
      n = cells[icell].nsurf;

      for (i = 0; i < n; i++)
        if (shash.find(list[i]) == shash.end()) {
          shash[list[i]] = 0;
          ncount++;
        }
    }

    // allocate memory for rvous input

    int *proclist;
    memory->create(proclist,ncount,"surf2grid:proclist");
    InRvous *inbuf =
      (InRvous *) memory->smalloc((bigint) ncount*sizeof(InRvous),
                                  "surf2grid:inbuf");

    // create rvous inputs
    // proclist = owner of each surf

    surfint surfID;

    ncount = 0;
    for (it = shash.begin(); it != shash.end(); ++it) {
      surfID = it->first;
      proclist[ncount] = (surfID-1) % nprocs;
      inbuf[ncount].proc = me;
      inbuf[ncount].surfID = surfID;
      ncount++;
    }

    // perform rendezvous operation
    // each proc owns subset of surfs
    // receives all surf requests to return surf to each proc who needs it

    char *outbuf;
    int outbytes;
    if (dim == 2) outbytes = sizeof(OutRvousLine);
    else outbytes = sizeof(OutRvousTri);

    int nreturn = comm->rendezvous(1,ncount,(char *) inbuf,sizeof(InRvous),
                                   0,proclist,rendezvous_surfrequest,
                                   0,outbuf,outbytes,(void *) this);

    memory->destroy(proclist);
    memory->sfree(inbuf);

    // copy entire rendezvous output buf into realloced Surf lines/tris

    surf->nlocal = surf->nghost = 0;
    int nmax_old = surf->nmax;
    surf->nmax = surf->nlocal = nreturn;
    surf->grow(nmax_old);

    if (dim == 2) memcpy(surf->lines,outbuf,nreturn*sizeof(Surf::Line));
    else memcpy(surf->tris,outbuf,nreturn*sizeof(Surf::Tri));

    memory->sfree(outbuf);

    // reset Surf hash to point to surf list in lines/tris

    Surf::Line *lines = surf->lines;
    Surf::Tri *tris = surf->tris;

    if (dim == 2) {
      for (i = 0; i < nreturn; i++) {
        surfID = lines[i].id;
        shash[surfID] = i;
      }
    } else {
      for (i = 0; i < nreturn; i++) {
        surfID = tris[i].id;
        shash[surfID] = i;
      }
    }

    // reset csurfs list for each of my owned cells
    // from storing surfID to storing local index of that surfID

    for (icell = 0; icell < nlocal; icell++) {
      if (!cells[icell].nsurf) continue;
      if (cells[icell].nsplit <= 0) continue;

      list = cells[icell].csurfs;
      n = cells[icell].nsurf;

      for (i = 0; i < n; i++)
        list[i] = shash[list[i]];
    }
  }

  // for performance, sort each cell's csurfs list, same order as cell alg
  // mark cells with surfs as OVERLAP, only if has a non-transparent surf

  lines = surf->lines;
  tris = surf->tris;

  surfint *list;
  int nontrans;

  for (icell = 0; icell < nlocal; icell++) {
    if (!cells[icell].nsurf) continue;
    if (cells[icell].nsplit <= 0) continue;

    qsort(cells[icell].csurfs,cells[icell].nsurf,
          sizeof(surfint),compare_surfIDs);

    list = cells[icell].csurfs;
    n = cells[icell].nsurf;
    nontrans = 0;

    if (dim == 2) {
      for (i = 0; i < n; i++) {
        if (!lines[list[i]].transparent) {
          nontrans = 1;
          break;
        }
      }
    } else {
      for (i = 0; i < n; i++) {
        if (!tris[list[i]].transparent) {
          nontrans = 1;
          break;
        }
      }
    }

    if (nontrans) cinfo[icell].type = OVERLAP;
  }

  if (outflag) {
    MPI_Barrier(world);
    t2 = MPI_Wtime();
    tcomm4 = t2-t1;
  }
}

/* ----------------------------------------------------------------------
   compute cut volume of each cell and any split cell info
   nsurf and csurfs list for each grid cell have already been computed
   if subflag = 1, create new owned split and sub cells as needed
     called from ReadSurf, RemoveSurf, MoveSurf, FixAblate
   if subflag = 0, split/sub cells already exist
     called from ReadRestart, only for explicit surfs
   outflag = 1 for timing and statistics info
   in cells: set nsplit, isplit
   in cinfo: set corner, volume
   initialize sinfo as needed
------------------------------------------------------------------------- */

void Grid::surf2grid_split(int subflag, int outflag)
{
  int i,isub,nsplitone,xsub;
  int *surfmap,*ptr;
  double t1,t2;
  double *vols;
  double xsplit[3];
  ChildCell *c;
  SplitInfo *s;

  int dim = domain->dimension;

  if (outflag) {
    MPI_Barrier(world);
    t1 = MPI_Wtime();
  }

  // compute cut volume and possible split of each grid cell by surfs
  // decrement nunsplitlocal if convert an unsplit cell to split cell
  // if nsplitone > 1, create new split cell sinfo and sub-cells
  // skip if nsplit <= 0 b/c split cells could exist if restarting

  int max = 0;
  int ncurrent = nlocal;

  for (int icell = 0; icell < ncurrent; icell++) {
    if (cells[icell].nsplit <= 0) continue;
    if (cinfo[icell].type != OVERLAP) continue;

    surfmap = csplits->vget();
    c = &cells[icell];

    if (dim == 3)
      nsplitone = cut3d->split(c->id,c->lo,c->hi,c->nsurf,c->csurfs,
                               vols,surfmap,cinfo[icell].corner,xsub,xsplit);
    else
      nsplitone = cut2d->split(c->id,c->lo,c->hi,c->nsurf,c->csurfs,
                               vols,surfmap,cinfo[icell].corner,xsub,xsplit);

    if (nsplitone == 1) {
      cinfo[icell].volume = vols[0];

    } else if (subflag) {
      if (nsplitone > maxsplitpercell) {
        max = MAX(max,nsplitone);
        csplits->vgot(0);

      } else {
        cells[icell].nsplit = nsplitone;
        nunsplitlocal--;

        cells[icell].isplit = nsplitlocal;
        add_split_cell(1);
        s = &sinfo[nsplitlocal-1];
        s->icell = icell;
        s->csplits = surfmap;
        s->xsub = xsub;
        s->xsplit[0] = xsplit[0];
        s->xsplit[1] = xsplit[1];
        if (dim == 3) s->xsplit[2] = xsplit[2];
        else s->xsplit[2] = 0.0;

        ptr = s->csubs = csubs->vget();

        // add nsplitone sub cells
        // collide and fixes also need to add cells

        for (i = 0; i < nsplitone; i++) {
          isub = nlocal;
          add_sub_cell(icell,1);
          if (collide) collide->add_grid_one();
          if (modify->n_pergrid) modify->add_grid_one();
          cells[isub].nsplit = -i;
          cinfo[isub].volume = vols[i];
          ptr[i] = isub;
        }

        csubs->vgot(nsplitone);
        csplits->vgot(cells[icell].nsurf);
      }

    } else {
      if (cells[icell].nsplit != nsplitone) {
        printf("BAD %d " CELLINT_FORMAT ": %d %d\n",icell,cells[icell].id,
               nsplitone,cells[icell].nsplit);
        error->one(FLERR,
                   "Inconsistent surface to grid mapping in read_restart");
      }

      s = &sinfo[cells[icell].isplit];
      s->csplits = surfmap;
      s->xsub = xsub;
      s->xsplit[0] = xsplit[0];
      s->xsplit[1] = xsplit[1];
      if (dim == 3) s->xsplit[2] = xsplit[2];
      else s->xsplit[2] = 0.0;

      ptr = s->csubs;
      for (i = 0; i < nsplitone; i++) {
        isub = ptr[i];
        cells[isub].nsurf = cells[icell].nsurf;
        cells[isub].csurfs = cells[icell].csurfs;
        cinfo[isub].volume = vols[i];
      }

      csplits->vgot(cells[icell].nsurf);
    }
  }

  // error if split count exceeds maxsplitpercell for any cell

  int maxall;
  MPI_Allreduce(&max,&maxall,1,MPI_INT,MPI_MAX,world);
  if (maxall) {
    if (me == 0) printf("Max split cells in any cell = %d\n",maxall);
    error->all(FLERR,"Too many split cells in a single cell - "
               "set global splitmax");
  }

  // stats on unmarked corner points in OVERLAP cells

  if (outflag) {
    int noverlap = 0;
    int ncorner = 0;
    for (int icell = 0; icell < nlocal; icell++) {
      if (cells[icell].nsplit <= 0) continue;
      if (cinfo[icell].type == OVERLAP) {
        noverlap++;
        if (cinfo[icell].corner[0] == UNKNOWN) ncorner++;
      }
    }

    bigint bncorner = ncorner;
    bigint bnoverlap = noverlap;
    bigint ncornerall,noverlapall;
    MPI_Allreduce(&bncorner,&ncornerall,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
    MPI_Allreduce(&bnoverlap,&noverlapall,1,MPI_SPARTA_BIGINT,MPI_SUM,world);

    if (comm->me == 0) {
      if (screen) fprintf(screen,"  " BIGINT_FORMAT " " BIGINT_FORMAT
                          " = cells overlapping surfs, "
                          "overlap cells with unmarked corner pts\n",
                          noverlapall,ncornerall);
      if (logfile) fprintf(logfile,"  " BIGINT_FORMAT " " BIGINT_FORMAT
                           " = cells overlapping surfs, "
                           "overlap cells with unmarked corner pts\n",
                           noverlapall,ncornerall);
    }
  }

  // print info on unusual surf split cases

  if (dim == 3) {
    bigint ntiny = cut3d->ntiny;
    bigint alltiny;
    MPI_Allreduce(&ntiny,&alltiny,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
    if (alltiny && comm->me == 0) {
      if (screen) fprintf(screen,"  " BIGINT_FORMAT " tiny edges removed\n",
                          alltiny);
      if (logfile) fprintf(logfile,"  " BIGINT_FORMAT " tiny edges removed\n",
                           alltiny);
    }

    bigint nshrink = cut3d->nshrink;
    bigint allshrink;
    MPI_Allreduce(&nshrink,&allshrink,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
    if (allshrink && comm->me == 0) {
      if (screen)
        fprintf(screen,"  " BIGINT_FORMAT
                " cells shrunk to enable splitting\n",allshrink);
      if (logfile)
        fprintf(logfile,"  " BIGINT_FORMAT
                " cells shrunk to enable splitting\n",allshrink);
    }
  }

  if (outflag) {
    MPI_Barrier(world);
    t2 = MPI_Wtime();
    tsplit = t2-t1;
  }
}

/* ----------------------------------------------------------------------
   enumerate all child cells in chash which a single line intersects with
   done recursively, 1st call from surf2grid_surf_algorithm() uses parentID = root
   phash stores IDs of all parent cells for child cells in chash
   bblo/hi = portion of bounding box for surf that is wholly within parentID
   parentID = parent cell
   level = level of parent cell
   plo/phi = corner points of parent cell
   surfindex, line = surf element and its local index in caller
   npair, maxpair, pair = growing list of I,J gridcell/surf overlap pairs
   chash, phash = hashes of cell IDs for child and parent cells in this proc's RCB box
   cut2d->surf2grid_one() is used to determine actual overlap
------------------------------------------------------------------------- */

void Grid::recurse2d(cellint parentID, int level, double *plo, double *phi,
                     int surfindex, Surf::Line *line, double *bblo, double *bbhi,
                     int &npair, int &maxpair, int **&pairs,
                     MyHash *chash, MyHash *phash)
{
  int ix,iy,cflag,pflag,overlap;
  cellint ichild,childID;
  double celledge;
  double clo[3],chi[3];
  double newlo[3],newhi[3];

  double *p1 = line->p1;
  double *p2 = line->p2;

  int nx = plevels[level].nx;
  int ny = plevels[level].ny;
  int nbits = plevels[level].nbits;

  // ij lohi = indices for range of child cells overlapped by surf bbox
  // overlap = surf bbox include any interior of grid cell or touches its boundary
  // id_point_child() returns cell indices for >= lo and < hi
  // so check if lower range should be decremented

  int ilo,ihi,jlo,jhi,klo,khi;
  id_point_child(bblo,plo,phi,nx,ny,1,ilo,jlo,klo);
  id_point_child(bbhi,plo,phi,nx,ny,1,ihi,jhi,khi);

  celledge = plo[0] + ilo*(phi[0]-plo[0])/nx;
  if (bblo[0] <= celledge && ilo > 0) ilo--;
  celledge = plo[1] + jlo*(phi[1]-plo[1])/ny;
  if (bblo[1] <= celledge && jlo > 0) jlo--;

  // loop over range of grid cells between ij lohi inclusive
  // if cell is neither a child or parent cell in chash/phash, skip it
  // if line does not intersect cell, skip it
  // if cell is a child, add intersection to pairs list
  // if cell is a parent:
  //   recurse using new lohi for intersection of surf bbox with new parent cell

  for (iy = jlo; iy <= jhi; iy++) {
    for (ix = ilo; ix <= ihi; ix++) {
      ichild = (cellint) iy*nx + ix + 1;
      childID = parentID | (ichild << nbits);

      if (chash->find(childID) == chash->end()) cflag = 0;
      else cflag = 1;
      if (phash->find(childID) == phash->end()) pflag = 0;
      else pflag = 1;
      if (!cflag && !pflag) continue;

      grid->id_child_lohi(level,plo,phi,ichild,clo,chi);
      overlap = cut2d->surf2grid_one(p1,p2,clo,chi);
      if (!overlap) continue;

      if (cflag) {
        if (npair == maxpair) {
          maxpair += DELTA_SEND;
          memory->grow(pairs,maxpair,2,"surf2grid:pairs");
        }
        pairs[npair][0] = surfindex;
        pairs[npair][1] = (*chash)[childID];
        npair++;
        continue;
      }

      if (pflag) {
        newlo[0] = MAX(bblo[0],clo[0]);
        newlo[1] = MAX(bblo[1],clo[1]);
        newhi[0] = MIN(bbhi[0],chi[0]);
        newhi[1] = MIN(bbhi[1],chi[1]);

        // 3rd dim is unused in 2d, but must still be set
        // id_point_child() reads it, and the caller of recurse2d() sets it to 0

        newlo[2] = 0.0;
        newhi[2] = 0.0;
        recurse2d(childID,level+1,clo,chi,surfindex,line,newlo,newhi,
                  npair,maxpair,pairs,chash,phash);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   enumerate all child cells in chash which a single tri intersects with
   done recursively, 1st call from surf2grid_surf_algorithm() uses parentID = root
     phash stores IDs of all parent cells for child cells in chash
   bblo/hi = portion of bounding box for surf that is wholly within parentID
   parentID = parent cell
   level = level of parent cell
   plo/phi = corner points of parent cell
   surfindex, tri = surf element and its local index in caller
   npair, maxpair, pair = growing list of I,J gridcell/surf overlap pairs
   chash, phash = hashes of cell IDs for child and parent cells in this proc's RCB box
   cut3d->surf2grid_one() is used to determine actual overlap
------------------------------------------------------------------------- */

void Grid::recurse3d(cellint parentID, int level, double *plo, double *phi,
                     int surfindex, Surf::Tri *tri, double *bblo, double *bbhi,
                     int &npair, int &maxpair, int **&pairs,
                     MyHash *chash, MyHash *phash)
{
  int ix,iy,iz,cflag,pflag,overlap;
  cellint ichild,childID;
  double celledge;
  double clo[3],chi[3];
  double newlo[3],newhi[3];

  double *p1 = tri->p1;
  double *p2 = tri->p2;
  double *p3 = tri->p3;

  int nx = plevels[level].nx;
  int ny = plevels[level].ny;
  int nz = plevels[level].nz;
  int nbits = plevels[level].nbits;

  // ijk lohi = indices for range of child cells overlapped by surf bbox
  // overlap = surf bbox include any interior of grid cell or touches its boundary
  // id_point_child() returns cell indices for >= lo and < hi
  // so check if lower range should be decremented

  int ilo,ihi,jlo,jhi,klo,khi;
  id_point_child(bblo,plo,phi,nx,ny,nz,ilo,jlo,klo);
  id_point_child(bbhi,plo,phi,nx,ny,nz,ihi,jhi,khi);

  celledge = plo[0] + ilo*(phi[0]-plo[0])/nx;
  if (bblo[0] <= celledge && ilo > 0) ilo--;
  celledge = plo[1] + jlo*(phi[1]-plo[1])/ny;
  if (bblo[1] <= celledge && jlo > 0) jlo--;
  celledge = plo[2] + klo*(phi[2]-plo[2])/nz;
  if (bblo[2] <= celledge && klo > 0) klo--;

  // loop over range of grid cells between ij lohi inclusive
  // if cell is neither a child or parent cell in chash/phash, skip it
  // if tri does not intersect cell, skip it
  // if cell is a child, add intersectino to pairs list
  // if pairs is a parent:
  //   recurse using new lohi for intersection of surf bbox with new parent cell

  for (iz = klo; iz <= khi; iz++) {
    for (iy = jlo; iy <= jhi; iy++) {
      for (ix = ilo; ix <= ihi; ix++) {
        ichild = (cellint) iz*nx*ny + (cellint) iy*nx + ix + 1;
        childID = parentID | (ichild << nbits);

        if (chash->find(childID) == chash->end()) cflag = 0;
        else cflag = 1;
        if (phash->find(childID) == phash->end()) pflag = 0;
        else pflag = 1;
        if (!cflag && !pflag) continue;

        grid->id_child_lohi(level,plo,phi,ichild,clo,chi);
        overlap = cut3d->surf2grid_one(p1,p2,p3,clo,chi);
        if (!overlap) continue;

        if (cflag) {
          if (npair == maxpair) {
            maxpair += DELTA_SEND;
            memory->grow(pairs,maxpair,2,"surf2grid:pairs");
          }
          pairs[npair][0] = surfindex;
          pairs[npair][1] = (*chash)[childID];
          npair++;
          continue;
        }

        if (pflag) {
          newlo[0] = MAX(bblo[0],clo[0]);
          newlo[1] = MAX(bblo[1],clo[1]);
          newlo[2] = MAX(bblo[2],clo[2]);
          newhi[0] = MIN(bbhi[0],chi[0]);
          newhi[1] = MIN(bbhi[1],chi[1]);
          newhi[2] = MIN(bbhi[2],chi[2]);
          recurse3d(childID,level+1,clo,chi,surfindex,tri,newlo,newhi,
                    npair,maxpair,pairs,chash,phash);
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   recursive method to partition a block of uniform grid cells
   uses RCB to create one sub-block per processor
   xyz lo/hi = extent of initial grid block
   proc lower/upper = 0 and Nprocs-1 initially
   output: gtree = tree of RCB cuts
------------------------------------------------------------------------- */

void Grid::partition_grid(int proclower, int procupper,
                          int xlo, int xhi, int ylo, int yhi, int zlo, int zhi,
                          GridTree *gtree)
{
  // end recursion when partition is a single proc

  if (proclower == procupper) return;

  int procmid = proclower + (procupper-proclower) / 2 + 1;
  int nplower = procmid-proclower;
  int npupper = procupper-procmid + 1;

  int xrange = xhi-xlo + 1;
  int yrange = yhi-ylo + 1;
  int zrange = zhi-zlo + 1;

  if (xrange >= yrange && xrange >= zrange) {
    int mid = xlo + static_cast<int> ((0.5*nplower/npupper) * xrange);
    gtree[procmid].dim = 0;
    gtree[procmid].cut = mid;
    partition_grid(proclower,procmid-1,xlo,mid-1,ylo,yhi,zlo,zhi,gtree);
    partition_grid(procmid,procupper,mid,xhi,ylo,yhi,zlo,zhi,gtree);
  } else if (yrange >= zrange) {
    int mid = ylo + static_cast<int> ((0.5*nplower/npupper) * yrange);
    gtree[procmid].dim = 1;
    gtree[procmid].cut = mid;
    partition_grid(proclower,procmid-1,xlo,xhi,ylo,mid-1,zlo,zhi,gtree);
    partition_grid(procmid,procupper,xlo,xhi,mid,yhi,zlo,zhi,gtree);
  } else {
    int mid = zlo + static_cast<int> ((0.5*nplower/npupper) * zrange);
    gtree[procmid].dim = 2;
    gtree[procmid].cut = mid;
    partition_grid(proclower,procmid-1,xlo,xhi,ylo,yhi,zlo,mid-1,gtree);
    partition_grid(procmid,procupper,xlo,xhi,ylo,yhi,mid,zhi,gtree);
  }
}

/* ----------------------------------------------------------------------
   recursive method to identify my sub-block uniform grid cell block
   xyz lohi = extent of initial grid block
   proc lower/upper = 0 and Nprocs-1 initially
   traverses RCB gtree of cuts to zoom in on this processor
   output: xyz lohi will be overwritten with this proc's sub-block extent
------------------------------------------------------------------------- */

void Grid::mybox(int me, int proclower, int procupper,
                 int &xlo, int &xhi, int &ylo, int &yhi, int &zlo, int &zhi,
                 GridTree *gtree)
{
  // end recursion when partition is a single proc

  if (proclower == procupper) return;

  int procmid = proclower + (procupper-proclower) / 2 + 1;

  if (me < procmid) {
    if (gtree[procmid].dim == 0) xhi = gtree[procmid].cut-1;
    else if (gtree[procmid].dim == 1) yhi = gtree[procmid].cut-1;
    else zhi = gtree[procmid].cut-1;
    mybox(me,proclower,procmid-1,xlo,xhi,ylo,yhi,zlo,zhi,gtree);
  } else {
    if (gtree[procmid].dim == 0) xlo = gtree[procmid].cut;
    else if (gtree[procmid].dim == 1) ylo = gtree[procmid].cut;
    else zlo = gtree[procmid].cut;
    mybox(me,procmid,procupper,xlo,xhi,ylo,yhi,zlo,zhi,gtree);
  }
}

/* ----------------------------------------------------------------------
   recursive method to drop a grid box down RCB tree to identify procs it overlaps
   lo/hi = indices (0 to N-1) of uniform grid block for range of box
   proc lower/upper = 0 and Nprocs-1 initially
   traverses RCB gtree to drop box on one or both sides of each cut
   output: noverlap = # of proc sub-boxes the box overlaps
   output: overlap = list of proc IDs the box overlaps
   overlap vector is allocated by caller
------------------------------------------------------------------------- */

void Grid::box_drop(int *lo, int *hi, int proclower, int procupper,
                    GridTree *gtree, int &noverlap, int *overlap)
{
  // end recursion when partition is a single proc
  // add proc to overlap list

  if (proclower == procupper) {
    overlap[noverlap++] = proclower;
    return;
  }

  // drop box on each side of cut it extends beyond
  // use of < and >= criteria are important
  // procmid = 1st processor in upper half of partition
  //         = location in tree that stores this cut
  // dim = 0,1,2 dimension of cut
  // cut = position of cut

  int procmid = proclower + (procupper - proclower) / 2 + 1;
  int dim = gtree[procmid].dim;
  int cut = gtree[procmid].cut;

  if (lo[dim] < cut)
    box_drop(lo,hi,proclower,procmid-1,gtree,noverlap,overlap);
  if (hi[dim] >= cut)
    box_drop(lo,hi,procmid,procupper,gtree,noverlap,overlap);
}

/* ----------------------------------------------------------------------
   rendezvous decomposition computation
   return surf info for each requested surf element
   recv (proc,surfID) pairs from requestor of each surf
   send memcpy of surf info back to the requesting proc
---------------------------------------------------------------------- */

int Grid::rendezvous_surfrequest(int n, char *inbuf, int &flag,
                                 int *&proclist, char *&outbuf,
                                 void *ptr)
{
  int i,m;

  Grid *gptr = (Grid *) ptr;
  Surf::Line *mylines = gptr->surf->mylines;
  Surf::Tri *mytris = gptr->surf->mytris;
  Memory *memory = gptr->memory;
  int dim = gptr->domain->dimension;
  int nprocs = gptr->comm->nprocs;

  InRvous *in = (InRvous *) inbuf;

  // proclist = list of procs who made requests
  // outbuf = list of lines or tris to pass back to comm->rendezvous

  memory->create(proclist,n,"gridsurf:proclist");

  int surfbytes;
  if (dim == 2) surfbytes = sizeof(OutRvousLine);
  else surfbytes = sizeof(OutRvousTri);
  outbuf = (char *) memory->smalloc((bigint) n*surfbytes,"gridsurf:outbuf");

  if (dim == 2) {
    for (i = 0; i < n; i++) {
      proclist[i] = in[i].proc;
      m = (in[i].surfID-1) / nprocs;
      memcpy(&outbuf[(bigint) i*surfbytes],&mylines[m],surfbytes);
    }
  } else {
    for (i = 0; i < n; i++) {
      proclist[i] = in[i].proc;
      m = (in[i].surfID-1) / nprocs;
      memcpy(&outbuf[(bigint) i*surfbytes],&mytris[m],surfbytes);
    }
  }

  // Comm::rendezvous will delete proclist and outbuf
  // flag = 2: new outbuf

  flag = 2;
  return n;
}

/* ----------------------------------------------------------------------
   turn an unsplit owned cell into a split cell with Nsplit sub cells
   the caller supplies the split: csplits maps each of the cell's surfs
     to the piece it bounds, xsub/xsplit locate the reference point
     Update::split2d/3d() cast a ray to, and vols are the flow volumes
     of the pieces, as returned by Cut2d/Cut3d::split()
   csplits is stored by reference: the caller owns it and must keep it
     alive and the same length as the cell's surf list, which
     Update::split2d/3d() index in lockstep with it
   csubs is filled in with the indices of the new sub cells and stored
     by reference on the same terms
   same sequence as the split branch of surf2grid_one(), and like it may
     only be called when this proc stores no ghost cells, since each new
     sub cell is appended at nlocal, which is otherwise the first ghost
   the caller owns the particles in the cell: on return they are all
     still labelled with the split cell, which holds none of its own, so
     assign_split_cell_particles() has to follow
------------------------------------------------------------------------- */

void Grid::split_cell_set(int icell, int nsplitone, int *csplits, int *csubs,
                          int xsub, double *xsplit, double *vols)
{
  int dim = domain->dimension;

  cells[icell].nsplit = nsplitone;
  nunsplitlocal--;

  cells[icell].isplit = nsplitlocal;
  add_split_cell(1);
  SplitInfo *s = &sinfo[nsplitlocal-1];
  s->icell = icell;
  s->csplits = csplits;
  s->csubs = csubs;
  s->xsub = xsub;
  s->xsplit[0] = xsplit[0];
  s->xsplit[1] = xsplit[1];
  if (dim == 3) s->xsplit[2] = xsplit[2];
  else s->xsplit[2] = 0.0;

  // one sub cell per piece, appended after the split cell
  // collide and fixes also need to add cells
  // add_sub_cell() copies the split cell, including its ilocal and its
  //   particle list, so both are reset here

  for (int i = 0; i < nsplitone; i++) {
    int isub = nlocal;
    add_sub_cell(icell,1);
    if (collide) collide->add_grid_one();
    if (modify->n_pergrid) modify->add_grid_one();
    cells[isub].nsplit = -i;
    cells[isub].ilocal = isub;
    cinfo[isub].volume = vols[i];
    cinfo[isub].count = 0;
    cinfo[isub].first = -1;
    csubs[i] = isub;
  }
}

/* ----------------------------------------------------------------------
   turn a split owned cell back into an unsplit cell
   its sub cells are detached and marked for removal rather than removed
     here, so that a batch of split changes can be applied before the
     cell list is compacted once by remove_marked_cells()
   the caller must move the particles out of the sub cells first, e.g.
     with combine_split_cell_particles(icell,1)
   the abandoned SplitInfo entry is reclaimed by remove_marked_cells()
------------------------------------------------------------------------- */

void Grid::split_cell_unset(int icell)
{
  if (cells[icell].nsplit <= 1) return;

  int isplit = cells[icell].isplit;
  int nsplit = cells[icell].nsplit;
  int *mycsubs = sinfo[isplit].csubs;

  for (int i = 0; i < nsplit; i++) {
    int isub = mycsubs[i];
    cells[isub].nsplit = 1;
    cells[isub].isplit = -1;
    cells[isub].nsurf = 0;
    cells[isub].csurfs = NULL;
    cells[isub].proc = -1;
    cinfo[isub].count = 0;
    cinfo[isub].first = -1;
  }

  cells[icell].nsplit = 1;
  cells[icell].isplit = -1;
}

/* ----------------------------------------------------------------------
   remove every owned cell marked for deletion with proc = -1
   cells are compacted by moving the last cell into each hole, as
     clear_surf() does, and every reference to a moved cell is repaired:
     its SplitInfo back pointer, its own index, the particles it holds,
     and the per-cell data collide and the fixes keep
   neighbor links hold cell IDs at this point (Grid::unset_neighbors())
     and ghost cells have been removed, so neither needs fixing here;
     the caller re-establishes both
   the SplitInfo list is compacted afterwards, dropping the entries of
     the cells which stopped being split
   assumes particles are sorted; they are still sorted on return
   returns the # of cells removed
------------------------------------------------------------------------- */

int Grid::remove_marked_cells()
{
  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;
  int nlocal_prev = nlocal;
  nmoved = 0;

  int icell = 0;
  while (icell < nlocal) {
    if (cells[icell].proc != -1) {
      icell++;
      continue;
    }

    if (icell != nlocal-1) {
      memcpy(&cells[icell],&cells[nlocal-1],sizeof(ChildCell));
      memcpy(&cinfo[icell],&cinfo[nlocal-1],sizeof(ChildInfo));
      if (ncustom) copy_custom(nlocal-1,icell);
      if (collide) collide->copy_grid_one(nlocal-1,icell);
      if (modify->n_pergrid) modify->copy_grid_one(nlocal-1,icell);

      // repair the references to the cell which moved into this slot

      cells[icell].ilocal = icell;
      if (cells[icell].nsplit > 1)
        sinfo[cells[icell].isplit].icell = icell;
      else if (cells[icell].nsplit <= 0)
        sinfo[cells[icell].isplit].csubs[-cells[icell].nsplit] = icell;

      // patch the hash for the cell which moved, so a caller which only
      //   removes a few cells (fix rigid re-cutting split cells every step)
      //   need not rehash the whole grid afterwards
      // conditions, in order:
      //   proc != -1: the cell which moved in may itself be marked for
      //     removal (the loop does not re-test icell, it removes it on the
      //     next pass), and such a cell must not be entered in the hash
      //   nsplit >= 1: a sub cell shares the ID of its split cell
      //     (add_sub_cell copies the parent's ChildCell) and is not the
      //     owner of that ID; rehash() skips it, so this must too

      if (hashcurrent && cells[icell].proc != -1 && cells[icell].nsplit >= 1)
        (*hash)[cells[icell].id] = icell;

      int ip = cinfo[icell].first;
      while (ip >= 0) {
        particles[ip].icell = icell;
        ip = next[ip];
      }

      // for a caller which labels particles outside Grid (KOKKOS)

      if (nmoved == maxmoved) {
        maxmoved += DELTA_MOVED;
        memory->grow(movedfrom,maxmoved,"grid:movedfrom");
        memory->grow(movedto,maxmoved,"grid:movedto");
      }
      movedfrom[nmoved] = nlocal-1;
      movedto[nmoved] = icell;
      nmoved++;
    }

    nlocal--;
  }

  if (nlocal == nlocal_prev) return 0;

  // compact sinfo, dropping the entry of every cell which is no longer
  //   split, and re-point each split cell and its sub cells at it

  int nsplitnew = 0;
  for (int i = 0; i < nsplitlocal; i++) {
    int ic = sinfo[i].icell;
    if (ic < 0 || ic >= nlocal) continue;
    if (cells[ic].nsplit <= 1) continue;
    if (cells[ic].isplit != i) continue;
    if (i != nsplitnew) memcpy(&sinfo[nsplitnew],&sinfo[i],sizeof(SplitInfo));
    cells[ic].isplit = nsplitnew;
    int *mycsubs = sinfo[nsplitnew].csubs;
    for (int j = 0; j < cells[ic].nsplit; j++)
      cells[mycsubs[j]].isplit = nsplitnew;
    nsplitnew++;
  }
  nsplitlocal = nsplitnew;

  // re-derive the cell-kind counts, as setup_owned() does for unsplit

  nsublocal = 0;
  for (int i = 0; i < nlocal; i++)
    if (cells[i].nsplit <= 0) nsublocal++;
  nunsplitlocal = nlocal - nsplitlocal - nsublocal;

  if (collide) collide->reset_grid_count(nlocal);
  if (modify->n_pergrid) modify->reset_grid_count(nlocal);

  // the hash is left VALID when it was valid on entry: only sub cells are
  //   ever marked for removal here, and a sub cell is not the owner of its
  //   ID so it has no hash entry of its own; every cell whose index moved
  //   had its entry patched in the loop above
  // halo_index maps a position to a local index the same way, so it moves
  //   with the hash and is refreshed here rather than invalidated
  // hashfilled stays as it was: a caller which never filled the hash still
  //   has an empty one

  if (hashcurrent) update_halo_index();
  else hashfilled = 0;
  clear_cell_bins();

  return nlocal_prev - nlocal;
}

/* ----------------------------------------------------------------------
   1 if restructure_split_cells() would move an owned cell which is not
     a sub cell, else 0
   only sub cells can move without informing other procs: a ghost copy of
     any other cell records its owner's index for particle migration,
     while a ghost sub cell routes to its split cell (subroute), so its
     record never goes stale
   the cells which move are the ones the compaction pulls off the tail
     of the owned list, found here by the same walk without moving them
   collective callers reduce the result so every proc takes the same path
------------------------------------------------------------------------- */

int Grid::restructure_check(int n, SplitChange *list)
{
  int i,m,icell,nsplit;

  // holes = sub cells given up, nadd = sub cells created
  // the first nadd holes (ascending) take the new sub cells,
  //   the rest take cells from the tail

  int nhole = 0;
  int nadd = 0;
  for (m = 0; m < n; m++) {
    nsplit = cells[list[m].icell].nsplit;
    if (nsplit > 1) nhole += nsplit;
    if (list[m].nsplitnew > 1) nadd += list[m].nsplitnew;
  }
  if (nadd >= nhole) return 0;

  int *holes;
  memory->create(holes,nhole,"grid:holes");
  nhole = 0;
  for (m = 0; m < n; m++) {
    icell = list[m].icell;
    nsplit = cells[icell].nsplit;
    if (nsplit <= 1) continue;
    int *mycsubs = sinfo[cells[icell].isplit].csubs;
    for (i = 0; i < nsplit; i++) holes[nhole++] = mycsubs[i];
  }
  std::sort(holes,holes+nhole);

  // walk the tail as the compaction will: a tail cell which is itself a
  //   hole is dropped, any other tail cell moves into the next hole

  int flag = 0;
  int nl = nlocal;
  int ihole = nhole-1;
  for (i = nadd; i < nhole; i++) {
    while (ihole >= 0 && holes[ihole] == nl-1) {
      nl--;
      ihole--;
    }
    if (holes[i] >= nl) break;
    if (cells[nl-1].nsplit > 0) {
      flag = 1;
      break;
    }
    nl--;
  }

  memory->destroy(holes);
  return flag;
}

/* ----------------------------------------------------------------------
   apply N changes of the number of flow pieces of owned cells in place
   the cut lists, flow volumes, types and corner marks of the cells are
     already current (see set_cell_surfs() etc.); only the sub cells
     change: the old ones of every changing cell are given up, new ones
     created, and the cell list compacted, without dropping the ghost
     cells and neighbor links or communicating
   the resulting layout of the owned cells is exactly the one
     split_cell_set() + remove_marked_cells() produce: the new sub cells
     are appended after the owned cells, then the holes left by the old
     sub cells are filled from the tail, lowest hole first, so results
     do not depend on which path was taken
   only sub cells move (restructure_check() must have returned 0 on
     every proc): they are referenced only by their split cell's csubs
     and by their particles, so no hash, neighbor or ghost record of
     another proc goes stale.  the ghost block is moved aside or closed
     up by moving a few ghost cells, whose references are repaired by
     move_cell()
   the caller has pulled the particles of every changing split cell up
     into the cell itself, so no particle is labelled with a hole
   the cells moved are listed in movedfrom/movedto for a caller which
     keeps particle labels outside Grid (KOKKOS)
------------------------------------------------------------------------- */

void Grid::restructure_split_cells(int n, SplitChange *list)
{
  int i,j,m,icell,isplit,isub,nsplit;

  int nlocal_old = nlocal;
  nmoved = 0;

  // holes = the sub cells every changing split cell gives up, detached
  //   as split_cell_unset() does; their sinfo entries become free slots

  int nhole = 0;
  int nadd = 0;
  int nfree = 0;
  for (m = 0; m < n; m++) {
    nsplit = cells[list[m].icell].nsplit;
    if (nsplit > 1) {
      nhole += nsplit;
      nfree++;
    }
    if (list[m].nsplitnew > 1) nadd += list[m].nsplitnew;
  }

  int *holes,*freeslots;
  memory->create(holes,nhole,"grid:holes");
  memory->create(freeslots,nfree,"grid:freeslots");
  nhole = nfree = 0;

  for (m = 0; m < n; m++) {
    icell = list[m].icell;
    nsplit = cells[icell].nsplit;
    if (nsplit <= 1) continue;
    isplit = cells[icell].isplit;
    int *mycsubs = sinfo[isplit].csubs;
    for (i = 0; i < nsplit; i++) holes[nhole++] = mycsubs[i];
    split_cell_unset(icell);
    sinfo[isplit].icell = -1;
    freeslots[nfree++] = isplit;
    nsublocal -= nsplit;
    if (journalflag) journal_cell(icell);
  }
  std::sort(holes,holes+nhole);
  std::sort(freeslots,freeslots+nfree);

  // layout: the first min(nadd,nhole) holes take new sub cells, in the
  //   reverse of their creation order; nkeep new sub cells remain
  //   appended at nlocal_old and up, for which the first nkeep ghost
  //   cells step aside to the end of the ghost block

  int nkeep = MAX(0,nadd-nhole);

  if (nkeep) {
    grow_cells(nkeep,nkeep);
    int nmove = MIN(nkeep,nghost);
    int start = nlocal_old + MAX(nkeep,nghost);
    for (i = 0; i < nmove; i++) move_cell(nlocal_old+i,start+i);
  }

  // create the new sub cells of every changing cell, in list order
  // per-cell data of collide and the per-grid fixes is appended in the
  //   same order and copied into a hole exactly as the compaction of
  //   the appended cells would have done

  int ifree = 0;
  j = 0;
  for (m = 0; m < n; m++) {
    if (list[m].nsplitnew <= 1) continue;
    icell = list[m].icell;
    nsplit = list[m].nsplitnew;

    // a free sinfo slot, else a new one before the ghost entries

    if (ifree < nfree) isplit = freeslots[ifree++];
    else {
      grow_sinfo(1);
      isplit = nsplitlocal;
      if (nsplitghost) move_sinfo(isplit,isplit+nsplitghost);
      nsplitlocal++;
    }

    int *mycsplits = csplits->get(list[m].nsurf);
    int *mycsubs = csubs->get(nsplit);
    if (!mycsplits || !mycsubs)
      error->one(FLERR,"Failed to allocate grid split cell lists");
    memcpy(mycsplits,list[m].map,list[m].nsurf*sizeof(int));
    surflist_churn += (list[m].nsurf + nsplit) * sizeof(int);

    SplitInfo *s = &sinfo[isplit];
    s->icell = icell;
    s->csplits = mycsplits;
    s->csubs = mycsubs;
    s->xsub = list[m].xsub;
    s->xsplit[0] = list[m].xsplit[0];
    s->xsplit[1] = list[m].xsplit[1];
    if (domain->dimension == 3) s->xsplit[2] = list[m].xsplit[2];
    else s->xsplit[2] = 0.0;

    cells[icell].nsplit = nsplit;
    cells[icell].isplit = isplit;
    if (journalflag) {
      journal_cell(icell);
      journal_sinfo(isplit);
    }

    for (i = 0; i < nsplit; i++) {
      if (j < nkeep) isub = nlocal_old + j;
      else isub = holes[nadd-1-j];
      add_sub_cell_at(icell,isub,i);
      cinfo[isub].volume = list[m].vols[i];
      mycsubs[i] = isub;
      if (journalflag) journal_cell(isub);
      if (collide) collide->add_grid_one();
      if (modify->n_pergrid) modify->add_grid_one();
      if (isub != nlocal_old+j) {
        if (collide) collide->copy_grid_one(nlocal_old+j,isub);
        if (modify->n_pergrid) modify->copy_grid_one(nlocal_old+j,isub);
      }
      j++;
    }
  }

  // the remaining holes are filled from the tail, lowest hole first:
  //   a tail cell which is itself a hole is dropped, any other moves in

  int nl = nlocal_old;
  for (i = nadd; i < nhole; i++) {
    while (nl > 0 && cells[nl-1].proc == -1) nl--;
    if (holes[i] >= nl) break;
    move_cell(nl-1,holes[i]);
    nl--;
  }

  // a shorter owned list: the ghost block closes up behind it

  if (nl < nlocal_old) {
    int nmove = MIN(nlocal_old-nl,nghost);
    for (i = 0; i < nmove; i++)
      move_cell(nlocal_old+nghost-nmove+i,nl+i);
    nlocal = nl;
  } else nlocal = nlocal_old + nkeep;

  // free sinfo slots not reused are filled from the tail of the owned
  //   entries; the ghost entries close up behind them

  int nsplit_old = nsplitlocal;
  for (i = ifree; i < nfree; i++) {
    while (nsplitlocal > 0 && sinfo[nsplitlocal-1].icell < 0) nsplitlocal--;
    if (freeslots[i] >= nsplitlocal) break;
    move_sinfo(nsplitlocal-1,freeslots[i]);
    nsplitlocal--;
  }
  while (nsplitlocal > 0 && sinfo[nsplitlocal-1].icell < 0) nsplitlocal--;
  if (nsplitlocal < nsplit_old) {
    int nmove = MIN(nsplit_old-nsplitlocal,nsplitghost);
    for (i = 0; i < nmove; i++)
      move_sinfo(nsplit_old+nsplitghost-nmove+i,nsplitlocal+i);
  }

  nunsplitlocal = nlocal - nsplitlocal - nsublocal;

  if (collide) collide->reset_grid_count(nlocal);
  if (modify->n_pergrid) modify->reset_grid_count(nlocal);

  // a grid with cells at several levels: a finer neighbor of a moved
  //   cell is not reachable from the cell itself, so the links of every
  //   cell are scanned for the old indices of the moved cells

  if (neighscan) {
    int nold = nlocal_old + nghost + nkeep;
    int *newidx;
    memory->create(newidx,nold,"grid:newidx");
    for (i = 0; i < nold; i++) newidx[i] = -1;
    for (m = 0; m < nmoved; m++)
      if (cells[movedto[m]].nsplit >= 1) newidx[movedfrom[m]] = movedto[m];

    int ntotal = nlocal + nghost;
    for (icell = 0; icell < ntotal; icell++) {
      cellint *neigh = cells[icell].neigh;
      int nmask = cells[icell].nmask;
      for (i = 0; i < 6; i++) {
        int nflag = neigh_decode(nmask,i);
        if (nflag != NCHILD && nflag != NPBCHILD) continue;
        if (neigh[i] >= nold || newidx[neigh[i]] < 0) continue;
        neigh[i] = newidx[neigh[i]];
        if (journalflag) journal_cell(icell);
      }
    }
    memory->destroy(newidx);
    neighscan = 0;
  }

  memory->destroy(holes);
  memory->destroy(freeslots);
}

/* ----------------------------------------------------------------------
   create sub cell I of split cell icell in slot isub, from the cell
     itself as add_sub_cell() does; isplit of the split cell is set
------------------------------------------------------------------------- */

void Grid::add_sub_cell_at(int icell, int isub, int i)
{
  memcpy(&cells[isub],&cells[icell],sizeof(ChildCell));
  cells[isub].ccoll = NULL;
  cells[isub].ilocal = isub;
  cells[isub].nsplit = -i;
  memcpy(&cinfo[isub],&cinfo[icell],sizeof(ChildInfo));
  cinfo[isub].count = 0;
  cinfo[isub].first = -1;
  if (ncustom) copy_custom(icell,isub);
  nsublocal++;
}

/* ----------------------------------------------------------------------
   move cell src into slot dst and repair every reference to it: its
     sinfo entry, the hash, the halo index, the cell bins, the neighbor
     links of its neighbors, its particles and the per-cell data of
     collide and the per-grid fixes
   src is an owned or ghost cell, dst a slot no live cell occupies
   the neighbors' links are repaired through the cell's own links, which
     is exact when every cell is at the same level; otherwise a finer
     neighbor referencing this cell is not reachable from it, and
     restructure_split_cells() repairs the links with a scan
------------------------------------------------------------------------- */

void Grid::move_cell(int src, int dst)
{
  int i,j,n;

  int ownflag = (cells[src].proc == me);
  int nsplit = cells[src].nsplit;

  memcpy(&cells[dst],&cells[src],sizeof(ChildCell));
  if (ownflag) {
    memcpy(&cinfo[dst],&cinfo[src],sizeof(ChildInfo));
    cells[dst].ilocal = dst;
    if (collide) collide->copy_grid_one(src,dst);
    if (modify->n_pergrid) modify->copy_grid_one(src,dst);
  }
  if (ncustom) copy_custom(src,dst);
  if (journalflag) journal_cell(dst);

  if (nsplit > 1) sinfo[cells[dst].isplit].icell = dst;
  else if (nsplit <= 0) sinfo[cells[dst].isplit].csubs[-nsplit] = dst;
  if (journalflag && nsplit != 1) journal_sinfo(cells[dst].isplit);

  // a sub cell is referenced only by its split cell and its particles

  if (nsplit >= 1) {
    if (hashfilled) (*hash)[cells[dst].id] = dst;
    if (halo_index) {
      int site = halo_site(dst);
      if (site >= 0) halo_index[site] = dst;
    }
    if (cellbinvalid) rebin_cell(src,dst);

    cellint *neigh = cells[dst].neigh;
    int nmask = cells[dst].nmask;
    for (i = 0; i < 6; i++) {
      int nflag = neigh_decode(nmask,i);
      if (nflag != NCHILD && nflag != NPBCHILD) continue;
      if (neigh[i] == src) neigh[i] = dst;
      n = neigh[i];
      j = i ^ 1;
      int jflag = neigh_decode(cells[n].nmask,j);
      if ((jflag == NCHILD || jflag == NPBCHILD) && cells[n].neigh[j] == src) {
        cells[n].neigh[j] = dst;
        if (journalflag) journal_cell(n);
      } else if (!uniform) neighscan = 1;
    }
    if (!uniform) neighscan = 1;
  }

  if (ownflag && particle->sorted) {
    Particle::OnePart *particles = particle->particles;
    int *next = particle->next;
    int ip = cinfo[dst].first;
    while (ip >= 0) {
      particles[ip].icell = dst;
      ip = next[ip];
    }
  }

  if (nmoved == maxmoved) {
    maxmoved += DELTA_MOVED;
    memory->grow(movedfrom,maxmoved,"grid:movedfrom");
    memory->grow(movedto,maxmoved,"grid:movedto");
  }
  movedfrom[nmoved] = src;
  movedto[nmoved] = dst;
  nmoved++;
}

/* ----------------------------------------------------------------------
   move sinfo entry src into slot dst and re-point its split cell and
     sub cells at it; a dropped ghost split cell no longer points at
     its entry and is left alone
------------------------------------------------------------------------- */

void Grid::move_sinfo(int src, int dst)
{
  memcpy(&sinfo[dst],&sinfo[src],sizeof(SplitInfo));
  if (journalflag) journal_sinfo(dst);
  int icell = sinfo[dst].icell;
  if (icell < 0) return;
  if (cells[icell].isplit != src) return;
  cells[icell].isplit = dst;
  if (journalflag) journal_cell(icell);
  int *mycsubs = sinfo[dst].csubs;
  for (int i = 0; i < cells[icell].nsplit; i++) {
    cells[mycsubs[i]].isplit = dst;
    if (journalflag) journal_cell(mycsubs[i]);
  }
}

/* ----------------------------------------------------------------------
   change journal, see grid.h
   a cut record holds a copy of the list; a collision record likewise,
     in its own buffer which reset_collision_surfs() empties
------------------------------------------------------------------------- */

void Grid::journal_cell(int icell)
{
  if (!journalcells) return;
  if (ndirtycell == maxdirtycell) {
    maxdirtycell += DELTA_JOURNAL;
    memory->grow(dirtycell,maxdirtycell,"grid:dirtycell");
  }
  dirtycell[ndirtycell++] = icell;
}

void Grid::journal_sinfo(int isplit)
{
  if (ndirtysinfo == maxdirtysinfo) {
    maxdirtysinfo += DELTA_JOURNAL;
    memory->grow(dirtysinfo,maxdirtysinfo,"grid:dirtysinfo");
  }
  dirtysinfo[ndirtysinfo++] = isplit;
}

void Grid::journal_list(int icell, int n, surfint *list, int collflag)
{
  int *nrec,*maxrec;
  ListRecord **rec;
  int **buf;
  bigint *nbuf,*maxbuf;

  if (collflag) {
    nrec = &ncollrec; maxrec = &maxcollrec; rec = &collrec;
    buf = &collbuf; nbuf = &ncollbuf; maxbuf = &maxcollbuf;
  } else {
    nrec = &ncutrec; maxrec = &maxcutrec; rec = &cutrec;
    buf = &cutbuf; nbuf = &ncutbuf; maxbuf = &maxcutbuf;
  }

  if (*nrec == *maxrec) {
    *maxrec += DELTA_JOURNAL;
    *rec = (ListRecord *)
      memory->srealloc(*rec,(*maxrec)*sizeof(ListRecord),"grid:listrec");
  }
  if (*nbuf + n > *maxbuf) {
    while (*nbuf + n > *maxbuf) *maxbuf += DELTA_JOURNAL;
    memory->grow(*buf,*maxbuf,"grid:listbuf");
  }

  ListRecord *r = &(*rec)[(*nrec)++];
  r->icell = icell;
  r->n = n;
  r->offset = *nbuf;
  int *ptr = &(*buf)[*nbuf];
  for (int i = 0; i < n; i++) ptr[i] = (int) list[i];
  *nbuf += n;
}

void Grid::journal_clear()
{
  ndirtycell = ndirtysinfo = 0;
  ncutrec = ncutbuf = 0;
  ncollrec = ncollbuf = 0;
  collreset = 0;
  nmoved = 0;
  nbinpatch = 0;
}

/* ----------------------------------------------------------------------
   site of cell icell in halo_index, -1 if it has none
   the same placement as update_halo_index()
------------------------------------------------------------------------- */

int Grid::halo_site(int icell)
{
  const int un[3] = {unx,uny,unz};
  const int lo[3] = {halo_ilo,halo_jlo,halo_klo};
  const int nh[3] = {halo_nx,halo_ny,halo_nz};
  int l[3];

  for (int d = 0; d < 3; d++) {
    double inv = un[d]/(domain->boxhi[d]-domain->boxlo[d]);
    l[d] = static_cast<int> ((cells[icell].lo[d]-domain->boxlo[d])*inv+0.5)
      - lo[d];
    if (l[d] < 0) l[d] += un[d];
    if (l[d] >= nh[d]) return -1;
  }
  return (l[2]*halo_ny + l[1])*halo_nx + l[0];
}

/* ----------------------------------------------------------------------
   cell src moved to slot dst: replace it in every bin of the bin index
------------------------------------------------------------------------- */

void Grid::rebin_cell(int src, int dst)
{
  int k,ibx,iby,ibz,ibin;
  int lo[3],hi[3];

  for (k = 0; k < 3; k++) {
    lo[k] = (int) ((cells[dst].lo[k]-cellbinlo[k]) * cellbininv[k]);
    hi[k] = (int) ((cells[dst].hi[k]-cellbinlo[k]) * cellbininv[k]);
    lo[k] = MAX(0,MIN(lo[k],cellnbin[k]-1));
    hi[k] = MAX(0,MIN(hi[k],cellnbin[k]-1));
  }

  for (ibz = lo[2]; ibz <= hi[2]; ibz++)
    for (iby = lo[1]; iby <= hi[1]; iby++)
      for (ibx = lo[0]; ibx <= hi[0]; ibx++) {
        ibin = (ibz*cellnbin[1] + iby)*cellnbin[0] + ibx;
        for (int i = cellbinstart[ibin]; i < cellbinstart[ibin+1]; i++)
          if (cellbinlist[i] == src) cellbinlist[i] = dst;
      }

  if (journalflag) {
    if (nbinpatch == maxbinpatch) {
      maxbinpatch += DELTA_MOVED;
      memory->grow(binpatchfrom,maxbinpatch,"grid:binpatchfrom");
      memory->grow(binpatchto,maxbinpatch,"grid:binpatchto");
    }
    binpatchfrom[nbinpatch] = src;
    binpatchto[nbinpatch] = dst;
    nbinpatch++;
  }
}

/* ----------------------------------------------------------------------
   route every ghost sub cell to its split cell for particle migration,
     for a caller which sets subroute after the ghosts were acquired
------------------------------------------------------------------------- */

void Grid::route_ghost_subcells()
{
  for (int isplit = nsplitlocal; isplit < nsplitlocal+nsplitghost; isplit++) {
    int icell = sinfo[isplit].icell;
    if (icell < 0 || cells[icell].isplit != isplit) continue;
    int *mycsubs = sinfo[isplit].csubs;
    for (int i = 0; i < cells[icell].nsplit; i++) {
      cells[mycsubs[i]].ilocal = cells[icell].ilocal;
      if (journalflag) journal_cell(mycsubs[i]);
    }
  }
}

/* ----------------------------------------------------------------------
   remove all surf info from owned grid cells and reset cell volumes
   also remove sub cells by compressing grid cells list
   set cell type and corner flags to UNKNOWN or OUTSIDE
   called before reassigning surfs to grid cells
   changes cells data structure since sub cells are removed
   if particles exist, reassign them to new cells
------------------------------------------------------------------------- */

void Grid::clear_surf()
{
  int dimension = domain->dimension;
  int ncorner = 8;
  if (dimension == 2) ncorner = 4;
  double *lo,*hi;

  hashfilled = 0;
  hashcurrent = 0;
  clear_cell_bins();

  // if surfs no longer exist, set cell type to OUTSIDE, else UNKNOWN
  // set corner points of every cell to UNKNOWN

  int celltype = UNKNOWN;
  if (!surf->exist) celltype = OUTSIDE;

  // compress cell list
  // collide and fixes also need to do the same

  int nlocal_prev = nlocal;

  int icell = 0;
  while (icell < nlocal) {
    if (cells[icell].nsplit <= 0) {
      if (icell != nlocal-1) {
        memcpy(&cells[icell],&cells[nlocal-1],sizeof(ChildCell));
        memcpy(&cinfo[icell],&cinfo[nlocal-1],sizeof(ChildInfo));
        if (collide && collide->ngroups) collide->copy_grid_one(nlocal-1,icell);
        if (modify->n_pergrid) modify->copy_grid_one(nlocal-1,icell);
      }
      nlocal--;
    } else {
      cells[icell].ilocal = icell;
      cells[icell].nsurf = 0;
      cells[icell].csurfs = NULL;
      cells[icell].nsplit = 1;
      cells[icell].isplit = -1;
      cinfo[icell].type = celltype;
      for (int m = 0; m < ncorner; m++) cinfo[icell].corner[m] = UNKNOWN;
      lo = cells[icell].lo;
      hi = cells[icell].hi;
      if (dimension == 3)
        cinfo[icell].volume = (hi[0]-lo[0]) * (hi[1]-lo[1]) * (hi[2]-lo[2]);
      else if (domain->axisymmetric)
        cinfo[icell].volume = MY_PI * (hi[1]*hi[1]-lo[1]*lo[1]) * (hi[0]-lo[0]);
      else
        cinfo[icell].volume = (hi[0]-lo[0]) * (hi[1]-lo[1]);
      icell++;
    }
  }

  // reset final grid cell count in collide and fixes

  if (collide) collide->reset_grid_count(nlocal);
  if (modify->n_pergrid) modify->reset_grid_count(nlocal);

  // if particles exist and local cell count changed
  // repoint particles to new icell indices
  // assumes particles are sorted,
  //   so count/first values in compressed cinfo data struct are still valid
  // when done, particles are still sorted

  if (particle->exist && nlocal < nlocal_prev) {
    Particle::OnePart *particles = particle->particles;
    int *next = particle->next;

    int ip;
    for (int icell = 0; icell < nlocal; icell++) {
      ip = cinfo[icell].first;
      while (ip >= 0) {
        particles[ip].icell = icell;
        ip = next[ip];
      }
    }
  }

  // reset csurfs and csplits and csubs so can refill

  csurfs->reset();
  csplits->reset();
  csubs->reset();

  // reset all cell counters

  nunsplitlocal = nlocal;
  nsplitlocal = nsublocal = 0;
}

/* ----------------------------------------------------------------------
   remove effects of implicit surf split cells on grid and particles
   remove sub-cells from grid list and set all cells to unsplit
   reassign particles in sub-cells to parent split cell
   allows a subsequent read_isurf command to work correctly
   called by read_restart after reading restart file for simulation
     which used implicit surfs and thus may have had split cells
------------------------------------------------------------------------- */

void Grid::clear_surf_implicit()
{
  int icell;
  double *lo,*hi;

  int dimension = domain->dimension;
  int ncorner = 8;
  if (dimension == 2) ncorner = 4;

  // store cellIDs = list of cellID each particle is in
  // sub-cells and parent split cell have the same cellID

  Particle::OnePart *particles = particle->particles;
  int nplocal = particle->nlocal;

  cellint *cellIDs;
  memory->create(cellIDs,nplocal,"grid:cellIDs");

  for (int i = 0; i < nplocal; i++)
    cellIDs[i] = cells[particles[i].icell].id;

  // compress cell list to remove all sub-cells
  // reset cell and corner types, and cell volume

  int celltype = OUTSIDE;

  icell = 0;
  while (icell < nlocal) {
    if (cells[icell].nsplit <= 0) {
      if (icell != nlocal-1) {
        memcpy(&cells[icell],&cells[nlocal-1],sizeof(ChildCell));
        memcpy(&cinfo[icell],&cinfo[nlocal-1],sizeof(ChildInfo));
      }
      nlocal--;
    } else {
      cells[icell].ilocal = icell;
      cells[icell].nsurf = 0;
      cells[icell].csurfs = NULL;
      cells[icell].nsplit = 1;
      cells[icell].isplit = -1;
      cinfo[icell].type = celltype;
      for (int m = 0; m < ncorner; m++) cinfo[icell].corner[m] = UNKNOWN;
      lo = cells[icell].lo;
      hi = cells[icell].hi;
      if (dimension == 3)
        cinfo[icell].volume = (hi[0]-lo[0]) * (hi[1]-lo[1]) * (hi[2]-lo[2]);
      else if (domain->axisymmetric)
        cinfo[icell].volume = MY_PI * (hi[1]*hi[1]-lo[1]*lo[1]) * (hi[0]-lo[0]);
      else
        cinfo[icell].volume = (hi[0]-lo[0]) * (hi[1]-lo[1]);
      icell++;
    }
  }

  // reset all cell counters

  nunsplitlocal = nlocal;
  nsplitlocal = nsublocal = 0;

  // create hash for new set of owned grid cells

  hash->clear();

  for (icell = 0; icell < nlocal; icell++)
    (*hash)[cells[icell].id] = icell;

  // use hash to reassign each particle's index into cell list
  // uses the temporary cellIDs list created above

  for (int i = 0; i < nplocal; i++)
    particles[i].icell = (*hash)[cellIDs[i]];

  // clean up

  memory->destroy(cellIDs);
  hash->clear();
  hashfilled = 0;
  hashcurrent = 0;
}

/* ----------------------------------------------------------------------
   remove all surf info from owned grid cells and reset cell volumes
   do NOT remove sub cells by compressing grid cells list
   called from read_restart before reassigning surfs to grid cells
   sub cells already exist from restart file
------------------------------------------------------------------------- */

void Grid::clear_surf_restart()
{
  // reset current grid cells as if no surfs existed
  // just skip sub cells
  // set values in cells/cinfo as if no surfaces, including volume

  int dimension = domain->dimension;
  int ncorner = 8;
  if (dimension == 2) ncorner = 4;
  double *lo,*hi;

  for (int icell = 0; icell < nlocal; icell++) {
    if (cells[icell].nsplit <= 0) continue;
    cinfo[icell].type = UNKNOWN;
    for (int m = 0; m < ncorner; m++) cinfo[icell].corner[m] = UNKNOWN;
    lo = cells[icell].lo;
    hi = cells[icell].hi;
    if (dimension == 3)
      cinfo[icell].volume = (hi[0]-lo[0]) * (hi[1]-lo[1]) * (hi[2]-lo[2]);
    else if (domain->axisymmetric)
      cinfo[icell].volume = MY_PI * (hi[1]*hi[1]-lo[1]*lo[1]) * (hi[0]-lo[0]);
    else
      cinfo[icell].volume = (hi[0]-lo[0]) * (hi[1]-lo[1]);
  }
}

/* ----------------------------------------------------------------------
   combine all particles in sub cells of a split icell to be in split cell
   assumes particles are sorted, returns them sorted in icell
   if relabel = 1, also change icell value for each particle, else do not
------------------------------------------------------------------------- */

void Grid::combine_split_cell_particles(int icell, int relabel)
{
  int ip,iplast,jcell;

  int nsplit = cells[icell].nsplit;
  int *mycsubs = sinfo[cells[icell].isplit].csubs;
  int count = 0;
  int first = -1;

  int *next = particle->next;

  for (int i = 0; i < nsplit; i++) {
    jcell = mycsubs[i];
    count += cinfo[jcell].count;
    if (cinfo[jcell].first < 0) continue;

    if (first < 0) first = cinfo[jcell].first;
    else next[iplast] = cinfo[jcell].first;

    ip = cinfo[jcell].first;
    while (ip >= 0) {
      iplast = ip;
      ip = next[ip];
    }
  }

  cinfo[icell].count = count;
  cinfo[icell].first = first;

  // repoint each particle now in parent split cell to the split cell

  if (relabel) {
    Particle::OnePart *particles = particle->particles;
    ip = first;
    while (ip >= 0) {
      particles[ip].icell = icell;
      ip = next[ip];
    }
  }
}

/* ----------------------------------------------------------------------
   the owned split cells, in ascending cell index
   sinfo holds the owned split cells before the ghost ones, so the list
     comes from it in O(nsplit) instead of a scan of every owned cell;
     an entry a restructure abandoned points at no split cell of its own
   returns the count, the list is a Grid buffer valid until the next call
------------------------------------------------------------------------- */

int Grid::owned_split_cells(int *&list)
{
  if (nsplitlocal > maxsplitlist) {
    maxsplitlist = nsplitlocal;
    memory->destroy(splitlist);
    memory->create(splitlist,maxsplitlist,"grid:splitlist");
  }

  int n = 0;
  for (int i = 0; i < nsplitlocal; i++) {
    int icell = sinfo[i].icell;
    if (icell < 0 || cells[icell].isplit != i || cells[icell].nsplit <= 1)
      continue;
    splitlist[n++] = icell;
  }

  // the callers walked the cells in index order, so the list does too

  std::sort(splitlist,splitlist+n);

  list = splitlist;
  return n;
}

/* ----------------------------------------------------------------------
   assign all particles in a split icell to appropriate sub cells
   assumes particles are sorted, are NOT sorted by sub cell when done
   also change particle icell label
------------------------------------------------------------------------- */

void Grid::assign_split_cell_particles(int icell)
{
  int ip,jcell;

  int dim = domain->dimension;
  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;

  ip = cinfo[icell].first;
  while (ip >= 0) {
    if (dim == 3) jcell = update->split3d(icell,particles[ip].x);
    else jcell = update->split2d(icell,particles[ip].x);
    particles[ip].icell = jcell;
    ip = next[ip];
  }

  cinfo[icell].count = 0;
  cinfo[icell].first = -1;
}

/* ----------------------------------------------------------------------
   return a point X in icell that is in the flow (outside of all surfs)
   do NOT call for a cell with no surfs
------------------------------------------------------------------------- */

int Grid::point_outside_surfs(int icell, double *x)
{
  if (surf->implicit)
    return point_outside_surfs_implicit(icell,x);
  else
    return point_outside_surfs_explicit(icell,x);
}

/* ----------------------------------------------------------------------
   variant for implicit surfs
   return a point X in icell that is in the flow (outside of all surfs)
   set X to midpt of first line (2d) or center pt of first triangle (3d)
   for implicit surfs this should be a pt in or on ICELL
   displace X by EPSSURF in the line/tri norm direction
   without EPSSURF displacement, test via outside_surfs() for
     a particle which is inside surfs  may not intersect due to round-off
------------------------------------------------------------------------- */

int Grid::point_outside_surfs_implicit(int icell, double *x)
{
  int dim = domain->dimension;
  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;
  surfint *csurfs = cells[icell].csurfs;

  double edge[3];
  double edgelen,minedge,displace;

  int isurf = csurfs[0];

  if (dim == 2) {
    x[0] = 0.5 * (lines[isurf].p1[0] + lines[isurf].p2[0]);
    x[1] = 0.5 * (lines[isurf].p1[1] + lines[isurf].p2[1]);
    x[2] = 0.0;

    MathExtra::sub3(lines[isurf].p1,lines[isurf].p2,edge);
    minedge = MathExtra::len3(edge);

    displace = EPSSURF * minedge;
    x[0] += displace*lines[isurf].norm[0];
    x[1] += displace*lines[isurf].norm[1];

  } else {
    double onethird = 1.0/3.0;
    x[0] = onethird *
      (tris[isurf].p1[0] + tris[isurf].p2[0] + tris[isurf].p3[0]);
    x[1] = onethird *
      (tris[isurf].p1[1] + tris[isurf].p2[1] + tris[isurf].p3[1]);
    x[2] = onethird *
      (tris[isurf].p1[2] + tris[isurf].p2[2] + tris[isurf].p3[2]);

    MathExtra::sub3(tris[isurf].p1,tris[isurf].p2,edge);
    edgelen = MathExtra::len3(edge);
    minedge = edgelen;
    MathExtra::sub3(tris[isurf].p2,tris[isurf].p3,edge);
    edgelen = MathExtra::len3(edge);
    minedge = MIN(minedge,edgelen);
    MathExtra::sub3(tris[isurf].p3,tris[isurf].p1,edge);
    edgelen = MathExtra::len3(edge);
    minedge = MIN(minedge,edgelen);

    displace = EPSSURF * minedge;
    x[0] += displace*tris[isurf].norm[0];
    x[1] += displace*tris[isurf].norm[1];
    x[2] += displace*tris[isurf].norm[2];
  }

  // robustness pass: X was pushed off only the first surf (csurfs[0]) in the
  //  cell, so for acute/spiky surface features it can land inside a
  //  neighboring surf; push it back outside every surf in the cell
  //  (no-op when X is already outside all of them)

  push_reference_outside_surfs(icell,x,displace);

  return 1; // implicit surfs always have a valid flow region
}

/* ----------------------------------------------------------------------
   ensure reference point X (already in or near cell ICELL) is in the flow,
     i.e. outside every non-transparent surf in the cell, not just the single
     surf it was pushed off of by the caller
   X is initially set by point_outside_surfs_explicit/implicit() by displacing
     the centroid of one surf in the cell by DISPLACE along that surf's
     outward normal
   for acute (spiky) surface features, where two neighboring surfs meet at a
     dihedral angle < 90 degrees, that single push can leave X slightly INSIDE
     a neighboring surf, i.e. still inside the surface rather than in the flow
   outside_surfs() then uses X as an in-flow reference point and can misclassify
     particles as being in the flow when they are not, placing particles inside
     the surface (manifests as fix grid/check errors on interior cells)
   fix: iteratively push X back to be at least DISPLACE outside every surf in
     the cell whose face X has intruded behind, until none remain (or a max
     iteration count is reached for pathological geometry)
   only surfs whose nearest feature to X lies within a few DISPLACE can
     constrain X, so far-away surfs never spuriously move it
   this is a no-op when X is already outside all surfs in the cell, so it does
     not change the reference point for the common (non-spiky) case
------------------------------------------------------------------------- */

void Grid::push_reference_outside_surfs(int icell, double *x, double displace)
{
  int dim = domain->dimension;
  int nsurf = cells[icell].nsurf;
  surfint *csurfs = cells[icell].csurfs;

  double band = 10.0*displace;
  double band2 = band*band;
  double sdist,d2,push;

  if (dim == 2) {
    Surf::Line *lines = surf->lines;
    for (int iter = 0; iter < 50; iter++) {
      int moved = 0;
      for (int i = 0; i < nsurf; i++) {
        Surf::Line *ln = &lines[csurfs[i]];
        if (ln->transparent) continue;
        d2 = Geometry::distsq_point_line(x,ln->p1,ln->p2);
        if (d2 > band2) continue;
        sdist = (x[0]-ln->p1[0])*ln->norm[0] + (x[1]-ln->p1[1])*ln->norm[1];
        if (sdist < displace) {
          push = displace - sdist;
          x[0] += push*ln->norm[0];
          x[1] += push*ln->norm[1];
          moved = 1;
        }
      }
      if (!moved) break;
    }
  } else {
    Surf::Tri *tris = surf->tris;
    for (int iter = 0; iter < 50; iter++) {
      int moved = 0;
      for (int i = 0; i < nsurf; i++) {
        Surf::Tri *tr = &tris[csurfs[i]];
        if (tr->transparent) continue;
        d2 = Geometry::distsq_point_tri(x,tr->p1,tr->p2,tr->p3,tr->norm);
        if (d2 > band2) continue;
        sdist = (x[0]-tr->p1[0])*tr->norm[0] + (x[1]-tr->p1[1])*tr->norm[1] +
                (x[2]-tr->p1[2])*tr->norm[2];
        if (sdist < displace) {
          push = displace - sdist;
          x[0] += push*tr->norm[0];
          x[1] += push*tr->norm[1];
          x[2] += push*tr->norm[2];
          moved = 1;
        }
      }
      if (!moved) break;
    }
  }
}

/* ----------------------------------------------------------------------
   variant for explicit surfs
   return a point X in icell that is in the flow (outside of all surfs)
   loop over surfs:
     clip surf to grid cell via Cut::clip_external()
     check that number of clipped points is not a single point or tri edge
     check that line/tri does not just graze cell edge or face
     graze = all clipped points on same edge/face and outward normal
     set X to midpt of clipped line (2d_ or center pt of first 3 clip pts (3d)
     for implicit surfs this should be a pt in or on ICELL
   displace X by EPSSURF in the line/tri norm direction
   without EPSSURF displacement, test via outside_surfs() for
     a particle which is inside surfs  may not intersect due to round-off
------------------------------------------------------------------------- */

int Grid::point_outside_surfs_explicit(int icell, double *x)
{
  int dim = domain->dimension;
  if (dim == 3 && !cut3d) cut3d = new Cut3d(sparta);
  else if (dim == 2 && !cut2d) cut2d = new Cut2d(sparta,domain->axisymmetric);

  double *lo = cells[icell].lo;
  double *hi = cells[icell].hi;
  surfint *csurfs = cells[icell].csurfs;
  int nsurf = cells[icell].nsurf;

  double minsize = MIN(hi[0]-lo[0],hi[1]-lo[1]);
  double displace = EPSSURF * minsize;

  double maxlength = 0.0;
  double maxarea = 0.0;
  int setflag = 0;

  if (dim == 2) {
    int npoint;
    double cpath[4];
    double *norm;
    Surf::Line *line;

    Surf::Line *lines = surf->lines;

    for (int i = 0; i < nsurf; i++) {
      line = &lines[csurfs[i]];
      if (line->transparent) continue;
      norm = line->norm;

      npoint = cut2d->clip_external(line->p1,line->p2,lo,hi,cpath);
      if (npoint < 2) continue;

      int edge = cut2d->sameedge_external(&cpath[0],&cpath[2],lo,hi);
      if (edge) {
        if (edge == 1 and norm[0] < 0.0) continue;
        if (edge == 2 and norm[0] > 0.0) continue;
        if (edge == 3 and norm[1] < 0.0) continue;
        if (edge == 4 and norm[1] > 0.0) continue;
      }

      // for surfaces with a tiny intersection, the point to push off
      //  from can be very close to another line
      // if the angle between the two lines is less than 90 degrees,
      //  the pushed-off point can end up "inside" the surface instead
      //  of "outside"
      // using the line with the largest clipped length makes this
      //  issue very unlikely to occur

      double x1 = cpath[2] - cpath[0];
      double y1 = cpath[3] - cpath[1];
      double length = sqrt(x1*x1 + y1*y1);
      if (length < maxlength) continue;
      maxlength = length;

      x[0] = 0.5*(cpath[0]+cpath[2]) + displace*norm[0];
      x[1] = 0.5*(cpath[1]+cpath[3]) + displace*norm[1];
      x[2] = 0.0;
      setflag = 1;
    }

  } else {
    int npoint;
    double cpath[24];
    double *norm;
    Surf::Tri *tri;

    Surf::Tri *tris = surf->tris;

    for (int i = 0; i < nsurf; i++) {
      tri = &tris[csurfs[i]];
      if (tri->transparent) continue;
      norm = tri->norm;

      npoint = cut3d->clip_external(tri->p1,tri->p2,tri->p3,lo,hi,cpath);
      if (npoint < 3) continue;

      int face = cut3d->sameface_external(&cpath[0],&cpath[3],&cpath[6],lo,hi);
      if (face) {
        if (face == 1 and norm[0] < 0.0) continue;
        if (face == 2 and norm[0] > 0.0) continue;
        if (face == 3 and norm[1] < 0.0) continue;
        if (face == 4 and norm[1] > 0.0) continue;
        if (face == 5 and norm[2] < 0.0) continue;
        if (face == 6 and norm[2] > 0.0) continue;
      }

      // for surfaces with a tiny intersection, the point to push off
      //  from can be very close to a shared edge with another triangle
      // if the angle between the two tris is less than 90 degrees,
      //  the pushed-off point can end up "inside" the surface instead
      //  of "outside"
      // using the triangle with the largest clipped area
      //  and pushing off its centroid makes this issue very unlikely
      //  to occur

      double center[3];
      double area = Geometry::poly_area(npoint,cpath,center);
      if (area < maxarea) continue;
      maxarea = area;

      x[0] = center[0] + displace*norm[0];
      x[1] = center[1] + displace*norm[1];
      x[2] = center[2] + displace*norm[2];
      setflag = 1;
    }
  }

  // robustness pass: ensure reference point X is truly in the flow
  //  (outside every surf in the cell), not just outside the one surf it
  //  was pushed off of; see push_reference_outside_surfs()

  if (setflag) push_reference_outside_surfs(icell,x,displace);

  // if setflag equal to 0
  //  unable to find a point in flow volume, all surfs invoked "continue"
  //  means entire cell is actually outside or inside, just touched by surfs
  //  if outside, caller does not need to call outside_surfs()
  //  if inside, caller can detect that its flow volume = zero

  return setflag;
}

/* ----------------------------------------------------------------------
   check if particle at X is outside any surfs in icell (in the flow)
   use Xcell as reference point in flow, calculated by point_outside_surfs()
   do NOT call for a cell with no surfs
   if outside return 1, else return 0
------------------------------------------------------------------------- */

int Grid::outside_surfs(int icell, double *x, double *xcell)
{
  int dim = domain->dimension;
  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;
  surfint *csurfs = cells[icell].csurfs;

  int m,isurf,hitflag,side;
  double param;
  double xc[3];
  Surf::Line *line;
  Surf::Tri *tri;

  // loop over surfs, ray-trace from x to xcell, see how many surfaces were hit

  int nsurf = cells[icell].nsurf;

  int cnt = 0;
  for (m = 0; m < nsurf; m++) {
    isurf = csurfs[m];
    if (dim == 3) {
      tri = &tris[isurf];
      hitflag = Geometry::
        line_tri_intersect_noeps(x,xcell,tri->p1,tri->p2,tri->p3,
                                 tri->norm,xc,param,side);
    } else {
      line = &lines[isurf];
      hitflag = Geometry::
        line_line_intersect(x,xcell,line->p1,line->p2,line->norm,xc,param,side);
    }
    if (hitflag) cnt++;
  }

  // if no surf was hit, particle is outside surfs
  // else check how many surfaces were hit
  //   odd number = inside, even number = outside

  if (cnt == 0) return 1;
  if (cnt % 2 == 0) return 1;

  return 0;
}

/* ----------------------------------------------------------------------
   re-allocate page data structs to hold variable-length surf and cell info
   used for mapping surfaces to grid cells
------------------------------------------------------------------------- */

void Grid::allocate_surf_arrays()
{
  delete csurfs;
  delete csplits;
  delete csubs;

  csurfs = new MyPage<surfint>(maxsurfpercell,MAX(100*maxsurfpercell,1024));
  csplits = new MyPage<int>(maxsurfpercell,MAX(100*maxsurfpercell,1024));
  csubs = new MyPage<int>(maxsplitpercell,MAX(100*maxsplitpercell,128));

  surflist_churn = 0;
}

/* ----------------------------------------------------------------------
   flow volume of a cell no surf cuts, from its corner points
------------------------------------------------------------------------- */

double Grid::cell_volume(double *lo, double *hi)
{
  if (domain->dimension == 3)
    return (hi[0]-lo[0]) * (hi[1]-lo[1]) * (hi[2]-lo[2]);
  if (domain->axisymmetric)
    return MY_PI * (hi[1]*hi[1]-lo[1]*lo[1]) * (hi[0]-lo[0]);
  return (hi[0]-lo[0]) * (hi[1]-lo[1]);
}

/* ----------------------------------------------------------------------
   add to the tiny-edge and shrink counts of the 3d cut, for cells a
     caller cut elsewhere (fix rigid/kk cuts them on the device)
------------------------------------------------------------------------- */

void Grid::add_cut3d_counts(bigint ntiny, bigint nshrink)
{
  if (!cut3d) return;
  cut3d->ntiny += ntiny;
  cut3d->nshrink += nshrink;
}

/* ----------------------------------------------------------------------
   operations a fix which moves surfs uses to re-map them to grid cells
     one cell at a time, in place of the full surf2grid() pipeline
   every list they install lives in the Grid pages, like the lists the
     pipeline installs, so the cells own their lists and nothing outside
     Grid ever has to free one
   surfs_in_cell() and cut_cell() run the same cut routines the pipeline
     runs, on the cell's own extent and cut list
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   which of the ncand surfs in cand overlap cell icell
   their local indices are returned in list, at most max of them
   returns the full count, which may exceed max: the caller decides
------------------------------------------------------------------------- */

int Grid::surfs_in_cell(int icell, int ncand, surfint *cand,
                        surfint *list, int max)
{
  ChildCell *c = &cells[icell];
  if (domain->dimension == 3)
    return cut3d->surf2grid_list(c->id,c->lo,c->hi,ncand,cand,list,max);
  return cut2d->surf2grid_list(c->id,c->lo,c->hi,ncand,cand,list,max);
}

/* ----------------------------------------------------------------------
   cut cell icell by its current cut list, changing nothing in the cell
   returns the number of flow pieces the surfs divide the cell into
   vols = flow volume of each piece, a buffer the cut routine owns which
     is valid until its next call
   map = piece each surf of the cut list belongs to, -1 if none
   corner = INSIDE/OUTSIDE/UNKNOWN mark of each cell corner point
   xsub,xsplit = reference piece and a point inside it, the start of the
     ray split2d/3d cast to assign a particle to a piece
------------------------------------------------------------------------- */

int Grid::cut_cell(int icell, double *&vols, int *map, int *corner,
                   int &xsub, double *xsplit)
{
  ChildCell *c = &cells[icell];
  if (domain->dimension == 3)
    return cut3d->split(c->id,c->lo,c->hi,c->nsurf,c->csurfs,
                        vols,map,corner,xsub,xsplit);
  return cut2d->split(c->id,c->lo,c->hi,c->nsurf,c->csurfs,
                      vols,map,corner,xsub,xsplit);
}

/* ----------------------------------------------------------------------
   replace the cut list of cell icell by the n surfs in list
   the sub cells of a split cell share its list, so they follow
   the previous list stays in its page: a page never frees a single
     list, compact_surf_lists() reclaims the space once enough has piled up
   the caller sets the cell's type, volume and split info afterwards
------------------------------------------------------------------------- */

void Grid::set_cell_surfs(int icell, int n, surfint *list)
{
  surfint *ptr = NULL;

  if (n) {
    ptr = csurfs->get(n);
    if (!ptr) error->one(FLERR,"Failed to allocate grid cell surf list");
    memcpy(ptr,list,n*sizeof(surfint));
    surflist_churn += n*sizeof(surfint);
  }

  cells[icell].nsurf = n;
  cells[icell].csurfs = ptr;

  if (cells[icell].nsplit > 1) {
    int *mycsubs = sinfo[cells[icell].isplit].csubs;
    for (int i = 0; i < cells[icell].nsplit; i++) {
      cells[mycsubs[i]].nsurf = n;
      cells[mycsubs[i]].csurfs = ptr;
      if (journalflag) journal_cell(mycsubs[i]);
    }
  }

  if (journalflag) {
    journal_cell(icell);
    journal_list(icell,n,list,0);
  }
}

/* ----------------------------------------------------------------------
   type an owned cell no surf cuts as INSIDE or OUTSIDE
   its corner marks follow its type, its flow volume is the full cell
     volume, or zero if INSIDE, as set_inout() assigns them
------------------------------------------------------------------------- */

void Grid::set_cell_type(int icell, int type)
{
  int ncorner = 4;
  if (domain->dimension == 3) ncorner = 8;

  cinfo[icell].type = type;
  for (int i = 0; i < ncorner; i++) cinfo[icell].corner[i] = type;
  if (type == INSIDE) cinfo[icell].volume = 0.0;
  else cinfo[icell].volume = cell_volume(cells[icell].lo,cells[icell].hi);
  if (journalflag) journal_cell(icell);
}

/* ----------------------------------------------------------------------
   type an owned cell a surf cuts as OVERLAP, with the flow volume and
     corner marks its cut produced
   a split cell's own volume is the full cell volume, as the pipeline
     leaves it; the caller passes that, its pieces get theirs from
     set_split_info()
------------------------------------------------------------------------- */

void Grid::set_cell_overlap(int icell, double volume, int *corner)
{
  int ncorner = 4;
  if (domain->dimension == 3) ncorner = 8;

  cinfo[icell].type = OVERLAP;
  for (int i = 0; i < ncorner; i++) cinfo[icell].corner[i] = corner[i];
  cinfo[icell].volume = volume;
  if (journalflag) journal_cell(icell);
}

/* ----------------------------------------------------------------------
   re-split cell icell in place, keeping its sub cells: a new piece map
     for its current cut list, a new reference piece and point, and the
     flow volume of each piece
   map has one entry per surf of the cut list, so it is replaced with it
   a ghost split cell has no ChildInfo, so its piece volumes are skipped
------------------------------------------------------------------------- */

void Grid::set_split_info(int icell, int *map, int xsub, double *xsplit,
                          double *vols)
{
  int isplit = cells[icell].isplit;
  int n = cells[icell].nsurf;

  int *ptr = csplits->get(n);
  if (!ptr) error->one(FLERR,"Failed to allocate grid split cell map");
  memcpy(ptr,map,n*sizeof(int));
  surflist_churn += n*sizeof(int);

  SplitInfo *s = &sinfo[isplit];
  s->csplits = ptr;
  s->xsub = xsub;
  s->xsplit[0] = xsplit[0];
  s->xsplit[1] = xsplit[1];
  if (domain->dimension == 3) s->xsplit[2] = xsplit[2];
  else s->xsplit[2] = 0.0;

  if (journalflag) journal_sinfo(isplit);
  if (icell >= nlocal) return;

  int *mycsubs = s->csubs;
  for (int i = 0; i < cells[icell].nsplit; i++) {
    cinfo[mycsubs[i]].volume = vols[i];
    if (journalflag) journal_cell(mycsubs[i]);
  }
}

/* ----------------------------------------------------------------------
   rebuild the pages of cut lists, piece maps and sub cell lists from the
     lists the cells currently hold, so the space of every list which
     set_cell_surfs() or set_split_info() replaced is reclaimed
   a fix which re-cuts cells every step replaces lists every step, and a
     page never frees a single list, so without this the pages would grow
     without bound; same idea as compress()
   only done once the replaced lists outweigh the ones in use, so the
     O(nlocal+nghost) pass is amortized over many steps
   sub cells share the list of their split cell: it is copied once for
     the split cell and the sub cells are re-pointed at the copy
------------------------------------------------------------------------- */

void Grid::compact_surf_lists()
{
  bigint bytes = csurfs->size() + csplits->size();
  if (surflist_churn < bytes/2) return;

  MyPage<surfint> *csurfs_old = csurfs;
  MyPage<int> *csplits_old = csplits;
  MyPage<int> *csubs_old = csubs;

  csurfs = NULL; csplits = NULL; csubs = NULL;
  allocate_surf_arrays();

  int i,n,icell,isplit;
  int *iptr;
  surfint *sptr;

  int ntotal = nlocal + nghost;

  for (icell = 0; icell < ntotal; icell++) {
    if (cells[icell].nsplit <= 0) continue;
    n = cells[icell].nsurf;
    if (n <= 0) continue;
    sptr = csurfs->get(n);
    if (!sptr) error->one(FLERR,"Failed to allocate grid cell surf list");
    memcpy(sptr,cells[icell].csurfs,n*sizeof(surfint));
    cells[icell].csurfs = sptr;
  }

  int nsplitall = nsplitlocal + nsplitghost;

  for (isplit = 0; isplit < nsplitall; isplit++) {
    icell = sinfo[isplit].icell;
    n = cells[icell].nsurf;
    int nsplitone = cells[icell].nsplit;

    iptr = csplits->get(n);
    if (!iptr) error->one(FLERR,"Failed to allocate grid split cell map");
    memcpy(iptr,sinfo[isplit].csplits,n*sizeof(int));
    sinfo[isplit].csplits = iptr;

    iptr = csubs->get(nsplitone);
    if (!iptr) error->one(FLERR,"Failed to allocate grid sub cell list");
    memcpy(iptr,sinfo[isplit].csubs,nsplitone*sizeof(int));
    sinfo[isplit].csubs = iptr;

    for (i = 0; i < nsplitone; i++) {
      cells[iptr[i]].nsurf = n;
      cells[iptr[i]].csurfs = cells[icell].csurfs;
    }
  }

  delete csurfs_old;
  delete csplits_old;
  delete csubs_old;
}

/* ----------------------------------------------------------------------
   per-step collision lists
   a fix which moves surfs adds them to the cells they sweep through
     during a step, so the mover tests particles anywhere on their path
     against them; the cut lists, which define the cell geometry and
     which split2d/3d index in lockstep with the piece maps, stay as they
     are.  the lists are set after the surfs are moved and reset before
     the cells are re-cut
   collision_page() sizes the storage for the longest list a fix will
     set, once per run
------------------------------------------------------------------------- */

void Grid::collision_page(int maxchunk)
{
  if (ccollpage && maxchunk <= ccollmax) return;
  delete ccollpage;
  ccollmax = maxchunk;
  ccollpage = new MyPage<surfint>(maxchunk,MAX(65536,4*maxchunk));
  if (ccollpage->errorflag)
    error->one(FLERR,"Failed to allocate grid collision list page");
}

/* ----------------------------------------------------------------------
   set the collision list of cell icell for this step to the n surfs in list
------------------------------------------------------------------------- */

void Grid::set_collision_surfs(int icell, int n, surfint *list)
{
  surfint *ptr = ccollpage->get(n);
  if (!ptr) error->one(FLERR,"Failed to allocate grid collision list");
  memcpy(ptr,list,n*sizeof(surfint));

  if (!cells[icell].ccoll) {
    if (ncollcells == maxcollcells) {
      maxcollcells += 1024;
      memory->grow(collcells,maxcollcells,"grid:collcells");
    }
    collcells[ncollcells++] = icell;
  }

  cells[icell].ncoll = n;
  cells[icell].ccoll = ptr;
  if (journalflag) journal_list(icell,n,list,1);
}

/* ----------------------------------------------------------------------
   every cell's collision list is its cut list again
------------------------------------------------------------------------- */

void Grid::reset_collision_surfs()
{
  for (int i = 0; i < ncollcells; i++) cells[collcells[i]].ccoll = NULL;
  ncollcells = 0;
  if (ccollpage) ccollpage->reset();
  if (journalflag) {
    ncollrec = 0;
    ncollbuf = 0;
    collreset = 1;
  }
}

/* ----------------------------------------------------------------------
   the ghost surfs were re-packed after the local surf range grew from
     nslocal_old: an entry >= nslocal_old in a ghost cell's cut list is
     the old ghost index, mapped to its new index by gmap
   sub cells share the list of their split cell, so each is visited once
------------------------------------------------------------------------- */

void Grid::reindex_ghost_surfs(int nslocal_old, int *gmap)
{
  int ntotal = nlocal + nghost;

  for (int icell = nlocal; icell < ntotal; icell++) {
    if (cells[icell].nsplit <= 0) continue;
    if (cells[icell].nsurf <= 0) continue;
    surfint *list = cells[icell].csurfs;
    int n = cells[icell].nsurf;
    int changed = 0;
    for (int j = 0; j < n; j++)
      if (list[j] >= nslocal_old) {
        list[j] = gmap[list[j]-nslocal_old];
        changed = 1;
      }
    if (changed && journalflag) journal_list(icell,n,list,0);
  }
}

/* ----------------------------------------------------------------------
   request N-length vector from csubs
   called by ReadRestart
------------------------------------------------------------------------- */

int *Grid::csubs_request(int n)
{
  int *ptr = csubs->vget();
  csubs->vgot(n);
  return ptr;
}

// ----------------------------------------------------------------------
// private class methods
// ----------------------------------------------------------------------

/* ----------------------------------------------------------------------
   comparison function invoked by qsort() called by surf2grid algorithm
   used to sort the csurfs list of a single cell
   this is not a class method
------------------------------------------------------------------------- */

int compare_surfIDs(const void *iptr, const void *jptr)
{
  int i = *((surfint *) iptr);
  int j = *((surfint *) jptr);
  if (i < j) return -1;
  if (i > j) return 1;
  return 0;
}

/* ----------------------------------------------------------------------
   output stats on overlap of surfs with grid cells
------------------------------------------------------------------------- */

void Grid::surf2grid_stats()
{
  double cmax,len;
  int dimension = domain->dimension;

  int scount = 0;
  bigint stotal = 0;
  int smax = 0;
  double sratio = BIG;

  for (int icell = 0; icell < nlocal; icell++) {
    if (cells[icell].nsplit <= 0) continue;
    if (cells[icell].nsurf) scount++;
    stotal += cells[icell].nsurf;
    smax = MAX(smax,cells[icell].nsurf);

    cmax = MAX(cells[icell].hi[0] - cells[icell].lo[0],
               cells[icell].hi[1] - cells[icell].lo[1]);
    if (dimension == 3)
      cmax = MAX(cmax,cells[icell].hi[2] - cells[icell].lo[2]);

    if (dimension == 2) {
      for (int i = 0; i < cells[icell].nsurf; i++) {

        // NOTE: this line is bad for distributed surfs
        //       b/c calling line_size that will access lines
        //       which is not yet initialized
        // maybe should do rendezvous already to get them?

        len = surf->line_size(cells[icell].csurfs[i]);
        sratio = MIN(sratio,len/cmax);
      }
    } else if (dimension == 3) {
      for (int i = 0; i < cells[icell].nsurf; i++) {
        surf->tri_size(cells[icell].csurfs[i],len);
        sratio = MIN(sratio,len/cmax);
      }
    }
  }

  bigint bscount = scount;
  bigint bstotal = stotal;   // stotal itself is bigint, sum can exceed 2^31
  bigint scountall,stotalall;
  int smaxall;
  double sratioall;
  MPI_Allreduce(&bscount,&scountall,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
  MPI_Allreduce(&bstotal,&stotalall,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
  MPI_Allreduce(&smax,&smaxall,1,MPI_INT,MPI_MAX,world);
  MPI_Allreduce(&sratio,&sratioall,1,MPI_DOUBLE,MPI_MIN,world);

  if (comm->me == 0) {
    if (screen) {
      fprintf(screen,"  " BIGINT_FORMAT " = cells with surfs\n",scountall);
      fprintf(screen,"  " BIGINT_FORMAT
              " = total surfs in all grid cells\n",stotalall);
      fprintf(screen,"  %d = max surfs in one grid cell\n",smaxall);
      fprintf(screen,"  %g = min surf-size/cell-size ratio\n",sratioall);
    }
    if (logfile) {
      fprintf(logfile,"  " BIGINT_FORMAT " = cells with surfs\n",scountall);
      fprintf(logfile,"  " BIGINT_FORMAT
              " = total surfs in all grid cells\n",stotalall);
      fprintf(logfile,"  %d = max surfs in one grid cell\n",smaxall);
      fprintf(logfile,"  %g = min surf-size/cell-size ratio\n",sratioall);
    }
  }
}

/* ----------------------------------------------------------------------
   output stats on aggregate flow volume and cells in/out of the flow
------------------------------------------------------------------------- */

void Grid::flow_stats()
{
  int i;

  int outside = 0;
  int inside = 0;
  int overlap = 0;
  int maxsplitone = 0;
  double cellvolume = 0.0;

  for (int icell = 0; icell < nlocal; icell++) {
    if (cells[icell].nsplit <= 0) continue;
    if (cinfo[icell].type == OUTSIDE) outside++;
    else if (cinfo[icell].type == INSIDE) inside++;
    else if (cinfo[icell].type == OVERLAP) overlap++;
    maxsplitone = MAX(maxsplitone,cells[icell].nsplit);
  }

  // sum volume for unsplit and sub cells
  // skip split cells and INSIDE cells

  for (int icell = 0; icell < nlocal; icell++) {
    if (cells[icell].nsplit > 1) continue;
    if (cinfo[icell].type != INSIDE) cellvolume += cinfo[icell].volume;
  }

  // sum cell counts in bigint, global counts can exceed 2^31

  bigint outall,inall,overall;
  int maxsplitall;
  double cellvolumeall;
  bigint one;
  one = outside;
  MPI_Allreduce(&one,&outall,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
  one = inside;
  MPI_Allreduce(&one,&inall,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
  one = overlap;
  MPI_Allreduce(&one,&overall,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
  MPI_Allreduce(&maxsplitone,&maxsplitall,1,MPI_INT,MPI_MAX,world);
  MPI_Allreduce(&cellvolume,&cellvolumeall,1,MPI_DOUBLE,MPI_SUM,world);

  double flowvolume = flow_volume();

  bigint *tally = new bigint[maxsplitall];
  bigint *tallyall = new bigint[maxsplitall];
  for (i = 0; i < maxsplitall; i++) tally[i] = 0;

  for (int icell = 0; icell < nlocal; icell++) {
    if (cells[icell].nsplit <= 0) continue;
    if (cinfo[icell].type == OVERLAP) tally[cells[icell].nsplit-1]++;
  }

  MPI_Allreduce(tally,tallyall,maxsplitall,MPI_SPARTA_BIGINT,MPI_SUM,world);

  if (comm->me == 0) {
    if (screen) {
      fprintf(screen,"  " BIGINT_FORMAT " " BIGINT_FORMAT " " BIGINT_FORMAT
              " = cells outside/inside/overlapping surfs\n",
              outall,inall,overall);
      fprintf(screen," ");
      for (i = 0; i < maxsplitall; i++)
        fprintf(screen," " BIGINT_FORMAT,tallyall[i]);
      fprintf(screen," = surf cells with 1,2,etc splits\n");
      fprintf(screen,"  %.15g %.15g = cell-wise and global flow volume\n",
              cellvolumeall,flowvolume);
    }
    if (logfile) {
      fprintf(logfile,"  " BIGINT_FORMAT " " BIGINT_FORMAT " " BIGINT_FORMAT
              " = cells outside/inside/overlapping surfs\n",
              outall,inall,overall);
      fprintf(logfile," ");
      for (i = 0; i < maxsplitall; i++)
        fprintf(logfile," " BIGINT_FORMAT,tallyall[i]);
      fprintf(logfile," = surf cells with 1,2,etc splits\n");
      fprintf(logfile,"  %g %g = cell-wise and global flow volume\n",
              cellvolumeall,flowvolume);
    }
  }

  delete [] tally;
  delete [] tallyall;
}

/* ----------------------------------------------------------------------
   compute flow volume for entire box, using list of surfs
   volume for one surf is projection to lower z face (3d) or y face (2d)
   skip transparent surfs
   NOTE: this does not work if any surfs are clipped to zlo or zhi faces in 3d
         this does not work if any surfs are clipped to ylo or yhi faces in 3d
         need to add contribution due to closing surfs on those faces
         fairly easy to add in 2d, not so easy in 3d
------------------------------------------------------------------------- */

double Grid::flow_volume()
{
  double zarea,volall;
  double *p1,*p2,*p3;

  int n;
  Surf::Line *lines;
  Surf::Tri *tris;

  if (domain->dimension == 3) {
    if (!surf->implicit && surf->distributed) {
      tris = surf->mytris;
      n = surf->nown;
    } else {
      tris = surf->tris;
      n = surf->nlocal;
    }
  } else {
    if (!surf->implicit && surf->distributed) {
      lines = surf->mylines;
      n = surf->nown;
    } else {
      lines = surf->lines;
      n = surf->nlocal;
    }
  }

  double *boxlo = domain->boxlo;
  double *boxhi = domain->boxhi;

  double volume = 0.0;

  if (domain->dimension == 3) {
    for (int i = 0; i < n; i++) {
      if (tris[i].transparent) continue;
      p1 = tris[i].p1;
      p2 = tris[i].p2;
      p3 = tris[i].p3;
      zarea = 0.5 * ((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]));
      volume -= zarea * ((p1[2]+p2[2]+p3[2])/3.0 - boxlo[2]);
    }

    if (surf->distributed)
      MPI_Allreduce(&volume,&volall,1,MPI_DOUBLE,MPI_SUM,world);
    else volall = volume;

    if (volall <= 0.0)
      volall += (boxhi[0]-boxlo[0]) * (boxhi[1]-boxlo[1]) *
        (boxhi[2]-boxlo[2]);

  // axisymmetric "volume" of line segment = volume of truncated cone
  // PI/3 (y1^2 + y1y2 + y2^2) (x2-x1)

  } else if (domain->axisymmetric) {
    for (int i = 0; i < n; i++) {
      if (lines[i].transparent) continue;
      p1 = lines[i].p1;
      p2 = lines[i].p2;
      volume -=
        MY_PI3 * (p1[1]*p1[1] + p1[1]*p2[1] + p2[1]*p2[1]) * (p2[0]-p1[0]);
    }

    if (surf->distributed)
      MPI_Allreduce(&volume,&volall,1,MPI_DOUBLE,MPI_SUM,world);
    else volall = volume;

    if (volall <= 0.0)
      volall += MY_PI * boxhi[1]*boxhi[1] * (boxhi[0]-boxlo[0]);

  } else {
    for (int i = 0; i < n; i++) {
      if (lines[i].transparent) continue;
      p1 = lines[i].p1;
      p2 = lines[i].p2;
      volume -= (0.5*(p1[1]+p2[1]) - boxlo[1]) * (p2[0]-p1[0]);
    }

    if (surf->distributed)
      MPI_Allreduce(&volume,&volall,1,MPI_DOUBLE,MPI_SUM,world);
    else volall = volume;

    if (volall <= 0.0) volall += (boxhi[0]-boxlo[0]) * (boxhi[1]-boxlo[1]);
  }

  return volall;
}
