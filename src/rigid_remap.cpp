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
#include "string.h"
#include "math.h"
#include <algorithm>
#include "rigid_remap.h"
#include "fix_rigid.h"
#include "update.h"
#include "domain.h"
#include "grid.h"
#include "surf.h"
#include "particle.h"
#include "comm.h"
#include "modify.h"
#include "compute.h"
#include "output.h"
#include "dump.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

#define DELTA 1024

enum{UNKNOWN,OUTSIDE,INSIDE,OVERLAP};   // several files

// local box/box overlap test, touching counts as overlap

static inline int box_overlap(double *alo, double *ahi,
                              double *blo, double *bhi)
{
  if (ahi[0] < blo[0] || alo[0] > bhi[0]) return 0;
  if (ahi[1] < blo[1] || alo[1] > bhi[1]) return 0;
  if (ahi[2] < blo[2] || alo[2] > bhi[2]) return 0;
  return 1;
}

/* ----------------------------------------------------------------------
   re-map of the surfs of rigid bodies to the grid cells, incrementally
   each step the bodies move, so which cells their surfs overlap, the
     flow volume and type of those cells, and how a cell is split into
     disconnected flow pieces all change.  a full re-map of every surf
     to every cell (Grid::surf2grid) and the collective flood fill which
     types the cells (Grid::set_inout) would do it, at a cost of the
     whole grid every step; this class does only the cells near the
     bodies, through the per-cell operations Grid provides
   two passes per step:
   collision_lists(), after the bodies' end-of-step pose is known: add
     every body surf to the collision list of every cell it sweeps
     through during the step, so the mover tests a particle anywhere on
     the body's path against the moving surf and reflects it instead of
     letting the body overtake it.  the cut lists are untouched
   recut(), after the surfs are moved to their end-of-step position:
     re-cut every cell in the region a body occupied before or after its
     move whose surf overlap changed or which holds a moving surf, and
     re-type the cells its interior entered or left by a parity test of
     the cell center against the body surfs.  a cell whose piece count
     changes is queued and restructured by apply_pending()
   ghost cells are left alone: their cut lists are covered by the
     collision lists every step, and a particle migrating into a ghost
     split cell is routed to the split cell itself (Grid::subroute), so
     a ghost copy of a split cell never needs its pieces re-derived
------------------------------------------------------------------------- */

RigidRemap::RigidRemap(SPARTA *sparta, FixRigid *fixrigid) : Pointers(sparta)
{
  fix = fixrigid;
  dim = domain->dimension;

  memory->create(prevlo,fix->nbody,3,"rigid_remap:prevlo");
  memory->create(prevxcm,fix->nbody,3,"rigid_remap:prevxcm");
  memory->create(cominside,fix->nbody,"rigid_remap:cominside");
  memory->create(prevhi,fix->nbody,3,"rigid_remap:prevhi");

  typechanged = listschanged = splitchanged = restructured = 0;
  npending = maxpending = 0;
  pending = NULL;
  maxmap = maxvols = NULL;

  swstamp = swhead = NULL;
  maxswcell = 0;
  swcur = 0;
  swcells = NULL;
  nswcell = maxswcells = 0;
  entnext = NULL;
  entelem = NULL;
  nent = maxent = 0;
  swlist = NULL;
  maxswlist = 0;
  nswept = 0;

  staticinside = NULL;
  staticgen = 0;
  maxstatic = 0;
  staticvalid = 0;
  nrcand = maxrcand = 0;
  rcand = NULL;
  newlist = NULL;
  newmap = NULL;
  maxnewlist = 0;
  reclist = NULL;
  maxreclist = 0;
}

/* ---------------------------------------------------------------------- */

RigidRemap::~RigidRemap()
{
  memory->destroy(prevlo);
  memory->destroy(prevhi);
  memory->destroy(prevxcm);
  memory->destroy(cominside);

  for (int m = 0; m < maxpending; m++) {
    memory->destroy(pending[m].map);
    memory->destroy(pending[m].vols);
  }
  memory->sfree(pending);
  memory->destroy(maxmap);
  memory->destroy(maxvols);

  memory->destroy(swstamp);
  memory->destroy(swhead);
  memory->destroy(swcells);
  memory->destroy(entnext);
  memory->destroy(entelem);
  memory->destroy(swlist);
  memory->destroy(staticinside);
  memory->destroy(rcand);
  memory->destroy(newlist);
  memory->destroy(newmap);
  memory->destroy(reclist);
}

/* ----------------------------------------------------------------------
   called at the start of each run, once the bodies' bounding boxes are
     set and the grid is consistent with the bodies at their positions
   work bufs are sized for the current global surfmax, which can change
     between runs
------------------------------------------------------------------------- */

void RigidRemap::setup()
{
  int nbody = fix->nbody;
  int nsurf = fix->nsurf;

  // a merged collision list can hold one cell's cut surfs plus the
  //   swept surfs of every body

  int maxchunk = grid->maxsurfpercell + nsurf;
  grid->collision_page(maxchunk);
  if (maxchunk > maxswlist) {
    maxswlist = maxchunk;
    memory->destroy(swlist);
    memory->create(swlist,maxswlist,"rigid_remap:swlist");
  }

  // newlist/newmap = the cell's new surf list and its piece map, both
  //   capped at maxsurfpercell by the cut routines
  // reclist = candidate surfs, the cell's current static surfs plus
  //   every element of every body

  if (grid->maxsurfpercell > maxnewlist) {
    maxnewlist = grid->maxsurfpercell;
    memory->destroy(newlist);
    memory->destroy(newmap);
    memory->create(newlist,maxnewlist,"rigid_remap:newlist");
    memory->create(newmap,maxnewlist,"rigid_remap:newmap");
  }

  int n = grid->maxsurfpercell + nsurf;
  if (n > maxreclist) {
    maxreclist = n;
    memory->destroy(reclist);
    memory->create(reclist,maxreclist,"rigid_remap:reclist");
  }

  // the grid is now consistent with the bodies at their current positions

  for (int ibody = 0; ibody < nbody; ibody++)
    for (int j = 0; j < 3; j++) {
      prevlo[ibody][j] = fix->bbodylo[ibody][j];
      prevhi[ibody][j] = fix->bbodyhi[ibody][j];
      prevxcm[ibody][j] = fix->xcm[ibody][j];
    }

  // whether each body's COM is interior to it, a property of its shape

  for (int ibody = 0; ibody < nbody; ibody++)
    cominside[ibody] = fix->inside_body(ibody,fix->xcm[ibody]);
}

/* ----------------------------------------------------------------------
   add the surfs of every body this proc holds to the collision lists
     of all grid cells they sweep through during this step, so
     particles anywhere in a body's swept path are tested against the
     moving surfs and reflected (rather than overtaken by a fast body
     and deleted)
   called each start_of_step, after the end-of-step pose of every body
     is known; a single pass over the grid cells handles all bodies
   for each overlapped cell the collision list = its cut list plus all
     swept surfs not already present; split cells share the list with
     their sub cells, where particles reside
   cut-cell volumes are not changed: this augments collision lists only
------------------------------------------------------------------------- */

void RigidRemap::collision_lists()
{
  int i,j,ibody,icell,isub,ncur,isplit,dup,nmerged;
  surfint *merged,*cur;

  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;
  int ntotal = grid->nlocal + grid->nghost;

  int nblist = fix->nblist;
  int *blist = fix->blist;
  int *bodystart = fix->bodystart;
  int *lblist = fix->lblist;
  double **elemlo = fix->elemlo;
  double **elemhi = fix->elemhi;

  // per-element swept bounding boxes of every body were set by
  //   FixRigid::body_bbox(ibody,1) in start_of_step()

  nswept = 0;

  // phase 1: gather (cell, swept element) entries per body, visiting
  //   only the candidate cells near each body from the box->cell
  //   index, so cost scales with the bodies' swept regions and not
  //   with the number of cells this proc owns
  // entries for one cell are chained; a per-cell stamp detects the
  //   first touch of a cell this step

  if (ntotal > maxswcell) {
    int oldmax = maxswcell;
    maxswcell = ntotal;
    memory->grow(swstamp,maxswcell,"rigid_remap:swstamp");
    memory->grow(swhead,maxswcell,"rigid_remap:swhead");
    for (i = oldmax; i < maxswcell; i++) swstamp[i] = 0;
  }
  swcur++;
  nswcell = 0;
  nent = 0;

  int ncand,icand;
  int *cand;

  for (int m = 0; m < nblist; m++) {
    ibody = blist[m];
    double *blo = fix->bbodylo[ibody];
    double *bhi = fix->bbodyhi[ibody];
    ncand = grid->cells_in_box(blo,bhi,&cand);

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
            maxswcells += DELTA;
            memory->grow(swcells,maxswcells,"rigid_remap:swcells");
          }
          swcells[nswcell++] = icell;
        }

        // lblist = local surf index of the element on this proc;
        // for distributed surfs grid_changed() keeps it current

        if (nent == maxent) {
          maxent += DELTA;
          memory->grow(entnext,maxent,"rigid_remap:entnext");
          memory->grow(entelem,maxent,"rigid_remap:entelem");
        }
        entelem[nent] = (surfint) lblist[i];
        entnext[nent] = swhead[icell];
        swhead[icell] = nent++;
      }
    }
  }

  // phase 2: for each touched cell, install a merged collision list =
  //   cut list + chained swept elements not already present
  // dedup vs the cut list skips body surfs the cut pipeline placed
  //   at a body's start-of-step position; chained entries are unique
  //   among themselves (bodies are disjoint, one entry per element)
  // the list goes where particles reside: the cell itself if unsplit,
  //   else its sub cells; a split cell keeps its cut list, which
  //   Update::split2d/3d() index in lockstep with its piece map

  for (int ic = 0; ic < nswcell; ic++) {
    icell = swcells[ic];

    ncur = cells[icell].nsurf;
    cur = cells[icell].csurfs;
    merged = swlist;
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

    if (cells[icell].nsplit == 1)
      grid->set_collision_surfs(icell,nmerged,merged);
    else {
      isplit = cells[icell].isplit;
      for (j = 0; j < cells[icell].nsplit; j++) {
        isub = sinfo[isplit].csubs[j];
        grid->set_collision_surfs(isub,nmerged,merged);
      }
    }
    nswept++;
  }

  // per-cell surf lists changed on the host (read by fix rigid/kk)

  if (nswept) listschanged = 1;
}

/* ----------------------------------------------------------------------
   drop the collision lists set by collision_lists()
------------------------------------------------------------------------- */

void RigidRemap::reset_collision_lists()
{
  if (nswept) listschanged = 1;
  grid->reset_collision_surfs();
  nswept = 0;
}

/* ----------------------------------------------------------------------
   bring the per-cell state up to date, called each step before the
     bodies move to their end-of-step positions
   the static-inside flags are derived from the cell types and the body
     positions which produced them, so they must be rebuilt before the
     bodies move, and only when something other than this class changed
     the cells (a full re-map, fix adapt, fix balance)
------------------------------------------------------------------------- */

void RigidRemap::refresh()
{
  if (staticvalid) return;

  // the bodies' bboxes are the swept boxes of this step: restore the
  //   boxes of the positions which typed the cells

  for (int m = 0; m < fix->nblist; m++) {
    int ibody = fix->blist[m];
    fix->host_geometry(ibody);
    fix->body_bbox(ibody,0);
  }
  mark_static();
}

/* ----------------------------------------------------------------------
   flag every owned cell which is INSIDE because of the static surfs:
     uncut, typed INSIDE, and its center not interior to any body
   requires the bodies' bboxes and bins for their current positions,
     which typed the cells
------------------------------------------------------------------------- */

void RigidRemap::mark_static()
{
  double ctr[3];

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  int nglocal = grid->nlocal;

  if (nglocal > maxstatic) {
    maxstatic = nglocal;
    memory->destroy(staticinside);
    memory->create(staticinside,maxstatic,"rigid_remap:staticinside");
  }

  for (int icell = 0; icell < nglocal; icell++) {
    staticinside[icell] = 0;
    if (cells[icell].nsplit != 1) continue;
    if (cinfo[icell].type != INSIDE) continue;
    if (cell_cut(icell)) continue;

    ctr[0] = 0.5 * (cells[icell].lo[0] + cells[icell].hi[0]);
    ctr[1] = 0.5 * (cells[icell].lo[1] + cells[icell].hi[1]);
    if (dim == 3) ctr[2] = 0.5 * (cells[icell].lo[2] + cells[icell].hi[2]);
    else ctr[2] = 0.0;
    if (fix->inside_any_body(ctr)) continue;

    staticinside[icell] = 1;
  }

  staticvalid = 1;
  staticgen++;
}

/* ----------------------------------------------------------------------
   re-cut only the grid cells near the bodies
   a cell is re-cut if the set of surfs overlapping it changed,
     or if it is overlapped by a body surf (whose geometry moved)
   candidate surfs for a cell = the static surfs already in its list
     (static surfs never move, so the set overlapping a cell is fixed)
     plus the elements of the bodies near it, so the cost per cell is
     O(surfs in cell + nearby body surfs) and independent of the total
     surf count; only local surf indices are ever referenced, as
     required for distributed surfs
   the uncut cells in the region a body occupied before or after its
     move are re-typed INSIDE/OUTSIDE via parity tests of their centers,
     all other cells are untouched; a cell cut by no surf (surf-free, or
     overlapped only by transparent surfs) is typed this way, as
     Grid::set_inout() types it
   ghost cell copies of re-cut cells become stale, which is acceptable:
     the ghost cell surf lists the mover consults are re-covered by the
     collision lists every step, and cell volumes/types of ghost cells
     are not used
   return FALLBACK_NONE if done, else a reason code requesting a full
     grid re-map: a cell's surf count would exceed maxsurfpercell
------------------------------------------------------------------------- */

int RigidRemap::recut()
{
  int i,n,ncand,icell,ibody,moving;
  double ctr[3],rlo[3],rhi[3];
  double *clo,*chi;

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  int nglocal = grid->nlocal;
  int maxsurfpercell = grid->maxsurfpercell;
  int *rigidmap = update->rigidmap;

  int nbody = fix->nbody;
  int *bodystart = fix->bodystart;
  int *lblist = fix->lblist;
  double **bbodylo = fix->bbodylo;
  double **bbodyhi = fix->bbodyhi;

  // R = union over all bodies of the region rlo/rhi each
  //   occupied before and after its move this step
  // collect the owned cells overlapping R from the box->cell index,
  //   one body region at a time, so bodies far apart do not sweep the
  //   cells between them; the re-cut and re-type passes below iterate
  //   only this list

  nrcand = 0;
  splitchanged = 0;
  npending = 0;

  for (int m = 0; m < fix->nblist; m++) {
    ibody = fix->blist[m];
    for (i = 0; i < 3; i++) {
      rlo[i] = MIN(prevlo[ibody][i],bbodylo[ibody][i]);
      rhi[i] = MAX(prevhi[ibody][i],bbodyhi[ibody][i]);
    }

    int *cand;
    int ncells = grid->cells_in_box(rlo,rhi,&cand);

    // owned cells only: a ghost cell's cut list is covered by the
    //   collision lists, and its pieces are never consulted (subroute)
    // a cell lying wholly closer to the COM than any element, at both
    //   the start and the end of the move, held no element of this body
    //   at any time and kept the COM's parity: nothing to do for it
    //   (the COM moves on a line and the farthest corner's distance to
    //   a point on a line is convex, so the cell stays inside the
    //   element-free ball throughout)

    double rmin2 = fix->rminbody[ibody] * fix->rminbody[ibody];
    int interior = cominside[ibody];

    for (int ic = 0; ic < ncells; ic++) {
      icell = cand[ic];
      if (icell >= nglocal) continue;
      if (cells[icell].nsplit <= 0) continue;
      if (!box_overlap(cells[icell].lo,cells[icell].hi,rlo,rhi)) continue;
      if (interior) {
        double dnew = 0.0;
        double dold = 0.0;
        for (i = 0; i < dim; i++) {
          double c = fix->xcm[ibody][i];
          double dk = MAX(fabs(c-cells[icell].lo[i]),fabs(c-cells[icell].hi[i]));
          dnew += dk*dk;
          c = prevxcm[ibody][i];
          dk = MAX(fabs(c-cells[icell].lo[i]),fabs(c-cells[icell].hi[i]));
          dold += dk*dk;
        }
        if (dnew < rmin2 && dold < rmin2) continue;
      }
      if (nrcand == maxrcand) {
        maxrcand += DELTA;
        memory->grow(rcand,maxrcand,"rigid_remap:rcand");
      }
      rcand[nrcand++] = icell;
    }
  }

  // a cell in the regions of several bodies is listed once

  if (nbody > 1) {
    std::sort(rcand,rcand+nrcand);
    nrcand = std::unique(rcand,rcand+nrcand) - rcand;
  }
  ncand_run += nrcand;

  double tlists = 0.0;
  double tcut = 0.0;
  double tstart = 0.0;
  int timeflag = fix->timeflag;

  // pass 1: re-cut cells in R whose surf overlap changed
  //   or which are overlapped by a moved body surf (from any body)
  // candidate list keeps the cell's static surfs in their current
  //   order, followed by the body elements, so an unchanged cell
  //   yields an identical list and is skipped

  for (int ic = 0; ic < nrcand; ic++) {
    icell = rcand[ic];

    int nsplitold = cells[icell].nsplit;
    if (timeflag) tstart = MPI_Wtime();

    ncand = 0;
    surfint *cur = cells[icell].csurfs;
    for (i = 0; i < cells[icell].nsurf; i++)
      if (rigidmap[cur[i]] < 0) reclist[ncand++] = cur[i];

    // only bodies whose bounding box overlaps this cell contribute
    //   candidates: bbodylo/bbodyhi bound every element of the body at
    //   its end-of-step position, so surfs_in_cell would reject all of
    //   them one at a time.  the body bins find them without a loop
    //   over all bodies, so the cost of a cell is independent of how
    //   many bodies are defined far away from it

    int *blist;
    int nb = fix->body_box(cells[icell].lo,cells[icell].hi,&blist);
    for (int m = 0; m < nb; m++) {
      ibody = blist[m];

      // radial prefilter: body_box() only compared this cell against the
      //   body's bounding BOX, so it still returns a body whose surfs all
      //   lie far from the cell -- the deep interior of the body, and the
      //   corners of its bbox, which for a rounded body is 1-pi/4 of it.
      //   every such surf would then be tested against the cell one at a
      //   time by surfs_in_cell() only to be rejected
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
        double c = fix->xcm[ibody][k];
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

      double rmax = fix->rmaxbody[ibody] + fix->bboxeps[ibody];
      if (dlo2 > rmax*rmax) continue;
      double rmin = fix->rminbody[ibody] - fix->bboxeps[ibody];
      if (rmin > 0.0 && dhi2 < rmin*rmin) continue;

      // only the elements whose own box reaches the cell are offered:
      //   the exact test rejects the others one at a time otherwise

      for (i = bodystart[ibody]; i < bodystart[ibody+1]; i++)
        if (box_overlap(fix->elemlo[i],fix->elemhi[i],clo2,chi2))
          reclist[ncand++] = lblist[i];
    }

    // new list of surfs overlapping this cell

    n = grid->surfs_in_cell(icell,ncand,reclist,newlist,maxsurfpercell);
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
      if (n == 0) {
        if (timeflag) tlists += MPI_Wtime() - tstart;
        continue;
      }
      if (memcmp(newlist,cells[icell].csurfs,n*sizeof(surfint)) == 0) {
        if (timeflag) tlists += MPI_Wtime() - tstart;
        continue;
      }
    }
    if (timeflag) {
      double now = MPI_Wtime();
      tlists += now - tstart;
      tstart = now;
    }
    nlist_run++;

    // cut the cell by its new list

    recut_cell(icell,n,newlist);
    if (timeflag) tcut += MPI_Wtime() - tstart;
  }

  if (timeflag) {
    fix->add_time(FixRigid::T_RECUT_LISTS,tlists);
    fix->add_time(FixRigid::T_RECUT_CUT,tcut);
    tstart = MPI_Wtime();
  }

  // pass 2: re-type the uncut owned cells in R which a body interior
  //   entered or left, by the parity test of their centers: cells swept
  //   over entirely within one step never overlap a body surf at the
  //   start- or end-of-step position, so pass 1 never sees them
  // a cell INSIDE because of the static surfs is left alone
  // the ray cast is only needed for a center in the shell of some body
  //   between its inner and outer radius about the COM: closer than the
  //   inner radius the center shares the parity of the COM, since no
  //   element lies between them, and beyond the outer radius of every
  //   body it is outside them all

  for (int ic = 0; ic < nrcand; ic++) {
    icell = rcand[ic];
    if (cells[icell].nsplit != 1) continue;
    if (cell_cut(icell)) continue;
    if (staticinside[icell]) continue;

    clo = cells[icell].lo;
    chi = cells[icell].hi;
    ctr[0] = 0.5 * (clo[0] + chi[0]);
    ctr[1] = 0.5 * (clo[1] + chi[1]);
    if (dim == 3) ctr[2] = 0.5 * (clo[2] + chi[2]);
    else ctr[2] = 0.0;

    int type = OUTSIDE;
    int shell = 0;
    int *blist;
    int nb = fix->body_box(clo,chi,&blist);
    for (int m = 0; m < nb; m++) {
      ibody = blist[m];
      double d2 = 0.0;
      for (int k = 0; k < dim; k++) {
        double dk = ctr[k] - fix->xcm[ibody][k];
        d2 += dk*dk;
      }
      double rmin = fix->rminbody[ibody];
      if (d2 < rmin*rmin) {
        if (cominside[ibody]) {
          type = INSIDE;
          shell = 0;
          break;
        }
        continue;
      }
      double rmax = fix->rmaxbody[ibody] + fix->bboxeps[ibody];
      if (d2 <= rmax*rmax) shell = 1;
    }
    if (shell) {
      if (fix->inside_any_body(ctr)) type = INSIDE;
      else type = OUTSIDE;
    }
    if (cinfo[icell].type == type) continue;

    grid->set_cell_type(icell,type);
    typechanged = 1;
  }

  if (timeflag) fix->add_time(FixRigid::T_RECUT_TYPE,MPI_Wtime() - tstart);

  // the bodies' current bboxes bound the region on the next step

  for (int m = 0; m < fix->nblist; m++) {
    ibody = fix->blist[m];
    for (i = 0; i < 3; i++) {
      prevlo[ibody][i] = bbodylo[ibody][i];
      prevhi[ibody][i] = bbodyhi[ibody][i];
      prevxcm[ibody][i] = fix->xcm[ibody][i];
    }
  }

  return FALLBACK_NONE;
}

/* ----------------------------------------------------------------------
   install the new cut list of owned cell icell and cut it: its type,
     corner marks and flow volume, its piece map if it is split, and a
     pending change if its number of flow pieces changed
   the per-cell part of recut(), shared with the device variant which
     computes the lists elsewhere and cuts the cell on the device
------------------------------------------------------------------------- */

void RigidRemap::recut_cell(int icell, int n, surfint *newlist)
{
  int i,nsplitone,xsub;
  int corner[8];
  double xsplit[3],ctr[3];
  double unknownvol;
  double *vols;
  double *clo,*chi;

  Grid::ChildCell *cells = grid->cells;

  // install the new cut list, which any sub cells share

  grid->set_cell_surfs(icell,n,newlist);
  listschanged = 1;

  clo = cells[icell].lo;
  chi = cells[icell].hi;
  ctr[0] = 0.5 * (clo[0] + chi[0]);
  ctr[1] = 0.5 * (clo[1] + chi[1]);
  if (dim == 3) ctr[2] = 0.5 * (clo[2] + chi[2]);
  else ctr[2] = 0.0;

  // a cell overlapped by no surf, or only by transparent ones, is not
  //   cut (Grid::surf2grid_split() skips non-OVERLAP cells): full
  //   flow volume, interior/exterior typing via parity test

  if (!cell_cut(icell)) {
    if (fix->inside_any_body(ctr)) corner[0] = INSIDE;
    else corner[0] = OUTSIDE;
    apply_cut(icell,0,NULL,NULL,corner,0,NULL);
    return;
  }

  // re-cut the cell

  ncut_run++;
  nsplitone = grid->cut_cell(icell,vols,newmap,corner,xsub,xsplit);

  // the cut leaves the corner marks UNKNOWN when every surf only
  //   touches the cell faces, so the cell is one flow piece lying
  //   entirely on one side of them; the full pipeline marks it from
  //   a neighbor by flood fill in Grid::set_inout(), the parity test
  //   of its center against the closed body gives the same answer
  // as set_inout(): the cell stays OVERLAP with its corners marked,
  //   and its volume is the full cell volume or zero

  if (corner[0] == UNKNOWN) {
    int mark = OUTSIDE;
    if (fix->inside_any_body(ctr)) mark = INSIDE;
    int ncorner = (dim == 3) ? 8 : 4;
    for (i = 0; i < ncorner; i++) corner[i] = mark;
    nsplitone = 1;
    vols = &unknownvol;
    if (mark == INSIDE) unknownvol = 0.0;
    else unknownvol = grid->cell_volume(clo,chi);
  }

  apply_cut(icell,nsplitone,vols,newmap,corner,xsub,xsplit);
}

/* ----------------------------------------------------------------------
   install the result of cutting owned cell icell by the cut list it
     already holds: nsplitone flow pieces with volumes vols, the piece
     map of its surfs, corner marks, and the reference piece xsub with
     a point xsplit in it if it is split
   nsplitone = 0 means no surf cuts the cell: corner[0] is then its
     INSIDE/OUTSIDE type, and the other arguments are unused
   the cut itself ran on the host (recut_cell) or on the device
------------------------------------------------------------------------- */

void RigidRemap::apply_cut(int icell, int nsplitone, double *vols,
                           int *map, int *corner, int xsub, double *xsplit)
{
  Grid::ChildCell *cells = grid->cells;
  int nsplitold = cells[icell].nsplit;
  int n = cells[icell].nsurf;
  double *clo = cells[icell].lo;
  double *chi = cells[icell].hi;

  // an uncut cell which was a split cell gives up its sub cells, which
  //   changes the cell count

  if (nsplitone == 0) {
    if (nsplitold > 1) split_pending(icell,1,0,NULL,0,NULL,NULL);
    grid->set_cell_type(icell,corner[0]);
    typechanged = 1;
    return;
  }

  // the number of disconnected flow pieces changed: the cell gains
  //   or loses sub cells, which changes this proc's cell count and
  //   the sub cell indices other procs migrate particles into
  // recorded now and applied by apply_pending() at the end of the
  //   step, together with every other such cell
  // a split cell's own volume is the whole cell volume

  if (nsplitone != nsplitold) {
    split_pending(icell,nsplitone,n,map,xsub,xsplit,vols);
    if (nsplitone > 1)
      grid->set_cell_overlap(icell,grid->cell_volume(clo,chi),corner);
    else grid->set_cell_overlap(icell,vols[0],corner);
    typechanged = 1;
    return;
  }

  // a cell which was and still is split keeps its sub cells and
  //   takes the new piece map and per-piece volumes in place
  // a split cell's own volume is the whole cell volume
  //   (Grid::surf2grid_split() likewise leaves it alone)

  if (nsplitone > 1) {
    grid->set_split_info(icell,map,xsub,xsplit,vols);
    splitchanged = 1;
    grid->set_cell_overlap(icell,grid->cell_volume(clo,chi),corner);
    typechanged = 1;
    return;
  }

  grid->set_cell_overlap(icell,vols[0],corner);
  typechanged = 1;
}

/* ----------------------------------------------------------------------
   return 1 if grid cell icell is cut by a surf, else 0
   a cell overlapped only by transparent surfs is not cut:
     Grid::surf2grid_split() leaves it uncut, and Grid::set_inout()
     types it INSIDE/OUTSIDE by flood fill like a surf-free cell,
     so the incremental re-cut must re-type it the same way
------------------------------------------------------------------------- */

int RigidRemap::cell_cut(int icell)
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
   record that one owned cell's number of flow pieces changes this step
   the cut of the next cell overwrites the work buffers, so the piece
     map and the piece volumes are copied out here
   nsplitnew = 1 means the cell stops being a split cell, and needs no
     map and no volumes
------------------------------------------------------------------------- */

void RigidRemap::split_pending(int icell, int nsplitnew, int n, int *map,
                               int xsub, double *xsplit, double *vols)
{
  if (npending == maxpending) {
    int oldmax = maxpending;
    maxpending += DELTA;
    pending = (Grid::SplitChange *)
      memory->srealloc(pending,maxpending*sizeof(Grid::SplitChange),
                       "rigid_remap:pending");
    memory->grow(maxmap,maxpending,"rigid_remap:maxmap");
    memory->grow(maxvols,maxpending,"rigid_remap:maxvols");
    for (int m = oldmax; m < maxpending; m++) {
      pending[m].map = NULL;
      pending[m].vols = NULL;
      maxmap[m] = maxvols[m] = 0;
    }
  }

  int m = npending++;
  Grid::SplitChange *p = &pending[m];
  p->icell = icell;
  p->nsplitnew = nsplitnew;
  p->nsurf = n;

  if (nsplitnew == 1) return;

  if (n > maxmap[m]) {
    maxmap[m] = n;
    memory->destroy(p->map);
    memory->create(p->map,n,"rigid_remap:pendingmap");
  }
  memcpy(p->map,map,n*sizeof(int));

  p->xsub = xsub;
  p->xsplit[0] = xsplit[0];
  p->xsplit[1] = xsplit[1];
  p->xsplit[2] = xsplit[2];

  if (nsplitnew > maxvols[m]) {
    maxvols[m] = nsplitnew;
    memory->destroy(p->vols);
    memory->create(p->vols,nsplitnew,"rigid_remap:pendingvols");
  }
  memcpy(p->vols,vols,nsplitnew*sizeof(double));
}

/* ----------------------------------------------------------------------
   1 if the pending changes cannot be applied in place on this proc
   Grid::restructure_split_cells() moves only sub cells, since a ghost
     copy of any other cell on another proc records its index; after a
     grid rebuild or balance the tail of the owned cell list may hold
     other cells, and a change which would pull one of them into a hole
     takes the collective rebuild instead
   the caller reduces the flag so every proc takes the same path
------------------------------------------------------------------------- */

int RigidRemap::rebuild_needed()
{
  if (!npending) return 0;
  return grid->restructure_check(npending,pending);
}

/* ----------------------------------------------------------------------
   apply every pending split change, in place of a full grid re-map
   recut() already produced the correct surf list, flow volume, cell
     type and corner marks of every cell, so the two most expensive
     steps of a full re-map are skipped: Grid::clear_surf() plus
     Grid::surf2grid(), which re-maps every surf to every cell, and
     Grid::set_inout(), whose flood fill is an iterative collective
   what is left is the structural part: sub cells are added and removed
     and the cell list compacted by Grid::restructure_split_cells(),
     which repairs every reference to the cells it moves, so the ghost
     cells and neighbor links survive and nothing is communicated
     beyond one reduction of the cell counts
   rebuild = 1 (collective, every proc agrees) takes the sequence fix
     adapt uses when it refines or coarsens cells during a run instead:
     no owned cell may be added or removed while ghost cells are
     stored, so they are dropped and re-acquired around the change
   the caller has pulled the particles of every split cell up into the
     cell itself, so none is labelled with a sub cell about to vanish,
     and has sorted them if the per-cell lists are to be walked
   collective: every proc enters together
------------------------------------------------------------------------- */

void RigidRemap::apply_pending(int rebuild)
{
  int m;
  int nstatic_old = grid->nlocal;

  if (rebuild) {

    // neighbor links become cell IDs, which survive the cells moving

    grid->unset_neighbors();
    grid->remove_ghosts();

    for (m = 0; m < npending; m++)
      grid->split_cell_unset(pending[m].icell);

    for (m = 0; m < npending; m++) {
      if (pending[m].nsplitnew == 1) continue;
      int *csplits = grid->csplits->get(pending[m].nsurf);
      int *csubs = grid->csubs->get(pending[m].nsplitnew);
      if (!csplits || !csubs)
        error->one(FLERR,"Failed to allocate grid split cell lists");
      memcpy(csplits,pending[m].map,pending[m].nsurf*sizeof(int));
      grid->split_cell_set(pending[m].icell,pending[m].nsplitnew,
                           csplits,csubs,pending[m].xsub,pending[m].xsplit,
                           pending[m].vols);
    }

    grid->remove_marked_cells();

    grid->setup_owned();
    grid->acquire_ghosts();
    grid->reset_neighbors();
    comm->reset_neighbors();

    // as after a grid rebuild with distributed surfs

    if (surf->distributed) {
      surf->localghost_changed_step = update->ntimestep;
      for (int i = 0; i < surf->ncustom; i++) surf->estatus[i] = 0;
    }

  } else {
    grid->restructure_split_cells(npending,pending);
    restructured = 1;
    listschanged = 1;

    // global cell counts, the one collective of the in-place change

    bigint mine[3],all[3];
    mine[0] = grid->nunsplitlocal;
    mine[1] = grid->nsplitlocal;
    mine[2] = grid->nsublocal;
    MPI_Allreduce(mine,all,3,MPI_SPARTA_BIGINT,MPI_SUM,world);
    grid->nunsplit = all[0];
    grid->nsplit = all[1];
    grid->nsub = all[2];

    // dumps count cells when they next write

    for (int i = 0; i < output->ndump; i++)
      output->dump[i]->reset_grid_count();
  }

  // a per-grid compute sized itself for the old cell count; one which
  //   sees the same count again would keep arrays that now refer to
  //   different cells, so they are dropped, as AdaptGrid does

  Compute **compute = modify->compute;
  for (int i = 0; i < modify->ncompute; i++)
    if (compute[i]->per_grid_flag) {
      compute[i]->reallocate();
      compute[i]->invoked_flag = 0;
    }

  if (rebuild) grid->notify_changed();

  // notify_changed() invalidates the static-inside flags as for any
  //   grid change, but this change only added, removed and moved sub
  //   cells, which are never static INSIDE: the flags of the other
  //   cells still hold, and the cells appended at the end are sub cells

  if (grid->nlocal > maxstatic) {
    maxstatic = grid->nlocal;
    memory->grow(staticinside,maxstatic,"rigid_remap:staticinside");
  }
  for (m = nstatic_old; m < grid->nlocal; m++) staticinside[m] = 0;
  staticvalid = 1;
}

/* ----------------------------------------------------------------------
   the previous-position state of one body: it travels with the body,
     since the re-cut region of the next step is the union of where the
     body was and where it goes
------------------------------------------------------------------------- */

void RigidRemap::pack_prev(int ibody, double *buf)
{
  memcpy(&buf[0],prevlo[ibody],3*sizeof(double));
  memcpy(&buf[3],prevhi[ibody],3*sizeof(double));
  memcpy(&buf[6],prevxcm[ibody],3*sizeof(double));
}

/* ---------------------------------------------------------------------- */

void RigidRemap::unpack_prev(int ibody, const double *buf)
{
  memcpy(prevlo[ibody],&buf[0],3*sizeof(double));
  memcpy(prevhi[ibody],&buf[3],3*sizeof(double));
  memcpy(prevxcm[ibody],&buf[6],3*sizeof(double));
}

/* ----------------------------------------------------------------------
   grid cells were rebuilt, adapted, or migrated to other procs
   any collision lists were discarded with the cells; the cut lists
     installed by recut() were copied by Grid::compress() or discarded
     by Grid::clear_surf() and are re-derived by the next recut()
------------------------------------------------------------------------- */

void RigidRemap::grid_changed()
{
  nswept = 0;
  listschanged = 1;
  staticvalid = 0;

  // the grid now matches the bodies where they are

  for (int ibody = 0; ibody < fix->nbody; ibody++)
    for (int j = 0; j < 3; j++) {
      prevlo[ibody][j] = fix->bbodylo[ibody][j];
      prevhi[ibody][j] = fix->bbodyhi[ibody][j];
      prevxcm[ibody][j] = fix->xcm[ibody][j];
    }
}

/* ---------------------------------------------------------------------- */

double RigidRemap::memory_usage()
{
  double bytes = 0.0;
  bytes += (double) fix->nbody * 6 * sizeof(double);         // prevlo/hi
  bytes += (double) maxswcell * 2 * sizeof(int);             // swstamp,swhead
  bytes += (double) maxswcells * sizeof(int);                // swcells
  bytes += (double) maxent * (sizeof(int) + sizeof(surfint)); // entries
  bytes += (double) maxswlist * sizeof(surfint);
  bytes += (double) maxstatic * sizeof(char);
  bytes += (double) maxrcand * sizeof(int);
  bytes += (double) maxreclist * sizeof(surfint);
  bytes += (double) 2 * maxnewlist * sizeof(int);
  bytes += (double) maxpending * (sizeof(Grid::SplitChange) + 2*sizeof(int));
  return bytes;
}
