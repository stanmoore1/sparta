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

#include "rigid_remap_kokkos.h"
#include "fix_rigid_kokkos.h"
#include "grid_kokkos.h"
#include "surf_kokkos.h"
#include "update_kokkos.h"
#include "cut_kokkos.h"
#include "cut2d_kokkos.h"
#include "cut3d_kokkos.h"
#include "domain.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "sparta_masks.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

#define CUTSCRATCH 268435456   // bytes of device scratch per chunk of 3d cuts

enum{CELLUNKNOWN,CELLOUTSIDE,CELLINSIDE,CELLOVERLAP};   // same as Grid

/* ----------------------------------------------------------------------
   the device twin of RigidRemap: the per-step collision lists are
     built as the mover's graph on the device, from the device cell
     bins, the fix's device body table and the device cut graph, with
     the same merge as the host (cut list, then the swept elements not
     already in it, in descending element order), so the lists agree
     entry for entry
------------------------------------------------------------------------- */

RigidRemapKokkos::RigidRemapKokkos(SPARTA *sparta, FixRigidKokkos *fix_in) :
  RigidRemap(sparta,fix_in)
{
  fix_kk = fix_in;
  maxpair = 0;
  maxswcell_kk = maxtouched_kk = maxhit_kk = 0;
  ntouched_prev = 0;
  d_ntouched = DAT::t_int_scalar("rigid_remap:ntouched");
  d_nhit = DAT::t_int_scalar("rigid_remap:nhit");
  staticgen_kk = -1;
  maxrcand_kk = maxrcandlist_kk = maxch_kk = maxchent_kk = 0;
  maxvert_kk = maxedge_kk = maxcline_kk = maxpt_kk = 0;
  d_cutstats = DAT::t_int_1d("rigid_remap:cutstats",2);
}

/* ----------------------------------------------------------------------
   box/box overlap, touching counts as overlap; a device twin of the
     host helper
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
static int box_overlap_kk(const double *alo, const double *ahi,
                          const double *blo, const double *bhi)
{
  if (ahi[0] < blo[0] || alo[0] > bhi[0]) return 0;
  if (ahi[1] < blo[1] || alo[1] > bhi[1]) return 0;
  if (ahi[2] < blo[2] || alo[2] > bhi[2]) return 0;
  return 1;
}

/* ----------------------------------------------------------------------
   add every body's swept surfs to the collision lists of the cells it
     sweeps this step, on the device: the mover's list for a cell is its
     cut list followed by the swept elements, the same merge as the host
     (cut list, then the swept elements not already in it, in descending
     element order)
   only the cells a body sweeps are touched: the count pass over the
     (body, bin) pairs records every (cell, element) hit and lists each
     touched cell once, the hits are scattered into one row per touched
     cell, and the mover reads a cell's row through d_swrow (a sub cell
     its split cell's row).  nothing is sized by the grid but the two
     per-cell arrays, which are reset from the previous step's touched
     list rather than rewritten
------------------------------------------------------------------------- */

void RigidRemapKokkos::collision_lists()
{
  int i,k,ibody;

  GridKokkos *grid_kk = (GridKokkos *) grid;
  nswept = 0;

  // pending host changes reach the device first, then the cell bins
  //   and this step's swept body table

  grid_kk->apply_changes();
  grid_kk->sync_cell_bins();
  fix_kk->pack_body_device();

  int ntotal = grid->nlocal + grid->nghost;
  int nbody = fix->nbody;
  int nblist = fix->nblist;
  int *blist = fix->blist;
  double **bbodylo = fix->bbodylo;
  double **bbodyhi = fix->bbodyhi;

  // (body, bin) pairs: the bins each body's swept box overlaps

  int npair = 0;
  int qlo[3],qhi[3];
  for (int m = 0; m < nblist; m++) {
    ibody = blist[m];
    int n = 1;
    for (k = 0; k < 3; k++) {
      qlo[k] = (int) ((bbodylo[ibody][k]-grid->cellbinlo[k]) * grid->cellbininv[k]);
      qhi[k] = (int) ((bbodyhi[ibody][k]-grid->cellbinlo[k]) * grid->cellbininv[k]);
      qlo[k] = MAX(0,MIN(qlo[k],grid->cellnbin[k]-1));
      qhi[k] = MAX(0,MIN(qhi[k],grid->cellnbin[k]-1));
      n *= qhi[k] - qlo[k] + 1;
    }
    npair += n;
  }

  if (npair > maxpair) {
    maxpair = npair;
    k_pairbody = DAT::tdual_int_1d("rigid_remap:pairbody",maxpair);
    k_pairbin = DAT::tdual_int_1d("rigid_remap:pairbin",maxpair);
  }
  if ((int) k_qlo.extent(0) < nbody) {
    k_qlo = DAT::tdual_int_2d("rigid_remap:qlo",nbody,3);
    k_qhi = DAT::tdual_int_2d("rigid_remap:qhi",nbody,3);
  }
  auto h_pairbody = k_pairbody.view_host();
  auto h_pairbin = k_pairbin.view_host();
  auto h_qlo = k_qlo.view_host();
  auto h_qhi = k_qhi.view_host();

  npair = 0;
  for (int m = 0; m < nblist; m++) {
    ibody = blist[m];
    for (k = 0; k < 3; k++) {
      qlo[k] = (int) ((bbodylo[ibody][k]-grid->cellbinlo[k]) * grid->cellbininv[k]);
      qhi[k] = (int) ((bbodyhi[ibody][k]-grid->cellbinlo[k]) * grid->cellbininv[k]);
      qlo[k] = MAX(0,MIN(qlo[k],grid->cellnbin[k]-1));
      qhi[k] = MAX(0,MIN(qhi[k],grid->cellnbin[k]-1));
      h_qlo(ibody,k) = qlo[k];
      h_qhi(ibody,k) = qhi[k];
    }
    for (int ibz = qlo[2]; ibz <= qhi[2]; ibz++)
      for (int iby = qlo[1]; iby <= qhi[1]; iby++)
        for (int ibx = qlo[0]; ibx <= qhi[0]; ibx++) {
          h_pairbody(npair) = ibody;
          h_pairbin(npair) = (ibz*grid->cellnbin[1] + iby)*grid->cellnbin[0] + ibx;
          npair++;
        }
  }
  k_pairbody.modify_host(); k_pairbody.sync_device();
  k_pairbin.modify_host(); k_pairbin.sync_device();
  k_qlo.modify_host(); k_qlo.sync_device();
  k_qhi.modify_host(); k_qhi.sync_device();

  // per-cell count and row: zero and -1 everywhere but the cells the
  //   previous step touched, which its touched list resets here; views
  //   which grow start fresh

  if (ntotal > maxswcell_kk) {
    maxswcell_kk = ntotal;
    d_swcount = DAT::t_int_1d("rigid_remap:swcount",maxswcell_kk);
    d_swrow = DAT::t_int_1d("rigid_remap:swrow",maxswcell_kk);
    Kokkos::deep_copy(d_swrow,-1);
    d_swtouched = DAT::t_int_1d("rigid_remap:swtouched",maxswcell_kk);
    ntouched_prev = 0;
  }

  auto d_swcount = this->d_swcount;
  auto d_swrow = this->d_swrow;
  auto d_swtouched = this->d_swtouched;

  if (ntouched_prev) {
    Kokkos::parallel_for("rigid_remap:sw_reset",ntouched_prev,
                         KOKKOS_LAMBDA(const int t) {
      const int icell = d_swtouched(t);
      d_swcount(icell) = 0;
      d_swrow(icell) = -1;
    });
    ntouched_prev = 0;
  }

  // the views the kernels read, as locals so the lambdas carry copies

  auto d_pairbody = k_pairbody.view_device();
  auto d_pairbin = k_pairbin.view_device();
  auto d_qlo = k_qlo.view_device();
  auto d_qhi = k_qhi.view_device();
  auto d_binstart = grid_kk->d_cellbinstart;
  auto d_binlist = grid_kk->d_cellbinlist;
  auto d_cells = grid_kk->k_cells.view_device();
  const RigidBodyKK body = fix_kk->body;
  auto d_ntouched = this->d_ntouched;
  auto d_nhit = this->d_nhit;
  const int nbinx = grid->cellnbin[0];
  const int nbiny = grid->cellnbin[1];
  const int nbinz = grid->cellnbin[2];
  const double binlo0 = grid->cellbinlo[0], binlo1 = grid->cellbinlo[1],
    binlo2 = grid->cellbinlo[2];
  const double bininv0 = grid->cellbininv[0], bininv1 = grid->cellbininv[1],
    bininv2 = grid->cellbininv[2];

  // a cell is listed in every bin its box overlaps: for one body's query
  //   it is handled from the lowest of those bins inside the query range,
  //   which is what the host's per-query stamp achieves

  auto first_bin = KOKKOS_LAMBDA(const int icell, const int ibody, const int ibin) {
    const double *lo = d_cells[icell].lo;
    const double *hi = d_cells[icell].hi;
    int clo[3],chi[3];
    clo[0] = (int) ((lo[0]-binlo0) * bininv0);
    chi[0] = (int) ((hi[0]-binlo0) * bininv0);
    clo[1] = (int) ((lo[1]-binlo1) * bininv1);
    chi[1] = (int) ((hi[1]-binlo1) * bininv1);
    clo[2] = (int) ((lo[2]-binlo2) * bininv2);
    chi[2] = (int) ((hi[2]-binlo2) * bininv2);
    clo[0] = MAX(0,MIN(clo[0],nbinx-1));
    chi[0] = MAX(0,MIN(chi[0],nbinx-1));
    clo[1] = MAX(0,MIN(clo[1],nbiny-1));
    chi[1] = MAX(0,MIN(chi[1],nbiny-1));
    clo[2] = MAX(0,MIN(clo[2],nbinz-1));
    chi[2] = MAX(0,MIN(chi[2],nbinz-1));
    const int fx = MAX(clo[0],d_qlo(ibody,0));
    const int fy = MAX(clo[1],d_qlo(ibody,1));
    const int fz = MAX(clo[2],d_qlo(ibody,2));
    return ibin == (fz*nbiny + fy)*nbinx + fx;
  };

  // count the swept elements of each cell, recording every hit and
  //   listing a cell the first time it is counted
  // a hit buffer too small for this step is grown and the pass re-run,
  //   which happens on the first step and rarely after

  int ntouched,nhit;

  while (1) {
    Kokkos::deep_copy(d_ntouched,0);
    Kokkos::deep_copy(d_nhit,0);
    auto d_hitcell = this->d_hitcell;
    auto d_hitelem = this->d_hitelem;
    const int maxhit = maxhit_kk;

    Kokkos::parallel_for("rigid_remap:sw_count",npair, KOKKOS_LAMBDA(const int p) {
      const int ibody = d_pairbody(p);
      const int ibin = d_pairbin(p);
      for (int m = d_binstart(ibin); m < d_binstart(ibin+1); m++) {
        const int icell = d_binlist(m);
        if (d_cells[icell].nsplit <= 0) continue;
        if (d_cells[icell].nsurf < 0) continue;
        if (!first_bin(icell,ibody,ibin)) continue;
        const double *clo = d_cells[icell].lo;
        const double *chi = d_cells[icell].hi;
        if (!body.box_overlap(ibody,clo,chi)) continue;
        int n = 0;
        for (int e = body.d_bodystart(ibody); e < body.d_bodystart(ibody+1); e++) {
          if (!body.elem_overlap(e,clo,chi)) continue;
          const int h = Kokkos::atomic_fetch_add(&d_nhit(),1);
          if (h < maxhit) {
            d_hitcell(h) = icell;
            d_hitelem(h) = e;
          }
          n++;
        }
        if (!n) continue;
        const int old = Kokkos::atomic_fetch_add(&d_swcount(icell),n);
        if (old == 0) {
          const int t = Kokkos::atomic_fetch_add(&d_ntouched(),1);
          d_swtouched(t) = icell;
        }
      }
    });

    Kokkos::deep_copy(ntouched,d_ntouched);
    Kokkos::deep_copy(nhit,d_nhit);
    if (nhit <= maxhit_kk) break;

    maxhit_kk = nhit + nhit/2;
    this->d_hitcell = DAT::t_int_1d("rigid_remap:hitcell",maxhit_kk);
    this->d_hitelem = DAT::t_int_1d("rigid_remap:hitelem",maxhit_kk);
    this->d_swelem = DAT::t_int_1d("rigid_remap:swelem",maxhit_kk);
    Kokkos::parallel_for("rigid_remap:sw_reset",ntouched,
                         KOKKOS_LAMBDA(const int t) {
      d_swcount(d_swtouched(t)) = 0;
    });
  }

  ntouched_prev = ntouched;

  if (!nhit) {
    grid_kk->d_csurfs_move = grid_kk->d_csurfs;
    grid_kk->swextras = 0;
    return;
  }

  // one row per touched cell, in list order: the cell's row index, the
  //   row offsets by a scan of the counts, then the hits scattered into
  //   the rows

  if (ntouched > maxtouched_kk) {
    maxtouched_kk = ntouched;
    d_swoff = DAT::t_int_1d("rigid_remap:swoff",maxtouched_kk+1);
    d_swcursor = DAT::t_int_1d("rigid_remap:swcursor",maxtouched_kk);
    d_swext = DAT::t_int_1d("rigid_remap:swext",maxtouched_kk);
  }
  auto d_swoff = this->d_swoff;
  auto d_swcursor = this->d_swcursor;
  auto d_swext = this->d_swext;
  auto d_swelem = this->d_swelem;
  auto d_hitcell = this->d_hitcell;
  auto d_hitelem = this->d_hitelem;

  Kokkos::parallel_for("rigid_remap:sw_rows",ntouched, KOKKOS_LAMBDA(const int t) {
    d_swrow(d_swtouched(t)) = t;
    d_swcursor(t) = 0;
  });

  Kokkos::parallel_scan("rigid_remap:sw_scan",ntouched,
                        KOKKOS_LAMBDA(const int t, int &sum, const bool final) {
    const int n = d_swcount(d_swtouched(t));
    if (final) d_swoff(t) = sum;
    sum += n;
    if (final && t == ntouched-1) d_swoff(ntouched) = sum;
  });

  Kokkos::parallel_for("rigid_remap:sw_scatter",nhit, KOKKOS_LAMBDA(const int h) {
    const int t = d_swrow(d_hitcell(h));
    const int j = Kokkos::atomic_fetch_add(&d_swcursor(t),1);
    d_swelem(d_swoff(t)+j) = d_hitelem(h);
  });

  // per row: the host chains its entries and reads the chain from its
  //   head, so the swept elements follow the cut list in descending
  //   element order; elements the cut list already holds are dropped

  auto d_csurfs = grid_kk->d_csurfs;

  Kokkos::parallel_for("rigid_remap:sw_sort_dedup",ntouched, KOKKOS_LAMBDA(const int t) {
    const int icell = d_swtouched(t);
    const int n = d_swcount(icell);
    const int off = d_swoff(t);
    for (int a = 1; a < n; a++) {
      const int v = d_swelem(off+a);
      int b = a - 1;
      while (b >= 0 && d_swelem(off+b) < v) {
        d_swelem(off+b+1) = d_swelem(off+b);
        b--;
      }
      d_swelem(off+b+1) = v;
    }
    const crs_size_type cstart = d_csurfs.row_map(icell);
    const int ncut = d_csurfs.row_map(icell+1) - cstart;
    int nextra = 0;
    for (int a = 0; a < n; a++) {
      const int s = body.d_lblist(d_swelem(off+a));
      int dup = 0;
      for (int j = 0; j < ncut; j++)
        if (d_csurfs.entries(cstart+j) == s) { dup = 1; break; }
      if (!dup) d_swelem(off+nextra++) = s;
    }
    d_swext(t) = nextra;
  });

  // the mover reads the cut graph plus these rows: a split cell keeps
  //   its cut list, its sub cells take its row

  grid_kk->d_csurfs_move = grid_kk->d_csurfs;
  grid_kk->d_swrow = d_swrow;
  grid_kk->d_swoff = d_swoff;
  grid_kk->d_swext = d_swext;
  grid_kk->d_swelem = d_swelem;
  grid_kk->swextras = 1;
  grid_kk->graph_generation++;
  nswept = 1;
}

/* ----------------------------------------------------------------------
   every cell's collision list is its cut list again
------------------------------------------------------------------------- */

void RigidRemapKokkos::reset_collision_lists()
{
  GridKokkos *grid_kk = (GridKokkos *) grid;
  grid_kk->d_csurfs_move = grid_kk->d_csurfs;
  grid_kk->swextras = 0;
  grid_kk->graph_generation++;
  nswept = 0;
}

/* ----------------------------------------------------------------------
   the incremental re-cut with its candidate cells, candidate surf lists
     and re-typing on the device, and the cut of the cells whose list
     changed on the host
   the device produces exactly the host's candidate set in ascending
     cell order, the same candidate surfs, the same overlap decisions
     and the same sorted lists, so the cells the host cuts, the order it
     cuts them in, and every type it sets are those of RigidRemap::recut()
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   size the scratch rows of the device cut to hold nvert vertices,
     nedge edges, ncline clipped lines and npt points, uninitialized
------------------------------------------------------------------------- */

void RigidRemapKokkos::grow_cut_scratch(int nvert, int nedge,
                                        int ncline, int npt)
{
  if (nvert > maxvert_kk) {
    maxvert_kk = nvert;
    d_verts = t_vertex_1d(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             "rigid_remap:verts"),maxvert_kk);
    d_loops3 = t_loop3_1d(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             "rigid_remap:loops3"),maxvert_kk);
    d_phs = t_ph_1d(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                       "rigid_remap:phs"),maxvert_kk);
    d_used3 = DAT::t_int_1d(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                               "rigid_remap:used3"),maxvert_kk);
    d_stack = DAT::t_int_1d(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                               "rigid_remap:stack"),maxvert_kk);
  }
  if (nedge > maxedge_kk) {
    maxedge_kk = nedge;
    d_edges = t_edge_1d(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                           "rigid_remap:edges"),maxedge_kk);
    d_facelist = DAT::t_int_1d(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                  "rigid_remap:facelist"),maxedge_kk);
    d_efaces = DAT::t_int_1d(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                "rigid_remap:efaces"),maxedge_kk);
  }
  if (ncline > maxcline_kk) {
    maxcline_kk = ncline;
    d_clines = t_cline_1d(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             "rigid_remap:clines"),maxcline_kk);
  }
  if (npt > maxpt_kk) {
    maxpt_kk = npt;
    d_points = t_point_1d(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             "rigid_remap:points"),maxpt_kk);
    d_loops = t_loop_1d(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                           "rigid_remap:loops"),maxpt_kk);
    d_pgs = t_pg_1d(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                       "rigid_remap:pgs"),maxpt_kk);
    d_used = DAT::t_int_1d(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                              "rigid_remap:used"),maxpt_kk);
  }
}

/* ----------------------------------------------------------------------
   the re-cut on the device
------------------------------------------------------------------------- */

int RigidRemapKokkos::recut()
{
  int i,k,m,ibody,icell;
  double rlo[3],rhi[3];

  GridKokkos *grid_kk = (GridKokkos *) grid;
  SurfKokkos *surf_kk = (SurfKokkos *) surf;
  UpdateKokkos *update_kk = (UpdateKokkos *) update;

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  int nglocal = grid->nlocal;
  int maxsurfpercell = grid->maxsurfpercell;
  int nbody = fix->nbody;
  int nblist = fix->nblist;
  int *blist = fix->blist;
  double **bbodylo = fix->bbodylo;
  double **bbodyhi = fix->bbodyhi;
  int timeflag = fix->timeflag;
  double tstart = 0.0;
  if (timeflag) tstart = MPI_Wtime();

  nrcand = 0;
  splitchanged = 0;
  npending = 0;

  // the device must hold the grid as it is, the surfs at the new pose,
  //   the body table at the new pose and the static-inside flags

  grid_kk->apply_changes();
  grid_kk->sync_cell_bins();
  fix_kk->pack_body_device();

  if (staticgen != staticgen_kk || (int) k_staticinside.extent(0) < nglocal) {
    if ((int) k_staticinside.extent(0) < nglocal)
      k_staticinside = DAT::tdual_int_1d("rigid_remap:staticinside",nglocal);
    auto h_static = k_staticinside.view_host();
    for (i = 0; i < nglocal; i++) h_static(i) = staticinside[i];
    k_staticinside.modify_host(); k_staticinside.sync_device();
    staticgen_kk = staticgen;
  }

  // per body: the region R = old bbox U new bbox, its bins, the COM now
  //   and before, the radii about it and whether the COM is interior

  if ((int) k_bodyparam.extent(0) < nbody) {
    k_bodyparam = tdual_dbl_2d("rigid_remap:bodyparam",nbody,15);
    k_qlo = DAT::tdual_int_2d("rigid_remap:qlo",nbody,3);
    k_qhi = DAT::tdual_int_2d("rigid_remap:qhi",nbody,3);
    k_cominside = DAT::tdual_int_1d("rigid_remap:cominside",nbody);
  }
  auto h_bodyparam = k_bodyparam.view_host();
  auto h_qlo = k_qlo.view_host();
  auto h_qhi = k_qhi.view_host();
  auto h_cominside = k_cominside.view_host();

  int npair = 0;
  for (int m = 0; m < nblist; m++) {
    ibody = blist[m];
    int n = 1;
    for (k = 0; k < 3; k++) {
      rlo[k] = MIN(prevlo[ibody][k],bbodylo[ibody][k]);
      rhi[k] = MAX(prevhi[ibody][k],bbodyhi[ibody][k]);
      h_bodyparam(ibody,k) = rlo[k];
      h_bodyparam(ibody,3+k) = rhi[k];
      h_bodyparam(ibody,6+k) = fix->xcm[ibody][k];
      h_bodyparam(ibody,9+k) = prevxcm[ibody][k];
      int qlo = (int) ((rlo[k]-grid->cellbinlo[k]) * grid->cellbininv[k]);
      int qhi = (int) ((rhi[k]-grid->cellbinlo[k]) * grid->cellbininv[k]);
      qlo = MAX(0,MIN(qlo,grid->cellnbin[k]-1));
      qhi = MAX(0,MIN(qhi,grid->cellnbin[k]-1));
      h_qlo(ibody,k) = qlo;
      h_qhi(ibody,k) = qhi;
      n *= qhi - qlo + 1;
    }
    h_bodyparam(ibody,12) = fix->rminbody[ibody];
    h_bodyparam(ibody,13) = fix->rmaxbody[ibody];
    h_bodyparam(ibody,14) = fix->bboxeps[ibody];
    h_cominside(ibody) = cominside[ibody];
    npair += n;
  }

  if (npair > maxpair) {
    maxpair = npair;
    k_pairbody = DAT::tdual_int_1d("rigid_remap:pairbody",maxpair);
    k_pairbin = DAT::tdual_int_1d("rigid_remap:pairbin",maxpair);
  }
  auto h_pairbody = k_pairbody.view_host();
  auto h_pairbin = k_pairbin.view_host();
  npair = 0;
  for (int m = 0; m < nblist; m++)
    for (int ibz = h_qlo(blist[m],2); ibz <= h_qhi(blist[m],2); ibz++)
      for (int iby = h_qlo(blist[m],1); iby <= h_qhi(blist[m],1); iby++)
        for (int ibx = h_qlo(blist[m],0); ibx <= h_qhi(blist[m],0); ibx++) {
          h_pairbody(npair) = blist[m];
          h_pairbin(npair) = (ibz*grid->cellnbin[1] + iby)*grid->cellnbin[0] + ibx;
          npair++;
        }

  k_bodyparam.modify_host(); k_bodyparam.sync_device();
  k_qlo.modify_host(); k_qlo.sync_device();
  k_qhi.modify_host(); k_qhi.sync_device();
  k_cominside.modify_host(); k_cominside.sync_device();
  k_pairbody.modify_host(); k_pairbody.sync_device();
  k_pairbin.modify_host(); k_pairbin.sync_device();

  if (nglocal+1 > maxrcand_kk) {
    maxrcand_kk = nglocal+1;
    d_candflag = DAT::t_int_1d("rigid_remap:candflag",maxrcand_kk);
    d_candoff = DAT::t_int_1d("rigid_remap:candoff",maxrcand_kk);
  }

  auto d_pairbody = k_pairbody.view_device();
  auto d_pairbin = k_pairbin.view_device();
  auto d_qlo = k_qlo.view_device();
  auto d_qhi = k_qhi.view_device();
  auto d_bodyparam = k_bodyparam.view_device();
  auto d_cominside = k_cominside.view_device();
  auto d_binstart = grid_kk->d_cellbinstart;
  auto d_binlist = grid_kk->d_cellbinlist;
  auto d_cells = grid_kk->k_cells.view_device();
  auto d_cinfo = grid_kk->k_cinfo.view_device();
  auto d_candflag = this->d_candflag;
  auto d_candoff = this->d_candoff;
  const int nbinx = grid->cellnbin[0];
  const int nbiny = grid->cellnbin[1];
  const int nbinz = grid->cellnbin[2];
  const double binlo0 = grid->cellbinlo[0], binlo1 = grid->cellbinlo[1],
    binlo2 = grid->cellbinlo[2];
  const double bininv0 = grid->cellbininv[0], bininv1 = grid->cellbininv[1],
    bininv2 = grid->cellbininv[2];
  const int dimk = dim;

  auto first_bin = KOKKOS_LAMBDA(const int ic, const int ib, const int ibin) {
    const double *lo = d_cells[ic].lo;
    const double *hi = d_cells[ic].hi;
    int clo[3];
    clo[0] = MAX(0,MIN((int) ((lo[0]-binlo0) * bininv0),nbinx-1));
    clo[1] = MAX(0,MIN((int) ((lo[1]-binlo1) * bininv1),nbiny-1));
    clo[2] = MAX(0,MIN((int) ((lo[2]-binlo2) * bininv2),nbinz-1));
    const int fx = MAX(clo[0],d_qlo(ib,0));
    const int fy = MAX(clo[1],d_qlo(ib,1));
    const int fz = MAX(clo[2],d_qlo(ib,2));
    return ibin == (fz*nbiny + fy)*nbinx + fx;
  };

  // pass 0: the owned cells in R, as RigidRemap::recut() collects them,
  //   flagged then listed in ascending order

  Kokkos::deep_copy(Kokkos::subview(d_candflag,std::make_pair(0,nglocal)),0);

  Kokkos::parallel_for("rigid_remap:rc_cand_flag",npair, KOKKOS_LAMBDA(const int p) {
    const int ib = d_pairbody(p);
    const int ibin = d_pairbin(p);
    double blo[3],bhi[3];
    for (int kk = 0; kk < 3; kk++) {
      blo[kk] = d_bodyparam(ib,kk);
      bhi[kk] = d_bodyparam(ib,3+kk);
    }
    const double rmin2 = d_bodyparam(ib,12) * d_bodyparam(ib,12);
    const int interior = d_cominside(ib);
    for (int mm = d_binstart(ibin); mm < d_binstart(ibin+1); mm++) {
      const int ic = d_binlist(mm);
      if (ic >= nglocal) continue;
      if (d_cells[ic].nsplit <= 0) continue;
      if (!first_bin(ic,ib,ibin)) continue;
      const double *clo = d_cells[ic].lo;
      const double *chi = d_cells[ic].hi;
      if (!box_overlap_kk(clo,chi,blo,bhi)) continue;
      if (interior) {
        double dnew = 0.0;
        double dold = 0.0;
        for (int kk = 0; kk < dimk; kk++) {
          double c = d_bodyparam(ib,6+kk);
          double dk = MAX(fabs(c-clo[kk]),fabs(c-chi[kk]));
          dnew += dk*dk;
          c = d_bodyparam(ib,9+kk);
          dk = MAX(fabs(c-clo[kk]),fabs(c-chi[kk]));
          dold += dk*dk;
        }
        if (dnew < rmin2 && dold < rmin2) continue;
      }
      d_candflag(ic) = 1;
    }
  });

  Kokkos::parallel_scan("rigid_remap:rc_cand_scan",nglocal, KOKKOS_LAMBDA(const int ic, int &sum,
                                                const bool final) {
    const int n = d_candflag(ic);
    if (final) d_candoff(ic) = sum;
    sum += n;
    if (final && ic == nglocal-1) d_candoff(nglocal) = sum;
  });
  int ncand_dev = 0;
  if (nglocal) Kokkos::deep_copy(ncand_dev,Kokkos::subview(d_candoff,nglocal));
  nrcand = ncand_dev;
  ncand_run += nrcand;

  if (nrcand > maxrcand) {
    maxrcand = nrcand;
    memory->destroy(rcand);
    memory->create(rcand,maxrcand,"rigid_remap:rcand");
  }
  if (nrcand > maxrcandlist_kk) {
    maxrcandlist_kk = nrcand;
    k_rcand = DAT::tdual_int_1d("rigid_remap:rcand",maxrcandlist_kk);
    d_newn = DAT::t_int_1d("rigid_remap:newn",maxrcandlist_kk);
    d_chflag = DAT::t_int_1d("rigid_remap:chflag",maxrcandlist_kk);
    d_choff = DAT::t_int_1d("rigid_remap:choff",maxrcandlist_kk+1);
    d_newtype = DAT::t_int_1d("rigid_remap:newtype",maxrcandlist_kk);
    k_newlist = DAT::tdual_int_1d("rigid_remap:newlist",
                                  (bigint) maxrcandlist_kk * maxsurfpercell);
  }
  auto d_rcand = k_rcand.view_device();
  auto d_newn = this->d_newn;
  auto d_chflag = this->d_chflag;
  auto d_choff = this->d_choff;
  auto d_newtype = this->d_newtype;
  auto d_newlist = k_newlist.view_device();

  Kokkos::parallel_for("rigid_remap:rc_cand_list",nglocal, KOKKOS_LAMBDA(const int ic) {
    if (d_candflag(ic)) d_rcand(d_candoff(ic)) = ic;
  });

  if (!nrcand) {
    for (int m = 0; m < nblist; m++) {
      ibody = blist[m];
      for (i = 0; i < 3; i++) {
        prevlo[ibody][i] = bbodylo[ibody][i];
        prevhi[ibody][i] = bbodyhi[ibody][i];
        prevxcm[ibody][i] = fix->xcm[ibody][i];
      }
    }
    if (timeflag) fix->add_time(FixRigid::T_RECUT_LISTS,MPI_Wtime()-tstart);
    return FALLBACK_NONE;
  }

  // pass 1 on the device: the new cut list of every candidate, and
  //   whether it differs from the current one or holds a moving surf

  auto d_csurfs = grid_kk->d_csurfs;
  auto d_rigidmap = update_kk->d_rigidmap;
  auto d_lines = surf_kk->k_lines.view_device();
  auto d_tris = surf_kk->k_tris.view_device();
  const RigidBodyKK body = fix_kk->body;
  const int maxsurf = maxsurfpercell;
  DAT::t_int_scalar d_fallback("rigid_remap:fallback");
  Kokkos::deep_copy(d_fallback,0);

  auto surf_in_cell = KOKKOS_LAMBDA(const int s, const double *clo,
                                    const double *chi) {
    if (dimk == 2) {
      const double *x1 = d_lines[s].p1;
      const double *x2 = d_lines[s].p2;
      if (MAX(x1[0],x2[0]) < clo[0]) return 0;
      if (MIN(x1[0],x2[0]) > chi[0]) return 0;
      if (MAX(x1[1],x2[1]) < clo[1]) return 0;
      if (MIN(x1[1],x2[1]) > chi[1]) return 0;
      return CutKokkos::cliptest2d(x1,x2,clo,chi);
    }
    const double *x1 = d_tris[s].p1;
    const double *x2 = d_tris[s].p2;
    const double *x3 = d_tris[s].p3;
    double value;
    value = MAX(x1[0],x2[0]);
    if (MAX(value,x3[0]) < clo[0]) return 0;
    value = MIN(x1[0],x2[0]);
    if (MIN(value,x3[0]) > chi[0]) return 0;
    value = MAX(x1[1],x2[1]);
    if (MAX(value,x3[1]) < clo[1]) return 0;
    value = MIN(x1[1],x2[1]);
    if (MIN(value,x3[1]) > chi[1]) return 0;
    value = MAX(x1[2],x2[2]);
    if (MAX(value,x3[2]) < clo[2]) return 0;
    value = MIN(x1[2],x2[2]);
    if (MIN(value,x3[2]) > chi[2]) return 0;
    return CutKokkos::clip3d(x1,x2,x3,clo,chi) ? 1 : 0;
  };

  Kokkos::parallel_for("rigid_remap:rc_newlists",nrcand, KOKKOS_LAMBDA(const int ic) {
    const int icell = d_rcand(ic);
    const double *clo = d_cells[icell].lo;
    const double *chi = d_cells[icell].hi;
    int *newlist = &d_newlist(((bigint) ic) * maxsurf);
    int n = 0;

    // the cell's static surfs, then the elements of the bodies whose
    //   bbox overlaps it (fix->body_box), radially and box prefiltered

    const crs_size_type cstart = d_csurfs.row_map(icell);
    const int ncur = d_csurfs.row_map(icell+1) - cstart;
    for (int j = 0; j < ncur; j++) {
      const int s = d_csurfs.entries(cstart+j);
      if (d_rigidmap(s) >= 0) continue;
      if (surf_in_cell(s,clo,chi)) {
        if (n < maxsurf) newlist[n] = s;
        n++;
      }
    }

    int blo[3],bhi[3];
    body.box_bins(clo,chi,blo,bhi);

    for (int ibz = blo[2]; ibz <= bhi[2]; ibz++)
      for (int iby = blo[1]; iby <= bhi[1]; iby++)
        for (int ibx = blo[0]; ibx <= bhi[0]; ibx++) {
          const int ibin = (ibz*body.nbin[1] + iby)*body.nbin[0] + ibx;
          for (int mm = body.d_binstart(ibin); mm < body.d_binstart(ibin+1); mm++) {
            const int ib = body.d_binlist(mm);
            if (!body.box_overlap(ib,clo,chi)) continue;

            double dlo2 = 0.0, dhi2 = 0.0;
            for (int kk = 0; kk < dimk; kk++) {
              const double c = d_bodyparam(ib,6+kk);
              double dlok = 0.0;
              if (c < clo[kk]) dlok = clo[kk] - c;
              else if (c > chi[kk]) dlok = c - chi[kk];
              const double dhik = MAX(fabs(c-clo[kk]),fabs(c-chi[kk]));
              dlo2 += dlok*dlok;
              dhi2 += dhik*dhik;
            }
            const double rmax = d_bodyparam(ib,13) + d_bodyparam(ib,14);
            if (dlo2 > rmax*rmax) continue;
            const double rmin = d_bodyparam(ib,12) - d_bodyparam(ib,14);
            if (rmin > 0.0 && dhi2 < rmin*rmin) continue;

            for (int e = body.d_bodystart(ib); e < body.d_bodystart(ib+1); e++) {
              if (!body.elem_overlap(e,clo,chi)) continue;
              const int s = body.d_lblist(e);
              if (surf_in_cell(s,clo,chi)) {
                if (n < maxsurf) newlist[n] = s;
                n++;
              }
            }
          }
        }

    d_newn(ic) = n;
    if (n > maxsurf) {
      d_fallback() = 1;
      d_chflag(ic) = 1;
      return;
    }

    // ordered by local surf index, as the host sorts it

    for (int a = 1; a < n; a++) {
      const int v = newlist[a];
      int b = a - 1;
      while (b >= 0 && newlist[b] > v) {
        newlist[b+1] = newlist[b];
        b--;
      }
      newlist[b+1] = v;
    }

    int moving = 0;
    for (int j = 0; j < n; j++)
      if (d_rigidmap(newlist[j]) >= 0) { moving = 1; break; }

    int changed = 1;
    if (!moving && n == ncur) {
      changed = 0;
      for (int j = 0; j < n; j++)
        if (newlist[j] != d_csurfs.entries(cstart+j)) { changed = 1; break; }
    }
    d_chflag(ic) = changed;
  });

  int fallback = 0;
  Kokkos::deep_copy(fallback,d_fallback);
  if (fallback) {
    if (timeflag) fix->add_time(FixRigid::T_RECUT_LISTS,MPI_Wtime()-tstart);
    return FALLBACK_SURFMAX;
  }

  // pass 2 on the device: the type of every uncut candidate, from the
  //   shell test and the ray cast of RigidRemap::recut(); a candidate
  //   the host is about to cut gets no type here
  // the cut list is the new one if it changed, else the current one

  auto d_staticinside = k_staticinside.view_device();

  Kokkos::parallel_for("rigid_remap:rc_compare",nrcand, KOKKOS_LAMBDA(const int ic) {
    d_newtype(ic) = -1;
    const int icell = d_rcand(ic);
    if (d_cells[icell].nsplit != 1) return;
    if (d_staticinside(icell)) return;

    const int *list;
    int n;
    if (d_chflag(ic)) {
      list = &d_newlist(((bigint) ic) * maxsurf);
      n = d_newn(ic);
    } else {
      const crs_size_type cstart = d_csurfs.row_map(icell);
      list = &d_csurfs.entries(cstart);
      n = d_csurfs.row_map(icell+1) - cstart;
    }
    for (int j = 0; j < n; j++) {
      const int transparent = (dimk == 2) ? d_lines[list[j]].transparent :
        d_tris[list[j]].transparent;
      if (!transparent) return;
    }

    const double *clo = d_cells[icell].lo;
    const double *chi = d_cells[icell].hi;
    double ctr[3];
    ctr[0] = 0.5 * (clo[0] + chi[0]);
    ctr[1] = 0.5 * (clo[1] + chi[1]);
    if (dimk == 3) ctr[2] = 0.5 * (clo[2] + chi[2]);
    else ctr[2] = 0.0;

    int type = CELLOUTSIDE;
    int shell = 0;
    int done = 0;
    int blo[3],bhi[3];
    body.box_bins(clo,chi,blo,bhi);

    for (int ibz = blo[2]; ibz <= bhi[2] && !done; ibz++)
      for (int iby = blo[1]; iby <= bhi[1] && !done; iby++)
        for (int ibx = blo[0]; ibx <= bhi[0] && !done; ibx++) {
          const int ibin = (ibz*body.nbin[1] + iby)*body.nbin[0] + ibx;
          for (int mm = body.d_binstart(ibin); mm < body.d_binstart(ibin+1); mm++) {
            const int ib = body.d_binlist(mm);
            if (!body.box_overlap(ib,clo,chi)) continue;
            double d2 = 0.0;
            for (int kk = 0; kk < dimk; kk++) {
              const double dk = ctr[kk] - d_bodyparam(ib,6+kk);
              d2 += dk*dk;
            }
            const double rmin = d_bodyparam(ib,12);
            if (d2 < rmin*rmin) {
              if (d_cominside(ib)) {
                type = CELLINSIDE;
                shell = 0;
                done = 1;
                break;
              }
              continue;
            }
            const double rmax = d_bodyparam(ib,13) + d_bodyparam(ib,14);
            if (d2 <= rmax*rmax) shell = 1;
          }
        }

    if (shell) {
      if (body.inside_any_body(ctr)) type = CELLINSIDE;
      else type = CELLOUTSIDE;
    }
    if (d_cinfo[icell].type != type) d_newtype(ic) = type;
  });

  // the changed lists are packed as rows of one entries array, the
  //   cell of each row and its offset listed by a scan of the lengths

  Kokkos::fence();
  // nrcand is a member of RigidRemap, so naming it inside the lambda would
  //   capture this, a HOST pointer, and dereferencing it on the device is an
  //   illegal access (it aborted every rigid-body run on a GPU with
  //   cudaErrorIllegalAddress).  copy it to a local first

  {
    const int nrc = nrcand;
    Kokkos::parallel_scan("rigid_remap:rc_ch_scan",nrc, KOKKOS_LAMBDA(const int ic, int &sum,
                                             const bool final) {
      const int n = d_chflag(ic);
      if (final) d_choff(ic) = sum;
      sum += n;
      if (final && ic == nrc-1) d_choff(nrc) = sum;
    });
  }
  int nch;
  Kokkos::deep_copy(nch,Kokkos::subview(d_choff,nrcand));

  if (nch+1 > maxch_kk) {
    maxch_kk = nch+1;
    k_chcand = DAT::tdual_int_1d("rigid_remap:chcand",maxch_kk);
    k_chn = DAT::tdual_int_1d("rigid_remap:chn",maxch_kk);
    k_chloff = DAT::tdual_int_1d("rigid_remap:chloff",maxch_kk);
    k_chnsplit = DAT::tdual_int_1d("rigid_remap:chnsplit",maxch_kk);
    k_chcorner = DAT::tdual_int_1d("rigid_remap:chcorner",8*maxch_kk);
    k_chxsub = DAT::tdual_int_1d("rigid_remap:chxsub",maxch_kk);
    k_cherr = DAT::tdual_int_1d("rigid_remap:cherr",maxch_kk);
    k_chxsplit = tdual_dbl_1d("rigid_remap:chxsplit",3*maxch_kk);
  }
  auto d_chcand = k_chcand.view_device();
  auto d_chn = k_chn.view_device();
  auto d_chloff = k_chloff.view_device();

  Kokkos::parallel_for("rigid_remap:rc_ch_pack",nrcand, KOKKOS_LAMBDA(const int ic) {
    if (!d_chflag(ic)) return;
    const int c = d_choff(ic);
    d_chcand(c) = ic;
    d_chn(c) = d_newn(ic);
  });

  // row offsets of the lists: the piece maps, the piece volumes and the
  //   scratch rows of the device cut are laid out by the same offsets

  Kokkos::parallel_scan("rigid_remap:rc_ch_offscan",nch, KOKKOS_LAMBDA(const int c, int &sum,
                                            const bool final) {
    const int n = d_chn(c);
    if (final) d_chloff(c) = sum;
    sum += n;
    if (final && c == nch-1) d_chloff(nch) = sum;
  });
  int nent = 0;
  if (nch) Kokkos::deep_copy(nent,Kokkos::subview(d_chloff,nch));

  if (nent > maxchent_kk) {
    maxchent_kk = nent;
    k_chlist = DAT::tdual_int_1d("rigid_remap:chlist",maxchent_kk);
    k_chmap = DAT::tdual_int_1d("rigid_remap:chmap",maxchent_kk);
    k_chvols = tdual_dbl_1d("rigid_remap:chvols",maxchent_kk);
  }
  auto d_chlist = k_chlist.view_device();

  Kokkos::parallel_for("rigid_remap:rc_ch_fill",nch, KOKKOS_LAMBDA(const int c) {
    const int ic = d_chcand(c);
    const int n = d_chn(c);
    const int off = d_chloff(c);
    for (int j = 0; j < n; j++)
      d_chlist(off+j) = d_newlist(((bigint) ic) * maxsurf + j);
  });

  if (timeflag) {
    double now = MPI_Wtime();
    fix->add_time(FixRigid::T_RECUT_LISTS,now-tstart);
    tstart = now;
  }

  // the cut of every changed cell on the device, as recut_cell() cuts
  //   it on the host: no cut if only transparent surfs overlap it
  //   (nsplit = 0, the parity test of its center gives its type), else
  //   Cut2d/Cut3d::split() on scratch rows sized by its # of surfs,
  //   with UNKNOWN corner marks resolved by the same parity test
  // a cell the device cut fails on is cut again on the host, which
  //   raises the error message
  // the 3d scratch is large per cell, so the cells are cut in chunks
  //   which fit a memory budget; the row offsets are linear in the
  //   running # of surfs and of cells within the chunk

  auto d_chnsplit = k_chnsplit.view_device();
  auto d_chcorner = k_chcorner.view_device();
  auto d_chxsub = k_chxsub.view_device();
  auto d_cherr = k_cherr.view_device();
  auto d_chxsplit = k_chxsplit.view_device();
  auto d_chmap = k_chmap.view_device();
  auto d_chvols = k_chvols.view_device();

  if (dim == 2) {
    grow_cut_scratch(0,0,nent,2*nent + 4*nch);
    auto d_clines = this->d_clines;
    auto d_points = this->d_points;
    auto d_loops = this->d_loops;
    auto d_pgs = this->d_pgs;
    auto d_used = this->d_used;
    const int axisymmetric = domain->axisymmetric;
    const Surf::Line *lines = d_lines.data();

    Kokkos::parallel_for("rigid_remap:rc_cut2d",nch, KOKKOS_LAMBDA(const int c) {
      const int icell = d_rcand(d_chcand(c));
      const int n = d_chn(c);
      const int off = d_chloff(c);
      const int *list = &d_chlist(off);
      const double *clo = d_cells[icell].lo;
      const double *chi = d_cells[icell].hi;
      d_cherr(c) = 0;

      double ctr[3];
      ctr[0] = 0.5 * (clo[0] + chi[0]);
      ctr[1] = 0.5 * (clo[1] + chi[1]);
      ctr[2] = 0.0;

      int cut = 0;
      for (int j = 0; j < n; j++)
        if (!lines[list[j]].transparent) { cut = 1; break; }
      if (!cut) {
        d_chnsplit(c) = 0;
        if (body.inside_any_body(ctr)) d_chcorner(8*c) = CELLINSIDE;
        else d_chcorner(8*c) = CELLOUTSIDE;
        return;
      }

      const int poff = 2*off + 4*c;
      Cut2dKokkos cut2d(lines,axisymmetric,clo,chi,n,list,
                        &d_clines(off),&d_points(poff),&d_loops(poff),
                        &d_pgs(poff),&d_used(poff));

      int corner[4];
      int xsub = 0;
      int errflag;
      double xsplit[3];
      xsplit[0] = xsplit[1] = xsplit[2] = 0.0;
      double *vols = &d_chvols(off);
      int *map = &d_chmap(off);

      int nsplit = cut2d.split(vols,map,corner,xsub,xsplit,errflag);
      if (errflag) {
        d_cherr(c) = errflag;
        d_chnsplit(c) = 1;
        return;
      }

      if (corner[0] == CELLUNKNOWN) {
        int mark = CELLOUTSIDE;
        if (body.inside_any_body(ctr)) mark = CELLINSIDE;
        corner[0] = corner[1] = corner[2] = corner[3] = mark;
        nsplit = 1;
        if (mark == CELLINSIDE) vols[0] = 0.0;
        else if (axisymmetric)
          vols[0] = MY_PI * (chi[1]*chi[1]-clo[1]*clo[1]) * (chi[0]-clo[0]);
        else vols[0] = (chi[0]-clo[0]) * (chi[1]-clo[1]);
      }

      d_chnsplit(c) = nsplit;
      for (int k = 0; k < 4; k++) d_chcorner(8*c+k) = corner[k];
      d_chxsub(c) = xsub;
      for (int k = 0; k < 3; k++) d_chxsplit(3*c+k) = xsplit[k];
    });

  } else {
    k_chloff.modify_device(); k_chloff.sync_host();
    k_chn.modify_device(); k_chn.sync_host();
    auto h_chloff = k_chloff.view_host();
    auto h_chn = k_chn.view_host();
    const Surf::Tri *tris = d_tris.data();
    Kokkos::deep_copy(d_cutstats,0);

    // bytes of scratch per surf of a cell and per cell, from the row
    //   bounds at the head of cut3d_kokkos.h

    const bigint persurf =
      27 * (sizeof(Cut3dKokkos::Edge) + 2*sizeof(int)) +
      10 * (sizeof(Cut3dKokkos::Vertex) + sizeof(Cut3dKokkos::Loop) +
            sizeof(Cut3dKokkos::PH) + 2*sizeof(int)) +
      9 * sizeof(Cut2dKokkos::Cline) +
      18 * (sizeof(Cut2dKokkos::Point) + sizeof(Cut2dKokkos::Loop) +
            sizeof(Cut2dKokkos::PG) + sizeof(int));
    const bigint percell =
      24 * (sizeof(Cut3dKokkos::Edge) + 2*sizeof(int)) +
      6 * (sizeof(Cut3dKokkos::Vertex) + sizeof(Cut3dKokkos::Loop) +
           sizeof(Cut3dKokkos::PH) + 2*sizeof(int)) +
      4 * (sizeof(Cut2dKokkos::Point) + sizeof(Cut2dKokkos::Loop) +
           sizeof(Cut2dKokkos::PG) + sizeof(int));

    int c0 = 0;
    while (c0 < nch) {
      int c1 = c0 + 1;
      bigint bytes = persurf * h_chn(c0) + percell;
      while (c1 < nch) {
        bigint more = persurf * h_chn(c1) + percell;
        if (bytes + more > CUTSCRATCH) break;
        bytes += more;
        c1++;
      }

      const int base = h_chloff(c0);
      const int dl = h_chloff(c1) - base;
      const int dc = c1 - c0;
      grow_cut_scratch(10*dl + 6*dc,27*dl + 24*dc,9*dl,18*dl + 4*dc);
      auto d_verts = this->d_verts;
      auto d_edges = this->d_edges;
      auto d_loops3 = this->d_loops3;
      auto d_phs = this->d_phs;
      auto d_facelist = this->d_facelist;
      auto d_efaces = this->d_efaces;
      auto d_used3 = this->d_used3;
      auto d_stack = this->d_stack;
      auto d_clines = this->d_clines;
      auto d_points = this->d_points;
      auto d_loops = this->d_loops;
      auto d_pgs = this->d_pgs;
      auto d_used = this->d_used;
      auto d_cutstats = this->d_cutstats;
      const int cfirst = c0;

      Kokkos::parallel_for("rigid_remap:rc_cut3d",Kokkos::RangePolicy<DeviceType>(c0,c1),
                           KOKKOS_LAMBDA(const int c) {
        const int icell = d_rcand(d_chcand(c));
        const int n = d_chn(c);
        const int off = d_chloff(c);
        const int *list = &d_chlist(off);
        const double *clo = d_cells[icell].lo;
        const double *chi = d_cells[icell].hi;
        d_cherr(c) = 0;

        double ctr[3];
        ctr[0] = 0.5 * (clo[0] + chi[0]);
        ctr[1] = 0.5 * (clo[1] + chi[1]);
        ctr[2] = 0.5 * (clo[2] + chi[2]);

        int cut = 0;
        for (int j = 0; j < n; j++)
          if (!tris[list[j]].transparent) { cut = 1; break; }
        if (!cut) {
          d_chnsplit(c) = 0;
          if (body.inside_any_body(ctr)) d_chcorner(8*c) = CELLINSIDE;
          else d_chcorner(8*c) = CELLOUTSIDE;
          return;
        }

        const int u = off - base;
        const int k = c - cfirst;
        const int voff = 10*u + 6*k;
        const int eoff = 27*u + 24*k;
        const int coff = 9*u;
        const int poff = 18*u + 4*k;
        Cut3dKokkos cut3d(tris,clo,chi,n,list,
                          &d_verts(voff),&d_edges(eoff),&d_loops3(voff),
                          &d_phs(voff),&d_facelist(eoff),&d_efaces(eoff),
                          &d_used3(voff),&d_stack(voff),10*n + 6,27*n + 24,
                          &d_clines(coff),&d_points(poff),&d_loops(poff),
                          &d_pgs(poff),&d_used(poff),9*n);

        int corner[8];
        int xsub = 0;
        int errflag;
        double xsplit[3];
        xsplit[0] = xsplit[1] = xsplit[2] = 0.0;
        double *vols = &d_chvols(off);
        int *map = &d_chmap(off);

        int nsplit = cut3d.split(vols,map,corner,xsub,xsplit,errflag);
        if (cut3d.ntiny) Kokkos::atomic_add(&d_cutstats(0),cut3d.ntiny);
        if (cut3d.nshrink) Kokkos::atomic_add(&d_cutstats(1),cut3d.nshrink);
        if (errflag) {
          d_cherr(c) = errflag;
          d_chnsplit(c) = 1;
          return;
        }

        if (corner[0] == CELLUNKNOWN) {
          int mark = CELLOUTSIDE;
          if (body.inside_any_body(ctr)) mark = CELLINSIDE;
          for (int j = 0; j < 8; j++) corner[j] = mark;
          nsplit = 1;
          if (mark == CELLINSIDE) vols[0] = 0.0;
          else vols[0] = (chi[0]-clo[0]) * (chi[1]-clo[1]) * (chi[2]-clo[2]);
        }

        d_chnsplit(c) = nsplit;
        for (int j = 0; j < 8; j++) d_chcorner(8*c+j) = corner[j];
        d_chxsub(c) = xsub;
        for (int j = 0; j < 3; j++) d_chxsplit(3*c+j) = xsplit[j];
      });

      c0 = c1;
    }

    int cutstats[2];
    Kokkos::deep_copy(Kokkos::View<int[2],Kokkos::HostSpace>(cutstats),
                      d_cutstats);
    grid->add_cut3d_counts(cutstats[0],cutstats[1]);
  }

  // the changed lists and the results of their cuts come to the host

  k_chcand.modify_device(); k_chcand.sync_host();
  k_chn.modify_device(); k_chn.sync_host();
  k_chloff.modify_device(); k_chloff.sync_host();
  k_chlist.modify_device(); k_chlist.sync_host();
  k_rcand.modify_device(); k_rcand.sync_host();
  k_chnsplit.modify_device(); k_chnsplit.sync_host();
  k_chcorner.modify_device(); k_chcorner.sync_host();
  k_chxsub.modify_device(); k_chxsub.sync_host();
  k_cherr.modify_device(); k_cherr.sync_host();
  k_chxsplit.modify_device(); k_chxsplit.sync_host();
  k_chmap.modify_device(); k_chmap.sync_host();
  k_chvols.modify_device(); k_chvols.sync_host();

  auto h_rcand = k_rcand.view_host();
  auto h_chcand = k_chcand.view_host();
  auto h_chn = k_chn.view_host();
  auto h_chloff = k_chloff.view_host();
  auto h_chlist = k_chlist.view_host();
  auto h_chnsplit = k_chnsplit.view_host();
  auto h_chcorner = k_chcorner.view_host();
  auto h_chxsub = k_chxsub.view_host();
  auto h_cherr = k_cherr.view_host();
  auto h_chxsplit = k_chxsplit.view_host();
  auto h_chmap = k_chmap.view_host();
  auto h_chvols = k_chvols.view_host();
  for (i = 0; i < nrcand; i++) rcand[i] = h_rcand(i);

  // install the new list and the cut of every changed cell, in
  //   ascending cell order, exactly as the host loop does

  int corner[8];
  double xsplit[3];
  const int ncorner = (dim == 3) ? 8 : 4;

  for (m = 0; m < nch; m++) {
    icell = rcand[h_chcand(m)];
    int n = h_chn(m);
    int off = h_chloff(m);
    for (int j = 0; j < n; j++) newlist[j] = (surfint) h_chlist(off+j);
    nlist_run++;

    if (h_cherr(m)) {
      fix_kk->refresh_host_surfs();
      recut_cell(icell,n,newlist);
      continue;
    }

    grid->set_cell_surfs(icell,n,newlist);
    listschanged = 1;

    int nsplitone = h_chnsplit(m);
    if (nsplitone) ncut_run++;
    for (int j = 0; j < ncorner; j++) corner[j] = h_chcorner(8*m+j);
    if (nsplitone > 1)
      for (int j = 0; j < n; j++) newmap[j] = h_chmap(off+j);
    for (int j = 0; j < 3; j++) xsplit[j] = h_chxsplit(3*m+j);

    apply_cut(icell,nsplitone,&h_chvols(off),newmap,corner,
              h_chxsub(m),xsplit);
  }

  if (timeflag) {
    double now = MPI_Wtime();
    fix->add_time(FixRigid::T_RECUT_CUT,now-tstart);
    tstart = now;
  }

  // pass 2 on the host: the new types

  if ((int) k_newtype.extent(0) < nrcand) {
    k_newtype = DAT::tdual_int_1d("rigid_remap:newtypeh",nrcand);
  }
  Kokkos::deep_copy(Kokkos::subview(k_newtype.view_host(),std::make_pair(0,nrcand)),
                    Kokkos::subview(d_newtype,std::make_pair(0,nrcand)));
  auto h_newtype = k_newtype.view_host();
  for (i = 0; i < nrcand; i++) {
    int type = h_newtype(i);
    if (type < 0) continue;
    icell = rcand[i];
    if (cinfo[icell].type == type) continue;
    grid->set_cell_type(icell,type);
    typechanged = 1;
  }

  if (timeflag) fix->add_time(FixRigid::T_RECUT_TYPE,MPI_Wtime()-tstart);

  // the bodies' current bboxes bound the region on the next step

  for (int m = 0; m < nblist; m++) {
    ibody = blist[m];
    for (i = 0; i < 3; i++) {
      prevlo[ibody][i] = bbodylo[ibody][i];
      prevhi[ibody][i] = bbodyhi[ibody][i];
      prevxcm[ibody][i] = fix->xcm[ibody][i];
    }
  }

  return FALLBACK_NONE;
}
