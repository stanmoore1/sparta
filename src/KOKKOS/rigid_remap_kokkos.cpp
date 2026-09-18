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
#include "domain.h"
#include "memory_kokkos.h"
#include "sparta_masks.h"
#include "error.h"

using namespace SPARTA_NS;

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
  maxswcell_kk = maxswent_kk = 0;
  staticgen_kk = -1;
  maxrcand_kk = maxrcandlist_kk = maxch_kk = 0;
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
     sweeps through, as the mover's graph d_csurfs_move on the device
   the host lists are not built: nothing on the host reads them when
     the KOKKOS mover runs
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
  fix_kk->pack_body_device(1);

  int ntotal = grid->nlocal + grid->nghost;
  int nbody = fix->nbody;
  double **bbodylo = fix->bbodylo;
  double **bbodyhi = fix->bbodyhi;

  // (body, bin) pairs: the bins each body's swept box overlaps

  int npair = 0;
  int qlo[3],qhi[3];
  for (ibody = 0; ibody < nbody; ibody++) {
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
  for (ibody = 0; ibody < nbody; ibody++) {
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

  if (ntotal > maxswcell_kk) {
    maxswcell_kk = ntotal;
    d_swcount = DAT::t_int_1d("rigid_remap:swcount",maxswcell_kk);
    d_swoff = DAT::t_int_1d("rigid_remap:swoff",maxswcell_kk+1);
    d_swcursor = DAT::t_int_1d("rigid_remap:swcursor",maxswcell_kk);
    d_swext = DAT::t_int_1d("rigid_remap:swext",maxswcell_kk);
    d_subparent = DAT::t_int_1d("rigid_remap:subparent",maxswcell_kk);
    d_rowcount = Kokkos::View<crs_size_type*,DeviceType>("rigid_remap:rowcount",maxswcell_kk+1);
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
  auto d_swcount = this->d_swcount;
  auto d_swoff = this->d_swoff;
  auto d_swcursor = this->d_swcursor;
  auto d_swext = this->d_swext;
  auto d_subparent = this->d_subparent;
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

  // count the swept elements of each cell

  Kokkos::deep_copy(Kokkos::subview(d_swcount,std::make_pair(0,ntotal)),0);

  Kokkos::parallel_for(npair, KOKKOS_LAMBDA(const int p) {
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
      for (int e = body.d_bodystart(ibody); e < body.d_bodystart(ibody+1); e++)
        if (body.elem_overlap(e,clo,chi)) n++;
      if (n) Kokkos::atomic_add(&d_swcount(icell),n);
    }
  });

  Kokkos::parallel_scan(ntotal, KOKKOS_LAMBDA(const int icell, int &sum,
                                               const bool final) {
    const int n = d_swcount(icell);
    if (final) d_swoff(icell) = sum;
    sum += n;
    if (final && icell == ntotal-1) d_swoff(ntotal) = sum;
  });
  int nent;
  Kokkos::deep_copy(nent,Kokkos::subview(d_swoff,ntotal));

  if (!nent) {
    grid_kk->d_csurfs_move = grid_kk->d_csurfs;
    return;
  }

  if (nent > maxswent_kk) {
    maxswent_kk = nent;
    d_swelem = DAT::t_int_1d("rigid_remap:swelem",maxswent_kk);
  }
  auto d_swelem = this->d_swelem;
  Kokkos::deep_copy(Kokkos::subview(d_swcursor,std::make_pair(0,ntotal)),0);

  // fill: the same enumeration, each cell's elements in arrival order

  Kokkos::parallel_for(npair, KOKKOS_LAMBDA(const int p) {
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
      for (int e = body.d_bodystart(ibody); e < body.d_bodystart(ibody+1); e++) {
        if (!body.elem_overlap(e,clo,chi)) continue;
        const int j = Kokkos::atomic_fetch_add(&d_swcursor(icell),1);
        d_swelem(d_swoff(icell)+j) = e;
      }
    }
  });

  // per cell: the host chains its entries and reads the chain from its
  //   head, so the swept elements follow the cut list in descending
  //   element order; elements the cut list already holds are dropped

  auto d_csurfs = grid_kk->d_csurfs;

  Kokkos::parallel_for(ntotal, KOKKOS_LAMBDA(const int icell) {
    const int n = d_swcount(icell);
    if (!n) {
      d_swext(icell) = 0;
      return;
    }
    const int off = d_swoff(icell);
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
    d_swext(icell) = nextra;
  });

  // the split cell of every sub cell, which takes its split cell's
  //   swept elements; a split cell itself keeps its cut list

  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;
  int nsinfo = grid->nsplitlocal + grid->nsplitghost;
  int nsub = 0;
  for (i = 0; i < nsinfo; i++) {
    int icell = sinfo[i].icell;
    if (icell < 0 || icell >= ntotal) continue;
    if (cells[icell].nsplit <= 1 || cells[icell].isplit != i) continue;
    nsub += cells[icell].nsplit;
  }
  if ((int) k_subcell.extent(0) < MAX(nsub,1)) {
    k_subcell = DAT::tdual_int_1d("rigid_remap:subcell",MAX(nsub,1));
    k_subpar = DAT::tdual_int_1d("rigid_remap:subpar",MAX(nsub,1));
  }
  auto h_subcell = k_subcell.view_host();
  auto h_subpar = k_subpar.view_host();
  k = 0;
  for (i = 0; i < nsinfo; i++) {
    int icell = sinfo[i].icell;
    if (icell < 0 || icell >= ntotal) continue;
    if (cells[icell].nsplit <= 1 || cells[icell].isplit != i) continue;
    int *mycsubs = sinfo[i].csubs;
    for (int j = 0; j < cells[icell].nsplit; j++) {
      h_subcell(k) = mycsubs[j];
      h_subpar(k) = icell;
      k++;
    }
  }
  k_subcell.modify_host(); k_subcell.sync_device();
  k_subpar.modify_host(); k_subpar.sync_device();
  auto d_subcell = k_subcell.view_device();
  auto d_subpar = k_subpar.view_device();

  Kokkos::deep_copy(Kokkos::subview(d_subparent,std::make_pair(0,ntotal)),-1);
  Kokkos::parallel_for(nsub, KOKKOS_LAMBDA(const int m) {
    d_subparent(d_subcell(m)) = d_subpar(m);
  });

  // the mover's graph: cut list plus the swept elements of the cell, or
  //   of its split cell for a sub cell

  auto d_rowcount = this->d_rowcount;
  Kokkos::parallel_for(ntotal, KOKKOS_LAMBDA(const int icell) {
    crs_size_type n = d_csurfs.row_map(icell+1) - d_csurfs.row_map(icell);
    const int nsplit = d_cells[icell].nsplit;
    if (nsplit == 1) n += d_swext(icell);
    else if (nsplit <= 0) {
      const int par = d_subparent(icell);
      if (par >= 0) n += d_swext(par);
    }
    d_rowcount(icell) = n;
  });

  if ((int) d_rowmap_move.extent(0) < ntotal+1)
    d_rowmap_move = Kokkos::View<crs_size_type*,DeviceType>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,"rigid_remap:rowmap_move"),
      ntotal+1);
  auto rowmap = d_rowmap_move;
  Kokkos::parallel_scan(ntotal, KOKKOS_LAMBDA(const int icell, crs_size_type &sum,
                                               const bool final) {
    const crs_size_type n = d_rowcount(icell);
    if (final) rowmap(icell) = sum;
    sum += n;
    if (final && icell == ntotal-1) rowmap(ntotal) = sum;
  });
  crs_size_type nentries;
  Kokkos::deep_copy(nentries,Kokkos::subview(rowmap,ntotal));

  if ((bigint) d_entries_move.extent(0) < (bigint) nentries)
    d_entries_move = DAT::t_int_1d(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,"rigid_remap:entries_move"),
      nentries);
  auto entries = d_entries_move;

  Kokkos::parallel_for(ntotal, KOKKOS_LAMBDA(const int icell) {
    const crs_size_type cstart = d_csurfs.row_map(icell);
    const int ncut = d_csurfs.row_map(icell+1) - cstart;
    crs_size_type start = rowmap(icell);
    for (int j = 0; j < ncut; j++) entries(start+j) = d_csurfs.entries(cstart+j);
    start += ncut;
    const int nsplit = d_cells[icell].nsplit;
    int src = -1;
    if (nsplit == 1) src = icell;
    else if (nsplit <= 0) src = d_subparent(icell);
    if (src < 0) return;
    const int nextra = d_swext(src);
    const int off = d_swoff(src);
    for (int j = 0; j < nextra; j++) entries(start+j) = d_swelem(off+j);
  });
  Kokkos::fence();

  grid_kk->d_csurfs_move.row_map = rowmap;
  grid_kk->d_csurfs_move.entries = entries;
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
  surf_kk->modify(Host,ALL_MASK);
  surf_kk->sync(Device,ALL_MASK);
  fix_kk->pack_body_device(0);

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
  for (ibody = 0; ibody < nbody; ibody++) {
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
  for (ibody = 0; ibody < nbody; ibody++)
    for (int ibz = h_qlo(ibody,2); ibz <= h_qhi(ibody,2); ibz++)
      for (int iby = h_qlo(ibody,1); iby <= h_qhi(ibody,1); iby++)
        for (int ibx = h_qlo(ibody,0); ibx <= h_qhi(ibody,0); ibx++) {
          h_pairbody(npair) = ibody;
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

  Kokkos::parallel_for(npair, KOKKOS_LAMBDA(const int p) {
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

  Kokkos::parallel_scan(nglocal, KOKKOS_LAMBDA(const int ic, int &sum,
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

  Kokkos::parallel_for(nglocal, KOKKOS_LAMBDA(const int ic) {
    if (d_candflag(ic)) d_rcand(d_candoff(ic)) = ic;
  });

  if (!nrcand) {
    for (ibody = 0; ibody < nbody; ibody++)
      for (i = 0; i < 3; i++) {
        prevlo[ibody][i] = bbodylo[ibody][i];
        prevhi[ibody][i] = bbodyhi[ibody][i];
        prevxcm[ibody][i] = fix->xcm[ibody][i];
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

  Kokkos::parallel_for(nrcand, KOKKOS_LAMBDA(const int ic) {
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

  Kokkos::parallel_for(nrcand, KOKKOS_LAMBDA(const int ic) {
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

  // the changed lists and the new types come to the host, compactly

  Kokkos::parallel_scan(nrcand, KOKKOS_LAMBDA(const int ic, int &sum,
                                               const bool final) {
    const int n = d_chflag(ic);
    if (final) d_choff(ic) = sum;
    sum += n;
    if (final && ic == nrcand-1) d_choff(nrcand) = sum;
  });
  int nch;
  Kokkos::deep_copy(nch,Kokkos::subview(d_choff,nrcand));

  if (nch > maxch_kk) {
    maxch_kk = nch;
    k_chcand = DAT::tdual_int_1d("rigid_remap:chcand",maxch_kk);
    k_chn = DAT::tdual_int_1d("rigid_remap:chn",maxch_kk);
    k_chlist = DAT::tdual_int_1d("rigid_remap:chlist",
                                 (bigint) maxch_kk * maxsurfpercell);
  }
  auto d_chcand = k_chcand.view_device();
  auto d_chn = k_chn.view_device();
  auto d_chlist = k_chlist.view_device();
  Kokkos::parallel_for(nrcand, KOKKOS_LAMBDA(const int ic) {
    if (!d_chflag(ic)) return;
    const int c = d_choff(ic);
    d_chcand(c) = ic;
    const int n = d_newn(ic);
    d_chn(c) = n;
    for (int j = 0; j < n; j++)
      d_chlist(((bigint) c) * maxsurf + j) = d_newlist(((bigint) ic) * maxsurf + j);
  });
  k_chcand.modify_device(); k_chcand.sync_host();
  k_chn.modify_device(); k_chn.sync_host();
  k_chlist.modify_device(); k_chlist.sync_host();
  k_rcand.modify_device(); k_rcand.sync_host();

  auto h_rcand = k_rcand.view_host();
  auto h_chcand = k_chcand.view_host();
  auto h_chn = k_chn.view_host();
  auto h_chlist = k_chlist.view_host();
  for (i = 0; i < nrcand; i++) rcand[i] = h_rcand(i);

  if (timeflag) {
    double now = MPI_Wtime();
    fix->add_time(FixRigid::T_RECUT_LISTS,now-tstart);
    tstart = now;
  }

  // pass 1 on the host: cut the cells whose list changed, in ascending
  //   cell order, exactly as the host loop does

  for (m = 0; m < nch; m++) {
    int ic = h_chcand(m);
    icell = rcand[ic];
    int n = h_chn(m);
    for (int j = 0; j < n; j++)
      newlist[j] = (surfint) h_chlist(((bigint) m) * maxsurfpercell + j);
    nlist_run++;
    recut_cell(icell,n,newlist);
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

  for (ibody = 0; ibody < nbody; ibody++)
    for (i = 0; i < 3; i++) {
      prevlo[ibody][i] = bbodylo[ibody][i];
      prevhi[ibody][i] = bbodyhi[ibody][i];
      prevxcm[ibody][i] = fix->xcm[ibody][i];
    }

  return FALLBACK_NONE;
}
