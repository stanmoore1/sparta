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
#include "domain.h"
#include "memory_kokkos.h"
#include "error.h"

using namespace SPARTA_NS;

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
  auto d_bbodylo = fix_kk->d_bbodylo;
  auto d_bbodyhi = fix_kk->d_bbodyhi;
  auto d_bodystart = fix_kk->d_bodystart;
  auto d_elemlo = fix_kk->d_elemlo;
  auto d_elemhi = fix_kk->d_elemhi;
  auto d_lblist = fix_kk->d_lblist;
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
    double blo[3],bhi[3];
    for (int kk = 0; kk < 3; kk++) {
      blo[kk] = d_bbodylo(ibody,kk);
      bhi[kk] = d_bbodyhi(ibody,kk);
    }
    for (int m = d_binstart(ibin); m < d_binstart(ibin+1); m++) {
      const int icell = d_binlist(m);
      if (d_cells[icell].nsplit <= 0) continue;
      if (d_cells[icell].nsurf < 0) continue;
      if (!first_bin(icell,ibody,ibin)) continue;
      const double *clo = d_cells[icell].lo;
      const double *chi = d_cells[icell].hi;
      if (!box_overlap_kk(clo,chi,blo,bhi)) continue;
      int n = 0;
      for (int e = d_bodystart(ibody); e < d_bodystart(ibody+1); e++) {
        double elo[3],ehi[3];
        for (int kk = 0; kk < 3; kk++) {
          elo[kk] = d_elemlo(e,kk);
          ehi[kk] = d_elemhi(e,kk);
        }
        if (box_overlap_kk(clo,chi,elo,ehi)) n++;
      }
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
    double blo[3],bhi[3];
    for (int kk = 0; kk < 3; kk++) {
      blo[kk] = d_bbodylo(ibody,kk);
      bhi[kk] = d_bbodyhi(ibody,kk);
    }
    for (int m = d_binstart(ibin); m < d_binstart(ibin+1); m++) {
      const int icell = d_binlist(m);
      if (d_cells[icell].nsplit <= 0) continue;
      if (d_cells[icell].nsurf < 0) continue;
      if (!first_bin(icell,ibody,ibin)) continue;
      const double *clo = d_cells[icell].lo;
      const double *chi = d_cells[icell].hi;
      if (!box_overlap_kk(clo,chi,blo,bhi)) continue;
      for (int e = d_bodystart(ibody); e < d_bodystart(ibody+1); e++) {
        double elo[3],ehi[3];
        for (int kk = 0; kk < 3; kk++) {
          elo[kk] = d_elemlo(e,kk);
          ehi[kk] = d_elemhi(e,kk);
        }
        if (!box_overlap_kk(clo,chi,elo,ehi)) continue;
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
      const int s = d_lblist(d_swelem(off+a));
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
