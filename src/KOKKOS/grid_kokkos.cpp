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
#include "grid_kokkos.h"
#include "geometry.h"
#include "domain.h"
#include "comm.h"
#include "irregular.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "error.h"
#include "kokkos.h"
#include "sparta_masks.h"
#include "surf_kokkos.h"
#include "particle_kokkos.h"

#include <type_traits>

using namespace SPARTA_NS;
using namespace MathConst;

#define DELTA 8192
#define DELTAPARENT 1024
#define BIG 1.0e20
#define MAXLEVEL 32

enum{XLO,XHI,YLO,YHI,ZLO,ZHI,INTERIOR};         // same as Domain
enum{PERIODIC,OUTFLOW,REFLECT,SURFACE,AXISYM};  // same as Domain
enum{REGION_ALL,REGION_ONE,REGION_CENTER};      // same as Surf

// cell is entirely outside/inside surfs or has any overlap with surfs
// corner pt is outside/inside surfs or is on a surf

enum{UNKNOWN,OUTSIDE,INSIDE,OVERLAP};           // several files
enum{NCHILD,NPARENT,NUNKNOWN,NPBCHILD,NPBPARENT,NPBUNKNOWN,NBOUND};  // Update
enum{NOWEIGHT,VOLWEIGHT,RADWEIGHT,RADUNWEIGHT};

// allocate space for static class variable

//Grid *Grid::gptr;

// corners[i][j] = J corner points of face I of a grid cell
// works for 2d quads and 3d hexes

//int corners[6][4] = {{0,2,4,6}, {1,3,5,7}, {0,1,4,5}, {2,3,6,7},
//                     {0,1,2,3}, {4,5,6,7}};

/* ---------------------------------------------------------------------- */

GridKokkos::GridKokkos(SPARTA *sparta) : Grid(sparta)
{
  delete [] plevels;
  memoryKK->create_kokkos(k_plevels,plevels,MAXLEVEL,"grid:plevels");

  k_eivec = tdual_struct_tdual_int_1d_1d("grid:eivec",0);
  k_eiarray = tdual_struct_tdual_int_2d_1d("grid:eiarray",0);
  k_edvec = tdual_struct_tdual_float_1d_1d("grid:edvec",0);
  k_edarray = tdual_struct_tdual_float_2d_1d("grid:edarray",0);

  journalflag = 1;
  ncsurfsrows = 0;
  graph_generation = 0;
  cellbingen_kk = -1;
  dirtystamp = sinfostamp = NULL;
  maxdirtystamp = maxsinfostamp = 0;
  dirtygen = 0;
  ibuf = 0;
}

GridKokkos::~GridKokkos()
{
  if (copy || copymode) return;

  memory->destroy(dirtystamp);
  memory->destroy(sinfostamp);

  cells = NULL;
  cinfo = NULL;
  sinfo = NULL;
  pcells = NULL;
  plevels = NULL;

  ewhich = NULL;
  eicol = NULL;
  edcol = NULL;

  for (int i = 0; i < ncustom_ivec; i++)
    memoryKK->destroy_kokkos(k_eivec.view_host()[i].k_view,eivec[i]);
  for (int i = 0; i < ncustom_iarray; i++)
    memoryKK->destroy_kokkos(k_eiarray.view_host()[i].k_view,eiarray[i]);
  for (int i = 0; i < ncustom_dvec; i++)
    memoryKK->destroy_kokkos(k_edvec.view_host()[i].k_view,edvec[i]);
  for (int i = 0; i < ncustom_darray; i++)
    memoryKK->destroy_kokkos(k_edarray.view_host()[i].k_view,edarray[i]);

  ncustom_ivec = ncustom_iarray = 0;
  ncustom_dvec = ncustom_darray = 0;
}

///////////////////////////////////////////////////////////////////////////
// grow cell list data structures
///////////////////////////////////////////////////////////////////////////

/* ----------------------------------------------------------------------
   insure cells and cinfo can hold N and M new cells respectively
------------------------------------------------------------------------- */

void GridKokkos::grow_cells(int n, int m)
{
  if (sparta->kokkos->prewrap) {
    Grid::grow_cells(n,m);
  } else {

    if (nlocal+nghost+n >= maxcell) {
      const int oldmax = maxcell;
      while (maxcell < nlocal+nghost+n) maxcell += DELTA;
      if (cells == NULL)
        MemKK::realloc_kokkos(k_cells,"grid:cells",maxcell);
      else {
        this->modify(Host,CELL_MASK);  // the host is authoritative
        this->sync(Device,CELL_MASK);  // force resize on device
        Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                       k_cells,maxcell);
        this->modify(Device,CELL_MASK); // needed for auto sync
      }
      cells = k_cells.view_host().data();

      if (ncustom) reallocate_custom(oldmax,maxcell);
    }

    if (nlocal+m >= maxlocal) {
      while (maxlocal < nlocal+m) maxlocal += DELTA;
      if (cinfo == NULL)
        MemKK::realloc_kokkos(k_cinfo,"grid:cinfo",maxlocal);
      else {
        this->modify(Host,CINFO_MASK);  // the host is authoritative
        this->sync(Device,CINFO_MASK);  // force resize on device
        Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                       k_cinfo,maxlocal);
        this->modify(Device,CINFO_MASK); // needed for auto sync
      }
      cinfo = k_cinfo.view_host().data();
    }
  }
}

/* ----------------------------------------------------------------------
   grow pcells
------------------------------------------------------------------------- */

void GridKokkos::grow_pcells()
{
  if (sparta->kokkos->prewrap) {
    Grid::grow_pcells();
  } else {

    maxparent += DELTA;
    if (pcells == NULL)
      MemKK::realloc_kokkos(k_pcells,"grid:pcells",maxparent);
    else {
      this->modify(Host,PCELL_MASK);  // the host is authoritative
      this->sync(Device,PCELL_MASK);  // force resize on device
      Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                     k_pcells,maxparent);
      this->modify(Device,PCELL_MASK); // needed for auto sync
    }
    pcells = k_pcells.view_host().data();
  }
}

/* ----------------------------------------------------------------------
   insure sinfo can hold N new split cells
------------------------------------------------------------------------- */

void GridKokkos::grow_sinfo(int n)
{
  if (sparta->kokkos->prewrap) {
    Grid::grow_sinfo(n);
  } else {

    if (nsplitlocal+nsplitghost+n >= maxsplit) {
      while (maxsplit < nsplitlocal+nsplitghost+n) maxsplit += DELTA;
      if (sinfo == NULL)
        MemKK::realloc_kokkos(k_sinfo,"grid:sinfo",maxsplit);
      else {
        this->modify(Host,SINFO_MASK);  // the host is authoritative
        this->sync(Device,SINFO_MASK);  // force resize on device
        Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing),
                       k_sinfo,maxsplit);
        this->modify(Device,SINFO_MASK); // needed for auto sync
      }
      sinfo = k_sinfo.view_host().data();
    }
  }
}

/* ---------------------------------------------------------------------- */

void GridKokkos::wrap_kokkos_graphs()
{
  if (!surf->exist) return;

  // csurfs = the cut list of every cell; ghost cells with nsurf < 0 (empty)
  //   have an empty row.  d_csurfs doesn't need to be surfint because at
  //   this point only local (not global) ids are stored in csurfs

  Kokkos::Crs<int, SPAHostType, void, crs_size_type> h_csurfs;
  auto csurfs_lambda = [&](int icell, int* fill) {
    int nsurf = cells[icell].nsurf;
    surfint* csurfs = cells[icell].csurfs;
    if (nsurf < 0) nsurf = 0;
    else if (fill)
      for (int j = 0; j < nsurf; ++j) fill[j] = (int) csurfs[j];
    return nsurf;
  };
  Kokkos::count_and_fill_crs(h_csurfs, nlocal+nghost, csurfs_lambda);
  d_csurfs.row_map = decltype(d_csurfs.row_map)(
      "csurfs.row_map", h_csurfs.row_map.size());
  d_csurfs.entries = decltype(d_csurfs.entries)(
      "csurfs.entries", h_csurfs.entries.size());
  Kokkos::deep_copy(d_csurfs.row_map, h_csurfs.row_map);
  Kokkos::deep_copy(d_csurfs.entries, h_csurfs.entries);
  ncsurfsrows = nlocal+nghost;
  d_rowmap_buf[ibuf] = d_csurfs.row_map;
  d_entries_buf[ibuf] = d_csurfs.entries;

  // the mover's graph: the collision lists of the cells which have one

  if (ncollcells) {
    Kokkos::Crs<int, SPAHostType, void, crs_size_type> h_move;
    auto move_lambda = [&](int icell, int* fill) {
      int nsurf = cells[icell].nsurf;
      surfint* csurfs = cells[icell].csurfs;
      if (cells[icell].ccoll) {
        nsurf = cells[icell].ncoll;
        csurfs = cells[icell].ccoll;
      }
      if (nsurf < 0) nsurf = 0;
      else if (fill)
        for (int j = 0; j < nsurf; ++j) fill[j] = (int) csurfs[j];
      return nsurf;
    };
    Kokkos::count_and_fill_crs(h_move, nlocal+nghost, move_lambda);
    d_csurfs_move.row_map = decltype(d_csurfs_move.row_map)(
        "csurfs_move.row_map", h_move.row_map.size());
    d_csurfs_move.entries = decltype(d_csurfs_move.entries)(
        "csurfs_move.entries", h_move.entries.size());
    Kokkos::deep_copy(d_csurfs_move.row_map, h_move.row_map);
    Kokkos::deep_copy(d_csurfs_move.entries, h_move.entries);
  } else d_csurfs_move = d_csurfs;

  wrap_split_graphs();
  graph_generation++;

  // the host state is now on the device in full

  journal_clear();
}

/* ----------------------------------------------------------------------
   d_csplits/d_csubs from the host split info, owned + ghost
   small: one row per split cell
------------------------------------------------------------------------- */

void GridKokkos::wrap_split_graphs()
{
  if (sinfo == NULL) return;

  Kokkos::Crs<int, SPAHostType, void, crs_size_type> h_csplits;
  auto csplits_lambda = [&](int isplit, int* fill) {
    int icell = sinfo[isplit].icell;
    if (icell < 0) return 0;
    int nsurf = cells[icell].nsurf;
    int nsplit = cells[icell].nsplit;
    if (nsurf < 0 || nsplit <= 1 || cells[icell].isplit != isplit) nsurf = 0;
    else if (fill) {
      int* csplits = sinfo[isplit].csplits;
      for (int j = 0; j < nsurf; ++j) fill[j] = csplits[j];
    }
    return nsurf;
  };
  Kokkos::count_and_fill_crs(h_csplits, nsplitlocal+nsplitghost, csplits_lambda);
  d_csplits.row_map = decltype(d_csplits.row_map)(
      "csplits.row_map", h_csplits.row_map.size());
  d_csplits.entries = decltype(d_csplits.entries)(
      "csplits.entries", h_csplits.entries.size());
  Kokkos::deep_copy(d_csplits.row_map, h_csplits.row_map);
  Kokkos::deep_copy(d_csplits.entries, h_csplits.entries);

  Kokkos::Crs<int, SPAHostType, void, crs_size_type> h_csubs;
  auto csubs_lambda = [&](int isplit, int* fill) {
    int icell = sinfo[isplit].icell;
    if (icell < 0) return 0;
    int nsurf = cells[icell].nsurf;
    int nsplit = cells[icell].nsplit;
    if (nsurf < 0 || nsplit <= 1 || cells[icell].isplit != isplit) nsplit = 0;
    else if (fill) {
      int* csubs = sinfo[isplit].csubs;
      for (int j = 0; j < nsplit; ++j) fill[j] = csubs[j];
    }
    return nsplit;
  };
  Kokkos::count_and_fill_crs(h_csubs, nsplitlocal+nsplitghost, csubs_lambda);
  d_csubs.row_map = decltype(d_csubs.row_map)(
      "csubs.row_map", h_csubs.row_map.size());
  d_csubs.entries = decltype(d_csubs.entries)(
      "csubs.entries", h_csubs.entries.size());
  Kokkos::deep_copy(d_csubs.row_map, h_csubs.row_map);
  Kokkos::deep_copy(d_csubs.entries, h_csubs.entries);
}

/* ----------------------------------------------------------------------
   patch the device copy of the grid with the change journal: the cells,
     split info, hash and halo entries the host primitives changed, and
     the per-cell surf graphs, rebuilt on the device from their old rows
     and the recorded lists
   the host is authoritative and the device is brought level with it,
     so neither DualView side is flagged modified afterwards
   O(changed) transfers and O(cells) device work, no host loop over the
     grid: the alternative, modify(Host) + wrap_kokkos_graphs(), is
     O(cells) host work and traffic on every call
------------------------------------------------------------------------- */

void GridKokkos::apply_changes()
{
  int i,k,icell;

  if (sparta->kokkos->prewrap) return;
  int ntotal = nlocal + nghost;

  // cells and their ChildInfo, deduplicated

  int *unique;
  int nunique = stage_records(ndirtycell,dirtycell,&unique);

  if (nunique) {
    if ((int) k_stagecell.extent(0) < nunique) {
      k_stagecell = tdual_cell_1d("grid:stagecell",nunique);
      k_stagecinfo = tdual_cinfo_1d("grid:stagecinfo",nunique);
      k_dirtycell = DAT::tdual_int_1d("grid:dirtycell",nunique);
      k_dirtyown = DAT::tdual_int_1d("grid:dirtyown",nunique);
      k_hashid = DAT::tdual_cellint_1d("grid:hashid",nunique);
      k_hashidx = DAT::tdual_int_1d("grid:hashidx",nunique);
      k_halosite = DAT::tdual_int_1d("grid:halosite",nunique);
      k_haloidx = DAT::tdual_int_1d("grid:haloidx",nunique);
    }
    auto h_stagecell = k_stagecell.view_host();
    auto h_stagecinfo = k_stagecinfo.view_host();
    auto h_dirtycell = k_dirtycell.view_host();
    auto h_dirtyown = k_dirtyown.view_host();
    auto h_hashid = k_hashid.view_host();
    auto h_hashidx = k_hashidx.view_host();
    auto h_halosite = k_halosite.view_host();
    auto h_haloidx = k_haloidx.view_host();

    int nhash = 0;
    int nhalo = 0;
    for (k = 0; k < nunique; k++) {
      icell = unique[k];
      h_dirtycell(k) = icell;
      h_stagecell(k) = cells[icell];
      if (icell < nlocal) {
        h_dirtyown(k) = 1;
        h_stagecinfo(k) = cinfo[icell];
      } else h_dirtyown(k) = 0;

      // a live cell with an ID of its own: its hash and halo entries

      if (icell < ntotal && cells[icell].nsplit >= 1 &&
          cells[icell].proc != -1) {
        h_hashid(nhash) = cells[icell].id;
        h_hashidx(nhash) = icell;
        nhash++;
        if (halo_index) {
          int site = halo_site(icell);
          if (site >= 0) {
            h_halosite(nhalo) = site;
            h_haloidx(nhalo) = icell;
            nhalo++;
          }
        }
      }
    }

    k_stagecell.modify_host(); k_stagecell.sync_device();
    k_stagecinfo.modify_host(); k_stagecinfo.sync_device();
    k_dirtycell.modify_host(); k_dirtycell.sync_device();
    k_dirtyown.modify_host(); k_dirtyown.sync_device();

    auto d_cells = k_cells.view_device();
    auto d_cinfo = k_cinfo.view_device();
    auto d_stagecell = k_stagecell.view_device();
    auto d_stagecinfo = k_stagecinfo.view_device();
    auto d_dirtycell = k_dirtycell.view_device();
    auto d_dirtyown = k_dirtyown.view_device();
    Kokkos::parallel_for(nunique, KOKKOS_LAMBDA(const int m) {
      const int ic = d_dirtycell(m);
      d_cells(ic) = d_stagecell(m);
      if (d_dirtyown(m)) d_cinfo(ic) = d_stagecinfo(m);
    });

    if (nhash) {
      k_hashid.modify_host(); k_hashid.sync_device();
      k_hashidx.modify_host(); k_hashidx.sync_device();
      auto d_hashid = k_hashid.view_device();
      auto d_hashidx = k_hashidx.view_device();
      auto hash_d = hash_kk;
      Kokkos::parallel_for(nhash, KOKKOS_LAMBDA(const int m) {
        auto h = hash_d.find(static_cast<key_type>(d_hashid(m)));
        if (hash_d.valid_at(h)) hash_d.value_at(h) = d_hashidx(m);
      });
    }

    if (nhalo && d_halo_index.extent(0)) {
      k_halosite.modify_host(); k_halosite.sync_device();
      k_haloidx.modify_host(); k_haloidx.sync_device();
      auto d_halosite = k_halosite.view_device();
      auto d_haloidx = k_haloidx.view_device();
      auto d_halo = d_halo_index;
      Kokkos::parallel_for(nhalo, KOKKOS_LAMBDA(const int m) {
        d_halo(d_halosite(m)) = d_haloidx(m);
      });
    }
  }

  // split info

  int nsunique = 0;
  int *sunique = NULL;
  if (ndirtysinfo) {
    if (maxsplit > maxsinfostamp) {
      memory->destroy(sinfostamp);
      maxsinfostamp = maxsplit;
      memory->create(sinfostamp,maxsinfostamp,"grid:sinfostamp");
      for (i = 0; i < maxsinfostamp; i++) sinfostamp[i] = 0;
    }
    int gen = ++dirtygen;
    memory->create(sunique,ndirtysinfo,"grid:sunique");
    for (i = 0; i < ndirtysinfo; i++) {
      int isplit = dirtysinfo[i];
      if (isplit < 0 || isplit >= maxsplit) continue;
      if (sinfostamp[isplit] == gen) continue;
      sinfostamp[isplit] = gen;
      sunique[nsunique++] = isplit;
    }
    if (nsunique) {
      if ((int) k_stagesinfo.extent(0) < nsunique) {
        k_stagesinfo = tdual_sinfo_1d("grid:stagesinfo",nsunique);
        k_dirtysinfo = DAT::tdual_int_1d("grid:dirtysinfo",nsunique);
      }
      auto h_stagesinfo = k_stagesinfo.view_host();
      auto h_dirtysinfo = k_dirtysinfo.view_host();
      for (k = 0; k < nsunique; k++) {
        h_dirtysinfo(k) = sunique[k];
        h_stagesinfo(k) = sinfo[sunique[k]];
      }
      k_stagesinfo.modify_host(); k_stagesinfo.sync_device();
      k_dirtysinfo.modify_host(); k_dirtysinfo.sync_device();
      auto d_sinfo = k_sinfo.view_device();
      auto d_stagesinfo = k_stagesinfo.view_device();
      auto d_dirtysinfo = k_dirtysinfo.view_device();
      Kokkos::parallel_for(nsunique, KOKKOS_LAMBDA(const int m) {
        d_sinfo(d_dirtysinfo(m)) = d_stagesinfo(m);
      });
    }
    memory->destroy(sunique);
  }

  // the cell bins: a moved cell is replaced in its bins, if the device
  //   copy is the one the host patched

  if (nbinpatch && cellbingen_kk == cellbingen) {
    if ((int) k_movedfrom.extent(0) < nbinpatch) {
      k_movedfrom = DAT::tdual_int_1d("grid:movedfrom",nbinpatch);
      k_movedto = DAT::tdual_int_1d("grid:movedto",nbinpatch);
    }
    auto h_from = k_movedfrom.view_host();
    auto h_to = k_movedto.view_host();
    for (i = 0; i < nbinpatch; i++) {
      h_from(i) = binpatchfrom[i];
      h_to(i) = binpatchto[i];
    }
    k_movedfrom.modify_host(); k_movedfrom.sync_device();
    k_movedto.modify_host(); k_movedto.sync_device();
    auto d_from = k_movedfrom.view_device();
    auto d_to = k_movedto.view_device();
    auto d_binstart = d_cellbinstart;
    auto d_binlist = d_cellbinlist;
    auto d_cells = k_cells.view_device();
    const int nbinx = cellnbin[0], nbiny = cellnbin[1], nbinz = cellnbin[2];
    const double blo0 = cellbinlo[0], blo1 = cellbinlo[1], blo2 = cellbinlo[2];
    const double binv0 = cellbininv[0], binv1 = cellbininv[1],
      binv2 = cellbininv[2];
    int npatch = nbinpatch;
    Kokkos::parallel_for(npatch, KOKKOS_LAMBDA(const int m) {
      const int src = d_from(m);
      const int dst = d_to(m);
      const double *lo = d_cells[dst].lo;
      const double *hi = d_cells[dst].hi;
      int clo[3],chi[3];
      clo[0] = MAX(0,MIN((int) ((lo[0]-blo0)*binv0),nbinx-1));
      chi[0] = MAX(0,MIN((int) ((hi[0]-blo0)*binv0),nbinx-1));
      clo[1] = MAX(0,MIN((int) ((lo[1]-blo1)*binv1),nbiny-1));
      chi[1] = MAX(0,MIN((int) ((hi[1]-blo1)*binv1),nbiny-1));
      clo[2] = MAX(0,MIN((int) ((lo[2]-blo2)*binv2),nbinz-1));
      chi[2] = MAX(0,MIN((int) ((hi[2]-blo2)*binv2),nbinz-1));
      for (int ibz = clo[2]; ibz <= chi[2]; ibz++)
        for (int iby = clo[1]; iby <= chi[1]; iby++)
          for (int ibx = clo[0]; ibx <= chi[0]; ibx++) {
            const int ibin = (ibz*nbiny + iby)*nbinx + ibx;
            for (int j = d_binstart(ibin); j < d_binstart(ibin+1); j++)
              if (d_binlist(j) == src) d_binlist(j) = dst;
          }
    });
    Kokkos::fence();
  }

  // the graphs: the cut graph when a list or the cell layout changed,
  //   the split graphs when a split cell changed, the mover's graph
  //   when collision lists were set or reset

  int structural = (nmoved > 0 || ntotal != ncsurfsrows);
  if (surf->exist) {
    if (ncutrec || structural) build_csurfs_device();
    if (nsunique || structural) wrap_split_graphs();
    if (ncollrec) build_move_graph_device();
    else if (collreset || ncutrec || structural) d_csurfs_move = d_csurfs;
    graph_generation++;
  }

  memory->destroy(unique);
  journal_clear();
}

/* ----------------------------------------------------------------------
   the device copy of the cell bin index, built by the host on demand
     and re-copied when it was rebuilt; the patches of moved cells are
     applied by apply_changes() in between
------------------------------------------------------------------------- */

void GridKokkos::sync_cell_bins()
{
  if (!cellbinvalid) build_cell_bins();
  if (cellbingen_kk == cellbingen) return;

  int nbins = cellnbin[0]*cellnbin[1]*cellnbin[2];
  int nlist = cellbinstart[nbins];
  if ((int) k_cellbinstart.extent(0) < nbins+1)
    k_cellbinstart = DAT::tdual_int_1d("grid:cellbinstart",nbins+1);
  if ((int) k_cellbinlist.extent(0) < MAX(nlist,1))
    k_cellbinlist = DAT::tdual_int_1d("grid:cellbinlist",MAX(nlist,1));
  auto h_start = k_cellbinstart.view_host();
  auto h_list = k_cellbinlist.view_host();
  for (int i = 0; i <= nbins; i++) h_start(i) = cellbinstart[i];
  for (int i = 0; i < nlist; i++) h_list(i) = cellbinlist[i];
  k_cellbinstart.modify_host(); k_cellbinstart.sync_device();
  k_cellbinlist.modify_host(); k_cellbinlist.sync_device();
  d_cellbinstart = k_cellbinstart.view_device();
  d_cellbinlist = k_cellbinlist.view_device();
  cellbingen_kk = cellbingen;
}

/* ----------------------------------------------------------------------
   deduplicate a dirty index list with the stamp array, sized to the
     cell allocation; returns the unique indices in a new array
------------------------------------------------------------------------- */

int GridKokkos::stage_records(int n, int *list, int **unique)
{
  *unique = NULL;
  if (!n) return 0;

  if (maxcell > maxdirtystamp) {
    memory->destroy(dirtystamp);
    maxdirtystamp = maxcell;
    memory->create(dirtystamp,maxdirtystamp,"grid:dirtystamp");
    for (int i = 0; i < maxdirtystamp; i++) dirtystamp[i] = 0;
  }
  int gen = ++dirtygen;

  memory->create(*unique,n,"grid:unique");
  int nunique = 0;
  for (int i = 0; i < n; i++) {
    int icell = list[i];
    if (icell < 0 || icell >= maxcell) continue;
    if (dirtystamp[icell] == gen) continue;
    dirtystamp[icell] = gen;
    (*unique)[nunique++] = icell;
  }
  return nunique;
}

/* ----------------------------------------------------------------------
   upload N list records and their buffer
------------------------------------------------------------------------- */

void GridKokkos::upload_list_records(int n, ListRecord *rec, int *buf,
                                     bigint nbuf)
{
  if ((int) k_recicell.extent(0) < n) {
    k_recicell = DAT::tdual_int_1d("grid:recicell",n);
    k_recn = DAT::tdual_int_1d("grid:recn",n);
    k_recoff = DAT::tdual_bigint_1d("grid:recoff",n);
  }
  if ((bigint) k_listbuf.extent(0) < nbuf)
    k_listbuf = DAT::tdual_int_1d("grid:listbuf",nbuf);

  auto h_recicell = k_recicell.view_host();
  auto h_recn = k_recn.view_host();
  auto h_recoff = k_recoff.view_host();
  auto h_listbuf = k_listbuf.view_host();
  for (int m = 0; m < n; m++) {
    h_recicell(m) = rec[m].icell;
    h_recn(m) = rec[m].n;
    h_recoff(m) = rec[m].offset;
  }
  for (bigint i = 0; i < nbuf; i++) h_listbuf(i) = buf[i];
  k_recicell.modify_host(); k_recicell.sync_device();
  k_recn.modify_host(); k_recn.sync_device();
  k_recoff.modify_host(); k_recoff.sync_device();
  k_listbuf.modify_host(); k_listbuf.sync_device();
}

/* ----------------------------------------------------------------------
   d_csurfs for the current cell layout from its old rows and the cut
     records: a cell's row comes from the record for it, else from its
     old row, and a sub cell's row is its split cell's
   the old row of a cell the last restructure moved is at its old index
     (Grid::movedfrom), and every other cell kept its index
------------------------------------------------------------------------- */

void GridKokkos::build_csurfs_device()
{
  int i,k;
  int ntotal = nlocal + nghost;
  int nold = ncsurfsrows;

  // rowsrc = old index of every current cell, rowpar = split cell of
  //   every sub cell (current index), -1 otherwise

  if ((int) d_rowsrc.extent(0) < ntotal) {
    d_rowsrc = DAT::t_int_1d("grid:rowsrc",ntotal);
    d_rowpar = DAT::t_int_1d("grid:rowpar",ntotal);
  }
  auto d_rowsrc = this->d_rowsrc;
  auto d_rowpar = this->d_rowpar;
  Kokkos::parallel_for(ntotal, KOKKOS_LAMBDA(const int m) {
    d_rowsrc(m) = (m < nold) ? m : -1;
    d_rowpar(m) = -1;
  });

  if (nmoved) {
    if ((int) k_movedfrom.extent(0) < nmoved) {
      k_movedfrom = DAT::tdual_int_1d("grid:movedfrom",nmoved);
      k_movedto = DAT::tdual_int_1d("grid:movedto",nmoved);
    }
    auto h_from = k_movedfrom.view_host();
    auto h_to = k_movedto.view_host();
    for (i = 0; i < nmoved; i++) {
      h_from(i) = movedfrom[i];
      h_to(i) = movedto[i];
    }
    k_movedfrom.modify_host(); k_movedfrom.sync_device();
    k_movedto.modify_host(); k_movedto.sync_device();
    auto d_from = k_movedfrom.view_device();
    auto d_to = k_movedto.view_device();
    Kokkos::parallel_for(nmoved, KOKKOS_LAMBDA(const int m) {
      const int from = d_from(m);
      d_rowsrc(d_to(m)) = (from < nold) ? from : -1;
    });
  }

  int nsub = 0;
  int nsinfo = nsplitlocal + nsplitghost;
  for (i = 0; i < nsinfo; i++) {
    int icell = sinfo[i].icell;
    if (icell < 0 || icell >= ntotal) continue;
    if (cells[icell].nsplit <= 1 || cells[icell].isplit != i) continue;
    nsub += cells[icell].nsplit;
  }
  if (nsub) {
    if ((int) k_subcell.extent(0) < nsub) {
      k_subcell = DAT::tdual_int_1d("grid:subcell",nsub);
      k_subparent = DAT::tdual_int_1d("grid:subparent",nsub);
    }
    auto h_subcell = k_subcell.view_host();
    auto h_subparent = k_subparent.view_host();
    k = 0;
    for (i = 0; i < nsinfo; i++) {
      int icell = sinfo[i].icell;
      if (icell < 0 || icell >= ntotal) continue;
      if (cells[icell].nsplit <= 1 || cells[icell].isplit != i) continue;
      int *mycsubs = sinfo[i].csubs;
      for (int j = 0; j < cells[icell].nsplit; j++) {
        h_subcell(k) = mycsubs[j];
        h_subparent(k) = icell;
        k++;
      }
    }
    k_subcell.modify_host(); k_subcell.sync_device();
    k_subparent.modify_host(); k_subparent.sync_device();
    auto d_subcell = k_subcell.view_device();
    auto d_subparent = k_subparent.view_device();
    Kokkos::parallel_for(nsub, KOKKOS_LAMBDA(const int m) {
      d_rowpar(d_subcell(m)) = d_subparent(m);
    });
  }

  // rowrec = record for an old index, -1 if none; the last record wins

  if ((int) d_rowrec.extent(0) < MAX(nold,1))
    d_rowrec = DAT::t_int_1d("grid:rowrec",MAX(nold,1));
  auto d_rowrec = this->d_rowrec;
  Kokkos::parallel_for(nold, KOKKOS_LAMBDA(const int m) { d_rowrec(m) = -1; });

  if (ncutrec) {
    upload_list_records(ncutrec,cutrec,cutbuf,ncutbuf);
    auto d_recicell = k_recicell.view_device();
    int nrec = ncutrec;
    Kokkos::parallel_for(nrec, KOKKOS_LAMBDA(const int m) {
      const int ic = d_recicell(m);
      if (ic < nold) d_rowrec(ic) = m;
    });
    Kokkos::fence();
    Kokkos::parallel_for(nrec, KOKKOS_LAMBDA(const int m) {
      const int ic = d_recicell(m);
      if (ic < nold && d_rowrec(ic) < m) d_rowrec(ic) = m;
    });
  }

  int jbuf = 1 - ibuf;
  build_crs(ntotal,d_csurfs,d_rowmap_buf[jbuf],d_entries_buf[jbuf],d_csurfs);
  ibuf = jbuf;
  ncsurfsrows = ntotal;
}

/* ----------------------------------------------------------------------
   d_csurfs_move from d_csurfs and the collision records, which are
     keyed by current cell indices
------------------------------------------------------------------------- */

void GridKokkos::build_move_graph_device()
{
  int ntotal = nlocal + nghost;

  if ((int) d_rowsrc.extent(0) < ntotal) {
    d_rowsrc = DAT::t_int_1d("grid:rowsrc",ntotal);
    d_rowpar = DAT::t_int_1d("grid:rowpar",ntotal);
  }
  if ((int) d_rowrec.extent(0) < ntotal)
    d_rowrec = DAT::t_int_1d("grid:rowrec",ntotal);
  auto d_rowsrc = this->d_rowsrc;
  auto d_rowpar = this->d_rowpar;
  auto d_rowrec = this->d_rowrec;
  Kokkos::parallel_for(ntotal, KOKKOS_LAMBDA(const int m) {
    d_rowsrc(m) = m;
    d_rowpar(m) = -1;
    d_rowrec(m) = -1;
  });

  upload_list_records(ncollrec,collrec,collbuf,ncollbuf);
  auto d_recicell = k_recicell.view_device();
  int nrec = ncollrec;
  Kokkos::parallel_for(nrec, KOKKOS_LAMBDA(const int m) {
    const int ic = d_recicell(m);
    if (ic < ntotal) d_rowrec(ic) = m;
  });
  Kokkos::fence();
  Kokkos::parallel_for(nrec, KOKKOS_LAMBDA(const int m) {
    const int ic = d_recicell(m);
    if (ic < ntotal && d_rowrec(ic) < m) d_rowrec(ic) = m;
  });

  build_crs(ntotal,d_csurfs,d_rowmap_move,d_entries_move,d_csurfs_move);
}

/* ----------------------------------------------------------------------
   count/scan/fill of a per-cell graph with nrows rows from the rows of
     graph old and the uploaded records, through d_rowsrc, d_rowpar and
     d_rowrec, into the persistent buffers rowmap_buf/entries_buf, grown
     as needed; graph dst is pointed at them
   old and dst may be the same graph, since the buffers are separate
------------------------------------------------------------------------- */

void GridKokkos::build_crs(int nrows,
                           Kokkos::Crs<int, DeviceType, void, crs_size_type> &old,
                           Kokkos::View<crs_size_type*,DeviceType> &rowmap_buf,
                           DAT::t_int_1d &entries_buf,
                           Kokkos::Crs<int, DeviceType, void, crs_size_type> &dst)
{
  auto d_rowsrc = this->d_rowsrc;
  auto d_rowpar = this->d_rowpar;
  auto d_rowrec = this->d_rowrec;
  auto old_rowmap = old.row_map;
  auto old_entries = old.entries;
  auto d_recn = k_recn.view_device();
  auto d_recoff = k_recoff.view_device();
  auto d_listbuf = k_listbuf.view_device();

  if ((int) d_rowcount.extent(0) < nrows+1)
    d_rowcount = Kokkos::View<crs_size_type*,DeviceType>("grid:rowcount",nrows+1);
  auto d_rowcount = this->d_rowcount;

  // count: the source row of cell i is its own old row, or its split
  //   cell's; a record for that source replaces it

  Kokkos::parallel_for(nrows, KOKKOS_LAMBDA(const int i) {
    int par = d_rowpar(i);
    int src = (par >= 0) ? d_rowsrc(par) : d_rowsrc(i);
    crs_size_type n = 0;
    if (src >= 0) {
      const int rec = d_rowrec(src);
      if (rec >= 0) n = d_recn(rec);
      else n = old_rowmap(src+1) - old_rowmap(src);
    }
    d_rowcount(i) = n;
  });

  // scan into the row map

  if ((int) rowmap_buf.extent(0) < nrows+1)
    rowmap_buf = Kokkos::View<crs_size_type*,DeviceType>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,"grid:crs_rowmap"),
      nrows+1);
  auto rowmap = rowmap_buf;

  Kokkos::parallel_scan(nrows, KOKKOS_LAMBDA(const int i, crs_size_type &sum,
                                             const bool final) {
    const crs_size_type n = d_rowcount(i);
    if (final) rowmap(i) = sum;
    sum += n;
    if (final && i == nrows-1) rowmap(nrows) = sum;
  });
  Kokkos::fence();

  crs_size_type nentries = 0;
  if (nrows > 0) Kokkos::deep_copy(nentries,Kokkos::subview(rowmap,nrows));
  else Kokkos::deep_copy(Kokkos::subview(rowmap,0),(crs_size_type) 0);

  if ((bigint) entries_buf.extent(0) < (bigint) nentries)
    entries_buf = DAT::t_int_1d(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,"grid:crs_entries"),
      nentries);
  auto entries = entries_buf;

  // fill

  Kokkos::parallel_for(nrows, KOKKOS_LAMBDA(const int i) {
    int par = d_rowpar(i);
    int src = (par >= 0) ? d_rowsrc(par) : d_rowsrc(i);
    if (src < 0) return;
    const crs_size_type start = rowmap(i);
    const int rec = d_rowrec(src);
    if (rec >= 0) {
      const int n = d_recn(rec);
      const bigint off = d_recoff(rec);
      for (int j = 0; j < n; j++) entries(start+j) = d_listbuf(off+j);
    } else {
      const crs_size_type ostart = old_rowmap(src);
      const int n = old_rowmap(src+1) - ostart;
      for (int j = 0; j < n; j++) entries(start+j) = old_entries(ostart+j);
    }
  });
  Kokkos::fence();

  dst.row_map = rowmap;
  dst.entries = entries;
}

/* ---------------------------------------------------------------------- */

void GridKokkos::wrap_kokkos()
{
  // cells

  if (cells != k_cells.view_host().data()) {
    memoryKK->wrap_kokkos(k_cells,cells,maxcell,"grid:cells");
    k_cells.modify_host();
    k_cells.sync_device();
    memory->sfree(cells);
    cells = k_cells.view_host().data();
  }

  // cinfo

  if (cinfo != k_cinfo.view_host().data()) {
    memoryKK->wrap_kokkos(k_cinfo,cinfo,maxlocal,"grid:cinfo");
    k_cinfo.modify_host();
    k_cinfo.sync_device();
    memory->sfree(cinfo);
    cinfo = k_cinfo.view_host().data();
  }

  // sinfo

  if (sinfo != k_sinfo.view_host().data()) {
    memoryKK->wrap_kokkos(k_sinfo,sinfo,maxsplit,"grid:sinfo");
    k_sinfo.modify_host();
    k_sinfo.sync_device();
    memory->sfree(sinfo);
    sinfo = k_sinfo.view_host().data();
  }

  wrap_kokkos_graphs();

  // pcells

  if (pcells != k_pcells.view_host().data()) {
    memoryKK->wrap_kokkos(k_pcells,pcells,maxparent,"grid:pcells");
    k_pcells.modify_host();
    k_pcells.sync_device();
    memory->sfree(pcells);
    pcells = k_pcells.view_host().data();
  }

  // plevels doesn't need wrap but was modified on host

  k_plevels.modify_host();
  k_plevels.sync_device();
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   re-establish device state after a host fix rebuilt the grid and surfs
   (fix ablate regenerating implicit surfaces); the host is authoritative,
   so push it to the device, rebuild the per-cell surf graphs and cell hash,
   and declare the device particle sort invalid, since the fix may have
   deleted particles and reassigned split cell particles to new sub cells
   must run before anything else reads the grid or the per-cell lists, i.e.
   immediately after the fix rather than after the whole end-of-step batch
------------------------------------------------------------------------- */

void GridKokkos::resync_after_host_change()
{
  modify(Host,ALL_MASK);
  update_hash();

  if (surf->exist) {
    ((SurfKokkos*) surf)->modify(Host,ALL_MASK);
    wrap_kokkos_graphs();
  } else journal_clear();

  ((ParticleKokkos*) particle)->sorted_kk = 0;

  changed = 0;
}

/* ---------------------------------------------------------------------- */

void GridKokkos::sync(ExecutionSpace space, unsigned int mask)
{
  if (sparta->kokkos->prewrap) {
    if (space == Device)
      error->one(FLERR,"Sync Device before wrap");
    else
      return;
  }

  if (space == Device) {
    if (sparta->kokkos->auto_sync)
      modify(Host,mask);
    if (mask & CELL_MASK) k_cells.sync_device();
    if (mask & CINFO_MASK) k_cinfo.sync_device();
    if (mask & PCELL_MASK) k_pcells.sync_device();
    if (mask & SINFO_MASK) k_sinfo.sync_device();
    if (mask & PLEVEL_MASK) k_plevels.sync_device();
    if (mask & CUSTOM_MASK) {
      if (ncustom) {
        if (ncustom_ivec)
          for (int i = 0; i < ncustom_ivec; i++)
            k_eivec.view_host()[i].k_view.sync_device();

        if (ncustom_iarray)
          for (int i = 0; i < ncustom_iarray; i++)
            k_eiarray.view_host()[i].k_view.sync_device();

        if (ncustom_dvec)
          for (int i = 0; i < ncustom_dvec; i++)
            k_edvec.view_host()[i].k_view.sync_device();

        if (ncustom_darray)
          for (int i = 0; i < ncustom_darray; i++)
            k_edarray.view_host()[i].k_view.sync_device();
      }
    }
  } else {
    if (mask & CELL_MASK) k_cells.sync_host();
    if (mask & CINFO_MASK) k_cinfo.sync_host();
    if (mask & PCELL_MASK) k_pcells.sync_host();
    if (mask & SINFO_MASK) k_sinfo.sync_host();
    if (mask & PLEVEL_MASK) k_plevels.sync_host();
    if (mask & CUSTOM_MASK) {
      if (ncustom_ivec)
        for (int i = 0; i < ncustom_ivec; i++)
          k_eivec.view_host()[i].k_view.sync_host();

      if (ncustom_iarray)
        for (int i = 0; i < ncustom_iarray; i++)
          k_eiarray.view_host()[i].k_view.sync_host();

      if (ncustom_dvec)
        for (int i = 0; i < ncustom_dvec; i++)
          k_edvec.view_host()[i].k_view.sync_host();

      if (ncustom_darray)
        for (int i = 0; i < ncustom_darray; i++)
          k_edarray.view_host()[i].k_view.sync_host();
    }
  }
}

/* ---------------------------------------------------------------------- */

void GridKokkos::modify(ExecutionSpace space, unsigned int mask)
{
  if (sparta->kokkos->prewrap) {
    if (space == Device)
      error->one(FLERR,"Modify Device before wrap");
    else
      return;
  }

  if (space == Device) {
    if (mask & CELL_MASK) k_cells.modify_device();
    if (mask & CINFO_MASK) k_cinfo.modify_device();
    if (mask & PCELL_MASK) k_pcells.modify_device();
    if (mask & SINFO_MASK) k_sinfo.modify_device();
    if (mask & PLEVEL_MASK) k_plevels.modify_device();
    if (mask & CUSTOM_MASK) {
      if (ncustom) {
        if (ncustom_ivec)
          for (int i = 0; i < ncustom_ivec; i++)
            k_eivec.view_host()[i].k_view.modify_device();

        if (ncustom_iarray)
          for (int i = 0; i < ncustom_iarray; i++)
            k_eiarray.view_host()[i].k_view.modify_device();

        if (ncustom_dvec)
          for (int i = 0; i < ncustom_dvec; i++)
            k_edvec.view_host()[i].k_view.modify_device();

        if (ncustom_darray)
          for (int i = 0; i < ncustom_darray; i++)
            k_edarray.view_host()[i].k_view.modify_device();
      }
    }
    if (sparta->kokkos->auto_sync)
      sync(Host,mask);
  } else {
    if (mask & CELL_MASK) k_cells.modify_host();
    if (mask & CINFO_MASK) k_cinfo.modify_host();
    if (mask & PCELL_MASK) k_pcells.modify_host();
    if (mask & SINFO_MASK) k_sinfo.modify_host();
    if (mask & PLEVEL_MASK) k_plevels.modify_host();
    if (mask & CUSTOM_MASK) {
      if (ncustom) {
        if (ncustom_ivec)
          for (int i = 0; i < ncustom_ivec; i++)
            k_eivec.view_host()[i].k_view.modify_host();

        if (ncustom_iarray)
          for (int i = 0; i < ncustom_iarray; i++)
            k_eiarray.view_host()[i].k_view.modify_host();

        if (ncustom_dvec)
          for (int i = 0; i < ncustom_dvec; i++)
            k_edvec.view_host()[i].k_view.modify_host();

        if (ncustom_darray)
          for (int i = 0; i < ncustom_darray; i++)
            k_edarray.view_host()[i].k_view.modify_host();
      }
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of Kokkos-managed data
   Grid::memory_usage() is deliberately not called: the cells/cinfo/sinfo
     arrays it measures are the host mirrors of the DualViews below.  its
     other two terms, the csurfs and csplits host pages, have no Kokkos
     counterpart and are carried over here
   the flattened Crs graphs, the per-cell particle lists and the halo index
     are device-only in both backends
------------------------------------------------------------------------- */

bigint GridKokkos::memory_usage()
{
  const bool device_distinct =
    !std::is_same<DeviceType::memory_space,Kokkos::HostSpace>::value;

  bigint bytes = csurfs->size();
  bytes += csplits->size();

  bytes += MemKK::memory_usage(k_cells.view_host());
  bytes += MemKK::memory_usage(k_cinfo.view_host());
  bytes += MemKK::memory_usage(k_sinfo.view_host());
  bytes += MemKK::memory_usage(k_pcells.view_host());
  bytes += MemKK::memory_usage(k_plevels.view_host());
  for (int i = 0; i < ncustom_ivec; i++)
    bytes += MemKK::memory_usage(k_eivec.view_host()[i].k_view.view_host());
  for (int i = 0; i < ncustom_iarray; i++)
    bytes += MemKK::memory_usage(k_eiarray.view_host()[i].k_view.view_host());
  for (int i = 0; i < ncustom_dvec; i++)
    bytes += MemKK::memory_usage(k_edvec.view_host()[i].k_view.view_host());
  for (int i = 0; i < ncustom_darray; i++)
    bytes += MemKK::memory_usage(k_edarray.view_host()[i].k_view.view_host());

  if (device_distinct) {
    bytes += MemKK::memory_usage(k_cells.view_device());
    bytes += MemKK::memory_usage(k_cinfo.view_device());
    bytes += MemKK::memory_usage(k_sinfo.view_device());
    bytes += MemKK::memory_usage(k_pcells.view_device());
    bytes += MemKK::memory_usage(k_plevels.view_device());
    for (int i = 0; i < ncustom_ivec; i++)
      bytes += MemKK::memory_usage(k_eivec.view_host()[i].k_view.view_device());
    for (int i = 0; i < ncustom_iarray; i++)
      bytes += MemKK::memory_usage(k_eiarray.view_host()[i].k_view.view_device());
    for (int i = 0; i < ncustom_dvec; i++)
      bytes += MemKK::memory_usage(k_edvec.view_host()[i].k_view.view_device());
    for (int i = 0; i < ncustom_darray; i++)
      bytes += MemKK::memory_usage(k_edarray.view_host()[i].k_view.view_device());
  }

  bytes += MemKK::memory_usage(d_csurfs.entries);
  bytes += MemKK::memory_usage(d_csurfs.row_map);
  bytes += MemKK::memory_usage(d_csplits.entries);
  bytes += MemKK::memory_usage(d_csplits.row_map);
  bytes += MemKK::memory_usage(d_csubs.entries);
  bytes += MemKK::memory_usage(d_csubs.row_map);

  bytes += MemKK::memory_usage(d_cellcount);
  bytes += MemKK::memory_usage(d_plist);
  bytes += MemKK::memory_usage(d_halo_index);

  return bytes;
}
