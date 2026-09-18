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

#ifndef SPARTA_GRID_KOKKOS_H
#define SPARTA_GRID_KOKKOS_H

#include "grid.h"
#include "kokkos_type.h"
#include <Kokkos_UnorderedMap.hpp>

namespace SPARTA_NS {

class GridKokkos : public Grid {
 public:
  typedef ArrayTypes<DeviceType> AT;

  typedef Kokkos::UnorderedMap<cellint,int> hash_type;
  typedef hash_type::size_type size_type;    // uint32_t
  typedef hash_type::key_type key_type;      // cellint
  typedef hash_type::value_type value_type;  // int

  // make into a view
  //ChildCell *cells;           // list of owned and ghost child cells

  // methods

  GridKokkos(class SPARTA *);
  ~GridKokkos();
  void wrap_kokkos();
  void wrap_kokkos_graphs();
  void sync(ExecutionSpace, unsigned int);
  void modify(ExecutionSpace, unsigned int);

  int add_custom(char *, int, int) override;
  void allocate_custom(int) override;
  void reallocate_custom(int, int) override;
  void remove_custom(int) override;
  void copy_custom(int,int) override;
  int pack_custom(int, char *, int) override;
  int unpack_custom(char *, int) override;

// operations with grid cell IDs
  void update_hash();

  // re-establish device state after a host fix rebuilt the grid/surfs
  void resync_after_host_change();

  // patch the device with the change journal (see Grid::journalflag)
  void apply_changes();

  // device copy of the cell bin index (Grid::cells_in_box), re-copied
  //   when the host rebuilds it and patched with it otherwise

  DAT::tdual_int_1d k_cellbinstart,k_cellbinlist;
  DAT::t_int_1d d_cellbinstart,d_cellbinlist;
  int cellbingen_kk;
  void sync_cell_bins();

  /* ----------------------------------------------------------------------
     compute lo/hi extent of a specific child cell within a parent cell
     plevel = level of parent
     plo/phi = parent cell corner points
     ichild ranges from 1 to Nx*Ny*Nz within parent cell
     return clo/chi corner points, caller must allocate them
  ------------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void id_child_lohi(int plevel, double *plo, double *phi,
                     cellint ichild, double *clo, double *chi) const
  {
    int nx = k_plevels.view_device()[plevel].nx;
    int ny = k_plevels.view_device()[plevel].ny;
    int nz = k_plevels.view_device()[plevel].nz;

    ichild--;
    int ix = ichild % nx;
    int iy = (ichild/nx) % ny;
    int iz = ichild / ((cellint) nx*ny);

    clo[0] = plo[0] + ix*(phi[0]-plo[0])/nx;
    clo[1] = plo[1] + iy*(phi[1]-plo[1])/ny;
    clo[2] = plo[2] + iz*(phi[2]-plo[2])/nz;

    chi[0] = plo[0] + (ix+1)*(phi[0]-plo[0])/nx;
    chi[1] = plo[1] + (iy+1)*(phi[1]-plo[1])/ny;
    chi[2] = plo[2] + (iz+1)*(phi[2]-plo[2])/nz;

    if (ix == nx-1) chi[0] = phi[0];
    if (iy == ny-1) chi[1] = phi[1];
    if (iz == nz-1) chi[2] = phi[2];
  }

  /* ----------------------------------------------------------------------
     find child cell within parentID which contains pt X
     level = level of parent cell
     oplo/ophi = original parent cell corner pts
     pt X can be inside or on any boundary of parent cell
     recurse from parent downward until find a child cell or reach maxlevel
     if find child cell this proc stores (owned or ghost), return its local index
     else return -1 for unknown
  ------------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int id_find_child(cellint parentID, int plevel,
                    double *oplo, double *ophi, double *x) const
  {
    int ix,iy,iz,nx,ny,nz;
    double plo[3],phi[3],clo[3],chi[3];
    cellint childID,ichild;

    cellint id = parentID;
    int level = plevel;
    double *lo = oplo;
    double *hi = ophi;

    while (level < maxlevel) {
      nx = k_plevels.view_device()[level].nx;
      ny = k_plevels.view_device()[level].ny;
      nz = k_plevels.view_device()[level].nz;
      ix = static_cast<int> ((x[0]-lo[0]) * nx/(hi[0]-lo[0]));
      iy = static_cast<int> ((x[1]-lo[1]) * ny/(hi[1]-lo[1]));
      iz = static_cast<int> ((x[2]-lo[2]) * nz/(hi[2]-lo[2]));
      if (ix == nx) ix--;
      if (iy == ny) iy--;
      if (iz == nz) iz--;

      ichild = (cellint) iz*nx*ny + (cellint) iy*nx + ix + 1;
      childID = (ichild << k_plevels.view_device()[level].nbits) | id;

      size_type h_index = hash_kk.find(static_cast<key_type>(childID));
      if (hash_kk.valid_at(h_index)) return static_cast<int>(hash_kk.value_at(h_index));

      id = childID;
      id_child_lohi(level,lo,hi,ichild,clo,chi);
      plo[0] = clo[0]; plo[1] = clo[1]; plo[2] = clo[2];
      phi[0] = chi[0]; phi[1] = chi[1]; phi[2] = chi[2];
      lo = plo; hi = phi;
      level++;
    }

    return -1;
  }

  // extract/return neighbor flag for iface from per-cell nmask
  // inlined for efficiency

  KOKKOS_INLINE_FUNCTION
  int neigh_decode(int nmask, int iface) const {
    return (nmask & neighmask[iface]) >> neighshift[iface];
  }

  // overwrite neighbor flag for iface in per-cell nmask
  // first line zeroes the iface bits via one's complement of mask
  // inlined for efficiency
  // return updated nmask

  KOKKOS_INLINE_FUNCTION
  int neigh_encode(int flag, int nmask, int iface) const {
    nmask &= ~neighmask[iface];
    nmask |= flag << neighshift[iface];
    return nmask;
  }

  tdual_cell_1d k_cells;
  tdual_cinfo_1d k_cinfo;
  tdual_sinfo_1d k_sinfo;
  tdual_pcell_1d k_pcells;
  tdual_plevel_1d k_plevels;

  // Crs row_map offsets index the flattened per-cell surf lists.  Under
  //   BIGBIG the total entry count on a rank can exceed 2^31, so the offset
  //   type must be 64-bit; under BIG it cannot, and a 32-bit row_map halves
  //   the bytes touched by the per-particle surf lookups in the move kernel.
  // d_csurfs = the cut list of every owned+ghost cell, which split2d/3d
  //   index in lockstep with d_csplits
  // d_csurfs_move = the list the mover tests particles against: the cut
  //   list, or the collision list a fix which moves surfs set for this
  //   step (Grid::set_collision_surfs); the same graph as d_csurfs when
  //   no cell has one
  // graph_generation counts rebuilds of any of them, so a consumer
  //   holding a copy of a graph can tell it is stale

  Kokkos::Crs<int, DeviceType, void, crs_size_type> d_csurfs;
  Kokkos::Crs<int, DeviceType, void, crs_size_type> d_csurfs_move;
  Kokkos::Crs<int, DeviceType, void, crs_size_type> d_csplits;
  Kokkos::Crs<int, DeviceType, void, crs_size_type> d_csubs;
  int ncsurfsrows;            // rows of d_csurfs, whose views may be longer
  int graph_generation;

  DAT::t_int_1d d_cellcount;
  DAT::t_int_2d d_plist;

  // hash for all cell IDs (owned,ghost,parent).  The _d postfix refers to the
  // fact that this hash lives on "device"
  hash_type hash_kk;

  // device copy of Grid::halo_index, the dense alternative to hash_kk for the
  //   uniform-grid fast path in UpdateKokkos::move(): maps a cell's position
  //   within this proc's halo straight to its local index, so a lookup is one
  //   indexed load rather than a dependent probe chain.
  // the halo_* extents that key it are the base class members, since the map
  //   is built there and this is only the upload.
  // extent 0 means unavailable, and callers must fall back to hash_kk.

  DAT::t_int_1d d_halo_index;

  void update_halo_index();
  bigint memory_usage() override;

  DAT::tdual_int_1d k_ewhich,k_eicol,k_edcol;

  tdual_struct_tdual_int_1d_1d k_eivec;
  tdual_struct_tdual_float_1d_1d k_edvec;
  tdual_struct_tdual_int_2d_1d k_eiarray;
  tdual_struct_tdual_float_2d_1d k_edarray;

  void wrap_split_graphs();   // d_csplits/d_csubs from the host sinfo
  void build_csurfs_device(); // d_csurfs from its old rows + cut records
  void build_move_graph_device(); // d_csurfs_move from d_csurfs + records

 private:
  void grow_cells(int, int) override;
  void grow_sinfo(int) override;
  void grow_pcells() override;

  // staging for apply_changes(): the journal's records, uploaded and
  //   scattered by kernels; the stamps deduplicate the dirty lists

  int *dirtystamp,*sinfostamp;
  int maxdirtystamp,maxsinfostamp,dirtygen;

  tdual_cell_1d k_stagecell;
  tdual_cinfo_1d k_stagecinfo;
  tdual_sinfo_1d k_stagesinfo;
  DAT::tdual_int_1d k_dirtycell,k_dirtyown,k_dirtysinfo;
  DAT::tdual_cellint_1d k_hashid;
  DAT::tdual_int_1d k_hashidx,k_halosite,k_haloidx;
  DAT::tdual_int_1d k_movedfrom,k_movedto,k_subcell,k_subparent;
  DAT::tdual_int_1d k_recicell,k_recn,k_listbuf;
  DAT::tdual_bigint_1d k_recoff;
  DAT::t_int_1d d_rowsrc,d_rowpar,d_rowrec;
  Kokkos::View<crs_size_type*,DeviceType> d_rowcount;
  Kokkos::View<crs_size_type*,DeviceType> d_rowmap_buf[2],d_rowmap_move;
  DAT::t_int_1d d_entries_buf[2],d_entries_move;
  int ibuf;                   // which buffer pair d_csurfs currently uses

  int stage_records(int, int *, int **);   // dedup a dirty list
  void upload_list_records(int, ListRecord *, int *, bigint);
  void build_crs(int, Kokkos::Crs<int, DeviceType, void, crs_size_type> &,
                 Kokkos::View<crs_size_type*,DeviceType> &, DAT::t_int_1d &,
                 Kokkos::Crs<int, DeviceType, void, crs_size_type> &);
};

}

#endif

/* ERROR/WARNING messages:

*/
