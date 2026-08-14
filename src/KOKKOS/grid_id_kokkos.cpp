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
#include "error.h"
#include "grid_kokkos.h"
#include "domain.h"
#include "memory.h"

using namespace SPARTA_NS;

void GridKokkos::update_hash()
{
  typedef hash_type::size_type size_type;    // uint32_t
  typedef hash_type::key_type key_type;      // cellint
  typedef hash_type::value_type value_type;  // int
  typedef hash_type::host_mirror_type host_hash_type;

  size_type failed_count = 0;

  // Copy the keys:values from hash to Kokkos::UnorderedMap that lives on host
  host_hash_type hash_h(2*hash->size()); // double hash capacity to prevent insertion failure
  hash_kk = hash_type(2*hash->size());
  for (volatile auto it : *hash) { // volatile keyword works around a suspected compiler bug
    key_type key = static_cast<key_type>(it.first);
    value_type val = static_cast<value_type>(it.second);
    auto insert_result = hash_h.insert(key, val);
    failed_count += insert_result.failed() ? 1 : 0;
  }
  if (failed_count) {
    error->one(FLERR, "Kokkos::UnorderedMap insertion failed");
  }

  Kokkos::deep_copy(hash_kk, hash_h);

  update_halo_index();
}

/* ----------------------------------------------------------------------
   build d_halo_index, the dense cell-position -> local-index map used by the
     uniform-grid fast path in UpdateKokkos::move()
   leaves it empty (extent 0) whenever it cannot be built cheaply, in which
     case callers use hash_kk instead
------------------------------------------------------------------------- */

void GridKokkos::update_halo_index()
{
  d_halo_index = DAT::t_int_1d();

  if (!uniform || unx <= 0 || uny <= 0 || unz <= 0) return;

  const int ntotal = nlocal + nghost;
  if (ntotal <= 0) return;

  // place cells on the lattice the same way the fast path places particles,
  //   from coordinates rather than from the cell ID, so the two cannot
  //   disagree about which site a cell occupies.  spacing is computed with the
  //   identical expression UpdateKokkos::setup() uses for dx/dy/dz
  // a cell's lo corner is a whole multiple of the spacing, so round to nearest

  const int un[3] = {unx,uny,unz};
  double inv[3];
  for (int d = 0; d < 3; d++)
    inv[d] = un[d]/(domain->boxhi[d]-domain->boxlo[d]);

  // halo arc per dimension: mark the lattice planes this proc holds, then take
  //   the complement of the widest empty run on the ring.  the complement of
  //   any single empty run contains every occupied plane, so this is correct
  //   for a non-arc decomposition too; taking the widest one makes it minimal
  // deriving the arc from nearest-image offsets to a reference cell instead
  //   would have to break a tie at exactly half the dimension, and that is
  //   precisely where the wrapped ghost layer sits when a dimension is split
  //   over two procs.  getting that tie wrong stretches the arc to the whole
  //   dimension, which either wastes memory or trips the bounding-box test
  //   below and silently drops this proc back to the hash

  int *occ[3];
  for (int d = 0; d < 3; d++) {
    memory->create(occ[d],un[d],"grid:halo_occupied");
    for (int i = 0; i < un[d]; i++) occ[d][i] = 0;
  }

  // one pass over the cells, marking all three dimensions, since ntotal is the
  //   whole owned plus ghost list and is much larger than unx+uny+unz

  for (int m = 0; m < ntotal; m++)
    for (int d = 0; d < 3; d++) {
      const int s =
        static_cast<int> ((cells[m].lo[d]-domain->boxlo[d])*inv[d]+0.5);
      if (s >= 0 && s < un[d]) occ[d][s] = 1;
    }

  int lo[3],n[3],empty = 0;
  double box = 1.0;

  for (int d = 0; d < 3; d++) {

    // scan the ring starting from an occupied plane so no empty run is split
    //   by the wrap.  gapend is the first occupied plane after the widest run,
    //   which is where the arc begins

    int first = -1;
    for (int i = 0; i < un[d]; i++)
      if (occ[d][i]) { first = i; break; }
    if (first < 0) { empty = 1; break; }

    int gapbest = 0,gapend = first,run = 0;
    for (int k = 1; k <= un[d]; k++) {
      const int i = (first+k) % un[d];
      if (!occ[d][i]) run++;
      else {
        if (run > gapbest) { gapbest = run; gapend = i; }
        run = 0;
      }
    }

    n[d] = un[d] - gapbest;
    lo[d] = gapend;
    box *= n[d];
  }

  for (int d = 0; d < 3; d++) memory->destroy(occ[d]);
  if (empty) return;

  // refuse when the bounding box holds far more sites than cells: the halo is
  //   not an arc (RCB, adaptive) and a dense map would mostly be holes

  if (box > 4.0*ntotal) return;

  halo_ilo = lo[0]; halo_jlo = lo[1]; halo_klo = lo[2];
  halo_nx = n[0];   halo_ny = n[1];   halo_nz = n[2];

  HAT::t_int_1d h_halo_index(
    Kokkos::view_alloc("grid:halo_index",Kokkos::WithoutInitializing),
    (size_t) halo_nx*halo_ny*halo_nz);
  for (size_t i = 0; i < h_halo_index.extent(0); i++) h_halo_index[i] = -1;

  for (int m = 0; m < ntotal; m++) {
    int l[3],ok = 1;
    for (int d = 0; d < 3; d++) {
      l[d] =
        static_cast<int> ((cells[m].lo[d]-domain->boxlo[d])*inv[d]+0.5)-lo[d];
      if (l[d] < 0) l[d] += un[d];
      if (l[d] >= n[d]) ok = 0;
    }
    if (ok) h_halo_index[((size_t) l[2]*halo_ny + l[1])*halo_nx + l[0]] = m;
  }

  d_halo_index = DAT::t_int_1d(
    Kokkos::view_alloc("grid:halo_index",Kokkos::WithoutInitializing),
    h_halo_index.extent(0));
  Kokkos::deep_copy(d_halo_index,h_halo_index);
}
