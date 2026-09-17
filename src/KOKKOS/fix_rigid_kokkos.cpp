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
#include "fix_rigid_kokkos.h"
#include "update.h"
#include "grid_kokkos.h"
#include "particle_kokkos.h"
#include "surf_kokkos.h"
#include "compute_surf.h"
#include "error.h"
#include "memory_kokkos.h"
#include "kokkos.h"
#include "sparta_masks.h"
#include "geometry_kokkos.h"

using namespace SPARTA_NS;

enum{CELLUNKNOWN,CELLOUTSIDE,CELLINSIDE,CELLOVERLAP};   // same as Grid

/* ----------------------------------------------------------------------
   KOKKOS version of fix rigid
   the body time integration, force/torque reduction, swept collision
     lists, grid re-map and deletion of particles inside the body run on
     the host, exactly as in FixRigid; the moving-surf collision tests
     run in the KOKKOS particle mover (UpdateKokkos::move)
   this class brackets the host work with the device<->host transfers
     it needs, and rebuilds the device per-cell surf graphs whenever the
     host work changed the per-cell surf lists
------------------------------------------------------------------------- */

FixRigidKokkos::FixRigidKokkos(SPARTA *sparta, int narg, char **arg) :
  FixRigid(sparta, narg, arg)
{
  kokkos_flag = 0; // need auto sync
  execution_space = Host;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
  kokkosable = 1;

  nelem_kk = nbin_kk = 0;
  maxdelete_kk = 0;
  d_ndelete_kk = DAT::t_int_scalar("fix_rigid:ndelete");
  h_ndelete_kk = Kokkos::create_mirror_view(d_ndelete_kk);
}

/* ---------------------------------------------------------------------- */

FixRigidKokkos::~FixRigidKokkos()
{
}

/* ---------------------------------------------------------------------- */

void FixRigidKokkos::init()
{
  ((GridKokkos*) grid)->sync(Host,ALL_MASK);
  ((SurfKokkos*) surf)->sync(Host,ALL_MASK);

  FixRigid::init();

  // force/torque tallies must come from the KOKKOS mover, which only
  //   tallies into the KOKKOS variant of compute surf

  if (!csurf->kokkos_flag)
    error->all(FLERR,"Fix rigid/kk requires compute surf/kk");
}

/* ----------------------------------------------------------------------
   bring grid and surfs to the host before host-side work
   the particles are NOT brought over: the only per-particle work in the
     step is remove_inside_all(), which runs as a device kernel, and the
     particle array is by far the largest in the problem (one crossing
     each way per step dominated the cost of the fix on a GPU).  the host
     fallback path inside remove_inside_all() syncs them itself when it
     is taken, and so does any routine below which needs them
------------------------------------------------------------------------- */

void FixRigidKokkos::host_begin()
{
  ((GridKokkos*) grid)->sync(Host,ALL_MASK);
  ((SurfKokkos*) surf)->sync(Host,ALL_MASK);
}

/* ----------------------------------------------------------------------
   after host-side work: the host copies are authoritative
   if the grid was rebuilt (full re-map, flagged via Grid::changed),
     GridKokkos::resync_after_host_change() re-establishes the device
     grid, hash and per-cell surf graphs; otherwise only the per-cell
     surf lists changed (swept lists, incremental re-cut), so rewrap
     the surf graphs
------------------------------------------------------------------------- */

void FixRigidKokkos::host_end()
{
  GridKokkos *grid_kk = (GridKokkos*) grid;
  ParticleKokkos *particle_kk = (ParticleKokkos*) particle;
  SurfKokkos *surf_kk = (SurfKokkos*) surf;

  // surfs: the body geometry was regenerated on the host
  // per-cell surf lists: only rewrap the device graphs if a list changed
  // particles: NOT flagged as host-modified.  the deletion pass ran on the
  //   device and left the device copy authoritative; flagging the host
  //   copy here would mark a stale array as the newer one and discard the
  //   deletions on the next sync.  the host fallback path in
  //   remove_inside_all() does the flagging itself

  grid_kk->modify(Host,ALL_MASK);
  surf_kk->modify(Host,ALL_MASK);
  particle_kk->sorted_kk = 0;

  if (grid->changed) grid_kk->resync_after_host_change();
  else if (listschanged) grid_kk->wrap_kokkos_graphs();
  listschanged = 0;
}

/* ---------------------------------------------------------------------- */

void FixRigidKokkos::setup()
{
  host_begin();
  FixRigid::setup();
  host_end();
}

/* ----------------------------------------------------------------------
   integrate the bodies and install the swept collision lists (host),
     then rebuild the device surf graphs the mover reads
   distributed surfs: local copies of bodies entering this proc may be
     appended to the surf arrays; cells are not otherwise changed here
------------------------------------------------------------------------- */

void FixRigidKokkos::start_of_step()
{
  GridKokkos *grid_kk = (GridKokkos*) grid;
  SurfKokkos *surf_kk = (SurfKokkos*) surf;

  grid_kk->sync(Host,ALL_MASK);
  surf_kk->sync(Host,ALL_MASK);

  FixRigid::start_of_step();

  // distributed surfs: local copies of a body entering this proc were
  //   appended to the host surf arrays

  if (copiesappended) {
    surf_kk->modify(Host,ALL_MASK);
    copiesappended = 0;
  }

  grid_kk->modify(Host,CELL_MASK);
  if (listschanged) grid_kk->wrap_kokkos_graphs();
  listschanged = 0;
}

/* ----------------------------------------------------------------------
   restore swept lists, reduce force/torque, move the bodies, re-map the
     grid and delete particles inside the bodies (host), then
     re-establish the device state
------------------------------------------------------------------------- */

void FixRigidKokkos::end_of_step()
{
  host_begin();
  FixRigid::end_of_step();
  host_end();
}

/* ----------------------------------------------------------------------
   host code below is about to read or write the particle array, which
     host_begin() deliberately left on the device: bring it over and let
     the host copy be the newer one
   also used by FixRigid::setup(), whose one-off passes stay on the host
------------------------------------------------------------------------- */

void FixRigidKokkos::particles_to_host()
{
  ParticleKokkos *particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Host,PARTICLE_MASK|CUSTOM_MASK);
  particle_kk->modify(Host,PARTICLE_MASK|CUSTOM_MASK);
  particle_kk->sorted_kk = 0;
}

/* ----------------------------------------------------------------------
   remove particles inside a body
   run the pass on the device when it can be, so the particle array need
     not be brought to the host; otherwise defer to FixRigid
   the host copy of the particles is stale on return from the device path,
     which is what host_end() would otherwise re-upload
------------------------------------------------------------------------- */

void FixRigidKokkos::remove_inside_all(int splitflag)
{
  if (remove_inside_all_kokkos(splitflag)) {
    end_of_run_delete_warning();
    return;
  }

  // host fallback: the particles must be on the host for FixRigid's pass

  ((ParticleKokkos*) particle)->sync(Host,PARTICLE_MASK|CUSTOM_MASK);
  FixRigid::remove_inside_all(splitflag);
  ((ParticleKokkos*) particle)->modify(Host,PARTICLE_MASK|CUSTOM_MASK);
}

/* ----------------------------------------------------------------------
   copy the replicated body geometry and the COM bins to the device
   the geometry is regenerated on the host every step by
     FixRigid::update_surf_copies(), and the bins by body_bins(), so this
     runs once per step, before the deletion kernel
   only the element count can change during a run (distributed surfs
     append local copies), so the views are grown, not reallocated
------------------------------------------------------------------------- */

void FixRigidKokkos::pack_body_device()
{
  int nelem = bodystart[nbody];
  int nbins = bodynbin[0]*bodynbin[1]*bodynbin[2];

  if (nelem > nelem_kk) {
    k_bodypt = tdual_dbl_3d("fix_rigid:bodypt",nelem,3,3);
    k_bodynorm = tdual_dbl_2d("fix_rigid:bodynorm",nelem,3);
    d_bodypt = k_bodypt.view_device();
    d_bodynorm = k_bodynorm.view_device();
    nelem_kk = nelem;
  }
  if (nbins > nbin_kk || k_bodybinstart.extent(0) < (size_t)(nbins+1)) {
    k_bodybinstart = DAT::tdual_int_1d("fix_rigid:bodybinstart",nbins+1);
    d_bodybinstart = k_bodybinstart.view_device();
    nbin_kk = nbins;
  }
  if (k_bodybinlist.extent(0) < (size_t)nbody) {
    k_bodybinlist = DAT::tdual_int_1d("fix_rigid:bodybinlist",nbody);
    d_bodybinlist = k_bodybinlist.view_device();
  }
  if (k_bodystart.extent(0) < (size_t)(nbody+1)) {
    k_bodystart = DAT::tdual_int_1d("fix_rigid:bodystart",nbody+1);
    k_bbodylo = tdual_dbl_2d("fix_rigid:bbodylo",nbody,3);
    k_bbodyhi = tdual_dbl_2d("fix_rigid:bbodyhi",nbody,3);
    d_bodystart = k_bodystart.view_device();
    d_bbodylo = k_bbodylo.view_device();
    d_bbodyhi = k_bbodyhi.view_device();
  }

  // bodypt is allocated [nsurf][dim][3]: a 2d element is a line and has
  //   only 2 corner points, so copy dim of them, not 3

  auto h_bodypt = k_bodypt.view_host();
  auto h_bodynorm = k_bodynorm.view_host();
  for (int i = 0; i < nelem; i++) {
    for (int j = 0; j < dim; j++)
      for (int k = 0; k < 3; k++) h_bodypt(i,j,k) = bodypt[i][j][k];
    for (int k = 0; k < 3; k++) h_bodynorm(i,k) = bodynorm[i][k];
  }
  k_bodypt.modify_host(); k_bodypt.sync_device();
  k_bodynorm.modify_host(); k_bodynorm.sync_device();

  auto h_bodystart = k_bodystart.view_host();
  auto h_bbodylo = k_bbodylo.view_host();
  auto h_bbodyhi = k_bbodyhi.view_host();
  for (int ib = 0; ib <= nbody; ib++) h_bodystart(ib) = bodystart[ib];
  for (int ib = 0; ib < nbody; ib++)
    for (int k = 0; k < 3; k++) {
      h_bbodylo(ib,k) = bbodylo[ib][k];
      h_bbodyhi(ib,k) = bbodyhi[ib][k];
    }
  k_bodystart.modify_host(); k_bodystart.sync_device();
  k_bbodylo.modify_host(); k_bbodylo.sync_device();
  k_bbodyhi.modify_host(); k_bbodyhi.sync_device();

  auto h_binstart = k_bodybinstart.view_host();
  auto h_binlist = k_bodybinlist.view_host();
  for (int i = 0; i <= nbins; i++) h_binstart(i) = bodybinstart[i];
  for (int i = 0; i < nbody; i++) h_binlist(i) = bodybinlist[i];
  k_bodybinstart.modify_host(); k_bodybinstart.sync_device();
  k_bodybinlist.modify_host(); k_bodybinlist.sync_device();

  for (int k = 0; k < 3; k++) {
    bodynbin_kk[k] = bodynbin[k];
    bodybinlo_kk[k] = bodybinlo[k];
    bodybininv_kk[k] = bodybininv[k];
  }
  rmaxall_kk = rmaxall;
  dim_kk = dim;
}

/* ----------------------------------------------------------------------
   one thread per particle: flag the particle for deletion if it is in an
     INSIDE cell or interior to any body, the same two tests and in the
     same order as FixRigid::remove_inside_all()
   reduction value = # of particles claimed by a body, which is what
     FixRigid counts in ndeleted/ndelrun; a particle deleted only for
     being in an INSIDE cell is not counted, as on the host
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixRigidKokkos::operator()(TagFixRigidRemoveInside,
                                const int &i, int &nbody_del) const
{
  const int icell = d_particles_kk[i].icell;
  if (icell < 0) return;

  const int ctype = d_celltype_kk[icell];
  const int inside = (ctype == CELLINSIDE);

  // a surf-free OUTSIDE cell cannot contain a point interior to a body,
  //   since a body boundary crossing the cell would put a surf in it

  if (!inside && ctype == CELLOUTSIDE && d_cellnsurf_kk[icell] == 0) return;

  double *x = d_particles_kk[i].x;

  // inside_any_body(): the bins overlapping x inflated by rmaxall give the
  //   candidate bodies, then each body's bbox is tested exactly

  int inbody = 0;
  int blo[3],bhi[3];
  for (int k = 0; k < 3; k++) {
    blo[k] = (int) ((x[k]-rmaxall_kk-bodybinlo_kk[k]) * bodybininv_kk[k]);
    bhi[k] = (int) ((x[k]+rmaxall_kk-bodybinlo_kk[k]) * bodybininv_kk[k]);
    blo[k] = MAX(0,MIN(blo[k],bodynbin_kk[k]-1));
    bhi[k] = MAX(0,MIN(bhi[k],bodynbin_kk[k]-1));
  }

  for (int ibz = blo[2]; ibz <= bhi[2] && !inbody; ibz++)
    for (int iby = blo[1]; iby <= bhi[1] && !inbody; iby++)
      for (int ibx = blo[0]; ibx <= bhi[0] && !inbody; ibx++) {
        const int ibin = (ibz*bodynbin_kk[1] + iby)*bodynbin_kk[0] + ibx;
        for (int m = d_bodybinstart[ibin]; m < d_bodybinstart[ibin+1]; m++) {
          const int ibody = d_bodybinlist[m];

          // body_box(): exact bbox test, x as a degenerate box

          int overlap = 1;
          for (int k = 0; k < 3; k++)
            if (x[k] < d_bbodylo(ibody,k) || x[k] > d_bbodyhi(ibody,k))
              { overlap = 0; break; }
          if (!overlap) continue;

          // inside_body(): parity of the crossings of a ray from x to a
          //   point outside the body's bbox.  the ray and the loop order
          //   are those of the host routine, so the count matches

          double blox = d_bbodylo(ibody,0), bhix = d_bbodyhi(ibody,0);
          double dmax = MAX(bhix-blox,d_bbodyhi(ibody,1)-d_bbodylo(ibody,1));
          dmax = MAX(dmax,d_bbodyhi(ibody,2)-d_bbodylo(ibody,2));

          double xout[3],xc[3];
          xout[0] = bhix + 0.414159*dmax;
          xout[1] = x[1] + 0.271828*dmax;
          if (dim_kk == 3) xout[2] = x[2] + 0.161803*dmax;
          else xout[2] = 0.0;

          int count = 0;
          for (int e = d_bodystart(ibody); e < d_bodystart(ibody+1); e++) {
            // only a 3d element has a third corner point; in 2d the
            //   third slot of d_bodypt is never written

            double p1[3],p2[3],p3[3],nrm[3];
            for (int k = 0; k < 3; k++) {
              p1[k] = d_bodypt(e,0,k);
              p2[k] = d_bodypt(e,1,k);
              p3[k] = (dim_kk == 3) ? d_bodypt(e,2,k) : 0.0;
              nrm[k] = d_bodynorm(e,k);
            }
            double param;
            int side;
            bool hitflag;
            if (dim_kk == 2)
              hitflag = GeometryKokkos::
                line_line_intersect(x,xout,p1,p2,nrm,xc,param,side);
            else
              hitflag = GeometryKokkos::
                line_tri_intersect(x,xout,p1,p2,p3,nrm,xc,param,side);
            if (hitflag) count++;
          }

          if (count % 2) { inbody = 1; break; }
        }
      }

  if (!inbody && !inside) return;

  const int n = Kokkos::atomic_fetch_add(&d_ndelete_kk(),1);
  if (n < (int) d_dellist_kk.extent(0)) d_dellist_kk(n) = i;
  if (inbody) nbody_del++;
}

/* ----------------------------------------------------------------------
   device port of FixRigid::remove_inside_all()
   the host routine is a pass over every particle, which is what forces
     the particle array to the host and back on every step; running it on
     the device removes two crossings of the largest array in the problem
   return 1 if the deletion was done here, 0 to let the host do it
   the split-cell reassignment path (splitflag) stays on the host: it
     needs sorted particle lists and per-cell sub cell structure, and it
     only runs on a step where the grid was rebuilt anyway
------------------------------------------------------------------------- */

int FixRigidKokkos::remove_inside_all_kokkos(int splitflag)
{
  // the split-cell reassignment relabels only the particles listed under a
  //   split cell, a small subset, but it needs them on the host and sorted.
  //   do just that part on the host and then run the inside test, which is
  //   the pass over ALL particles, on the device
  // a body which repeatedly creates and destroys split cells sets splitflag
  //   on every step (splitchanged/structural), so declining the device path
  //   here would give up on it entirely for exactly the problems that need
  //   it most

  if (splitflag && grid->nsplitlocal) {
    ParticleKokkos *particle_kk = (ParticleKokkos*) particle;
    particle_kk->sync(Host,PARTICLE_MASK|CUSTOM_MASK);

    if (!particle->sorted) particle->sort();
    Grid::ChildCell *cells = grid->cells;
    int nglocal = grid->nlocal;
    for (int icell = 0; icell < nglocal; icell++)
      if (cells[icell].nsplit > 1)
        grid->assign_split_cell_particles(icell);
    particle->sorted = 0;

    // the host relabelled icell, so the host copy is the newer one; the
    //   device pass below syncs it back before the kernel

    particle_kk->modify(Host,PARTICLE_MASK|CUSTOM_MASK);
    particle_kk->sorted_kk = 0;
  }
  if (!particle->exist) return 1;

  ParticleKokkos *particle_kk = (ParticleKokkos*) particle;
  GridKokkos *grid_kk = (GridKokkos*) grid;

  pack_body_device();

  // the two per-cell fields the test reads, over owned + ghost cells,
  //   since a particle's icell may be a ghost after migration

  int nglocal = grid->nlocal + grid->nghost;
  if (k_celltype_kk.extent(0) < (size_t)nglocal) {
    k_celltype_kk = DAT::tdual_int_1d("fix_rigid:celltype",nglocal);
    k_cellnsurf_kk = DAT::tdual_int_1d("fix_rigid:cellnsurf",nglocal);
    d_celltype_kk = k_celltype_kk.view_device();
    d_cellnsurf_kk = k_cellnsurf_kk.view_device();
  }
  auto h_celltype = k_celltype_kk.view_host();
  auto h_cellnsurf = k_cellnsurf_kk.view_host();
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  for (int ic = 0; ic < nglocal; ic++) {
    h_celltype(ic) = cinfo[ic].type;
    h_cellnsurf(ic) = cells[ic].nsurf;
  }
  k_celltype_kk.modify_host(); k_celltype_kk.sync_device();
  k_cellnsurf_kk.modify_host(); k_cellnsurf_kk.sync_device();

  nplocal_kk = particle->nlocal;

  // the dellist is sized to the particle count so the kernel can never
  //   overflow it, which keeps this a single pass with no retry

  if (nplocal_kk > maxdelete_kk) {
    maxdelete_kk = nplocal_kk;
    k_dellist_kk = DAT::tdual_int_1d("fix_rigid:dellist",maxdelete_kk);
    d_dellist_kk = k_dellist_kk.view_device();
  }

  // ModifyKokkos sets auto_sync while a non-kokkos_flag fix runs, which
  //   makes every modify(Device) copy the array straight back to the host.
  //   that is the behaviour this port exists to avoid, so switch it off
  //   for the device pass and restore it afterwards

  const int prev_auto_sync = sparta->kokkos->auto_sync;
  sparta->kokkos->auto_sync = 0;

  particle_kk->sync(Device,PARTICLE_MASK);
  d_particles_kk = particle_kk->k_particles.view_device();

  Kokkos::deep_copy(d_ndelete_kk,0);

  int nbody_del = 0;
  copymode = 1;
  Kokkos::parallel_reduce(
    Kokkos::RangePolicy<DeviceType,TagFixRigidRemoveInside>(0,nplocal_kk),
    *this,nbody_del);
  copymode = 0;

  Kokkos::deep_copy(h_ndelete_kk,d_ndelete_kk);
  const int ndelete = h_ndelete_kk();

  ndeleted += nbody_del;
  ndelrun += nbody_del;

  if (!ndelete) {
    particle_kk->modify(Device,PARTICLE_MASK);
    sparta->kokkos->auto_sync = prev_auto_sync;
    return 1;
  }

  // the kernel appended to the dellist under an atomic, so its order is
  //   not reproducible; ParticleKokkos::compress_migrate() walks it
  //   assuming ascending indices, and an unsorted list would both break
  //   that walk and make the surviving particle order depend on thread
  //   scheduling.  sort it, so a run is reproducible and matches the host

  k_dellist_kk.modify_device();
  k_dellist_kk.sync_host();
  int *dellist_h = k_dellist_kk.view_host().data();
  std::sort(dellist_h,dellist_h+ndelete);
  k_dellist_kk.modify_host();
  k_dellist_kk.sync_device();

  particle_kk->modify(Device,PARTICLE_MASK);
  particle_kk->compress_migrate(ndelete,dellist_h);
  particle->sorted = 0;
  particle_kk->sorted_kk = 0;

  sparta->kokkos->auto_sync = prev_auto_sync;

  return 1;
}

/* ----------------------------------------------------------------------
   grid cells were rebuilt, adapted, or migrated: FixRigid re-establishes
     its local body-surf copies on the host, which may append surfs and
     re-index ghost cell lists, so flag the host surfs and cells as
     modified; the caller (grid rebuild, fix balance/kk, fix adapt/kk)
     rewraps the device surf graphs afterward
------------------------------------------------------------------------- */

void FixRigidKokkos::grid_changed()
{
  FixRigid::grid_changed();
  if (surf->distributed) ((SurfKokkos*) surf)->modify(Host,ALL_MASK);
  ((GridKokkos*) grid)->modify(Host,CELL_MASK);
}
