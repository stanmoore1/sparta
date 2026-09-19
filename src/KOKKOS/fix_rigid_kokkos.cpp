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
#include "rigid_remap.h"
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
#include "output.h"
#include "domain.h"

using namespace SPARTA_NS;

#define EPSSURF 1.0e-4          // same as FixRigid
#define BIG 1.0e20

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
  nsub_kk = 0;
  nasg_kk = 0;
  nsplit_kk = 0;

  // the re-map helper with its device stages

  delete remap;
  remap = new RigidRemapKokkos(sparta,this);
  hostgeom = NULL;
  maxhostgeom = 0;
  ncopy_kk = 0;
  devicegeom = 0;
  maxdelete_kk = 0;
  d_ndelete_kk = DAT::t_int_scalar("fix_rigid:ndelete");
  h_ndelete_kk = Kokkos::create_mirror_view(d_ndelete_kk);
}

/* ---------------------------------------------------------------------- */

FixRigidKokkos::~FixRigidKokkos()
{
  if (copy || copymode) return;
  memory->destroy(hostgeom);
}

/* ---------------------------------------------------------------------- */

void FixRigidKokkos::init()
{
  ((GridKokkos*) grid)->sync(Host,ALL_MASK);
  ((SurfKokkos*) surf)->sync(Host,ALL_MASK);

  FixRigid::init();
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

  // surfs: the body geometry was regenerated on the device by set_xv()
  //   and the host copies are refreshed only for a host consumer
  // grid: the cells, split info, hash and per-cell surf graphs are
  //   patched from the change journal the Grid primitives kept, so
  //   nothing sized by the grid crosses to the device; a full re-map
  //   (Grid::changed) is the exception and re-establishes everything
  // particles: NOT flagged as host-modified.  the deletion pass ran on the
  //   device and left the device copy authoritative; flagging the host
  //   copy here would mark a stale array as the newer one and discard the
  //   deletions on the next sync.  the host fallback path in
  //   remove_inside_all() does the flagging itself

  particle_kk->sorted_kk = 0;

  if (grid->changed) grid_kk->resync_after_host_change();
  else grid_kk->apply_changes();
  remap->listschanged = 0;
  remap->restructured = 0;

}

/* ----------------------------------------------------------------------
   the cells a restructure moved hold particles labelled with their old
     index; the host routine relabels them from the sorted per-cell
     lists, which the device copy of the particles never sees
   an old -> new map over the old cell range, applied to every particle
------------------------------------------------------------------------- */

void FixRigidKokkos::relabel_moved_cells()
{
  int nmoved = grid->nmoved;
  if (!nmoved || !particle->exist) return;

  ParticleKokkos *particle_kk = (ParticleKokkos*) particle;

  int nold = 0;
  for (int i = 0; i < nmoved; i++) nold = MAX(nold,grid->movedfrom[i]+1);

  if ((int) d_cellmap.extent(0) < nold)
    d_cellmap = DAT::t_int_1d("fix_rigid:cellmap",nold);
  if ((int) k_movedfrom.extent(0) < nmoved) {
    k_movedfrom = DAT::tdual_int_1d("fix_rigid:movedfrom",nmoved);
    k_movedto = DAT::tdual_int_1d("fix_rigid:movedto",nmoved);
  }
  auto h_from = k_movedfrom.view_host();
  auto h_to = k_movedto.view_host();
  for (int i = 0; i < nmoved; i++) {
    h_from(i) = grid->movedfrom[i];
    h_to(i) = grid->movedto[i];
  }
  k_movedfrom.modify_host(); k_movedfrom.sync_device();
  k_movedto.modify_host(); k_movedto.sync_device();
  d_movedfrom = k_movedfrom.view_device();
  d_movedto = k_movedto.view_device();

  const int prev_auto_sync = sparta->kokkos->auto_sync;
  sparta->kokkos->auto_sync = 0;

  particle_kk->sync(Device,PARTICLE_MASK);
  d_particles_kk = particle_kk->k_particles.view_device();
  nplocal_kk = particle->nlocal;

  copymode = 1;
  Kokkos::parallel_for(
    Kokkos::RangePolicy<DeviceType,TagFixRigidCellMapInit>(0,nold),*this);
  Kokkos::parallel_for(
    Kokkos::RangePolicy<DeviceType,TagFixRigidCellMapSet>(0,nmoved),*this);
  Kokkos::parallel_for(
    Kokkos::RangePolicy<DeviceType,TagFixRigidRelabel>(0,nplocal_kk),*this);
  copymode = 0;

  particle_kk->modify(Device,PARTICLE_MASK);
  particle->sorted = 0;
  particle_kk->sorted_kk = 0;

  sparta->kokkos->auto_sync = prev_auto_sync;
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixRigidKokkos::operator()(TagFixRigidCellMapInit, const int &i) const
{
  d_cellmap(i) = i;
}

KOKKOS_INLINE_FUNCTION
void FixRigidKokkos::operator()(TagFixRigidCellMapSet, const int &m) const
{
  d_cellmap(d_movedfrom(m)) = d_movedto(m);
}

KOKKOS_INLINE_FUNCTION
void FixRigidKokkos::operator()(TagFixRigidRelabel, const int &i) const
{
  const int icell = d_particles_kk[i].icell;
  if (icell < (int) d_cellmap.extent(0))
    d_particles_kk[i].icell = d_cellmap(icell);
}

/* ---------------------------------------------------------------------- */

void FixRigidKokkos::setup()
{
  host_begin();
  FixRigid::setup();

  // the host geometry and surfs are current at setup: the device takes
  //   them from the host once, then keeps them itself

  ((SurfKokkos*) surf)->modify(Host,ALL_MASK);
  pack_body_static();
  pack_body_geometry();
  devicegeom = 1;
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

  FixRigid::start_of_step();

  // distributed surfs: local copies of a body entering this proc were
  //   appended to the host surf arrays

  if (copiesappended) {
    surf_kk->modify(Host,ALL_MASK);
    copiesappended = 0;
  }

  // the move kernel's per-element force/torque tallies for this step

  if ((int) k_ftally.extent(0) < nsurf) {
    k_ftally = tdual_dbl_2d("fix_rigid:ftally",nsurf,6);
    d_ftally = k_ftally.view_device();
  }
  Kokkos::deep_copy(d_ftally,0.0);

  // the collision lists, and any re-indexed ghost cell lists, reach the
  //   device through the journal

  grid_kk->apply_changes();
  remap->listschanged = 0;
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

  // output at this step reads the host surfs; with distributed surfs a
  //   load balance or grid change restructures the host surf arrays
  //   before this fix hears of it, so they are kept current every step

  if (surf->distributed || output->next == update->ntimestep)
    refresh_host_surfs();
  host_end();
}

/* ----------------------------------------------------------------------
   full re-map of the surfs to the grid on the host, then the device grid
     re-established at once: the deletion and split-assign pass which
     follows runs on the device and reads the cells and the split graphs
     the re-map replaced, so it must not wait for host_end()
------------------------------------------------------------------------- */

void FixRigidKokkos::grid_rebuild()
{
  refresh_host_surfs();
  FixRigid::grid_rebuild();
  ((GridKokkos*) grid)->resync_after_host_change();
}

/* ----------------------------------------------------------------------
   the host surfs are read after the run: dumps, restart files, a
     write_surf, the next run's setup
------------------------------------------------------------------------- */

void FixRigidKokkos::post_run()
{
  refresh_host_surfs();
  FixRigid::post_run();
}

/* ----------------------------------------------------------------------
   per-surf maps, and the element tables the device keeps: the local
     copies of the body surfs changed
------------------------------------------------------------------------- */

void FixRigidKokkos::surf_maps()
{
  FixRigid::surf_maps();
  if (devicegeom) pack_body_static();
}

/* ----------------------------------------------------------------------
   the bodies to the end-of-step pose, then their geometry, boxes and
     surfs regenerated on the device; the host copies go stale and are
     regenerated per body on demand by host_geometry()
------------------------------------------------------------------------- */

void FixRigidKokkos::set_xv()
{
  set_pose();
  device_geometry(0);
  body_bins();
}

/* ----------------------------------------------------------------------
   the swept boxes of this step, on the device
------------------------------------------------------------------------- */

void FixRigidKokkos::swept_boxes()
{
  posesplit = 1;
  device_geometry(1);
}

/* ----------------------------------------------------------------------
   the host copy of one body's geometry, for a host consumer: the same
     arithmetic as the device kernel, so it is the same geometry
------------------------------------------------------------------------- */

void FixRigidKokkos::host_geometry(int ibody)
{
  if (!devicegeom || hostgeom[ibody]) return;
  body_geometry(ibody);
  hostgeom[ibody] = 1;
}

/* ----------------------------------------------------------------------
   the host copies of every body surf, for a host consumer of the surf
     arrays: output, a grid rebuild, every step with distributed surfs
   only with the copy tables of the current surf arrays: never between
     a restructure of the arrays and FixRigid::grid_changed()
------------------------------------------------------------------------- */

void FixRigidKokkos::refresh_host_surfs()
{
  if (!devicegeom) return;
  for (int ibody = 0; ibody < nbody; ibody++) host_geometry(ibody);
  update_surf_copies();
}

/* ----------------------------------------------------------------------
   the element tables to the device: displace, the body of each element,
     the element ranges, the local surf of each element and every local
     copy of an element
   called at setup and whenever the local copies change
------------------------------------------------------------------------- */

void FixRigidKokkos::pack_body_static()
{
  int nelem = bodystart[nbody];

  if (nelem > nelem_kk) {
    k_displace = tdual_dbl_3d("fix_rigid:displace",nelem,3,3);
    k_body = DAT::tdual_int_1d("fix_rigid:body",nelem);
    k_bodypt = tdual_dbl_3d("fix_rigid:bodypt",nelem,3,3);
    k_bodynorm = tdual_dbl_2d("fix_rigid:bodynorm",nelem,3);
    k_elemlo = tdual_dbl_2d("fix_rigid:elemlo",nelem,3);
    k_elemhi = tdual_dbl_2d("fix_rigid:elemhi",nelem,3);
    k_lblist = DAT::tdual_int_1d("fix_rigid:lblist",nelem);
    nelem_kk = nelem;
  }
  if (ncopy > ncopy_kk) {
    k_copy_index = DAT::tdual_int_1d("fix_rigid:copy_index",ncopy);
    k_copy_elem = DAT::tdual_int_1d("fix_rigid:copy_elem",ncopy);
    ncopy_kk = ncopy;
  }
  if (k_bodystart.extent(0) < (size_t)(nbody+1)) {
    k_bodystart = DAT::tdual_int_1d("fix_rigid:bodystart",nbody+1);
    k_bbodylo = tdual_dbl_2d("fix_rigid:bbodylo",nbody,3);
    k_bbodyhi = tdual_dbl_2d("fix_rigid:bbodyhi",nbody,3);
    k_pose = tdual_dbl_2d("fix_rigid:pose",nbody,16);
    k_bboxeps = tdual_dbl_1d("fix_rigid:bboxeps",nbody);
  }
  if (nbody > maxhostgeom) {
    memory->grow(hostgeom,nbody,"fix_rigid:hostgeom");
    for (int ib = maxhostgeom; ib < nbody; ib++) hostgeom[ib] = 1;
    maxhostgeom = nbody;
  }

  auto h_displace = k_displace.view_host();
  auto h_body = k_body.view_host();
  auto h_lblist = k_lblist.view_host();
  for (int i = 0; i < nelem; i++) {
    for (int j = 0; j < dim; j++)
      for (int k = 0; k < 3; k++) h_displace(i,j,k) = displace[i][j][k];
    h_body(i) = FixRigid::body[i];
    h_lblist(i) = lblist[i];
  }
  k_displace.modify_host(); k_displace.sync_device();
  k_body.modify_host(); k_body.sync_device();
  k_lblist.modify_host(); k_lblist.sync_device();

  auto h_copy_index = k_copy_index.view_host();
  auto h_copy_elem = k_copy_elem.view_host();
  for (int m = 0; m < ncopy; m++) {
    h_copy_index(m) = copy_index[m];
    h_copy_elem(m) = copy_elem[m];
  }
  k_copy_index.modify_host(); k_copy_index.sync_device();
  k_copy_elem.modify_host(); k_copy_elem.sync_device();

  auto h_bodystart = k_bodystart.view_host();
  for (int ib = 0; ib <= nbody; ib++) h_bodystart(ib) = bodystart[ib];
  k_bodystart.modify_host(); k_bodystart.sync_device();
}

/* ----------------------------------------------------------------------
   the host geometry to the device, at setup: the corner points,
     normals, element boxes and body boxes the host computed
------------------------------------------------------------------------- */

void FixRigidKokkos::pack_body_geometry()
{
  int nelem = bodystart[nbody];

  auto h_bodypt = k_bodypt.view_host();
  auto h_bodynorm = k_bodynorm.view_host();
  auto h_elemlo = k_elemlo.view_host();
  auto h_elemhi = k_elemhi.view_host();
  for (int i = 0; i < nelem; i++) {
    for (int j = 0; j < dim; j++)
      for (int k = 0; k < 3; k++) h_bodypt(i,j,k) = bodypt[i][j][k];
    for (int k = 0; k < 3; k++) {
      h_bodynorm(i,k) = bodynorm[i][k];
      h_elemlo(i,k) = elemlo[i][k];
      h_elemhi(i,k) = elemhi[i][k];
    }
  }
  k_bodypt.modify_host(); k_bodypt.sync_device();
  k_bodynorm.modify_host(); k_bodynorm.sync_device();
  k_elemlo.modify_host(); k_elemlo.sync_device();
  k_elemhi.modify_host(); k_elemhi.sync_device();

  auto h_bbodylo = k_bbodylo.view_host();
  auto h_bbodyhi = k_bbodyhi.view_host();
  auto h_bboxeps = k_bboxeps.view_host();
  for (int ib = 0; ib < nbody; ib++) {
    for (int k = 0; k < 3; k++) {
      h_bbodylo(ib,k) = bbodylo[ib][k];
      h_bbodyhi(ib,k) = bbodyhi[ib][k];
    }
    h_bboxeps(ib) = bboxeps[ib];
    hostgeom[ib] = 1;
  }
  k_bbodylo.modify_host(); k_bbodylo.sync_device();
  k_bbodyhi.modify_host(); k_bbodyhi.sync_device();
  k_bboxeps.modify_host(); k_bboxeps.sync_device();
}

/* ----------------------------------------------------------------------
   the body geometry from the pose, on the device: the twin of
     FixRigid::body_geometry() for every body, or of body_bbox(ibody,1)
     for every body if sweepflag is set
   sweepflag = 0: corner points and normals from the current pose, the
     surfs rewritten, element boxes and body bboxes of the current
     positions
   sweepflag = 1: the element boxes and body bboxes also bound the
     end-of-step positions; the points and surfs are left alone
   the body bboxes and their inflation come back to the host, which
     reads them; the host copies of the geometry go stale
------------------------------------------------------------------------- */

void FixRigidKokkos::device_geometry(int sweepflag)
{
  int nelem = bodystart[nbody];

  auto h_pose = k_pose.view_host();
  for (int ib = 0; ib < nbody; ib++) {
    double *x = sweepflag ? xcmnew[ib] : xcm[ib];
    for (int k = 0; k < 3; k++) {
      h_pose(ib,k) = x[k];
      h_pose(ib,3+k) = ex_space[ib][k];
      h_pose(ib,6+k) = ey_space[ib][k];
      h_pose(ib,9+k) = ez_space[ib][k];
      h_pose(ib,12+k) = omega[ib][k];
    }
    h_pose(ib,15) = rmaxbody[ib];
  }
  k_pose.modify_host(); k_pose.sync_device();

  d_pose_kk = k_pose.view_device();
  d_displace_kk = k_displace.view_device();
  d_bodypt_kk = k_bodypt.view_device();
  d_bodynorm_kk = k_bodynorm.view_device();
  d_elemlo_kk = k_elemlo.view_device();
  d_elemhi_kk = k_elemhi.view_device();
  d_bbodylo_kk = k_bbodylo.view_device();
  d_bbodyhi_kk = k_bbodyhi.view_device();
  d_bboxeps_kk = k_bboxeps.view_device();
  d_body_kk = k_body.view_device();
  d_bodystart_kk = k_bodystart.view_device();
  d_copy_index_kk = k_copy_index.view_device();
  d_copy_elem_kk = k_copy_elem.view_device();
  SurfKokkos *surf_kk = (SurfKokkos*) surf;
  if (dim == 2) d_lines_kk = surf_kk->k_lines.view_device();
  else d_tris_kk = surf_kk->k_tris.view_device();
  sweep_kk = sweepflag;
  axiflag_kk = axiflag;
  dim_kk = dim;
  dt_kk = update->dt;

  copymode = 1;
  Kokkos::parallel_for(
    Kokkos::RangePolicy<DeviceType,TagFixRigidGeometry>(0,nelem),*this);
  Kokkos::parallel_for(
    Kokkos::RangePolicy<DeviceType,TagFixRigidBodyBox>(0,nbody),*this);
  Kokkos::parallel_for(
    Kokkos::RangePolicy<DeviceType,TagFixRigidInflate>(0,nelem),*this);
  if (!sweepflag)
    Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType,TagFixRigidScatterSurfs>(0,ncopy),*this);
  copymode = 0;

  k_bbodylo.modify_device(); k_bbodylo.sync_host();
  k_bbodyhi.modify_device(); k_bbodyhi.sync_host();
  k_bboxeps.modify_device(); k_bboxeps.sync_host();
  auto h_bbodylo = k_bbodylo.view_host();
  auto h_bbodyhi = k_bbodyhi.view_host();
  auto h_bboxeps = k_bboxeps.view_host();
  for (int ib = 0; ib < nbody; ib++) {
    for (int k = 0; k < 3; k++) {
      bbodylo[ib][k] = h_bbodylo(ib,k);
      bbodyhi[ib][k] = h_bbodyhi(ib,k);
    }
    bboxeps[ib] = h_bboxeps(ib);
    if (!sweepflag) hostgeom[ib] = 0;
  }
}

/* ----------------------------------------------------------------------
   one thread per element: FixRigid::body_geometry() for its points and
     normal (not when sweeping: the points are those of the start of the
     step), then its box, over the current points and, when sweeping,
     the end-of-step points
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixRigidKokkos::operator()(TagFixRigidGeometry, const int &i) const
{
  int j,k;
  double v[3],delta[3],delta12[3],delta13[3],ptnew[3],lo[3],hi[3];
  double z[3];
  z[0] = 0.0; z[1] = 0.0; z[2] = 1.0;

  const int ib = d_body_kk(i);
  double xcm1[3],ex[3],ey[3],ez[3];
  for (k = 0; k < 3; k++) {
    xcm1[k] = d_pose_kk(ib,k);
    ex[k] = d_pose_kk(ib,3+k);
    ey[k] = d_pose_kk(ib,6+k);
    ez[k] = d_pose_kk(ib,9+k);
  }

  if (!sweep_kk) {
    for (j = 0; j < dim_kk; j++) {
      for (k = 0; k < 3; k++) v[k] = d_displace_kk(i,j,k);
      if (axiflag_kk) {
        d_bodypt_kk(i,j,0) = xcm1[0] + v[0];
        d_bodypt_kk(i,j,1) = v[1];
        d_bodypt_kk(i,j,2) = 0.0;
        continue;
      }
      delta[0] = ex[0]*v[0] + ey[0]*v[1] + ez[0]*v[2];
      delta[1] = ex[1]*v[0] + ey[1]*v[1] + ez[1]*v[2];
      delta[2] = ex[2]*v[0] + ey[2]*v[1] + ez[2]*v[2];
      if (dim_kk == 2) delta[2] = 0.0;
      for (k = 0; k < 3; k++) d_bodypt_kk(i,j,k) = xcm1[k] + delta[k];
    }

    double norm[3];
    if (dim_kk == 2) {
      for (k = 0; k < 3; k++) delta[k] = d_bodypt_kk(i,1,k) - d_bodypt_kk(i,0,k);
      norm[0] = z[1]*delta[2] - z[2]*delta[1];
      norm[1] = z[2]*delta[0] - z[0]*delta[2];
      norm[2] = z[0]*delta[1] - z[1]*delta[0];
    } else {
      for (k = 0; k < 3; k++) {
        delta12[k] = d_bodypt_kk(i,1,k) - d_bodypt_kk(i,0,k);
        delta13[k] = d_bodypt_kk(i,2,k) - d_bodypt_kk(i,0,k);
      }
      norm[0] = delta12[1]*delta13[2] - delta12[2]*delta13[1];
      norm[1] = delta12[2]*delta13[0] - delta12[0]*delta13[2];
      norm[2] = delta12[0]*delta13[1] - delta12[1]*delta13[0];
    }
    const double scale = 1.0/sqrt(norm[0]*norm[0]+norm[1]*norm[1]+norm[2]*norm[2]);
    norm[0] *= scale;
    norm[1] *= scale;
    norm[2] *= scale;
    if (dim_kk == 2) norm[2] = 0.0;
    for (k = 0; k < 3; k++) d_bodynorm_kk(i,k) = norm[k];
  }

  // FixRigid::body_bbox(): the box of the element, current points and,
  //   when sweeping, the end-of-step points from the new pose

  lo[0] = lo[1] = lo[2] = BIG;
  hi[0] = hi[1] = hi[2] = -BIG;

  for (j = 0; j < dim_kk; j++)
    for (k = 0; k < 3; k++) {
      lo[k] = MIN(lo[k],d_bodypt_kk(i,j,k));
      hi[k] = MAX(hi[k],d_bodypt_kk(i,j,k));
    }

  if (sweep_kk) {
    for (j = 0; j < dim_kk; j++) {
      for (k = 0; k < 3; k++) v[k] = d_displace_kk(i,j,k);
      if (axiflag_kk) {
        ptnew[0] = xcm1[0] + v[0];
        ptnew[1] = v[1];
        ptnew[2] = 0.0;
      } else {
        delta[0] = ex[0]*v[0] + ey[0]*v[1] + ez[0]*v[2];
        delta[1] = ex[1]*v[0] + ey[1]*v[1] + ez[1]*v[2];
        delta[2] = ex[2]*v[0] + ey[2]*v[1] + ez[2]*v[2];
        if (dim_kk == 2) delta[2] = 0.0;
        for (k = 0; k < 3; k++) ptnew[k] = xcm1[k] + delta[k];
      }
      for (k = 0; k < 3; k++) {
        lo[k] = MIN(lo[k],ptnew[k]);
        hi[k] = MAX(hi[k],ptnew[k]);
      }
    }
  }

  for (k = 0; k < 3; k++) {
    d_elemlo_kk(i,k) = lo[k];
    d_elemhi_kk(i,k) = hi[k];
  }
}

/* ----------------------------------------------------------------------
   one thread per body: its bbox over its element boxes and the
     inflation FixRigid::body_bbox() applies to both, the arc bulge of
     the rotation included when sweeping
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixRigidKokkos::operator()(TagFixRigidBodyBox, const int &ib) const
{
  int k;
  double blo[3],bhi[3];

  blo[0] = blo[1] = blo[2] = BIG;
  bhi[0] = bhi[1] = bhi[2] = -BIG;

  for (int i = d_bodystart_kk(ib); i < d_bodystart_kk(ib+1); i++)
    for (k = 0; k < 3; k++) {
      blo[k] = MIN(blo[k],d_elemlo_kk(i,k));
      bhi[k] = MAX(bhi[k],d_elemhi_kk(i,k));
    }

  double eps = EPSSURF * MAX(bhi[0]-blo[0],bhi[1]-blo[1]);
  eps = EPSSURF * MAX(eps/EPSSURF,bhi[2]-blo[2]);

  if (sweep_kk && !axiflag_kk) {
    const double wx = d_pose_kk(ib,12);
    const double wy = d_pose_kk(ib,13);
    const double wz = d_pose_kk(ib,14);
    const double angle = sqrt(wx*wx + wy*wy + wz*wz) * dt_kk;
    eps += 0.125 * d_pose_kk(ib,15) * angle * angle;
  }

  d_bboxeps_kk(ib) = eps;
  for (k = 0; k < 3; k++) {
    d_bbodylo_kk(ib,k) = blo[k] - eps;
    d_bbodyhi_kk(ib,k) = bhi[k] + eps;
  }
}

/* ----------------------------------------------------------------------
   one thread per element: its box inflated by its body's eps
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixRigidKokkos::operator()(TagFixRigidInflate, const int &i) const
{
  const double eps = d_bboxeps_kk(d_body_kk(i));
  for (int k = 0; k < 3; k++) {
    d_elemlo_kk(i,k) -= eps;
    d_elemhi_kk(i,k) += eps;
  }
}

/* ----------------------------------------------------------------------
   one thread per local surf copy of an element: the surf takes the
     element's points and normal, FixRigid::update_surf_copies()
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixRigidKokkos::operator()(TagFixRigidScatterSurfs, const int &m) const
{
  const int index = d_copy_index_kk(m);
  const int i = d_copy_elem_kk(m);
  if (dim_kk == 2) {
    for (int k = 0; k < 3; k++) {
      d_lines_kk[index].p1[k] = d_bodypt_kk(i,0,k);
      d_lines_kk[index].p2[k] = d_bodypt_kk(i,1,k);
      d_lines_kk[index].norm[k] = d_bodynorm_kk(i,k);
    }
  } else {
    for (int k = 0; k < 3; k++) {
      d_tris_kk[index].p1[k] = d_bodypt_kk(i,0,k);
      d_tris_kk[index].p2[k] = d_bodypt_kk(i,1,k);
      d_tris_kk[index].p3[k] = d_bodypt_kk(i,2,k);
      d_tris_kk[index].norm[k] = d_bodynorm_kk(i,k);
    }
  }
}


/* ----------------------------------------------------------------------
   per-body sums of the move kernel's per-element tallies into ftbuf_mine
   one thread per body, its elements in element order on every backend,
     the order the host sums its rows in
------------------------------------------------------------------------- */

void FixRigidKokkos::sum_tallies()
{
  if ((int) k_ft.extent(0) < nbody) {
    k_ft = tdual_dbl_2d("fix_rigid:ft",nbody,6);
    d_ft = k_ft.view_device();
  }
  d_bodystart = k_bodystart.view_device();

  copymode = 1;
  Kokkos::parallel_for(
    Kokkos::RangePolicy<DeviceType,TagFixRigidSumTallies>(0,nbody),*this);
  copymode = 0;

  k_ft.modify_device();
  k_ft.sync_host();
  auto h_ft = k_ft.view_host();
  for (int ibody = 0; ibody < nbody; ibody++)
    for (int j = 0; j < 6; j++) ftbuf_mine[6*ibody+j] = h_ft(ibody,j);
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixRigidKokkos::operator()(TagFixRigidSumTallies, const int &ibody) const
{
  double f[6];
  for (int j = 0; j < 6; j++) f[j] = 0.0;

  const int start = d_bodystart(ibody);
  const int stop = d_bodystart(ibody+1);
  for (int k = start; k < stop; k++)
    for (int j = 0; j < 6; j++) f[j] += d_ftally(k,j);

  for (int j = 0; j < 6; j++) d_ft(ibody,j) = f[j];
}

/* ----------------------------------------------------------------------
   pull the particles of every changed split cell up into the split cell
   device version of FixRigid::combine_split_all(), which was the last
     per-step host pass over the particle array
   only the icell LABEL matters here: the host routine also splices the sub
     cells' particle lists into the split cell's, so that
     Grid::remove_marked_cells() can walk them, but the cells it moves are
     the sub cells which split_cell_unset() has already emptied, so that
     walk never sees a particle (measured: 7149 moved cells, 0 particles
     relabelled, across every split-cell test in the suite).  the lists
     themselves are rebuilt by the next sort
------------------------------------------------------------------------- */

void FixRigidKokkos::combine_split_all()
{
  if (combine_split_kokkos()) return;
  FixRigid::combine_split_all();
}

/* ----------------------------------------------------------------------
   one thread per sub cell of a changed split cell: relabel every particle
     of the sub cell to the split cell which owns it
   d_plist/d_cellcount are the device per-cell particle lists built by
     ParticleKokkos::sort_kokkos(), the device form of cinfo.first/next[]
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixRigidKokkos::operator()(TagFixRigidCombineSplit, const int &m) const
{
  const int isub = d_subcell(m);
  const int iparent = d_subparent(m);
  const int n = d_cellcount_kk(isub);

  for (int k = 0; k < n; k++) {
    const int ip = d_plist2_kk(isub,k);
    d_particles_kk[ip].icell = iparent;
  }
}

/* ----------------------------------------------------------------------
   return 1 if the relabel was done on the device, 0 to use the host
------------------------------------------------------------------------- */

int FixRigidKokkos::combine_split_kokkos()
{
  ParticleKokkos *particle_kk = (ParticleKokkos*) particle;
  GridKokkos *grid_kk = (GridKokkos*) grid;

  // build the (sub cell -> split cell) pairs on the host: O(nsplit), no
  //   particle access, and the sub cell indices are needed by the kernel

  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;
  int nsplitlocal = grid->nsplitlocal;

  // the owned split cells, from the split info: O(nsplit), no scan of
  //   the cells; an entry a restructure abandoned points at no cell

  int nsub = 0;
  for (int i = 0; i < nsplitlocal; i++) {
    int icell = sinfo[i].icell;
    if (icell < 0 || cells[icell].isplit != i || cells[icell].nsplit <= 1)
      continue;
    nsub += cells[icell].nsplit;
  }
  if (!nsub) return 1;

  if (nsub > nsub_kk) {
    nsub_kk = nsub;
    k_subcell = DAT::tdual_int_1d("fix_rigid:subcell",nsub_kk);
    k_subparent = DAT::tdual_int_1d("fix_rigid:subparent",nsub_kk);
    d_subcell = k_subcell.view_device();
    d_subparent = k_subparent.view_device();
  }

  auto h_subcell = k_subcell.view_host();
  auto h_subparent = k_subparent.view_host();

  int m = 0;
  for (int i = 0; i < nsplitlocal; i++) {
    int icell = sinfo[i].icell;
    if (icell < 0 || cells[icell].isplit != i || cells[icell].nsplit <= 1)
      continue;
    int *mycsubs = sinfo[i].csubs;
    for (int j = 0; j < cells[icell].nsplit; j++) {
      h_subcell(m) = mycsubs[j];
      h_subparent(m) = icell;
      m++;
    }
  }
  k_subcell.modify_host(); k_subcell.sync_device();
  k_subparent.modify_host(); k_subparent.sync_device();

  // the per-cell lists must be current and must cover every sub cell index
  //   in the work list: sort_kokkos() sizes them to grid->nlocal as it was
  //   at the sort, so a sub cell added since then would be out of bounds

  const int prev_auto_sync = sparta->kokkos->auto_sync;
  sparta->kokkos->auto_sync = 0;

  particle_kk->sync(Device,PARTICLE_MASK);
  if (!particle_kk->sorted_kk) particle_kk->sort_kokkos();

  d_particles_kk = particle_kk->k_particles.view_device();
  d_plist2_kk = grid_kk->d_plist;
  d_cellcount_kk = grid_kk->d_cellcount;

  // every sub cell index must be inside the lists built by the sort

  int maxsub = 0;
  for (int i = 0; i < nsub; i++) maxsub = MAX(maxsub,h_subcell(i));
  if (maxsub >= (int) d_cellcount_kk.extent(0)) {
    sparta->kokkos->auto_sync = prev_auto_sync;
    return 0;
  }

  copymode = 1;
  Kokkos::parallel_for(
    Kokkos::RangePolicy<DeviceType,TagFixRigidCombineSplit>(0,nsub),*this);
  copymode = 0;

  particle_kk->modify(Device,PARTICLE_MASK);

  // the particles are no longer listed under the cells they are labelled
  //   with, exactly as after the host routine

  particle->sorted = 0;
  particle_kk->sorted_kk = 0;

  sparta->kokkos->auto_sync = prev_auto_sync;

  // cinfo.count/first for the split cell and its sub cells: host-side
  //   bookkeeping which Grid::remove_marked_cells() reads, and which the
  //   host routine sets as it splices.  no particle access

  Grid::ChildInfo *cinfo = grid->cinfo;
  for (int i = 0; i < nsplitlocal; i++) {
    int icell = sinfo[i].icell;
    if (icell < 0 || cells[icell].isplit != i || cells[icell].nsplit <= 1)
      continue;
    int *mycsubs = sinfo[i].csubs;
    int count = 0;
    for (int j = 0; j < cells[icell].nsplit; j++) {
      count += cinfo[mycsubs[j]].count;
      cinfo[mycsubs[j]].count = 0;
      cinfo[mycsubs[j]].first = -1;
    }
    cinfo[icell].count = count;
    cinfo[icell].first = -1;   // the list is stale; the next sort rebuilds it
  }

  return 1;
}


/* ----------------------------------------------------------------------
   which sub cell of split cell icell holds the point x
   the same tests, in the same order, as Update::split2d/split3d and their
     UpdateKokkos twins, reading the same device views
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
int FixRigidKokkos::split2d_kk(int icell, double *x) const
{
  const int nsurf = d_cells_kk[icell].nsurf;
  const int isplit = d_cells_kk[icell].isplit;
  double *xnew = d_sinfo_kk[isplit].xsplit;

  int cflag = 0, minsurfindex = 0;
  double minparam = 2.0;
  auto csplits_begin = d_csplits_kk.row_map(isplit);
  auto csurfs_begin = d_csurfs_kk.row_map(icell);

  for (int m = 0; m < nsurf; m++) {
    if (d_csplits_kk.entries(csplits_begin + m) < 0) continue;
    const int isurf = d_csurfs_kk.entries(csurfs_begin + m);
    double xc[3],param;
    int side;
    const bool hitflag = GeometryKokkos::
      line_line_intersect(x,xnew,d_lines_kk[isurf].p1,d_lines_kk[isurf].p2,
                          d_lines_kk[isurf].norm,xc,param,side);
    if (hitflag && side != INSIDE && param < minparam) {
      cflag = 1;
      minparam = param;
      minsurfindex = m;
    }
  }

  auto csubs_begin = d_csubs_kk.row_map(isplit);
  if (!cflag) return d_csubs_kk.entries(csubs_begin + d_sinfo_kk[isplit].xsub);
  const int index = d_csplits_kk.entries(csplits_begin + minsurfindex);
  return d_csubs_kk.entries(csubs_begin + index);
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
int FixRigidKokkos::split3d_kk(int icell, double *x) const
{
  const int nsurf = d_cells_kk[icell].nsurf;
  const int isplit = d_cells_kk[icell].isplit;
  double *xnew = d_sinfo_kk[isplit].xsplit;

  int cflag = 0, minsurfindex = 0;
  double minparam = 2.0;
  auto csplits_begin = d_csplits_kk.row_map(isplit);
  auto csurfs_begin = d_csurfs_kk.row_map(icell);

  for (int m = 0; m < nsurf; m++) {
    if (d_csplits_kk.entries(csplits_begin + m) < 0) continue;
    const int isurf = d_csurfs_kk.entries(csurfs_begin + m);
    double xc[3],param;
    int side;
    const bool hitflag = GeometryKokkos::
      line_tri_intersect(x,xnew,d_tris_kk[isurf].p1,d_tris_kk[isurf].p2,
                         d_tris_kk[isurf].p3,d_tris_kk[isurf].norm,
                         xc,param,side);
    if (hitflag && side != INSIDE && param < minparam) {
      cflag = 1;
      minparam = param;
      minsurfindex = m;
    }
  }

  auto csubs_begin = d_csubs_kk.row_map(isplit);
  if (!cflag) return d_csubs_kk.entries(csubs_begin + d_sinfo_kk[isplit].xsub);
  const int index = d_csplits_kk.entries(csplits_begin + minsurfindex);
  return d_csubs_kk.entries(csubs_begin + index);
}

/* ----------------------------------------------------------------------
   one thread per particle of a changed split cell: pick its sub cell
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixRigidKokkos::operator()(TagFixRigidAssignSplit, const int &m) const
{
  const int icell = d_asgcell(m);
  const int ip = d_asgpart(m);
  double *x = d_particles_kk[ip].x;
  d_particles_kk[ip].icell =
    (dim_kk == 3) ? split3d_kk(icell,x) : split2d_kk(icell,x);
}

/* ----------------------------------------------------------------------
   device replacement for the host assign_split_cell_particles() pass
   return 1 if it was done here, 0 to leave it to the host
------------------------------------------------------------------------- */

int FixRigidKokkos::assign_split_kokkos()
{
  ParticleKokkos *particle_kk = (ParticleKokkos*) particle;
  GridKokkos *grid_kk = (GridKokkos*) grid;
  SurfKokkos *surf_kk = (SurfKokkos*) surf;

  const int prev_auto_sync = sparta->kokkos->auto_sync;
  sparta->kokkos->auto_sync = 0;

  // the per-cell lists must cover the split cells at their CURRENT indices,
  //   so sort after the cell set is final

  particle_kk->sync(Device,PARTICLE_MASK);
  if (!particle_kk->sorted_kk) particle_kk->sort_kokkos();

  // the owned split cells, from the split info: O(nsplit), no scan of
  //   the cells; an entry a restructure abandoned points at no cell

  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;
  int nsplitlocal = grid->nsplitlocal;

  int nsplit = 0;
  for (int i = 0; i < nsplitlocal; i++) {
    int icell = sinfo[i].icell;
    if (icell < 0 || cells[icell].isplit != i || cells[icell].nsplit <= 1)
      continue;
    nsplit++;
  }
  if (!nsplit) {
    sparta->kokkos->auto_sync = prev_auto_sync;
    return 1;
  }

  // d_cellcount/d_plist are sized to grid->nlocal as it was at the last
  //   sort_kokkos(); a cell added since then is out of their range, and
  //   a count beyond the list capacity means an incomplete list: let
  //   the host handle such a step rather than read past the end

  d_cellcount_kk = grid_kk->d_cellcount;
  d_plist2_kk = grid_kk->d_plist;
  const int ncnt = (int) d_cellcount_kk.extent(0);
  if (grid->nlocal > ncnt) {
    sparta->kokkos->auto_sync = prev_auto_sync;
    return 0;
  }
  const int pcap = (int) d_plist2_kk.extent(1);

  if (nsplit > nsplit_kk) {
    nsplit_kk = nsplit;
    k_splitcells = DAT::tdual_int_1d("fix_rigid:splitcells",nsplit_kk);
    d_splitcells = k_splitcells.view_device();
    d_splitoff = DAT::t_int_1d("fix_rigid:splitoff",nsplit_kk+1);
  }
  auto h_splitcells = k_splitcells.view_host();
  int m = 0;
  for (int i = 0; i < nsplitlocal; i++) {
    int icell = sinfo[i].icell;
    if (icell < 0 || cells[icell].isplit != i || cells[icell].nsplit <= 1)
      continue;
    h_splitcells(m++) = icell;
  }
  k_splitcells.modify_host(); k_splitcells.sync_device();

  // flat work list: one entry per particle of a split cell, from the
  //   device per-cell counts and lists, so the host never reads either

  auto d_splitcells = this->d_splitcells;
  auto d_splitoff = this->d_splitoff;
  auto d_cellcount = d_cellcount_kk;
  auto d_plist = d_plist2_kk;

  int maxcount = 0;
  Kokkos::parallel_reduce(nsplit, KOKKOS_LAMBDA(const int i, int &mx) {
    const int n = d_cellcount(d_splitcells(i));
    if (n > mx) mx = n;
  },Kokkos::Max<int>(maxcount));
  if (maxcount > pcap) {
    sparta->kokkos->auto_sync = prev_auto_sync;
    return 0;
  }

  Kokkos::parallel_scan(nsplit, KOKKOS_LAMBDA(const int i, int &sum,
                                              const bool final) {
    const int n = d_cellcount(d_splitcells(i));
    if (final) d_splitoff(i) = sum;
    sum += n;
    if (final && i == nsplit-1) d_splitoff(nsplit) = sum;
  });
  int nasg;
  Kokkos::deep_copy(nasg,Kokkos::subview(d_splitoff,nsplit));

  if (!nasg) {
    sparta->kokkos->auto_sync = prev_auto_sync;
    return 1;
  }

  if (nasg > nasg_kk) {
    nasg_kk = nasg;
    d_asgcell = DAT::t_int_1d("fix_rigid:asgcell",nasg_kk);
    d_asgpart = DAT::t_int_1d("fix_rigid:asgpart",nasg_kk);
  }
  auto d_asgcell = this->d_asgcell;
  auto d_asgpart = this->d_asgpart;
  Kokkos::parallel_for(nsplit, KOKKOS_LAMBDA(const int i) {
    const int icell = d_splitcells(i);
    const int off = d_splitoff(i);
    const int n = d_cellcount(icell);
    for (int k = 0; k < n; k++) {
      d_asgcell(off+k) = icell;
      d_asgpart(off+k) = d_plist(icell,k);
    }
  });

  d_particles_kk = particle_kk->k_particles.view_device();

  // the split tests read cells, sinfo, the per-cell graphs and the surf
  //   lines/tris on the device.  FixRigid has just rewritten the cells on
  //   the HOST (the re-cut, the restructure), so the device is patched
  //   from the change journal first: reading the device state as it was
  //   would dereference split info of cells which no longer exist; the
  //   surfs were regenerated on the device by set_xv()

  grid_kk->apply_changes();

  d_cells_kk = grid_kk->k_cells.view_device();
  d_sinfo_kk = grid_kk->k_sinfo.view_device();
  d_csurfs_kk = grid_kk->d_csurfs;
  d_csplits_kk = grid_kk->d_csplits;
  d_csubs_kk = grid_kk->d_csubs;
  if (dim == 2) d_lines_kk = surf_kk->k_lines.view_device();
  else d_tris_kk = surf_kk->k_tris.view_device();
  dim_kk = dim;

  copymode = 1;
  Kokkos::parallel_for(
    Kokkos::RangePolicy<DeviceType,TagFixRigidAssignSplit>(0,nasg),*this);
  copymode = 0;

  particle_kk->modify(Device,PARTICLE_MASK);
  particle->sorted = 0;
  particle_kk->sorted_kk = 0;

  sparta->kokkos->auto_sync = prev_auto_sync;

  // the split cells no longer hold the particles, their sub cells do

  Grid::ChildInfo *cinfo = grid->cinfo;
  for (int i = 0; i < nsplit; i++) {
    cinfo[h_splitcells(i)].count = 0;
    cinfo[h_splitcells(i)].first = -1;
  }

  return 1;
}

/* ----------------------------------------------------------------------
   Grid::remove_marked_cells() walks the per-cell particle list of each cell
     it moves, and FixRigid sorts the particles first to make those lists
     valid.  the cells it moves are the detached sub cells, whose lists
     split_cell_unset() has already emptied, so the walk never reaches a
     particle: measured over every split-cell test in the suite, 7149 cells
     moved and 0 particles relabelled
   the sort is therefore only making empty lists valid, and skipping it is
     what keeps the particle array on the device for a whole step
   combine_split_kokkos() has already emptied every sub cell's list and left
     the split cell's head at -1, which is what an empty list looks like, so
     the host state the routine reads is consistent
------------------------------------------------------------------------- */

void FixRigidKokkos::sort_for_split_rebuild()
{
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
  int nbins = bodynbin[0]*bodynbin[1]*bodynbin[2];

  if (nbins > nbin_kk || k_bodybinstart.extent(0) < (size_t)(nbins+1)) {
    k_bodybinstart = DAT::tdual_int_1d("fix_rigid:bodybinstart",nbins+1);
    nbin_kk = nbins;
  }
  if (k_bodybinlist.extent(0) < (size_t)nbody)
    k_bodybinlist = DAT::tdual_int_1d("fix_rigid:bodybinlist",nbody);

  auto h_binstart = k_bodybinstart.view_host();
  auto h_binlist = k_bodybinlist.view_host();
  for (int i = 0; i <= nbins; i++) h_binstart(i) = bodybinstart[i];
  for (int i = 0; i < nbody; i++) h_binlist(i) = bodybinlist[i];
  k_bodybinstart.modify_host(); k_bodybinstart.sync_device();
  k_bodybinlist.modify_host(); k_bodybinlist.sync_device();

  body.nbody = nbody;
  body.dim = dim;
  body.d_bodypt = k_bodypt.view_device();
  body.d_bodynorm = k_bodynorm.view_device();
  body.d_elemlo = k_elemlo.view_device();
  body.d_elemhi = k_elemhi.view_device();
  body.d_bbodylo = k_bbodylo.view_device();
  body.d_bbodyhi = k_bbodyhi.view_device();
  body.d_bodystart = k_bodystart.view_device();
  body.d_lblist = k_lblist.view_device();
  body.d_binstart = k_bodybinstart.view_device();
  body.d_binlist = k_bodybinlist.view_device();
  for (int k = 0; k < 3; k++) {
    body.nbin[k] = bodynbin[k];
    body.binlo[k] = bodybinlo[k];
    body.bininv[k] = bodybininv[k];
  }
  body.rmaxall = rmaxall;
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

  int ctype = CELLOUTSIDE;
  if (icell < nlocal_kk) ctype = d_cinfo_kk[icell].type;
  const int inside = (ctype == CELLINSIDE);

  // a surf-free OUTSIDE cell cannot contain a point interior to a body,
  //   since a body boundary crossing the cell would put a surf in it

  if (!inside && ctype == CELLOUTSIDE && d_cells_kk[icell].nsurf == 0) return;

  double *x = d_particles_kk[i].x;
  const int inbody = body.inside_any_body(x);

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
   the split-cell reassignment runs first, on the device, so the particles
     never have to come back to the host for it
------------------------------------------------------------------------- */

int FixRigidKokkos::remove_inside_all_kokkos(int splitflag)
{
  // the split-cell reassignment re-decides the sub cell of every particle of
  //   a changed split cell.  it must not be skipped: it changes results (the
  //   trajectory of the 1000-body deck moves, and moves TOWARD the
  //   non-KOKKOS reference on 6 of 7 reported fields), so if the device pass
  //   declines we have to bring the particles over and let FixRigid do it
  // a body which repeatedly creates and destroys split cells sets splitflag
  //   on every step (splitchanged/structural), so this is the common case,
  //   not a corner

  if (splitflag && grid->nsplitlocal && !assign_split_kokkos()) {
    particles_to_host();
    if (!particle->sorted) particle->sort();
    Grid::ChildCell *cells = grid->cells;
    int nglocal = grid->nlocal;
    for (int icell = 0; icell < nglocal; icell++)
      if (cells[icell].nsplit > 1)
        grid->assign_split_cell_particles(icell);
    particle->sorted = 0;
    ((ParticleKokkos*) particle)->modify(Host,PARTICLE_MASK|CUSTOM_MASK);
    ((ParticleKokkos*) particle)->sorted_kk = 0;
  }
  // a body which repeatedly creates and destroys split cells sets splitflag
  //   on every step (splitchanged/structural), so declining the device path
  //   here would give up on it entirely for exactly the problems that need
  //   it most

  if (!particle->exist) return 1;

  ParticleKokkos *particle_kk = (ParticleKokkos*) particle;
  GridKokkos *grid_kk = (GridKokkos*) grid;

  pack_body_device();

  // the two per-cell fields the test reads come from the device grid,
  //   patched from the change journal; a ghost cell has no ChildInfo

  grid_kk->apply_changes();
  d_cells_kk = grid_kk->k_cells.view_device();
  d_cinfo_kk = grid_kk->k_cinfo.view_device();
  nlocal_kk = grid->nlocal;

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
