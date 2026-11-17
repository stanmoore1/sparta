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
#include "stdlib.h"
#include "string.h"
#include "comm_kokkos.h"
#include "collide_vss_kokkos.h"
#include "irregular_kokkos.h"
#include "particle_kokkos.h"
#include "grid_kokkos.h"
#include "update.h"
//#include "adapt_grid.h"
#include "memory_kokkos.h"
#include "error.h"
#include "kokkos.h"
#include "sparta_masks.h"

using namespace SPARTA_NS;

enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};   // several files

#define BUFFACTOR 1.5

/* ---------------------------------------------------------------------- */

CommKokkos::CommKokkos(SPARTA *sparta) : Comm(sparta),
  particle_kk_copy(sparta)
{
  delete iparticle;
  iparticle = new IrregularKokkos(sparta);

  k_nsend = DAT::tdual_int_scalar("comm:nsend");
  d_nsend = k_nsend.view_device();
  h_nsend = k_nsend.view_host();
}

/* ---------------------------------------------------------------------- */

CommKokkos::~CommKokkos()
{
  if (copymode) return;

  if (!sparta->kokkos->comm_serial) {
    pproc = NULL;
  }
}

/* ----------------------------------------------------------------------
   migrate particles to new procs after particle move
   return particle nlocal after compression,
     so Update can iterate on particle move
------------------------------------------------------------------------- */

int CommKokkos::migrate_particles(int nmigrate, int *plist, const DAT::t_int_1d &d_plist_in)
{
  GridKokkos* grid_kk = (GridKokkos*) grid;
  ParticleKokkos* particle_kk = ((ParticleKokkos*)particle);
  particle_kk->update_class_variables();
  particle_kk_copy.copy(particle_kk);

  const int need_atomics = sparta->kokkos->need_atomics;

  if (sparta->kokkos->comm_serial) {
    particle_kk->sync(Host,ALL_MASK);
    //grid_kk->sync(Host,ALL_MASK);
    int prev_auto_sync = sparta->kokkos->auto_sync;
    sparta->kokkos->auto_sync = 1;

    int ncompress = Comm::migrate_particles(nmigrate,plist);

    particle_kk->sync(Device,ALL_MASK);
    //grid_kk->sync(Device,ALL_MASK);
    sparta->kokkos->auto_sync = prev_auto_sync;

    return ncompress;
  }

  // int i,j;

  d_plist = d_plist_in;

  int ncustom = particle->ncustom;
  nbytes_particle = sizeof(Particle::OnePart);
  int nbytes_custom = particle->sizeof_custom();
  nbytes_total = nbytes_particle + nbytes_custom;

  // Kokkos
  // memory access: cells, particles
  // local views: pproc, sbuf, rbuf
  // functions: particle->compress_migrate, grow,
  //  iparticle->augment_data_uniform, iparticle->create_data_uniform,
  //  iparticle->exchange_uniform
  // parallel_for: loop over nmigrate to pack buffer,
  //  loop over nrecv to unpack buffer
  // atomic variables: ?

  // grow pproc and sbuf if necessary, with headroom and never shrinking,
  //   so device (and pinned host) allocations do not happen every step
  // sbuf no longer needs its exact size when not GPU-aware:
  //   IrregularKokkos::exchange_uniform() only copies the bytes it sends

  if (nmigrate > maxpproc) {
    maxpproc = static_cast<int>(BUFFACTOR*nmigrate);
    d_pproc = DAT::t_int_1d(Kokkos::view_alloc("comm:pproc",Kokkos::WithoutInitializing),maxpproc);
    h_pproc = t_int_1d_pinned(Kokkos::view_alloc("comm:pproc:host",Kokkos::WithoutInitializing),maxpproc);
    pproc = h_pproc.data();
  }

  bigint maxsendbuf = (bigint)nmigrate*nbytes_total;
  if (maxsendbuf > bigint(d_sbuf.extent(0)))
    d_sbuf = DAT::t_char_1d(Kokkos::view_alloc("comm:sbuf",Kokkos::WithoutInitializing),
                            (bigint)(BUFFACTOR*maxsendbuf));

  // fill proclist with procs to send to
  // pack sbuf with particles to migrate
  // if flag == PDISCARD, particle is deleted but not sent
  // change icell of migrated particle to owning cell on receiving proc
  // nsend = particles that actually migrate
  // if no custom attributes, pack particles directly via memcpy()
  // else pack_custom() performs packing into sbuf

  int nsend = 0;
  //int offset = 0;

  h_nsend() = 0;
  k_nsend.modify_host();
  k_nsend.sync_device();

  particle_kk->sync(Device,PARTICLE_MASK);
  if (ncustom) particle_kk->sync(Device,CUSTOM_MASK);
  grid_kk->sync(Device,CELL_MASK);

  d_cells = grid_kk->k_cells.view_device();
  d_particles = particle_kk->k_particles.view_device();

  copymode = 1;
  if (!ncustom) {

    if (need_atomics)
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCommMigrateParticles<1,0> >(0,nmigrate),*this);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCommMigrateParticles<0,0> >(0,nmigrate),*this);

  } else {

    if (need_atomics)
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCommMigrateParticles<1,1> >(0,nmigrate),*this);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCommMigrateParticles<0,1> >(0,nmigrate),*this);

  }
  copymode = 0;

  particle_kk->modify(Device,PARTICLE_MASK);
  d_particles = t_particle_1d(); // destroy reference to reduce memory use

  // sync_host() of nsend fences, so the pack kernel is done
  // then copy only the nsend entries of pproc that were filled

  k_nsend.modify_device();
  k_nsend.sync_host();
  nsend = h_nsend();

  if (nsend) {
    const std::pair<int,int> range(0,nsend);
    Kokkos::deep_copy(Kokkos::subview(h_pproc,range),Kokkos::subview(d_pproc,range));
  }

  // compress my list of particles
  // EXACT build has no ParticleKokkos::compress_migrate() override,
  //   so compress on the host, matching Comm::migrate_particles()

#ifdef SPARTA_KOKKOS_EXACT
  particle_kk->sync(Host,PARTICLE_MASK);
  if (ncustom) particle_kk->sync(Host,CUSTOM_MASK);

  int ascending = 1;
  for (int i = 1; i < nmigrate; i++)
    if (plist[i] <= plist[i-1]) {
      ascending = 0;
      break;
    }

  if (ascending) particle->compress_migrate(nmigrate,plist);
  else particle->compress_reactions(nmigrate,plist);

  particle_kk->modify(Host,PARTICLE_MASK);
  if (ncustom) particle_kk->modify(Host,CUSTOM_MASK);
#else
  particle->compress_migrate(nmigrate,plist);
#endif
  int ncompress = particle->nlocal;

  // create or augment irregular communication plan
  // nrecv = # of incoming particles

  IrregularKokkos* iparticle_kk = (IrregularKokkos*) iparticle;

  int nrecv;
  if (neighflag) {
    if (sparta->kokkos->comm_batch_flag)
      nrecv = iparticle_kk->augment_data_uniform_batch(nsend,pproc);
    else
      nrecv = iparticle_kk->augment_data_uniform(nsend,pproc);
  }
  else
    nrecv = iparticle_kk->create_data_uniform(nsend,pproc,commsortflag);

  // extend particle list if necessary

  particle->grow(nrecv);

  // perform irregular communication
  // if no custom attributes, append recv particles directly to particle list
  // else receive into rbuf, unpack particles one by one via unpack_custom()

  particle_kk->sync(Device,PARTICLE_MASK);
  if (ncustom) particle_kk->sync(Device,CUSTOM_MASK);
  d_particles = particle_kk->k_particles.view_device();

  if (!ncustom) {

    // received particles land straight in the particle list, past nlocal,
    //   either from MPI directly (GPU-aware) or by one copy from the host
    //   buffer exchange_uniform() received into, so no rbuf and no unpack
    //   kernel are needed in either case.
    // the view handed to exchange_uniform() must describe the same buffer
    //   the receives target, since self datums (never produced here: a
    //   migrating particle's new cell is on another proc) and the host copy
    //   are written through it

    char *d_pdest = (char *) (d_particles.data()+particle->nlocal);
    DAT::t_char_1d d_pbuf(d_pdest,(size_t)nrecv*nbytes_total);
    iparticle_kk->exchange_uniform(d_sbuf,nbytes_total,d_pdest,d_pbuf);
  } else {

    // custom attributes: receive into rbuf, then unpack particle by particle
    // grow rbuf with headroom, never shrink it
    // exchange_uniform() only copies the received bytes, so rbuf may be larger

    bigint maxrecvbuf = (bigint)nrecv*nbytes_total;
    if (maxrecvbuf > bigint(d_rbuf.extent(0)))
      d_rbuf = DAT::t_char_1d(Kokkos::view_alloc("comm:rbuf",Kokkos::WithoutInitializing),
                              (bigint)(BUFFACTOR*maxrecvbuf));

    nlocal = particle->nlocal;
    iparticle_kk->exchange_uniform(d_sbuf,nbytes_total,(char *)d_rbuf.data(),d_rbuf);

    // no fence after the unpack: later kernels are ordered after it,
    //   host access to particles goes through a DualView sync, which fences,
    //   and d_rbuf is only reallocated after the next call's fencing sync

    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCommMigrateUnpackParticles<1> >(0,nrecv),*this);
    copymode = 0;

  }

  particle_kk->modify(Device,PARTICLE_MASK);
  if (ncustom) particle_kk->modify(Device,CUSTOM_MASK);
  d_particles = t_particle_1d(); // destroy reference to reduce memory use
  d_plist = {};

  particle->nlocal += nrecv;
  ncomm += nsend;
  return ncompress;
}

template<int NEED_ATOMICS, int HAVE_CUSTOM>
KOKKOS_INLINE_FUNCTION
void CommKokkos::operator()(TagCommMigrateParticles<NEED_ATOMICS, HAVE_CUSTOM>, const int &i) const {
  const int j = d_plist[i];
  if (d_particles[j].flag == PDISCARD) return;
  int nsend;
  if (NEED_ATOMICS)
    nsend = Kokkos::atomic_fetch_add(&d_nsend(),1);
  else {
    nsend = d_nsend();
    d_nsend()++;
  }
  d_pproc[nsend] = d_cells[d_particles[j].icell].proc;
  d_particles[j].icell = d_cells[d_particles[j].icell].ilocal;
  const bigint offset = (bigint)nsend*nbytes_total;
  memcpy(&d_sbuf[offset],&d_particles[j],nbytes_particle);
  if (HAVE_CUSTOM)
    particle_kk_copy.obj.pack_custom_kokkos(j,(char*)(d_sbuf.data()+offset+nbytes_particle));
}

template<int HAVE_CUSTOM>
KOKKOS_INLINE_FUNCTION
void CommKokkos::operator()(TagCommMigrateUnpackParticles<HAVE_CUSTOM>, const int &irecv) const {
  const int i = nlocal + irecv;
  const bigint offset = (bigint)irecv*nbytes_total;
  memcpy(&d_particles[i],&d_rbuf[offset],nbytes_particle);
  if (HAVE_CUSTOM)
    particle_kk_copy.obj.unpack_custom_kokkos((char*)(d_rbuf.data()+offset+nbytes_particle),i);
}

/* ----------------------------------------------------------------------
   migrate grid cells with their particles to new procs
   called from BalanceGrid and FixBalance
------------------------------------------------------------------------- */

void CommKokkos::migrate_cells(int nmigrate)
{
  CollideVSSKokkos* collide_kk = (CollideVSSKokkos*) collide;
  if (collide)
    collide_kk->sync(Host,ALL_MASK);

  Comm::migrate_cells(nmigrate);

  if (collide)
    collide_kk->modified(Host,ALL_MASK);
}
