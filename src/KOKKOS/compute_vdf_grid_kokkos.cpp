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

#include "math.h"
#include "compute_vdf_grid_kokkos.h"
#include "particle_kokkos.h"
#include "mixture.h"
#include "grid_kokkos.h"
#include "update.h"
#include "memory_kokkos.h"
#include "error.h"
#include "sparta_masks.h"
#include "kokkos.h"

using namespace SPARTA_NS;

// user keywords, must match compute_vdf_grid.cpp

enum{SPEED,VX,VY,VZ,KE,EROT,EVIB};

// out-of-range handling, must match compute_vdf_grid.cpp

enum{IGNORE,CLAMP};

/* ---------------------------------------------------------------------- */

ComputeVDFGridKokkos::ComputeVDFGridKokkos(SPARTA *sparta, int narg, char **arg) :
  ComputeVDFGrid(sparta, narg, arg)
{
  kokkos_flag = 1;

  d_value = DAT::t_int_1d("vdf/grid:value",nvalue);
  d_nbin = DAT::t_int_1d("vdf/grid:nbin",nvalue);
  d_binoffset = DAT::t_int_1d("vdf/grid:binoffset",nvalue);
  d_lo = DAT::t_float_1d("vdf/grid:lo",nvalue);
  d_hi = DAT::t_float_1d("vdf/grid:hi",nvalue);
  d_invdelta = DAT::t_float_1d("vdf/grid:invdelta",nvalue);
}

/* ---------------------------------------------------------------------- */

ComputeVDFGridKokkos::~ComputeVDFGridKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_array_grid,array_grid);
  array_grid = NULL;
}

/* ---------------------------------------------------------------------- */

void ComputeVDFGridKokkos::init()
{
  ComputeVDFGrid::init();

  // mirror the per-value binning parameters to the device

  auto h_value = Kokkos::create_mirror_view(d_value);
  auto h_nbin = Kokkos::create_mirror_view(d_nbin);
  auto h_binoffset = Kokkos::create_mirror_view(d_binoffset);
  auto h_lo = Kokkos::create_mirror_view(d_lo);
  auto h_hi = Kokkos::create_mirror_view(d_hi);
  auto h_invdelta = Kokkos::create_mirror_view(d_invdelta);

  for (int m = 0; m < nvalue; m++) {
    h_value(m) = value[m];
    h_nbin(m) = nbin[m];
    h_binoffset(m) = binoffset[m];
    h_lo(m) = lo[m];
    h_hi(m) = hi[m];
    h_invdelta(m) = invdelta[m];
  }

  Kokkos::deep_copy(d_value,h_value);
  Kokkos::deep_copy(d_nbin,h_nbin);
  Kokkos::deep_copy(d_binoffset,h_binoffset);
  Kokkos::deep_copy(d_lo,h_lo);
  Kokkos::deep_copy(d_hi,h_hi);
  Kokkos::deep_copy(d_invdelta,h_invdelta);
}

/* ---------------------------------------------------------------------- */

void ComputeVDFGridKokkos::compute_per_grid()
{
  if (sparta->kokkos->prewrap) {
    ComputeVDFGrid::compute_per_grid();
  } else {
    compute_per_grid_kokkos();
    k_array_grid.modify_device();
    k_array_grid.sync_host();
  }
}

/* ---------------------------------------------------------------------- */

void ComputeVDFGridKokkos::compute_per_grid_kokkos()
{
  invoked_per_grid = update->ntimestep;

  mvv2e = update->mvv2e;
  useweight = weightflag && cellweightflag;

  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK);
  d_particles = particle_kk->k_particles.view_device();
  d_species = particle_kk->k_species.view_device();
  d_s2g = particle_kk->k_species2group.view_device();

  GridKokkos* grid_kk = (GridKokkos*) grid;
  d_cellcount = grid_kk->d_cellcount;
  d_plist = grid_kk->d_plist;
  grid_kk->sync(Device,CINFO_MASK);
  d_cinfo = grid_kk->k_cinfo.view_device();

  int nlocal = particle->nlocal;

  // zero all accumulators

  Kokkos::deep_copy(d_array_grid,0.0);

  // if particles are sorted by cell, tally per cell with no atomics needed
  // else tally per particle, with a duplicated or atomic scatter view

  need_dup = sparta->kokkos->need_dup<DeviceType>();
  if (particle_kk->sorted_kk && sparta->kokkos->need_atomics &&
      !sparta->kokkos->atomic_reduction)
    need_dup = 0;

  if (need_dup)
    dup_array_grid = Kokkos::Experimental::create_scatter_view<typename Kokkos::Experimental::ScatterSum, typename Kokkos::Experimental::ScatterDuplicated>(d_array_grid);
  else
    ndup_array_grid = Kokkos::Experimental::create_scatter_view<typename Kokkos::Experimental::ScatterSum, typename Kokkos::Experimental::ScatterNonDuplicated>(d_array_grid);

  copymode = 1;
  if (particle_kk->sorted_kk && sparta->kokkos->need_atomics &&
      !sparta->kokkos->atomic_reduction)
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagComputeVDFGrid_compute_per_grid>(0,nglocal),*this);
  else {
    if (sparta->kokkos->need_atomics)
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagComputeVDFGrid_compute_per_grid_atomic<1> >(0,nlocal),*this);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagComputeVDFGrid_compute_per_grid_atomic<0> >(0,nlocal),*this);
  }
  copymode = 0;

  if (need_dup) {
    Kokkos::Experimental::contribute(d_array_grid, dup_array_grid);
    dup_array_grid = {}; // free duplicated memory
  }

  d_particles = t_particle_1d(); // destroy reference to reduce memory use
  d_plist = DAT::t_int_2d();     // destroy reference to reduce memory use
}

/* ----------------------------------------------------------------------
   value of the quantity binned for value m of particle i
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double ComputeVDFGridKokkos::sample_of(const int i, const int m,
                                       const double mass) const
{
  double *v = d_particles[i].v;

  switch (d_value(m)) {
  case SPEED:
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  case VX:
    return v[0];
  case VY:
    return v[1];
  case VZ:
    return v[2];
  case KE:
    return 0.5*mvv2e*mass * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  case EROT:
    return d_particles[i].erot;
  case EVIB:
    return d_particles[i].evib;
  }
  return 0.0;
}

/* ----------------------------------------------------------------------
   bin index for a sample of value m, or -1 if out of range and discarded
   must match the host version in compute_vdf_grid.cpp exactly
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
int ComputeVDFGridKokkos::bin_of(const double sample, const int m) const
{
  const int nb = d_nbin(m);

  if (sample < d_lo(m) || sample > d_hi(m)) {
    if (oobstyle == IGNORE) return -1;
    return (sample < d_lo(m)) ? 0 : nb-1;
  }

  int ibin = static_cast<int> ((sample - d_lo(m)) * d_invdelta(m));
  if (ibin >= nb) ibin = nb - 1;
  return ibin;
}

/* ---------------------------------------------------------------------- */

template<int NEED_ATOMICS>
KOKKOS_INLINE_FUNCTION
void ComputeVDFGridKokkos::
operator()(TagComputeVDFGrid_compute_per_grid_atomic<NEED_ATOMICS>,
           const int &i) const
{
  // the tally array is duplicated for OpenMP, atomic for GPUs, neither for Serial

  auto v_array_grid = ScatterViewHelper<typename NeedDup<NEED_ATOMICS,DeviceType>::value,decltype(dup_array_grid),decltype(ndup_array_grid)>::get(dup_array_grid,ndup_array_grid);
  auto a_array_grid = v_array_grid.template access<typename AtomicDup<NEED_ATOMICS,DeviceType>::value>();

  const int ispecies = d_particles[i].ispecies;
  const int igroup = d_s2g(imix,ispecies);
  if (igroup < 0) return;

  const int icell = d_particles[i].icell;
  if (!(d_cinfo[icell].mask & groupbit)) return;

  const double mass = needmass ? d_species[ispecies].mass : 0.0;

  // use the cell weight, not Particle::OnePart::weight, which is only
  //   scratch state maintained by Particle::pre_weight() during a move

  const double wt = useweight ? d_cinfo[icell].weight : 1.0;

  const int kbase = igroup*nbintotal;

  for (int m = 0; m < nvalue; m++) {
    const int ibin = bin_of(sample_of(i,m,mass),m);
    if (ibin < 0) continue;
    a_array_grid(icell,kbase + d_binoffset(m) + ibin) += wt;
  }
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void ComputeVDFGridKokkos::operator()(TagComputeVDFGrid_compute_per_grid,
                                      const int &icell) const
{
  if (!(d_cinfo[icell].mask & groupbit)) return;

  const double wt = useweight ? d_cinfo[icell].weight : 1.0;
  const int np = d_cellcount[icell];

  for (int n = 0; n < np; n++) {
    const int i = d_plist(icell,n);

    const int ispecies = d_particles[i].ispecies;
    const int igroup = d_s2g(imix,ispecies);
    if (igroup < 0) continue;

    const double mass = needmass ? d_species[ispecies].mass : 0.0;
    const int kbase = igroup*nbintotal;

    for (int m = 0; m < nvalue; m++) {
      const int ibin = bin_of(sample_of(i,m,mass),m);
      if (ibin < 0) continue;
      d_array_grid(icell,kbase + d_binoffset(m) + ibin) += wt;
    }
  }
}

/* ----------------------------------------------------------------------
   reallocate array if nglocal has changed
   called by init() and whenever grid changes
------------------------------------------------------------------------- */

void ComputeVDFGridKokkos::reallocate()
{
  if (grid->nlocal == nglocal) return;

  memoryKK->destroy_kokkos(k_array_grid,array_grid);
  nglocal = grid->nlocal;
  memoryKK->create_kokkos(k_array_grid,array_grid,nglocal,ntotal,
                          "vdf/grid:array_grid");
  d_array_grid = k_array_grid.view_device();
}
