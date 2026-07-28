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

#ifdef COLLIDE_CLASS

CollideStyle(vss/kk,CollideVSSKokkos)

#else

#ifndef SPARTA_COLLIDE_VSS_KOKKOS_H
#define SPARTA_COLLIDE_VSS_KOKKOS_H

#include "collide_table.h"
#include "collide_vss_kokkos.h"
#include "particle_kokkos.h"
#include "grid_kokkos.h"
#include "react_tce_kokkos.h"
#include "kokkos_type.h"
#include "Kokkos_Random.hpp"
#include "rand_pool_wrap.h"
#include "kokkos_copy.h"

namespace SPARTA_NS {

struct s_COLLIDE_REDUCE {
  int nattempt_one,ncollide_one,nreact_one;
  KOKKOS_INLINE_FUNCTION
  s_COLLIDE_REDUCE() {
    nattempt_one = 0;
    ncollide_one = 0;
    nreact_one = 0;
  }

  KOKKOS_INLINE_FUNCTION
  void operator+=(const s_COLLIDE_REDUCE &rhs) {
    nattempt_one += rhs.nattempt_one;
    ncollide_one += rhs.ncollide_one;
    nreact_one += rhs.nreact_one;
  }
};
typedef struct s_COLLIDE_REDUCE COLLIDE_REDUCE;

struct TagCollideResetVremax{};
struct TagCollideZeroNN{};

template < int NEARCP, int GASTALLY, int ATOMIC_REDUCTION >
struct TagCollideCollisionsOne{};

template < int GASTALLY, int ATOMIC_REDUCTION >
struct TagCollideCollisionsOneAmbipolar{};

// derives from CollideTable rather than CollideVSS: the collision kernels
// are launched as parallel_for(policy,*this), which slices the object to this
// class, so device code cannot dispatch virtually to a style derived from it.
// the table state has to be reachable from here for collide table/kk to work.
// with no tables built every CollideTable method falls back to the analytic
// VSS form, so collide vss/kk is unchanged.

class CollideVSSKokkos : public CollideTable {
 public:
  typedef COLLIDE_REDUCE value_type;

  CollideVSSKokkos(class SPARTA *, int, char **);
  CollideVSSKokkos(class SPARTA *, int, char **, int);   // for derived styles
  ~CollideVSSKokkos();
  void init();
  void collisions();
  void sync(ExecutionSpace, unsigned int);
  void modified(ExecutionSpace, unsigned int);

#ifndef SPARTA_KOKKOS_EXACT
  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;

  //Kokkos::Random_XorShift1024_Pool<DeviceType> rand_pool;
  //typedef typename Kokkos::Random_XorShift1024_Pool<DeviceType>::generator_type rand_type;
#else
  RandPoolWrap rand_pool;
  typedef RandWrap rand_type;
#endif

  KOKKOS_INLINE_FUNCTION
  double attempt_collision_kokkos(int, int, double, rand_type &) const;
  KOKKOS_INLINE_FUNCTION
  double poisson_kokkos(double, rand_type &) const;
  KOKKOS_INLINE_FUNCTION
  int test_collision_kokkos(int, int, int, Particle::OnePart *, Particle::OnePart *, struct State &, rand_type &) const;
  KOKKOS_INLINE_FUNCTION
  void setup_collision_kokkos(Particle::OnePart *, Particle::OnePart *, struct State &, struct State &) const;
  KOKKOS_INLINE_FUNCTION
  int perform_collision_kokkos(Particle::OnePart *&, Particle::OnePart *&,
                        Particle::OnePart *&, struct State &, struct State &, rand_type &,
                        Particle::OnePart *&, int &, double &,
                        int &) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagCollideResetVremax, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagCollideZeroNN, const int&) const;

  template < int NEARCP, int GASTALLY, int ATOMIC_REDUCTION >
  KOKKOS_INLINE_FUNCTION
  void operator()(TagCollideCollisionsOne< NEARCP, GASTALLY, ATOMIC_REDUCTION >, const int&) const;

  template < int NEARCP, int GASTALLY, int ATOMIC_REDUCTION >
  KOKKOS_INLINE_FUNCTION
  void operator()(TagCollideCollisionsOne< NEARCP, GASTALLY, ATOMIC_REDUCTION >, const int&, COLLIDE_REDUCE&) const;

  template < int GASTALLY, int ATOMIC_REDUCTION >
  KOKKOS_INLINE_FUNCTION
  void operator()(TagCollideCollisionsOneAmbipolar< GASTALLY, ATOMIC_REDUCTION >, const int&) const;

  template < int GASTALLY, int ATOMIC_REDUCTION >
  KOKKOS_INLINE_FUNCTION
  void operator()(TagCollideCollisionsOneAmbipolar< GASTALLY, ATOMIC_REDUCTION >, const int&, COLLIDE_REDUCE&) const;

  typedef Kokkos::
    DualView<Params**, Kokkos::LayoutRight, DeviceType> tdual_params_2d;
  typedef tdual_params_2d::t_dev t_params_2d;
  typedef tdual_params_2d::t_dev_const t_params_2d_const;
  t_params_2d_const d_params_const;

 protected:

  // device copies of the tabulated cross sections, filled by
  // CollideTableKokkos.  ntab_kk is 0 for the vss style and none of this is
  // read.  TabMeta mirrors the private state of InterpTable which
  // InterpTable::evaluate() needs: the bit-indexed bin lookup and the power
  // law extrapolation on each end.

  struct TabMeta {
    double xlo,xhi,alo,plo,ahi,phi;
    int64_t offset;             // bin index of the first bin
    int64_t coffset;            // start of this table inside d_tabcoeff
    int shift;                  // right shift mapping the x bits to a bin
    int nbins,ncoeff,tabstyle;
  };

  typedef Kokkos::DualView<TabMeta*,DeviceType::array_layout,DeviceType>
    tdual_tabmeta_1d;
  typedef tdual_tabmeta_1d::t_dev t_tabmeta_1d;

  int ntab_kk;                  // # of cross section tables, 0 for vss
  int nalphatab_kk;             // # of alpha tables, 0 for vss
  tdual_tabmeta_1d k_tabmeta;
  t_tabmeta_1d d_tabmeta;
  DAT::tdual_float_1d k_tabcoeff;
  DAT::t_float_1d d_tabcoeff;
  DAT::tdual_int_2d k_sigidx;
  DAT::t_int_2d d_sigidx;
  DAT::tdual_int_2d k_alphaidx;
  DAT::t_int_2d d_alphaidx;

  // the same expression as InterpTable::evaluate()

  KOKKOS_INLINE_FUNCTION
  double tab_evaluate(int m, double x) const {
    const TabMeta &t = d_tabmeta(m);
    if (x <= t.xlo) return t.alo * pow(x,t.plo);
    if (x >= t.xhi) return t.ahi * pow(x,t.phi);

    union { double d; uint64_t u; } v;
    v.d = x;
    int64_t k = (int64_t) (v.u >> t.shift) - t.offset;
    if (k < 0) k = 0;
    else if (k > t.nbins-1) k = t.nbins-1;

    const int64_t b = t.coffset + t.ncoeff*k;
    if (t.tabstyle == 1) return d_tabcoeff(b) + x*d_tabcoeff(b+1);  // linear
    if (t.tabstyle == 0) return d_tabcoeff(b);                      // lookup
    const double u = x - d_tabcoeff(b);                             // spline
    return d_tabcoeff(b+1) +
      u*(d_tabcoeff(b+2) + u*(d_tabcoeff(b+3) + u*d_tabcoeff(b+4)));
  }

  // VSS alpha for this pair, from a table when one is defined

  KOKKOS_INLINE_FUNCTION
  double scatter_alpha_kokkos(int isp, int jsp, double vr2) const {
    if (!nalphatab_kk) return d_params(isp,jsp).alpha;
    const int m = d_alphaidx(isp,jsp);
    if (m < 0) return d_params(isp,jsp).alpha;
    return tab_evaluate(m,vr2);
  }

 private:
  KOKKOS_INLINE_FUNCTION
  void ambi_reset_kokkos(int, int, int, int,
                         Particle::OnePart *, Particle::OnePart *,
                         Particle::OnePart *, const DAT::t_int_1d &) const;
  void reset_vremax();
  int pack_grid_one(int, char *, int);
  int unpack_grid_one(int, char *);
  void copy_grid_one(int, int);
  void reset_grid_count(int);
  void add_grid_one();
  void adapt_grid();
  void grow_percell(int);

  KKCopy<GridKokkos> grid_kk_copy;
  KKCopy<ReactTCEKokkos> react_kk_copy;

  t_particle_1d d_particles;
  t_species_1d_const d_species;
  DAT::t_int_2d d_plist;

  DAT::t_int_1d d_ewhich;
  tdual_struct_tdual_int_1d_1d k_eivec;
  tdual_struct_tdual_int_2d_1d k_eiarray;
  tdual_struct_tdual_float_2d_1d k_edarray;
  DAT::t_int_1d d_ionambi;
  DAT::t_int_1d d_ions;
  DAT::t_float_2d_lr d_velambi;
  t_particle_2d d_elist;

  DAT::tdual_float_2d k_vremax_initial;
  DAT::t_float_2d d_vremax_initial;
  DAT::tdual_float_3d k_vremax;
  DAT::t_float_3d d_vremax;
  DAT::tdual_float_3d k_remain;
  DAT::t_float_3d d_remain;

  typedef Kokkos::DualView<int[11], DeviceType::array_layout, DeviceType> tdual_int_11;
  typedef tdual_int_11::t_dev t_int_11;
  typedef tdual_int_11::t_host t_host_int_11;
  t_int_11 d_scalars;
  t_host_int_11 h_scalars;

  DAT::t_int_scalar d_nattempt_one;
  HAT::t_int_scalar h_nattempt_one;

  DAT::t_int_scalar d_ncollide_one;
  HAT::t_int_scalar h_ncollide_one;

  DAT::t_int_scalar d_nreact_one;
  HAT::t_int_scalar h_nreact_one;

  DAT::t_int_scalar d_error_flag;
  HAT::t_int_scalar h_error_flag;

  DAT::t_int_scalar d_retry;
  HAT::t_int_scalar h_retry;

  DAT::t_int_scalar d_maxdelete;
  HAT::t_int_scalar h_maxdelete;

  DAT::t_int_scalar d_maxcellcount;
  HAT::t_int_scalar h_maxcellcount;

  DAT::t_int_scalar d_part_grow;
  HAT::t_int_scalar h_part_grow;

  DAT::t_int_scalar d_ndelete;
  HAT::t_int_scalar h_ndelete;

  DAT::t_int_scalar d_nlocal;
  HAT::t_int_scalar h_nlocal;

  DAT::t_int_scalar d_maxelectron;
  HAT::t_int_scalar h_maxelectron;

  DAT::tdual_int_1d k_dellist;
  DAT::t_int_1d d_dellist;

  DAT::t_float_2d d_recomb_ijflag;

  DAT::t_int_2d d_nn_last_partner;

  template < int NEARCP, int GASTALLY > void collisions_one(COLLIDE_REDUCE&);
  template < int GASTALLY > void collisions_one_ambipolar(COLLIDE_REDUCE&);

  // VSS specific

  DAT::tdual_float_2d k_prefactor;
  DAT::t_float_2d d_prefactor;

  tdual_params_2d k_params;
  t_params_2d d_params;

  double dt,fnum,boltz;
  int maxcellcount,react_defined;

  KOKKOS_INLINE_FUNCTION
  void SCATTER_TwoBodyScattering(Particle::OnePart *,
                                 Particle::OnePart *,
                                 struct State &, struct State &, rand_type &) const;
  KOKKOS_INLINE_FUNCTION
  void EEXCHANGE_NonReactingEDisposal(Particle::OnePart *,
                                      Particle::OnePart *,
                                      struct State &, struct State &, rand_type &) const;

  KOKKOS_INLINE_FUNCTION
  void SCATTER_ThreeBodyScattering(Particle::OnePart *,
                                   Particle::OnePart *,
                                   Particle::OnePart *,
                                   struct State &, struct State &, rand_type &) const;
  KOKKOS_INLINE_FUNCTION
  void EEXCHANGE_ReactingEDisposal(Particle::OnePart *,
                                   Particle::OnePart *,
                                   Particle::OnePart *,
                                   struct State &, struct State &, rand_type &) const;

  KOKKOS_INLINE_FUNCTION
  double sample_bl(rand_type &, double, double) const;
  KOKKOS_INLINE_FUNCTION
  double eff_vib_dof(double, double) const;
  KOKKOS_INLINE_FUNCTION
  double vib_pool_temp(double, int, double *, double) const;
  KOKKOS_INLINE_FUNCTION
  double rotrel (int, double) const;
  KOKKOS_INLINE_FUNCTION
  double vibrel (int, double) const;

  KOKKOS_INLINE_FUNCTION
  int set_nn(int, int) const;
  KOKKOS_INLINE_FUNCTION
  int find_nn(rand_type &, int, int, int) const;

  void backup();
  void restore();

  t_particle_1d d_particles_backup;
  DAT::t_int_2d d_plist_backup;
  DAT::t_float_3d d_vremax_backup;
  DAT::t_float_3d d_remain_backup;
  DAT::t_int_2d d_nn_last_partner_backup;
  DAT::t_int_1d d_ionambi_backup;
  DAT::t_float_2d_lr d_velambi_backup;
  RanKnuth* random_backup;
};

}

#endif
#endif

/* ERROR/WARNING messages:

*/
