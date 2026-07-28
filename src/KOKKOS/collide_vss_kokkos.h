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
#include "interp_table_kokkos.h"
#include "collide_vss_kokkos.h"
#include "particle_kokkos.h"
#include "grid_kokkos.h"
#include "react_tce_kokkos.h"
#include "react_table_kokkos.h"
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

  // public alongside d_params_const, and for the same reason: compute
  //   lambda/grid reads the cross section of a pair from the collide style,
  //   and takes it from the table when the pair has one

  DAT::tdual_int_2d k_sigidx;
  DAT::t_int_2d d_sigidx;

  // effective total cross section vs temperature, mirroring CollideTable's
  //   sigeff.  nsigeff_kk is 0 for the vss style, and a caller which sees
  //   that keeps whatever VHS arithmetic it used before

  int nsigeff_kk;               // # of rows in d_sigeff, 0 for vss
  int ntemp_kk;                 // # of temperature points per row
  double sigeff_tlo_kk,sigeff_tinvdelta_kk;
  DAT::tdual_float_2d k_sigeff;
  DAT::t_float_2d d_sigeff;

 protected:

  // device copies of the tabulated cross sections, filled by
  // CollideTableKokkos.  the tables are held in one InterpTableKokkos in
  // the order sigma, alpha, scatter, and ntab_kk is 0 for the vss style so
  // none of this is read.

  int ntab_kk;                  // # of cross section tables, 0 for vss
  int nalphatab_kk;             // # of alpha tables, 0 for vss
  int nscattertab_kk;           // # of scatter tables, 0 for vss
  int nlbpair_kk;               // # of pairs needing the LB correction
  InterpTableKokkos tabdev;
  DAT::tdual_int_2d k_alphaidx;
  DAT::t_int_2d d_alphaidx;
  DAT::tdual_int_2d k_scatteridx;
  DAT::t_int_2d d_scatteridx;

  // Larsen-Borgnakke acceptance normalization, mirroring CollideTable

  DAT::tdual_int_2d k_lbidx;
  DAT::t_int_2d d_lbidx;
  DAT::tdual_float_2d k_lbratio;
  DAT::t_float_2d d_lbratio;
  DAT::tdual_float_2d k_lbmax;
  DAT::t_float_2d d_lbmax;
  double lblo_kk,lbinvdelta_kk;
  int nlbgrid_kk;

  // set on device when a Larsen-Borgnakke acceptance loop runs out of
  //   retries, so the host can warn exactly as CollideVSS::lb_capcheck does

  DAT::t_int_scalar d_lb_cap;
  HAT::t_int_scalar h_lb_cap;

  // set on device when a collision energy falls outside the acceptance
  //   normalization grid, the warning CollideTable::lb_weight issues inline

  DAT::t_int_scalar d_lb_range;
  HAT::t_int_scalar h_lb_range;

  // the same expression as InterpTable::evaluate()

  // cos(chi) sampled from a tabulated differential cross section
  //   the random number is drawn inside, and only when the pair has a
  //   table, so the stream matches CollideTable::scatter_cosX() exactly

  KOKKOS_INLINE_FUNCTION
  int scatter_cosX_kokkos(int isp, int jsp, double vr2, double &cosX,
                          rand_type &rand_gen) const {
    if (!nscattertab_kk) return 0;
    const int m = d_scatteridx(isp,jsp);
    if (m < 0) return 0;
    cosX = tabdev.interpolate_row(m,vr2,rand_gen.drand());
    if (cosX > 1.0) cosX = 1.0;
    else if (cosX < -1.0) cosX = -1.0;
    return 1;
  }

  // VSS alpha for this pair, from a table when one is defined

  KOKKOS_INLINE_FUNCTION
  double scatter_alpha_kokkos(int isp, int jsp, double vr2) const {
    if (!nalphatab_kk) return d_params(isp,jsp).alpha;
    const int m = d_alphaidx(isp,jsp);
    if (m < 0) return d_params(isp,jsp).alpha;
    return tabdev.evaluate(m,vr2);
  }

  // does this pair carry a tabulated cross section?
  //   the same test as CollideTable::tabulated_pair()

  KOKKOS_INLINE_FUNCTION
  int tabulated_pair_kokkos(int isp, int jsp) const {
    if (!nsigeff_kk) return 0;
    return d_sigidx(isp,jsp) >= 0;
  }

  // effective total cross section at temperature TEMP, the same expression
  //   as CollideTable::sigma_eff(); only call it when the test above passes

  KOKKOS_INLINE_FUNCTION
  double sigma_eff_kokkos(int isp, int jsp, double temp) const {
    const int m = d_sigidx(isp,jsp);
    if (temp <= 0.0) temp = d_params(isp,jsp).tref;

    double f = (log(temp) - sigeff_tlo_kk) * sigeff_tinvdelta_kk;
    if (f <= 0.0) return d_sigeff(m,0);
    if (f >= ntemp_kk-1) return d_sigeff(m,ntemp_kk-1);
    int k = (int) f;
    return d_sigeff(m,k) + (f-k)*(d_sigeff(m,k+1)-d_sigeff(m,k));
  }

  // Larsen-Borgnakke acceptance probability, the same expression as
  //   CollideTable::lb_weight(); -1.0 when the pair needs no correction

  KOKKOS_INLINE_FUNCTION
  double lb_weight_kokkos(int isp, int jsp, double etrans, double ec) const {
    if (!nlbpair_kk) return -1.0;
    const int row = d_lbidx(isp,jsp);
    if (row < 0) return -1.0;
    if (etrans <= 0.0) return 0.0;

    double f = (log(etrans/1.602176634e-19) - lblo_kk) * lbinvdelta_kk;
    double g = (log(ec/1.602176634e-19) - lblo_kk) * lbinvdelta_kk;

    // an energy above the grid cannot be bounded by it, so the draw reverts
    //   toward the VSS law.  record it for the one-time warning the host
    //   style issues inline, since device code cannot warn itself

    if (g >= nlbgrid_kk-1) d_lb_range() = 1;

    if (f < 0.0) f = 0.0;
    if (g > nlbgrid_kk-1) g = nlbgrid_kk-1;

    int i = (int) f;
    double r;
    if (i >= nlbgrid_kk-1) r = d_lbratio(row,nlbgrid_kk-1);
    else r = d_lbratio(row,i) + (f-i)*(d_lbratio(row,i+1)-d_lbratio(row,i));

    int j = (int) g;
    if (j < 0) j = 0;
    else if (j > nlbgrid_kk-2) j = nlbgrid_kk-2;
    const double rmax = d_lbmax(row,j+1);

    if (rmax <= 0.0) return -1.0;
    const double w = r / rmax;
    return (w > 1.0) ? 1.0 : w;
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
  KKCopy<ReactTableKokkos> react_table_kk_copy;
  int react_table_defined;   // 1 if the react style is table/kk

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
