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

CollideStyle(vss,CollideVSS)

#else

#ifndef SPARTA_COLLIDE_VSS_H
#define SPARTA_COLLIDE_VSS_H

#include "collide.h"
#include "particle.h"

namespace SPARTA_NS {

class CollideVSS : public Collide {
 public:
  CollideVSS(class SPARTA *, int, char **);
  virtual ~CollideVSS();
  virtual void init();

  double vremax_init(int, int);
  virtual double attempt_collision(int, int, double);
  double attempt_collision(int, int, int, double);
  virtual int test_collision(int, int, int, Particle::OnePart *,
                             Particle::OnePart *);
  virtual void setup_collision(Particle::OnePart *, Particle::OnePart *);
  virtual int perform_collision(Particle::OnePart *&, Particle::OnePart *&,
                                Particle::OnePart *&);
  double extract(int, int, const char *);

  struct State {      // two-particle state
    double vr2;
    double vr;
    double imass,jmass;
    double ave_rotdof;
    double ave_vibdof;
    double ave_dof;
    double etrans;
    double erot;
    double evib;
    double eexchange;
    double eint;
    double etotal;
    double ucmf;
    double vcmf;
    double wcmf;
  };

  struct Params {             // VSS model parameters
    double diam;
    double omega;
    double tref;
    double alpha;
    double rotc1;
    double rotc2;
    double rotc3;
    double vibc1;
    double vibc2;
    double mr;
  };

 protected:
  CollideVSS(class SPARTA *, int, char **, int);   // ctor for derived classes

  int relaxflag,eng_exchange;
  double vr_indice;
  double **prefactor; // static portion of collision attempt frequency

  struct State precoln;       // state before collision
  struct State postcoln;      // state after collision

  Params **params;             // VSS params for each species
  int nparams;                // # of per-species params read in

  virtual void SCATTER_TwoBodyScattering(Particle::OnePart *,
                                         Particle::OnePart *);

  // hooks so a child style can replace the deflection angle model
  // scatter_alpha returns the VSS alpha to use for this pair
  // scatter_cosX returns 1 and sets cosX if the child sampled it directly

  virtual double scatter_alpha(int isp, int jsp) {return params[isp][jsp].alpha;}
  virtual int scatter_cosX(int, int, double &) {return 0;}

  // Larsen-Borgnakke detailed balance for a non-VHS cross section.
  //
  // for a colliding pair in equilibrium the joint density of the relative
  // translational energy and one internal mode of zeta degrees of freedom is
  //
  //   f(Et,Ei) ~ sigma(Et) Et Ei^(zeta/2-1) exp(-(Et+Ei)/kT)
  //
  // the Et factor being sigma(Et)*g times the Et^1/2 density of states.  an
  // exchange holds Ec = Et + Ei fixed, so it leaves that density stationary
  // exactly when the post-collision Et is drawn from
  //
  //   P(Et|Ec) ~ sigma(Et) Et (Ec-Et)^(zeta/2-1)
  //
  // for the VHS law sigma ~ Et^(1/2-omega) this is the Et^(3/2-omega)
  // weight used below, which is why the samplers carry that exponent.  for
  // an arbitrary sigma the same draws are made and then accepted with
  // probability sigma(Et)/sigma_VHS(Et), normalized by its maximum over the
  // energies reachable at this Ec, which turns the VHS sampler into an exact
  // sampler for P(Et|Ec) whatever the cross section is.
  //
  // lb_weight returns that acceptance probability, or a negative value when
  // the pair needs no correction.  lbflag is 0 unless some pair does, so the
  // vss style never evaluates it and its random number stream is unchanged.

  int lbflag;
  int lbcapflag;                  // 1 once the retry-cap warning has fired
  virtual double lb_weight(int, int, double, double) {return -1.0;}
  void lb_capcheck(int);

  void EEXCHANGE_NonReactingEDisposal(Particle::OnePart *,
                                      Particle::OnePart *);
  void SCATTER_ThreeBodyScattering(Particle::OnePart *,
                                   Particle::OnePart *,
                                   Particle::OnePart *);
  void EEXCHANGE_ReactingEDisposal(Particle::OnePart *,
                                   Particle::OnePart *,
                                   Particle::OnePart *);

  double sample_bl(RanKnuth *, double, double);
  double eff_vib_dof(double, double);
  double vib_pool_temp(double, int, double *, double);
  double rotrel (int, double);
  double vibrel (int, double);

  void parse_vss_args(int, int, char **);
  void allocate_params();
  virtual void read_param_file(char *);
  virtual int skip_param_line(int, char **) {return 0;}
  int wordparse(int, char *, char **);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: Species %s did not appear in VSS parameter file

Self-explanatory.

E: VSS parameters do not match current species

Species cannot be added after VSS colision file is read.

E: Cannot open VSS parameter file %s

Self-explanatory.

E: Incorrect line format in VSS parameter file

Number of parameters in a line read from file is not valid.

E: Request for unknown parameter from collide

VSS model does not have the parameter being requested.

*/
