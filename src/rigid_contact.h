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

#ifndef SPARTA_RIGID_CONTACT_H
#define SPARTA_RIGID_CONTACT_H

#include "pointers.h"

namespace SPARTA_NS {

class RigidContact : protected Pointers {
 public:
  enum{LINEAR,HERTZ};             // force laws

  RigidContact(class SPARTA *, class FixRigid *,
               int, double, double, double, int);
  ~RigidContact();
  void setup();                   // per run: bin the static surfs
  void compute(double **, double **);  // push-off force/torque on all bodies
  double memory_usage();

 private:
  class FixRigid *fix;
  int dim,axiflag;

  int style;              // LINEAR or HERTZ force law
  double kpush;           // spring constant
  double cutoff;          // distance below which push-off is applied
  double gamma;           // dashpot damping coefficient, 0 = elastic
  int boundflag;          // 1 to also push off non-periodic boundaries

  // bins over static surfs for candidate pruning
  // built once per run, static surfs never move during a run

  int nbin[3];            // # of bins in each dim
  double binlo[3];        // bin grid origin
  double bininv[3];       // inverse bin edge lengths
  int *binstart;          // CSR offsets into binlist per bin
  int *binlist;           // static surf indices, binned by surf bbox
  int *stamp;             // per-surf visit stamp to dedup multi-bin surfs
  int stampcur;

  // work buffers for the distributed-surf merge

  double *buf_mine,*buf_all;

  // which side of a contact is kept: both (replicated), the force on
  //   the body being processed, or the reaction on its partner

  enum{BOTH,PRIMARY,REACTION};

  // partner bodies of an owned body, sorted ascending

  int *jsort;
  int maxjsort;

  void body(int, double **, double **);
  int local_surfs();            // bins over the local copies, not owned
  void partner_pass(int, int, double **, double **);
  void contact(int, double *, double *, double *, double *, int,
               double **, double **, int);
};

}

#endif
