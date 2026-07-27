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

#ifndef SPARTA_RAN_KNUTH_H
#define SPARTA_RAN_KNUTH_H

namespace SPARTA_NS {

class RanKnuth {
 public:
  RanKnuth(int);
  RanKnuth(double);
  ~RanKnuth() {}
  void reset(double, int, int);
  double gaussian();
  double poisson(double);

  // uniform() is defined here rather than in random_knuth.cpp so it can be
  //   inlined; it is called several times per collision attempt, and the
  //   lagged-Fibonacci state update is only a handful of instructions, so
  //   the call itself dominated
  // the arithmetic is unchanged, so the random number stream is identical
  // the one-time seeding is left out of line in init_state()

  inline double uniform() {
    int mj;
    double rn;

    if (!initflag) init_state();

    while (1) {
      if (++inext == 56) inext = 1;
      if (++inextp == 56) inextp = 1;
      mj = ma[inext] - ma[inextp];
      if (mj < 0) mj += RK_MBIG;
      ma[inext] = mj;
      rn = mj*RK_FAC;

      // make sure the random number is valid

      if (rn > 0.0 && rn < 1.0) break;
    }

    return rn;
  }

 private:
  // same values as MBIG and FAC in random_knuth.cpp, needed here because
  //   uniform() is inline; constexpr so they fold into the instruction
  //   stream instead of becoming loads

  enum { RK_MBIG = 1000000000 };
  static constexpr double RK_FAC = 1.0/1000000000.0;

  int seed,save;
  double second;
  int initflag,inext,inextp;
  int ma[56];

  void init_state();
};

}

#endif
