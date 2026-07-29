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

// one class implements both styles, they differ only in the binned quantity

#ifdef COMPUTE_CLASS

ComputeStyle(edf/surf,ComputeEDFSurf)
ComputeStyle(adf/surf,ComputeEDFSurf)

#else

#ifndef SPARTA_COMPUTE_EDF_SURF_H
#define SPARTA_COMPUTE_EDF_SURF_H

#include "compute.h"
#include "surf.h"
#include "hash3.h"

namespace SPARTA_NS {

class ComputeEDFSurf : public Compute {
 public:
  ComputeEDFSurf(class SPARTA *, int, char **);
  ComputeEDFSurf(class SPARTA* sparta) : Compute(sparta) {} // needed for Kokkos
  ~ComputeEDFSurf();
  virtual void init();
  void compute_per_surf();
  virtual void clear();
  virtual void surf_tally(double, int, int, int, Particle::OnePart *,
                          Particle::OnePart *, Particle::OnePart *);
  virtual int tallyinfo(surfint *&);
  virtual void post_process_surf();
  bigint memory_usage();

 protected:
  int groupbit,imix,ngroup,ntotal;
  int maxsurf,combined;

  int distflag;            // ENERGY or ANGLE, set from style name
  int dirstyle;            // INCIDENT or REFLECTED particles
  int engstyle;            // KE, EROT, EVIB, or ETOT (ENERGY only)
  int oobstyle;            // IGNORE or CLAMP for out-of-range samples

  int nbin;                // # of bins per mixture group
  double lo,hi;            // bin range
  double invdelta;         // nbin/(hi-lo)

  int weightflag;          // 1 to tally the particle weight, 0 to tally counts
  int cellweightflag;      // 1 if cell weighting is enabled

  int ntally;              // # of surfs I have tallied for
  int maxtally;            // # of tallies currently allocated
  surfint *tally2surf;     // tally2surf[I] = surf ID of Ith tally

  // hash for surf IDs

#ifdef SPARTA_MAP
  typedef std::map<surfint,int> MyHash;
#elif defined SPARTA_UNORDERED_MAP
  typedef std::unordered_map<surfint,int> MyHash;
#else
  typedef std::tr1::unordered_map<surfint,int> MyHash;
#endif

  MyHash *hash;

  int dim;                 // local copies
  Surf::Line *lines;
  Surf::Tri *tris;

  // allocate the tally arrays, called by init() and whenever the grid changes
  // the host version grows them on demand instead, so it is a no-op here

  virtual void allocate_tally() {}

  virtual void grow_tally();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: Compute edf/surf group ID does not exist

Self-explanatory.

E: Compute edf/surf mixture ID does not exist

Self-explanatory.

E: Compute edf/surf Nbin must be > 0

Self-explanatory.

E: Compute edf/surf bin range must have lo < hi

Self-explanatory.

E: Invalid compute edf/surf optional keyword

Self-explanatory.

E: Compute adf/surf does not support the value keyword

The {value} keyword selects which energy to histogram, so it only
applies to compute edf/surf.  Compute adf/surf always histograms the
polar angle from the surface normal.

E: Cannot use compute edf/surf when surfs do not exist

Self-explanatory.

E: Cannot use compute edf/surf with implicit surfs

This compute only tallies collisions with explicit surface elements.

E: Number of groups in compute edf/surf mixture has changed

This mixture property cannot be changed after this compute command is
issued.

E: Must use compute edf/surf/kk if Kokkos is enabled

UpdateKokkos only drives surface tallying through Kokkos-enabled
computes, so the Kokkos version must be used.  The -sf kk command-line
switch selects it automatically.

*/
