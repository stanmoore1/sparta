/* -*- c++ -*- ----------------------------------------------------------
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

#ifndef SPARTA_DATAMASK_AUDIT_KOKKOS_H
#define SPARTA_DATAMASK_AUDIT_KOKKOS_H

#include "pointers.h"

#include <string>
#include <vector>

namespace SPARTA_NS {

#ifndef SPARTA_KOKKOS_DEBUG_SYNC

// Without the sync debugging option this compiles away.  The class still exists
// so the call sites need no conditional compilation of their own.

class DatamaskAudit {
 public:
  DatamaskAudit(SPARTA *, const char *, const char *, unsigned int) {}
  static void enable(int) {}
  static void note_modified(unsigned int) {}
  static void note_synced(unsigned int) {}
  static void report(SPARTA *) {}
  static void trace_end(const char *, const char *) {}
};

#else

/* ----------------------------------------------------------------------
   Check a style against what it declares that it changes.

   A KOKKOS style declares the shared arrays it reads and writes in
   datamask_read and datamask_modify, and ModifyKokkos copies data between the
   host and the device around the call based on those declarations.  A style
   that changes an array it did not declare, and did not mark itself, leaves the
   other copy stale, and a later copy in the opposite direction then overwrites
   the new values with the old ones.  On a GPU that silently changes the
   results; on a CPU it cannot be seen at all, because both copies are the same
   memory there.

   This compares the contents of the arrays rather than the coherence flags,
   because a style that forgets to declare a write leaves those flags looking
   perfectly clean.  Only the data itself shows what happened.

   SPARTA's KOKKOS styles declare EMPTY_MASK and mark what they changed as they
   go, with particle_kk->modify(Device,PARTICLE_MASK) and the like.  That is the
   contract this checks: EMPTY_MASK means every array is snapshotted, and only
   what the style actually marked is excused.

   Declaring more than is written is only wasteful and is not reported, except
   for a style that declares every array: that one is reported, because there is
   then nothing left to compare and silence would read as a clean result.  It is
   what a style gets by leaving datamask_modify at the ALL_MASK that Fix sets.
------------------------------------------------------------------------- */

class DatamaskAudit {
 public:
  DatamaskAudit(SPARTA *sparta, const char *what, const char *style,
                unsigned int datamask_modify);
  ~DatamaskAudit();

  // Off unless SPARTA_KOKKOS_AUDIT is set, and off outside the timestep loop
  // even then: setup and input processing rewrite whatever they like and would
  // bury the reports.
  static void enable(int flag);

  // A style may declare EMPTY_MASK and mark what it changed itself, per routine,
  // which is just as correct as declaring it up front.  ParticleKokkos::modify()
  // and its Grid and Surf counterparts report the masks they are given here so
  // that those count as declared too.
  static void note_modified(unsigned int mask);

  // A sync writes the very side being watched, so it would look like the style
  // had written it.  Take the affected arrays' contents again instead, which
  // keeps a later write by the style itself visible.
  static void note_synced(unsigned int mask);

  // called from the dual view once a sync has written the device side
  void rebaseline_one(const void *device_data);
  static void report(SPARTA *sparta);
  static void trace_end(const char *what, const char *style);

  // One entry per array watched.  Several arrays share a bit -- every custom
  // per-particle attribute is CUSTOM_MASK -- so an entry is identified by its
  // bit together with its address, never by the bit alone.
  struct Array {
    unsigned int bit;
    std::string name;
    const char *data;
    size_t bytes;
    int stride;    // bytes per entry, to name the particle or cell that changed
    bool stale;    // the device side owed a sync when the style started
  };

  // The lengths every snapshot is taken against.  A style that adds particles
  // or refines the grid leaves nothing comparable, and this is how that is
  // noticed.
  struct Extents {
    int nparticle, nspecies;
    int gnlocal, gnghost, gnsplitlocal, gnsplitghost, gnparent, gmaxlevel;
    int snlocal, snown;
    bool operator!=(const Extents &o) const
    {
      return nparticle != o.nparticle || nspecies != o.nspecies || gnlocal != o.gnlocal ||
          gnghost != o.gnghost || gnsplitlocal != o.gnsplitlocal ||
          gnsplitghost != o.gnsplitghost || gnparent != o.gnparent ||
          gmaxlevel != o.gmaxlevel || snlocal != o.snlocal || snown != o.snown;
    }
  };

 private:
  SPARTA *sparta;
  const char *what;
  std::string style;
  unsigned int declared;
  Extents extents;
  std::vector<Array> arrays;
  std::vector<std::vector<char>> before;
  bool active;

  void rebaseline(unsigned int mask);
};

#endif    // SPARTA_KOKKOS_DEBUG_SYNC

}    // namespace SPARTA_NS

#endif
