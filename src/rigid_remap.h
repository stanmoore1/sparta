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

#ifndef SPARTA_RIGID_REMAP_H
#define SPARTA_RIGID_REMAP_H

#include "pointers.h"

namespace SPARTA_NS {

// reasons recut() requests a full grid re-map instead

enum{FALLBACK_NONE,FALLBACK_SURFMAX};

class RigidRemap : protected Pointers {
 public:
  int typechanged;        // 1 if recut() changed a cell type this step
  int listschanged;       // 1 if a per-cell surf list changed on the host
                          //   since the flag was cleared (fix rigid/kk)
  int splitchanged;       // 1 if a split cell was re-cut in place this
                          //   step, so its particles must be
                          //   redistributed over its new pieces
  int npending;           // # of owned cells whose piece count changes
                          //   this step, applied by apply_pending()

  RigidRemap(class SPARTA *, class FixRigid *);
  ~RigidRemap();
  void setup();                   // per run, after the bodies are placed
  void collision_lists();         // add swept body surfs to the cells
  void reset_collision_lists();
  void refresh();                 // per-cell state, before the bodies move
  int recut();                    // re-cut cells near the bodies
  void apply_pending();           // restructure the piece-count changes
  void grid_changed();
  double memory_usage();

 private:
  class FixRigid *fix;
  int dim;

  // previous position of each body: bbox at the end of the last step,
  //   with which the current bbox bounds the region to re-cut

  double **prevlo,**prevhi;

  // per-cell accumulation of (cell, swept element) entries across
  //   bodies, so the collision list pass visits only candidate cells

  int *swstamp;           // per-cell visit stamp
  int *swhead;            // head of per-cell entry chain, per stamp
  int maxswcell;          // allocated length of swstamp/swhead
  int swcur;              // current stamp
  int *swcells;           // cells touched this step
  int nswcell,maxswcells;
  int *entnext;           // entry chains: next index and element's
  surfint *entelem;       //   local surf index
  int nent,maxent;
  surfint *swlist;        // work buf for one cell's merged collision list
  int maxswlist;
  int nswept;             // # of cells given a collision list this step

  // per owned cell: 1 if it is INSIDE because of the static surfs, so
  //   the bodies moving over and away from it never re-type it
  // sub cells are never INSIDE, and only sub cells are added, removed
  //   or moved by apply_pending(), so the flags stay valid across it
  //   and only a grid rebuild, adapt or migration invalidates them

  char *staticinside;
  int maxstatic;
  int staticvalid;

  // work list of cells overlapping the re-cut region this step

  int nrcand,maxrcand;
  int *rcand;

  // work bufs for re-cutting one cell

  surfint *newlist;       // the cell's new surf list
  int *newmap;            //   and its piece map
  int maxnewlist;         // allocated length of both
  surfint *reclist;       // candidate surfs for the cell
  int maxreclist;

  // owned cells whose number of disconnected flow pieces changes this
  //   step, applied together by apply_pending(): adding or removing a
  //   sub cell changes this proc's cell count and the sub cell indices
  //   other procs migrate particles into, neither of which can be done
  //   while ghost cells are stored

  struct PendingSplit {
    int icell;            // owned cell whose piece count changes
    int nsplitnew;        // its new # of pieces, 1 = no longer split
    int nsurf;            // # of surfs in the cell, = length of map
    int *map;             // the new piece map, copied out of the work
    int maxmap;           //   buffer the next cell's cut overwrites
    int xsub;             // reference piece and point for split2d/3d
    double xsplit[3];
    double *vols;         // flow volume of each new piece
    int maxvols;
  };

  PendingSplit *pending;
  int maxpending;

  int cell_cut(int);
  void mark_static();
  void split_ghost_drop(int);
  void split_pending(int, int, int, int *, int, double *, double *);
};

}

#endif
