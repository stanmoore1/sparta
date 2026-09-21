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
#include "grid.h"

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
  int restructured;       // 1 if apply_pending() moved cells in place
                          //   since the flag was cleared (fix rigid/kk)
  bigint ncand_run;       // per-run counts: candidate cells, cells whose
  bigint nlist_run;       //   cut list changed, cells cut
  bigint ncut_run;

  RigidRemap(class SPARTA *, class FixRigid *);
  virtual ~RigidRemap();
  void setup();                   // per run, after the bodies are placed
  virtual void collision_lists(); // add swept body surfs to the cells
  virtual void reset_collision_lists();
  void refresh();                 // per-cell state, before the bodies move
  virtual int recut();            // re-cut cells near the bodies
  int rebuild_needed();           // 1 if the pending changes need the
                                  //   collective grid rebuild
  void apply_pending(int);        // restructure the piece-count changes
  void grid_changed();
  void pack_prev(int, double *);        // a body's previous-position
  void unpack_prev(int, const double *);  //   state, for the exchanges
  double memory_usage();

 protected:
  class FixRigid *fix;
  int dim;
  int nswept;             // # of cells given a collision list this step

  // previous position of each body: bbox at the end of the last step,
  //   with which the current bbox bounds the region to re-cut

  double **prevlo,**prevhi;
  double **prevxcm;

  // per body, 1 if its COM is interior to it: a point closer to the COM
  //   than any element then shares its parity without a ray cast; a
  //   property of the shape, set once per run

  int *cominside;

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

  // per owned cell: 1 if it is INSIDE because of the static surfs, so
  //   the bodies moving over and away from it never re-type it
  // sub cells are never INSIDE, and only sub cells are added, removed
  //   or moved by apply_pending(), so the flags stay valid across it
  //   and only a grid rebuild, adapt or migration invalidates them

  char *staticinside;
  int maxstatic;
  int staticvalid;
  int staticgen;          // # of rebuilds of the flags, for a device copy

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
  //   step, applied together by apply_pending(): the piece map and
  //   volumes are copied out of the work buffers the next cell's cut
  //   overwrites

  Grid::SplitChange *pending;
  int *maxmap,*maxvols;   // allocated lengths of each entry's map/vols
  int maxpending;

  int cell_cut(int);
  void mark_static();
  void recut_cell(int, int, surfint *);   // cut one cell by a new list
  void apply_cut(int, int, double *, int *, int *, int, double *);
  void split_pending(int, int, int, int *, int, double *, double *);
};

}

#endif
