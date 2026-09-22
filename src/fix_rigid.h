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

#ifdef FIX_CLASS

FixStyle(rigid,FixRigid)

#else

#ifndef SPARTA_FIX_RIGID_H
#define SPARTA_FIX_RIGID_H

#include "fix.h"
#include "my_page.h"
#include <map>
#include <unordered_map>
#include <string.h>

namespace SPARTA_NS {

// 1 if a fix style string names a fix rigid, with or without the KOKKOS suffix

inline int fix_rigid_style(const char *style)
{
  if (strncmp(style,"rigid",5) != 0) return 0;
  return (style[5] == '\0' || strcmp(&style[5],"/kk") == 0);
}

class FixRigid : public Fix {
 public:
  int nbody;              // # of rigid bodies

  // per-body state read by the particle mover and compute surf
  // all per-body arrays are indexed by body index 0 to nbody-1
  //   which is the user body ID - 1

  double **xcm,**vcm;     // COM and velocity of COM
  double **omega;         // omega in space frame
  double *invmass;        // 1/massbody
  double **invinertia;    // 3x3 space-frame inverse inertia tensor at
                          //   start of step, for collision recoil
  double **quat;          // quaternion for orientation of body
  double **xcmnew;        // new COM at end of timestep
  double **quatnew;       // new quaternion at end of timestep
  double **xcmmid;        // mid-step COM, torques are tallied about it

  // per-element tables, elements are ordered by body

  double ***displace;     // displacement in body frame of line/tri points from COM
  int *body;              // body index of each element
  int *bodystart;         // elements of body I = bodystart[I] to bodystart[I+1]-1

  // per local+ghost surf: body index and element index, -1 = static
  // read by the mover (via Update::rigidmap) and the force sum

  // the elements of a body in groups of EGROUP with a box around each,
  //   so a per-cell scan can skip a whole group; rebuilt with the
  //   element boxes by body_bbox()

  enum{EGROUP = 8};
  int *groupstart;              // groups of body I = start[I]..start[I+1]-1
  double **elemglo,**elemghi;

  // elements of group g of body ibody

  void group_range(int ibody, int g, int &ilo, int &ihi) {
    ilo = bodystart[ibody] + (g - groupstart[ibody]) * EGROUP;
    ihi = MIN(ilo + EGROUP, bodystart[ibody+1]);
  }

  // the deletion pass: one byte per cell saying whether its particles
  //   can be inside a body, and the bodies which reach the last cell

  char *celldel;
  int maxcelldel;
  int *celbody;
  int maxcelbody;

  int *surfbody;
  int *surfelem;
  int maxsurfmap;         // allocated length of both

  // body_elem() = element index of a global surf ID, -1 if not in a body

  int body_elem(surfint id)
  {
    std::unordered_map<surfint,int>::const_iterator it = idmap.find(id);
    if (it == idmap.end()) return -1;
    return it->second;
  }

  FixRigid(class SPARTA *, int, char **);
  virtual ~FixRigid();
  int setmask();
  void init();
  void setup();
  virtual void start_of_step();
  virtual void end_of_step();
  void post_run();
  void grid_changed();
  double memory_usage();
  void init_surfs();            // per-run setup of the body surfs
  int ensure_local_copies();    // distributed: local copies of body surfs
                                //   returns 1 if surf arrays were changed
  void proc_bbox();             // bbox of this proc's owned + ghost cells
  void proc_boxes();            // every proc's owned and owned+ghost bboxes
  void neighbor_list();         // owned: procs a body exchange can reach
  int body_owner(double *);     // rank which owns a body at this COM
  void body_status(int = 0);    // body ownership/status and the loop lists
                                //   1 = every proc holds every body
  void gather_all();            // every proc's owned rows to all procs
  virtual void refresh_all();   // gather + host geometry of every body
                                //   virtual: fix rigid/kk uses the device
  int host_surfs_needed();      // 1 if a host consumer reads the surf
                                //   arrays this step
  void surfs_changed(int, int = 0);  // notify per-surf models of the above
                                     //   2nd arg = 0 in run, 1 init, 2 setup
  void rebin_contacts();        // owned: the push-off bins, on a change
  virtual void surf_maps();     // rebuild surfbody/surfelem
  void body_geometry(int);      // one body's geometry from its pose
  virtual void host_geometry(int) {}   // fix rigid/kk: host copy of it
  virtual void refresh_host_surfs() {} // fix rigid/kk: host surf copies

  // force/torque on a body surf from one particle collision,
  //   called by the particle mover for every collision with a body surf
  // iorig = particle before the collision, ip/jp = particles after it

  void surf_tally(int, Particle::OnePart *,
                  Particle::OnePart *, Particle::OnePart *);

  double compute_scalar();
  double compute_vector(int);
  double compute_array(int,int);

 protected:
  int igroup,groupbit;
  int bodystyle;          // SINGLE, TYPE, or CUSTOM
  char *customname;       // name of per-surf custom vector for CUSTOM
  int ngroupsurf;         // # of surfs in group when fix was defined
  int initflag;           // 1 once init() has run, 0 for a fix defined
                          //   after the last init, which has no state
  char *infile;
  char *outfile;
  int outevery;

  int dim;
  int axiflag;            // 1 if the domain is axisymmetric, in which
                          //   case the body is a body of revolution about
                          //   the x axis and may only translate along x
                          //   and spin about x (see start_of_step)
  int bodyflag;           // 1 if dstyle = body: params for a single body
  int densityflag;        // 1 if dstyle = density: mass/com/moi are
                          //   computed from the body geometry
  double density;         // uniform body density for the above

  // body params from dstyle = body, or vcom/angmom for every body
  //   from the optional keywords with dstyle = density

  double massone,xcmone[3],moione[6],vcmone[3],angmomone[3];

  double fext[3];         // constant external force on each COM

  // force/torque read from an infile, for run continuation
  // a body moves on a step under the force accumulated on the previous
  //   one, so a continuation which starts from zero force loses one
  //   step's impulse; the outfile writes them and read_infile restores
  //   them, after setup_body() has zeroed them

  int forceinfile;
  double **fcm_infile,**torque_infile;
  int kokkosable;         // 1 if this is the KOKKOS variant (rigid/kk)

 public:

  // the tables below are read by the RigidRemap and RigidContact helpers

  int nsurf;     // # of surfs which comprise surfaces of all rigid bodies
  int *lblist;            // local surf index of each element
  double **elemlo;        // per-element bounding boxes: swept boxes for
  double **elemhi;        //   this step, or current boxes after commit
  double ***bodypt;       // bodypt[i][j] = corner pt j of element i
  double **bodynorm;      // outward normal of element i
  double **bbodylo;       // bounding box around each body
  double **bbodyhi;
  double *bboxeps;        // inflation applied to bbodylo/bbodyhi
  double *rmaxbody;       // max distance of any body corner pt from COM
  double *rminbody;       // min distance of any point of any body element
                          //   from the COM.  a cell which lies wholly
                          //   inside this radius cannot hold a body surf

  // body ownership and the per-step loop lists, set by body_status()
  // replicated: every proc owns every body, both lists = identity

  int bodymode;           // REPLICATED or OWNED
  int *bodystatus;        // OWNEDBODY, GHOSTBODY, or FARBODY on this proc
  int *bodyowner;         // rank which owns each body
  bigint *bodystamp;      // step this proc's copy of each body arrived on
  int nown,*ownlist;      // bodies this proc owns
  int nblist,*blist;      // owned + ghost bodies, ascending body index
  int nnewghost,*newghost;  // GHOST now, FAR on the previous step
  int blistgen;           // bumped whenever blist changes

  // the records the exchanges of owned mode move: the dynamic state of
  //   one body, and one proc's partial force/torque sum for a ghost
  // static per-body data stays replicated from setup

  enum{NBODYDATUM = 63};
  struct BodyDatum { int ibody,pad; double v[NBODYDATUM]; };
  struct PartDatum { int ibody,pad; double f[6]; };

  int body_box(double *, double *, int **);  // bodies overlapping a box
  void body_bbox(int, int);     // bbox of body, current or swept over step
  int inside_body(int, double *); // 1 if point is inside rigid body, else 0
  int inside_any_body(double *); // 1 if inside any rigid body

 protected:
  // replicated body geometry (bodypt/bodynorm above), the authoritative
  //   source for all body computations (bbox, inside tests, watertight,
  //   contacts); for distributed surfs it is gathered from the owned
  //   copies at setup and regenerated from the body pose each step

  surfint *sids;          // global surf ID of each element
  int *bodymask;          // per-element group mask
  int *bodytype;          // per-element surf type
  int *bodytrans;         // per-element transparent flag
  int *bodyisc;           // per-element collision model index
  int *bodyisr;           // per-element reaction model index
  std::unordered_map<surfint,int> idmap;  // global surf ID -> element index

  // where this proc stores copies of body elements in the Surf arrays
  //   (lblist and the copy list above), see scan_copies()

  int nsurfall;           // surf->nlocal when the fix was defined
  int *bodyneed;          // 1 if this proc needs local copies of a body
  double proclo[3],prochi[3];  // bbox of this proc's owned + ghost cells

  // bodies owned: per-proc boxes the ownership and ghost rules read

  double *ownboxall;      // nprocs x 6: bbox of each proc's owned cells
  double *procboxall;     // nprocs x 6: ditto, owned + ghost cells
  double bodycut;         // a proc holds a body whose COM is within this
                          //   distance of its owned + ghost cells
  double bodycut_user;    // cutoff keyword value, <= 0.0 = use the default
  double rmaxmax;         // max over bodies of rmaxbody, for that default

  // bodies owned: the fixed neighbor list both exchanges run over,
  //   set by proc_boxes() whenever the cells change
  // the test is symmetric in the two procs, so r is my neighbor exactly
  //   when I am r's: the exchanges need no handshake

  int nneigh;             // procs within bodycut of mine, ascending
  int *neighlist;         // their ranks, excluding me
  int *neighslot;         // nprocs: rank -> slot, -1 if not a neighbor

  // bodies owned: the per-step exchanges and the gathered outputs
  // records are bucketed by neighbor slot, so the receive buffer is
  //   laid out in ascending source rank

  BodyDatum *bodysend,*bodyrecv;
  int maxbodysend,maxbodyrecv;
  PartDatum *partsend,*partrecv;
  int maxpartsend,maxpartrecv;
  int *nsendslot,*nrecvslot;   // records exchanged with each neighbor
  int *sendoffset,*recvoffset; // first record of each slot in the buffers
  MPI_Request *neighreq;       // 2*nneigh, both rounds of an exchange
  int *gathernum;         // records each proc contributes to gather_all()
  int *gathercount;       // and the byte counts/displacements of them
  int *gatherdispl;
  bigint gathervalid;     // step the gathered rows are valid for

  void pack_datum(int, BodyDatum &);
  void unpack_datum(const BodyDatum &);
  int neighbor_counts(int);     // trade the per-slot record counts
  void neighbor_data(char *, char *, int, int);   // trade the records
  void exchange_forward();      // owner -> holders, after the integration
  void exchange_reverse();      // ghost partial sums -> owner
  virtual void newghost_geometry();  // start-of-step geometry of
                                     //   FAR -> GHOST; virtual: fix
                                     //   rigid/kk uses the device
  void check_bodycut();         // swept bbox vs the bodies cutoff
  void body_warning(int &, int);  // once per run, per rank

  int nemitfix;           // emit fixes, which read the host surfs on a
                          //   re-map which changed cells or markings
  int copiesappended;     // 1 if start_of_step() appended local copies
  int ncopy,maxcopy;      // all local copies of body elements
  int *copy_index;        //   local surf index of each copy
  int *copy_elem;         //   element index of each copy
  int nolist;             // # of body elements this proc owns
  int *olist_own;         // owned-array index of each
  int *olist_elem;        // element index of each

  double mincellsize;     // smallest edge length of any grid cell
  int warnrotate;         // 1 after warning about rotation rate
  int warntranslate;      // 1 after warning about translation rate
  int warnexit;           // 1 after warning that a body exited the box
  int warndefer[3];       // owned: the once-per-run warnings of the body
                          //   loops, reduced and printed by post_run()
  int warnfallback;       // 1 after warning about incremental fallback
  int warndelete;         // 1 after warning about deletions during a run
  bigint ndelrun;         // per-proc deletions since this run started

  // per-body dynamic state

  double *massbody;       // total mass of rigid body enclosed by surfs
  double **moi;           // 6 MOI in space frame
  double **inertia;       // 3 diagonalized MOI in body frame
  double **angmom;        // angular momentum in space frame
  double **ex_space,**ey_space,**ez_space;  // principal axes of body
  double **fcm;           // force on COM in space frame
  double **torque;        // torque on body in space frame
  double **fpush;         // push-off force on COM: static surfs, other
                          //   bodies' reactions, box boundaries
  double **tqpush;        // torque from push-off contacts this step

  int pushflag;           // 1 if push-off forces are enabled
  int pushboundflag;      // 1 to also push off non-periodic boundaries
  int pushstyle;          // force law, see RigidContact
  double kpush;           // spring constant for push-off force
  double pushcutoff;      // distance below which push-off is applied
  double gammapush;       // dashpot damping coefficient, 0 = elastic
  class RigidContact *contact;   // the contact model, NULL if no push

  // bins over bodies by COM, rebuilt each step after the bodies move,
  //   for finding the bodies near another body, a grid cell, or a point
  //   without a loop over all bodies

  int bodynbin[3];        // # of bins in each dim
  double bodybinlo[3];    // bin grid origin
  double bodybininv[3];   // inverse bin edge lengths
  int *bodybinstart;      // CSR offsets into bodybinlist per bin
  int *bodybinlist;       // body indices, binned by COM
  int *bodycand;          // query result buffer
  int maxbodycand;
  double rmaxall;         // max over bodies of rmaxbody + bbox inflation

  // per-element force/torque tallies of the particle mover this step
  // one row per body element hit, shared by every local copy of the
  //   element, so the per-body sums are formed element by element in
  //   element order, which does not depend on how many local copies of
  //   an element a proc holds and is the order the device sums them in
  // the tally converts the momentum a collision gives the surf into a
  //   force via nfactor_inverse = fnum/dt, exactly as compute surf does

  int ntally,maxtally;
  int *tally2elem;        // element index of each row
  int *elem2tally;        // row of each element, -1 if not hit this step
  double **ftally;        // fx,fy,fz,tx,ty,tz per row
  double nfactor_inverse;
  int weightflag;         // 1 if particles carry cell weights

  // work buffers for the fused force/torque Allreduce over all bodies

  double *ftbuf_mine;
  double *ftbuf_all;

  // remap of body surfs to grid cells

  int remapmode;          // CUTCELL or INCREMENTAL
  int rotstyle;           // EULER or RICHARDSON quaternion update
  class RigidRemap *remap;

  // per-run counts of how the re-map went, printed by post_run() when
  //   the SPARTA_RIGID_TIMING environment variable is set

  bigint nstep_run;       // steps this run
  bigint nstep_inplace;   // steps whose split changes were applied in place
  bigint nstep_rebuild;   // steps which took the collective ghost rebuild
  bigint nstep_fallback;  // steps which fell back to a full re-map

  // per-stage wall time this run, accumulated when timing is on
  // the helper classes time their own sections with add_time()

 public:
  enum{T_INTEGRATE,T_INTEG_EX,T_COPIES,T_COLLIDELIST,T_COLL_ENUM,
       T_COLL_MERGE,T_COLL_RESET,T_TALLY,
       T_FORCES,T_FORCE_TALLY,T_FORCE_EX,T_SETXV,T_CONTACT,
       T_RECUT,T_RECUT_CAND,T_RECUT_LISTS,T_RECUT_CMP,T_RECUT_CUT,
       T_RECUT_TYPE,T_RECUT_RED,T_RECUT_COMB,
       T_APPLY,T_APPLY_REST,T_APPLY_COUNT,
       T_REMOVE,T_REMOVE_SPLIT,T_REMOVE_PASS,T_REMOVE_COMP,
       T_NSTAGE};

  // per-run counts of the work each stage did, printed with the timers

  enum{C_PTEST,C_PDEL,C_CAND,C_CANDSKIP,C_LISTCH,C_CUT,C_PENDING,
       C_BODYCELL,C_ELEMBOX,C_SWCELL,C_SWENT,C_SWEXTRA,C_NCOUNT};

  int timeflag;
  double stagetime[T_NSTAGE];
  bigint stagecount[C_NCOUNT];
  double stagestart;
  // with SPARTA_RIGID_TIMING the KOKKOS version fences the device at the
  //   stage boundaries, so a stage is charged its own asynchronous kernels
  //   rather than the next stage which happens to wait for them

  virtual void stage_fence() {}
  void stage_begin() { if (timeflag) { stage_fence(); stagestart = MPI_Wtime(); } }
  void stage_end(int i) { if (timeflag) { stage_fence(); stagetime[i] += MPI_Wtime() - stagestart; } }
  void add_time(int i, double t) { stagetime[i] += t; }
  void add_count(int i, bigint n) { stagecount[i] += n; }
 protected:

  bigint ndeleted;        // per-proc count of deleted particles
  bigint ndeleted_all;    // cached global sum for compute_scalar()
  bigint ndelvalid;       // timestep the cached sum is valid for

  void read_infile(char *);
  void write_outfile();
  void body_error(int, const char *);  // error naming a body
  void setup_body();
  void allocate_bodies();       // per-body arrays, once nbody is known
  void check_body_params(int);  // 2d/axisymmetric consistency of params
  void setup_body_one(int);     // body frame axes and quaternion
  void setup_body_displace(int); // body-frame pts etc, once axes are set
  void body_properties(int, double);  // mass/com/moi from the geometry
  void body_properties_axi(int, double); // the same for a body of revolution
  void axi_project(double *, double *);   // azimuthal average of a
                                          //   force and torque on the body
  void check_watertight(int);
  void set_recoil(int);         // set invmass/invinertia from current axes
  void final_kick(int);         // second half kick of velocity Verlet
  void check_enclosed(int);     // reject a body which encloses no area/volume

  void clear_tally();           // reset the per-element tallies
  void grow_tally();
  void sum_forces();            // per-body force/torque from the tallies
                                //   and the push-off contacts
  void set_pose();              // bodies to the end-of-step pose
  int posesplit;                // 1 between initial_integrate() and
                                //   set_pose(): xcm/quat at the start of
                                //   the step, ex/ey/ez_space at its end
  virtual void swept_boxes();   // per-element swept boxes of the step
  virtual void sum_tallies();   // local per-body sums of the tallies
                                //   virtual: fix rigid/kk sums the device
                                //   tallies with a kernel

  // the per-step skeleton, see start_of_step() and end_of_step()

  void initial_integrate();     // half kick + drift to the end-of-step pose
  virtual void set_xv();                // commit the pose, regenerate the geometry
  void check_bounds();          // body vs simulation box
  void final_integrate();       // push-off forces + second half kick
  void remap_grid();            // re-map the body surfs to grid cells

  void body_bins();             // bin bodies by COM
  void gather_body();           // build replicated body element table
  void check_body_attributes(); // error if body surf attributes changed
  int same_coords(double *, double *, double *, int);  // coords = elem
  void scan_copies();           // lblist and the local copy list
  void update_surf_copies();    // write bodypt/bodynorm into Surf storage
  virtual void grid_rebuild();  // full re-map of all surfs to grid cells
  bigint remove_inside_particles(int);  // all bodies, used at setup
  virtual void remove_inside_all(int);  // fused pass over all bodies, per step
                                        //   virtual so fix rigid/kk can run
                                        //   it as a device kernel instead
  void end_of_run_delete_warning();     // once-per-run warning for the above

  // pull the particles of every changed split cell up into the split cell
  //   itself, before any cell is restructured
  // virtual so fix rigid/kk can do it with a device kernel and keep the
  //   particle array on the device

  virtual void combine_split_all();

  // sort the particles so Grid::remove_marked_cells() can walk the per-cell
  //   lists of the cells it moves
  // virtual so fix rigid/kk can skip it: the cells that routine moves are
  //   the detached sub cells, which split_cell_unset() has already emptied,
  //   so the walk never reaches a particle and the sort only exists to make
  //   empty lists valid

  virtual void sort_for_split_rebuild();

  // particles of the cells Grid::restructure_split_cells() moved: the
  //   host routine relabels them itself from the sorted per-cell lists,
  //   fix rigid/kk relabels the device copy from Grid::movedfrom/movedto

  virtual void relabel_moved_cells() {}

  // hook called before host code touches the particle array, so fix
  //   rigid/kk can bring the particles back from the device: it otherwise
  //   leaves them there for the whole step (see host_begin)
  // no-op in the non-KOKKOS class

  virtual void particles_to_host() {}
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: Fix rigid bodies owned requires remap incremental

The cutcell re-map needs every body on every proc.

E: Fix rigid bodies owned requires a clumped grid decomposition

Body ownership is decided from the bounding box of each proc's owned
cells.  Use create_grid clump or block, balance_grid rcb, or fix balance
rcb.

E: Fix rigid body moved beyond the bodies cutoff

With bodies owned, a body may only reach the procs which were sent it,
which are those within the cutoff of its COM.  Raise the cutoff value of
the bodies keyword, or lower the timestep.

E: Fix rigid bodies owned with distributed surfs and push requires global gridcut >= N

The owner of a body computes its push-off contacts from the surfs of its
own and ghost cells, so that layer must reach as far as a contact does.
Raise the global gridcut, or set it negative to copy every cell.

E: Fix rigid body exchange reached a non-neighbor proc

The body exchanges only reach procs within the bodies cutoff of this one.
This should not be possible and indicates a bug.

*/
