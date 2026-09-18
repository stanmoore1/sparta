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

  int *irigid;            // per-surf flags, indexed by local surf index:
                          // -1 = static surf, else body index
                          // non-distributed surfs only, else NULL
  int nsurfall;           // length of irigid = surf->nlocal when allocated
                          // Update::init() clamps its scan to this length

  // body_elem() = element index of a global surf ID, -1 if not in a body
  // used by Update::build_rigidmap() for distributed surfs

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
  void grid_changed();
  double memory_usage();
  int ensure_local_copies();    // distributed: local copies of body surfs
  void proc_bbox();             // bbox of this proc's owned + ghost cells
                                //   returns 1 if surf arrays were changed
  void surfs_changed(int, int = 0);  // notify per-surf models of the above
                                     //   2nd arg = 0 in run, 1 init, 2 setup

  double compute_scalar();
  double compute_vector(int);
  double compute_array(int,int);

 protected:
  int igroup,groupbit;
  int bodystyle;          // SINGLE, TYPE, or CUSTOM
  char *customname;       // name of per-surf custom vector for CUSTOM
  int ngroupsurf;         // # of surfs in group when fix was defined
  char *csurfID;
  class ComputeSurf *csurf;
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

  int body_box(double *, double *, int **);  // bodies overlapping a box
  void body_bbox(int, int);     // bbox of body, current or swept over step
  int inside_body(int, double *); // 1 if point is inside rigid body, else 0
  int inside_any_body(double *); // 1 if inside any rigid body

 protected:
  int *slist;    // list of local surf indices for body surfs
                 //   non-distributed surfs only, else NULL

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
  // non-distributed: lblist = slist, one copy per element
  // distributed: ensure_local_copies() guarantees at least one copy of
  //   every element in the local (non-ghost) range; if the surf comm
  //   left duplicates, every copy is tracked so all are kept current

  int *bodyneed;          // 1 if this proc needs local copies of a body
  double proclo[3],prochi[3];  // bbox of this proc's owned + ghost cells
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

  // work buffers for the fused force/torque Allreduce over all bodies

  double *ftbuf_mine;
  double *ftbuf_all;

  // remap of body surfs to grid cells

  int remapmode;          // CUTCELL or INCREMENTAL
  int rotstyle;           // EULER or RICHARDSON quaternion update
  class RigidRemap *remap;

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

  void body_bins();             // bin bodies by COM
  void gather_body();           // build replicated body element table
  void check_body_attributes(); // error if body surf attributes changed
  int same_coords(double *, double *, double *, int);  // coords = elem
  void update_surf_copies();    // write bodypt/bodynorm into Surf storage
  void grid_rebuild();          // full re-map of all surfs to grid cells
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

*/
