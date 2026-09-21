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

#include "mpi.h"
#include "math.h"
#include "rigid_contact.h"
#include "fix_rigid.h"
#include "update.h"
#include "domain.h"
#include "surf.h"
#include "comm.h"
#include "geometry.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

enum{PERIODIC,OUTFLOW,REFLECT,SURFACE,AXISYM};  // same as Domain

// local box/box overlap test, touching counts as overlap

static inline int box_overlap(double *alo, double *ahi,
                              double *blo, double *bhi)
{
  if (ahi[0] < blo[0] || alo[0] > bhi[0]) return 0;
  if (ahi[1] < blo[1] || alo[1] > bhi[1]) return 0;
  if (ahi[2] < blo[2] || alo[2] > bhi[2]) return 0;
  return 1;
}

/* ----------------------------------------------------------------------
   push-off contact forces between rigid bodies, static surfs and the
     non-periodic box boundaries
   every corner point of a body within the cutoff of a source element
     is pushed away from it by a spring, linear or Hertzian in the
     overlap, with an optional dashpot in the normal approach rate; the
     reaction of a contact with another body is applied to that body at
     the same point, so body-body contacts conserve momentum
   the source elements near a body are found from bins over the static
     surfs, built once per run, and from the bins over the bodies which
     FixRigid rebuilds each step
------------------------------------------------------------------------- */

RigidContact::RigidContact(SPARTA *sparta, FixRigid *fixrigid, int style_in,
                           double k_in, double cutoff_in, double gamma_in,
                           int boundflag_in) : Pointers(sparta)
{
  fix = fixrigid;
  dim = domain->dimension;
  axiflag = domain->axisymmetric;

  style = style_in;
  kpush = k_in;
  cutoff = cutoff_in;
  gamma = gamma_in;
  boundflag = boundflag_in;

  binstart = binlist = stamp = NULL;
  stampcur = 0;

  memory->create(buf_mine,6*fix->nbody,"rigid_contact:buf_mine");
  memory->create(buf_all,6*fix->nbody,"rigid_contact:buf_all");
}

/* ---------------------------------------------------------------------- */

RigidContact::~RigidContact()
{
  memory->destroy(binstart);
  memory->destroy(binlist);
  memory->destroy(stamp);
  memory->destroy(buf_mine);
  memory->destroy(buf_all);
}

/* ----------------------------------------------------------------------
   bin static surfs for candidate pruning
   built once per run; static surfs never move during a run
   each static surf is added to every bin its bbox overlaps (CSR layout);
     a query gathers the bins overlapping the cutoff-inflated body
     bbox and dedups multi-bin surfs with a visit stamp
   bin edge lengths are at least the cutoff, at most 64 bins per dim
   non-distributed: bin the static surfs of the local arrays, which
     hold all surfs on every proc (identical bins everywhere)
   distributed: bin the static surfs this proc OWNS; each surf is
     binned on exactly one proc, so per-proc contributions are
     disjoint partial sums which compute() merges
   static = surf not in any rigid body
------------------------------------------------------------------------- */

void RigidContact::setup()
{
  int i,k,m,ibx,iby,ibz;
  int blo[3],bhi[3];

  int distributed = surf->distributed;

  Surf::Line *lines;
  Surf::Tri *tris;
  int nslocal;
  if (!distributed) {
    lines = surf->lines;
    tris = surf->tris;
    nslocal = surf->nlocal;
  } else {
    lines = surf->mylines;
    tris = surf->mytris;
    nslocal = surf->nown;
  }

  int *surfbody = fix->surfbody;

  double *boxlo = domain->boxlo;
  double *boxhi = domain->boxhi;

  for (k = 0; k < 3; k++) {
    binlo[k] = boxlo[k];
    double len = boxhi[k] - boxlo[k];
    int n = (int) (len/cutoff);
    n = MAX(n,1);
    n = MIN(n,64);
    if (dim == 2 && k == 2) n = 1;
    nbin[k] = n;
    bininv[k] = n/len;
  }
  int nbins = nbin[0]*nbin[1]*nbin[2];

  memory->destroy(binstart);
  memory->destroy(binlist);
  memory->destroy(stamp);
  memory->create(binstart,nbins+1,"rigid_contact:binstart");
  memory->create(stamp,nslocal,"rigid_contact:stamp");
  for (i = 0; i < nslocal; i++) stamp[i] = 0;
  stampcur = 0;

  // two passes: count entries per bin, then fill

  double slo[3],shi[3];

  for (int pass = 0; pass < 2; pass++) {
    if (pass == 0)
      for (i = 0; i <= nbins; i++) binstart[i] = 0;

    for (m = 0; m < nslocal; m++) {

      // skip surfs belonging to any rigid body

      if (!distributed) {
        if (surfbody[m] >= 0) continue;
      } else {
        surfint id = (dim == 2) ? lines[m].id : tris[m].id;
        if (fix->body_elem(id) >= 0) continue;
      }

      if (dim == 2) {
        for (k = 0; k < 2; k++) {
          slo[k] = MIN(lines[m].p1[k],lines[m].p2[k]);
          shi[k] = MAX(lines[m].p1[k],lines[m].p2[k]);
        }
        slo[2] = shi[2] = 0.0;
      } else {
        for (k = 0; k < 3; k++) {
          slo[k] = MIN(tris[m].p1[k],MIN(tris[m].p2[k],tris[m].p3[k]));
          shi[k] = MAX(tris[m].p1[k],MAX(tris[m].p2[k],tris[m].p3[k]));
        }
      }

      for (k = 0; k < 3; k++) {
        blo[k] = (int) ((slo[k]-binlo[k]) * bininv[k]);
        bhi[k] = (int) ((shi[k]-binlo[k]) * bininv[k]);
        blo[k] = MAX(0,MIN(blo[k],nbin[k]-1));
        bhi[k] = MAX(0,MIN(bhi[k],nbin[k]-1));
      }

      for (ibz = blo[2]; ibz <= bhi[2]; ibz++)
        for (iby = blo[1]; iby <= bhi[1]; iby++)
          for (ibx = blo[0]; ibx <= bhi[0]; ibx++) {
            int ibin = (ibz*nbin[1] + iby)*nbin[0] + ibx;
            if (pass == 0) binstart[ibin+1]++;
            else binlist[binstart[ibin]++] = m;
          }
    }

    if (pass == 0) {
      for (i = 0; i < nbins; i++) binstart[i+1] += binstart[i];
      memory->create(binlist,binstart[nbins],"rigid_contact:binlist");
    } else {
      // filling advanced the starts by one bin: shift them back
      for (i = nbins; i > 0; i--) binstart[i] = binstart[i-1];
      binstart[0] = 0;
    }
  }
}

/* ----------------------------------------------------------------------
   push-off force and torque of the bodies this proc owns into
     fpush/tqpush; a contact also writes the partner's row
   called after all bodies have committed their end-of-step geometry
   non-distributed surfs: computed identically on every proc, so no
     communication is needed
   distributed surfs: the static-surf contributions are disjoint per-proc
     partial sums (each proc handles the static surfs it owns), and
     proc 0 alone computes the body-body and boundary contributions, so
     that one Allreduce sums them once and in the same order as a
     non-distributed run, which computes them on every proc
------------------------------------------------------------------------- */

void RigidContact::compute(double **fpush, double **tqpush)
{
  int ibody;
  int nbody = fix->nbody;

  for (ibody = 0; ibody < nbody; ibody++) {
    fpush[ibody][0] = fpush[ibody][1] = fpush[ibody][2] = 0.0;
    tqpush[ibody][0] = tqpush[ibody][1] = tqpush[ibody][2] = 0.0;
  }

  for (int m = 0; m < fix->nown; m++) body(fix->ownlist[m],fpush,tqpush);

  if (!surf->distributed) return;

  for (ibody = 0; ibody < nbody; ibody++) {
    buf_mine[6*ibody]   = fpush[ibody][0];
    buf_mine[6*ibody+1] = fpush[ibody][1];
    buf_mine[6*ibody+2] = fpush[ibody][2];
    buf_mine[6*ibody+3] = tqpush[ibody][0];
    buf_mine[6*ibody+4] = tqpush[ibody][1];
    buf_mine[6*ibody+5] = tqpush[ibody][2];
  }
  MPI_Allreduce(buf_mine,buf_all,6*nbody,MPI_DOUBLE,MPI_SUM,world);
  for (ibody = 0; ibody < nbody; ibody++) {
    fpush[ibody][0] = buf_all[6*ibody];
    fpush[ibody][1] = buf_all[6*ibody+1];
    fpush[ibody][2] = buf_all[6*ibody+2];
    tqpush[ibody][0] = buf_all[6*ibody+3];
    tqpush[ibody][1] = buf_all[6*ibody+4];
    tqpush[ibody][2] = buf_all[6*ibody+5];
  }
}

/* ----------------------------------------------------------------------
   push-off forces on body ibody from too-close static surfs, other
     rigid bodies, and (if boundflag) non-periodic box boundaries
   static surf candidates come from the bins built by setup();
     other bodies are pruned by a body-body bbox test, then per element
   NOTE: a corner pt shared by adjacent body elements contributes once
     per element, and a corner close to several source elements
     interacts with each of them, so kpush is a per-contact stiffness;
     two bodies engage the corner pts of each against the elements of
     the other, about twice the contacts of one body against a static
     surf of the same shape (each set is a distinct geometric contact,
     and dropping either would make the force depend on body order)
------------------------------------------------------------------------- */

void RigidContact::body(int ibody, double **fpush, double **tqpush)
{
  int i,j,m,e,jbody;
  double d,scale;
  double **pts;
  double fone[3],rdelta[3],tq[3];
  int blo[3],bhi[3];

  fix->host_geometry(ibody);

  double *xcm1 = fix->xcm[ibody];
  double *fpush1 = fpush[ibody];
  double *tqpush1 = tqpush[ibody];
  int *bodystart = fix->bodystart;
  double ***bodypt = fix->bodypt;
  double **bodynorm = fix->bodynorm;

  int distributed = surf->distributed;
  Surf::Line *lines;
  Surf::Tri *tris;
  if (!distributed) {
    lines = surf->lines;
    tris = surf->tris;
  } else {
    lines = surf->mylines;
    tris = surf->mytris;
  }

  int npoint = dim;     // 2 corner pts per line, 3 per tri

  // cutlo/cuthi = bbox around body inflated by the cutoff
  // requires FixRigid::body_bbox() was called for the current position

  double cutlo[3],cuthi[3];
  for (j = 0; j < 3; j++) {
    cutlo[j] = fix->bbodylo[ibody][j] - cutoff;
    cuthi[j] = fix->bbodyhi[ibody][j] + cutoff;
  }

  // static surf candidates: bins overlapping the inflated body bbox
  // stamp dedups surfs binned into more than one of the bins

  for (j = 0; j < 3; j++) {
    blo[j] = (int) ((cutlo[j]-binlo[j]) * bininv[j]);
    bhi[j] = (int) ((cuthi[j]-binlo[j]) * bininv[j]);
    blo[j] = MAX(0,MIN(blo[j],nbin[j]-1));
    bhi[j] = MAX(0,MIN(bhi[j],nbin[j]-1));
  }

  stampcur++;

  for (int ibz = blo[2]; ibz <= bhi[2]; ibz++)
    for (int iby = blo[1]; iby <= bhi[1]; iby++)
      for (int ibx = blo[0]; ibx <= bhi[0]; ibx++) {
        int ibin = (ibz*nbin[1] + iby)*nbin[0] + ibx;
        for (i = binstart[ibin]; i < binstart[ibin+1]; i++) {
          m = binlist[i];
          if (stamp[m] == stampcur) continue;
          stamp[m] = stampcur;

          if (dim == 2) {
            if (MAX(lines[m].p1[0],lines[m].p2[0]) < cutlo[0]) continue;
            if (MIN(lines[m].p1[0],lines[m].p2[0]) > cuthi[0]) continue;
            if (MAX(lines[m].p1[1],lines[m].p2[1]) < cutlo[1]) continue;
            if (MIN(lines[m].p1[1],lines[m].p2[1]) > cuthi[1]) continue;
            contact(ibody,lines[m].p1,lines[m].p2,NULL,
                    lines[m].norm,-1,fpush,tqpush);
          } else {
            if (MAX(tris[m].p1[0],MAX(tris[m].p2[0],tris[m].p3[0])) <
                cutlo[0]) continue;
            if (MIN(tris[m].p1[0],MIN(tris[m].p2[0],tris[m].p3[0])) >
                cuthi[0]) continue;
            if (MAX(tris[m].p1[1],MAX(tris[m].p2[1],tris[m].p3[1])) <
                cutlo[1]) continue;
            if (MIN(tris[m].p1[1],MIN(tris[m].p2[1],tris[m].p3[1])) >
                cuthi[1]) continue;
            if (MAX(tris[m].p1[2],MAX(tris[m].p2[2],tris[m].p3[2])) <
                cutlo[2]) continue;
            if (MIN(tris[m].p1[2],MIN(tris[m].p2[2],tris[m].p3[2])) >
                cuthi[2]) continue;
            contact(ibody,tris[m].p1,tris[m].p2,tris[m].p3,
                    tris[m].norm,-1,fpush,tqpush);
          }
        }
      }

  // other rigid bodies: those whose bbox overlaps the inflated body
  //   bbox, from the body bins, then per-element bbox tests using the
  //   current-position element boxes set by FixRigid::body_bbox() when
  //   each body committed its end-of-step geometry
  // each contact applies equal-and-opposite forces to both bodies
  // distributed: proc 0 alone computes the pair and boundary
  //   contributions, see compute()
  // the bins keep this proc-0 work small: a body only tests the bodies
  //   in its neighboring bins

  int mine = 1;
  if (distributed && comm->me) mine = 0;

  if (mine) {
    int *jlist;
    int nj = fix->body_box(cutlo,cuthi,&jlist);
    double **elemlo = fix->elemlo;
    double **elemhi = fix->elemhi;

    for (int jj = 0; jj < nj; jj++) {
      jbody = jlist[jj];
      fix->host_geometry(jbody);
      if (jbody == ibody) continue;

      for (e = bodystart[jbody]; e < bodystart[jbody+1]; e++) {
        if (!box_overlap(cutlo,cuthi,elemlo[e],elemhi[e])) continue;
        if (dim == 2)
          contact(ibody,bodypt[e][0],bodypt[e][1],NULL,
                  bodynorm[e],jbody,fpush,tqpush);
        else
          contact(ibody,bodypt[e][0],bodypt[e][1],bodypt[e][2],
                  bodynorm[e],jbody,fpush,tqpush);
      }
    }
  }

  // spring force from non-periodic simulation box boundaries
  // corner pts from the replicated body geometry

  if (boundflag && mine) {
    double *boxlo = domain->boxlo;
    double *boxhi = domain->boxhi;
    int *bflag = domain->bflag;

    int nface = 2*dim;
    double fsign[6] = {1.0,-1.0,1.0,-1.0,1.0,-1.0};

    for (i = bodystart[ibody]; i < bodystart[ibody+1]; i++) {
      pts = bodypt[i];

      for (j = 0; j < npoint; j++) {
        for (int iface = 0; iface < nface; iface++) {
          if (bflag[iface] == PERIODIC) continue;

          // the axisymmetric axis is not a wall: a body of revolution
          //   is expected to reach r = 0, and pushing it off the axis
          //   would break the symmetry it is built on

          if (bflag[iface] == AXISYM) continue;

          int idim = iface/2;
          if (iface % 2 == 0) d = pts[j][idim] - boxlo[idim];
          else d = boxhi[idim] - pts[j][idim];
          if (d >= cutoff) continue;
          if (d < 0.0) d = 0.0;      // past the face: max force k*cutoff

          if (style == LINEAR) scale = kpush * (cutoff-d);
          else scale = kpush * (cutoff-d) * sqrt(cutoff-d);

          // dashpot vs the static boundary, face normal = fsign*e_idim

          if (gamma > 0.0) {
            double vpt[3],rd[3];
            MathExtra::sub3(pts[j],xcm1,rd);
            MathExtra::cross3(fix->omega[ibody],rd,vpt);
            MathExtra::add3(fix->vcm[ibody],vpt,vpt);
            scale -= gamma * fsign[iface] * vpt[idim];
            if (scale < 0.0) scale = 0.0;
          }

          scale *= fsign[iface];
          if (axiflag) scale *= MY_2PI * pts[j][1];
          fone[0] = fone[1] = fone[2] = 0.0;
          fone[idim] = scale;

          fpush1[0] += fone[0];
          fpush1[1] += fone[1];
          fpush1[2] += fone[2];
          MathExtra::sub3(pts[j],xcm1,rdelta);
          MathExtra::cross3(rdelta,fone,tq);
          tqpush1[0] += tq[0];
          tqpush1[1] += tq[1];
          tqpush1[2] += tq[2];
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   contact forces between all corner pts of body ibody and one source
     element with corner pts p1,p2 (p3 for 3d) and outward normal norm
   for each body corner pt within the cutoff of the element, apply a
     repulsive force directed from the closest point of the element to
     the corner pt (the element outward normal for a face-on contact),
     with overlap delta = cutoff - dist:
     linear spring F = kpush * delta, or
     Hertzian contact F = kpush * delta^3/2 (smooth onset, standard
     model for elastic contact of spherical particulates)
   every element within the cutoff contributes, on either side of it:
     each force is the gradient of its own spring potential in the
     distance to the element, so the total is the gradient of the sum
     and the contacts conserve energy exactly
   a corner pt inside a wall thinner than 2*cutoff is repelled by
     both faces at once; their potentials add to a barrier whose peak
     is at the near surface, so the wall repels the body rather than
     driving it through, and the body passes only if it arrives with
     more energy than the barrier
   if gamma > 0, a dashpot term F += gamma * d(delta)/dt is added (the
     DEM spring-dashpot pair), i.e. minus gamma times the normal
     separation rate of the corner pt relative to the source surface;
     the total contact force is clamped at zero, so the dashpot never
     produces adhesion as a contact ends
   jbody = the rigid body the element belongs to, or -1 if static
   if jbody >= 0, the reaction force -F is applied to it at the same
     contact point, so body-body contacts conserve momentum exactly
------------------------------------------------------------------------- */

void RigidContact::contact(int ibody, double *p1, double *p2, double *p3,
                           double *norm, int jbody,
                           double **fpush, double **tqpush)
{
  int i,j;
  double dsq,d,scale;
  double **pts;
  double fone[3],rdelta[3],tq[3],cp[3],fdir[3];

  int npoint = dim;     // 2 corner pts per line, 3 per tri
  double cutsq = cutoff*cutoff;

  int *bodystart = fix->bodystart;
  double ***bodypt = fix->bodypt;
  double **elemlo = fix->elemlo;
  double **elemhi = fix->elemhi;

  double *xcm1 = fix->xcm[ibody];
  double *fpush1 = fpush[ibody];
  double *tqpush1 = tqpush[ibody];

  // bounding box of the source element inflated by the cutoff:
  //   only body elements whose own box overlaps it can be in contact

  double slo[3],shi[3];
  for (int k = 0; k < 3; k++) {
    slo[k] = MIN(p1[k],p2[k]);
    shi[k] = MAX(p1[k],p2[k]);
    if (dim == 3) {
      slo[k] = MIN(slo[k],p3[k]);
      shi[k] = MAX(shi[k],p3[k]);
    }
    slo[k] -= cutoff;
    shi[k] += cutoff;
  }

  for (i = bodystart[ibody]; i < bodystart[ibody+1]; i++) {
    if (elemlo[i][0] > shi[0] || elemhi[i][0] < slo[0]) continue;
    if (elemlo[i][1] > shi[1] || elemhi[i][1] < slo[1]) continue;
    if (dim == 3 && (elemlo[i][2] > shi[2] || elemhi[i][2] < slo[2]))
      continue;
    pts = bodypt[i];

    for (j = 0; j < npoint; j++) {

      if (dim == 2)
        dsq = Geometry::closest_point_line(pts[j],p1,p2,cp);
      else
        dsq = Geometry::closest_point_tri(pts[j],p1,p2,p3,norm,cp);
      if (dsq >= cutsq) continue;

      d = sqrt(dsq);
      if (style == LINEAR) scale = kpush * (cutoff-d);
      else scale = kpush * (cutoff-d) * sqrt(cutoff-d);

      // force direction = from the closest point of the element to the
      //   corner pt, the gradient of the spring potential in d: equals
      //   the element normal when the closest feature is the interior,
      //   and stays conservative when it is an edge or vertex, as it is
      //   for most contacts with a faceted curved surface
      // a corner pt on the element (d = 0) is pushed along the normal

      if (d > 0.0) {
        fdir[0] = (pts[j][0]-cp[0]) / d;
        fdir[1] = (pts[j][1]-cp[1]) / d;
        fdir[2] = (pts[j][2]-cp[2]) / d;
      } else {
        fdir[0] = norm[0]; fdir[1] = norm[1]; fdir[2] = norm[2];
      }

      // dashpot: damp by the normal approach rate of the corner pt
      //   relative to the source surface,
      //   which moves if it belongs to another rigid body

      if (gamma > 0.0) {
        double vpt[3],vsrc[3],rd[3];
        MathExtra::sub3(pts[j],xcm1,rd);
        MathExtra::cross3(fix->omega[ibody],rd,vpt);
        MathExtra::add3(fix->vcm[ibody],vpt,vpt);
        if (jbody >= 0) {
          MathExtra::sub3(pts[j],fix->xcm[jbody],rd);
          MathExtra::cross3(fix->omega[jbody],rd,vsrc);
          MathExtra::add3(fix->vcm[jbody],vsrc,vsrc);
          MathExtra::sub3(vpt,vsrc,vpt);
        }
        scale -= gamma * MathExtra::dot3(vpt,fdir);
        if (scale < 0.0) scale = 0.0;
      }

      // axisymmetric: the corner pt stands for a ring of radius r and
      //   the source element for another, so the spring law gives a
      //   force per unit length of contact and the total is 2 pi r
      //   times it.  a pt on the axis then feels no push, which is
      //   right: its ring has no circumference.  the force stays in the
      //   (x,r) plane, so a contact exerts no torque about the axis and
      //   cannot spin the body

      if (axiflag) scale *= MY_2PI * pts[j][1];

      fone[0] = scale*fdir[0];
      fone[1] = scale*fdir[1];
      fone[2] = scale*fdir[2];

      fpush1[0] += fone[0];
      fpush1[1] += fone[1];
      fpush1[2] += fone[2];
      MathExtra::sub3(pts[j],xcm1,rdelta);
      MathExtra::cross3(rdelta,fone,tq);
      tqpush1[0] += tq[0];
      tqpush1[1] += tq[1];
      tqpush1[2] += tq[2];

      // equal-and-opposite reaction on the source body,
      //   applied at the same contact point

      if (jbody >= 0) {
        fpush[jbody][0] -= fone[0];
        fpush[jbody][1] -= fone[1];
        fpush[jbody][2] -= fone[2];
        MathExtra::sub3(pts[j],fix->xcm[jbody],rdelta);
        MathExtra::cross3(rdelta,fone,tq);
        tqpush[jbody][0] -= tq[0];
        tqpush[jbody][1] -= tq[1];
        tqpush[jbody][2] -= tq[2];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

double RigidContact::memory_usage()
{
  double bytes = 0.0;
  if (binstart) {
    int nbins = nbin[0]*nbin[1]*nbin[2];
    bytes += (double) (nbins+1) * sizeof(int);
    bytes += (double) binstart[nbins] * sizeof(int);
    if (surf->distributed) bytes += (double) surf->nown * sizeof(int);
    else bytes += (double) surf->nlocal * sizeof(int);
  }
  bytes += (double) 12 * fix->nbody * sizeof(double);
  return bytes;
}
