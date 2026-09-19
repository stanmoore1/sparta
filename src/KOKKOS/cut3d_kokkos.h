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

#ifndef SPARTA_CUT3D_KOKKOS_H
#define SPARTA_CUT3D_KOKKOS_H

#include "kokkos_type.h"
#include "surf.h"
#include "cut2d_kokkos.h"

namespace SPARTA_NS {

/* ----------------------------------------------------------------------
   device twin of Cut3d::split(): the cut of one 3d grid cell by the
     triangles of its cut list, the same arithmetic in the same order as
     the host cut, so a cell cut on the device gets the flow volumes,
     corner marks, piece map and split point the host cut gives it
   one instance per thread, built on the cell and on scratch rows the
     caller sized; Cut3d grows its vectors as it goes, so the rows carry
     capacities and an overflow is an error (9) the caller resolves by
     cutting the cell on the host; the bounds below hold for every cell
     the host cut handles, so the host cut is reached only for a cell
     the host cut would fail on too
   for a cell of n triangles:
     verts    <= 10n+6    n tris, one polygon per clipped edge on a
                         face at most, 6 whole faces
     edges    <= 27n+24   3n tri edges, one gap edge per tri per face
                         plane, then one face edge per 2d point
     clines   <= 9n       every singlet edge of one face
     points   <= 18n+4    2 per cline + 4 corners (Cut2d)
     loops, polyhedra, walk flags and stack <= verts; face lists <= edges;
     2d loops, polygons and flags <= points
   the error returns of the host cut come back as its errflag values
     (see Cut3d::split_error(); 8 = more pieces than surfs, which
     cannot happen); the caller re-runs the host cut on such a cell,
     which prints the cell and raises the message
   the implicit-surf split point is not ported: fix rigid rejects
     implicit surfs
------------------------------------------------------------------------- */

struct Cut3dKokkos {
  struct Vertex {
    int active;      // 1/0 if active or not
    int style;       // CTRI or CTRIFACE or FACEPGON or FACE
    int label;       // index in list of tris that intersect this cell
                     //   for CTRI or CTRIFACE
                     // face index (0-5) for FACEPGON or FACE
    int next;        // index of next vertex when walking a loop
    int nedge;       // # of edges in this vertex
    int first;       // first edge in vertex
    int dirfirst;    // dir of first edge in vertex
    int last;        // last edge in vertex
    int dirlast;     // dir of last edge in vertex
    double volume;   // volume of vertex projected against lower z face of cell
    const double *norm;  // ptr to norm of tri, NULL for other styles
  };

  struct Edge {
    double p1[3],p2[3];  // 2 points in edge
    int active;          // 1/0 if active or not
    int style;           // CTRI or CTRIFACE or FACEPGON or FACE
    int clipped;         // 1/0 if already clipped during face iteration
    int nvert;           // flag for verts containing this edge
                         // 0 = no verts
                         // 1 = just 1 vert in forward dir
                         // 2 = just 1 vert in reverse dir
                         // 3 = 2 verts in both dirs
                         // all vecs are [0] in forward dir, [1] in reverse dir
    int verts[2];        // index of vertices containing this edge, -1 if not
    int next[2];         // index of next edge for each vertex, -1 for end
    int dirnext[2];      // next edge for each vertex is forward/reverse (0,1)
    int prev[2];         // index of prev edge for each vertex, -1 for start
    int dirprev[2];      // prev edge for each vertex is forward/reverse (0,1)
  };

  struct Loop {
    double volume;        // volume of loop
    int flag;             // INTERIOR (if all CTRI vertices) or BORDER
    int n;                // # of vertices in loop
    int first;            // index of first vertex in loop
    int next;             // index of next loop in same PH, -1 if last loop
  };

  struct PH {
    double volume;
    int n;
    int first;
  };

  enum{UNKNOWN,OUTSIDE,INSIDE,OVERLAP};     // same as Cut3d
  enum{CTRI,CTRIFACE,FACEPGON,FACE};
  enum{EXTERIOR,INTERIOR,BORDER};

  static constexpr double EPSEDGE = 1.0e-9;  // minimum edge length (fraction of cell size)
  static constexpr double SHRINK = 1.0e-8;   // shrink grid cell by this fraction when split fails

  const Surf::Tri *tris;     // the surf tris on the device

  double lo[3],hi[3];        // opposite corner pts of cell being worked on
  int nsurf;                 // # of surf elements in cell
  const int *surfs;          // local indices of surf elements in cell

  int grazecount;            // count of tris that graze cell surf w/ outward norm
  int touchcount;            // count of tris that only touch cell surf
  int touchmark;             // corner marking inferred by touching tris
  double epsilon;            // epsilon size for this cell
  int empty;
  int ntiny,nshrink;         // counts for this cell, summed by the caller

  double path1[12][3],path2[12][3];

  Vertex *verts;             // scratch rows and their capacities
  Edge *edges;
  Loop *loops;
  PH *phs;
  int *facelist;             // edges on each cell face, rows of one array
  int facestart[7];
  int *efaces;               // face of each singlet edge, work array
  int *used;                 // 0/1 flag for each vertex when walking loops
  int *stack;                // list of vertices to check when walking loops
  int nverts,nedges,nloops,nphs;
  int maxverts,maxedges,maxclines;

  double lo2d[2],hi2d[2];    // the face the 2d cut works on
  Cut2dKokkos cut2d;

  KOKKOS_INLINE_FUNCTION
  Cut3dKokkos(const Surf::Tri *tris_in,
              const double *lo_in, const double *hi_in,
              int nsurf_in, const int *surfs_in,
              Vertex *verts_in, Edge *edges_in, Loop *loops_in, PH *phs_in,
              int *facelist_in, int *efaces_in, int *used_in, int *stack_in,
              int maxverts_in, int maxedges_in,
              Cut2dKokkos::Cline *clines_in, Cut2dKokkos::Point *points_in,
              Cut2dKokkos::Loop *loops2d_in, Cut2dKokkos::PG *pgs_in,
              int *used2d_in, int maxclines_in) :
    tris(tris_in), nsurf(nsurf_in), surfs(surfs_in),
    grazecount(0), touchcount(0), touchmark(UNKNOWN), epsilon(0.0),
    empty(0), ntiny(0), nshrink(0),
    verts(verts_in), edges(edges_in), loops(loops_in), phs(phs_in),
    facelist(facelist_in), efaces(efaces_in), used(used_in), stack(stack_in),
    nverts(0), nedges(0), nloops(0), nphs(0),
    maxverts(maxverts_in), maxedges(maxedges_in), maxclines(maxclines_in),
    cut2d(NULL,0,lo2d,hi2d,0,NULL,clines_in,points_in,loops2d_in,pgs_in,
          used2d_in)
  {
    lo[0] = lo_in[0]; lo[1] = lo_in[1]; lo[2] = lo_in[2];
    hi[0] = hi_in[0]; hi[1] = hi_in[1]; hi[2] = hi_in[2];
  }

  /* --------------------------------------------------------------------
     Cut3d::split(): cut volume and pieces of the cell
     return nsplit = # of pieces, 1 for no split, 0 on an error with
       errflag = the host's error code
     vols = one flow volume per piece, in a row of nsurf
     corners = UNKNOWN/INSIDE/OUTSIDE for each of 8 corner pts
     if nsplit > 1: surfmap = piece of each surf, -1 if in none;
       xsplit = a point in piece xsub
     work is done by split_try(), called once more with a shrunk cell
       if the first attempt fails
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int split(double *vols, int *surfmap, int *corners,
            int &xsub, double *xsplit, int &errflag)
  {
    int nsplit = 0;

    errflag = split_try(vols,surfmap,corners,xsub,xsplit,nsplit);

    // error return
    // try again after shrinking grid cell by SHRINK factor
    // this gets rid of pesky errors due to tri pts/edges on cell faces

    if (errflag) {
      nshrink++;

      double newlo = lo[0] + SHRINK*(hi[0]-lo[0]);
      double newhi = hi[0] - SHRINK*(hi[0]-lo[0]);
      lo[0] = newlo;
      hi[0] = newhi;

      newlo = lo[1] + SHRINK*(hi[1]-lo[1]);
      newhi = hi[1] - SHRINK*(hi[1]-lo[1]);
      lo[1] = newlo;
      hi[1] = newhi;

      newlo = lo[2] + SHRINK*(hi[2]-lo[2]);
      newhi = hi[2] - SHRINK*(hi[2]-lo[2]);
      lo[2] = newlo;
      hi[2] = newhi;

      errflag = split_try(vols,surfmap,corners,xsub,xsplit,nsplit);
    }

    if (errflag) return 0;
    return nsplit;
  }

  /* --------------------------------------------------------------------
     Cut3d::split_try(): attempt to split cell
     return 0 if successful, otherwise return an error flag
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int split_try(double *vols, int *surfmap, int *corners,
                int &xsub, double *xsplit, int &nsplit)
  {
    int errflag = add_tris();
    if (errflag) return errflag;

    errflag = clip_tris();
    if (errflag) return errflag;
    clip_adjust();

    // all triangles just touched cell surface
    // mark corner points based on grazecount or touchmark value
    // return vol = 0.0 for UNKNOWN/INSIDE, full cell vol for OUTSIDE

    if (empty) {
      int mark = UNKNOWN;
      if (grazecount || touchmark == INSIDE) mark = INSIDE;
      else if (touchmark == OUTSIDE) mark = OUTSIDE;
      for (int i = 0; i < 8; i++) corners[i] = mark;

      double vol = 0.0;
      if (mark == OUTSIDE) vol = (hi[0]-lo[0]) * (hi[1]-lo[1]) * (hi[2]-lo[2]);

      vols[0] = vol;
      nsplit = 1;
      return 0;
    }

    ctri_volume();
    errflag = edge2face();
    if (errflag) return errflag;

    for (int iface = 0; iface < 6; iface++) {
      if (facestart[iface+1] > facestart[iface]) {
        face_from_cell(iface);
        errflag = edge2clines(iface);
        if (errflag) return errflag;
        errflag = cut2d.split_face();
        if (errflag) return errflag;
        errflag = add_face_pgons(iface);
        if (errflag) return errflag;
      } else {
        face_from_cell(iface);
        errflag = add_face(iface);
        if (errflag) return errflag;
      }
    }

    remove_faces();

    errflag = check();
    if (errflag) return errflag;

    walk();

    errflag = loop2ph();

    // loop2ph detected no positive-volume loops, cell is inside the surf

    if (errflag == 4) {
      for (int i = 0; i < 8; i++) corners[i] = INSIDE;
      vols[0] = 0.0;
      nsplit = 1;
      return 0;
    }

    if (errflag) return errflag;

    // if multiple splits, find a split point

    nsplit = nphs;
    if (nsplit > 1) {
      create_surfmap(surfmap);
      errflag = split_point_explicit(surfmap,xsplit,xsub);
      if (errflag) return errflag;
    }

    // successful cut/split
    // set corners = OUTSIDE if corner pt is in list of edge points
    // else set corners = INSIDE

    int icorner;

    for (int i = 0; i < 8; i++) corners[i] = INSIDE;

    for (int iedge = 0; iedge < nedges; iedge++) {
      if (!edges[iedge].active) continue;
      icorner = corner(edges[iedge].p1);
      if (icorner >= 0) corners[icorner] = OUTSIDE;
      icorner = corner(edges[iedge].p2);
      if (icorner >= 0) corners[icorner] = OUTSIDE;
    }

    // a piece holds at least one tri of its own, so nsplit <= nsurf and
    //   the volumes fit a row of nsurf; anything else is an error (8)

    if (nsplit > nsurf) return 8;
    for (int i = 0; i < nsplit; i++) vols[i] = phs[i].volume;

    return 0;
  }

  /* --------------------------------------------------------------------
     Cut3d::clip(): Sutherland-Hodgman clipping of tri P0 P1 P2 against
       the cell; return # of clipped points, left in path1
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int clip(const double *p0, const double *p1, const double *p2)
  {
    int i,npath,nnew;
    double value;
    double *s,*e;
    double (*path)[3];
    double (*newpath)[3];

    // initial path = tri vertices

    nnew = 3;
    for (i = 0; i < 3; i++) {
      path1[0][i] = p0[i];
      path1[1][i] = p1[i];
      path1[2][i] = p2[i];
    }

    // intersect if any of tri vertices is within grid cell

    if (p0[0] >= lo[0] && p0[0] <= hi[0] &&
        p0[1] >= lo[1] && p0[1] <= hi[1] &&
        p0[2] >= lo[2] && p0[2] <= hi[2] &&
        p1[0] >= lo[0] && p1[0] <= hi[0] &&
        p1[1] >= lo[1] && p1[1] <= hi[1] &&
        p1[2] >= lo[2] && p1[2] <= hi[2] &&
        p2[0] >= lo[0] && p2[0] <= hi[0] &&
        p2[1] >= lo[1] && p2[1] <= hi[1] &&
        p2[2] >= lo[2] && p2[2] <= hi[2]) return 1;

    // clip tri against each of 6 grid face planes

    for (int dim = 0; dim < 3; dim++) {
      path = path1;
      newpath = path2;
      npath = nnew;
      nnew = 0;

      value = lo[dim];
      s = path[npath-1];
      for (i = 0; i < npath; i++) {
        e = path[i];
        if (e[dim] >= value) {
          if (s[dim] < value) between(s,e,dim,value,newpath[nnew++]);
          copy3(e,newpath[nnew++]);
        } else if (s[dim] >= value) between(e,s,dim,value,newpath[nnew++]);
        s = e;
      }
      if (!nnew) return 0;

      path = path2;
      newpath = path1;
      npath = nnew;
      nnew = 0;

      value = hi[dim];
      s = path[npath-1];
      for (i = 0; i < npath; i++) {
        e = path[i];
        if (e[dim] <= value) {
          if (s[dim] > value) between(s,e,dim,value,newpath[nnew++]);
          copy3(e,newpath[nnew++]);
        } else if (s[dim] <= value) between(e,s,dim,value,newpath[nnew++]);
        s = e;
      }
      if (!nnew) return 0;
    }

    return nnew;
  }

  /* --------------------------------------------------------------------
     Cut3d::add_tris(): add each triangle as vertex and edges to BPG
     add full edge even if outside cell, clipping comes later
     skip transparent surfs
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int add_tris()
  {
    int i,m;
    int e1,e2,e3,dir1,dir2,dir3;
    const double *p1,*p2,*p3;
    const Surf::Tri *tri;
    Vertex *vert;
    Edge *edge;

    nverts = 0;
    nedges = 0;

    int nvert = 0;
    for (i = 0; i < nsurf; i++) {
      m = surfs[i];
      tri = &tris[m];
      if (tri->transparent) continue;

      p1 = tri->p1;
      p2 = tri->p2;
      p3 = tri->p3;

      if (nvert >= maxverts) return 9;
      vert = &verts[nvert];
      vert->active = 1;
      vert->style = CTRI;
      vert->label = i;
      vert->nedge = 0;
      vert->volume = 0.0;
      vert->norm = tri->norm;

      // look for each edge of tri
      // add to edges in forward dir if doesn't yet exist
      // add to edges in reverse dir if already exists

      e1 = findedge(p1,p2,0,dir1);
      if (e1 == -2) return 1;

      if (e1 < 0) {
        if (nedges >= maxedges) return 9;
        e1 = nedges++;
        dir1 = 0;
        edge = &edges[e1];
        edge->style = CTRI;
        edge->nvert = 0;
        edge->verts[0] = edge->verts[1] = -1;
        copy3(p1,edge->p1);
        copy3(p2,edge->p2);
      }
      edge_insert(e1,dir1,nvert,-1,-1,-1,-1);

      e2 = findedge(p2,p3,0,dir2);
      if (e2 == -2) return 1;

      if (e2 < 0) {
        if (nedges >= maxedges) return 9;
        e2 = nedges++;
        dir2 = 0;
        edge = &edges[e2];
        edge->style = CTRI;
        edge->nvert = 0;
        edge->verts[0] = edge->verts[1] = -1;
        copy3(p2,edge->p1);
        copy3(p3,edge->p2);
      }
      edge_insert(e2,dir2,nvert,e1,dir1,-1,-1);

      e3 = findedge(p3,p1,0,dir3);
      if (e3 == -2) return 1;

      if (e3 < 0) {
        if (nedges >= maxedges) return 9;
        e3 = nedges++;
        dir3 = 0;
        edge = &edges[e3];
        edge->style = CTRI;
        edge->nvert = 0;
        edge->verts[0] = edge->verts[1] = -1;
        copy3(p3,edge->p1);
        copy3(p1,edge->p2);
      }
      edge_insert(e3,dir3,nvert,e2,dir2,-1,-1);

      nvert++;
    }

    nverts = nvert;
    return 0;
  }

  /* --------------------------------------------------------------------
     Cut3d::clip_tris(): clip collection of tris that overlap cell by
       6 faces of cell
     edges fully outside the cell are removed
     shared edges that intersect the cell are clipped consistently
     return 9 if the edge row overflows
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int clip_tris()
  {
    int i,n,dim,lohi,ivert,iedge,jedge,idir,jdir,nedge;
    int p1flag,p2flag;
    double value;
    double *p1,*p2;
    Edge *edge,*newedge;

    // loop over all 6 faces of cell

    int nvert = nverts;

    for (int iface = 0; iface < 6; iface++) {
      dim = iface / 2;
      lohi = iface % 2;
      if (lohi == 0) value = lo[dim];
      else value = hi[dim];

      // mark all edges as unclipped
      // some may have been clipped and not cleared on previous face

      nedge = nedges;
      for (iedge = 0; iedge < nedge; iedge++)
        if (edges[iedge].active) edges[iedge].clipped = 0;

      // loop over vertices, clip each of its edges to face
      // if edge already clipped, unset clip flag and keep edge as-is

      for (ivert = 0; ivert < nvert; ivert++) {
        iedge = verts[ivert].first;
        idir = verts[ivert].dirfirst;
        nedge = verts[ivert].nedge;

        for (i = 0; i < nedge; i++) {
          edge = &edges[iedge];

          if (edge->clipped) {
            edge->clipped = 0;
            iedge = edge->next[idir];
            idir = edge->dirnext[idir];
            continue;
          }

          // p1/p2 are pts in order of traversal

          if (idir == 0) {
            p1 = edge->p1;
            p2 = edge->p2;
          } else {
            p1 = edge->p2;
            p2 = edge->p1;
          }

          // p1/p2 flag = OUTSIDE/ON/INSIDE for edge pts

          if (lohi == 0) {
            if (p1[dim] < value) p1flag = OUTSIDE;
            else if (p1[dim] > value) p1flag = INSIDE;
            else p1flag = OVERLAP;
            if (p2[dim] < value) p2flag = OUTSIDE;
            else if (p2[dim] > value) p2flag = INSIDE;
            else p2flag = OVERLAP;
          } else {
            if (p1[dim] < value) p1flag = INSIDE;
            else if (p1[dim] > value) p1flag = OUTSIDE;
            else p1flag = OVERLAP;
            if (p2[dim] < value) p2flag = INSIDE;
            else if (p2[dim] > value) p2flag = OUTSIDE;
            else p2flag = OVERLAP;
          }

          // if both OUTSIDE or one OUTSIDE and other ON, delete edge
          // if both INSIDE or one INSIDE and other ON or both ON, keep as-is
          // if one INSIDE and one OUTSIDE, replace OUTSIDE pt with clip pt

          if (p1flag == OUTSIDE) {
            if (p2flag == OUTSIDE || p2flag == OVERLAP) edge_remove(edge,idir);
            else {
              if (idir == 0) between(p1,p2,dim,value,edge->p1);
              else between(p1,p2,dim,value,edge->p2);
              edge->clipped = 1;
            }
          } else if (p1flag == INSIDE) {
            if (p2flag == OUTSIDE) {
              if (idir == 0) between(p1,p2,dim,value,edge->p2);
              else between(p1,p2,dim,value,edge->p1);
              edge->clipped = 1;
            }
          } else {
            if (p2flag == OUTSIDE) edge_remove(edge,idir);
          }

          iedge = edge->next[idir];
          idir = edge->dirnext[idir];
        }

        // loop over edges in vertex again
        // iedge = this edge, jedge = next edge
        // p1 = last pt in iedge, pt = first pt in jedge
        // if p1 != p2, add edge between them

        iedge = verts[ivert].first;
        idir = verts[ivert].dirfirst;

        for (i = 0; i < verts[ivert].nedge; i++) {
          edge = &edges[iedge];
          jedge = edge->next[idir];
          jdir = edge->dirnext[idir];
          if (jedge < 0) {
            jedge = verts[ivert].first;
            jdir = verts[ivert].dirfirst;
          }

          if (idir == 0) p1 = edge->p2;
          else p1 = edge->p1;
          if (jdir == 0) p2 = edges[jedge].p1;
          else p2 = edges[jedge].p2;

          if (!samepoint(p1,p2)) {
            if (nedges >= maxedges) return 9;
            n = nedges++;
            newedge = &edges[n];
            newedge->style = CTRI;
            newedge->nvert = 0;
            newedge->verts[0] = newedge->verts[1] = -1;
            copy3(p1,newedge->p1);
            copy3(p2,newedge->p2);
            // convert jedge back to -1 for last vertex
            if (jedge == verts[ivert].first) jedge = -1;
            edge_insert(n,0,ivert,iedge,idir,jedge,jdir);
            i++;
          }

          iedge = jedge;
          idir = jdir;
        }
      }
    }

    return 0;
  }

  /* --------------------------------------------------------------------
     Cut3d::clip_adjust(): adjust the collection of clipped triangles
     discard if clipped tri is a single point, increment touchcount
       touchmark = corner point marking inferred from touching tri orientations
     discard if grazes cell with outward normal, increment grazecount
     if all clipped tris are discarded
       set and return empty, touchcount, grazecount, touchmark
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void clip_adjust()
  {
    int nvert,nedge,nface1,nface2;
    int faces1[6],faces2[6];
    double pboth[3],move1[3],move2[3];
    const double *p1,*p2,*p3;
    Edge *edge;

    // epsilon = EPSEDGE fraction of largest cell dimension

    epsilon = EPSEDGE*(hi[0]-lo[0]);
    epsilon = MAX(epsilon,EPSEDGE*(hi[1]-lo[1]));
    epsilon = MAX(epsilon,EPSEDGE*(hi[2]-lo[2]));

    // collapse edges shorter than epsilon to a single point (so will be removed)
    // one or both of the points should be on cell faces

    nedge = nedges;

    for (int iedge = 0; iedge < nedge; iedge++) {
      if (!edges[iedge].active) continue;

      edge = &edges[iedge];
      p1 = edge->p1;
      p2 = edge->p2;
      double dx = p1[0]-p2[0];
      double dy = p1[1]-p2[1];
      double dz = p1[2]-p2[2];
      double edgelen = sqrt(dx*dx+dy*dy+dz*dz);

      if (edgelen < epsilon) {
        ntiny++;

        nface1 = on_faces(p1,faces1);
        nface2 = on_faces(p2,faces2);

        // set both p1 and p2 to same pboth
        // if both pts are interior (should not happen), pboth = p1
        // if only one pt X is on a face, pboth = X
        // if both pts are on one or more faces:
        //   push both to face(s), recalculate on_faces()
        //   if pt X is on more faces, pboth = X, else pboth = p1

        if (!nface1 && !nface2) {
          copy3(p1,pboth);
        } else if (nface1 && !nface2) {
          copy3(p1,pboth);
        } else if (nface2 && !nface1) {
          copy3(p2,pboth);
        } else {
          copy3(p1,move1);
          copy3(p2,move2);
          move_to_faces(move1);
          move_to_faces(move2);
          nface1 = on_faces(move1,faces1);
          nface2 = on_faces(move2,faces2);
          if (nface2 > nface1) copy3(move2,pboth);
          else copy3(move1,pboth);
        }

        // set all points that are same as old p1 or p2 to pboth
        // reset first for all jedge != iedge, then reset iedge

        for (int jedge = 0; jedge < nedge; jedge++) {
          if (!edges[jedge].active) continue;
          if (jedge == iedge) continue;

          if (samepoint(edges[jedge].p1,p1)) copy3(pboth,edges[jedge].p1);
          if (samepoint(edges[jedge].p2,p1)) copy3(pboth,edges[jedge].p2);

          if (samepoint(edges[jedge].p1,p2)) copy3(pboth,edges[jedge].p1);
          if (samepoint(edges[jedge].p2,p2)) copy3(pboth,edges[jedge].p2);
        }

        copy3(pboth,edges[iedge].p1);
        copy3(pboth,edges[iedge].p2);
      }
    }

    // remove zero-length edges

    nedge = nedges;

    for (int iedge = 0; iedge < nedge; iedge++) {
      if (!edges[iedge].active) continue;
      edge = &edges[iedge];
      if (samepoint(edge->p1,edge->p2)) edge_remove(edge);
    }

    // remove vertices (triangles) which now have less than 3 edges
    // tally inside/outside for all removed tris
    // outside = tri norm from tri ctr points towards cell ctr
    // inside = tri norm from tri ctr points away from cell ctr
    // cbox = cell center pt, ctri = triangle center pt, t2b = cbox-ctri

    touchcount = 0;
    grazecount = 0;

    int noutside = 0;
    int ninside = 0;

    double cbox[3],ctri[3],t2b[3];

    nvert = nverts;

    for (int ivert = 0; ivert < nvert; ivert++)
      if (verts[ivert].nedge <= 2) {
        touchcount++;
        cbox[0] = 0.5*(lo[0]+hi[0]);
        cbox[1] = 0.5*(lo[1]+hi[1]);
        cbox[2] = 0.5*(lo[2]+hi[2]);
        int itri = surfs[verts[ivert].label];
        p1 = tris[itri].p1;
        p2 = tris[itri].p2;
        p3 = tris[itri].p3;
        ctri[0] = (p1[0]+p2[0]+p3[0])/3.0;
        ctri[1] = (p1[1]+p2[1]+p3[1])/3.0;
        ctri[2] = (p1[2]+p2[2]+p3[2])/3.0;
        t2b[0] = cbox[0] - ctri[0];
        t2b[1] = cbox[1] - ctri[1];
        t2b[2] = cbox[2] - ctri[2];
        const double *norm = verts[ivert].norm;
        double dot = norm[0]*t2b[0] + norm[1]*t2b[1] + norm[2]*t2b[2];
        if (dot > 0.0) noutside++;
        if (dot < 0.0) ninside++;
        vertex_remove(&verts[ivert]);
      }

    // discard clipped tri if lies on a cell face w/ normal out of cell
    // increment grazecount in this case

    for (int ivert = 0; ivert < nvert; ivert++) {
      if (!verts[ivert].active) continue;
      if (grazing(&verts[ivert])) {
        vertex_remove(&verts[ivert]);
        grazecount++;
      }
    }

    // remove edges which now have no vertices

    for (int iedge = 0; iedge < nedge; iedge++) {
      if (!edges[iedge].active) continue;
      if (edges[iedge].nvert == 0) edges[iedge].active = 0;
    }

    // set BPG empty flag if no active vertices

    empty = 1;
    for (int ivert = 0; ivert < nvert; ivert++)
      if (verts[ivert].active) {
        empty = 0;
        break;
      }

    // if no lines, set touchmark which will be used to mark corner points
    // only set touchmark if all deleted tris had same orientation

    touchmark = UNKNOWN;
    if (empty) {
      if (ninside && noutside == 0) touchmark = INSIDE;
      else if (noutside && ninside == 0) touchmark = OUTSIDE;
    }
  }

  /* --------------------------------------------------------------------
     Cut3d::ctri_volume(): compute volume of vertices
     when called, only clipped triangles exist
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void ctri_volume()
  {
    int i,iedge,idir,nedge;
    double zarea,volume;
    const double *p0,*p1,*p2;
    Edge *edge;

    int nvert = nverts;
    for (int ivert = 0; ivert < nvert; ivert++) {
      if (!verts[ivert].active) continue;
      iedge = verts[ivert].first;
      idir = verts[ivert].dirfirst;
      nedge = verts[ivert].nedge;

      if (idir == 0) p0 = edges[iedge].p1;
      else p0 = edges[iedge].p2;

      volume = 0.0;

      for (i = 0; i < nedge; i++) {
        edge = &edges[iedge];

        // compute projected volume of a convex polygon to zlo face
        // split polygon into triangles
        // each tri makes a tri-capped volume with zlo face
        // zarea = area of oriented tri projected into z plane
        // volume based on height of z midpt of tri above zlo face

        if (idir == 0) {
          p1 = edge->p1;
          p2 = edge->p2;
        } else {
          p1 = edge->p2;
          p2 = edge->p1;
        }
        zarea = 0.5 * ((p1[0]-p0[0])*(p2[1]-p0[1]) -
                       (p1[1]-p0[1])*(p2[0]-p0[0]));
        volume -= zarea * ((p0[2]+p1[2]+p2[2])/3.0 - lo[2]);

        iedge = edge->next[idir];
        idir = edge->dirnext[idir];
      }

      verts[ivert].volume = volume;
    }
  }

  /* --------------------------------------------------------------------
     Cut3d::edge2face(): assign all singlet edges to faces (0-5)
     singlet edge must be on one or two faces, two if on cell edge
     if along cell edge, assign to one of two faces based on
       which has larger dot product of its inward face norm
       and the norm of the tri containing the edge
     the lists are rows of one array, each in ascending edge order as
       the host builds them
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int edge2face()
  {
    int iface,nface,ivert;
    int faces[6];
    int count[6];
    double dot0;
    double norm_inward[3];
    const double *trinorm;
    Edge *edge;

    for (iface = 0; iface < 6; iface++) count[iface] = 0;

    // loop over edges, assign singlets to exactly one face

    int nedge = nedges;

    for (int iedge = 0; iedge < nedge; iedge++) {
      efaces[iedge] = -1;
      if (!edges[iedge].active) continue;
      if (edges[iedge].nvert == 3) continue;
      edge = &edges[iedge];

      nface = which_faces(edge->p1,edge->p2,faces);
      if (nface == 0) return 2;

      else if (nface == 1) iface = faces[0];

      else if (nface == 2) {
        if (edge->nvert == 1) ivert = edge->verts[0];
        else ivert = edge->verts[1];
        trinorm = verts[ivert].norm;

        iface = faces[0];
        norm_inward[0] = norm_inward[1] = norm_inward[2] = 0.0;
        if (iface % 2) norm_inward[iface/2] = -1.0;
        else norm_inward[iface/2] = 1.0;
        dot0 = norm_inward[0]*trinorm[0] + norm_inward[1]*trinorm[1] +
          norm_inward[2]*trinorm[2];
        if (dot0 > 0.0) iface = faces[1];

      } else return 3;

      efaces[iedge] = iface;
      count[iface]++;
    }

    facestart[0] = 0;
    for (iface = 0; iface < 6; iface++)
      facestart[iface+1] = facestart[iface] + count[iface];

    int cursor[6];
    for (iface = 0; iface < 6; iface++) cursor[iface] = facestart[iface];
    for (int iedge = 0; iedge < nedge; iedge++)
      if (efaces[iedge] >= 0) facelist[cursor[efaces[iedge]]++] = iedge;

    return 0;
  }

  /* --------------------------------------------------------------------
     Cut3d::edge2clines(): build a 2d CLINES data structure
       from all singlet edges assigned to iface (0-5)
     order pts in edge for tri traversing edge in forward order
     flip edge if in a flip face = faces 0,3,4
     edge label in clines = edge index in BPG
     return 9 if the cline row overflows
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int edge2clines(int iface)
  {
    int iedge;
    const double *p1,*p2;
    Edge *edge;
    Cut2dKokkos::Cline *cline;

    int flip = 0;
    if (iface == 0 || iface == 3 || iface == 4) flip = 1;

    int nline = facestart[iface+1] - facestart[iface];
    if (nline > maxclines) return 9;
    cut2d.nclines = 0;

    for (int i = 0; i < nline; i++) {
      iedge = facelist[facestart[iface]+i];
      edge = &edges[iedge];
      if (edge->nvert == 1) {
        p1 = edge->p1;
        p2 = edge->p2;
      } else {
        p1 = edge->p2;
        p2 = edge->p1;
      }
      cline = &cut2d.clines[i];
      cline->line = iedge;
      if (flip) {
        compress2d(iface,p1,cline->y);
        compress2d(iface,p2,cline->x);
      } else {
        compress2d(iface,p1,cline->x);
        compress2d(iface,p2,cline->y);
      }
    }

    cut2d.nclines = nline;
    return 0;
  }

  /* --------------------------------------------------------------------
     Cut3d::add_face_pgons(): add one or more face polygons as vertices
       to BPG
     have to convert pts computed by cut2d back into 3d pts on face
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int add_face_pgons(int iface)
  {
    int iloop,mloop,nloop,ipt,mpt,npt;
    int iedge,dir,prev,dirprev;
    double p1[3],p2[3];
    Vertex *vert;
    Edge *edge;
    Cut2dKokkos::PG *pg;
    Cut2dKokkos::Loop *loop;
    Cut2dKokkos::Point *p12d,*p22d;

    Cut2dKokkos::PG *pgs = cut2d.pgs;
    Cut2dKokkos::Loop *loops2d = cut2d.loops;
    Cut2dKokkos::Point *points = cut2d.points;

    int flip = 0;
    if (iface == 0 || iface == 3 || iface == 4) flip = 1;

    double value;
    int dim = iface / 2;
    int lohi = iface % 2;
    if (lohi == 0) value = lo[dim];
    else value = hi[dim];

    int npg = cut2d.npgs;
    int nvert = nverts;

    for (int ipg = 0; ipg < npg; ipg++) {
      pg = &pgs[ipg];

      if (nvert >= maxverts) return 9;
      vert = &verts[nvert];
      vert->active = 1;
      vert->style = FACEPGON;
      vert->label = iface;
      if (iface == 5) vert->volume = pg->area * (hi[2]-lo[2]);
      else vert->volume = 0.0;
      vert->nedge = 0;
      vert->norm = NULL;

      prev = -1;
      dirprev = -1;

      nloop = pg->n;
      mloop = pg->first;
      for (iloop = 0; iloop < nloop; iloop++) {
        loop = &loops2d[mloop];
        npt = loop->n;
        mpt = loop->first;

        for (ipt = 0; ipt < npt; ipt++) {
          p12d = &points[mpt];
          mpt = p12d->next;
          p22d = &points[mpt];
          expand2d(iface,value,p12d->x,p1);
          expand2d(iface,value,p22d->x,p2);

          // edge was from a CTRI vertex
          // match in opposite order that CTRI vertex matched it

          if (p12d->type == Cut2dKokkos::ENTRY ||
              p12d->type == Cut2dKokkos::TWO) {
            iedge = p12d->line;
            edge = &edges[iedge];
            edge->style = CTRIFACE;
            if (edge->nvert == 1) dir = 1;
            else dir = 0;
            edge_insert(iedge,dir,nvert,prev,dirprev,-1,-1);
            prev = iedge;
            dirprev = dir;
            continue;
          }

          // face edge not from a CTRI
          // unflip edge if in a flip face

          if (flip) iedge = findedge(p2,p1,0,dir);
          else iedge = findedge(p1,p2,0,dir);
          if (iedge == -2) return 1;

          if (iedge >= 0) {
            edge_insert(iedge,dir,nvert,prev,dirprev,-1,-1);
            prev = iedge;
            dirprev = 1;
            continue;
          }

          if (nedges >= maxedges) return 9;
          iedge = nedges++;
          edge = &edges[iedge];
          edge->style = FACEPGON;
          edge->nvert = 0;
          edge->verts[0] = edge->verts[1] = -1;
          if (flip) {
            copy3(p2,edge->p1);
            copy3(p1,edge->p2);
          } else {
            copy3(p1,edge->p1);
            copy3(p2,edge->p2);
          }
          dir = 0;
          edge_insert(iedge,dir,nvert,prev,dirprev,-1,-1);
          prev = iedge;
          dirprev = 0;
        }
        mloop = loop->next;
      }

      nvert++;
    }

    nverts = nvert;
    return 0;
  }

  /* --------------------------------------------------------------------
     Cut3d::add_face(): add an entire cell face as vertex to BPG
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int add_face(int iface)
  {
    int i,j,iedge,dir,prev,dirprev;
    double p1[3],p2[3];
    Vertex *vert;
    Edge *edge;

    if (nverts >= maxverts) return 9;
    int nvert = nverts++;
    vert = &verts[nvert];
    vert->active = 1;
    vert->style = FACE;
    vert->label = iface;
    if (iface == 5)
      vert->volume = (hi[0]-lo[0]) * (hi[1]-lo[1]) * (hi[2]-lo[2]);
    else vert->volume = 0.0;
    vert->nedge = 0;
    vert->norm = NULL;

    double value;
    int dim = iface / 2;
    int lohi = iface % 2;
    if (lohi == 0) value = lo[dim];
    else value = hi[dim];

    // usual ordering of points in face as LL,LR,UR,UL
    // flip order if in a flip face

    int flip = 0;
    if (iface == 0 || iface == 3 || iface == 4) flip = 1;

    double cpts[4][2];

    if (flip) {
      cpts[0][0] = lo2d[0]; cpts[0][1] = lo2d[1];
      cpts[1][0] = lo2d[0]; cpts[1][1] = hi2d[1];
      cpts[2][0] = hi2d[0]; cpts[2][1] = hi2d[1];
      cpts[3][0] = hi2d[0]; cpts[3][1] = lo2d[1];
    } else {
      cpts[0][0] = lo2d[0]; cpts[0][1] = lo2d[1];
      cpts[1][0] = hi2d[0]; cpts[1][1] = lo2d[1];
      cpts[2][0] = hi2d[0]; cpts[2][1] = hi2d[1];
      cpts[3][0] = lo2d[0]; cpts[3][1] = hi2d[1];
    }

    if (vert->nedge) {
      prev = vert->last;
      dirprev = vert->dirlast;
    } else {
      prev = -1;
      dirprev = -1;
    }

    for (i = 0; i < 4; i++) {
      j = i+1;
      if (j == 4) j = 0;
      expand2d(iface,value,&cpts[i][0],p1);
      expand2d(iface,value,&cpts[j][0],p2);
      iedge = findedge(p1,p2,1,dir);
      if (iedge == -2) return 1;

      if (iedge >= 0) {
        edge_insert(iedge,dir,nvert,prev,dirprev,-1,-1);
        prev = iedge;
        dirprev = 1;
        continue;
      }

      if (nedges >= maxedges) return 9;
      iedge = nedges++;
      edge = &edges[iedge];
      edge->style = vert->style;
      edge->nvert = 0;
      edge->verts[0] = edge->verts[1] = -1;
      copy3(p1,edge->p1);
      copy3(p2,edge->p2);
      dir = 0;
      edge_insert(iedge,dir,nvert,prev,dirprev,-1,-1);
      prev = iedge;
      dirprev = 0;
    }

    return 0;
  }

  /* --------------------------------------------------------------------
     Cut3d::remove_faces(): remove any FACE vertices with one or more
       unconnected edges
     iterate twice since another face may become unconnected
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void remove_faces()
  {
    int i,ivert,iedge,dir;
    Vertex *vert;
    Edge *edge;

    int nvert = nverts;

    for (int iter = 0; iter < 2; iter++)
      for (ivert = 0; ivert < nvert; ivert++) {
        if (!verts[ivert].active) continue;
        if (verts[ivert].style != FACE) continue;
        vert = &verts[ivert];

        iedge = vert->first;
        dir = vert->dirfirst;
        for (i = 0; i < 4; i++) {
          edge = &edges[iedge];
          if (edge->nvert == 1 || edge->nvert == 2) break;
          iedge = edge->next[dir];
          dir = edge->dirnext[dir];
        }
        if (i < 4) vertex_remove(vert);
      }
  }

  /* --------------------------------------------------------------------
     Cut3d::check(): check BPG for consistency
     vertices have 3 or more unique edges that point back to it
     edges have 2 unique vertices
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int check()
  {
    int i,iedge,dir,nedge,last,dirlast;
    Vertex *vert;
    Edge *edge;

    // mark all edges as unclipped
    // use for detecting duplicate edges in same vertex

    nedge = nedges;
    for (iedge = 0; iedge < nedge; iedge++)
      if (edges[iedge].active) edges[iedge].clipped = 0;

    // check vertices
    // for each vertex: mark edges as see them, unmark all edges at end

    int nvert = nverts;
    for (int ivert = 0; ivert < nvert; ivert++) {
      if (!verts[ivert].active) continue;
      vert = &verts[ivert];
      if (vert->nedge < 3) return 11;

      nedge = vert->nedge;
      iedge = vert->first;
      dir = vert->dirfirst;
      last = dirlast = -1;

      for (i = 0; i < nedge; i++) {
        edge = &edges[iedge];
        if (!edge->active) return 12;
        if (edge->verts[dir] != ivert) return 13;
        if (edge->clipped) return 14;
        edge->clipped = 1;
        last = iedge;
        dirlast = dir;
        iedge = edge->next[dir];
        dir = edge->dirnext[dir];
      }

      if (last != vert->last || dirlast != vert->dirlast) return 15;

      iedge = vert->first;
      dir = vert->dirfirst;
      for (i = 0; i < nedge; i++) {
        edge = &edges[iedge];
        edge->clipped = 0;
        iedge = edge->next[dir];
        dir = edge->dirnext[dir];
      }
    }

    // check edges

    nedge = nedges;
    for (int iedge = 0; iedge < nedge; iedge++) {
      if (!edges[iedge].active) continue;
      edge = &edges[iedge];

      if (edge->nvert != 3) return 16;
      if (edge->verts[0] == edge->verts[1]) return 17;
      if (edge->verts[0] >= nvert || !verts[edge->verts[0]].active) return 18;
      if (edge->verts[1] >= nvert || !verts[edge->verts[1]].active) return 19;
    }

    return 0;
  }

  /* --------------------------------------------------------------------
     Cut3d::walk(): convert BPG into simple closed polyhedra, not nested
     walk BPG from any unused vertex, flagging vertices as used
     stack is list of new vertices to process
     loop over edges of pgon, add its unused neighbors to stack
     when stack is empty, loop is closed
     accumulate volume of polyhedra as walk it from volume of each vertex
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void walk()
  {
    int j,flag,ncount,ivert,firstvert,iedge,dir,nedge,prev;
    double volume;
    Vertex *vert;
    Edge *edge;

    // used = 0/1 flag for whether a vertex is already part of a loop
    // only active vertices are eligible

    int nvert = nverts;
    for (int ivert = 0; ivert < nvert; ivert++) {
      if (verts[ivert].active) used[ivert] = 0;
      else used[ivert] = 1;
    }

    int nstack = 0;
    int nloop = 0;

    for (int i = 0; i < nvert; i++) {
      if (used[i]) continue;
      volume = 0.0;
      flag = INTERIOR;
      ncount = 0;

      stack[0] = firstvert = i;
      nstack = 1;
      used[i] = 1;
      prev = -1;
      vert = &verts[i];

      while (nstack) {
        nstack--;
        ivert = stack[nstack];
        ncount++;

        vert = &verts[ivert];
        if (vert->style != CTRI) flag = BORDER;
        volume += vert->volume;

        nedge = vert->nedge;
        iedge = vert->first;
        dir = vert->dirfirst;

        for (j = 0; j < nedge; j++) {
          edge = &edges[iedge];
          if (!used[edge->verts[0]]) {
            stack[nstack++] = edge->verts[0];
            used[edge->verts[0]] = 1;
          }
          if (!used[edge->verts[1]]) {
            stack[nstack++] = edge->verts[1];
            used[edge->verts[1]] = 1;
          }
          iedge = edge->next[dir];
          dir = edge->dirnext[dir];
        }

        if (prev >= 0) verts[prev].next = ivert;
        prev = ivert;
      }
      vert->next = -1;

      loops[nloop].volume = volume;
      loops[nloop].flag = flag;
      loops[nloop].n = ncount;
      loops[nloop].first = firstvert;
      nloop++;
    }

    nloops = nloop;
  }

  /* --------------------------------------------------------------------
     Cut3d::loop2ph(): one PH (polyhedron) for each disjoint flow volume
     error returns 4,5,6 as the host
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int loop2ph()
  {
    int positive = 0;
    int negative = 0;

    int nloop = nloops;

    for (int i = 0; i < nloop; i++) {
      if (loops[i].volume > 0.0) positive++;
      else negative++;
    }

    // if no positive vols, cell is entirely inside the surf, caller handles it

    if (positive == 0) return 4;

    // do not allow mulitple positive with one or more negative

    if (positive > 1 && negative) return 5;

    // positive = 1 means 1 PH with vol = sum of all pos/neg loops
    // positive > 1 means each loop is a PH

    if (positive == 1) {
      double volume = 0.0;
      for (int i = 0; i < nloop; i++) {
        volume += loops[i].volume;
        loops[i].next = i+1;
      }
      loops[nloop-1].next = -1;

      if (volume < 0.0) return 6;

      phs[0].volume = volume;
      phs[0].n = nloop;
      phs[0].first = 0;

    } else {
      for (int i = 0; i < nloop; i++) {
        phs[i].volume = loops[i].volume;
        phs[i].n = 1;
        phs[i].first = i;
        loops[i].next = -1;
      }
    }

    nphs = positive;
    return 0;
  }

  /* --------------------------------------------------------------------
     Cut3d::create_surfmap(): surfmap[i] = PH the Ith tri is assigned to
     -1 if the tri did not end up in a PH
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void create_surfmap(int *surfmap)
  {
    for (int i = 0; i < nsurf; i++) surfmap[i] = -1;

    int iloop,nloop,mloop,ivert,nvert,mvert;

    for (int iph = 0; iph < nphs; iph++) {
      nloop = phs[iph].n;
      mloop = phs[iph].first;
      for (iloop = 0; iloop < nloop; iloop++) {
        nvert = loops[mloop].n;
        mvert = loops[mloop].first;
        for (ivert = 0; ivert < nvert; ivert++) {
          if (verts[mvert].style == CTRI || verts[mvert].style == CTRIFACE)
            surfmap[verts[mvert].label] = iph;
          mvert = verts[mvert].next;
        }
        mloop = loops[mloop].next;
      }
    }
  }

  /* --------------------------------------------------------------------
     Cut3d::split_point_explicit(): a surf point in or on the cell
     return xsplit = coords of point, xsub = its sub-cell index
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int split_point_explicit(int *surfmap, double *xsplit, int &xsub)
  {
    int itri;
    const double *x1,*x2,*x3;

    // if end pt of any tri with non-negative surfmap is in/on cell, return

    for (int i = 0; i < nsurf; i++) {
      if (surfmap[i] < 0) continue;
      itri = surfs[i];
      x1 = tris[itri].p1;
      x2 = tris[itri].p2;
      x3 = tris[itri].p3;
      if (ptflag(x1) != EXTERIOR) {
        xsplit[0] = x1[0]; xsplit[1] = x1[1]; xsplit[2] = x1[2];
        xsub = surfmap[i];
        return 0;
      }
      if (ptflag(x2) != EXTERIOR) {
        xsplit[0] = x2[0]; xsplit[1] = x2[1]; xsplit[2] = x2[2];
        xsub = surfmap[i];
        return 0;
      }
      if (ptflag(x3) != EXTERIOR) {
        xsplit[0] = x3[0]; xsplit[1] = x3[1]; xsplit[2] = x3[2];
        xsub = surfmap[i];
        return 0;
      }
    }

    // clip 1st tri with non-negative surfmap to cell, and return clip point

    for (int i = 0; i < nsurf; i++) {
      if (surfmap[i] < 0) continue;
      itri = surfs[i];
      x1 = tris[itri].p1;
      x2 = tris[itri].p2;
      x3 = tris[itri].p3;
      clip(x1,x2,x3);
      xsplit[0] = path1[0][0]; xsplit[1] = path1[0][1]; xsplit[2] = path1[0][2];
      xsub = surfmap[i];
      return 0;
    }

    return 7;
  }

  /* --------------------------------------------------------------------
     Cut3d::edge_insert(): insert edge IEDGE in DIR for ivert
     also update vertex info for added edge
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void edge_insert(int iedge, int dir, int ivert,
                   int iprev, int dirprev, int inext, int dirnext)
  {
    Edge *edge = &edges[iedge];

    if (dir == 0) {
      edge->nvert += 1;
      edge->verts[0] = ivert;
    } else {
      edge->nvert += 2;
      edge->verts[1] = ivert;
    }

    edge->active = 1;
    edge->clipped = 0;

    // set prev/next pointers for doubly linked list of edges

    edge->next[dir] = inext;
    edge->prev[dir] = iprev;

    if (inext >= 0) {
      edge->dirnext[dir] = dirnext;
      Edge *next = &edges[inext];
      next->prev[dirnext] = iedge;
      next->dirprev[dirnext] = dir;
    } else edge->dirnext[dir] = -1;

    if (iprev >= 0) {
      edge->dirprev[dir] = dirprev;
      Edge *prev = &edges[iprev];
      prev->next[dirprev] = iedge;
      prev->dirnext[dirprev] = dir;
    } else edge->dirprev[dir] = -1;

    // add edge info to owning vertex

    verts[ivert].nedge++;
    if (iprev < 0) {
      verts[ivert].first = iedge;
      verts[ivert].dirfirst = dir;
    }
    if (inext < 0) {
      verts[ivert].last = iedge;
      verts[ivert].dirlast = dir;
    }
  }

  /* --------------------------------------------------------------------
     Cut3d::edge_remove(): complete edge removal in both dirs
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void edge_remove(Edge *edge)
  {
    if (edge->verts[0] >= 0) edge_remove(edge,0);
    if (edge->verts[1] >= 0) edge_remove(edge,1);
  }

  /* --------------------------------------------------------------------
     Cut3d::edge_remove(): edge removal in DIR
     also update vertex info for removed edge
     mark edge inactive if its nvert -> 0
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void edge_remove(Edge *edge, int dir)
  {
    int ivert = edge->verts[dir];
    edge->verts[dir] = -1;
    if (dir == 0) edge->nvert--;
    else edge->nvert -= 2;
    if (edge->nvert == 0) edge->active = 0;

    // reset prev/next pointers for doubly linked list to skip this edge

    if (edge->prev[dir] >= 0) {
      Edge *prev = &edges[edge->prev[dir]];
      int dirprev = edge->dirprev[dir];
      prev->next[dirprev] = edge->next[dir];
      prev->dirnext[dirprev] = edge->dirnext[dir];
    }

    if (edge->next[dir] >= 0) {
      Edge *next = &edges[edge->next[dir]];
      int dirnext = edge->dirnext[dir];
      next->prev[dirnext] = edge->prev[dir];
      next->dirprev[dirnext] = edge->dirprev[dir];
    }

    // update vertex for removal of this edge

    verts[ivert].nedge--;
    if (edge->prev[dir] < 0) {
      verts[ivert].first = edge->next[dir];
      verts[ivert].dirfirst = edge->dirnext[dir];
    }
    if (edge->next[dir] < 0) {
      verts[ivert].last = edge->prev[dir];
      verts[ivert].dirlast = edge->dirprev[dir];
    }
  }

  /* --------------------------------------------------------------------
     Cut3d::vertex_remove(): remove a vertex and all edges it includes
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void vertex_remove(Vertex *vert)
  {
    Edge *edge;

    vert->active = 0;

    int iedge = vert->first;
    int dir = vert->dirfirst;
    int nedge = vert->nedge;

    for (int i = 0; i < nedge; i++) {
      edge = &edges[iedge];
      if (dir == 0) edge->nvert--;
      else edge->nvert -= 2;
      if (edge->nvert == 0) edge->active = 0;
      edge->verts[dir] = -1;
      iedge = edge->next[dir];
      dir = edge->dirnext[dir];
    }
  }

  /* --------------------------------------------------------------------
     Cut3d::grazing(): 1 if a planar polygon lies entirely in plane of
       any face of cell and its normal is outward with respect to cell
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int grazing(Vertex *vert)
  {
    int count[6];
    const double *p;
    Edge *edge;

    int iedge = vert->first;
    int idir = vert->dirfirst;
    int nedge = vert->nedge;

    count[0] = count[1] = count[2] = count[3] = count[4] = count[5] = 0;

    for (int i = 0; i < nedge; i++) {
      edge = &edges[iedge];
      if (idir == 0) p = edge->p1;
      else p = edge->p2;

      if (p[0] == lo[0]) count[0]++;
      if (p[0] == hi[0]) count[1]++;
      if (p[1] == lo[1]) count[2]++;
      if (p[1] == hi[1]) count[3]++;
      if (p[2] == lo[2]) count[4]++;
      if (p[2] == hi[2]) count[5]++;

      iedge = edge->next[idir];
      idir = edge->dirnext[idir];
    }

    const double *norm = vert->norm;
    if (count[0] == nedge && norm[0] < 0.0) return 1;
    if (count[1] == nedge && norm[0] > 0.0) return 1;
    if (count[2] == nedge && norm[1] < 0.0) return 1;
    if (count[3] == nedge && norm[1] > 0.0) return 1;
    if (count[4] == nedge && norm[2] < 0.0) return 1;
    if (count[5] == nedge && norm[2] > 0.0) return 1;
    return 0;
  }

  /* --------------------------------------------------------------------
     Cut3d::on_faces(): which cell faces point P is on, 0,1,2 of them
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int on_faces(const double *p, int *faces) const
  {
    int n = 0;
    if (p[0] == lo[0]) faces[n++] = 0;
    if (p[0] == hi[0]) faces[n++] = 1;
    if (p[1] == lo[1]) faces[n++] = 2;
    if (p[1] == hi[1]) faces[n++] = 3;
    if (p[2] == lo[2]) faces[n++] = 4;
    if (p[2] == hi[2]) faces[n++] = 5;
    return n;
  }

  /* --------------------------------------------------------------------
     Cut3d::which_faces(): which cell faces edge between p1,p2 is on
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int which_faces(const double *p1, const double *p2, int *faces) const
  {
    int n = 0;
    if (p1[0] == lo[0] && p2[0] == lo[0]) faces[n++] = 0;
    if (p1[0] == hi[0] && p2[0] == hi[0]) faces[n++] = 1;
    if (p1[1] == lo[1] && p2[1] == lo[1]) faces[n++] = 2;
    if (p1[1] == hi[1] && p2[1] == hi[1]) faces[n++] = 3;
    if (p1[2] == lo[2] && p2[2] == lo[2]) faces[n++] = 4;
    if (p1[2] == hi[2] && p2[2] == hi[2]) faces[n++] = 5;
    return n;
  }

  /* --------------------------------------------------------------------
     Cut3d::face_from_cell(): extract 2d cell lo2d/hi2d from iface (0-5)
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void face_from_cell(int iface)
  {
    if (iface < 2) {
      lo2d[0] = lo[1]; hi2d[0] = hi[1];
      lo2d[1] = lo[2]; hi2d[1] = hi[2];
    } else if (iface < 4) {
      lo2d[0] = lo[0]; hi2d[0] = hi[0];
      lo2d[1] = lo[2]; hi2d[1] = hi[2];
    } else {
      lo2d[0] = lo[0]; hi2d[0] = hi[0];
      lo2d[1] = lo[1]; hi2d[1] = hi[1];
    }
  }

  /* --------------------------------------------------------------------
     Cut3d::compress2d(): compress a 3d pt into a 2d pt on iface
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void compress2d(int iface, const double *p3, double *p2) const
  {
    if (iface < 2) {
      p2[0] = p3[1]; p2[1] = p3[2];
    } else if (iface < 4) {
      p2[0] = p3[0]; p2[1] = p3[2];
    } else {
      p2[0] = p3[0]; p2[1] = p3[1];
    }
  }

  /* --------------------------------------------------------------------
     Cut3d::expand2d(): expand a 2d pt into 3d pt on iface with extra
       coord = value
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void expand2d(int iface, double value, const double *p2, double *p3) const
  {
    if (iface < 2) {
      p3[0] = value; p3[1] = p2[0]; p3[2] = p2[1];
    } else if (iface < 4) {
      p3[0] = p2[0]; p3[1] = value; p3[2] = p2[1];
    } else {
      p3[0] = p2[0]; p3[1] = p2[1]; p3[2] = value;
    }
  }

  /* --------------------------------------------------------------------
     Cut3d::findedge(): look for edge (x,y) in list of edges
     match as (x,y) or (y,x)
     if flag, do not match edges that are part of a CTRI
     return = index if find it, -1 if do not find it
     return dir = 0 if matches as (x,y), 1 if matches as (y,x), -1 if no match
     return -2 as error if edge already exists in same dir as this one
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int findedge(const double *x, const double *y, int flag, int &dir) const
  {
    const double *p1,*p2;

    int nedge = nedges;

    for (int i = 0; i < nedge; i++) {
      if (!edges[i].active) continue;
      if (flag && (edges[i].style == CTRI || edges[i].style == CTRIFACE))
        continue;
      p1 = edges[i].p1;
      p2 = edges[i].p2;
      if (samepoint(x,p1) && samepoint(y,p2)) {
        if (edges[i].nvert % 2 == 1) return -2;
        dir = 0;
        return i;
      }
      if (samepoint(x,p2) && samepoint(y,p1)) {
        if (edges[i].nvert / 2 == 1) return -2;
        dir = 1;
        return i;
      }
    }

    dir = -1;
    return -1;
  }

  /* --------------------------------------------------------------------
     Cut3d::between(): intersection pt C of line segment A,B in dim
       with coord value; C can be same as A or B, will just overwrite
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  static void between(const double *a, const double *b, int dim,
                      double value, double *c)
  {
    if (dim == 0) {
      c[1] = a[1] + (value-a[dim])/(b[dim]-a[dim]) * (b[1]-a[1]);
      c[2] = a[2] + (value-a[dim])/(b[dim]-a[dim]) * (b[2]-a[2]);
      c[0] = value;
    } else if (dim == 1) {
      c[0] = a[0] + (value-a[dim])/(b[dim]-a[dim]) * (b[0]-a[0]);
      c[2] = a[2] + (value-a[dim])/(b[dim]-a[dim]) * (b[2]-a[2]);
      c[1] = value;
    } else {
      c[0] = a[0] + (value-a[dim])/(b[dim]-a[dim]) * (b[0]-a[0]);
      c[1] = a[1] + (value-a[dim])/(b[dim]-a[dim]) * (b[1]-a[1]);
      c[2] = value;
    }
  }

  KOKKOS_INLINE_FUNCTION
  static int samepoint(const double *x, const double *y)
  {
    if (x[0] == y[0] && x[1] == y[1] && x[2] == y[2]) return 1;
    return 0;
  }

  KOKKOS_INLINE_FUNCTION
  static void copy3(const double *a, double *b)
  {
    b[0] = a[0]; b[1] = a[1]; b[2] = a[2];
  }

  /* --------------------------------------------------------------------
     Cut3d::corner(): 0-7 if pt is a corner pt of grid cell, else -1
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int corner(const double *pt) const
  {
    if (pt[2] == lo[2]) {
      if (pt[1] == lo[1]) {
        if (pt[0] == lo[0]) return 0;
        else if (pt[0] == hi[0]) return 1;
      } else if (pt[1] == hi[1]) {
        if (pt[0] == lo[0]) return 2;
        else if (pt[0] == hi[0]) return 3;
      }
    } else if (pt[2] == hi[2]) {
      if (pt[1] == lo[1]) {
        if (pt[0] == lo[0]) return 4;
        else if (pt[0] == hi[0]) return 5;
      } else if (pt[1] == hi[1]) {
        if (pt[0] == lo[0]) return 6;
        else if (pt[0] == hi[0]) return 7;
      }
    }

    return -1;
  }

  /* --------------------------------------------------------------------
     Cut3d::move_to_faces(): move point within epsilon of any cell face
       to be on cell faces
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void move_to_faces(double *pt) const
  {
    if (fabs(pt[0]-lo[0]) < epsilon) pt[0] = lo[0];
    if (fabs(pt[0]-hi[0]) < epsilon) pt[0] = hi[0];
    if (fabs(pt[1]-lo[1]) < epsilon) pt[1] = lo[1];
    if (fabs(pt[1]-hi[1]) < epsilon) pt[1] = hi[1];
    if (fabs(pt[2]-lo[2]) < epsilon) pt[2] = lo[2];
    if (fabs(pt[2]-hi[2]) < epsilon) pt[2] = hi[2];
  }

  /* --------------------------------------------------------------------
     Cut3d::ptflag(): EXTERIOR, INTERIOR or BORDER of the cell
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int ptflag(const double *pt) const
  {
    double x = pt[0];
    double y = pt[1];
    double z = pt[2];
    if (x < lo[0] || x > hi[0] || y < lo[1] || y > hi[1] ||
        z < lo[2] || z > hi[2]) return EXTERIOR;
    if (x > lo[0] && x < hi[0] && y > lo[1] && y < hi[1] &&
        z > lo[2] && z < hi[2]) return INTERIOR;
    return BORDER;
  }
};

}

#endif
