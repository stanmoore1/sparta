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

#ifndef SPARTA_CUT2D_KOKKOS_H
#define SPARTA_CUT2D_KOKKOS_H

#include "kokkos_type.h"
#include "surf.h"
#include "math_const.h"

namespace SPARTA_NS {

/* ----------------------------------------------------------------------
   device twin of Cut2d::split(): the cut of one 2d grid cell by the
     lines of its cut list, the same arithmetic in the same order, so a
     cell cut on the device gets the flow areas, corner marks, piece map
     and split point the host cut gives it
   one instance per thread, built on the cell and on scratch rows the
     caller sized: nsurf clipped lines, 2*nsurf+4 points, and as many
     loops, polygons and walk flags, the bounds Cut2d grows its vectors to
   the error returns of the host cut come back as its errflag values
     (1-7, see the foot of cut2d.h; 8 = more pieces than surfs, which
     cannot happen); the caller re-runs the host cut on such a cell,
     which prints the cell and raises the message
   the implicit-surf split point is not ported: fix rigid rejects
     implicit surfs
------------------------------------------------------------------------- */

struct Cut2dKokkos {
  struct Cline {
    double x[2],y[2];   // coords of end points of line clipped to cell
    int line;           // index in list of lines that intersect this cell
  };

  struct Point {
    double x[2];        // coords of point
    int type;           // type of pt = ENTRY,EXIT,TWO,CORNER
    int next;           // index of next point when walking a flow area loop
    int line;           // index of line this pt starts in intersecting line list
    int corner;         // 0,1,2,3 if pt is geometrically a corner point, else -1
    int cprev,cnext;    // indices of pts in linked list around cell perimeter
    int side;           // which side of cell (0,1,2,3) pt is on
    double value;       // coord along the side
  };

  struct Loop {
    double area;        // area of loop
    int active;         // 1/0 if active or not
    int flag;           // INTERIOR, BORDER, INTBORD
    int n;              // # of points in loop
    int first;          // index of first point in loop
    int next;           // index of next loop in same PG, -1 if last loop
  };

  struct PG {
    double area;        // summed area (over loops) of PG
    int n;              // # of loops in PG
    int first;          // index of first loop in PG
  };

  enum{UNKNOWN,OUTSIDE,INSIDE,OVERLAP};     // same as Cut2d
  enum{EXTERIOR,INTERIOR,BORDER,INTBORD};
  enum{ENTRY,EXIT,TWO,CORNER};

  const Surf::Line *lines;   // the surf lines on the device
  int axisymmetric;

  const double *lo,*hi;      // opposite corner pts of cell
  int nsurf;                 // # of surf elements in cell
  const int *surfs;          // local indices of surf elements in cell

  Cline *clines;             // scratch rows, sized by the caller
  Point *points;
  Loop *loops;
  PG *pgs;
  int *used;
  int nclines,npoints,nloops,npgs;

  int grazecount;            // count of lines that graze cell surf w/ outward norm
  int touchcount;            // count of line that only touch cell surf
  int touchmark;             // corner marking inferred by touching lines

  KOKKOS_INLINE_FUNCTION
  Cut2dKokkos(const Surf::Line *lines_in, int axisymmetric_in,
              const double *lo_in, const double *hi_in,
              int nsurf_in, const int *surfs_in,
              Cline *clines_in, Point *points_in, Loop *loops_in,
              PG *pgs_in, int *used_in) :
    lines(lines_in), axisymmetric(axisymmetric_in), lo(lo_in), hi(hi_in),
    nsurf(nsurf_in), surfs(surfs_in), clines(clines_in), points(points_in),
    loops(loops_in), pgs(pgs_in), used(used_in),
    nclines(0), npoints(0), nloops(0), npgs(0),
    grazecount(0), touchcount(0), touchmark(UNKNOWN) {}

  /* --------------------------------------------------------------------
     Cut2d::split(): cut area and pieces of the cell
     return nsplit = # of pieces, 1 for no split, 0 on an error with
       errflag = the host's error code
     areas = one flow area per piece, in a row of nsurf
     corners = UNKNOWN/INSIDE/OUTSIDE for each of 4 corner pts
     if nsplit > 1: surfmap = piece of each surf, -1 if in none;
       xsplit = a point in piece xsub
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int split(double *areas, int *surfmap, int *corners,
            int &xsub, double *xsplit, int &errflag)
  {
    int nsplit;

    errflag = 0;
    build_clines();

    // all lines just touched cell surface
    // mark corner points based non-zero grazecount or touchmark value
    // return area = 0.0 for UNKNOWN/INSIDE, full cell area for OUTSIDE

    if (nclines == 0) {
      int mark = UNKNOWN;
      if (grazecount || touchmark == INSIDE) mark = INSIDE;
      else if (touchmark == OUTSIDE) mark = OUTSIDE;
      corners[0] = corners[1] = corners[2] = corners[3] = mark;

      double area = 0.0;
      if (mark == OUTSIDE) {
        if (axisymmetric)
          area = MathConst::MY_PI * (hi[1]*hi[1] - lo[1]*lo[1]) * (hi[0]-lo[0]);
        else area = (hi[0]-lo[0]) * (hi[1]-lo[1]);
      }

      areas[0] = area;
      return 1;
    }

    // 3 operations can generate errors: weiler_build, loop2pg, split_point

    errflag = weiler_build();
    if (errflag) return 0;

    weiler_loops();
    errflag = loop2pg();

    // loop2pg detected no positive-area loops, cell is inside the surf

    if (errflag == 4) {
      errflag = 0;
      corners[0] = corners[1] = corners[2] = corners[3] = INSIDE;
      areas[0] = 0.0;
      return 1;
    }

    if (errflag) return 0;

    nsplit = npgs;
    if (nsplit > 1) {
      create_surfmap(surfmap);
      errflag = split_point_explicit(surfmap,xsplit,xsub);
      if (errflag) return 0;
    }

    // successful cut/split
    // set corners = OUTSIDE if corner pt is in list of points in PGs
    // else set corners = INSIDE

    corners[0] = corners[1] = corners[2] = corners[3] = INSIDE;

    int iloop,nloop,mloop,ipt,npt,mpt;

    for (int ipg = 0; ipg < npgs; ipg++) {
      nloop = pgs[ipg].n;
      mloop = pgs[ipg].first;
      for (iloop = 0; iloop < nloop; iloop++) {
        npt = loops[mloop].n;
        mpt = loops[mloop].first;
        for (ipt = 0; ipt < npt; ipt++) {
          if (points[mpt].corner >= 0)
            corners[points[mpt].corner] = OUTSIDE;
          mpt = points[mpt].next;
        }
        mloop = loops[mloop].next;
      }
    }

    // a piece holds at least one line of its own, so nsplit <= nsurf and
    //   the areas fit a row of nsurf; anything else is an error (8)

    if (nsplit > nsurf) {
      errflag = 8;
      return 0;
    }
    for (int i = 0; i < nsplit; i++) areas[i] = pgs[i].area;

    return nsplit;
  }

  /* --------------------------------------------------------------------
     Cut2d::split_face(): the cut of one face of a 3d cell by the
       clines the caller filled in; return the errflag + 20
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int split_face()
  {
    int errflag = weiler_build();
    if (errflag) return errflag+20;
    weiler_loops();
    errflag = loop2pg();
    if (errflag) return errflag+20;
    return 0;
  }

  /* --------------------------------------------------------------------
     Cut2d::build_clines(): clines = list of lines clipped to cell
     skip transparent surfs
     also set touchcount and grazecount and touchmark
       only used if all clipped lines are discarded
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void build_clines()
  {
    int m;
    double p1[2],p2[2],cbox[3],cmid[3],l2b[3];
    double *x,*y;
    const double *norm,*pp1,*pp2;
    const Surf::Line *line;
    Cline *cline;

    nclines = 0;
    touchcount = 0;
    grazecount = 0;

    int noutside = 0;
    int ninside = 0;
    int n = 0;

    for (int i = 0; i < nsurf; i++) {
      m = surfs[i];
      line = &lines[m];
      if (line->transparent) continue;
      p1[0] = line->p1[0]; p1[1] = line->p1[1];
      p2[0] = line->p2[0]; p2[1] = line->p2[1];

      cline = &clines[n];
      cline->line = i;

      // clip PQ to cell and store as XY in Cline

      x = cline->x;
      y = cline->y;
      clip(p1,p2,x,y);

      // discard clipped line if only one point, increment touchcount
      // tally inside/outside for all removed lines
      // outside = line norm from line ctr points towards cell ctr
      // inside = line norm from line ctr points away from cell ctr
      // cbox = cell center pt, cmid = line center pt, l2b = cbox-cmid

      if (x[0] == y[0] && x[1] == y[1]) {
        touchcount++;
        cbox[0] = 0.5*(lo[0]+hi[0]);
        cbox[1] = 0.5*(lo[1]+hi[1]);
        cbox[2] = 0.0;
        pp1 = line->p1;
        pp2 = line->p2;
        cmid[0] = 0.5*(pp1[0]+pp2[0]);
        cmid[1] = 0.5*(pp1[1]+pp2[1]);
        cmid[2] = 0.0;
        l2b[0] = cbox[0] - cmid[0];
        l2b[1] = cbox[1] - cmid[1];
        l2b[2] = cbox[2] - cmid[2];
        norm = line->norm;
        double dot = norm[0]*l2b[0] + norm[1]*l2b[1] + norm[2]*l2b[2];
        if (dot > 0.0) noutside++;
        if (dot < 0.0) ninside++;
        continue;
      }

      // discard clipped line if lies on a cell edge w/ normal out of cell
      // increment grazecount in this case

      if (ptflag(x) == BORDER && ptflag(y) == BORDER) {
        int edge = sameedge(x,y);
        if (edge) {
          grazecount++;
          norm = line->norm;
          if (edge == 1 && norm[0] < 0.0) continue;
          if (edge == 2 && norm[0] > 0.0) continue;
          if (edge == 3 && norm[1] < 0.0) continue;
          if (edge == 4 && norm[1] > 0.0) continue;
          grazecount--;
        }
      }

      n++;
    }

    // if no lines, set touchmark which will be used to mark corner points
    // only set touchmark if all single-point deleted lines had same orientation

    touchmark = UNKNOWN;
    if (n == 0) {
      if (ninside && noutside == 0) touchmark = INSIDE;
      else if (noutside && ninside == 0) touchmark = OUTSIDE;
    }

    nclines = n;
  }

  /* --------------------------------------------------------------------
     Cut2d::weiler_build(): the Weiler/Atherton point data structure
     3 possible error returns, then the 4 cell corner pts are added and
       the loop and cell perimeter linked lists created
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int weiler_build()
  {
    int i,j;
    int firstpt,lastpt,nextpt;
    const double *pt;

    int nlines = nclines;
    int npt = 0;

    // add each cline end pt to points
    // set x,type,next,line for each pt

    for (i = 0; i < nlines; i++) {

      // 1st point in cline

      pt = clines[i].x;
      for (j = 0; j < npt; j++)
        if (pt[0] == points[j].x[0] && pt[1] == points[j].x[1]) break;

      if (j < npt) {
        if (points[j].type == ENTRY || points[j].type == TWO) return 1;
        points[j].type = TWO;
        points[j].line = clines[i].line;
        firstpt = j;
      } else {
        points[npt].x[0] = pt[0];
        points[npt].x[1] = pt[1];
        points[npt].type = ENTRY;
        points[npt].line = clines[i].line;
        firstpt = npt;
        npt++;
      }

      // 2nd point in cline

      pt = clines[i].y;
      for (j = 0; j < npt; j++)
        if (pt[0] == points[j].x[0] && pt[1] == points[j].x[1]) break;

      if (j < npt) {
        if (points[j].type == EXIT || points[j].type == TWO) return 2;
        points[j].type = TWO;
        points[firstpt].next = j;
      } else {
        points[npt].x[0] = pt[0];
        points[npt].x[1] = pt[1];
        points[npt].type = EXIT;
        points[firstpt].next = npt;
        npt++;
      }
    }

    // error check that every singlet point is on cell border

    for (i = 0; i < npt; i++)
      if (points[i].type != TWO && ptflag(points[i].x) != BORDER) return 3;

    // add 4 cell CORNER pts to points
    // only if corner pt is not already an ENTRY or EXIT pt
    // if a TWO pt, still add corner pt as CORNER
    // corner flag = 0,1,2,3 for LL,LR,UL,UR, same as in Grid::ChildInfo,
    //   but ordering of corner pts in linked list is LL,LR,UR,UL
    // side = 0,1,2,3 for lower,right,upper,left = traversal order

    double cpt[2];
    int ipt1,ipt2,ipt3,ipt4;

    for (i = 0; i < npt; i++) points[i].corner = -1;

    cpt[0] = lo[0]; cpt[1] = lo[1];
    for (j = 0; j < npt; j++)
      if (cpt[0] == points[j].x[0] && cpt[1] == points[j].x[1]) break;
    if (j == npt || points[j].type == TWO) {
      points[npt].x[0] = cpt[0];
      points[npt].x[1] = cpt[1];
      points[npt].type = CORNER;
      ipt1 = npt++;
    } else ipt1 = j;
    points[ipt1].corner = 0;
    points[ipt1].side = 0;
    points[ipt1].value = lo[0];

    cpt[0] = hi[0]; cpt[1] = lo[1];
    for (j = 0; j < npt; j++)
      if (cpt[0] == points[j].x[0] && cpt[1] == points[j].x[1]) break;
    if (j == npt || points[j].type == TWO) {
      points[npt].x[0] = cpt[0];
      points[npt].x[1] = cpt[1];
      points[npt].type = CORNER;
      ipt2 = npt++;
    } else ipt2 = j;
    points[ipt2].corner = 1;
    points[ipt2].side = 1;
    points[ipt2].value = lo[1];

    cpt[0] = hi[0]; cpt[1] = hi[1];
    for (j = 0; j < npt; j++)
      if (cpt[0] == points[j].x[0] && cpt[1] == points[j].x[1]) break;
    if (j == npt || points[j].type == TWO) {
      points[npt].x[0] = cpt[0];
      points[npt].x[1] = cpt[1];
      points[npt].type = CORNER;
      ipt3 = npt++;
    } else ipt3 = j;
    points[ipt3].corner = 3;
    points[ipt3].side = 2;
    points[ipt3].value = hi[0];

    cpt[0] = lo[0]; cpt[1] = hi[1];
    for (j = 0; j < npt; j++)
      if (cpt[0] == points[j].x[0] && cpt[1] == points[j].x[1]) break;
    if (j == npt || points[j].type == TWO) {
      points[npt].x[0] = cpt[0];
      points[npt].x[1] = cpt[1];
      points[npt].type = CORNER;
      ipt4 = npt++;
    } else ipt4 = j;
    points[ipt4].corner = 2;
    points[ipt4].side = 3;
    points[ipt4].value = hi[1];

    npoints = npt;

    // create initial counter-clockwise linked list around cell perimeter
    // just the 4 corner pts

    firstpt = ipt1;
    lastpt = ipt4;

    points[ipt1].cprev = -1;
    points[ipt1].cnext = ipt2;
    points[ipt2].cprev = ipt1;
    points[ipt2].cnext = ipt3;
    points[ipt3].cprev = ipt2;
    points[ipt3].cnext = ipt4;
    points[ipt4].cprev = ipt3;
    points[ipt4].cnext = -1;

    // add all non-corner ENTRY/EXIT pts to counter-clockwise linked list
    // side = 0,1,2,3 for lower,right,upper,left sides
    // value = coord of pt along the side it is on

    int ipt,iprev,side;
    double value;

    iprev = -1;
    for (i = 0; i < npt; i++) {
      if (points[i].type == TWO || points[i].type == CORNER) continue;
      if (points[i].corner >= 0) continue;

      side = whichside(points[i].x);
      if (side % 2) value = points[i].x[1];
      else value = points[i].x[0];

      // interleave Ith point into linked list between firstpt and lastpt
      // insertion location is between iprev and ipt

      ipt = firstpt;
      while (ipt >= 0) {
        if (side < points[ipt].side) break;
        if (side == points[ipt].side) {
          if (side < 2) {
            if (value < points[ipt].value) break;
          } else {
            if (value > points[ipt].value) break;
          }
        }
        iprev = ipt;
        ipt = points[ipt].cnext;
      }

      points[i].side = side;
      points[i].value = value;
      points[i].cprev = iprev;
      points[i].cnext = ipt;

      points[iprev].cnext = i;
      if (ipt >= 0) points[ipt].cprev = i;
      else lastpt = i;
    }

    // set next field for cell perimeter points in linked list
    // do not reset next for ENTRY pts
    // after loop, explicitly connect lastpt to firstpt

    ipt = firstpt;
    while (ipt >= 0) {
      nextpt = points[ipt].cnext;
      if (points[ipt].type != ENTRY) points[ipt].next = nextpt;
      ipt = nextpt;
    }
    if (points[lastpt].type != ENTRY) points[lastpt].next = firstpt;

    return 0;
  }

  /* --------------------------------------------------------------------
     Cut2d::weiler_loops(): one Loop for each closed path in the points
     discard a path if did not close on itself, b/c just corner pts
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void weiler_loops()
  {
    int n = npoints;
    for (int i = 0; i < n; i++) used[i] = 0;

    int ipt,iflag,cflag,ncount,firstpt,nextpt;
    double area;
    const double *x,*y;

    int nloop = 0;

    for (int i = 0; i < n; i++) {
      if (used[i]) continue;
      area = 0.0;
      iflag = cflag = 1;
      ncount = 0;

      ipt = firstpt = i;
      x = points[ipt].x;

      while (!used[ipt]) {
        used[ipt] = 1;
        ncount++;
        if (points[ipt].type != TWO) iflag = 0;
        if (points[ipt].type != CORNER) cflag = 0;
        nextpt = points[ipt].next;
        y = points[nextpt].x;
        if (axisymmetric)
          area -= MathConst::MY_PI3 *
            (x[1]*x[1] + x[1]*y[1] + y[1]*y[1]) * (y[0]-x[0]);
        else area -= (0.5*(x[1]+y[1]) - lo[1]) * (y[0]-x[0]);
        x = y;
        ipt = nextpt;
        if (ipt == firstpt) break;
      }
      if (ipt != firstpt) continue;

      loops[nloop].area = area;
      loops[nloop].active = 1;
      if (iflag) loops[nloop].flag = INTERIOR;
      else if (cflag) loops[nloop].flag = BORDER;
      else loops[nloop].flag = INTBORD;
      loops[nloop].n = ncount;
      loops[nloop].first = firstpt;
      nloop++;
    }

    nloops = nloop;
  }

  /* --------------------------------------------------------------------
     Cut2d::loop2pg(): one PG (polygon) for each disjoint flow area
     error returns 4,5,6 as the host
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int loop2pg()
  {
    int positive = 0;
    int negative = 0;

    int nloop = nloops;

    for (int i = 0; i < nloop; i++) {
      if (loops[i].area > 0.0) positive++;
      else if (loops[i].area < 0.0) negative++;
    }

    // if no positive areas, cell is entirely inside the surf, caller handles it

    if (positive == 0) return 4;

    // do not allow mulitple positive with one or more negative

    if (positive > 1 && negative) return 5;

    // if multiple positive, mark positive BORDER loop as inactive if exists

    if (positive > 1) {
      for (int i = 0; i < nloop; i++)
        if (loops[i].flag == BORDER) {
          loops[i].active = 0;
          positive--;
        }
    }

    // positive = 1 means 1 PG with area = sum of all pos/neg loops
    // positive > 1 means each loop is a PG

    if (positive == 1) {
      double area = 0.0;
      int prev = -1;
      int count = 0;
      int first = -1;

      for (int i = 0; i < nloop; i++) {
        if (!loops[i].active) continue;
        area += loops[i].area;
        count++;
        if (prev < 0) first = i;
        else loops[prev].next = i;
        prev = i;
      }
      loops[prev].next = -1;

      // do not allow an inverse donut geometry, positive inside a negative

      if (area < 0.0) return 6;

      pgs[0].area = area;
      pgs[0].n = count;
      pgs[0].first = first;

    } else {
      int m = 0;
      for (int i = 0; i < nloop; i++) {
        if (!loops[i].active) continue;
        pgs[m].area = loops[i].area;
        pgs[m].n = 1;
        pgs[m].first = i;
        m++;
        loops[i].next = -1;
      }
    }

    npgs = positive;
    return 0;
  }

  /* --------------------------------------------------------------------
     Cut2d::create_surfmap(): surfmap[i] = PG the Ith line is assigned to
     -1 if the line did not end up in a PG
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void create_surfmap(int *surfmap)
  {
    for (int i = 0; i < nsurf; i++) surfmap[i] = -1;

    int iloop,nloop,mloop,ipt,npt,mpt;

    for (int ipg = 0; ipg < npgs; ipg++) {
      nloop = pgs[ipg].n;
      mloop = pgs[ipg].first;
      for (iloop = 0; iloop < nloop; iloop++) {
        npt = loops[mloop].n;
        mpt = loops[mloop].first;
        for (ipt = 0; ipt < npt; ipt++) {
          if (points[mpt].type == TWO || points[mpt].type == ENTRY)
            surfmap[points[mpt].line] = ipg;
          mpt = points[mpt].next;
        }
        mloop = loops[mloop].next;
      }
    }
  }

  /* --------------------------------------------------------------------
     Cut2d::split_point_explicit(): a surf point in or on the cell
     return xsplit = coords of point, xsub = its sub-cell index
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int split_point_explicit(int *surfmap, double *xsplit, int &xsub)
  {
    int iline;
    const double *x1,*x2;
    double a[2],b[2];

    // if end pt of any line with non-negative surfmap is in/on cell, return

    for (int i = 0; i < nsurf; i++) {
      if (surfmap[i] < 0) continue;
      iline = surfs[i];
      x1 = lines[iline].p1;
      x2 = lines[iline].p2;
      if (ptflag(x1) != EXTERIOR) {
        xsplit[0] = x1[0]; xsplit[1] = x1[1];
        xsub = surfmap[i];
        return 0;
      }
      if (ptflag(x2) != EXTERIOR) {
        xsplit[0] = x2[0]; xsplit[1] = x2[1];
        xsub = surfmap[i];
        return 0;
      }
    }

    // clip 1st line with non-negative surfmap to cell, and return clip point

    for (int i = 0; i < nsurf; i++) {
      if (surfmap[i] < 0) continue;
      iline = surfs[i];
      x1 = lines[iline].p1;
      x2 = lines[iline].p2;
      clip(x1,x2,a,b);
      xsplit[0] = a[0]; xsplit[1] = a[1];
      xsub = surfmap[i];
      return 0;
    }

    return 7;
  }

  /* --------------------------------------------------------------------
     Cut2d::clip(): PQ is known to intersect cell, return AB = clipped PQ
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void clip(const double *p, const double *q, double *a, double *b) const
  {
    double x,y;

    a[0] = p[0]; a[1] = p[1];
    b[0] = q[0]; b[1] = q[1];

    if (p[0] >= lo[0] && p[0] <= hi[0] &&
        p[1] >= lo[1] && p[1] <= hi[1] &&
        q[0] >= lo[0] && q[0] <= hi[0] &&
        q[1] >= lo[1] && q[1] <= hi[1]) return;

    if (a[0] < lo[0] || b[0] < lo[0]) {
      y = a[1] + (lo[0]-a[0])/(b[0]-a[0])*(b[1]-a[1]);
      if (a[0] < lo[0]) {
        a[0] = lo[0]; a[1] = y;
      } else {
        b[0] = lo[0]; b[1] = y;
      }
    }
    if (a[0] > hi[0] || b[0] > hi[0]) {
      y = a[1] + (hi[0]-a[0])/(b[0]-a[0])*(b[1]-a[1]);
      if (a[0] > hi[0]) {
        a[0] = hi[0]; a[1] = y;
      } else {
        b[0] = hi[0]; b[1] = y;
      }
    }
    if (a[1] < lo[1] || b[1] < lo[1]) {
      x = a[0] + (lo[1]-a[1])/(b[1]-a[1])*(b[0]-a[0]);
      if (a[1] < lo[1]) {
        a[0] = x; a[1] = lo[1];
      } else {
        b[0] = x; b[1] = lo[1];
      }
    }
    if (a[1] > hi[1] || b[1] > hi[1]) {
      x = a[0] + (hi[1]-a[1])/(b[1]-a[1])*(b[0]-a[0]);
      if (a[1] > hi[1]) {
        a[0] = x; a[1] = hi[1];
      } else {
        b[0] = x; b[1] = hi[1];
      }
    }
  }

  /* --------------------------------------------------------------------
     Cut2d::sameedge(): 1,2,3,4 if A,B both on left,right,lower,upper
       edge of cell, else 0
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int sameedge(const double *a, const double *b) const
  {
    if (a[0] == lo[0] && b[0] == lo[0]) return 1;
    if (a[0] == hi[0] && b[0] == hi[0]) return 2;
    if (a[1] == lo[1] && b[1] == lo[1]) return 3;
    if (a[1] == hi[1] && b[1] == hi[1]) return 4;
    return 0;
  }

  /* --------------------------------------------------------------------
     Cut2d::ptflag(): EXTERIOR, INTERIOR or BORDER of the cell
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int ptflag(const double *pt) const
  {
    double x = pt[0];
    double y = pt[1];
    if (x < lo[0] || x > hi[0] || y < lo[1] || y > hi[1]) return EXTERIOR;
    if (x > lo[0] && x < hi[0] && y > lo[1] && y < hi[1]) return INTERIOR;
    return BORDER;
  }

  /* --------------------------------------------------------------------
     Cut2d::whichside(): 0,1,2,3 = lower,right,upper,left side of cell
       the border pt is on, -1 if not on the border
  -------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int whichside(const double *pt) const
  {
    if (pt[0] == lo[0]) return 3;
    if (pt[0] == hi[0]) return 1;
    if (pt[1] == lo[1]) return 0;
    if (pt[1] == hi[1]) return 2;
    return -1;
  }
};

}

#endif
