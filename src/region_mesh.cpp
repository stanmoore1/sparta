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

#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "region_mesh.h"
#include "stl_reader.h"
#include "comm.h"
#include "input.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

#define EPSILON 1.0e-9       // relative tolerance for a degenerate ray hit
#define JITTER 1.0e-6        // relative offset used to move a degenerate ray
#define MAXJITTER 8          // max # of times a degenerate ray is moved
#define WATERTIGHT 1.0e-6    // relative tolerance on the closure of the mesh
#define MAXBIN 1024          // max # of ray-casting bins in y or z

// angle between successive jittered rays, the golden angle in radians
// keeps the rays from repeating a direction that was already degenerate

#define GOLDEN_ANGLE 2.39996322972865332

/* ---------------------------------------------------------------------- */

RegMesh::RegMesh(SPARTA *sparta, int narg, char **arg) :
  Region(sparta, narg, arg)
{
  if (narg < 3) error->all(FLERR,"Illegal region mesh command");

  tris = NULL;
  tnorm = NULL;
  binfirst = binlist = NULL;

  // read the triangles, every proc keeps a copy of the entire mesh
  // the reader owns the array it fills, so keep our own copy of it

  STLReader reader(sparta);
  double **readtris;
  ntri = reader.read_file(arg[2],readtris);

  memory->create(tris,ntri,9,"region/mesh:tris");
  memcpy(&tris[0][0],&readtris[0][0],9*ntri*sizeof(double));

  // optional args
  // side is parsed by the base class, everything else transforms the mesh
  // transforms are applied in the order they appear, as in read_surf

  origin[0] = origin[1] = origin[2] = 0.0;
  interior = 1;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"origin") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal region mesh command");
      origin[0] = input->numeric(FLERR,arg[iarg+1]);
      origin[1] = input->numeric(FLERR,arg[iarg+2]);
      origin[2] = input->numeric(FLERR,arg[iarg+3]);
      iarg += 4;
    } else if (strcmp(arg[iarg],"trans") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal region mesh command");
      double dx = input->numeric(FLERR,arg[iarg+1]);
      double dy = input->numeric(FLERR,arg[iarg+2]);
      double dz = input->numeric(FLERR,arg[iarg+3]);
      origin[0] += dx;
      origin[1] += dy;
      origin[2] += dz;
      translate(dx,dy,dz);
      iarg += 4;
    } else if (strcmp(arg[iarg],"scale") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal region mesh command");
      double sx = input->numeric(FLERR,arg[iarg+1]);
      double sy = input->numeric(FLERR,arg[iarg+2]);
      double sz = input->numeric(FLERR,arg[iarg+3]);
      scale(sx,sy,sz);
      iarg += 4;
    } else if (strcmp(arg[iarg],"rotate") == 0) {
      if (iarg+5 > narg) error->all(FLERR,"Illegal region mesh command");
      double theta = input->numeric(FLERR,arg[iarg+1]);
      double rx = input->numeric(FLERR,arg[iarg+2]);
      double ry = input->numeric(FLERR,arg[iarg+3]);
      double rz = input->numeric(FLERR,arg[iarg+4]);
      if (rx == 0.0 && ry == 0.0 && rz == 0.0)
        error->all(FLERR,"Illegal region mesh command");
      rotate(theta,rx,ry,rz);
      iarg += 5;
    } else if (strcmp(arg[iarg],"side") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal region mesh command");
      options(2,&arg[iarg]);
      iarg += 2;
    } else error->all(FLERR,"Illegal region mesh command");
  }

  // bounding box, enclosed volume, and the bins used by inside()

  setup();

  // extent of the mesh
  // as for the other styles, only an interior region has a bounding box

  if (interior) {
    bboxflag = 1;
    extent_xlo = bblo[0];
    extent_xhi = bbhi[0];
    extent_ylo = bblo[1];
    extent_yhi = bbhi[1];
    extent_zlo = bblo[2];
    extent_zhi = bbhi[2];
  } else bboxflag = 0;

  if (comm->me == 0) {
    if (screen) {
      fprintf(screen,"  %g %g %g to %g %g %g mesh bounding box\n",
              bblo[0],bblo[1],bblo[2],bbhi[0],bbhi[1],bbhi[2]);
      fprintf(screen,"  %g enclosed mesh volume\n",volume);
    }
    if (logfile) {
      fprintf(logfile,"  %g %g %g to %g %g %g mesh bounding box\n",
              bblo[0],bblo[1],bblo[2],bbhi[0],bbhi[1],bbhi[2]);
      fprintf(logfile,"  %g enclosed mesh volume\n",volume);
    }
  }
}

/* ---------------------------------------------------------------------- */

RegMesh::~RegMesh()
{
  memory->destroy(tris);
  memory->destroy(tnorm);
  memory->destroy(binfirst);
  memory->destroy(binlist);
}

/* ----------------------------------------------------------------------
   inside = 1 if x,y,z is inside the closed surface, else 0
   cast a ray in +x from the point and count the triangles it crosses
   an odd count means the point is inside
------------------------------------------------------------------------- */

int RegMesh::inside(double *x)
{
  if (x[0] < bblo[0] || x[0] > bbhi[0] ||
      x[1] < bblo[1] || x[1] > bbhi[1] ||
      x[2] < bblo[2] || x[2] > bbhi[2]) return 0;

  // a ray that grazes an edge or vertex cannot be counted reliably
  // move it a little in the y-z plane and cast it again

  double dy = 0.0;
  double dz = 0.0;

  for (int attempt = 0; attempt <= MAXJITTER; attempt++) {
    int ncross = crossings(x[0],x[1]+dy,x[2]+dz);
    if (ncross >= 0) return ncross & 1;

    double angle = GOLDEN_ANGLE * (attempt+1);
    double offset = jitter * (attempt+1);
    dy = offset * cos(angle);
    dz = offset * sin(angle);
  }

  // every ray was degenerate, so the point is on the surface itself
  // as for the other region styles, the surface counts as inside

  return 1;
}

/* ----------------------------------------------------------------------
   count the triangles crossed by a +x ray from the point px,py,pz
   return -1 if the ray grazes an edge or vertex, so the count is unreliable
------------------------------------------------------------------------- */

int RegMesh::crossings(double px, double py, double pz)
{
  int iy = static_cast<int> ((py-bblo[1]) * invbiny);
  iy = MAX(0,MIN(iy,nbiny-1));
  int iz = static_cast<int> ((pz-bblo[2]) * invbinz);
  iz = MAX(0,MIN(iz,nbinz-1));
  int ibin = iz*nbiny + iy;

  int ncross = 0;

  for (int m = binfirst[ibin]; m < binfirst[ibin+1]; m++) {
    int itri = binlist[m];
    double *t = tris[itri];

    // 2x the signed areas of the 3 triangles that the point makes with each
    // edge, in the y-z plane the ray projects onto
    // the point projects inside the triangle if all 3 have the same sign

    double d0 = (t[4]-t[1])*(pz-t[2]) - (t[5]-t[2])*(py-t[1]);
    double d1 = (t[7]-t[4])*(pz-t[5]) - (t[8]-t[5])*(py-t[4]);
    double d2 = (t[1]-t[7])*(pz-t[8]) - (t[2]-t[8])*(py-t[7]);

    // d0+d1+d2 is 2x the signed area of the projected triangle,
    // which is also the x component of the triangle normal
    // comparing it to the length of the normal tests how edge-on the
    // triangle is to the ray, independent of how big the triangle is

    double area2 = d0 + d1 + d2;
    double epsarea = EPSILON * tnorm[itri];
    if (fabs(area2) <= epsarea) continue;

    int nneg = 0;
    int npos = 0;
    int nzero = 0;
    if (d0 > epsarea) npos++;
    else if (d0 < -epsarea) nneg++;
    else nzero++;
    if (d1 > epsarea) npos++;
    else if (d1 < -epsarea) nneg++;
    else nzero++;
    if (d2 > epsarea) npos++;
    else if (d2 < -epsarea) nneg++;
    else nzero++;

    if (npos && nneg) continue;      // ray misses the triangle
    if (nzero) return -1;            // ray grazes an edge or a vertex

    // x where the ray pierces the triangle, from the barycentric coords

    double xhit = (d1*t[0] + d2*t[3] + d0*t[6]) / area2;
    if (fabs(xhit-px) < epslen) return -1;   // point is on the triangle
    if (xhit > px) ncross++;
  }

  return ncross;
}

/* ----------------------------------------------------------------------
   bounding box, enclosed volume, tolerances, and ray-casting bins
------------------------------------------------------------------------- */

void RegMesh::setup()
{
  // bounding box around all the vertices

  bblo[0] = bbhi[0] = tris[0][0];
  bblo[1] = bbhi[1] = tris[0][1];
  bblo[2] = bbhi[2] = tris[0][2];

  for (int i = 0; i < ntri; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++) {
        bblo[k] = MIN(bblo[k],tris[i][3*j+k]);
        bbhi[k] = MAX(bbhi[k],tris[i][3*j+k]);
      }

  // tolerances scale with the size of the mesh
  // the jitter is much larger than the tolerance it has to escape

  double meshlen = MAX(bbhi[0]-bblo[0],MAX(bbhi[1]-bblo[1],bbhi[2]-bblo[2]));
  if (meshlen == 0.0) error->all(FLERR,"Region mesh has zero extent");

  epslen = EPSILON * meshlen;
  jitter = JITTER * meshlen;

  // normal of each triangle, its length is 2x the area
  // enclosed volume from the divergence theorem
  // the area-weighted normals of a closed surface sum to zero

  memory->create(tnorm,ntri,"region/mesh:tnorm");

  double nsum[3] = {0.0,0.0,0.0};
  double asum = 0.0;
  double vsum = 0.0;

  for (int i = 0; i < ntri; i++) {
    double *a = &tris[i][0];
    double *b = &tris[i][3];
    double *c = &tris[i][6];
    double e1[3],e2[3],n[3];

    MathExtra::sub3(b,a,e1);
    MathExtra::sub3(c,a,e2);
    MathExtra::cross3(e1,e2,n);
    tnorm[i] = MathExtra::len3(n);

    asum += tnorm[i];
    nsum[0] += n[0];
    nsum[1] += n[1];
    nsum[2] += n[2];

    MathExtra::cross3(b,c,e1);
    vsum += MathExtra::dot3(a,e1);
  }

  volume = fabs(vsum) / 6.0;

  if (MathExtra::len3(nsum) > WATERTIGHT * asum)
    error->warning(FLERR,"Region mesh surface is not watertight");

  // bin the triangles by their y-z extent so a +x ray only has to test
  // the triangles in one bin
  // aim for one triangle per bin, with square bins in the y-z plane

  double ylen = bbhi[1] - bblo[1];
  double zlen = bbhi[2] - bblo[2];

  double binsize = sqrt(ylen*zlen/ntri);
  if (binsize == 0.0) binsize = meshlen;

  nbiny = MAX(1,MIN(static_cast<int> (ylen/binsize) + 1,MAXBIN));
  nbinz = MAX(1,MIN(static_cast<int> (zlen/binsize) + 1,MAXBIN));

  invbiny = (ylen > 0.0) ? nbiny/ylen : 0.0;
  invbinz = (zlen > 0.0) ? nbinz/zlen : 0.0;

  int nbin = nbiny*nbinz;
  memory->create(binfirst,nbin+1,"region/mesh:binfirst");
  for (int i = 0; i <= nbin; i++) binfirst[i] = 0;

  // count the triangles in each bin, then turn the counts into offsets

  int iylo,iyhi,izlo,izhi;

  for (int i = 0; i < ntri; i++) {
    bin_range(i,iylo,iyhi,izlo,izhi);
    for (int iz = izlo; iz <= izhi; iz++)
      for (int iy = iylo; iy <= iyhi; iy++)
        binfirst[iz*nbiny+iy+1]++;
  }

  for (int i = 0; i < nbin; i++) binfirst[i+1] += binfirst[i];

  memory->create(binlist,binfirst[nbin],"region/mesh:binlist");

  int *next;
  memory->create(next,nbin,"region/mesh:next");
  for (int i = 0; i < nbin; i++) next[i] = binfirst[i];

  for (int i = 0; i < ntri; i++) {
    bin_range(i,iylo,iyhi,izlo,izhi);
    for (int iz = izlo; iz <= izhi; iz++)
      for (int iy = iylo; iy <= iyhi; iy++)
        binlist[next[iz*nbiny+iy]++] = i;
  }

  memory->destroy(next);
}

/* ----------------------------------------------------------------------
   range of bins that triangle I overlaps in the y-z plane
------------------------------------------------------------------------- */

void RegMesh::bin_range(int i, int &iylo, int &iyhi, int &izlo, int &izhi)
{
  double *t = tris[i];

  double ylo = MIN(t[1],MIN(t[4],t[7]));
  double yhi = MAX(t[1],MAX(t[4],t[7]));
  double zlo = MIN(t[2],MIN(t[5],t[8]));
  double zhi = MAX(t[2],MAX(t[5],t[8]));

  iylo = MAX(0,MIN(static_cast<int> ((ylo-bblo[1])*invbiny),nbiny-1));
  iyhi = MAX(0,MIN(static_cast<int> ((yhi-bblo[1])*invbiny),nbiny-1));
  izlo = MAX(0,MIN(static_cast<int> ((zlo-bblo[2])*invbinz),nbinz-1));
  izhi = MAX(0,MIN(static_cast<int> ((zhi-bblo[2])*invbinz),nbinz-1));
}

/* ----------------------------------------------------------------------
   translate the mesh by dx,dy,dz
------------------------------------------------------------------------- */

void RegMesh::translate(double dx, double dy, double dz)
{
  for (int i = 0; i < ntri; i++)
    for (int j = 0; j < 3; j++) {
      tris[i][3*j+0] += dx;
      tris[i][3*j+1] += dy;
      tris[i][3*j+2] += dz;
    }
}

/* ----------------------------------------------------------------------
   scale the mesh by sx,sy,sz around the origin point
------------------------------------------------------------------------- */

void RegMesh::scale(double sx, double sy, double sz)
{
  for (int i = 0; i < ntri; i++)
    for (int j = 0; j < 3; j++) {
      tris[i][3*j+0] = sx*(tris[i][3*j+0]-origin[0]) + origin[0];
      tris[i][3*j+1] = sy*(tris[i][3*j+1]-origin[1]) + origin[1];
      tris[i][3*j+2] = sz*(tris[i][3*j+2]-origin[2]) + origin[2];
    }
}

/* ----------------------------------------------------------------------
   rotate the mesh theta degrees around the axis rx,ry,rz thru the origin point
------------------------------------------------------------------------- */

void RegMesh::rotate(double theta, double rx, double ry, double rz)
{
  double r[3],q[4],d[3],dnew[3];
  double rotmat[3][3];

  theta *= MY_PI/180.0;

  r[0] = rx; r[1] = ry; r[2] = rz;
  MathExtra::norm3(r);
  MathExtra::axisangle_to_quat(r,theta,q);
  MathExtra::quat_to_mat(q,rotmat);

  for (int i = 0; i < ntri; i++)
    for (int j = 0; j < 3; j++) {
      d[0] = tris[i][3*j+0] - origin[0];
      d[1] = tris[i][3*j+1] - origin[1];
      d[2] = tris[i][3*j+2] - origin[2];
      MathExtra::matvec(rotmat,d,dnew);
      tris[i][3*j+0] = dnew[0] + origin[0];
      tris[i][3*j+1] = dnew[1] + origin[1];
      tris[i][3*j+2] = dnew[2] + origin[2];
    }
}
