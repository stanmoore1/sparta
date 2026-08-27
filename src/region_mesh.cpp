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
#include "utils.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

#define BIG 1.0e20           // same "infinity" the other region styles use
#define EPSILON 1.0e-9       // relative tolerance for a degenerate ray hit
#define JITTER 1.0e-6        // relative offset used to move a degenerate ray
#define MAXJITTER 8          // max # of times a degenerate ray is moved
#define CLOSED 1.0e-6        // relative tolerance on the closure of the mesh
#define MAXBIN 1024          // max # of ray-casting bins in y or z
#define MAXLINE 1024
#define MAXWORD 16           // more words on a line than any valid format has

// angle between successive jittered rays, the golden angle in radians
// keeps the rays from repeating a direction that was already degenerate

#define GOLDEN_ANGLE 2.39996322972865332

/* ----------------------------------------------------------------------
   return the next non-blank line of a file with any comment stripped off,
   or NULL at end of file
------------------------------------------------------------------------- */

static char *next_line(FILE *fp, char *line)
{
  while (fgets(line,MAXLINE,fp)) {
    char *ptr = strchr(line,'#');
    if (ptr) *ptr = '\0';
    if (strspn(line," \t\n\r") != strlen(line)) return line;
  }
  return NULL;
}

/* ----------------------------------------------------------------------
   split a line into whitespace delimited words, return # of words
   words point into line, which is modified
------------------------------------------------------------------------- */

static int split_line(char *line, char **words)
{
  int nwords = 0;
  char *ptr = strtok(line," \t\n\r\f");

  while (ptr && nwords < MAXWORD) {
    words[nwords++] = ptr;
    ptr = strtok(NULL," \t\n\r\f");
  }

  return nwords;
}

/* ---------------------------------------------------------------------- */

RegMesh::RegMesh(SPARTA *sparta, int narg, char **arg) :
  Region(sparta, narg, arg)
{
  if (narg < 3) error->all(FLERR,"Illegal region mesh command");

  verts = NULL;
  esize = NULL;
  binfirst = binlist = NULL;

  // read the surface, every proc keeps a copy of the entire mesh

  read_mesh(arg[2]);

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
  // a 2d mesh is a closed curve in x-y, so the region is a prism in z

  if (interior) {
    bboxflag = 1;
    extent_xlo = bblo[0];
    extent_xhi = bbhi[0];
    extent_ylo = bblo[1];
    extent_yhi = bbhi[1];
    if (meshdim == 3) {
      extent_zlo = bblo[2];
      extent_zhi = bbhi[2];
    } else {
      extent_zlo = -BIG;
      extent_zhi = BIG;
    }
  } else bboxflag = 0;

  if (comm->me == 0) {
    const char *what = (meshdim == 2) ? "area" : "volume";
    if (screen) {
      fprintf(screen,"  %g %g %g to %g %g %g mesh bounding box\n",
              bblo[0],bblo[1],bblo[2],bbhi[0],bbhi[1],bbhi[2]);
      fprintf(screen,"  %g enclosed mesh %s\n",enclosed,what);
    }
    if (logfile) {
      fprintf(logfile,"  %g %g %g to %g %g %g mesh bounding box\n",
              bblo[0],bblo[1],bblo[2],bbhi[0],bbhi[1],bbhi[2]);
      fprintf(logfile,"  %g enclosed mesh %s\n",enclosed,what);
    }
  }
}

/* ---------------------------------------------------------------------- */

RegMesh::~RegMesh()
{
  memory->destroy(verts);
  memory->destroy(esize);
  memory->destroy(binfirst);
  memory->destroy(binlist);
}

/* ----------------------------------------------------------------------
   read the surface from an STL file or a SPARTA surf file
   which one it is is detected from the contents of the file
------------------------------------------------------------------------- */

void RegMesh::read_mesh(char *file)
{
  int stlflag = 0;

  if (comm->me == 0) {
    try {
      stlflag = STLReader::is_stl_file(file);
    } catch (std::exception &e) {
      error->one(FLERR,e.what());
    }
  }
  MPI_Bcast(&stlflag,1,MPI_INT,0,world);

  if (stlflag) read_stl_file(file);
  else read_surf_file(file);
}

/* ----------------------------------------------------------------------
   read triangles from an ASCII or binary STL file
------------------------------------------------------------------------- */

void RegMesh::read_stl_file(char *file)
{
  STLReader reader(sparta);
  double **tris;

  meshdim = 3;
  nelem = reader.read_file(file,tris);
  nvert = 3*nelem;

  // the reader owns the array it fills, so keep our own copy of it

  memory->create(verts,nvert,3,"region/mesh:verts");

  for (int i = 0; i < nelem; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
        verts[3*i+j][k] = tris[i][3*j+k];
}

/* ----------------------------------------------------------------------
   read lines or triangles from a SPARTA surf file, the same format
   read_surf reads, on proc 0, then broadcast them
   only the geometry is used, so a type column is skipped and the ids are
   ignored, but per-surf custom attributes are not supported
------------------------------------------------------------------------- */

void RegMesh::read_surf_file(char *file)
{
  if (comm->me == 0) {
    char line[MAXLINE];
    char *words[MAXWORD];
    double **pts = NULL;

    FILE *fp = fopen(file,"r");
    if (fp == NULL) surf_error(file,"Cannot open mesh file");

    // 1st line of the file is a comment

    if (fgets(line,MAXLINE,fp) == NULL)
      surf_error(file,"Unexpected end of surf file");

    // header = keyword lines with counts on them
    // it ends at the 1st line with no header keyword, the section keyword

    int npoint = 0;
    int nline = 0;
    int ntri = 0;

    while (1) {
      if (next_line(fp,line) == NULL) {
        line[0] = '\0';
        break;
      }
      if (strstr(line,"points")) npoint = atoi(line);
      else if (strstr(line,"lines")) nline = atoi(line);
      else if (strstr(line,"triangles")) ntri = atoi(line);
      else break;
    }

    if (nline && ntri)
      surf_error(file,"Surf file contains both lines and triangles");
    if (nline <= 0 && ntri <= 0)
      surf_error(file,"Surf file does not contain lines or triangles");

    meshdim = nline ? 2 : 3;
    nelem = nline ? nline : ntri;
    nvert = meshdim*nelem;
    memory->create(verts,nvert,3,"region/mesh:verts");

    // sections, in whatever order they appear
    // a Points section is optional, elements may carry their own coords

    int nelemread = 0;

    while (strlen(line)) {
      if (strstr(line,"Points")) {
        if (npoint <= 0) surf_error(file,"Surf file has no points count");
        memory->create(pts,npoint,3,"region/mesh:pts");

        for (int i = 0; i < npoint; i++) {
          if (next_line(fp,line) == NULL)
            surf_error(file,"Unexpected end of surf file");
          int nwords = split_line(line,words);
          if (nwords != 3 && nwords != 4)
            surf_error(file,"Incorrect point format in surf file");
          pts[i][0] = surf_numeric(file,words[1]);
          pts[i][1] = surf_numeric(file,words[2]);
          pts[i][2] = (nwords == 4) ? surf_numeric(file,words[3]) : 0.0;
        }

      } else if (strstr(line,"Lines") || strstr(line,"Triangles")) {
        int nvper = meshdim;

        // # of words per element, with and without a leading type column
        // an id is always present, and point indices replace the coords
        // if the file has a Points section

        int nbase = npoint ? 1+nvper : 1+3*nvper;
        if (meshdim == 2 && !npoint) nbase = 1+2*nvper;

        for (int i = 0; i < nelem; i++) {
          if (next_line(fp,line) == NULL)
            surf_error(file,"Unexpected end of surf file");
          int nwords = split_line(line,words);
          if (nwords != nbase && nwords != nbase+1)
            surf_error(file,(meshdim == 2) ?
                       "Incorrect line format in surf file" :
                       "Incorrect triangle format in surf file");

          // skip the id, and the type column if there is one

          int iw = (nwords == nbase) ? 1 : 2;

          for (int j = 0; j < nvper; j++) {
            double *v = verts[nvper*i+j];
            if (npoint) {
              int ipoint = surf_inumeric(file,words[iw++]);
              if (ipoint < 1 || ipoint > npoint)
                surf_error(file,"Invalid point index in surf file");
              v[0] = pts[ipoint-1][0];
              v[1] = pts[ipoint-1][1];
              v[2] = pts[ipoint-1][2];
            } else {
              v[0] = surf_numeric(file,words[iw++]);
              v[1] = surf_numeric(file,words[iw++]);
              v[2] = (meshdim == 3) ? surf_numeric(file,words[iw++]) : 0.0;
            }
          }
        }

        nelemread = nelem;

      } else surf_error(file,"Unknown section in surf file");

      if (next_line(fp,line) == NULL) line[0] = '\0';
    }

    memory->destroy(pts);
    fclose(fp);

    if (nelemread == 0)
      surf_error(file,(meshdim == 2) ?
                 "Surf file has no Lines section" :
                 "Surf file has no Triangles section");

    if (screen)
      fprintf(screen,"Reading %d %s from surf file %s\n",
              nelem,(meshdim == 2) ? "lines" : "triangles",file);
    if (logfile)
      fprintf(logfile,"Reading %d %s from surf file %s\n",
              nelem,(meshdim == 2) ? "lines" : "triangles",file);
  }

  bcast_verts();
}

/* ----------------------------------------------------------------------
   send the mesh proc 0 read to all the other procs
------------------------------------------------------------------------- */

void RegMesh::bcast_verts()
{
  MPI_Bcast(&meshdim,1,MPI_INT,0,world);
  MPI_Bcast(&nelem,1,MPI_INT,0,world);
  MPI_Bcast(&nvert,1,MPI_INT,0,world);

  if (comm->me) memory->create(verts,nvert,3,"region/mesh:verts");

  // allow for 3*nvert to exceed the max allowed size of a single MPI_Bcast()

  bigint ntotal = (bigint) nvert * 3;
  if (ntotal < MAXSMALLINT)
    MPI_Bcast(&verts[0][0],3*nvert,MPI_DOUBLE,0,world);
  else {
    double *source = &verts[0][0];
    bigint n = 0;
    while (n < ntotal) {
      int nsize = MIN(MAXSMALLINT,ntotal-n);
      MPI_Bcast(&source[n],nsize,MPI_DOUBLE,0,world);
      n += nsize;
    }
  }
}

/* ---------------------------------------------------------------------- */

void RegMesh::surf_error(const char *file, const char *mesg)
{
  char str[256];
  snprintf(str,256,"%s: %s",mesg,file);
  error->one(FLERR,str);
}

/* ----------------------------------------------------------------------
   convert a word of a surf file to a number
   Input::numeric() cannot be used here, it reports a bad value with
   Error::all(), which the other procs would never reach while proc 0 is
   the only one reading the file
------------------------------------------------------------------------- */

double RegMesh::surf_numeric(const char *file, const char *word)
{
  if (!utils::is_double(word))
    surf_error(file,"Expected floating point value in surf file");
  return atof(word);
}

/* ---------------------------------------------------------------------- */

int RegMesh::surf_inumeric(const char *file, const char *word)
{
  if (!utils::is_integer(word))
    surf_error(file,"Expected integer value in surf file");
  return atoi(word);
}

/* ----------------------------------------------------------------------
   inside = 1 if x,y,z is inside the closed surface, else 0
   cast a ray in +x from the point and count the elements it crosses
   an odd count means the point is inside
   a 2d mesh is a closed curve in the x-y plane, so the ray is cast in that
   plane and the z coord of the point does not matter
------------------------------------------------------------------------- */

int RegMesh::inside(double *x)
{
  if (x[0] < bblo[0] || x[0] > bbhi[0] ||
      x[1] < bblo[1] || x[1] > bbhi[1]) return 0;
  if (meshdim == 3 && (x[2] < bblo[2] || x[2] > bbhi[2])) return 0;

  // a ray that grazes an edge or a vertex cannot be counted reliably
  // move it a little across the ray direction and cast it again

  double dy = 0.0;
  double dz = 0.0;

  for (int attempt = 0; attempt <= MAXJITTER; attempt++) {
    int ncross = crossings(x[0],x[1]+dy,x[2]+dz);
    if (ncross >= 0) return ncross & 1;

    double offset = jitter * (attempt+1);
    if (meshdim == 2) {
      dy = (attempt & 1) ? -offset : offset;
      dz = 0.0;
    } else {
      double angle = GOLDEN_ANGLE * (attempt+1);
      dy = offset * cos(angle);
      dz = offset * sin(angle);
    }
  }

  // every ray was degenerate, so the point is on the surface itself
  // as for the other region styles, the surface counts as inside

  return 1;
}

/* ----------------------------------------------------------------------
   count the elements crossed by a +x ray from the point px,py,pz
   return -1 if the ray grazes an edge or a vertex, so the count is unreliable
------------------------------------------------------------------------- */

int RegMesh::crossings(double px, double py, double pz)
{
  int iy = static_cast<int> ((py-bblo[1]) * invbiny);
  iy = MAX(0,MIN(iy,nbiny-1));
  int iz = static_cast<int> ((pz-bblo[2]) * invbinz);
  iz = MAX(0,MIN(iz,nbinz-1));
  int ibin = iz*nbiny + iy;

  if (meshdim == 2)
    return crossings_line(px,py,binfirst[ibin],binfirst[ibin+1]);
  return crossings_tri(px,py,pz,binfirst[ibin],binfirst[ibin+1]);
}

/* ----------------------------------------------------------------------
   crossings of the line segments in one bin, for a 2d mesh
------------------------------------------------------------------------- */

int RegMesh::crossings_line(double px, double py, int mlo, int mhi)
{
  int ncross = 0;

  for (int m = mlo; m < mhi; m++) {
    int ielem = binlist[m];
    double *a = verts[2*ielem];
    double *b = verts[2*ielem+1];

    // how far the point is from each end of the segment, measured along y
    // the 2 are the same sign only if the segment spans the ray

    double d0 = py - a[1];
    double d1 = b[1] - py;

    // d0+d1 is the extent of the segment in y
    // comparing it to the length of the segment tests how parallel the
    // segment is to the ray, independent of how long the segment is

    double dy = d0 + d1;
    double epsproj = EPSILON * esize[ielem];
    if (fabs(dy) <= epsproj) continue;

    if (fabs(d0) <= epsproj || fabs(d1) <= epsproj) return -1;
    if ((d0 > 0.0) != (d1 > 0.0)) continue;

    // x where the ray crosses the segment, from the interpolation weights

    double xhit = (d1*a[0] + d0*b[0]) / dy;
    if (fabs(xhit-px) < epslen) return -1;   // point is on the segment
    if (xhit > px) ncross++;
  }

  return ncross;
}

/* ----------------------------------------------------------------------
   crossings of the triangles in one bin, for a 3d mesh
------------------------------------------------------------------------- */

int RegMesh::crossings_tri(double px, double py, double pz, int mlo, int mhi)
{
  int ncross = 0;

  for (int m = mlo; m < mhi; m++) {
    int ielem = binlist[m];
    double *a = verts[3*ielem];
    double *b = verts[3*ielem+1];
    double *c = verts[3*ielem+2];

    // 2x the signed areas of the 3 triangles that the point makes with each
    // edge, in the y-z plane the ray projects onto
    // the point projects inside the triangle if all 3 have the same sign

    double d0 = (b[1]-a[1])*(pz-a[2]) - (b[2]-a[2])*(py-a[1]);
    double d1 = (c[1]-b[1])*(pz-b[2]) - (c[2]-b[2])*(py-b[1]);
    double d2 = (a[1]-c[1])*(pz-c[2]) - (a[2]-c[2])*(py-c[1]);

    // d0+d1+d2 is 2x the signed area of the projected triangle,
    // which is also the x component of the triangle normal
    // comparing it to the length of the normal tests how edge-on the
    // triangle is to the ray, independent of how big the triangle is

    double area2 = d0 + d1 + d2;
    double epsproj = EPSILON * esize[ielem];
    if (fabs(area2) <= epsproj) continue;

    int nneg = 0;
    int npos = 0;
    int nzero = 0;
    if (d0 > epsproj) npos++;
    else if (d0 < -epsproj) nneg++;
    else nzero++;
    if (d1 > epsproj) npos++;
    else if (d1 < -epsproj) nneg++;
    else nzero++;
    if (d2 > epsproj) npos++;
    else if (d2 < -epsproj) nneg++;
    else nzero++;

    if (npos && nneg) continue;      // ray misses the triangle
    if (nzero) return -1;            // ray grazes an edge or a vertex

    // x where the ray pierces the triangle, from the barycentric coords

    double xhit = (d1*a[0] + d2*b[0] + d0*c[0]) / area2;
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

  bblo[0] = bbhi[0] = verts[0][0];
  bblo[1] = bbhi[1] = verts[0][1];
  bblo[2] = bbhi[2] = verts[0][2];

  for (int i = 0; i < nvert; i++)
    for (int k = 0; k < 3; k++) {
      bblo[k] = MIN(bblo[k],verts[i][k]);
      bbhi[k] = MAX(bbhi[k],verts[i][k]);
    }

  // tolerances scale with the size of the mesh
  // the jitter is much larger than the tolerance it has to escape

  double meshlen = MAX(bbhi[0]-bblo[0],MAX(bbhi[1]-bblo[1],bbhi[2]-bblo[2]));
  if (meshlen == 0.0) error->all(FLERR,"Region mesh has zero extent");

  epslen = EPSILON * meshlen;
  jitter = JITTER * meshlen;

  // size of each element = 2x area of a triangle, length of a line
  // enclosed volume from the divergence theorem, area from the shoelace sum
  // the area-weighted normals of a closed surface sum to zero

  memory->create(esize,nelem,"region/mesh:esize");

  double nsum[3] = {0.0,0.0,0.0};
  double ssum = 0.0;
  double gsum = 0.0;

  if (meshdim == 3) {
    for (int i = 0; i < nelem; i++) {
      double *a = verts[3*i];
      double *b = verts[3*i+1];
      double *c = verts[3*i+2];
      double e1[3],e2[3],n[3];

      MathExtra::sub3(b,a,e1);
      MathExtra::sub3(c,a,e2);
      MathExtra::cross3(e1,e2,n);
      esize[i] = MathExtra::len3(n);

      ssum += esize[i];
      nsum[0] += n[0];
      nsum[1] += n[1];
      nsum[2] += n[2];

      MathExtra::cross3(b,c,e1);
      gsum += MathExtra::dot3(a,e1);
    }
    enclosed = fabs(gsum) / 6.0;

  } else {
    for (int i = 0; i < nelem; i++) {
      double *a = verts[2*i];
      double *b = verts[2*i+1];

      double dx = b[0] - a[0];
      double dy = b[1] - a[1];
      esize[i] = sqrt(dx*dx + dy*dy);

      // the normal of a segment has the same length as the segment

      ssum += esize[i];
      nsum[0] += dy;
      nsum[1] -= dx;

      gsum += a[0]*b[1] - b[0]*a[1];
    }
    enclosed = fabs(gsum) / 2.0;
  }

  if (MathExtra::len3(nsum) > CLOSED * ssum)
    error->warning(FLERR,"Region mesh surface is not closed");

  // bin the elements by their y-z extent so a +x ray only has to test
  // the elements in one bin
  // in 3d aim for one triangle per bin, with square bins in the y-z plane
  // in 2d there is nothing to bin in z

  double ylen = bbhi[1] - bblo[1];
  double zlen = bbhi[2] - bblo[2];

  if (meshdim == 2) {
    nbiny = MAX(1,MIN(nelem,MAXBIN));
    nbinz = 1;
  } else {
    double binsize = sqrt(ylen*zlen/nelem);
    if (binsize == 0.0) binsize = meshlen;
    nbiny = MAX(1,MIN(static_cast<int> (ylen/binsize) + 1,MAXBIN));
    nbinz = MAX(1,MIN(static_cast<int> (zlen/binsize) + 1,MAXBIN));
  }

  invbiny = (ylen > 0.0) ? nbiny/ylen : 0.0;
  invbinz = (zlen > 0.0) ? nbinz/zlen : 0.0;

  int nbin = nbiny*nbinz;
  memory->create(binfirst,nbin+1,"region/mesh:binfirst");
  for (int i = 0; i <= nbin; i++) binfirst[i] = 0;

  // count the elements in each bin, then turn the counts into offsets

  int iylo,iyhi,izlo,izhi;

  for (int i = 0; i < nelem; i++) {
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

  for (int i = 0; i < nelem; i++) {
    bin_range(i,iylo,iyhi,izlo,izhi);
    for (int iz = izlo; iz <= izhi; iz++)
      for (int iy = iylo; iy <= iyhi; iy++)
        binlist[next[iz*nbiny+iy]++] = i;
  }

  memory->destroy(next);
}

/* ----------------------------------------------------------------------
   range of bins that element I overlaps in the y-z plane
------------------------------------------------------------------------- */

void RegMesh::bin_range(int i, int &iylo, int &iyhi, int &izlo, int &izhi)
{
  int first = meshdim*i;

  double ylo,yhi,zlo,zhi;
  ylo = yhi = verts[first][1];
  zlo = zhi = verts[first][2];

  for (int j = 1; j < meshdim; j++) {
    ylo = MIN(ylo,verts[first+j][1]);
    yhi = MAX(yhi,verts[first+j][1]);
    zlo = MIN(zlo,verts[first+j][2]);
    zhi = MAX(zhi,verts[first+j][2]);
  }

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
  for (int i = 0; i < nvert; i++) {
    verts[i][0] += dx;
    verts[i][1] += dy;
    verts[i][2] += dz;
  }
}

/* ----------------------------------------------------------------------
   scale the mesh by sx,sy,sz around the origin point
------------------------------------------------------------------------- */

void RegMesh::scale(double sx, double sy, double sz)
{
  for (int i = 0; i < nvert; i++) {
    verts[i][0] = sx*(verts[i][0]-origin[0]) + origin[0];
    verts[i][1] = sy*(verts[i][1]-origin[1]) + origin[1];
    verts[i][2] = sz*(verts[i][2]-origin[2]) + origin[2];
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

  for (int i = 0; i < nvert; i++) {
    d[0] = verts[i][0] - origin[0];
    d[1] = verts[i][1] - origin[1];
    d[2] = verts[i][2] - origin[2];
    MathExtra::matvec(rotmat,d,dnew);
    verts[i][0] = dnew[0] + origin[0];
    verts[i][1] = dnew[1] + origin[1];
    verts[i][2] = dnew[2] + origin[2];
  }
}
