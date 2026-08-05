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

/* ----------------------------------------------------------------------
   Contributing author: ported from the LAMMPS VTK package (dump_vtk),
   originally from LIGGGHTS (www.liggghts.com).  The VTK files are written
   by SPARTA's own VTKWriter, see src/vtk_writer.cpp.
------------------------------------------------------------------------- */

#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "dump_particle_vtk.h"
#include "update.h"
#include "domain.h"
#include "memory.h"
#include "error.h"

#include <utility>

using namespace SPARTA_NS;

enum{INT,DOUBLE,BIGINT,UINT,BIGUINT,STRING};    // same as Dump
enum{VTK,VTP,VTU,PVTP,PVTU};                    // VTK file formats

/* ---------------------------------------------------------------------- */

DumpParticleVTK::DumpParticleVTK(SPARTA *sparta, int narg, char **arg) :
  DumpParticle(sparta, narg, arg)
{
  // VTK output uses raw double buffers, not SPARTA's ASCII string buffering

  buffer_allow = 0;
  buffer_flag = 0;

  n_calls_ = 0;
  filecurrent = NULL;
  parallelfilecurrent = NULL;
  gx = gy = gz = -1;
  prec = VTKWriter::DOUBLE;
  warned_precision = 0;

  // determine which VTK file format from the filename extension
  // default = legacy .vtk; .vtp = PolyData, .vtu = UnstructuredGrid
  // multiproc (% in filename) selects the parallel XML variants

  vtk_file_format = VTK;
  char *suffix = filename + strlen(filename) - strlen(".vtp");
  if (suffix > filename && strcmp(suffix,".vtp") == 0) {
    vtk_file_format = multiproc ? PVTP : VTP;
  } else if (suffix > filename && strcmp(suffix,".vtu") == 0) {
    vtk_file_format = multiproc ? PVTU : VTU;
  }

  // legacy .vtk format has no per-proc support (no '%' in filename)
  // without '%', the base class already gathers all procs to proc 0

  if (vtk_file_format == VTK && multiproc)
    error->all(FLERR,"Dump particle/vtk legacy .vtk format does not support "
               "'%' in filename; use .vtu or .vtp");

  // VTK writes one file per timestep, so require '*' in the filename

  if (!multifile)
    error->all(FLERR,"Dump particle/vtk requires '*' in filename for "
               "per-timestep output");

  // locate x,y,z columns and build the list of output data arrays

  setup_vtk_fields();

  if (filewriter) reset_vtk_data_containers();
}

/* ---------------------------------------------------------------------- */

DumpParticleVTK::~DumpParticleVTK()
{
  delete [] filecurrent;
  delete [] parallelfilecurrent;
}

/* ----------------------------------------------------------------------
   split the columns string into field names, locate x,y,z point coords,
   and assemble the list of point-data arrays (grouping vx,vy,vz and
   xs,ys,zs into 3-component vectors)
------------------------------------------------------------------------- */

void DumpParticleVTK::setup_vtk_fields()
{
  // tokenize columns (space separated, trailing space) into per-field names

  std::vector<std::string> names;
  char *copy = new char[strlen(columns)+1];
  strcpy(copy,columns);
  char *tok = strtok(copy," ");
  while (tok) { names.push_back(tok); tok = strtok(NULL," "); }
  delete [] copy;

  if ((int) names.size() != size_one)
    error->all(FLERR,"Dump particle/vtk internal field count mismatch");

  for (int i = 0; i < size_one; i++) {
    if (names[i] == "x") gx = i;
    else if (names[i] == "y") gy = i;
    else if (names[i] == "z") gz = i;
  }
  if (gx < 0 || gy < 0 || gz < 0)
    error->all(FLERR,"Dump particle/vtk requires x, y, and z attributes");

  // build data arrays from all non-coordinate columns
  // group consecutive vx,vy,vz and xs,ys,zs (all DOUBLE) into 3-vectors

  fields.clear();
  for (int i = 0; i < size_one; i++) {
    if (i == gx || i == gy || i == gz) continue;

    if (vtype[i] == STRING)
      error->all(FLERR,"Dump particle/vtk does not support string attributes");

    VTKField f;
    f.col = i;
    f.ncomp = 1;
    f.type = vtype[i];
    f.name = names[i];

    if (i+2 < size_one && vtype[i] == DOUBLE &&
        vtype[i+1] == DOUBLE && vtype[i+2] == DOUBLE) {
      if (names[i] == "vx" && names[i+1] == "vy" && names[i+2] == "vz") {
        f.ncomp = 3; f.name = "v"; fields.push_back(f); i += 2; continue;
      }
      if (names[i] == "xs" && names[i+1] == "ys" && names[i+2] == "zs") {
        f.ncomp = 3; f.name = "coords_scaled"; fields.push_back(f); i += 2;
        continue;
      }
    }
    fields.push_back(f);
  }
}

/* ---------------------------------------------------------------------- */

void DumpParticleVTK::init_style()
{
  // reuse DumpParticle to resolve compute/fix/variable/custom pointers
  // multifile is enforced in the constructor, so no dump file is opened here

  DumpParticle::init_style();
}

/* ----------------------------------------------------------------------
   the VTK writers manage their own files, so suppress the base class's
   FILE* handling: openfile() is a no-op (fp stays NULL and Dump::write()
   skips its fflush/fclose), and write_header() only resets the per-snapshot
   accumulation counter (it is called once per snapshot on the filewriter,
   just before Dump::gather_and_write() drives write_data()).
------------------------------------------------------------------------- */

void DumpParticleVTK::openfile() {}

void DumpParticleVTK::write_header(bigint) { n_calls_ = 0; }

/* ---------------------------------------------------------------------- */

void DumpParticleVTK::write_data(int n, double *mybuf)
{
  write_file(n,mybuf);
}

/* ----------------------------------------------------------------------
   append the packed rows in mybuf to the accumulating data containers
------------------------------------------------------------------------- */

void DumpParticleVTK::buf2arrays(int n, double *mybuf)
{
  for (int i = 0; i < n; i++) {
    int base = i*size_one;

    points.push_back(mybuf[base+gx]);
    points.push_back(mybuf[base+gy]);
    points.push_back(mybuf[base+gz]);

    for (int f = 0; f < (int) fields.size(); f++) {
      int c = base + fields[f].col;
      if (fields[f].ncomp == 3) {
        ddata[f].push_back(mybuf[c]);
        ddata[f].push_back(mybuf[c+1]);
        ddata[f].push_back(mybuf[c+2]);
      } else if (fields[f].type == DOUBLE)
        ddata[f].push_back(mybuf[c]);
      else
        ldata[f].push_back((int64_t) ubuf(mybuf[c]).i);
    }
  }
}

/* ---------------------------------------------------------------------- */

void DumpParticleVTK::reset_vtk_data_containers()
{
  points.clear();
  ddata.assign(fields.size(),std::vector<double>());
  ldata.assign(fields.size(),std::vector<int64_t>());
}

/* ----------------------------------------------------------------------
   write one piece file: particles as VTK_VERTEX cells, one point each,
   with the data arrays attached as point data
------------------------------------------------------------------------- */

void DumpParticleVTK::write_file(int n, double *mybuf)
{
  ++n_calls_;
  buf2arrays(n,mybuf);
  if (n_calls_ < nclusterprocs) return;

  setFileCurrent();

  const int xml = (vtk_file_format != VTK);
  VTKWriter writer(xml ? VTKWriter::XML : VTKWriter::LEGACY,binary,prec);

  try {
    // legacy .vtk and .vtp are polydata, .vtu is an unstructured grid

    if (vtk_file_format == VTU || vtk_file_format == PVTU)
      writer.set_unstructured_grid(std::move(points),VTKWriter::VERTEX,1);
    else
      writer.set_polydata(std::move(points),VTKWriter::VERTEX,1);

    for (int f = 0; f < (int) fields.size(); f++) {
      if (fields[f].ncomp == 3 || fields[f].type == DOUBLE)
        writer.add_point_array(fields[f].name,fields[f].ncomp,std::move(ddata[f]));
      else
        writer.add_point_array(fields[f].name,1,std::move(ldata[f]));
    }

    writer.write(filecurrent);

    // single precision only resolves so much of a box far from the origin

    double lbox = MAX(domain->xprd,domain->yprd);
    if (domain->dimension == 3) lbox = MAX(lbox,domain->zprd);
    double res = VTKWriter::single_precision_resolution
      (writer.max_single_precision_value(),lbox);
    if (res > 0.0 && !warned_precision) {
      warned_precision = 1;
      char str[128];
      sprintf(str,"Dump particle/vtk in single precision resolves coordinates "
              "only to %g",res);
      error->warning(FLERR,str);
    }
  } catch (VTKWriterException &e) {
    error->one(FLERR,e.what());
  }

  if (me == 0 && multiproc) write_pvtk(vtk_file_format);

  reset_vtk_data_containers();
}

/* ----------------------------------------------------------------------
   helpers to substitute '%' (proc/cluster id) and '*' (timestep) in a
   filename template, returning a newly allocated string
------------------------------------------------------------------------- */

static char *subst_percent(const char *name, int id)
{
  const char *ptr = strchr(name,'%');
  if (!ptr) { char *s = new char[strlen(name)+1]; strcpy(s,name); return s; }
  char *s = new char[strlen(name)+16];
  int pre = (int)(ptr-name);
  strncpy(s,name,pre); s[pre] = '\0';
  sprintf(s+pre,"%d%s",id,ptr+1);
  return s;
}

static char *subst_star(const char *name, bigint ntimestep, int padflag)
{
  const char *ptr = strchr(name,'*');
  if (!ptr) { char *s = new char[strlen(name)+1]; strcpy(s,name); return s; }
  char *s = new char[strlen(name)+16];
  int pre = (int)(ptr-name);
  strncpy(s,name,pre); s[pre] = '\0';
  if (padflag == 0)
    sprintf(s+pre,BIGINT_FORMAT "%s",ntimestep,ptr+1);
  else {
    char bif[8],pad[16];
    strcpy(bif,BIGINT_FORMAT);
    sprintf(pad,"%%0%d%s%%s",padflag,&bif[1]);
    sprintf(s+pre,pad,ntimestep,ptr+1);
  }
  return s;
}

/* ----------------------------------------------------------------------
   build the piece filename for this proc/timestep, and (rank 0) the name
   of the parallel .pvtp/.pvtu summary file
------------------------------------------------------------------------- */

void DumpParticleVTK::setFileCurrent()
{
  delete [] filecurrent;
  filecurrent = NULL;

  int id = me;
  if (multiproc > 1)
    id = (me + nclusterprocs == nprocs) ? multiproc-1 : me/nclusterprocs;

  char *tmp = subst_percent(filename,id);
  filecurrent = subst_star(tmp,update->ntimestep,padflag);
  delete [] tmp;

  if (multiproc && me == 0) {
    delete [] parallelfilecurrent;
    parallelfilecurrent = NULL;

    // remove '%' then change the extension to the parallel form
    char *ptr = strchr(filename,'%');
    char *base = new char[strlen(filename)+16];
    int pre = (int)(ptr-filename);
    strncpy(base,filename,pre); base[pre] = '\0';
    sprintf(base+pre,"%s",ptr+1);
    char *ext = strrchr(base,'.');
    ext++;
    ext[0] = 'p'; ext[1] = 'v'; ext[2] = 't';
    ext[3] = (vtk_file_format == PVTP) ? 'p' : 'u';
    ext[4] = '\0';

    parallelfilecurrent = subst_star(base,update->ntimestep,padflag);
    delete [] base;
  }
}

/* ----------------------------------------------------------------------
   name of the per-proc piece file for piece id, relative to the summary
------------------------------------------------------------------------- */

std::string DumpParticleVTK::pvtk_piece_filename(int id)
{
  char *tmp = subst_percent(filename,id);
  char *full = subst_star(tmp,update->ntimestep,padflag);
  delete [] tmp;
  std::string fname = full;
  delete [] full;
  size_t pos = fname.find_last_of("/\\");
  if (pos != std::string::npos) fname = fname.substr(pos+1);
  return fname;
}

/* ----------------------------------------------------------------------
   rank 0 writes the parallel summary (.pvtp/.pvtu) file by hand
------------------------------------------------------------------------- */

void DumpParticleVTK::write_pvtk(int fileformat)
{
  FILE *fp = fopen(parallelfilecurrent,"w");
  if (!fp) error->one(FLERR,"Cannot open dump particle/vtk parallel file");

  const char *gridtype =
    (fileformat == PVTP) ? "PPolyData" : "PUnstructuredGrid";

  // the summary must declare exactly the types the piece files contain

  const char *realtype = VTKWriter::xml_real_type(prec);

  fprintf(fp,"<?xml version=\"1.0\"?>\n");
  fprintf(fp,"<VTKFile type=\"%s\" version=\"1.0\" byte_order=\"%s\">\n",
          gridtype,VTKWriter::xml_byte_order());
  fprintf(fp,"  <%s GhostLevel=\"0\">\n",gridtype);

  fprintf(fp,"    <PPointData>\n");
  for (int f = 0; f < (int) fields.size(); f++) {
    const char *type = (fields[f].type == DOUBLE || fields[f].ncomp == 3) ?
      realtype : "Int64";
    if (fields[f].ncomp == 3)
      fprintf(fp,"      <PDataArray type=\"%s\" Name=\"%s\" "
              "NumberOfComponents=\"3\"/>\n",type,fields[f].name.c_str());
    else
      fprintf(fp,"      <PDataArray type=\"%s\" Name=\"%s\"/>\n",
              type,fields[f].name.c_str());
  }
  fprintf(fp,"    </PPointData>\n");

  fprintf(fp,"    <PPoints>\n");
  fprintf(fp,"      <PDataArray type=\"%s\" Name=\"Points\" "
          "NumberOfComponents=\"3\"/>\n",realtype);
  fprintf(fp,"    </PPoints>\n");

  int npieces = (multiproc > 1) ? multiproc : nprocs;
  for (int i = 0; i < npieces; i++)
    fprintf(fp,"    <Piece Source=\"%s\"/>\n",pvtk_piece_filename(i).c_str());

  fprintf(fp,"  </%s>\n",gridtype);
  fprintf(fp,"</VTKFile>\n");
  fclose(fp);
}

/* ----------------------------------------------------------------------
   dump_modify options: "binary yes/no" toggles VTK binary output and
   "double yes/no" selects Float64 vs Float32 for all floating point data,
   everything else is handled by DumpParticle (region, thresh)
------------------------------------------------------------------------- */

int DumpParticleVTK::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"binary") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal dump_modify command");
    if (strcmp(arg[1],"yes") == 0) binary = 1;
    else if (strcmp(arg[1],"no") == 0) binary = 0;
    else error->all(FLERR,"Illegal dump_modify command");
    return 2;
  }
  if (strcmp(arg[0],"double") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal dump_modify command");
    if (strcmp(arg[1],"yes") == 0) prec = VTKWriter::DOUBLE;
    else if (strcmp(arg[1],"no") == 0) prec = VTKWriter::SINGLE;
    else error->all(FLERR,"Illegal dump_modify command");
    return 2;
  }
  return DumpParticle::modify_param(narg,arg);
}
