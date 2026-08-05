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
   Contributing author: ported from the LAMMPS VTKWriter class
   (github.com/lammps/lammps pull request #5106).

   Writer for the subset of the VTK file formats used by SPARTA.  The
   layouts implemented here were determined from the output of VTK 9.2 so
   that files written without the VTK library are read back the same way.
   The relevant details, none of which are obvious from the format
   documentation:

   legacy format
   * the version line says 5.1.  that revision replaced the old inline cell
     lists with separate OFFSETS and CONNECTIVITY arrays.
   * data arrays that are not the active scalars end up in a single
     "FIELD FieldData <n>" section, one header line plus values per array.
   * binary payloads follow the newline of their keyword line as raw
     BIG endian values and are followed by another newline.
   * 64-bit integers are named "vtktypeint64", strings are named "string".
   * strings carry a length prefix.  lengths below 64 use the single byte
     0xc0|len; longer strings use undocumented multi byte forms, so we
     refuse to write those rather than guess.

   XML format
   * the file version is 0.1 and the byte count header type is UInt32.
   * inline binary data is base64 encoded.  without compression the header
     and the data form a single encoded stream, but with compression the
     header is encoded separately and the two encodings are concatenated.
   * the compressed header is [nblocks][block size][size of last partial
     block][compressed size per block].  we always use a single block.
   * cell offsets are END offsets and have one entry per cell, unlike the
     legacy format where they start at zero and have one extra entry.
   * string arrays use <Array>, not <DataArray>, and their ASCII form is a
     list of decimal character codes with a zero terminating each string.

   The VTK library also writes RangeMin/RangeMax attributes, InformationKey
   and METADATA blocks, and empty containers for the polydata cell types it
   is not using.  All of those are optional and are left out here.
------------------------------------------------------------------------- */

#include "vtk_writer.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <limits>
#include <utility>

#if defined(SPARTA_ZLIB)
#include <zlib.h>
#endif

using namespace SPARTA_NS;

static const int PER_LINE = 9;

// warn once single precision can no longer resolve this fraction of the box

static const double SINGLE_PRECISION_FRACTION = 1.0e-6;

// longest string we can write into a binary legacy file, see file header

static const size_t MAX_BINARY_STRING = 63;

// flush ASCII output in chunks of this size instead of one call per value

static const size_t WRITE_CHUNK = 1 << 20;

namespace {

const char b64chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int little_endian()
{
  const int one = 1;
  return *((const char *) &one) == 1;
}

std::string base64(const void *data, size_t len)
{
  const unsigned char *d = static_cast<const unsigned char *>(data);
  std::string out;
  out.reserve(((len + 2) / 3) * 4);

  size_t i = 0;
  for (; i + 3 <= len; i += 3) {
    const unsigned v = (d[i] << 16) | (d[i + 1] << 8) | d[i + 2];
    out += b64chars[(v >> 18) & 63];
    out += b64chars[(v >> 12) & 63];
    out += b64chars[(v >> 6) & 63];
    out += b64chars[v & 63];
  }
  if (i + 1 == len) {
    const unsigned v = d[i] << 16;
    out += b64chars[(v >> 18) & 63];
    out += b64chars[(v >> 12) & 63];
    out += "==";
  } else if (i + 2 == len) {
    const unsigned v = (d[i] << 16) | (d[i + 1] << 8);
    out += b64chars[(v >> 18) & 63];
    out += b64chars[(v >> 12) & 63];
    out += b64chars[(v >> 6) & 63];
    out += '=';
  }
  return out;
}

// encode a block of raw bytes the way the VTK library does for inline binary
// XML data.  see the notes at the top of this file for the layout.

std::string encode_bytes(const std::string &raw)
{
#if defined(SPARTA_ZLIB)
  uLongf bound = compressBound((uLong) raw.size());
  std::vector<unsigned char> buf(bound ? bound : 1);
  if (compress(&buf[0],&bound,(const Bytef *) raw.data(),(uLong) raw.size()) == Z_OK) {
    const uint32_t header[4] = {1U,(uint32_t) raw.size(),0U,(uint32_t) bound};
    return base64(header,sizeof(header)) + base64(&buf[0],bound);
  }
  // fall through to writing the data uncompressed if zlib fails
#endif
  std::string block;
  block.reserve(raw.size() + sizeof(uint32_t));
  const uint32_t nbytes = (uint32_t) raw.size();
  block.append((const char *) &nbytes,sizeof(nbytes));
  block.append(raw);
  return base64(block.data(),block.size());
}

// append the raw bytes of a value in the host byte order

template <typename T> void append_raw(std::string &out, T value)
{
  out.append((const char *) &value,sizeof(T));
}

// write raw values in big endian byte order, as legacy binary files require.
// the bytes are reordered through char pointers on purpose and are never
// loaded back into a variable of the original type: reversing floating point
// numbers by way of a floating point register can alter them on some
// platforms.  keep this loop byte-wise if it is ever optimized.  the fixed
// chunk buffer bounds the transient memory for large arrays.

template <typename T> void fwrite_be(FILE *fp, const std::vector<T> &values)
{
  if (values.empty()) return;
  if (!little_endian()) {
    fwrite(&values[0],sizeof(T),values.size(),fp);
    return;
  }
  char buf[8192];
  size_t used = 0;
  for (size_t n = 0; n < values.size(); n++) {
    const char *src = (const char *) &values[n];
    for (size_t b = 0; b < sizeof(T); b++) buf[used + b] = src[sizeof(T) - 1 - b];
    used += sizeof(T);
    if (used + sizeof(T) > sizeof(buf)) {
      fwrite(buf,1,used,fp);
      used = 0;
    }
  }
  if (used) fwrite(buf,1,used,fp);
}

/* ----------------------------------------------------------------------
   shortest decimal representation that reads back as the same number.
   the VTK library writes a fixed 11 significant digits for doubles, which
   does not round-trip and so defeats the point of double precision output.
   %g already strips trailing zeros, so starting at the digit count that is
   always sufficient for exact values (FLT_DIG, DBL_DIG) and growing only
   when the value does not read back gives the shortest exact form.
------------------------------------------------------------------------- */

std::string fmt_float(float value)
{
  char buf[32];
  for (int prec = 6; prec < 9; prec++) {
    snprintf(buf,sizeof(buf),"%.*g",prec,(double) value);
    if (strtof(buf,NULL) == value) return buf;
  }
  snprintf(buf,sizeof(buf),"%.9g",(double) value);
  return buf;
}

std::string fmt_double(double value)
{
  char buf[32];
  for (int prec = 15; prec < 17; prec++) {
    snprintf(buf,sizeof(buf),"%.*g",prec,value);
    if (strtod(buf,NULL) == value) return buf;
  }
  snprintf(buf,sizeof(buf),"%.17g",value);
  return buf;
}

// text form of one value of each type that can appear in a data array.
// these must be declared before the templates below that call them.

void append_value(std::string &out, int value)
{
  char buf[16];
  snprintf(buf,sizeof(buf),"%d",value);
  out += buf;
}

void append_value(std::string &out, uint8_t value)
{
  char buf[8];
  snprintf(buf,sizeof(buf),"%d",(int) value);
  out += buf;
}

void append_value(std::string &out, int64_t value)
{
  char buf[24];
  snprintf(buf,sizeof(buf),"%lld",(long long) value);
  out += buf;
}

void append_value(std::string &out, float value)
{
  out += fmt_float(value);
}

void append_value(std::string &out, double value)
{
  out += fmt_double(value);
}

// write the values of a legacy data section, either as text or as binary.
// the text is flushed in chunks so that large arrays need neither one stdio
// call per value nor the whole payload in memory.

template <typename T> void write_legacy_values(FILE *fp, int binary, const std::vector<T> &values)
{
  if (binary) {
    fwrite_be(fp,values);
    fputc('\n',fp);
    return;
  }
  std::string out;
  for (size_t i = 0; i < values.size(); i++) {
    append_value(out,values[i]);
    out += ((i + 1) % PER_LINE == 0) ? '\n' : ' ';
    if (out.size() > WRITE_CHUNK) {
      fwrite(out.data(),1,out.size(),fp);
      out.clear();
    }
  }
  if (values.size() % PER_LINE != 0) out += '\n';
  fwrite(out.data(),1,out.size(),fp);
}

// build the content of an XML data array, either as text or base64 encoded

template <typename T> std::string xml_values(int binary, const std::vector<T> &values, int indent)
{
  if (binary) {
    std::string raw;
    raw.reserve(values.size() * sizeof(T));
    for (size_t i = 0; i < values.size(); i++) append_raw(raw,values[i]);
    return std::string(indent,' ') + encode_bytes(raw) + "\n";
  }

  const std::string pad(indent,' ');
  std::string out;
  out.reserve(values.size() * 8 + (values.size() / PER_LINE + 1) * pad.size());
  for (size_t i = 0; i < values.size(); i++) {
    if (i % PER_LINE == 0) out += pad;
    append_value(out,values[i]);
    out += ((i + 1) % PER_LINE == 0) ? '\n' : ' ';
  }
  if (values.size() % PER_LINE != 0) out += '\n';
  return out;
}

}    // namespace

/* ----------------------------------------------------------------------
   remember the largest coordinate magnitude that goes through single
   precision, so that callers can warn when its resolution is no longer
   sufficient.  only the set_*() methods call this: data arrays need no
   tracking since their values only require relative resolution
------------------------------------------------------------------------- */

void VTKWriter::track_single(const std::vector<double> &values)
{
  if (prec != SINGLE) return;
  for (size_t i = 0; i < values.size(); i++)
    if (fabs(values[i]) > maxsingle) maxsingle = fabs(values[i]);
}

/* ----------------------------------------------------------------------
   write a list of floating point values in the precision selected for
   this writer
------------------------------------------------------------------------- */

void VTKWriter::write_legacy_reals(FILE *fp, const std::vector<double> &values)
{
  if (prec == SINGLE) {
    const std::vector<float> single(values.begin(),values.end());
    write_legacy_values(fp,binary,single);
  } else {
    write_legacy_values(fp,binary,values);
  }
}

std::string VTKWriter::xml_reals(const std::vector<double> &values, int indent) const
{
  if (prec == SINGLE) {
    const std::vector<float> single(values.begin(),values.end());
    return xml_values(binary,single,indent);
  }
  return xml_values(binary,values,indent);
}

const char *VTKWriter::legacy_real_type() const
{
  return (prec == SINGLE) ? "float" : "double";
}

const char *VTKWriter::xml_real_type() const
{
  return xml_real_type(prec);
}

const char *VTKWriter::xml_real_type(Precision precision)
{
  return (precision == SINGLE) ? "Float32" : "Float64";
}

const char *VTKWriter::xml_byte_order()
{
  return little_endian() ? "LittleEndian" : "BigEndian";
}

double VTKWriter::single_precision_resolution(double maxcoord, double boxlength)
{
  const double resolution = maxcoord * std::numeric_limits<float>::epsilon();
  if (resolution <= SINGLE_PRECISION_FRACTION * boxlength) return 0.0;
  return resolution;
}

/* ---------------------------------------------------------------------- */

VTKWriter::VTKWriter(Flavor _flavor, int _binary, Precision _prec) :
  flavor(_flavor), binary(_binary), prec(_prec), maxsingle(0.0), dataset(NONE),
  title("Generated by SPARTA"), npoints(0), ncells(0), celltype(VERTEX), ptspercell(1) {}

/* ---------------------------------------------------------------------- */

void VTKWriter::set_title(const std::string &_title)
{
  // the legacy format reserves a single line of at most 256 characters

  title = _title.substr(0,256);
  for (size_t i = 0; i < title.size(); i++)
    if (title[i] == '\n' || title[i] == '\r') title[i] = ' ';
}

/* ----------------------------------------------------------------------
   select a dataset of cells that each own npts consecutive points of the
   coordinate list, so the connectivity is simply 0,1,2,...
------------------------------------------------------------------------- */

void VTKWriter::set_cells(std::vector<double> &xyz, Dataset type, int ctype, int npts)
{
  if (dataset != NONE) throw VTKWriterException("VTK writer already has a dataset");
  if (xyz.size() % 3) throw VTKWriterException("VTK point list is not a multiple of 3");
  if (npts < 1) throw VTKWriterException("VTK cells need at least one point each");

  if (xyz.size() / 3 > (size_t) INT_MAX)
    throw VTKWriterException("Too many points for one VTK file");

  points.swap(xyz);
  track_single(points);
  npoints = (int) (points.size() / 3);
  if (npoints % npts)
    throw VTKWriterException("VTK point count is not a multiple of the points per cell");
  ncells = npoints / npts;
  celltype = ctype;
  ptspercell = npts;
  dataset = type;
}

void VTKWriter::set_polydata(std::vector<double> xyz, int ctype, int npts)
{
  if (ctype != VERTEX && ctype != LINE && ctype != TRIANGLE)
    throw VTKWriterException("VTK polydata supports only vertex, line and polygon cells");
  set_cells(xyz,POLYDATA,ctype,npts);
}

void VTKWriter::set_unstructured_grid(std::vector<double> xyz, int ctype, int npts)
{
  set_cells(xyz,UNSTRUCTURED,ctype,npts);
}

/* ----------------------------------------------------------------------
   polydata keeps its cells in a container named after the cell type
------------------------------------------------------------------------- */

const char *VTKWriter::legacy_poly_keyword() const
{
  if (celltype == LINE) return "LINES";
  if (celltype == TRIANGLE) return "POLYGONS";
  return "VERTICES";
}

const char *VTKWriter::xml_poly_tag() const
{
  if (celltype == LINE) return "Lines";
  if (celltype == TRIANGLE) return "Polys";
  return "Verts";
}

/* ---------------------------------------------------------------------- */

void VTKWriter::add_array(std::vector<DataArray> &arrays, int nitems, const char *kind,
                          DataArray &array)
{
  if (dataset == NONE) throw VTKWriterException("VTK writer has no dataset selected");
  if (array.ncomp < 1) throw VTKWriterException("VTK data array needs at least one component");

  size_t nvalues = 0;
  switch (array.type) {
  case TYPE_INT: nvalues = array.ivalues.size(); break;
  case TYPE_INT64: nvalues = array.lvalues.size(); break;
  case TYPE_DOUBLE: nvalues = array.dvalues.size(); break;
  case TYPE_STRING: nvalues = array.svalues.size(); break;
  }

  if (nvalues != (size_t) nitems * array.ncomp) {
    char str[256];
    snprintf(str,sizeof(str),
             "VTK data array %s has %d values but %d %s times %d components are required",
             array.name.c_str(),(int) nvalues,nitems,kind,array.ncomp);
    throw VTKWriterException(str);
  }

  arrays.push_back(std::move(array));
}

/* ---------------------------------------------------------------------- */

void VTKWriter::add_point_array(const std::string &name, int ncomp, std::vector<int> data)
{
  DataArray array;
  array.name = name; array.type = TYPE_INT; array.ncomp = ncomp; array.ivalues.swap(data);
  add_array(point_arrays,npoints,"points",array);
}

void VTKWriter::add_point_array(const std::string &name, int ncomp, std::vector<int64_t> data)
{
  DataArray array;
  array.name = name; array.type = TYPE_INT64; array.ncomp = ncomp; array.lvalues.swap(data);
  add_array(point_arrays,npoints,"points",array);
}

void VTKWriter::add_point_array(const std::string &name, int ncomp, std::vector<double> data)
{
  DataArray array;
  array.name = name; array.type = TYPE_DOUBLE; array.ncomp = ncomp; array.dvalues.swap(data);
  add_array(point_arrays,npoints,"points",array);
}

void VTKWriter::add_point_array(const std::string &name, std::vector<std::string> data)
{
  DataArray array;
  array.name = name; array.type = TYPE_STRING; array.ncomp = 1; array.svalues.swap(data);
  add_array(point_arrays,npoints,"points",array);
}

void VTKWriter::add_cell_array(const std::string &name, int ncomp, std::vector<int> data)
{
  DataArray array;
  array.name = name; array.type = TYPE_INT; array.ncomp = ncomp; array.ivalues.swap(data);
  add_array(cell_arrays,ncells,"cells",array);
}

void VTKWriter::add_cell_array(const std::string &name, int ncomp, std::vector<int64_t> data)
{
  DataArray array;
  array.name = name; array.type = TYPE_INT64; array.ncomp = ncomp; array.lvalues.swap(data);
  add_array(cell_arrays,ncells,"cells",array);
}

void VTKWriter::add_cell_array(const std::string &name, int ncomp, std::vector<double> data)
{
  DataArray array;
  array.name = name; array.type = TYPE_DOUBLE; array.ncomp = ncomp; array.dvalues.swap(data);
  add_array(cell_arrays,ncells,"cells",array);
}

void VTKWriter::add_cell_array(const std::string &name, std::vector<std::string> data)
{
  DataArray array;
  array.name = name; array.type = TYPE_STRING; array.ncomp = 1; array.svalues.swap(data);
  add_array(cell_arrays,ncells,"cells",array);
}

/* ---------------------------------------------------------------------- */

void VTKWriter::set_active_scalars(const std::string &name)
{
  for (size_t i = 0; i < point_arrays.size(); i++)
    if (point_arrays[i].name == name) { scalars = name; return; }
  for (size_t i = 0; i < cell_arrays.size(); i++)
    if (cell_arrays[i].name == name) { scalars = name; return; }

  char str[256];
  snprintf(str,sizeof(str),"VTK data array %s was not added to this writer",name.c_str());
  throw VTKWriterException(str);
}

/* ---------------------------------------------------------------------- */

void VTKWriter::write(const std::string &filename)
{
  FILE *fp = fopen(filename.c_str(),"wb");
  if (!fp) {
    char str[256];
    snprintf(str,sizeof(str),"Cannot open VTK file %s",filename.c_str());
    throw VTKWriterException(str);
  }

  // close the file whether or not writing it succeeds

  try {
    write(fp);
  } catch (VTKWriterException &) {
    fclose(fp);
    throw;
  }
  fclose(fp);
}

void VTKWriter::write(FILE *fp)
{
  if (dataset == NONE) throw VTKWriterException("VTK writer has no dataset selected");
  if (flavor == LEGACY) write_legacy(fp);
  else write_xml(fp);
}

/* ----------------------------------------------------------------------
   legacy format
------------------------------------------------------------------------- */

void VTKWriter::write_legacy_cells(FILE *fp, const char *keyword)
{
  // the legacy offsets start at zero, so there is one more of them than cells

  std::vector<int64_t> offsets(ncells + 1);
  for (int i = 0; i <= ncells; i++) offsets[i] = (int64_t) i * ptspercell;

  std::vector<int64_t> connectivity((size_t) ncells * ptspercell);
  for (size_t i = 0; i < connectivity.size(); i++) connectivity[i] = (int64_t) i;

  fprintf(fp,"%s %d %d\n",keyword,ncells+1,(int) connectivity.size());
  fprintf(fp,"OFFSETS vtktypeint64\n");
  write_legacy_values(fp,binary,offsets);
  fprintf(fp,"CONNECTIVITY vtktypeint64\n");
  write_legacy_values(fp,binary,connectivity);
}

/* ---------------------------------------------------------------------- */

void VTKWriter::write_legacy_array_data(FILE *fp, const DataArray &array)
{
  switch (array.type) {
  case TYPE_INT:
    write_legacy_values(fp,binary,array.ivalues);
    break;
  case TYPE_INT64:
    write_legacy_values(fp,binary,array.lvalues);
    break;
  case TYPE_DOUBLE:
    write_legacy_reals(fp,array.dvalues);
    break;
  case TYPE_STRING:
    if (binary) {
      std::string raw;
      for (size_t i = 0; i < array.svalues.size(); i++) {
        const std::string &s = array.svalues[i];
        if (s.size() > MAX_BINARY_STRING) {
          char str[256];
          snprintf(str,sizeof(str),
                   "Cannot write string \"%s\" to a binary legacy VTK file, "
                   "it is longer than %d characters",s.c_str(),(int) MAX_BINARY_STRING);
          throw VTKWriterException(str);
        }
        raw += (char) (0xc0 | s.size());
        raw += s;
      }
      fwrite(raw.data(),1,raw.size(),fp);
      fputc('\n',fp);
    } else {
      for (size_t i = 0; i < array.svalues.size(); i++)
        fprintf(fp,"%s\n",array.svalues[i].c_str());
    }
    break;
  }
}

/* ---------------------------------------------------------------------- */

const char *VTKWriter::legacy_array_type(const DataArray &array) const
{
  if (array.type == TYPE_INT) return "int";
  if (array.type == TYPE_INT64) return "vtktypeint64";
  if (array.type == TYPE_STRING) return "string";
  return legacy_real_type();
}

/* ---------------------------------------------------------------------- */

void VTKWriter::write_legacy_arrays(FILE *fp, const std::vector<DataArray> &arrays,
                                    const char *keyword, int nitems)
{
  if (arrays.empty()) return;
  fprintf(fp,"%s %d\n",keyword,nitems);

  // the active scalars get their own section, everything else is field data

  int nfield = 0;
  for (size_t i = 0; i < arrays.size(); i++) {
    const DataArray &array = arrays[i];
    if (array.name == scalars) {
      if (array.type == TYPE_STRING)
        throw VTKWriterException("VTK string arrays cannot be used as active scalars");
      fprintf(fp,"SCALARS %s %s",array.name.c_str(),legacy_array_type(array));
      if (array.ncomp > 1) fprintf(fp," %d",array.ncomp);
      fprintf(fp,"\nLOOKUP_TABLE default\n");
      write_legacy_array_data(fp,array);
    } else nfield++;
  }
  if (nfield == 0) return;

  fprintf(fp,"FIELD FieldData %d\n",nfield);
  for (size_t i = 0; i < arrays.size(); i++) {
    const DataArray &array = arrays[i];
    if (array.name == scalars) continue;
    fprintf(fp,"%s %d %d %s\n",array.name.c_str(),array.ncomp,nitems,legacy_array_type(array));
    write_legacy_array_data(fp,array);
  }
}

/* ---------------------------------------------------------------------- */

void VTKWriter::write_legacy(FILE *fp)
{
  fprintf(fp,"# vtk DataFile Version 5.1\n%s\n%s\n",title.c_str(),binary ? "BINARY" : "ASCII");

  if (dataset == POLYDATA) {
    fprintf(fp,"DATASET POLYDATA\nPOINTS %d %s\n",npoints,legacy_real_type());
    write_legacy_reals(fp,points);
    write_legacy_cells(fp,legacy_poly_keyword());

  } else {
    fprintf(fp,"DATASET UNSTRUCTURED_GRID\nPOINTS %d %s\n",npoints,legacy_real_type());
    write_legacy_reals(fp,points);
    write_legacy_cells(fp,"CELLS");
    std::vector<int> types(ncells,celltype);
    fprintf(fp,"CELL_TYPES %d\n",ncells);
    write_legacy_values(fp,binary,types);
  }

  write_legacy_arrays(fp,point_arrays,"POINT_DATA",npoints);
  write_legacy_arrays(fp,cell_arrays,"CELL_DATA",ncells);
}

/* ----------------------------------------------------------------------
   XML format
------------------------------------------------------------------------- */

void VTKWriter::write_xml_data_array(FILE *fp, const char *type, const std::string &name,
                                     int ncomp, const std::string &payload, int indent)
{
  const std::string pad(indent,' ');
  fprintf(fp,"%s<DataArray type=\"%s\" Name=\"%s\"",pad.c_str(),type,name.c_str());
  if (ncomp > 1) fprintf(fp," NumberOfComponents=\"%d\"",ncomp);
  fprintf(fp," format=\"%s\">\n",binary ? "binary" : "ascii");
  fputs(payload.c_str(),fp);
  fprintf(fp,"%s</DataArray>\n",pad.c_str());
}

/* ---------------------------------------------------------------------- */

void VTKWriter::write_xml_array(FILE *fp, const DataArray &array, int indent)
{
  switch (array.type) {
  case TYPE_INT:
    write_xml_data_array(fp,"Int32",array.name,array.ncomp,
                         xml_values(binary,array.ivalues,indent+2),indent);
    break;

  case TYPE_INT64:
    write_xml_data_array(fp,"Int64",array.name,array.ncomp,
                         xml_values(binary,array.lvalues,indent+2),indent);
    break;

  case TYPE_DOUBLE:
    write_xml_data_array(fp,xml_real_type(),array.name,array.ncomp,
                         xml_reals(array.dvalues,indent+2),indent);
    break;

  case TYPE_STRING: {
    // string arrays use <Array> and store the characters of all strings,
    // each terminated by a zero byte

    const std::string pad(indent,' ');
    std::string payload;
    if (binary) {
      std::string raw;
      for (size_t i = 0; i < array.svalues.size(); i++) {
        raw += array.svalues[i];
        raw += '\0';
      }
      payload = std::string(indent+2,' ') + encode_bytes(raw) + "\n";
    } else {
      std::vector<int> codes;
      for (size_t i = 0; i < array.svalues.size(); i++) {
        const std::string &s = array.svalues[i];
        for (size_t c = 0; c < s.size(); c++) codes.push_back((unsigned char) s[c]);
        codes.push_back(0);
      }
      payload = xml_values(0,codes,indent+2);
    }
    fprintf(fp,"%s<Array type=\"String\" Name=\"%s\" format=\"%s\">\n",
            pad.c_str(),array.name.c_str(),binary ? "binary" : "ascii");
    fputs(payload.c_str(),fp);
    fprintf(fp,"%s</Array>\n",pad.c_str());
    break;
  }
  }
}

/* ---------------------------------------------------------------------- */

void VTKWriter::write_xml_arrays(FILE *fp, const std::vector<DataArray> &arrays, const char *tag,
                                 int indent)
{
  const std::string pad(indent,' ');
  int has_scalars = 0;
  for (size_t i = 0; i < arrays.size(); i++)
    if (arrays[i].name == scalars) has_scalars = 1;

  if (has_scalars) fprintf(fp,"%s<%s Scalars=\"%s\">\n",pad.c_str(),tag,scalars.c_str());
  else fprintf(fp,"%s<%s>\n",pad.c_str(),tag);

  for (size_t i = 0; i < arrays.size(); i++) write_xml_array(fp,arrays[i],indent+2);
  fprintf(fp,"%s</%s>\n",pad.c_str(),tag);
}

/* ---------------------------------------------------------------------- */

void VTKWriter::write_xml_cells(FILE *fp, const char *tag, int indent)
{
  const std::string pad(indent,' ');

  // XML offsets are end offsets, one per cell

  std::vector<int64_t> offsets(ncells);
  for (int i = 0; i < ncells; i++) offsets[i] = (int64_t) (i+1) * ptspercell;

  std::vector<int64_t> connectivity((size_t) ncells * ptspercell);
  for (size_t i = 0; i < connectivity.size(); i++) connectivity[i] = (int64_t) i;

  fprintf(fp,"%s<%s>\n",pad.c_str(),tag);
  write_xml_data_array(fp,"Int64","connectivity",1,xml_values(binary,connectivity,indent+4),
                       indent+2);
  write_xml_data_array(fp,"Int64","offsets",1,xml_values(binary,offsets,indent+4),indent+2);

  // only an unstructured grid carries a per cell type, polydata does not

  if (strcmp(tag,"Cells") == 0) {
    std::vector<uint8_t> types(ncells,(uint8_t) celltype);
    write_xml_data_array(fp,"UInt8","types",1,xml_values(binary,types,indent+4),indent+2);
  }
  fprintf(fp,"%s</%s>\n",pad.c_str(),tag);
}

/* ---------------------------------------------------------------------- */

void VTKWriter::write_xml(FILE *fp)
{
  const char *gridtype = (dataset == POLYDATA) ? "PolyData" : "UnstructuredGrid";

  fprintf(fp,"<?xml version=\"1.0\"?>\n");
  fprintf(fp,"<VTKFile type=\"%s\" version=\"0.1\" byte_order=\"%s\" header_type=\"UInt32\"",
          gridtype,xml_byte_order());
#if defined(SPARTA_ZLIB)
  if (binary) fprintf(fp," compressor=\"vtkZLibDataCompressor\"");
#endif
  fprintf(fp,">\n");

  fprintf(fp,"  <%s>\n",gridtype);
  if (dataset == POLYDATA) {
    // the cell count belongs to the one container that holds the cells

    const int nverts = (celltype == VERTEX) ? ncells : 0;
    const int nlines = (celltype == LINE) ? ncells : 0;
    const int npolys = (celltype == TRIANGLE) ? ncells : 0;
    fprintf(fp,"    <Piece NumberOfPoints=\"%d\" NumberOfVerts=\"%d\" NumberOfLines=\"%d\" "
            "NumberOfStrips=\"0\" NumberOfPolys=\"%d\">\n",npoints,nverts,nlines,npolys);
  } else {
    fprintf(fp,"    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n",npoints,ncells);
  }

  write_xml_arrays(fp,point_arrays,"PointData",6);
  write_xml_arrays(fp,cell_arrays,"CellData",6);

  fprintf(fp,"      <Points>\n");
  write_xml_data_array(fp,xml_real_type(),"Points",3,xml_reals(points,10),8);
  fprintf(fp,"      </Points>\n");

  if (dataset == POLYDATA) write_xml_cells(fp,xml_poly_tag(),6);
  else write_xml_cells(fp,"Cells",6);

  fprintf(fp,"    </Piece>\n  </%s>\n</VTKFile>\n",gridtype);
}
