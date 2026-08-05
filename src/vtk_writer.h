/* -*- c++ -*- ----------------------------------------------------------
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
------------------------------------------------------------------------- */

#ifndef SPARTA_VTK_WRITER_H
#define SPARTA_VTK_WRITER_H

#include <stdint.h>
#include <stdio.h>
#include <exception>
#include <string>
#include <vector>

namespace SPARTA_NS {

// exception thrown for inconsistent data or files that cannot be written

class VTKWriterException : public std::exception {
  std::string message;

 public:
  explicit VTKWriterException(const std::string &msg) : message(msg) {}
  virtual ~VTKWriterException() throw() {}
  virtual const char *what() const throw() { return message.c_str(); }
};

/* ----------------------------------------------------------------------
   Write data files in the formats used by the VTK visualization toolkit,
   without requiring the VTK library itself.  Both the "legacy" format and
   the XML format are supported, each with ASCII or binary encoding.
   Reading VTK files is not supported.

   Usage is always the same: create a writer for the desired format, select
   exactly one dataset, attach any number of data arrays, and write.  All
   inconsistencies (mismatched array lengths, missing dataset, a file that
   cannot be opened) are reported by throwing a VTKWriterException, so
   callers are expected to catch it and turn it into a SPARTA error:

     VTKWriter writer(VTKWriter::XML,0);
     writer.set_polydata(coords);
     writer.add_point_array("id",1,ids);
     try {
       writer.write("dump.vtp");
     } catch (VTKWriterException &e) {
       error->one(FLERR,e.what());
     }

   Only the two datasets SPARTA writes are implemented: point/cell sets as
   POLYDATA and as UNSTRUCTURED_GRID.  The LAMMPS class this was ported from
   also has RECTILINEAR_GRID and STRUCTURED_POINTS datasets; they are left
   out here so that every code path has a caller and is exercised by the
   examples/vtk tests.
------------------------------------------------------------------------- */

class VTKWriter {
 public:
  enum Flavor { LEGACY, XML };
  enum Precision { SINGLE, DOUBLE };

  // the VTK cell type ids SPARTA uses, same values as VTK's vtkCellType.h

  enum CellType { VERTEX = 1, LINE = 3, TRIANGLE = 5, PIXEL = 8, VOXEL = 11 };

  /* --------------------------------------------------------------------
     Resolution of single precision storage at the given coordinate
     magnitude.  Callers pass the value reported by
     max_single_precision_value() together with the size of the simulation
     box, and warn with the returned resolution when it is not zero.

     Single precision has a relative resolution of about 1.2e-7, so what
     matters is the coordinate magnitude compared to the size of the box:
     a small box far from the origin loses absolute resolution.  We warn
     once single precision can no longer resolve one part in a million of
     the box.  (LAMMPS scales a fixed coordinate limit by force->angstrom
     instead; that does not carry over, because SPARTA's si and cgs units
     make every coordinate small in absolute terms.)

     maxcoord   = largest coordinate magnitude written in single precision
     boxlength  = extent of the simulation box
     returns    = absolute resolution at that magnitude when it warrants a
                  warning, 0.0 otherwise
  -------------------------------------------------------------------- */

  static double single_precision_resolution(double maxcoord, double boxlength);

  /* --------------------------------------------------------------------
     XML type name for floating point data at the given precision, either
     "Float32" or "Float64".  Exposed so that the parallel summary files
     the dump styles write by hand always declare exactly the types that
     the piece files contain.
  -------------------------------------------------------------------- */

  static const char *xml_real_type(Precision precision);

  // byte order attribute written into XML files on this machine,
  // either "LittleEndian" or "BigEndian"

  static const char *xml_byte_order();

  /* --------------------------------------------------------------------
     Create a writer for one file.

     flavor     = LEGACY for the simple legacy format, XML for the XML format
     binary     = 1 to store the data as binary numbers instead of as text
     precision  = precision used for all floating point data, that is point
                  coordinates and double data arrays alike
  -------------------------------------------------------------------- */

  VTKWriter(Flavor flavor, int binary, Precision precision = DOUBLE);

  // title written into the header of legacy files, ignored for XML,
  // truncated to the 256 characters the format allows

  void set_title(const std::string &title);

  /* --------------------------------------------------------------------
     Dataset selection.  Exactly one of these must be called before
     write().  Coordinates are passed as 3 doubles per point, and each
     cell owns npts consecutive points of its own, which is how the dump
     styles build their geometry.  The coordinate vector is taken by
     value, so callers can swap() it in to avoid a copy.

     celltype = one of the CellType values
     npts     = points per cell, 1 for VERTEX, 2 for LINE, 3 for TRIANGLE,
                4 for PIXEL, 8 for VOXEL
  -------------------------------------------------------------------- */

  void set_polydata(std::vector<double> xyz, int celltype = VERTEX, int npts = 1);
  void set_unstructured_grid(std::vector<double> xyz, int celltype = VERTEX, int npts = 1);

  // data arrays, holding ncomp consecutive values per point or per cell.
  // like the coordinates these are taken by value and moved into the writer,
  // so pass them with std::move() to avoid copying data that the caller no
  // longer needs.

  void add_point_array(const std::string &name, int ncomp, std::vector<int> data);
  void add_point_array(const std::string &name, int ncomp, std::vector<int64_t> data);
  void add_point_array(const std::string &name, int ncomp, std::vector<double> data);
  void add_point_array(const std::string &name, std::vector<std::string> data);
  void add_cell_array(const std::string &name, int ncomp, std::vector<int> data);
  void add_cell_array(const std::string &name, int ncomp, std::vector<int64_t> data);
  void add_cell_array(const std::string &name, int ncomp, std::vector<double> data);
  void add_cell_array(const std::string &name, std::vector<std::string> data);

  /* --------------------------------------------------------------------
     Mark a previously added array as the default one for coloring.  It is
     written as SCALARS in legacy files and referenced by the Scalars
     attribute in XML files.
  -------------------------------------------------------------------- */

  void set_active_scalars(const std::string &name);

  /* --------------------------------------------------------------------
     Largest absolute value among the coordinates that are written in
     single precision, or 0.0 if there are none.  Callers use this with
     single_precision_resolution() to warn when single precision is no
     longer sufficient.  Data arrays are not tracked: their values only
     need the relative resolution that single precision always provides.
  -------------------------------------------------------------------- */

  double max_single_precision_value() const { return maxsingle; }

  int number_of_points() const { return npoints; }
  int number_of_cells() const { return ncells; }

  void write(const std::string &filename);
  void write(FILE *fp);

 private:
  enum Dataset { NONE, POLYDATA, UNSTRUCTURED };
  enum DataType { TYPE_INT, TYPE_INT64, TYPE_DOUBLE, TYPE_STRING };

  struct DataArray {
    std::string name;
    DataType type;
    int ncomp;
    std::vector<int> ivalues;
    std::vector<int64_t> lvalues;
    std::vector<double> dvalues;
    std::vector<std::string> svalues;
  };

  Flavor flavor;
  int binary;
  Precision prec;
  double maxsingle;
  Dataset dataset;
  std::string title;
  std::string scalars;

  int npoints, ncells;
  std::vector<double> points;
  int celltype, ptspercell;

  std::vector<DataArray> point_arrays, cell_arrays;

  void set_cells(std::vector<double> &xyz, Dataset type, int ctype, int npts);
  void add_array(std::vector<DataArray> &arrays, int nitems, const char *kind, DataArray &array);
  const char *legacy_array_type(const DataArray &array) const;

  // coordinates and double data arrays honor the selected precision.
  // only coordinates are tracked for the single precision warning,
  // because only they need absolute resolution.

  void track_single(const std::vector<double> &values);
  void write_legacy_reals(FILE *fp, const std::vector<double> &values);
  std::string xml_reals(const std::vector<double> &values, int indent) const;
  const char *legacy_real_type() const;
  const char *xml_real_type() const;

  // name of the container that holds the cells of a POLYDATA dataset,
  // which depends on the cell type: vertices, lines or polygons

  const char *legacy_poly_keyword() const;
  const char *xml_poly_tag() const;

  // legacy format

  void write_legacy(FILE *fp);
  void write_legacy_arrays(FILE *fp, const std::vector<DataArray> &arrays, const char *keyword,
                           int nitems);
  void write_legacy_array_data(FILE *fp, const DataArray &array);
  void write_legacy_cells(FILE *fp, const char *keyword);

  // XML format

  void write_xml(FILE *fp);
  void write_xml_arrays(FILE *fp, const std::vector<DataArray> &arrays, const char *tag, int indent);
  void write_xml_array(FILE *fp, const DataArray &array, int indent);
  void write_xml_data_array(FILE *fp, const char *type, const std::string &name, int ncomp,
                            const std::string &payload, int indent);
  void write_xml_cells(FILE *fp, const char *tag, int indent);
};

}    // namespace SPARTA_NS

#endif
