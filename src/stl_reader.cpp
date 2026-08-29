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
   Ported from the LAMMPS STLReader class, see stl_reader.h
------------------------------------------------------------------------- */

#include "ctype.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "stl_reader.h"
#include "comm.h"
#include "utils.h"
#include "memory.h"
#include "error.h"

#include <stdint.h>

using namespace SPARTA_NS;

// byte layout of a binary STL file:
//   80-byte header + 4-byte (uint32) triangle count, then per triangle
//   12 floats (normal + 3 vertices) + a 2-byte attribute count

static const long STL_BIN_HEADER = 80 + sizeof(uint32_t);                        // 84
static const long STL_BIN_PER_TRI = 12*sizeof(float) + sizeof(uint16_t);         // 50

#define MAXLINE 1024

/* ----------------------------------------------------------------------
   local helpers that stand in for the LAMMPS utils and TextFileReader
   classes the original version of this file uses
------------------------------------------------------------------------- */

// strip leading and trailing whitespace from a string

static std::string trim(const std::string &text)
{
  std::string::size_type first = 0;
  std::string::size_type last = text.size();

  while (first < last && isspace((unsigned char) text[first])) first++;
  while (last > first && isspace((unsigned char) text[last-1])) last--;

  return text.substr(first,last-first);
}

// split a line on whitespace into a list of words

static std::vector<std::string> split_words(const char *line)
{
  std::vector<std::string> words;
  const char *ptr = line;

  while (*ptr) {
    while (*ptr && isspace((unsigned char) *ptr)) ptr++;
    const char *start = ptr;
    while (*ptr && !isspace((unsigned char) *ptr)) ptr++;
    if (ptr > start) words.push_back(std::string(start,ptr-start));
  }

  return words;
}

// convert a word to a double, throw if it is not a valid number

static double next_double(const std::string &word, const std::string &context)
{
  if (!utils::is_double(word))
    throw STLReaderException("Error parsing STL file: " + context +
                             ": invalid number " + word);
  return atof(word.c_str());
}

// return the next non-blank line of a file, or NULL at end of file
// buf must be at least MAXLINE chars

static char *next_line(FILE *fp, char *buf)
{
  while (fgets(buf,MAXLINE,fp)) {
    for (char *ptr = buf; *ptr; ptr++)
      if (!isspace((unsigned char) *ptr)) return buf;
  }
  return NULL;
}

/* ----------------------------------------------------------------------
   a class instance is only needed for read_file(), which owns the storage
   for the triangle vertices that it returns
------------------------------------------------------------------------- */

STLReader::STLReader(SPARTA *sparta) : Pointers(sparta)
{
  ntris = maxtris = 0;
  tris = NULL;
}

/* ----------------------------------------------------------------------
   free the storage for the triangle vertices
------------------------------------------------------------------------- */

STLReader::~STLReader()
{
  memory->destroy(tris);
}

/* ----------------------------------------------------------------------
   read an STL file on proc 0 and communicate the vertices to all procs
   the facet normals are discarded
   the storage for the vertices is owned by the class instance,
   so it is freed by the destructor and the caller must copy what it needs
   reading an empty or unreadable file aborts SPARTA with an error message
   return # of triangles
------------------------------------------------------------------------- */

int STLReader::read_file(const char *filename, double **&caller_tris)
{
  int me = comm->me;

  if (me == 0) {
    try {
      std::string title;
      std::vector<Triangle> parsed = parse(filename,&title);
      ntris = (int) parsed.size();
      maxtris = ntris;
      utils::logmesg(sparta,"Reading STL object " +
                     (title.empty() ? std::string("(unnamed)") : title) + " with " +
                     std::to_string(ntris) + " triangles from file " + filename + "\n");

      memory->create(tris,(ntris > 0) ? ntris : 1,9,"stl_reader:tris");
      for (int i = 0; i < ntris; i++) {
        int m = 0;
        for (int j = 0; j < 3; j++)
          for (int k = 0; k < 3; k++)
            tris[i][m++] = parsed[i].vert[j][k];
      }
    } catch (std::exception &e) {
      error->one(FLERR,e.what());
    }
  }

  MPI_Bcast(&ntris,1,MPI_INT,0,world);
  if (ntris == 0) {
    std::string mesg = "STL file " + std::string(filename) + " has no triangles";
    error->all(FLERR,mesg.c_str());
  }
  if (me) memory->create(tris,ntris,9,"stl_reader:tris");

  // allow for 9*ntris to exceed the max allowed size of a single MPI_Bcast()

  bigint ntotal = (bigint) ntris * 9;
  if (ntotal < MAXSMALLINT)
    MPI_Bcast(&tris[0][0],9*ntris,MPI_DOUBLE,0,world);
  else {
    double *source = &tris[0][0];
    bigint n = 0;
    while (n < ntotal) {
      int nsize = MIN(MAXSMALLINT,ntotal-n);
      MPI_Bcast(&source[n],nsize,MPI_DOUBLE,0,world);
      n += nsize;
    }
  }

  caller_tris = tris;
  return ntris;
}

/* ----------------------------------------------------------------------
   1 if the file is in STL format, else 0
   a binary STL file has a size of exactly 84 + 50*N bytes for N triangles
   an ASCII STL file starts with the solid keyword
   throws an STLReaderException if the file cannot be opened
------------------------------------------------------------------------- */

bool STLReader::is_stl_file(const std::string &filename)
{
  FILE *fp = fopen(filename.c_str(),"rb");
  if (!fp) throw STLReaderException("Cannot open mesh file " + filename);

  bool is_stl = false;

  if (fseek(fp,0,SEEK_END) == 0) {
    long filesize = ftell(fp);
    if (filesize >= STL_BIN_HEADER) {
      uint32_t ntri_claim = 0;
      if (fseek(fp,80,SEEK_SET) == 0 &&
          fread(&ntri_claim,sizeof(ntri_claim),1,fp) == 1) {
        bigint expected = (bigint) STL_BIN_HEADER +
          (bigint) ntri_claim * STL_BIN_PER_TRI;
        if (expected == (bigint) filesize) is_stl = true;
      }
    }
  }

  if (!is_stl) {
    rewind(fp);
    char buf[MAXLINE];
    char *line = next_line(fp,buf);
    if (line && utils::strmatch(line,"^ *solid")) is_stl = true;
  }

  fclose(fp);
  return is_stl;
}

/* ----------------------------------------------------------------------
   read and parse an STL file and return its triangles
   independent of any SPARTA instance and does not communicate
   whether the file is text or binary is detected from its contents:
     a binary STL file has a size of exactly 84 + 50*N bytes for N triangles,
     which is more robust than looking for the word "solid" at the start of
     the file, since the header of a binary file may legally begin with it
   any error while opening, reading, or parsing the file is reported by
   throwing an STLReaderException
   a file with zero triangles is not an error, it gives an empty vector
------------------------------------------------------------------------- */

std::vector<STLReader::Triangle>
STLReader::parse(const std::string &filename, std::string *title_out)
{
  FILE *fp = fopen(filename.c_str(),"rb");
  if (!fp) throw STLReaderException("Cannot open STL file " + filename);

  // determine the file size and the triangle count claimed by a binary header

  bool is_binary = false;
  if (fseek(fp,0,SEEK_END) == 0) {
    long filesize = ftell(fp);
    if (filesize >= STL_BIN_HEADER) {
      uint32_t ntri_claim = 0;
      if (fseek(fp,80,SEEK_SET) == 0 &&
          fread(&ntri_claim,sizeof(ntri_claim),1,fp) == 1) {
        bigint expected = (bigint) STL_BIN_HEADER +
          (bigint) ntri_claim * STL_BIN_PER_TRI;
        if (expected == (bigint) filesize) is_binary = true;
      }
    }
  }
  rewind(fp);

  std::string title;
  std::vector<Triangle> triangles;

  try {
    if (is_binary) {
      triangles = parse_binary(fp,title);
    } else {
      fclose(fp);
      fp = NULL;
      triangles = parse_text(filename,title);
    }
  } catch (...) {
    if (fp) fclose(fp);
    throw;
  }
  if (fp) fclose(fp);

  if (title_out) *title_out = title;
  return triangles;
}

/* ----------------------------------------------------------------------
   parse an ASCII STL file
------------------------------------------------------------------------- */

std::vector<STLReader::Triangle>
STLReader::parse_text(const std::string &filename, std::string &title)
{
  char buf[MAXLINE];

  FILE *fp = fopen(filename.c_str(),"r");
  if (!fp) throw STLReaderException("Cannot open STL file " + filename);

  char *line = next_line(fp,buf);
  if (!line || !utils::strmatch(line,"^ *solid")) {
    fclose(fp);
    throw STLReaderException("File " + filename + " is not a valid ASCII STL file");
  }

  // solid name may be empty; use a std::string so there is no risk of running
  // past the end of the buffer (as bare pointer arithmetic on "solid" would)

  std::string header(line);
  std::string::size_type pos = header.find("solid");
  title = trim(header.substr(pos+5));

  std::vector<Triangle> triangles;

  try {
    while ((line = next_line(fp,buf))) {
      std::vector<std::string> words = split_words(line);
      if (words.empty()) continue;
      if (utils::strmatch(words[0],"^endsolid")) break;
      if (!utils::strmatch(words[0],"^facet"))
        throw STLReaderException(std::string("Expected 'facet' or 'endsolid' in "
                                             "STL file, got: ") + trim(line));

      Triangle tri;
      tri.normal[0] = tri.normal[1] = tri.normal[2] = 0.0;

      // facet line is "facet normal nx ny nz"; the normal is optional and
      // tolerated to be absent or unparsable (it is recomputed when needed)

      if (words.size() >= 5 && utils::strmatch(words[1],"^normal") &&
          utils::is_double(words[2]) && utils::is_double(words[3]) &&
          utils::is_double(words[4]))
        for (int k = 0; k < 3; k++) tri.normal[k] = atof(words[2+k].c_str());

      line = next_line(fp,buf);
      if (!line || !utils::strmatch(line,"^ *outer *loop"))
        throw STLReaderException("Error reading 'outer loop' in STL file");

      for (int k = 0; k < 3; k++) {
        line = next_line(fp,buf);
        std::vector<std::string> values = line ? split_words(line) :
          std::vector<std::string>();
        if (values.size() < 4 || values[0] != "vertex")
          throw STLReaderException("Error reading vertex " + std::to_string(k+1) +
                                   " of facet in STL file");
        for (int m = 0; m < 3; m++)
          tri.vert[k][m] = next_double(values[1+m],"vertex " + std::to_string(k+1));
      }

      line = next_line(fp,buf);
      if (!line || !utils::strmatch(line,"^ *endloop"))
        throw STLReaderException("Error reading 'endloop' in STL file");
      line = next_line(fp,buf);
      if (!line || !utils::strmatch(line,"^ *endfacet"))
        throw STLReaderException("Error reading 'endfacet' in STL file");

      triangles.push_back(tri);
    }
  } catch (...) {
    fclose(fp);
    throw;
  }

  fclose(fp);
  return triangles;
}

/* ----------------------------------------------------------------------
   parse a binary STL file: 80-byte header, uint32 triangle count, then
   per triangle 12 little-endian floats (normal + 3 vertices) + 2-byte attr
------------------------------------------------------------------------- */

std::vector<STLReader::Triangle>
STLReader::parse_binary(FILE *fp, std::string &title)
{
  rewind(fp);

  char head[80];
  if (fread(head,1,80,fp) != 80)
    throw STLReaderException("Unexpected end of binary STL file while reading header");

  // the header is a fixed 80-byte field that need not be null terminated

  std::size_t len = 0;
  while (len < sizeof(head) && head[len] != '\0') ++len;
  title = trim(std::string(head,len));

  uint32_t ntri = 0;
  if (fread(&ntri,sizeof(ntri),1,fp) != 1)
    throw STLReaderException("Unexpected end of binary STL file while reading "
                             "triangle count");
  if (ntri > (uint32_t) MAXSMALLINT)
    throw STLReaderException("Number of triangles in STL file exceeds integer limit");

  std::vector<Triangle> triangles;
  triangles.reserve(ntri);

  float buf[12];
  uint16_t attr;
  for (uint32_t i = 0; i < ntri; i++) {
    if (fread(buf,sizeof(float),12,fp) != 12)
      throw STLReaderException("Unexpected end of binary STL file at triangle " +
                               std::to_string(i+1) + " of " + std::to_string(ntri));
    if (fread(&attr,sizeof(attr),1,fp) != 1)
      throw STLReaderException("Unexpected end of binary STL file reading "
                               "attributes of triangle " + std::to_string(i+1));

    Triangle tri;
    for (int k = 0; k < 3; k++) tri.normal[k] = buf[k];
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
        tri.vert[j][k] = buf[3 + 3*j + k];
    triangles.push_back(tri);
  }

  return triangles;
}
