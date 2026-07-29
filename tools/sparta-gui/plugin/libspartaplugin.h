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

/* Adapted for SPARTA-GUI from the LAMMPS-GUI plugin loader
   (https://github.com/akohlmey/lammps-gui, liblammpsplugin.h,
   Copyright (c) 2023 - 2026  Axel Kohlmeyer), reduced to the
   functions actually exported by the SPARTA C library interface
   in src/library.h. */

#ifndef LIBSPARTAPLUGIN_H
#define LIBSPARTAPLUGIN_H
/*
   Variant of the C style library interface to SPARTA
   that uses a shared library and dynamically opens it,
   so this can be used as a prototype code to integrate
   a SPARTA plugin to some other software.
*/

#if defined(SPARTA_LIB_MPI)
#include <mpi.h>
#endif

#include <stdint.h>  /* for int64_t */

/* The following enums must be kept in sync with the equivalent enums
 * or constants in src/library.h */

/* Data type constants for extracted or returned data */

enum _SPARTA_DATATYPE_CONST {
  SPARTA_NONE = -1,     /*!< no data type assigned (yet) */
  SPARTA_INT = 0,       /*!< 32-bit integer (array) */
  SPARTA_INT_2D = 1,    /*!< two-dimensional 32-bit integer array */
  SPARTA_DOUBLE = 2,    /*!< 64-bit double (array) */
  SPARTA_DOUBLE_2D = 3, /*!< two-dimensional 64-bit double array */
  SPARTA_INT64 = 4,     /*!< 64-bit integer (array) */
  SPARTA_INT64_2D = 5,  /*!< two-dimensional 64-bit integer array */
  SPARTA_STRING = 6     /*!< C-String */
};

/* The same constants are declared in SPARTA's src/library.h.  This header has
 * to stand alone -- the loader dlopen()s a library it has no headers for --
 * so it carries its own copy, skipped when library.h got here first. */

#ifndef SPARTA_LIBRARY_H_HAS_SPA_CONSTANTS

/* Style constants for extracting data from computes and fixes.
 * These differ from LAMMPS: SPARTA data is global, per-particle,
 * per-grid, per-surf, or per-tally. */

enum _SPA_STYLE_CONST {
  SPA_STYLE_GLOBAL = 0,   /*!< return global data */
  SPA_STYLE_PARTICLE = 1, /*!< return per-particle data */
  SPA_STYLE_GRID = 2,     /*!< return per-grid data */
  SPA_STYLE_SURF = 3,     /*!< return per-surf data */
  SPA_STYLE_TALLY = 4     /*!< return per-tally data */
};

/* Type constants for extracting data from computes and fixes.
 * For global data: scalar, vector, or array.
 * For per-particle/grid/surf/tally data the type selects the vector
 * (SPA_TYPE_VECTOR) or a 1-based array column (> 1). */

enum _SPA_TYPE_CONST {
  SPA_TYPE_SCALAR = 0, /*!< return scalar */
  SPA_TYPE_VECTOR = 1, /*!< return vector */
  SPA_TYPE_ARRAY = 2   /*!< return array */
};

/* Variable style constants for extracting data from variables.
 * Must be kept in sync with the constants in src/library.h */

enum _SPA_VAR_CONST {
  SPA_VAR_EQUAL = 0,    /*!< compatible with equal-style variables */
  SPA_VAR_PARTICLE = 1, /*!< compatible with particle-style variables */
  SPA_VAR_GRID = 2,     /*!< compatible with grid-style variables */
  SPA_VAR_SURF = 3,     /*!< compatible with surf-style variables */
  SPA_VAR_STRING = 4,   /*!< return value will be a string (catch-all) */
  SPA_VAR_INTERNAL = 5  /*!< internal variables */
};

#endif /* SPARTA_LIBRARY_H_HAS_SPA_CONSTANTS */

#ifdef __cplusplus
extern "C" {
#endif

#define SPARTAPLUGIN_ABI_VERSION 1
struct _libspartaplugin {
  int abiversion;
  int has_exceptions;
  void *handle;

#if defined(SPARTA_LIB_MPI)
  void *(*open)(int, char **, MPI_Comm, void **);
#else
  void *open;
#endif
  void *(*open_no_mpi)(int, char **, void **);
  void (*close)(void *);

  void (*file)(void *, const char *);
  char *(*command)(void *, const char *);
  void (*commands_string)(void *, const char *);

  double (*get_thermo)(void *, const char *);
  void *(*last_thermo)(void *, const char *, int);

  int (*extract_setting)(void *, const char *);
  void *(*extract_global)(void *, const char *);

  void *(*extract_compute)(void *, const char *, int, int);
  void *(*extract_fix)(void *, const char *, int, int);
  void *(*extract_variable)(void *, const char *);
  int (*extract_variable_datatype)(void *, const char *);
  int (*variable_info)(void *, int, char *, int);

  int (*version)(void *);

  int (*config_has_mpi_support)();
  int (*config_has_gzip_support)();
  int (*config_has_png_support)();
  int (*config_has_jpeg_support)();
  int (*config_has_ffmpeg_support)();
  int (*config_has_exceptions)();

  int (*config_has_package)(const char *);
  int (*config_accelerator)(const char *, const char *, const char *);

  int (*style_count)(void *, const char *);
  int (*style_name)(void *, const char *, int, char *, int);

  int (*has_id)(void *, const char *, const char *);
  int (*id_count)(void *, const char *);
  int (*id_name)(void *, const char *, int, char *, int);

  void (*free)(void *);

  int (*is_running)(void *);
  void (*force_timeout)(void *);

  int (*has_error)(void *);
  int (*get_last_error_message)(void *, char *, int);
};

typedef struct _libspartaplugin libspartaplugin_t;

libspartaplugin_t *libspartaplugin_load(const char *);
int libspartaplugin_release(libspartaplugin_t *);

#ifdef __cplusplus
}
#endif

#endif
