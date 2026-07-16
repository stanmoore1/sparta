/* -*- c++ -*- ----------------------------------------------------------
   SPARTA - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://sparta.github.io/, Sandia National Laboratories
   SPARTA development team: developers@sparta.github.io

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#ifndef LIBSPARTAPLUGIN_H
#define LIBSPARTAPLUGIN_H
/*
   Variant of the C style library interface to SPARTA
   that uses a shared library and dynamically opens it,
   so this can be used as a prototype code to integrate
   a SPARTA plugin to some other software.
*/

/*
 * Follow the behavior of regular SPARTA compilation and assume
 * -DSPARTA_SMALLBIG when no define is set.
 */
#if !defined(SPARTA_BIGBIG) && !defined(SPARTA_SMALLBIG)
#define SPARTA_SMALLBIG
#endif

#if defined(SPARTA_LIB_MPI)
#include <mpi.h>
#endif

#if defined(SPARTA_BIGBIG) || defined(SPARTA_SMALLBIG)
#include <stdint.h>  /* for int64_t */
#endif

/* The following enums must be kept in sync with the equivalent enums
 * or constants in src/library.h, src/lmptype.h, python/sparta/constants.py,
 * fortran/sparta.f90, and tools/swig/sparta.i */

/* Data type constants for extracting data from atoms, computes and fixes */

enum _LMP_DATATYPE_CONST {
  SPARTA_NONE = -1,     /*!< no data type assigned (yet) */
  SPARTA_INT = 0,       /*!< 32-bit integer (array) */
  SPARTA_INT_2D = 1,    /*!< two-dimensional 32-bit integer array */
  SPARTA_DOUBLE = 2,    /*!< 64-bit double (array) */
  SPARTA_DOUBLE_2D = 3, /*!< two-dimensional 64-bit double array */
  SPARTA_INT64 = 4,     /*!< 64-bit integer (array) */
  SPARTA_INT64_2D = 5,  /*!< two-dimensional 64-bit integer array */
  SPARTA_STRING = 6     /*!< C-String */
};

/* Style constants for extracting data from computes and fixes. */

enum _SPA_STYLE_CONST {
  SPA_STYLE_GLOBAL = 0, /*!< return global data */
  SPA_STYLE_ATOM = 1,   /*!< return per-atom data */
  SPA_STYLE_LOCAL = 2   /*!< return local data */
};

/* Type and size constants for extracting data from computes and fixes. */

enum _SPA_TYPE_CONST {
  SPA_TYPE_SCALAR = 0, /*!< return scalar */
  SPA_TYPE_VECTOR = 1, /*!< return vector */
  SPA_TYPE_ARRAY = 2,  /*!< return array */
  SPA_SIZE_VECTOR = 3, /*!< return length of vector */
  SPA_SIZE_ROWS = 4,   /*!< return number of rows */
  SPA_SIZE_COLS = 5    /*!< return number of columns */
};

/* Error codes to select the suitable function in the Error class */

enum _SPA_ERROR_CONST {
  SPA_ERROR_WARNING = 0, /*!< call Error::warning() */
  SPA_ERROR_ONE = 1,     /*!< called from one MPI rank */
  SPA_ERROR_ALL = 2,     /*!< called from all MPI ranks */
  SPA_ERROR_WORLD = 4,   /*!< error on Comm::world */
  SPA_ERROR_UNIVERSE = 8 /*!< error on Comm::universe */
};

/** Variable style constants for extracting data from variables.
 *
 * Must be kept in sync with the equivalent constants in python/sparta/constants.py,
 * fortran/sparta.f90, and tools/swig/sparta.i */

enum _SPA_VAR_CONST {
  SPA_VAR_EQUAL = 0,  /*!< compatible with equal-style variables */
  SPA_VAR_ATOM = 1,   /*!< compatible with atom-style variables */
  SPA_VAR_VECTOR = 2, /*!< compatible with vector-style variables */
  SPA_VAR_STRING = 3  /*!< return value will be a string (catch-all) */
};

/** Neighbor list settings constants
 *
 * Must be kept in sync with the equivalent constants in ``python/sparta/constants.py``,
 * ``fortran/sparta.f90``, ``tools/swig/sparta.i``, and
 * ``examples/COUPLE/plugin/libspartaplugin.h`` */

enum _LMP_NEIGH_CONST {
  LMP_NEIGH_HALF = 0,  /*!< request (default) half neighbor list */
  LMP_NEIGH_FULL = 1,  /*!< request full neighbor list */
};

#ifdef __cplusplus
extern "C" {
#endif

#if defined(SPARTA_BIGBIG)
typedef void (*FixExternalFnPtr)(void *, int64_t, int, int64_t *, double **, double **);
#else
typedef void (*FixExternalFnPtr)(void *, int64_t, int, int *, double **, double **);
#endif

#define SPARTAPLUGIN_ABI_VERSION 2
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
  void *(*open_fortran)(int, char **, void **, int);
  void (*close)(void *);

  void (*mpi_init)();
  void (*mpi_finalize)();
  void (*kokkos_finalize)();
  void (*python_finalize)();
  void (*plugin_finalize)();

  void (*error)(void *, int, const char *);
  char *(*expand)(void *, const char *);

  void (*file)(void *, const char *);
  char *(*command)(void *, const char *);
  void (*commands_list)(void *, int, const char **);
  void (*commands_string)(void *, const char *);

  double (*get_natoms)(void *);
  double (*get_thermo)(void *, const char *);
  void *(*last_thermo)(void *, const char *, int);

  void (*extract_box)(void *, double *, double *,
                      double *, double *, double *, int *, int *);
  void (*reset_box)(void *, double *, double *, double, double, double);

  void (*memory_usage)(void *, double *);
  int (*get_mpi_comm)(void *);

  int (*extract_setting)(void *, const char *);
  int (*extract_global_datatype)(void *, const char *);
  void *(*extract_global)(void *, const char *);
  int (*extract_pair_dimension)(void *, const char *);
  void *(*extract_pair)(void *, const char *);
  int (*map_atom)(void *, const void *);

  int (*extract_atom_datatype)(void *, const char *);
  int (*extract_atom_size)(void *, const char *, int);
  void *(*extract_atom)(void *, const char *);

  void *(*extract_compute)(void *, const char *, int, int);
  void *(*extract_fix)(void *, const char *, int, int, int, int);
  void *(*extract_variable)(void *, const char *, const char *);
  int (*extract_variable_datatype)(void *, const char *);
  int (*set_variable)(void *, const char *, const char *);
  int (*set_string_variable)(void *, const char *, const char *);
  int (*set_internal_variable)(void *, const char *, double);
  int (*variable_info)(void *, int, char *, int);
  double (*eval)(void *, const char *);
  void (*clearstep_compute)(void *);
  void (*addstep_compute)(void *, void *);
  void (*addstep_compute_all)(void *, void *);

  void (*gather_atoms)(void *, const char *, int, int, void *);
  void (*gather_atoms_concat)(void *, const char *, int, int, void *);
  void (*gather_atoms_subset)(void *, const char *, int, int, int, int *, void *);
  void (*scatter_atoms)(void *, const char *, int, int, void *);
  void (*scatter_atoms_subset)(void *, const char *, int, int, int, int *, void *);

  void (*gather_bonds)(void *, void *);
  void (*gather_angles)(void *, void *);
  void (*gather_dihedrals)(void *, void *);
  void (*gather_impropers)(void *, void *);

  void (*gather)(void *, const char *, int, int, void *);
  void (*gather_concat)(void *, const char *, int, int, void *);
  void (*gather_subset)(void *, const char *, int, int, int, int *,void *);
  void (*scatter)(void *, const char *, int, int, void *);
  void (*scatter_subset)(void *, const char *, int, int, int, int *, void *);

/* sparta_create_atoms() takes tagint and imageint as args
 * the ifdef ensures they are compatible with rest of SPARTA
 * caller must match to how SPARTA library is built */

#if !defined(SPARTA_BIGBIG)
 int (*create_atoms)(void *, int, const int *, const int *, const double *, const double *,
                     const int *, int);
#else
  int (*create_atoms)(void *, int, const int64_t *, const int *, const double *, const double *,
                      const int64_t *, int);
#endif
  int (*create_molecule)(void *, const char *, const char *);

  int (*find_pair_neighlist)(void *, const char *, int, int, int);
  int (*find_fix_neighlist)(void *, const char *, int);
  int (*find_compute_neighlist)(void *, const char *, int);
  int (*request_single_neighlist)(void *, const char *, int, double);
  int (*neighlist_num_elements)(void *, int);
  void (*neighlist_element_neighbors)(void *, int, int, int *, int *, int **);

  int (*version)(void *);
  void (*get_os_info)(char *, int);

  int (*config_has_mpi_support)();
  int (*config_has_omp_support)();
  int (*config_has_gzip_support)();
  int (*config_has_png_support)();
  int (*config_has_jpeg_support)();
  int (*config_has_ffmpeg_support)();
  int (*config_has_curl_support)();
  int (*config_has_exceptions)();

  int (*config_has_package)(const char *);
  int (*config_package_count)();
  int (*config_package_name)(int, char *, int);

  int (*config_accelerator)(const char *, const char *, const char *);
  int (*has_gpu_device)();
  void (*get_gpu_device_info)(char *, int);

  int (*has_style)(void *, const char *, const char *);
  int (*style_count)(void *, const char *);
  int (*style_name)(void *, const char *, int, char *, int);

  int (*has_id)(void *, const char *, const char *);
  int (*id_count)(void *, const char *);
  int (*id_name)(void *, const char *, int, char *, int);

  int (*plugin_count)();
  int (*plugin_name)(int, char *, char *, int);

#if !defined(SPARTA_BIGBIG)
  int (*encode_image_flags)(int, int, int);
  void (*decode_image_flags)(int, int *);
#else
  int64_t (*encode_image_flags)(int, int, int);
  void (*decode_image_flags)(int64_t, int *);
#endif

  void (*set_fix_external_callback)(void *, const char *, FixExternalFnPtr, void *);
  double **(*fix_external_get_force)(void *, const char *);
  void (*fix_external_set_energy_global)(void *, const char *, double);
  void (*fix_external_set_energy_peratom)(void *, const char *, double *);
  void (*fix_external_set_virial_global)(void *, const char *, double *);
  void (*fix_external_set_virial_peratom)(void *, const char *, double **);
  void (*fix_external_set_vector_length)(void *, const char *, int);
  void (*fix_external_set_vector)(void *, const char *, int, double);

  void (*flush_buffers)(void *);

  void (*free)(void *);

  int (*is_running)(void *);
  void (*force_timeout)(void *);

  int (*has_error)(void *);
  int (*get_last_error_message)(void *, char *, int);
  int (*set_show_error)(void *, const int);

  int (*python_api_version)();
};

typedef struct _libspartaplugin libspartaplugin_t;

libspartaplugin_t *libspartaplugin_load(const char *);
int libspartaplugin_release(libspartaplugin_t *);

#ifdef __cplusplus
}
#endif

#endif
