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
   (https://github.com/akohlmey/lammps-gui, liblammpsplugin.c,
   Copyright (c) 2023 - 2026  Axel Kohlmeyer), reduced to the
   functions actually exported by the SPARTA C library interface. */

/*
   Variant of the C style library interface to SPARTA
   that uses a shared library and dynamically opens it,
   so this can be used as a prototype code to integrate
   a SPARTA plugin to some other software.
*/

#include "libspartaplugin.h"

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#if defined(_WIN32_WINNT)
#undef _WIN32_WINNT
#endif

// target Windows version is windows 7 and later
#define _WIN32_WINNT _WIN32_WINNT_WIN7
#define PSAPI_VERSION 2

#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <stdlib.h>

libspartaplugin_t *libspartaplugin_load(const char *lib)
{
  libspartaplugin_t *lmp;
  void *handle;

  if (lib == NULL) return NULL;

#ifdef _WIN32
  handle = (void *) LoadLibraryA(lib);
#else
  handle = dlopen(lib,RTLD_NOW|RTLD_GLOBAL);
#endif
  if (handle == NULL) return NULL;

  lmp = (libspartaplugin_t *) calloc(1, sizeof(libspartaplugin_t));
  lmp->abiversion = SPARTAPLUGIN_ABI_VERSION;
  lmp->handle = handle;

#ifdef _WIN32
#define ADDSYM(symbol) *(void **) (&lmp->symbol) = (void *) GetProcAddress((HINSTANCE) handle, "sparta_" #symbol)
#else
#define ADDSYM(symbol) *(void **) (&lmp->symbol) = dlsym(handle,"sparta_" #symbol)
#endif

#if defined(SPARTA_LIB_MPI)
  ADDSYM(open);
#else
  lmp->open = NULL;
#endif

  ADDSYM(open_no_mpi);
  ADDSYM(close);

  ADDSYM(file);
  ADDSYM(command);
  ADDSYM(commands_string);

  ADDSYM(get_thermo);
  ADDSYM(last_thermo);

  ADDSYM(extract_setting);
  ADDSYM(extract_global);

  ADDSYM(extract_compute);
  ADDSYM(extract_fix);
  ADDSYM(extract_variable);
  ADDSYM(extract_variable_datatype);
  ADDSYM(variable_info);

  ADDSYM(version);

  ADDSYM(config_has_mpi_support);
  ADDSYM(config_has_gzip_support);
  ADDSYM(config_has_png_support);
  ADDSYM(config_has_jpeg_support);
  ADDSYM(config_has_ffmpeg_support);
  ADDSYM(config_has_exceptions);

  ADDSYM(config_has_package);
  ADDSYM(config_accelerator);

  ADDSYM(style_count);
  ADDSYM(style_name);

  ADDSYM(has_id);
  ADDSYM(id_count);
  ADDSYM(id_name);

  ADDSYM(free);

  ADDSYM(is_running);
  ADDSYM(force_timeout);

  /* symbol not present: release the handle and storage before bailing out */
  if (!lmp->config_has_exceptions) {
    libspartaplugin_release(lmp);
    return NULL;
  }

  lmp->has_exceptions = lmp->config_has_exceptions();
  if (lmp->has_exceptions) {
    ADDSYM(has_error);
    ADDSYM(get_last_error_message);
  }

  return lmp;
}

int libspartaplugin_release(libspartaplugin_t *lmp)
{
  if (lmp == NULL) return 1;
  if (lmp->handle == NULL) return 2;

#ifdef _WIN32
  FreeLibrary((HINSTANCE) lmp->handle);
#else
  dlclose(lmp->handle);
#endif
  free((void *)lmp);
  return 0;
}
