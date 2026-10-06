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

#include "mpi.h"
#include "sparta.h"
#include "input.h"
#include "library.h"
#include "spaexception.h"

#include "stdio.h"
#include "stdlib.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace SPARTA_NS;

// for convenience

static void finalize()
{
  sparta_kokkos_finalize();
}

/* ----------------------------------------------------------------------
   check if process has superuser (root) or administrator privileges
------------------------------------------------------------------------- */

static bool is_superuser()
{
#if defined(_WIN32)
  // a process started with "Run as administrator" has an elevated access token
  bool elevated = false;
  HANDLE token = NULL;
  if (OpenProcessToken(GetCurrentProcess(),TOKEN_QUERY,&token)) {
    TOKEN_ELEVATION elevation;
    DWORD size = sizeof(elevation);
    if (GetTokenInformation(token,TokenElevation,&elevation,
                            sizeof(elevation),&size))
      elevated = (elevation.TokenIsElevated != 0);
    CloseHandle(token);
  }
  return elevated;
#else
  return (geteuid() == 0);
#endif
}

/* ----------------------------------------------------------------------
   main program to drive SPARTA
------------------------------------------------------------------------- */

int main(int argc, char **argv)
{
  MPI_Init(&argc,&argv);

  // warn if SPARTA is run with superuser or administrator privileges.
  // this is done as early as possible and only by the first MPI rank

  int me = 0;
  MPI_Comm_rank(MPI_COMM_WORLD,&me);
  if ((me == 0) && is_superuser())
    fprintf(stderr,
            "\nWARNING: SPARTA is run with superuser or administrator "
            "privileges.\n"
            "WARNING: This is STRONGLY discouraged, because typos or "
            "mistakes in an\n"
            "WARNING: input file or errors in SPARTA itself can damage the "
            "entire\n"
            "WARNING: system to the point of requiring a re-installation.\n"
            "WARNING: For more information see "
            "https://github.com/sparta/sparta/blob/master/SECURITY.md\n\n");

  try {
    SPARTA *sparta = new SPARTA(argc,argv,MPI_COMM_WORLD);
    sparta->input->file();
    delete sparta;
  } catch (SpartaAbortException &e) {
    finalize();
    MPI_Abort(e.get_universe(),1);
  } catch (SpartaException &) {
    // error message was already printed by the Error class
    finalize();
    MPI_Finalize();
    exit(1);
  }

  finalize();

  MPI_Finalize();
}
