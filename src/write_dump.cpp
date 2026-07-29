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

#include "spatype.h"
#include "string.h"
#include "write_dump.h"
#include "style_dump.h"
#include "dump.h"
#include "dump_image.h"
#include "modify.h"
#include "update.h"
#include "error.h"

using namespace SPARTA_NS;

/* ----------------------------------------------------------------------
   write a single snapshot of the current state to a dump file
   syntax: write_dump style select-ID file args ... [modify dump_modify-args ...]
   creates a Dump instance on the fly, writes one snapshot, destroys it
------------------------------------------------------------------------- */

void WriteDump::command(int narg, char **arg)
{
  if (narg < 3) error->all(FLERR,"Illegal write_dump command");

  // a movie is a sequence of frames piped to ffmpeg during a run,
  //   so a single snapshot of it is not meaningful

  if (strcmp(arg[0],"movie") == 0)
    error->all(FLERR,"Dump style movie cannot be used with write_dump");

  // modindex = index of optional "modify" keyword in arg
  // = narg if not present, so dump args are arg[3] to arg[modindex-1]

  int modindex;
  for (modindex = 3; modindex < narg; modindex++)
    if (strcmp(arg[modindex],"modify") == 0) break;

  // build the arg list the Dump constructor expects:
  //   ID style select-ID N file args ...
  // ID is a reserved name, N = 1 is never used since the Dump
  //   is not added to the Output list of dumps

  int dumpargc = modindex + 2;
  char **dumpargv = new char*[dumpargc];

  dumpargv[0] = (char *) "WRITE_DUMP";
  dumpargv[1] = arg[0];
  dumpargv[2] = arg[1];
  dumpargv[3] = (char *) "1";
  dumpargv[4] = arg[2];
  for (int i = 3; i < modindex; i++) dumpargv[i+2] = arg[i];

  // create the Dump instance

  Dump *dump = NULL;

  if (0) return;         // dummy line to enable else-if macro expansion

#define DUMP_CLASS
#define DumpStyle(key,Class) \
  else if (strcmp(arg[0],#key) == 0) dump = new Class(sparta,dumpargc,dumpargv);
#include "style_dump.h"
#undef DumpStyle
#undef DUMP_CLASS

  else {
    delete [] dumpargv;
    error->all(FLERR,"Unrecognized dump style in write_dump command");
  }

  // an image dump normally requires "*" in the filename, one image per frame
  // write_dump writes a single image, so allow a filename without "*"

  if (strcmp(arg[0],"image") == 0)
    ((DumpImage *) dump)->multifile_override = 1;

  // apply any dump_modify args that followed the "modify" keyword

  if (modindex < narg)
    dump->modify_params(narg-modindex-1,&arg[modindex+1]);

  // write one snapshot
  // wrap with clear/add if the dump invokes computes, as Output::write() does
  // addstep for the next timestep so computes invoked here stay available

  if (dump->clearstep) modify->clearstep_compute();
  dump->init();
  dump->write();
  if (dump->clearstep) modify->addstep_compute(update->ntimestep+1);

  delete dump;
  delete [] dumpargv;
}
