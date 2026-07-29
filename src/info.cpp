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
#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include "info.h"
#include "version.h"
#include "update.h"
#include "domain.h"
#include "region.h"
#include "comm.h"
#include "particle.h"
#include "mixture.h"
#include "grid.h"
#include "surf.h"
#include "surf_collide.h"
#include "surf_react.h"
#include "collide.h"
#include "react.h"
#include "modify.h"
#include "compute.h"
#include "fix.h"
#include "output.h"
#include "stats.h"
#include "dump.h"
#include "input.h"
#include "variable.h"
#include "timer.h"
#include "memory.h"
#include "error.h"

#if defined(__linux__) || defined(__APPLE__) || defined(__unix__)
#include "sys/resource.h"
#define SPARTA_HAVE_RUSAGE 1
#endif

using namespace SPARTA_NS;

enum{PERIODIC,OUTFLOW,REFLECT,SURFACE,AXISYM};   // same as Domain

// bit flags for which sections to print

#define INFO_CONFIG        0x00001
#define INFO_SYSTEM        0x00002
#define INFO_COMM          0x00004
#define INFO_COMPUTES      0x00008
#define INFO_FIXES         0x00010
#define INFO_DUMPS         0x00020
#define INFO_VARIABLES     0x00040
#define INFO_REGIONS       0x00080
#define INFO_GROUPS        0x00100
#define INFO_SPECIES       0x00200
#define INFO_MIXTURES      0x00400
#define INFO_SURF_COLLIDE  0x00800
#define INFO_SURF_REACT    0x01000
#define INFO_MEMORY        0x02000
#define INFO_TIME          0x04000
#define INFO_STYLES        0x08000

// "all" = everything except the (long) list of compiled-in styles

#define INFO_ALL (INFO_CONFIG | INFO_SYSTEM | INFO_COMM | INFO_COMPUTES | \
                  INFO_FIXES | INFO_DUMPS | INFO_VARIABLES | INFO_REGIONS | \
                  INFO_GROUPS | INFO_SPECIES | INFO_MIXTURES | \
                  INFO_SURF_COLLIDE | INFO_SURF_REACT | INFO_MEMORY | \
                  INFO_TIME)

/* ----------------------------------------------------------------------
   print a summary of the current state of the simulation
   syntax: info keyword ... [out screen/log/append file/overwrite file]
------------------------------------------------------------------------- */

void Info::command(int narg, char **arg)
{
  // out defaults to the screen on proc 0
  // only proc 0 produces output, but all procs participate
  //   since some sections do collective MPI calls

  out = screen;
  FILE *fpclose = NULL;

  // no keywords = "all"

  int flags = 0;
  if (narg == 0) flags = INFO_ALL;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"all") == 0) {
      flags |= INFO_ALL;
      iarg++;
    } else if (strcmp(arg[iarg],"config") == 0) {
      flags |= INFO_CONFIG;
      iarg++;
    } else if (strcmp(arg[iarg],"system") == 0) {
      flags |= INFO_SYSTEM;
      iarg++;
    } else if (strcmp(arg[iarg],"communication") == 0) {
      flags |= INFO_COMM;
      iarg++;
    } else if (strcmp(arg[iarg],"computes") == 0) {
      flags |= INFO_COMPUTES;
      iarg++;
    } else if (strcmp(arg[iarg],"fixes") == 0) {
      flags |= INFO_FIXES;
      iarg++;
    } else if (strcmp(arg[iarg],"dumps") == 0) {
      flags |= INFO_DUMPS;
      iarg++;
    } else if (strcmp(arg[iarg],"variables") == 0) {
      flags |= INFO_VARIABLES;
      iarg++;
    } else if (strcmp(arg[iarg],"regions") == 0) {
      flags |= INFO_REGIONS;
      iarg++;
    } else if (strcmp(arg[iarg],"groups") == 0) {
      flags |= INFO_GROUPS;
      iarg++;
    } else if (strcmp(arg[iarg],"species") == 0) {
      flags |= INFO_SPECIES;
      iarg++;
    } else if (strcmp(arg[iarg],"mixtures") == 0) {
      flags |= INFO_MIXTURES;
      iarg++;
    } else if (strcmp(arg[iarg],"surf_collide") == 0) {
      flags |= INFO_SURF_COLLIDE;
      iarg++;
    } else if (strcmp(arg[iarg],"surf_react") == 0) {
      flags |= INFO_SURF_REACT;
      iarg++;
    } else if (strcmp(arg[iarg],"memory") == 0) {
      flags |= INFO_MEMORY;
      iarg++;
    } else if (strcmp(arg[iarg],"time") == 0) {
      flags |= INFO_TIME;
      iarg++;
    } else if (strcmp(arg[iarg],"styles") == 0) {
      flags |= INFO_STYLES;
      iarg++;

    } else if (strcmp(arg[iarg],"out") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal info command");
      if (strcmp(arg[iarg+1],"screen") == 0) {
        out = screen;
        iarg += 2;
      } else if (strcmp(arg[iarg+1],"log") == 0) {
        out = logfile;
        iarg += 2;
      } else if (strcmp(arg[iarg+1],"append") == 0 ||
                 strcmp(arg[iarg+1],"overwrite") == 0) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal info command");
        const char *mode = (strcmp(arg[iarg+1],"append") == 0) ? "a" : "w";
        if (comm->me == 0) {
          if (fpclose) fclose(fpclose);
          fpclose = fopen(arg[iarg+2],mode);
          if (fpclose == NULL) {
            char str[128];
            sprintf(str,"Cannot open info file %s",arg[iarg+2]);
            error->one(FLERR,str);
          }
        }
        out = fpclose;
        iarg += 3;
      } else error->all(FLERR,"Illegal info command");

    } else error->all(FLERR,"Unknown info command keyword");
  }

  if (comm->me != 0) out = NULL;

  if (out) {
    time_t now = time(NULL);
    fprintf(out,"\nInfo-Info-Info-Info-Info-Info-Info-Info-Info-Info-Info\n");
    fprintf(out,"Printed on %s",ctime(&now));
  }

  if (flags & INFO_CONFIG) config();
  if (flags & INFO_SYSTEM) sysinfo();
  if (flags & INFO_COMM) comminfo();
  if (flags & INFO_COMPUTES) computes();
  if (flags & INFO_FIXES) fixes();
  if (flags & INFO_DUMPS) dumps();
  if (flags & INFO_VARIABLES) variables();
  if (flags & INFO_REGIONS) regions();
  if (flags & INFO_GROUPS) groups();
  if (flags & INFO_SPECIES) species();
  if (flags & INFO_MIXTURES) mixtures();
  if (flags & INFO_SURF_COLLIDE) surf_collide();
  if (flags & INFO_SURF_REACT) surf_react();
  if (flags & INFO_MEMORY) meminfo();
  if (flags & INFO_TIME) timeinfo();
  if (flags & INFO_STYLES) styles();

  if (out) {
    fprintf(out,"\nInfo-Info-Info-Info-Info-Info-Info-Info-Info-Info-Info\n\n");
    fflush(out);
  }

  if (fpclose) fclose(fpclose);
}

/* ----------------------------------------------------------------------
   version and compile-time settings of this executable
------------------------------------------------------------------------- */

void Info::config()
{
  if (!out) return;

  fprintf(out,"\nSPARTA version: %s\n",SPARTA_VERSION);

  fprintf(out,"sizeof(smallint) = %d, sizeof(cellint) = %d, "
          "sizeof(bigint) = %d\n",
          (int) sizeof(smallint),(int) sizeof(cellint),(int) sizeof(bigint));

#if defined(__cplusplus)
  fprintf(out,"C++ standard: %ld\n",(long) __cplusplus);
#endif
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
  fprintf(out,"Compiler: GNU C++ %d.%d.%d\n",
          __GNUC__,__GNUC_MINOR__,__GNUC_PATCHLEVEL__);
#elif defined(__clang__)
  fprintf(out,"Compiler: Clang C++ %d.%d.%d\n",
          __clang_major__,__clang_minor__,__clang_patchlevel__);
#elif defined(__INTEL_COMPILER)
  fprintf(out,"Compiler: Intel C++ %d\n",__INTEL_COMPILER);
#elif defined(_MSC_VER)
  fprintf(out,"Compiler: MSVC %d\n",_MSC_VER);
#endif

  fprintf(out,"\nCompile-time settings:\n");

#ifdef MPI_STUBS
  fprintf(out,"  MPI support:   no (MPI STUBS library)\n");
#else
  fprintf(out,"  MPI support:   yes\n");
#endif
#ifdef SPARTA_GZIP
  fprintf(out,"  GZIP support:  yes\n");
#else
  fprintf(out,"  GZIP support:  no\n");
#endif
#ifdef SPARTA_JPEG
  fprintf(out,"  JPEG support:  yes\n");
#else
  fprintf(out,"  JPEG support:  no\n");
#endif
#ifdef SPARTA_PNG
  fprintf(out,"  PNG support:   yes\n");
#else
  fprintf(out,"  PNG support:   no\n");
#endif
#ifdef SPARTA_FFMPEG
  fprintf(out,"  FFMPEG support: yes\n");
#else
  fprintf(out,"  FFMPEG support: no\n");
#endif
#ifdef SPARTA_EXCEPTIONS
  fprintf(out,"  Exceptions:    yes\n");
#else
  fprintf(out,"  Exceptions:    no\n");
#endif

  int npackage = 0;
  fprintf(out,"\nInstalled packages:\n ");
#ifdef SPARTA_KOKKOS
  fprintf(out," KOKKOS");
  npackage++;
#endif
#if defined(FFT_KISS) || defined(FFT_FFTW3) || defined(FFT_MKL)
  fprintf(out," FFT");
  npackage++;
#endif
#ifdef SPARTA_VTK
  fprintf(out," VTK");
  npackage++;
#endif
#ifdef SPARTA_PYTHON
  fprintf(out," PYTHON");
  npackage++;
#endif
  if (npackage == 0) fprintf(out," none");
  fprintf(out,"\n");

#ifdef SPARTA_KOKKOS
  fprintf(out,"\nKOKKOS backends:\n ");
#ifdef KOKKOS_ENABLE_SERIAL
  fprintf(out," Serial");
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  fprintf(out," OpenMP");
#endif
#ifdef KOKKOS_ENABLE_CUDA
  fprintf(out," Cuda");
#endif
#ifdef KOKKOS_ENABLE_HIP
  fprintf(out," HIP");
#endif
#ifdef KOKKOS_ENABLE_SYCL
  fprintf(out," SYCL");
#endif
  fprintf(out,"\n");
#endif

  if (sparta->suffix_enable && sparta->suffix)
    fprintf(out,"\nActive style suffix: %s\n",sparta->suffix);
}

/* ----------------------------------------------------------------------
   global settings, box, particles, grid, surfs, collide/react models
------------------------------------------------------------------------- */

void Info::sysinfo()
{
  if (out) {
    fprintf(out,"\nSystem information:\n");
    fprintf(out,"Units      = %s\n",update->unit_style);
    fprintf(out,"Dimension  = %d%s\n",domain->dimension,
            domain->axisymmetric ? " (axisymmetric)" : "");
    fprintf(out,"Timestep   = " BIGINT_FORMAT "\n",update->ntimestep);
    fprintf(out,"Time       = %g\n",update->time +
            (update->ntimestep - update->time_last_update) * update->dt);
    fprintf(out,"dt         = %g\n",update->dt);
    fprintf(out,"fnum       = %g\n",update->fnum);
    fprintf(out,"nrho       = %g\n",update->nrho);
    fprintf(out,"temp       = %g\n",update->temp_thermal);
    fprintf(out,"vstream    = %g %g %g\n",
            update->vstream[0],update->vstream[1],update->vstream[2]);

    if (!domain->box_exist) fprintf(out,"\nBox has not been created\n");
    else {
      static const char *bstr[] = {"periodic","outflow","reflect",
                                   "surface","axisymmetric"};
      fprintf(out,"\nBox = %g %g %g to %g %g %g\n",
              domain->boxlo[0],domain->boxlo[1],domain->boxlo[2],
              domain->boxhi[0],domain->boxhi[1],domain->boxhi[2]);
      const char *face[] = {"xlo","xhi","ylo","yhi","zlo","zhi"};
      fprintf(out,"Boundaries =");
      for (int i = 0; i < 6; i++) {
        int b = domain->bflag[i];
        const char *name = (b >= 0 && b <= AXISYM) ? bstr[b] : "unknown";
        fprintf(out," %s:%s",face[i],name);
      }
      fprintf(out,"\n");
    }
  }

  // particle counts are already global, grid/surf counts are global too
  // nlocal counts are per-proc, so reduce them for a min/max/ave summary

  bigint pnlocal = particle->nlocal;
  bigint pmin,pmax,psum;
  MPI_Allreduce(&pnlocal,&pmin,1,MPI_SPARTA_BIGINT,MPI_MIN,world);
  MPI_Allreduce(&pnlocal,&pmax,1,MPI_SPARTA_BIGINT,MPI_MAX,world);
  MPI_Allreduce(&pnlocal,&psum,1,MPI_SPARTA_BIGINT,MPI_SUM,world);

  bigint gnlocal = grid->nlocal;
  bigint gmin,gmax;
  MPI_Allreduce(&gnlocal,&gmin,1,MPI_SPARTA_BIGINT,MPI_MIN,world);
  MPI_Allreduce(&gnlocal,&gmax,1,MPI_SPARTA_BIGINT,MPI_MAX,world);

  if (!out) return;

  fprintf(out,"\nParticles = " BIGINT_FORMAT " total, per proc: "
          "ave = %g, min = " BIGINT_FORMAT ", max = " BIGINT_FORMAT "\n",
          psum,1.0*psum/comm->nprocs,pmin,pmax);
  fprintf(out,"Species = %d, Mixtures = %d\n",
          particle->nspecies,particle->nmixture);
  if (particle->ncustom) {
    fprintf(out,"Custom particle attributes =");
    for (int i = 0; i < particle->ncustom; i++)
      if (particle->ename[i]) fprintf(out," %s",particle->ename[i]);
    fprintf(out,"\n");
  }

  if (!grid->exist) fprintf(out,"\nGrid has not been created\n");
  else {
    fprintf(out,"\nGrid cells = " BIGINT_FORMAT " total ("
            BIGINT_FORMAT " unsplit, %d split, %d sub)\n",
            grid->ncell,grid->nunsplit,grid->nsplit,grid->nsub);
    fprintf(out,"Grid cells per proc: ave = %g, min = " BIGINT_FORMAT
            ", max = " BIGINT_FORMAT "\n",
            1.0*grid->ncell/comm->nprocs,gmin,gmax);
    fprintf(out,"Max grid level = %d, uniform = %s, ghost cutoff = %g\n",
            grid->maxlevel,grid->uniform ? "yes" : "no",grid->cutoff);
    fprintf(out,"Grid groups = %d:",grid->ngroup);
    for (int i = 0; i < grid->ngroup; i++) fprintf(out," %s",grid->gnames[i]);
    fprintf(out,"\n");
    if (grid->ncustom) {
      fprintf(out,"Custom grid attributes =");
      for (int i = 0; i < grid->ncustom; i++)
        if (grid->ename[i]) fprintf(out," %s",grid->ename[i]);
      fprintf(out,"\n");
    }
  }

  if (!surf->exist) fprintf(out,"\nNo surfaces are defined\n");
  else {
    fprintf(out,"\nSurf elements = " BIGINT_FORMAT " (%s, %s)\n",
            surf->nsurf,surf->implicit ? "implicit" : "explicit",
            surf->distributed ? "distributed" : "replicated");
    fprintf(out,"Surf groups = %d:",surf->ngroup);
    for (int i = 0; i < surf->ngroup; i++) fprintf(out," %s",surf->gnames[i]);
    fprintf(out,"\n");
    if (surf->ncustom) {
      fprintf(out,"Custom surf attributes =");
      for (int i = 0; i < surf->ncustom; i++)
        if (surf->ename[i]) fprintf(out," %s",surf->ename[i]);
      fprintf(out,"\n");
    }
  }

  fprintf(out,"\nCollide style = %s\n",
          collide ? collide->style : "none");
  fprintf(out,"React style   = %s\n",react ? react->style : "none");
}

/* ---------------------------------------------------------------------- */

void Info::comminfo()
{
  if (!out) return;

  fprintf(out,"\nCommunication information:\n");

#ifdef MPI_STUBS
  fprintf(out,"MPI library = SPARTA MPI STUBS (serial)\n");
#else
  int major,minor;
  MPI_Get_version(&major,&minor);
#if defined(MPI_VERSION) && (MPI_VERSION >= 3)
  char version[MPI_MAX_LIBRARY_VERSION_STRING];
  int len;
  MPI_Get_library_version(version,&len);

  // MPI library version strings can be multi-line, print only the first line

  char *newline = strchr(version,'\n');
  if (newline) *newline = '\0';

  fprintf(out,"MPI library = MPI v%d.%d: %s\n",major,minor,version);
#else
  fprintf(out,"MPI library = MPI v%d.%d\n",major,minor);
#endif
#endif

  fprintf(out,"Number of procs = %d\n",comm->nprocs);
  fprintf(out,"Particle comm = %s\n",
          comm->commpartstyle ? "neighbor" : "all");
  fprintf(out,"Comm sort = %s\n",comm->commsortflag ? "yes" : "no");
  fprintf(out,"Grid ownership is %s\n",
          grid->clumped ? "clumped" : "not clumped");
}

/* ---------------------------------------------------------------------- */

void Info::computes()
{
  if (!out) return;

  fprintf(out,"\nCompute information:\n");
  for (int i = 0; i < modify->ncompute; i++)
    fprintf(out,"Compute[%3d]: %-24s style = %s\n",
            i,modify->compute[i]->id,modify->compute[i]->style);
  if (modify->ncompute == 0) fprintf(out,"  none defined\n");
}

/* ---------------------------------------------------------------------- */

void Info::fixes()
{
  if (!out) return;

  fprintf(out,"\nFix information:\n");
  for (int i = 0; i < modify->nfix; i++)
    fprintf(out,"Fix[%3d]: %-24s style = %-20s every = %d\n",
            i,modify->fix[i]->id,modify->fix[i]->style,modify->fix[i]->nevery);
  if (modify->nfix == 0) fprintf(out,"  none defined\n");
}

/* ---------------------------------------------------------------------- */

void Info::dumps()
{
  if (!out) return;

  fprintf(out,"\nDump information:\n");
  for (int i = 0; i < output->ndump; i++)
    fprintf(out,"Dump[%3d]: %-24s style = %-12s every = %d\n",
            i,output->dump[i]->id,output->dump[i]->style,
            output->every_dump[i]);
  if (output->ndump == 0) fprintf(out,"  none defined\n");
}

/* ---------------------------------------------------------------------- */

void Info::variables()
{
  if (!out) return;

  Variable *variable = input->variable;
  int nvar = variable->nvar_active();

  fprintf(out,"\nVariable information:\n");
  for (int i = 0; i < nvar; i++)
    fprintf(out,"%s",variable->get_info(i).c_str());
  if (nvar == 0) fprintf(out,"  none defined\n");
}

/* ---------------------------------------------------------------------- */

void Info::regions()
{
  if (!out) return;

  fprintf(out,"\nRegion information:\n");
  for (int i = 0; i < domain->nregion; i++)
    fprintf(out,"Region[%3d]: %-24s style = %s\n",
            i,domain->regions[i]->id,domain->regions[i]->style);
  if (domain->nregion == 0) fprintf(out,"  none defined\n");
}

/* ---------------------------------------------------------------------- */

void Info::groups()
{
  if (!out) return;

  fprintf(out,"\nGroup information:\n");
  for (int i = 0; i < grid->ngroup; i++)
    fprintf(out,"Grid group[%3d]: %s\n",i,grid->gnames[i]);
  for (int i = 0; i < surf->ngroup; i++)
    fprintf(out,"Surf group[%3d]: %s\n",i,surf->gnames[i]);
}

/* ---------------------------------------------------------------------- */

void Info::species()
{
  if (!out) return;

  fprintf(out,"\nSpecies information:\n");
  for (int i = 0; i < particle->nspecies; i++) {
    Particle::Species *sp = &particle->species[i];
    fprintf(out,"Species[%3d]: %-16s molwt = %g, charge = %g, "
            "rotdof = %d, vibdof = %d\n",
            i,sp->id,sp->molwt,sp->charge,sp->rotdof,sp->vibdof);
  }
  if (particle->nspecies == 0) fprintf(out,"  none defined\n");
}

/* ---------------------------------------------------------------------- */

void Info::mixtures()
{
  if (!out) return;

  fprintf(out,"\nMixture information:\n");
  for (int i = 0; i < particle->nmixture; i++) {
    Mixture *mix = particle->mixture[i];
    fprintf(out,"Mixture[%3d]: %-16s nspecies = %d, ngroup = %d, species =",
            i,mix->id,mix->nspecies,mix->ngroup);
    for (int j = 0; j < mix->nspecies; j++)
      fprintf(out," %s",particle->species[mix->species[j]].id);
    fprintf(out,"\n");
  }
  if (particle->nmixture == 0) fprintf(out,"  none defined\n");
}

/* ---------------------------------------------------------------------- */

void Info::surf_collide()
{
  if (!out) return;

  fprintf(out,"\nSurface collision model information:\n");
  for (int i = 0; i < surf->nsc; i++)
    fprintf(out,"Surf collide[%3d]: %-24s style = %s\n",
            i,surf->sc[i]->id,surf->sc[i]->style);
  if (surf->nsc == 0) fprintf(out,"  none defined\n");
}

/* ---------------------------------------------------------------------- */

void Info::surf_react()
{
  if (!out) return;

  fprintf(out,"\nSurface reaction model information:\n");
  for (int i = 0; i < surf->nsr; i++)
    fprintf(out,"Surf react[%3d]: %-24s style = %s\n",
            i,surf->sr[i]->id,surf->sr[i]->style);
  if (surf->nsr == 0) fprintf(out,"  none defined\n");
}

/* ----------------------------------------------------------------------
   per-proc memory usage, same accounting as the end-of-run summary
------------------------------------------------------------------------- */

void Info::meminfo()
{
  bigint pbytes = particle->memory_usage();
  bigint gbytes = grid->memory_usage();
  bigint sbytes = surf->memory_usage();
  bigint mbytes = modify->memory_usage();
  bigint total = pbytes + gbytes + sbytes + mbytes;

  double scale = 1.0/1024.0/1024.0;
  const char *label[5] = {"particles","grid","surf","fixes/computes","total"};
  bigint value[5] = {pbytes,gbytes,sbytes,mbytes,total};

  double ave[5],mn[5],mx[5];
  for (int i = 0; i < 5; i++) {
    bigint sum,lo,hi;
    MPI_Allreduce(&value[i],&sum,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
    MPI_Allreduce(&value[i],&lo,1,MPI_SPARTA_BIGINT,MPI_MIN,world);
    MPI_Allreduce(&value[i],&hi,1,MPI_SPARTA_BIGINT,MPI_MAX,world);
    ave[i] = scale * sum/comm->nprocs;
    mn[i] = scale * lo;
    mx[i] = scale * hi;
  }

  if (!out) return;

  fprintf(out,"\nMemory usage per proc in Mbytes:\n");
  for (int i = 0; i < 5; i++)
    fprintf(out,"  %-16s (ave,min,max) = %g %g %g\n",
            label[i],ave[i],mn[i],mx[i]);
}

/* ---------------------------------------------------------------------- */

void Info::timeinfo()
{
  double wall = MPI_Wtime() - output->stats->wall_start();

  double cpu = 0.0;
#ifdef SPARTA_HAVE_RUSAGE
  struct rusage ru;
  if (getrusage(RUSAGE_SELF,&ru) == 0)
    cpu = (double) ru.ru_utime.tv_sec + 1.0e-6*ru.ru_utime.tv_usec +
      (double) ru.ru_stime.tv_sec + 1.0e-6*ru.ru_stime.tv_usec;
#endif

  if (!out) return;

  fprintf(out,"\nTotal time information (MPI rank 0):\n");
  fprintf(out,"  CPU time:  %8.4f seconds\n",cpu);
  fprintf(out,"  Wall time: %8.4f seconds\n",wall);

  // timer->array holds the breakdown of the most recent run

  if (timer->array[TIME_LOOP] > 0.0) {
    static const char *tname[TIME_N] =
      {"Loop","Move","Collide","Sort","Comm","Modify","Output"};
    double loop = timer->array[TIME_LOOP];
    fprintf(out,"\nBreakdown of the most recent run (MPI rank 0):\n");
    for (int i = 0; i < TIME_N; i++)
      fprintf(out,"  %-8s %10.4f seconds  %6.2f%%\n",
              tname[i],timer->array[i],100.0*timer->array[i]/loop);
  }
}

/* ----------------------------------------------------------------------
   list all styles compiled into this executable
------------------------------------------------------------------------- */

void Info::styles_category(const char *name, int n, const char **list)
{
  fprintf(out,"\n* %s styles (%d):\n",name,n);

  // column width from the longest style name, so nothing runs together

  int width = 0;
  for (int i = 0; i < n; i++)
    width = MAX(width,(int) strlen(list[i]));
  width += 2;

  int percol = MAX(1,76/width);

  for (int i = 0; i < n; i++) {
    if (i % percol == 0) fprintf(out,"  ");
    fprintf(out,"%-*s",width,list[i]);
    if (i % percol == percol-1) fprintf(out,"\n");
  }
  if (n % percol) fprintf(out,"\n");
}

void Info::styles()
{
  if (!out) return;

  fprintf(out,"\nStyles compiled into this executable:\n");

  {
    static const char *list[] = {
#define COMMAND_CLASS
#define CommandStyle(key,Class) #key,
#include "style_command.h"
#undef CommandStyle
#undef COMMAND_CLASS
      NULL};
    int n = sizeof(list)/sizeof(char *) - 1;
    styles_category("Command",n,list);
  }
  {
    static const char *list[] = {
#define COMPUTE_CLASS
#define ComputeStyle(key,Class) #key,
#include "style_compute.h"
#undef ComputeStyle
#undef COMPUTE_CLASS
      NULL};
    int n = sizeof(list)/sizeof(char *) - 1;
    styles_category("Compute",n,list);
  }
  {
    static const char *list[] = {
#define FIX_CLASS
#define FixStyle(key,Class) #key,
#include "style_fix.h"
#undef FixStyle
#undef FIX_CLASS
      NULL};
    int n = sizeof(list)/sizeof(char *) - 1;
    styles_category("Fix",n,list);
  }
  {
    static const char *list[] = {
#define DUMP_CLASS
#define DumpStyle(key,Class) #key,
#include "style_dump.h"
#undef DumpStyle
#undef DUMP_CLASS
      NULL};
    int n = sizeof(list)/sizeof(char *) - 1;
    styles_category("Dump",n,list);
  }
  {
    static const char *list[] = {
#define REGION_CLASS
#define RegionStyle(key,Class) #key,
#include "style_region.h"
#undef RegionStyle
#undef REGION_CLASS
      NULL};
    int n = sizeof(list)/sizeof(char *) - 1;
    styles_category("Region",n,list);
  }
  {
    static const char *list[] = {
#define COLLIDE_CLASS
#define CollideStyle(key,Class) #key,
#include "style_collide.h"
#undef CollideStyle
#undef COLLIDE_CLASS
      NULL};
    int n = sizeof(list)/sizeof(char *) - 1;
    styles_category("Collide",n,list);
  }
  {
    static const char *list[] = {
#define REACT_CLASS
#define ReactStyle(key,Class) #key,
#include "style_react.h"
#undef ReactStyle
#undef REACT_CLASS
      NULL};
    int n = sizeof(list)/sizeof(char *) - 1;
    styles_category("React",n,list);
  }
  {
    static const char *list[] = {
#define SURF_COLLIDE_CLASS
#define SurfCollideStyle(key,Class) #key,
#include "style_surf_collide.h"
#undef SurfCollideStyle
#undef SURF_COLLIDE_CLASS
      NULL};
    int n = sizeof(list)/sizeof(char *) - 1;
    styles_category("Surf collide",n,list);
  }
  {
    static const char *list[] = {
#define SURF_REACT_CLASS
#define SurfReactStyle(key,Class) #key,
#include "style_surf_react.h"
#undef SurfReactStyle
#undef SURF_REACT_CLASS
      NULL};
    int n = sizeof(list)/sizeof(char *) - 1;
    styles_category("Surf react",n,list);
  }
}
