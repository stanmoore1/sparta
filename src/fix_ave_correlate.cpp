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
   Contributing authors: Benoit Leblanc, Dave Rigby, Paul Saxe (Materials
   Design) and Reese Jones (Sandia), original LAMMPS version
------------------------------------------------------------------------- */

#include "spatype.h"
#include "stdlib.h"
#include "string.h"
#include "unistd.h"
#include "fix_ave_correlate.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

enum{COMPUTE,FIX,VARIABLE};
enum{ONE,RUNNING};
enum{AUTO,UPPER,LOWER,AUTOUPPER,AUTOLOWER,FULL};

#define INVOKED_SCALAR 1
#define INVOKED_VECTOR 2

/* ---------------------------------------------------------------------- */

FixAveCorrelate::FixAveCorrelate(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix ave/correlate command");

  MPI_Comm_rank(world,&me);

  nevery = atoi(arg[2]);
  nrepeat = atoi(arg[3]);
  nfreq = atoi(arg[4]);

  time_depend = 1;
  global_freq = nfreq;

  // scan values to count them, then read options

  nvalues = 0;

  int iarg = 5;
  while (iarg < narg) {
    if ((strncmp(arg[iarg],"c_",2) == 0) ||
        (strncmp(arg[iarg],"f_",2) == 0) ||
        (strncmp(arg[iarg],"v_",2) == 0)) {
      nvalues++;
      iarg++;
    } else break;
  }

  if (nvalues == 0) error->all(FLERR,"No values in fix ave/correlate command");

  // option defaults

  fp = NULL;
  type = AUTO;
  ave = ONE;
  startstep = 0;
  prefactor = 1.0;
  overwrite = 0;
  char *title1 = NULL;
  char *title2 = NULL;
  char *title3 = NULL;

  int ioarg = iarg;
  while (ioarg < narg) {
    if (strcmp(arg[ioarg],"type") == 0) {
      if (ioarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      if (strcmp(arg[ioarg+1],"auto") == 0) type = AUTO;
      else if (strcmp(arg[ioarg+1],"upper") == 0) type = UPPER;
      else if (strcmp(arg[ioarg+1],"lower") == 0) type = LOWER;
      else if (strcmp(arg[ioarg+1],"auto/upper") == 0) type = AUTOUPPER;
      else if (strcmp(arg[ioarg+1],"auto/lower") == 0) type = AUTOLOWER;
      else if (strcmp(arg[ioarg+1],"full") == 0) type = FULL;
      else error->all(FLERR,"Illegal fix ave/correlate command");
      ioarg += 2;
    } else if (strcmp(arg[ioarg],"ave") == 0) {
      if (ioarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      if (strcmp(arg[ioarg+1],"one") == 0) ave = ONE;
      else if (strcmp(arg[ioarg+1],"running") == 0) ave = RUNNING;
      else error->all(FLERR,"Illegal fix ave/correlate command");
      ioarg += 2;
    } else if (strcmp(arg[ioarg],"start") == 0) {
      if (ioarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      startstep = atoi(arg[ioarg+1]);
      ioarg += 2;
    } else if (strcmp(arg[ioarg],"prefactor") == 0) {
      if (ioarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      prefactor = atof(arg[ioarg+1]);
      ioarg += 2;
    } else if (strcmp(arg[ioarg],"file") == 0) {
      if (ioarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      if (me == 0) {
        fp = fopen(arg[ioarg+1],"w");
        if (fp == NULL) {
          char str[128];
          sprintf(str,"Cannot open fix ave/correlate file %s",arg[ioarg+1]);
          error->one(FLERR,str);
        }
      }
      ioarg += 2;
    } else if (strcmp(arg[ioarg],"overwrite") == 0) {
      overwrite = 1;
      ioarg += 1;
    } else if (strcmp(arg[ioarg],"title1") == 0) {
      if (ioarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      delete [] title1;
      int n = strlen(arg[ioarg+1]) + 1;
      title1 = new char[n];
      strcpy(title1,arg[ioarg+1]);
      ioarg += 2;
    } else if (strcmp(arg[ioarg],"title2") == 0) {
      if (ioarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      delete [] title2;
      int n = strlen(arg[ioarg+1]) + 1;
      title2 = new char[n];
      strcpy(title2,arg[ioarg+1]);
      ioarg += 2;
    } else if (strcmp(arg[ioarg],"title3") == 0) {
      if (ioarg+2 > narg) error->all(FLERR,"Illegal fix ave/correlate command");
      delete [] title3;
      int n = strlen(arg[ioarg+1]) + 1;
      title3 = new char[n];
      strcpy(title3,arg[ioarg+1]);
      ioarg += 2;
    } else error->all(FLERR,"Illegal fix ave/correlate command");
  }

  // expand values if any have wildcard character "*"
  // this can reset nvalues
  // mode 0 = expand a global vector into its elements

  int expand = 0;
  char **earg;
  nvalues = input->expand_args(nvalues,&arg[5],0,earg);

  if (earg != &arg[5]) expand = 1;
  char **varg = earg;

  // parse values

  which = new int[nvalues];
  argindex = new int[nvalues];
  value2index = new int[nvalues];
  ids = new char*[nvalues];

  for (int i = 0; i < nvalues; i++) {
    if (varg[i][0] == 'c') which[i] = COMPUTE;
    else if (varg[i][0] == 'f') which[i] = FIX;
    else if (varg[i][0] == 'v') which[i] = VARIABLE;

    int n = strlen(varg[i]);
    char *suffix = new char[n];
    strcpy(suffix,&varg[i][2]);

    char *ptr = strchr(suffix,'[');
    if (ptr) {
      if (suffix[strlen(suffix)-1] != ']')
        error->all(FLERR,"Illegal fix ave/correlate command");
      argindex[i] = atoi(ptr+1);
      *ptr = '\0';
    } else argindex[i] = 0;

    n = strlen(suffix) + 1;
    ids[i] = new char[n];
    strcpy(ids[i],suffix);
    delete [] suffix;
  }

  // setup and error check
  // for fix inputs, check that fix frequency is acceptable

  if (nevery <= 0 || nrepeat <= 0 || nfreq <= 0)
    error->all(FLERR,"Illegal fix ave/correlate command");
  if (nfreq % nevery) error->all(FLERR,"Illegal fix ave/correlate command");
  if (ave != RUNNING && overwrite)
    error->all(FLERR,"Illegal fix ave/correlate command");

  for (int i = 0; i < nvalues; i++) {
    if (which[i] == COMPUTE) {
      int icompute = modify->find_compute(ids[i]);
      if (icompute < 0)
        error->all(FLERR,"Compute ID for fix ave/correlate does not exist");
      if (argindex[i] == 0 && modify->compute[icompute]->scalar_flag == 0)
        error->all(FLERR,
                   "Fix ave/correlate compute does not calculate a scalar");
      if (argindex[i] && modify->compute[icompute]->vector_flag == 0)
        error->all(FLERR,
                   "Fix ave/correlate compute does not calculate a vector");
      if (argindex[i] && argindex[i] > modify->compute[icompute]->size_vector)
        error->all(FLERR,
                   "Fix ave/correlate compute vector is accessed out-of-range");

    } else if (which[i] == FIX) {
      int ifix = modify->find_fix(ids[i]);
      if (ifix < 0)
        error->all(FLERR,"Fix ID for fix ave/correlate does not exist");
      if (argindex[i] == 0 && modify->fix[ifix]->scalar_flag == 0)
        error->all(FLERR,"Fix ave/correlate fix does not calculate a scalar");
      if (argindex[i] && modify->fix[ifix]->vector_flag == 0)
        error->all(FLERR,"Fix ave/correlate fix does not calculate a vector");
      if (argindex[i] && argindex[i] > modify->fix[ifix]->size_vector)
        error->all(FLERR,
                   "Fix ave/correlate fix vector is accessed out-of-range");
      if (nevery % modify->fix[ifix]->global_freq)
        error->all(FLERR,
                   "Fix for fix ave/correlate not computed at compatible time");

    } else if (which[i] == VARIABLE) {
      int ivariable = input->variable->find(ids[i]);
      if (ivariable < 0)
        error->all(FLERR,
                   "Variable name for fix ave/correlate does not exist");
      if (input->variable->equal_style(ivariable) == 0)
        error->all(FLERR,
                   "Fix ave/correlate variable is not equal-style variable");
    }
  }

  // npair = # of correlation pairs to calculate

  if (type == AUTO) npair = nvalues;
  else if (type == UPPER || type == LOWER) npair = nvalues*(nvalues-1)/2;
  else if (type == AUTOUPPER || type == AUTOLOWER)
    npair = nvalues*(nvalues+1)/2;
  else npair = nvalues*nvalues;

  // print file comment lines

  if (fp && me == 0) {
    clearerr(fp);
    if (title1) fprintf(fp,"%s\n",title1);
    else fprintf(fp,"# Time-correlated data for fix %s\n",id);
    if (title2) fprintf(fp,"%s\n",title2);
    else fprintf(fp,"# TimeStep Number-of-time-windows\n");
    if (title3) fprintf(fp,"%s\n",title3);
    else {
      fprintf(fp,"# Index TimeDelta Ncount");
      if (type == AUTO)
        for (int i = 0; i < nvalues; i++)
          fprintf(fp," %s*%s",varg[i],varg[i]);
      else if (type == UPPER)
        for (int i = 0; i < nvalues; i++)
          for (int j = i+1; j < nvalues; j++)
            fprintf(fp," %s*%s",varg[i],varg[j]);
      else if (type == LOWER)
        for (int i = 0; i < nvalues; i++)
          for (int j = 0; j < i; j++)
            fprintf(fp," %s*%s",varg[i],varg[j]);
      else if (type == AUTOUPPER)
        for (int i = 0; i < nvalues; i++)
          for (int j = i; j < nvalues; j++)
            fprintf(fp," %s*%s",varg[i],varg[j]);
      else if (type == AUTOLOWER)
        for (int i = 0; i < nvalues; i++)
          for (int j = 0; j <= i; j++)
            fprintf(fp," %s*%s",varg[i],varg[j]);
      else
        for (int i = 0; i < nvalues; i++)
          for (int j = 0; j < nvalues; j++)
            fprintf(fp," %s*%s",varg[i],varg[j]);
      fprintf(fp,"\n");
    }
    if (ferror(fp))
      error->one(FLERR,"Error writing out correlation data");
    filepos = ftell(fp);
  }

  delete [] title1;
  delete [] title2;
  delete [] title3;

  // if wildcard expansion occurred, free earg memory from expand_args()
  // wait to do this until after file comment lines are printed

  if (expand) {
    for (int i = 0; i < nvalues; i++) delete [] earg[i];
    memory->sfree(earg);
  }

  // allocate and initialize memory
  // values = 2d array with nvalues columns, nrepeat rows
  // count and corr = nrepeat by npair

  memory->create(values,nrepeat,nvalues,"ave/correlate:values");
  memory->create(count,nrepeat,"ave/correlate:count");
  memory->create(corr,nrepeat,npair,"ave/correlate:corr");

  memory->create(save_count,nrepeat,"ave/correlate:save_count");
  memory->create(save_corr,nrepeat,npair,"ave/correlate:save_corr");

  for (int i = 0; i < nrepeat; i++) {
    save_count[i] = 0.0;
    for (int j = 0; j < npair; j++) save_corr[i][j] = 0.0;
  }

  for (int i = 0; i < nrepeat; i++) {
    count[i] = 0.0;
    for (int j = 0; j < npair; j++) corr[i][j] = 0.0;
  }

  // this fix produces a global array
  // rows = time windows, cols = time delta, sample count, Cij values

  array_flag = 1;
  size_array_rows = nrepeat;
  size_array_cols = npair+2;

  // nvalid = next step on which end_of_step does something
  // add nvalid to all computes that store invocation times
  // since don't know a priori which are invoked by this fix
  // once in end_of_step() can set timestep for ones actually invoked

  firstindex = 0;
  lastindex = -1;
  nsample = 0;

  nvalid = nextvalid();
  modify->addstep_compute_all(nvalid);
}

/* ---------------------------------------------------------------------- */

FixAveCorrelate::~FixAveCorrelate()
{
  delete [] which;
  delete [] argindex;
  delete [] value2index;
  for (int i = 0; i < nvalues; i++) delete [] ids[i];
  delete [] ids;

  memory->destroy(values);
  memory->destroy(count);
  memory->destroy(corr);
  memory->destroy(save_count);
  memory->destroy(save_corr);

  if (fp && me == 0) fclose(fp);
}

/* ---------------------------------------------------------------------- */

int FixAveCorrelate::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAveCorrelate::init()
{
  // set current indices for all computes,fixes,variables

  for (int i = 0; i < nvalues; i++) {
    if (which[i] == COMPUTE) {
      int icompute = modify->find_compute(ids[i]);
      if (icompute < 0)
        error->all(FLERR,"Compute ID for fix ave/correlate does not exist");
      value2index[i] = icompute;

    } else if (which[i] == FIX) {
      int ifix = modify->find_fix(ids[i]);
      if (ifix < 0)
        error->all(FLERR,"Fix ID for fix ave/correlate does not exist");
      value2index[i] = ifix;

    } else if (which[i] == VARIABLE) {
      int ivariable = input->variable->find(ids[i]);
      if (ivariable < 0)
        error->all(FLERR,"Variable name for fix ave/correlate does not exist");
      value2index[i] = ivariable;
    }
  }
}

/* ----------------------------------------------------------------------
   only does something if nvalid = current timestep
------------------------------------------------------------------------- */

void FixAveCorrelate::setup()
{
  end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixAveCorrelate::end_of_step()
{
  int i,j,m;
  double scalar;

  // skip if not step which requires doing something

  bigint ntimestep = update->ntimestep;
  if (ntimestep != nvalid) return;

  // accumulate results of computes,fixes,variables to local copy
  // compute/fix/variable may invoke computes so wrap with clear/add

  modify->clearstep_compute();

  lastindex++;
  if (lastindex == nrepeat) lastindex = 0;

  for (i = 0; i < nvalues; i++) {
    m = value2index[i];

    // invoke compute if not previously invoked

    if (which[i] == COMPUTE) {
      Compute *compute = modify->compute[m];

      if (argindex[i] == 0) {
        if (!(compute->invoked_flag & INVOKED_SCALAR)) {
          compute->compute_scalar();
          compute->invoked_flag |= INVOKED_SCALAR;
        }
        scalar = compute->scalar;
      } else {
        if (!(compute->invoked_flag & INVOKED_VECTOR)) {
          compute->compute_vector();
          compute->invoked_flag |= INVOKED_VECTOR;
        }
        scalar = compute->vector[argindex[i]-1];
      }

    // access fix fields, guaranteed to be ready

    } else if (which[i] == FIX) {
      if (argindex[i] == 0)
        scalar = modify->fix[m]->compute_scalar();
      else
        scalar = modify->fix[m]->compute_vector(argindex[i]-1);

    // evaluate equal-style variable

    } else if (which[i] == VARIABLE)
      scalar = input->variable->compute_equal(m);

    values[lastindex][i] = scalar;
  }

  // firstindex = index in values ring of earliest time sample
  // nsample = number of time samples in values ring

  if (nsample < nrepeat) nsample++;
  else {
    firstindex++;
    if (firstindex == nrepeat) firstindex = 0;
  }

  nvalid += nevery;
  modify->addstep_compute(nvalid);

  // calculate all Cij() enabled by latest values

  accumulate();

  if (ntimestep % nfreq) return;

  // save results in save_count and save_corr

  for (i = 0; i < nrepeat; i++) {
    save_count[i] = count[i];
    if (count[i])
      for (j = 0; j < npair; j++)
        save_corr[i][j] = prefactor*corr[i][j]/count[i];
    else
      for (j = 0; j < npair; j++)
        save_corr[i][j] = 0.0;
  }

  // output result to file

  if (fp && me == 0) {
    clearerr(fp);
    if (overwrite) fseek(fp,filepos,SEEK_SET);
    fprintf(fp,BIGINT_FORMAT " %d\n",ntimestep,nrepeat);
    for (i = 0; i < nrepeat; i++) {
      fprintf(fp,"%d %d %g",i+1,i*nevery,count[i]);
      if (count[i])
        for (j = 0; j < npair; j++)
          fprintf(fp," %g",prefactor*corr[i][j]/count[i]);
      else
        for (j = 0; j < npair; j++)
          fprintf(fp," 0.0");
      fprintf(fp,"\n");
    }
    if (ferror(fp))
      error->one(FLERR,"Error writing out correlation data");
    fflush(fp);
    if (overwrite) {
      long fileend = ftell(fp);
      if (fileend > 0) {
        int tmp = ftruncate(fileno(fp),fileend);
        (void) tmp;
      }
    }
  }

  // zero accumulation if requested
  // recalculate Cij(0)

  if (ave == ONE) {
    for (i = 0; i < nrepeat; i++) {
      count[i] = 0.0;
      for (j = 0; j < npair; j++) corr[i][j] = 0.0;
    }
    nsample = 1;
    accumulate();
  }
}

/* ----------------------------------------------------------------------
   accumulate correlation data using more recently added values
------------------------------------------------------------------------- */

void FixAveCorrelate::accumulate()
{
  int i,j,k,m,n,ipair;

  for (k = 0; k < nsample; k++) count[k] += 1.0;

  if (type == AUTO) {
    m = n = lastindex;
    for (k = 0; k < nsample; k++) {
      ipair = 0;
      for (i = 0; i < nvalues; i++) {
        corr[k][ipair++] += values[m][i]*values[n][i];
      }
      m--;
      if (m < 0) m = nrepeat-1;
    }
  } else if (type == UPPER) {
    m = n = lastindex;
    for (k = 0; k < nsample; k++) {
      ipair = 0;
      for (i = 0; i < nvalues; i++)
        for (j = i+1; j < nvalues; j++)
          corr[k][ipair++] += values[m][i]*values[n][j];
      m--;
      if (m < 0) m = nrepeat-1;
    }
  } else if (type == LOWER) {
    m = n = lastindex;
    for (k = 0; k < nsample; k++) {
      ipair = 0;
      for (i = 0; i < nvalues; i++)
        for (j = 0; j < i; j++)
          corr[k][ipair++] += values[m][i]*values[n][j];
      m--;
      if (m < 0) m = nrepeat-1;
    }
  } else if (type == AUTOUPPER) {
    m = n = lastindex;
    for (k = 0; k < nsample; k++) {
      ipair = 0;
      for (i = 0; i < nvalues; i++)
        for (j = i; j < nvalues; j++)
          corr[k][ipair++] += values[m][i]*values[n][j];
      m--;
      if (m < 0) m = nrepeat-1;
    }
  } else if (type == AUTOLOWER) {
    m = n = lastindex;
    for (k = 0; k < nsample; k++) {
      ipair = 0;
      for (i = 0; i < nvalues; i++)
        for (j = 0; j <= i; j++)
          corr[k][ipair++] += values[m][i]*values[n][j];
      m--;
      if (m < 0) m = nrepeat-1;
    }
  } else if (type == FULL) {
    m = n = lastindex;
    for (k = 0; k < nsample; k++) {
      ipair = 0;
      for (i = 0; i < nvalues; i++)
        for (j = 0; j < nvalues; j++)
          corr[k][ipair++] += values[m][i]*values[n][j];
      m--;
      if (m < 0) m = nrepeat-1;
    }
  }
}

/* ----------------------------------------------------------------------
   return I,J array value
------------------------------------------------------------------------- */

double FixAveCorrelate::compute_array(int i, int j)
{
  if (j == 0) return 1.0*i*nevery;
  else if (j == 1) return save_count[i];
  else if (save_count[i]) return save_corr[i][j-2];
  return 0.0;
}

/* ----------------------------------------------------------------------
   nvalid = next timestep on which end_of_step does something
------------------------------------------------------------------------- */

bigint FixAveCorrelate::nextvalid()
{
  bigint nvalid = update->ntimestep;
  if (startstep > nvalid) nvalid = startstep;
  if (nvalid % nevery) nvalid = (nvalid/nevery)*nevery + nevery;
  return nvalid;
}
