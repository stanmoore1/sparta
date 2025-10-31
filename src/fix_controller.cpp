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
   File adapted from LAMMPS (https://www.lammps.org), October 2024
   Ported to SPARTA by: Stan Moore (SNL)
   Original Author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include "fix_controller.h"

#include "compute.h"
#include "error.h"
#include "input.h"
#include "modify.h"
#include "update.h"
#include "variable.h"

using namespace SPARTA_NS;
using namespace FixConst;

enum { COMPUTE, FIX, VARIABLE };

/* ---------------------------------------------------------------------- */

FixController::FixController(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg),
  pvID(nullptr), cvID(nullptr)
{
  if (narg != 11) error->all(FLERR,"Illegal fix controller command");

  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extvector = 0;

  nevery = utils::inumeric(FLERR,arg[3],false,sparta);
  if (nevery <= 0) {
    char msg[128];
    sprintf(msg, "Illegal fix controller nevery value %i", nevery);
    error->all(FLERR, msg);
  }

  alpha = utils::numeric(FLERR,arg[4],false,sparta);
  kp = utils::numeric(FLERR,arg[5],false,sparta);
  ki = utils::numeric(FLERR,arg[6],false,sparta);
  kd = utils::numeric(FLERR,arg[7],false,sparta);

  // process variable arg

  char* str = arg[iarg];
  int n = strlen(&str[2]) + 1;
  pvID = new char[n];
  strcpy(idvar,&str[2]);

  if (!utils::strmatch(arg[8],"^c_")) {
    pvwhich = COMPUTE;
    pvindex = input->compute->find(pvID);
  } else if (!utils::strmatch(arg[8],"^f_")) {
    pvwhich = FIX;
    pvindex = input->fix->find(pvID);
  } else if (!utils::strmatch(arg[8],"^v_")) {
    pvwhich = VARIABLE;
    pvindex = input->variable->find(pvID);
  } else {
    char msg[128];
    sprintf(msg, "Illegal fix controller argument %i", arg[8]);
    error->all(FLERR, msg);
  }

  // setpoint arg

  setpoint = utils::numeric(FLERR,arg[9],false,sparta);

  // control variable arg

  cvID = utils::strdup(arg[10]);

  // error check

  if (pvwhich == COMPUTE) {
    Compute *c = modify->get_compute_by_id(pvID);
    if (!c) {
      char msg[128];
      sprintf(msg, "Compute ID %s for fix controller does not exist", pvID);
      error->all(FLERR, msg);
    }
    int flag = 0;
    if (c->scalar_flag && pvindex == 0) flag = 1;
    else if (c->vector_flag && pvindex > 0) flag = 1;
    if (!flag)
      error->all(FLERR, 8, "Fix controller compute {} does not calculate a global scalar or "
                 "vector", pvID);
    if (pvindex && pvindex > c->size_vector)
      error->all(FLERR, 8, "Fix controller compute {} vector is accessed out-of-range{}",
                 pvID, utils::errorurl(20));
  } else if (pvwhich == FIX) {
    Fix *f = modify->get_fix_by_id(pvID);
    if (!f) error->all(FLERR, 8, "Fix ID {} for fix controller does not exist", pvID);
    int flag = 0;
    if (f->scalar_flag && pvindex == 0) flag = 1;
    else if (f->vector_flag && pvindex > 0) flag = 1;
    if (!flag)
      error->all(FLERR, 8, "Fix controller fix {} does not calculate a global scalar or vector",
                 pvID);
    if (pvindex && pvindex > f->size_vector)
      error->all(FLERR, 8, "Fix controller fix {} vector is accessed out-of-range{}", pvID,
                 utils::errorurl(20));
  } else if (pvwhich == VARIABLE) {
    int ivariable = input->variable->find(pvID);
    if (ivariable < 0)
      error->all(FLERR, 8, "Variable name {} for fix controller does not exist", pvID);
    if (input->variable->equalstyle(ivariable) == 0)
      error->all(FLERR, 8, "Fix controller variable {} is not equal-style variable", pvID);
  }

  int ivariable = input->variable->find(cvID);
  if (ivariable < 0)
    error->all(FLERR, 10, "Variable name {} for fix controller does not exist", cvID);
  if (input->variable->internalstyle(ivariable) == 0)
    error->all(FLERR, 10, "Fix controller variable {} is not internal-style variable", cvID);
  control = input->variable->compute_equal(ivariable);

  firsttime = 1;
}

/* ---------------------------------------------------------------------- */

FixController::~FixController()
{
  delete [] pvID;
  delete [] cvID;
}

/* ---------------------------------------------------------------------- */

int FixController::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixController::init()
{
  if (pvwhich == COMPUTE) {
    pcompute = modify->get_compute_by_id(pvID);
    if (!pcompute)
      error->all(FLERR, Error::NOLASTLINE,
                 "Compute ID {} for fix controller does not exist", pvID);

  } else if (pvwhich == FIX) {
    pfix = modify->get_fix_by_id(pvID);
    if (!pfix)
      error->all(FLERR, Error::NOLASTLINE, "Fix ID {} for fix controller does not exist", pvID);

  } else if (pvwhich == VARIABLE) {
    pvar = input->variable->find(pvID);
    if (pvar < 0)
      error->all(FLERR, Error::NOLASTLINE, "Variable name {} for fix controller does not exist",
                 pvID);
  }

  cvar = input->variable->find(cvID);
  if (cvar < 0)
    error->all(FLERR, Error::NOLASTLINE, "Variable name {} for fix controller does not exist",
               cvID);

  // set sampling time

  tau = nevery * update->dt;
}

/* ---------------------------------------------------------------------- */

void FixController::end_of_step()
{
  // current value of pv = invocation of compute,fix,variable
  // compute/fix/variable may invoke computes so wrap with clear/add

  modify->clearstep_compute();

  // invoke compute if not previously invoked

  double current = 0.0;

  if (pvwhich == COMPUTE) {
    if (pvindex == 0) {
      if (!(pcompute->invoked_flag & Compute::INVOKED_SCALAR)) {
        pcompute->compute_scalar();
        pcompute->invoked_flag |= Compute::INVOKED_SCALAR;
      }
      current = pcompute->scalar;
    } else {
      if (!(pcompute->invoked_flag & Compute::INVOKED_VECTOR)) {
        pcompute->compute_vector();
        pcompute->invoked_flag |= Compute::INVOKED_VECTOR;
      }
      current = pcompute->vector[pvindex-1];
    }

  // access fix field, guaranteed to be ready

  } else if (pvwhich == FIX) {
    if (pvindex == 0) current = pfix->compute_scalar();
    else current = pfix->compute_vector(pvindex-1);

  // evaluate equal-style variable

  } else if (pvwhich == VARIABLE) {
    current = input->variable->compute_equal(pvar);
  }

  modify->addstep_compute(update->ntimestep + nevery);

  // new control var = f(old value, current process var, setpoint)
  // cv = cvold -kp*err -ki*sumerr -kd*deltaerr
  // note: this deviates from standard notation, which is
  // cv = kp*err +ki*sumerr +kd*deltaerr
  // the difference is in the sign and the time integral

  err = current - setpoint;

  if (firsttime) {
    firsttime = 0;
    deltaerr = sumerr = 0.0;
  } else {
    deltaerr = err - olderr;
  }
  sumerr += err;

  // 3 terms of PID equation

  control += -kp * alpha * tau * err;
  control += -ki * alpha * tau * tau * sumerr;
  control += -kd * alpha * deltaerr;
  olderr = err;

  // reset control variable

  input->variable->internal_set(cvar,control);
}

/* ---------------------------------------------------------------------- */

void FixController::reset_dt()
{
  tau = nevery * update->dt;
}

/* ----------------------------------------------------------------------
   return 3 terms of PID controller at last invocation of end_of_step()
------------------------------------------------------------------------- */

double FixController::compute_vector(int n)
{
  if (n == 0) return (-kp * alpha * tau * err);
  else if (n == 1) return (-ki * alpha * tau * tau * sumerr);
  else return (-kd * alpha * deltaerr);
}
