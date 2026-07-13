#!/usr/bin/env python3
# ----------------------------------------------------------------------
#   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
#   http://sparta.github.io
#   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
#   Sandia National Laboratories
#
#   Copyright (2014) Sandia Corporation.  Under the terms of Contract
#   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
#   certain rights in this software.  This software is distributed under
#   the GNU General Public License.
#
#   See the README file in the top-level SPARTA directory.
# -------------------------------------------------------------------------

# Smoke test for the SPARTA Python wrapper (python/sparta.py) and the
# underlying C library interface (src/library.cpp).  Neither is exercised
# by the regression suite, even though the CI builds SPARTA with PKG_PYTHON.
#
# The test drives a small free-molecular simulation entirely through the
# library API (command / file / run / extract) and checks that the values
# returned across the ctypes boundary are correct.  It requires SPARTA to be
# built as a shared library (-DBUILD_SHARED_LIBS=ON) so libsparta.so can be
# loaded by ctypes.
#
# Exit code is 0 on success, nonzero on any failed check, which CTest reads
# as pass/fail.

import os
import sys

# locate the sparta.py wrapper (this file lives next to it)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparta import sparta

nfail = 0


def check(cond, msg):
    global nfail
    if cond:
        print("ok   - %s" % msg)
    else:
        nfail += 1
        print("FAIL - %s" % msg)


def check_close(a, b, tol, msg):
    check(abs(a - b) <= tol, "%s (got %.10g, expected %.10g)" % (msg, a, b))


# ------------------------------------------------------------------
# drive a small thermal gas in a closed box via the command API
# ------------------------------------------------------------------

spa = sparta(cmdargs=["-screen", "none", "-log", "none"])

cmds = """
seed             12345
dimension        3
global           gridcut 1.0e-5 comm/sort yes
boundary         rr rr rr
create_box       0 0.0001 0 0.0001 0 0.0001
create_grid      10 10 10
balance_grid     rcb part
species          ../examples/free/ar.species Ar
mixture          air Ar vstream 0.0 0.0 0.0 temp 273.15
global           nrho 7.07043E22
global           fnum 7.07043E6
create_particles air n 10000 twopass
compute          gtemp temp
stats            100
stats_style      step np c_gtemp
timestep         7.00E-9
"""

for line in cmds.strip().splitlines():
    line = line.strip()
    if line:
        spa.command(line)

# scalar globals set above should round-trip back through the API
check_close(spa.extract_global("fnum", 1), 7.07043e6, 1.0, "extract_global fnum")
check_close(spa.extract_global("nrho", 1), 7.07043e22, 1.0e16, "extract_global nrho")
check_close(spa.extract_global("dt", 1), 7.00e-9, 1.0e-15, "extract_global dt")

# closed box: all 10000 created particles should be present locally (serial)
nplocal = spa.extract_global("nplocal", 0)
check(nplocal == 10000, "extract_global nplocal == 10000 (got %d)" % nplocal)

# run and read the temperature compute back through the library
spa.command("run 200")

temp = spa.extract_compute("gtemp", 0, 0)
check(temp is not None, "extract_compute gtemp returned a value")
# thermal gas held near its initial temperature in a closed box
check(200.0 < temp < 350.0, "compute temp in physical range (got %.4f)" % temp)

# particle count is conserved in a closed box
nplocal2 = spa.extract_global("nplocal", 0)
check(nplocal2 == 10000, "particle count conserved after run (got %d)" % nplocal2)

spa.close()

print("\n%d failures" % nfail)
if nfail == 0:
    print("All library API tests passed")
sys.exit(1 if nfail else 0)
