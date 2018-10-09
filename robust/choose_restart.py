#!/usr/bin/python

import sys
import os
import glob

DIR=sys.argv[1]
NUMBER=0

not1=False
not2=False
prefer1=False
prefer2=False

os.chdir(DIR)

time1 = 0.0
time2 = 0.0

# toggle between restart.%.1 and restart.%.2 in ${DIR}/restart/ 
# choose best restart file based on timestamp


# make sure files exist

try:
  nfiles1 = len(glob.glob("restart.*.1"))
  time1 = os.path.getmtime("restart.base.1")
except:
  not1=True
  #print "not1 True from file not found"

try:
  nfiles2 = len(glob.glob("restart.*.2"))
  time2 = os.path.getmtime("restart.base.2")
except:
  not2=True
  #print "not2 True from file not found"


# make sure previous restart didn't fail due to corruption

if os.path.exists('./SPARTA_RESTART_FAIL_1'):
  not1=True
  #print "not1 True from previous failure"

if os.path.exists('./SPARTA_RESTART_FAIL_2'):
  not2=True
  #print "not2 True from previous failure"

if os.path.exists('./SPARTA_RESTART_FAIL_1') and os.path.exists('./SPARTA_RESTART_FAIL_2'):
  print -2
  exit()

# find the latest file:

if time1 > time2: prefer1 = True
else: prefer2 = True

if not1 and not2: NUMBER=-1
elif not1: NUMBER=2
elif not2: NUMBER=1

if NUMBER == 0:
  if prefer1: NUMBER=1
  else: NUMBER=2

print NUMBER

