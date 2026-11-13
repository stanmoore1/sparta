#!/bin/bash
# idempotent: (re)build every binary the verification needs. F = commit in $FR/F.txt, M = origin/master.
source /home/user/sparta/.fix-review/tools/env.sh
which mpicxx > /dev/null || { DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openmpi-bin libopenmpi-dev > /dev/null 2>&1; }
F=$(cat $FR/F.txt); M=$(git -C /home/user/sparta rev-parse origin/master)
$FR/tools/build.sh fin $F && $FR/tools/build.sh mst $M && $FR/tools/kbuild.sh kfin $F && $FR/tools/kbuild.sh kmst $M
echo "ENSURE DONE rc=$?"
