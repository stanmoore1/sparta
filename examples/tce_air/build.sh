#!/bin/bash
# Build the CPU executables the package uses into bin/:
#   spa_release   this branch (required)
#   spa_compare   the comparison build with the earlier TCE treatments
#                 (react_modify vib_energy dof, elec_energy yes|no|micro);
#                 only with --compare.  It is the electronic-excitation line
#                 of sparta/sparta pull request #446 (with a merge of master),
#                 branch claude/sparta-electronic-excitation-r9cb99 of
#                 https://github.com/stanmoore1/sparta, commit 0ed27de71.
#   spa_precap    this branch before the cap correction of the reverse
#                 tables (commit e3f72907, the parent of d150e011): the
#                 forward-only cap of step 7; only with --precap.
# Needs an MPI C++ compiler (mpicxx) and make.
set -e
PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$PKG/../.." && pwd)
JOBS=${JOBS:-4}
mkdir -p "$PKG/bin"
( cd "$ROOT/src" && make -j "$JOBS" mpi )
cp "$ROOT/src/spa_mpi" "$PKG/bin/spa_release"
for opt in "$@"; do
if [ "$opt" = "--precap" ]; then
  PRE="$PKG/build/precap"
  if [ ! -d "$PRE" ]; then
    git -C "$ROOT" worktree add --detach "$PRE" e3f72907 \
      || git clone --depth 200 --branch claude/sparta-backward-rates-r9cb99 \
           https://github.com/stanmoore1/sparta "$PRE"
  fi
  ( cd "$PRE" && git checkout -q e3f72907 && cd src && make -j "$JOBS" mpi )
  cp "$PRE/src/spa_mpi" "$PKG/bin/spa_precap"
fi
done
if [ "$1" = "--compare" ] || [ "$2" = "--compare" ]; then
  CMP="$PKG/build/compare"
  if [ ! -d "$CMP" ]; then
    git clone --depth 50 --branch claude/sparta-electronic-excitation-r9cb99 \
        https://github.com/stanmoore1/sparta "$CMP"
  fi
  ( cd "$CMP" && git checkout -q 0ed27de71 && cd src && make -j "$JOBS" mpi )
  cp "$CMP/src/spa_mpi" "$PKG/bin/spa_compare"
fi
python3 -c "import numpy, scipy, matplotlib" || \
  echo "Python packages numpy, scipy and matplotlib are needed: pip install numpy scipy matplotlib"
ls -l "$PKG/bin"
