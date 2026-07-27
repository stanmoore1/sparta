#!/bin/bash
# Build a SPARTA serial binary with a given flavor of flags and stash it under
# bench/opt/bin/ so old binaries stay around for A/B comparison.
#
# usage: build.sh FLAVOR TAG
#   FLAVOR = base    -O3 -std=c++11                (the shipped Makefile.serial flags)
#            prof    -O3 -std=c++11 -g -pg         (gprof)
#            debugsym -O3 -std=c++11 -g            (callgrind, keeps -O3)
#            native  -O3 -std=c++11 -march=native  (flag experiment)
#            fast    -O3 -std=c++11 -march=native -ffast-math -funroll-loops
#
# The flags for "base" are frozen: every code-change measurement uses them, so
# speedups are attributable to code rather than to compiler options.
set -eu

FLAVOR="${1:?need flavor}"
TAG="${2:?need tag}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT/bench/opt/bin"
mkdir -p "$BIN"

case "$FLAVOR" in
  base)     CCF="-O3 -std=c++11";                    LNK="-O3" ;;
  prof)     CCF="-O3 -std=c++11 -g -pg";             LNK="-O3 -pg" ;;
  debugsym) CCF="-O3 -std=c++11 -g -fno-omit-frame-pointer"; LNK="-O3 -g" ;;
  native)   CCF="-O3 -std=c++11 -march=native";      LNK="-O3 -march=native" ;;
  fast)     CCF="-O3 -std=c++11 -march=native -ffast-math -funroll-loops"
            LNK="-O3 -march=native -ffast-math" ;;
  *) echo "unknown flavor $FLAVOR" >&2; exit 1 ;;
esac

OBJ="Obj_$FLAVOR"
cd "$ROOT/src"

# a per-flavor Makefile so the object dirs never collide
MK="MAKE/Makefile.$FLAVOR"
sed -e "s|^CCFLAGS =.*|CCFLAGS =\t$CCF|" \
    -e "s|^LINKFLAGS =.*|LINKFLAGS =\t$LNK|" \
    MAKE/Makefile.serial > "$MK"

make "$FLAVOR" -j4 > "$ROOT/bench/opt/logs/build.$TAG.log" 2>&1 || {
  echo "BUILD FAILED, tail of log:" >&2
  tail -30 "$ROOT/bench/opt/logs/build.$TAG.log" >&2
  exit 1
}

cp "$ROOT/src/spa_$FLAVOR" "$BIN/spa_$TAG"
echo "built $BIN/spa_$TAG   [$CCF]"
