#!/bin/bash
# Build SPARTA and SPARTA-GUI on macOS with Homebrew dependencies.
#
# Usage:  ./tools/sparta-gui/build-macos.sh
# (run from the top level of the SPARTA source tree)
#
# The script builds:
#   1. a serial SPARTA shared library with PNG/JPEG support
#      in build-sparta-macos/
#   2. the SPARTA-GUI app bundle (plugin mode) in build-sparta-gui-macos/
# and prints how to launch the result.

set -e

if [ "$(uname)" != "Darwin" ]; then
    echo "This script is intended for macOS. Use CMake directly on other platforms." >&2
    exit 1
fi

if [ ! -f src/sparta.h ] || [ ! -d tools/sparta-gui ]; then
    echo "Please run this script from the top level of the SPARTA source tree." >&2
    exit 1
fi

# check Homebrew dependencies

if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required: https://brew.sh" >&2
    exit 1
fi

for pkg in cmake qt libpng jpeg; do
    if ! brew list --versions "$pkg" >/dev/null 2>&1; then
        echo "Installing missing Homebrew package: $pkg"
        brew install "$pkg"
    fi
done

QT_PREFIX=$(brew --prefix qt)
NPROC=$(sysctl -n hw.ncpu)

# 1. serial SPARTA shared library

echo "==> Building SPARTA shared library (serial, PNG/JPEG enabled)"
cmake -S cmake -B build-sparta-macos \
      -D BUILD_SHARED_LIBS=ON \
      -D BUILD_MPI=OFF \
      -D BUILD_PNG=ON \
      -D BUILD_JPEG=ON \
      -D CMAKE_BUILD_TYPE=Release
cmake --build build-sparta-macos -j "$NPROC"

SPARTA_LIB=$(find "$PWD/build-sparta-macos" -name 'libsparta*.dylib' | head -1)
if [ -z "$SPARTA_LIB" ]; then
    echo "Could not find the built SPARTA shared library" >&2
    exit 1
fi
echo "==> SPARTA library: $SPARTA_LIB"

# 2. SPARTA-GUI app bundle (plugin mode)

echo "==> Building SPARTA-GUI"
cmake -S tools/sparta-gui -B build-sparta-gui-macos \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_PREFIX_PATH="$QT_PREFIX"
cmake --build build-sparta-gui-macos -j "$NPROC"

APP=$(find "$PWD/build-sparta-gui-macos" -maxdepth 2 -name '*.app' | head -1)

echo
echo "Done."
if [ -n "$APP" ]; then
    echo "Launch with:   open \"$APP\""
else
    echo "Launch with:   ./build-sparta-gui-macos/sparta-gui"
fi
echo
echo "On first start, set the path to the SPARTA shared library in the"
echo "Preferences dialog:"
echo "    $SPARTA_LIB"
