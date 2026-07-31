#!/bin/bash
# this updates the list of fixes and computes supporting dump image
#
# For SPARTA these are the computes and fixes producing per-grid or per-surf
# data, since those are valid color sources for the grid/surf keywords of
# the dump image command.

if [ $# -lt 1 ]
then
    echo "usage: $0 <sparta-src-dir>"
    exit 1
fi

# resolve the source directory before cd'ing away from the caller's directory,
# so a relative path on the command line still means what the caller meant
SRCDIR="$(cd "$1" && pwd)" || exit 1

cd "$(dirname $0)" || exit 1

STYLELIST=""
for s in $(grep -l 'per_\(grid\|surf\)_flag = 1' "${SRCDIR}"/{compute,fix}_*.cpp \
               "${SRCDIR}"/*/{compute,fix}_*.cpp 2> /dev/null)
do \
    h="${s%.cpp}.h"
    [ -f "$h" ] && STYLELIST="${STYLELIST} ${h}"
done

touch image_style.table
mv image_style.table image_style.oldtable
touch image_style.tmp

for s in $STYLELIST
do \
    sed -n -e '/^Compute/s/ComputeStyle(\(.*\),.*/compute \1/p' \
        -e '/^Fix/s/FixStyle(\(.*\),.*/fix \1/p' $s >> image_style.tmp
done
sort -u image_style.tmp > image_style.table
rm image_style.tmp
cmp image_style.table image_style.oldtable > /dev/null || touch spartagui.qrc
