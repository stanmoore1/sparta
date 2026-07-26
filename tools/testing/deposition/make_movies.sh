#!/bin/sh
# Side-by-side movies of the two length/time rate conversions, response
# normal against response volume, for the five cases in the README table.
#
# Each panel shows the grid cells and the implicit surface.  What the movies
# show that the numbers do not is the SHAPE: a rate that depends on which way
# the surface faces does not just grow the body at the wrong speed, it grows
# it into the wrong shape.  The diamond on a binary field turns into an
# octagon under response normal, because its 45 degree faces outrun its tips.
#
# Needs ffmpeg.  Writes .mp4 into the directory given as $1 (default ./movies).
#
#   sh make_movies.sh [outdir] [path-to-spa_serial]

OUT=${1:-movies}
EXE=${2:-../../../src/spa_serial}
FONT=$(ls /usr/share/fonts/truetype/*/*Sans-Bold.ttf 2>/dev/null | head -1)
mkdir -p "$OUT"

# the corner point fields the table is measured on; diamond.* are committed,
# the other three are generated here since they are only used for the movies
python3 - "$OUT" <<'PY'
import struct, math, sys
N = 60; n = N+1; c = 30.0
def write(name, sd, taper=None):
    out = struct.pack('<ii', n, n); vals = []
    for j in range(n):
        for i in range(n):
            d = sd(i-c, j-c)
            v = (255 if d > 0 else 0) if taper is None else \
                int(round(255*min(1.0, max(0.0, 0.5 + d/taper))))
            if i in (0, n-1) or j in (0, n-1): v = 0
            vals.append(v)
    open(name, 'wb').write(out + bytes(vals))
write("square.binary", lambda x, y: 12.0 - max(abs(x), abs(y)))
write("blob.binary",   lambda x, y: 15.0 - math.hypot(x, y))
write("blob.smooth",   lambda x, y: 15.0 - math.hypot(x, y), 4.0)
PY

for CASE in diamond.smooth diamond.binary square.binary blob.smooth blob.binary; do
  for R in normal volume; do
    rm -rf "$OUT/${CASE}_$R"; mkdir -p "$OUT/${CASE}_$R"
    $EXE -in in.movie -var FIELD $CASE -var RESP $R -var OUT "$OUT/${CASE}_$R" \
         -var NSTEP 12000 -var NIMG 120 > "$OUT/${CASE}_$R/run.log" 2>&1
  done
  ffmpeg -y -loglevel error \
    -framerate 12 -pattern_type glob -i "$OUT/${CASE}_normal/f.*.ppm" \
    -framerate 12 -pattern_type glob -i "$OUT/${CASE}_volume/f.*.ppm" \
    -filter_complex "\
[0:v]pad=iw:ih+56:0:56:white,drawtext=fontfile=$FONT:text='response normal':x=(w-tw)/2:y=14:fontsize=30:fontcolor=0x111111[l];\
[1:v]pad=iw:ih+56:0:56:white,drawtext=fontfile=$FONT:text='response volume':x=(w-tw)/2:y=14:fontsize=30:fontcolor=0x111111[r];\
[l][r]hstack=inputs=2,pad=iw:ih+62:0:62:white,\
drawtext=fontfile=$FONT:text='$CASE':x=(w-tw)/2:y=16:fontsize=34:fontcolor=black[v]" \
    -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT/$CASE.mp4"
  echo "  $OUT/$CASE.mp4"
done
