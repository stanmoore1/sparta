#!/bin/sh
# Movies of what a growing surface does to the gas standing in its way.
#
# The claim the deposition path makes is that a molecule the advancing front
# encloses is not simply deleted.  The fix looks for somewhere ahead of the
# front to put it back, across a processor boundary if it has to, and buries
# it -- booking its mass, momentum and energy into the film -- only when
# there is nowhere left.  These movies show both outcomes happening.
#
# Every domain here is closed on all four sides, so the particle count can
# only fall by burial.  The counter strip under each panel reads:
#
#   in gas      = current particle count
#   pushed back = value (8), molecules the front enclosed and put back
#   buried      = value (3), molecules it enclosed with nowhere to put them
#
# and "in gas" + "buried" must stay equal to the starting count, frame by
# frame.  That the two numbers track is the conservation statement; that
# "pushed back" is the larger of them is the feature.
#
# Cells are shaded by how much volume the film has left them -- solid is
# what no particle can occupy -- with the isosurface drawn on top in red.
#
# Needs ffmpeg.  Writes .mp4 into the directory given as $1 (default
# ./movies).
#
#   sh make_engulf_movies.sh [outdir] [path-to-spa_serial]

OUT=${1:-movies}
EXE=${2:-../../../src/spa_serial}
FONT=$(ls /usr/share/fonts/truetype/*/*Sans-Bold.ttf 2>/dev/null | head -1)
FPS=12
mkdir -p "$OUT"

# corner point fields, generated rather than committed since they exist only
# for these movies.  All three are graded, so the front moves at the speed it
# is asked for in every direction and the movies are about the particles
# rather than about the rate.

python3 - <<'PY'
import struct, math
N = 60; n = N+1; c = 30.0
def write(name, sd, taper=4.0):
    vals = []
    for j in range(n):
        for i in range(n):
            d = sd(float(i), float(j))
            v = int(round(255*min(1.0, max(0.0, 0.5 + d/taper))))
            if i in (0, n-1) or j in (0, n-1): v = 0
            vals.append(v)
    open(name, 'wb').write(struct.pack('<ii', n, n) + bytes(vals))

# a bar across the box, clear of the boundary, growing up and down into gas
write("slab.smooth", lambda x, y: min(y-14.0, 24.0-y))

# a cup: an annular shell with a mouth cut out of its +y side.  As the shell
# thickens the mouth closes, and the gas inside is sealed in with it
def cup(x, y):
    r = math.hypot(x-c, y-c)
    return min(min(r-9.0, 20.0-r), -min(6.0-abs(x-c), y-c))
write("cup.smooth", cup)

# a disc that grows until it has taken the whole box
write("disc.smooth", lambda x, y: 9.0 - math.hypot(x-c, y-c))
PY

# run one case: $1 tag, then -var pairs
run_case() {
  tag=$1; shift
  rm -rf "$OUT/$tag"; mkdir -p "$OUT/$tag"
  $EXE -in in.movie.engulf -var OUT "$OUT/$tag" "$@" \
       > "$OUT/$tag/run.log" 2>&1 || {
    echo "  run failed: $tag (see $OUT/$tag/run.log)"; return 1; }
}

# turn a run log into an ffmpeg sendcmd script that retitles the counter
# strip on every frame.  stats and the image dump share an interval, so the
# stats lines and the frames are one to one.
cmds() {
  python3 - "$1" "$2" "$FPS" "$3" <<'PY'
import sys
log, out, fps = sys.argv[1], sys.argv[2], float(sys.argv[3])
name = sys.argv[4]
rows, on = [], False
for line in open(log):
    if line.startswith("Step "): on = True; continue
    if not on: continue
    if not line[:1].isspace() and not line[:1].isdigit(): break
    f = line.split()
    if len(f) < 4 or not f[0].isdigit(): break
    rows.append([int(float(x)) for x in f[:4]])
with open(out, "w") as fh:
    for i, (step, np_, buried, back) in enumerate(rows):
        fh.write("%.4f %s reinit "
                 "'text=step %d    in gas %d    pushed back %d"
                 "    buried %d';\n"
                 % (i/fps, name, step, np_, back, buried))
PY
}

# ---------------------------------------------------------------- one panel
single() {
  tag=$1; title=$2; shift 2
  run_case "$tag" "$@" || return
  cmds "$OUT/$tag/run.log" "$OUT/$tag/cmds.txt" drawtext@c
  ffmpeg -y -loglevel error -framerate $FPS \
    -pattern_type glob -i "$OUT/$tag/f.*.ppm" \
    -filter_complex "\
[0:v]pad=iw:ih+58:0:58:white,\
drawtext=fontfile=$FONT:text='$title':x=(w-tw)/2:y=15:fontsize=30:fontcolor=black,\
pad=iw:ih+46:0:0:white,sendcmd=f='$OUT/$tag/cmds.txt',\
drawtext@c=fontfile=$FONT:text=' ':x=(w-tw)/2:y=h-32:fontsize=20:fontcolor=0x333333[v]" \
    -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT/$tag.mp4" \
    && echo "  $OUT/$tag.mp4"
}

# ---------------------------------------------------------------- two panels
pair() {
  out=$1; ltag=$2; ltitle=$3; rtag=$4; rtitle=$5; caption=$6
  cmds "$OUT/$ltag/run.log" "$OUT/$ltag/cmds.txt" drawtext@cl
  cmds "$OUT/$rtag/run.log" "$OUT/$rtag/cmds.txt" drawtext@cr
  ffmpeg -y -loglevel error \
    -framerate $FPS -pattern_type glob -i "$OUT/$ltag/f.*.ppm" \
    -framerate $FPS -pattern_type glob -i "$OUT/$rtag/f.*.ppm" \
    -filter_complex "\
[0:v]pad=iw:ih+58:0:58:white,\
drawtext=fontfile=$FONT:text='$ltitle':x=(w-tw)/2:y=15:fontsize=30:fontcolor=black,\
pad=iw:ih+46:0:0:white,sendcmd=f='$OUT/$ltag/cmds.txt',\
drawtext@cl=fontfile=$FONT:text=' ':x=(w-tw)/2:y=h-33:fontsize=20:fontcolor=0x333333[l];\
[1:v]pad=iw:ih+58:0:58:white,\
drawtext=fontfile=$FONT:text='$rtitle':x=(w-tw)/2:y=15:fontsize=30:fontcolor=black,\
pad=iw:ih+46:0:0:white,sendcmd=f='$OUT/$rtag/cmds.txt',\
drawtext@cr=fontfile=$FONT:text=' ':x=(w-tw)/2:y=h-33:fontsize=20:fontcolor=0x333333[r];\
[l][r]hstack=inputs=2,pad=iw:ih+62:0:62:white,\
drawtext=fontfile=$FONT:text='$caption':x=(w-tw)/2:y=17:fontsize=32:fontcolor=black[v]" \
    -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT/$out.mp4" \
    && echo "  $OUT/$out.mp4"
}

# 1. how often the surface is rebuilt is what decides burial.  Same rate,
#    same total growth, same number of steps: only Nevery differs, so the
#    right hand front jumps 100 times further between rebuilds and starts
#    enclosing molecules faster than they can be put back.
run_case front_n1  -var FIELD slab.smooth -var RATE 0.5 -var NEVERY 1 \
                   -var NSTEP 24000 -var NIMG 200
run_case front_n100 -var FIELD slab.smooth -var RATE 0.5 -var NEVERY 100 \
                   -var NSTEP 24000 -var NIMG 200
pair engulf-front front_n1 "rebuilt every step" \
                  front_n100 "rebuilt every 100 steps" \
                  "a flat front sweeping through gas"

# 2. a molecule with nowhere to go.  The mouth of the cup closes, and the gas
#    inside is sealed into a shrinking pocket -- no neighbour ahead of the
#    front to be pushed into, so the pocket is buried molecule by molecule.
single engulf-pocket "gas sealed into a closing cavity" \
       -var FIELD cup.smooth -var RATE 0.5 -var NEVERY 1 \
       -var NSTEP 21000 -var NIMG 175

# 3. what usually happens instead.  The disc grows until it has taken nearly
#    the whole box, and the gas is not consumed -- it is compressed ahead of
#    the front into what is left.  8000 molecules start, the film ends up
#    holding most of the domain, and twelve of them are buried: the rest were
#    pushed back into the shrinking gas, thousands of times over.  A
#    delete-on-engulf implementation would have destroyed all of them.
single engulf-fill "the gas is compressed, not consumed" \
       -var FIELD disc.smooth -var RATE 1.0 -var NEVERY 1 \
       -var NSTEP 32000 -var NIMG 200
