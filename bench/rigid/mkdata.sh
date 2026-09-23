#!/bin/sh
# Generate the surf and body files these decks read.
#
#   sh mkdata.sh [/path/to/sparta]
#
# writes circle.surf, nc{1000,2000,4000}.surf, nc{1000,2000,4000}.bodies
# and air.species/air.vss into the current directory.  The .surf files are
# 2-8 MB, so they are generated rather than stored.

set -e
SPARTA=${1:-$(cd "$(dirname "$0")/../.." && pwd)}

# the template body: a 32 segment circle of radius 3.2 mm at the origin,
# wound so the surf normals point outward, which fix rigid requires

python3 - <<'PY'
import math
n, r = 32, 3.2e-3
with open("circle.surf", "w") as f:
    f.write("circle %d segs R=%g\n\n" % (n, r))
    f.write("%d points\n%d lines\n\nPoints\n\n" % (n, n))
    for i in range(n):
        f.write("%d %.17g %.17g\n"
                % (i + 1, r * math.cos(2 * math.pi * i / n),
                   r * math.sin(2 * math.pi * i / n)))
    f.write("\nLines\n\n")
    for i in range(n):
        f.write("%d %d %d\n" % (i + 1, (i + 1) % n + 1, i + 1))
PY

# N copies on a jittered random placement, each with its own surf type,
# plus a matching per-body parameter file.  the box edge is
# sqrt(N * 3.2e-4) so the body number density is the same at every N,
# which is what makes the weak series a weak series

for N in 1000 2000 4000; do
  L=$(python3 -c "import math;print(repr(math.sqrt($N*3.2e-4)))")
  python3 "$SPARTA/tools/rigid/replicate.py" circle.surf 2 $N nc$N.surf \
      -box 0 $L 0 $L -gap 1.0 -seed 777 \
      -bodies nc$N.bodies 1.0e-7 5.12e-13 5.12e-13 5.12e-13 100.0

  # one speed, 200 m/s, random direction: replicate.py draws speeds up to
  # its vmax, and a single speed makes the contact model's peak overlap
  # the same for every pair, 5.7e-4 m, inside the 8.0e-4 m push cutoff

  python3 - "$N" <<'PY'
import math, random, sys
random.seed(20260917)
N = sys.argv[1]
SPEED = 200.0
out = []
for line in open("nc%s.bodies" % N):
    if line.startswith("#"):
        out.append(line); continue
    t = line.split()
    th = random.uniform(0, 2 * math.pi)
    t[11] = "%.17g" % (SPEED * math.cos(th))
    t[12] = "%.17g" % (SPEED * math.sin(th))
    out.append(" ".join(t) + "\n")
open("nc%s.bodies" % N, "w").writelines(out)
PY
done

cp "$SPARTA/examples/rigid/air.species" "$SPARTA/examples/rigid/air.vss" .
echo "wrote circle.surf nc{1000,2000,4000}.{surf,bodies} air.species air.vss"
