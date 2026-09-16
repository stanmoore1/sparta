#!/usr/bin/env python3

# replicate.py: stamp a surface file N times to create N rigid bodies
#
# Syntax: replicate.py template dim N outsurf [options]
#   template = SPARTA surf file of ONE body, in 2d or 3d
#   dim = 2 or 3
#   N = # of copies to create
#   outsurf = surf file to write, with a type column = body ID 1 to N
# Options:
#   -box xlo xhi ylo yhi [zlo zhi]   region to place the bodies in
#                                   (default = 0 to 1 in each dim)
#   -lattice nx ny [nz]             place bodies on a lattice with this
#                                   many sites per dim (default = random)
#   -gap g                          min gap between bodies and to the box
#                                   edges, in body radii (default = 0.5)
#   -seed s                         random # seed (default = 12345)
#   -bodies file mass ixx iyy izz [vmax]
#                                   also write a fix rigid infile with
#                                   the same mass and moi for each body,
#                                   and a random velocity of magnitude
#                                   up to vmax (default = 0)
#
# The template body should be centered on the origin, since each copy is
#   translated to its site, and its lines or triangles must be ordered so
#   that the surf normals point outward, as fix rigid requires (in 2d,
#   clockwise around the body).  Each copy is written with surf type = its body
#   ID, so it can be read with "read_surf outsurf type" and driven by
#   "fix rigid ... type infile file" or "fix rigid ... type density rho".
#
# Example: 500 circles in a 0.1 x 0.1 box, each with random velocity <= 100
#   replicate.py circle.surf 2 500 circles.surf -box 0 0.1 0 0.1 \
#     -bodies circles.bodies 1.0e-7 5.12e-13 5.12e-13 5.12e-13 100.0

import sys, math, random

def error(msg):
    print("ERROR:", msg)
    sys.exit(1)

# ---------------------------------------------------------------------
# read the template surf file: points and lines or triangles

def read_template(filename, dim):
    pts = []
    elems = []
    section = None
    npoint = nelem = 0
    for line in open(filename):
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        if words[0] == "points" or (len(words) == 2 and words[1] == "points"):
            npoint = int(words[0]); continue
        if len(words) == 2 and words[1] in ("lines", "triangles"):
            nelem = int(words[0]); continue
        if words[0] == "Points":
            section = "points"; continue
        if words[0] in ("Lines", "Triangles"):
            section = "elems"; continue
        if section == "points":
            pts.append([float(w) for w in words[1:1+dim]])
        elif section == "elems":
            elems.append([int(w) for w in words[1:1+dim]])
    if len(pts) != npoint or len(elems) != nelem:
        error("template file %s has inconsistent counts" % filename)
    return pts, elems

# ---------------------------------------------------------------------
# main

args = sys.argv[1:]
if len(args) < 4:
    print(__doc__ or "Syntax: replicate.py template dim N outsurf [options]")
    sys.exit(1)

template = args[0]
dim = int(args[1])
n = int(args[2])
outsurf = args[3]
if dim not in (2, 3): error("dim must be 2 or 3")
if n < 1: error("N must be positive")

box = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
lattice = None
gap = 0.5
seed = 12345
bodies = None

iarg = 4
while iarg < len(args):
    if args[iarg] == "-box":
        box[:2*dim] = [float(w) for w in args[iarg+1:iarg+1+2*dim]]
        iarg += 1 + 2*dim
    elif args[iarg] == "-lattice":
        lattice = [int(w) for w in args[iarg+1:iarg+1+dim]]
        iarg += 1 + dim
    elif args[iarg] == "-gap":
        gap = float(args[iarg+1]); iarg += 2
    elif args[iarg] == "-seed":
        seed = int(args[iarg+1]); iarg += 2
    elif args[iarg] == "-bodies":
        vals = args[iarg+1:iarg+7]
        if len(vals) < 5: error("-bodies needs file mass ixx iyy izz [vmax]")
        bodies = {"file": vals[0], "mass": vals[1],
                  "moi": vals[2:5], "vmax": 0.0}
        iarg += 6
        if iarg < len(args) and not args[iarg].startswith("-"):
            bodies["vmax"] = float(args[iarg]); iarg += 1
    else: error("unknown option %s" % args[iarg])

random.seed(seed)
pts, elems = read_template(template, dim)
radius = max(math.sqrt(sum(c*c for c in p)) for p in pts)
spacing = 2.0 * radius * (1.0 + gap)

# body sites: lattice or random rejection sampling

sites = []
lo = [box[2*d] + spacing/2 for d in range(dim)]
hi = [box[2*d+1] - spacing/2 for d in range(dim)]
for d in range(dim):
    if hi[d] < lo[d]: error("box is too small for one body")

if lattice:
    if len(lattice) != dim: error("-lattice needs %d values" % dim)
    nsite = 1
    for count in lattice: nsite *= count
    if nsite < n: error("lattice has %d sites, fewer than N" % nsite)
    index = [0] * dim
    while len(sites) < n:
        site = []
        for d in range(dim):
            frac = 0.5 if lattice[d] == 1 else index[d] / (lattice[d] - 1.0)
            site.append(lo[d] + frac * (hi[d] - lo[d]))
        sites.append(site)
        for d in range(dim):
            index[d] += 1
            if index[d] < lattice[d]: break
            index[d] = 0
else:
    attempts = 0
    while len(sites) < n:
        attempts += 1
        if attempts > 1000 * n:
            error("could not place %d bodies without overlap, "
                  "enlarge the box or reduce -gap" % n)
        site = [random.uniform(lo[d], hi[d]) for d in range(dim)]
        ok = True
        for other in sites:
            dsq = sum((site[d] - other[d])**2 for d in range(dim))
            if dsq < spacing*spacing:
                ok = False; break
        if ok: sites.append(site)

# write the replicated surf file, type column = body ID

npoint = len(pts)
nelem = len(elems)
with open(outsurf, "w") as f:
    f.write("%d copies of %s, surf type = body ID\n\n" % (n, template))
    f.write("%d points\n" % (n*npoint))
    if dim == 2: f.write("%d lines\n\n" % (n*nelem))
    else: f.write("%d triangles\n\n" % (n*nelem))
    f.write("Points\n\n")
    ip = 0
    for ibody in range(n):
        for p in pts:
            ip += 1
            coords = " ".join("%.17g" % (p[d] + sites[ibody][d])
                              for d in range(dim))
            f.write("%d %s\n" % (ip, coords))
    if dim == 2: f.write("\nLines\n\n")
    else: f.write("\nTriangles\n\n")
    ie = 0
    for ibody in range(n):
        offset = ibody * npoint
        for e in elems:
            ie += 1
            f.write("%d %d %s\n" % (ie, ibody+1,
                                    " ".join(str(v+offset) for v in e)))

print("wrote %d bodies, %d points, %d elements to %s"
      % (n, n*npoint, n*nelem, outsurf))

# write the fix rigid infile

if bodies:
    with open(bodies["file"], "w") as f:
        f.write("# ID mtotal xcm ycm zcm ixx iyy izz ixy ixz iyz "
                "vxcm vycm vzcm lx ly lz\n")
        for ibody in range(n):
            com = sites[ibody] + [0.0] * (3 - dim)
            v = [0.0, 0.0, 0.0]
            if bodies["vmax"] > 0.0:
                if dim == 2:
                    theta = random.uniform(0.0, 2.0*math.pi)
                    v = [math.cos(theta), math.sin(theta), 0.0]
                else:
                    while True:
                        v = [random.uniform(-1, 1) for d in range(3)]
                        vsq = sum(c*c for c in v)
                        if 0.0 < vsq <= 1.0: break
                    vlen = math.sqrt(vsq)
                    v = [c/vlen for c in v]
                v = [c * bodies["vmax"] for c in v]
            f.write("%d %s %.17g %.17g %.17g %s %s %s 0 0 0 "
                    "%.17g %.17g %.17g 0 0 0\n"
                    % (ibody+1, bodies["mass"], com[0], com[1], com[2],
                       bodies["moi"][0], bodies["moi"][1], bodies["moi"][2],
                       v[0], v[1], v[2]))
    print("wrote %d body params to %s" % (n, bodies["file"]))
