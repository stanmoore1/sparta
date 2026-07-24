#!/usr/bin/env python3
"""Generate a synthetic scalar field for exercising the VTK viewer's filters.

A radial bump in a cube: iso-surfaces are spheres, so a wrong result is
obvious by eye, and the second array gives the field calculator two operands.
"""
import sys

N = 21
out = sys.argv[1] if len(sys.argv) > 1 else "field.vtk"
pts, vals = [], []
for k in range(N):
    for j in range(N):
        for i in range(N):
            x, y, z = i / (N - 1), j / (N - 1), k / (N - 1)
            pts.append((x, y, z))
            r2 = (x - .5) ** 2 + (y - .5) ** 2 + (z - .5) ** 2
            vals.append(1.0 / (1.0 + 40 * r2))

with open(out, "w") as f:
    f.write("# vtk DataFile Version 3.0\nsynthetic scalar field\nASCII\n")
    f.write(f"DATASET STRUCTURED_GRID\nDIMENSIONS {N} {N} {N}\n")
    f.write(f"POINTS {len(pts)} float\n")
    for p in pts:
        f.write("%g %g %g\n" % p)
    f.write(f"\nPOINT_DATA {len(vals)}\nSCALARS density float 1\nLOOKUP_TABLE default\n")
    for v in vals:
        f.write("%g\n" % v)
    f.write("\nSCALARS temperature float 1\nLOOKUP_TABLE default\n")
    for v in vals:
        f.write("%g\n" % (300.0 + 200.0 * v))
print(f"{out}: {N}^3 = {len(pts)} points, arrays density + temperature")
