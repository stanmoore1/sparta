#!/usr/bin/env python3
"""Check the precision conversion of particle data between a double
precision and a reduced precision KOKKOS build.

Run in.conversion (free molecular flow, no collisions, so the motion is
deterministic) with each build, in its own directory:

    spa_double -k on -sf kk -in in.conversion      (in dir double/)
    spa_mixed  -k on -sf kk -in in.conversion      (in dir mixed/)

then compare the step 5 dumps:

    check_conversion.py double/dump.conv.5 mixed/dump.conv.5 mixed
    check_conversion.py double/dump.conv.5 single/dump.conv.5 single

Velocities do not change without collisions, so they must survive the
host/device round trip exactly.  Positions differ only by float round-off:
mixed (double positions, float velocities) by ~|v| dt nsteps 2^-24,
single by a few float ulps of the coordinate (the box sits at x = 0.1 m
so single precision resolution of x is ~7e-9 m).
"""

import sys

BOX = (1e-4, 1e-4, 1e-4)


def load(fn):
    data = {}
    atoms = open(fn).read().split("ITEM: ATOMS")[1].strip().split("\n")[1:]
    for line in atoms:
        w = line.split()
        data[int(w[0])] = [float(x) for x in w[2:]]
    return data


def main():
    ref, new, mode = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
    if ref.keys() != new.keys():
        print("FAIL: different particles")
        return 1
    dx = [0.0] * 3
    dv = [0.0] * 3
    for k in ref:
        for c in range(3):
            d = abs(ref[k][c] - new[k][c])
            if d > 0.5 * BOX[c]:          # wrapped across the periodic boundary
                d = abs(d - BOX[c])
            dx[c] = max(dx[c], d)
            dv[c] = max(dv[c], abs(ref[k][c + 3] - new[k][c + 3]))
    # tolerances: mixed ~ 600 m/s * 3.5e-8 s * 2^-24 = 1.3e-12 m
    #             single ~ a few ulps of 0.1 (7.5e-9) in x, of 1e-4 in y,z
    tol = {"mixed": (5e-12, 5e-12, 5e-12), "single": (1e-7, 1e-10, 1e-10)}[mode]
    ok = all(v == 0.0 for v in dv) and all(d <= t for d, t in zip(dx, tol))
    print("max |dx| %s m, max |dv| %s m/s: %s"
          % (["%.2e" % d for d in dx], ["%.2e" % d for d in dv], "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
