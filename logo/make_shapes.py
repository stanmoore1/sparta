"""Generate the two shapes that are not derived from the emblem itself.

dish.surf  a concave reflector: the emblem's ring with a gap cut in its right
           side, so a stream entering through the gap reflects off the inner
           wall and converges.  Wound clockwise, which is what puts the normals
           outward and leaves the flow region outside the body.
coin.surf  a plain cylinder for the 3d variant, in SPARTA surf format rather
           than STL - read_surf reads only the former, region mesh reads both.
"""
import math

def dish(path='dish.surf', n=90, ro=55.0, ri=40.0, gap=70.0):
    a0, a1 = math.radians(gap / 2), math.radians(360 - gap / 2)
    arc = lambda r: [(r * math.cos(a0 + (a1 - a0) * i / n),
                      r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]
    pts = (arc(ro) + arc(ri)[::-1])[::-1]          # clockwise
    with open(path, 'w') as f:
        f.write("# concave reflector: an annulus with a gap facing +x\n\n")
        f.write("%d points\n%d lines\n\nPoints\n\n" % (len(pts), len(pts)))
        for i, (x, y) in enumerate(pts): f.write("%d %.8e %.8e\n" % (i + 1, x, y))
        f.write("\nLines\n\n")
        for i in range(len(pts)): f.write("%d %d %d\n" % (i + 1, i + 1, (i + 1) % len(pts) + 1))
    print(f'{path}: {len(pts)} lines')

def coin(path='coin.surf', n=48, R=20.0, H=4.0):
    pts, tris = [], []
    def P(x, y, z):
        pts.append((x, y, z)); return len(pts)
    top = [P(R * math.cos(2*math.pi*i/n), R * math.sin(2*math.pi*i/n),  H) for i in range(n)]
    bot = [P(R * math.cos(2*math.pi*i/n), R * math.sin(2*math.pi*i/n), -H) for i in range(n)]
    ct, cb = P(0, 0, H), P(0, 0, -H)
    for i in range(n):
        j = (i + 1) % n
        tris += [(ct, top[i], top[j]), (cb, bot[j], bot[i]),
                 (bot[i], bot[j], top[j]), (bot[i], top[j], top[i])]
    with open(path, 'w') as f:
        f.write("# a coin: cylinder radius %g, half-thickness %g, axis z\n\n" % (R, H))
        f.write("%d points\n%d triangles\n\nPoints\n\n" % (len(pts), len(tris)))
        for i, p in enumerate(pts): f.write("%d %.8e %.8e %.8e\n" % ((i + 1,) + p))
        f.write("\nTriangles\n\n")
        for i, t in enumerate(tris): f.write("%d %d %d %d\n" % ((i + 1,) + t))
    print(f'{path}: {len(tris)} triangles')

dish(); coin()
