"""Pull the lambda and SPARTA lettering out of logotext2d.surf as a surface.

That file is the inner disk with the letters cut out of it, so it holds the
r = 37 boundary circle as well as the letter loops.  For a body to fly a flow
around, only the letters are wanted, and each loop needs clockwise winding:
the winding each loop already has is kept, since the letters with counters
depend on the hole being wound against its parent.
"""
import math

txt = open('logotext2d.surf').read().split('\n')
i = txt.index('Points') + 2
pts = {}
while txt[i].strip():
    f = txt[i].split(); pts[int(f[0])] = (float(f[1]), float(f[2])); i += 1
j = txt.index('Lines') + 2
nxt = {}
while j < len(txt) and txt[j].strip():
    f = txt[j].split(); nxt[int(f[1])] = int(f[2]); j += 1

loops, seen = [], set()
for start in nxt:
    if start in seen: continue
    loop, p = [], start
    while p not in seen:
        seen.add(p); loop.append(p); p = nxt[p]
    if len(loop) > 2: loops.append(loop)

def radii(loop): return [math.hypot(*pts[p]) for p in loop]
def area(loop):
    a = 0.0
    for k in range(len(loop)):
        x1, y1 = pts[loop[k]]; x2, y2 = pts[loop[(k+1) % len(loop)]]
        a += x1*y2 - x2*y1
    return a / 2

# the boundary circle is the one loop that sits entirely at r = 37
letters = [L for L in loops if not (min(radii(L)) > 36.5 and max(radii(L)) < 37.5)]
dropped = len(loops) - len(letters)

out_pts, out_seg = [], []
# Do not re-wind the loops.  Four of the eleven are counters - the holes in
# A, P and R - and the file already orients each loop correctly relative to
# its parent.  Forcing them all one way makes a hole face the same way as the
# letter around it, which SPARTA rejects with "Cell type mis-match on self".
for L in letters:
    base = len(out_pts) + 1
    out_pts += [pts[p] for p in L]
    n = len(L)
    out_seg += [(base + k, base + (k + 1) % n) for k in range(n)]

with open('letters.surf', 'w') as f:
    f.write("# lambda and SPARTA lettering, lifted out of logotext2d.surf\n\n")
    f.write("%d points\n%d lines\n\nPoints\n\n" % (len(out_pts), len(out_seg)))
    for k, (x, y) in enumerate(out_pts): f.write("%d %.8e %.8e\n" % (k+1, x, y))
    f.write("\nLines\n\n")
    for k, (a, b) in enumerate(out_seg): f.write("%d %d %d\n" % (k+1, a, b))
print(f'{len(loops)} loops, dropped {dropped} boundary circle, '
      f'kept {len(letters)} letter loops -> {len(out_seg)} lines')
