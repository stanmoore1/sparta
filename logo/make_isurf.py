"""Rasterise the emblem into the corner-value file read_isurf reads.

The solid is the red of the emblem - the field behind the lettering and the
field behind the swirls - and the gold is void.  So the lambda, the text, the
swirls and both rings are holes in a real slab of material, which erodes back
around them.  An earlier version made the gold the solid instead; those are
strands a cell or two wide and broke up almost immediately.

Corner values are 255 inside the solid and 0 outside, thresholded at 180.5.
"""
import numpy as np

N, LO, HI = 100, -60.0, 60.0            # grid cells, box bounds
g = np.linspace(LO, HI, N + 1)
X, Y = np.meshgrid(g, g)                 # x fastest, matching the file order
px, py = X.ravel(), Y.ravel()

def read_lines(path, scale=1.0):
    txt = open(path).read().split('\n')
    i = txt.index('Points') + 2
    pts = {}
    while txt[i].strip():
        f = txt[i].split(); pts[int(f[0])] = (float(f[1]) * scale, float(f[2]) * scale)
        i += 1
    j = txt.index('Lines') + 2
    seg = []
    while j < len(txt) and txt[j].strip():
        f = txt[j].split(); seg.append(pts[int(f[1])] + pts[int(f[2])])
        j += 1
    return np.array(seg)                  # (n,4): x1,y1,x2,y2

def inside(seg, px, py, chunk=400):
    """even-odd ray cast in +x, chunked to keep the arrays manageable"""
    x1, y1, x2, y2 = seg[:,0], seg[:,1], seg[:,2], seg[:,3]
    out = np.zeros(px.size, dtype=bool)
    for s in range(0, px.size, chunk):
        X, Y = px[s:s+chunk,None], py[s:s+chunk,None]
        straddles = (y1 > Y) != (y2 > Y)
        with np.errstate(divide='ignore', invalid='ignore'):
            xcross = x1 + (Y - y1) * (x2 - x1) / (y2 - y1)
        out[s:s+chunk] = (straddles & (X < xcross)).sum(axis=1) % 2 == 1
    return out

r = np.hypot(px, py)
inletters = inside(read_lines('logotext2d.surf'), px, py)   # disk minus letters
inswirls  = inside(read_lines('swirl2d.surf', 0.1), px, py)

redcentre = (r < 35.0) & inletters
redband   = (r > 37.0) & (r < 55.0) & ~inswirls
solid     = redcentre | redband
# read_isurf expects an 8 byte header of two int32 corner counts, then the data
with open('emblem.isurf', 'wb') as fh:
    fh.write(np.array([N + 1, N + 1], dtype=np.int32).tobytes())
    fh.write(np.where(solid, 255, 0).astype(np.uint8).tobytes())
print(f'{(N+1)**2} corners, {solid.sum()} solid ({100*solid.mean():.1f}%) -> emblem.isurf')
