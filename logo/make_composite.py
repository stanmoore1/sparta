"""Build the logo composite: the three animations one after another in a single
looping gif.  Each plays its whole arc out and then back again, so every
segment returns to the same crisp emblem it started from and the handovers -
and the loop itself - land on that shared image rather than cutting from
debris.  No captions: the emblem is the whole picture."""

import glob, os, sys
from PIL import Image

D    = '/home/user/sparta/logo/'
Q    = int(sys.argv[1]) if len(sys.argv) > 1 else 180   # frame size
NF   = int(sys.argv[2]) if len(sys.argv) > 2 else 16    # frames per outward leg
NC   = int(sys.argv[3]) if len(sys.argv) > 3 else 10
DUR  = int(sys.argv[4]) if len(sys.argv) > 4 else 140
HOLD = 900                                              # ms on the crisp emblem

SEGMENTS = ['pis', 'bs', 'exp']

frames, durs = [], []
for prefix in SEGMENTS:
    fs = sorted(glob.glob(D + prefix + '.*.ppm'))
    fs = [fs[round(i * (len(fs) - 1) / (NF - 1))] for i in range(NF)]
    ims = [Image.open(f).convert('RGB').resize((Q, Q), Image.LANCZOS) for f in fs]
    leg = ims + ims[-2:0:-1]          # out and back, neither endpoint repeated
    frames += leg
    durs   += [HOLD] + [DUR] * (len(leg) - 1)

pal = frames[0].quantize(colors=NC, method=Image.MEDIANCUT)
qf  = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]

out = D + 'logo_composite.gif'
qf[0].save(out, save_all=True, append_images=qf[1:], duration=durs, loop=0, optimize=False)
print(f'{len(qf)} frames, {Q}x{Q}, {NC} colors -> '
      f'{os.path.getsize(out)/1e6:.2f} MB, cycle {sum(durs)/1000:.1f}s')
