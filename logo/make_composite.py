"""Build the logo composite: the animations one after another in a single
looping gif.

Most segments play out and come back the way they went, so each returns to the
crisp emblem and the handovers land on it.  The recurrence segment is different
in kind: it is collisionless and specular, so it returns to the emblem by
itself and is played forward only - reversing it would be claiming a symmetry
it does not need.  It also needs far more frames than the others, since what
makes it read as motion rather than noise is how far a particle moves between
frames.
"""

import glob, os, sys
from PIL import Image

D    = '/home/user/sparta/logo/'
Q    = int(sys.argv[1]) if len(sys.argv) > 1 else 150
NF   = int(sys.argv[2]) if len(sys.argv) > 2 else 12    # frames per outward leg
NC   = int(sys.argv[3]) if len(sys.argv) > 3 else 10
DUR  = int(sys.argv[4]) if len(sys.argv) > 4 else 150
HOLD = 900

#            prefix  frames  ping-pong
SEGMENTS = [('slam',  NF,    True),
            ('bs',    NF,    True),
            ('exp',   NF,    True),
            ('rec',   64,    False)]

frames, durs = [], []
for prefix, nf, pingpong in SEGMENTS:
    fs = sorted(glob.glob(D + prefix + '.*.ppm'))
    if not pingpong:
        fs = fs[:-1]                      # last frame repeats the first exactly
    fs = [fs[round(i * (len(fs) - 1) / (nf - 1))] for i in range(nf)]
    ims = [Image.open(f).convert('RGB').resize((Q, Q), Image.LANCZOS) for f in fs]
    leg = ims + ims[-2:0:-1] if pingpong else ims
    frames += leg
    durs   += [HOLD] + [DUR] * (len(leg) - 1)

pal = frames[0].quantize(colors=NC, method=Image.MEDIANCUT)
qf  = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]

out = D + 'logo_composite.gif'
qf[0].save(out, save_all=True, append_images=qf[1:], duration=durs, loop=0, optimize=False)
print(f'{len(qf)} frames, {Q}x{Q}, {NC} colors -> '
      f'{os.path.getsize(out)/1e6:.2f} MB, cycle {sum(durs)/1000:.1f}s')
