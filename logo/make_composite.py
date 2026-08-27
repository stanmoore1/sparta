"""Build the logo composite: the three animations one after another in a single
looping gif.  Each plays its whole arc out and then back again, so a segment
returns to whatever it opened on and the handovers - and the loop itself -
land there rather than cutting from mid-debris.  No captions: the emblem is
the whole picture.

Wind and expansion both open on the crisp emblem, so those two hand over on
the same image.  The bow shock opens on an empty box instead, since it fills
one, and it holds there before the gas arrives."""

import glob, os, sys
from PIL import Image

D    = '/home/user/sparta/logo/'
Q    = int(sys.argv[1]) if len(sys.argv) > 1 else 180   # frame size
NF   = int(sys.argv[2]) if len(sys.argv) > 2 else 16    # frames per outward leg
NC   = int(sys.argv[3]) if len(sys.argv) > 3 else 10
DUR  = int(sys.argv[4]) if len(sys.argv) > 4 else 140
HOLD = 900                                              # ms on the crisp emblem

SEGMENTS = ['bs', 'exp', 'let']

# The bow shock opens on an empty box, so it gets a second of the crisp
# particle emblem in front of it: the same transition the standalone gif uses,
# borrowed from the wind run, which is the same geometry drawn as particles.
OPENER = {'let': 'bs.00000.ppm'}

frames, durs = [], []
for prefix in SEGMENTS:
    fs = sorted(glob.glob(D + prefix + '.*.ppm'))
    fs = [fs[round(i * (len(fs) - 1) / (NF - 1))] for i in range(NF)]
    ims = [Image.open(f).convert('RGB').resize((Q, Q), Image.LANCZOS) for f in fs]
    leg = ims + ims[-2:0:-1]          # out and back, neither endpoint repeated
    lead = []
    if prefix in OPENER:
        lead = [Image.open(D + OPENER[prefix]).convert('RGB').resize((Q, Q), Image.LANCZOS)]
    frames += lead + leg
    durs   += [HOLD] * (len(lead) + 1) + [DUR] * (len(leg) - 1)

# Palette from the crisp emblem and a gas-filled bow shock frame together,
# with black, white and a mid grey forced in: the bow shock's lettering is a
# thin black outline, and left to the medians it lands on the nearest gold.
probe = Image.new('RGB', (Q * 2, Q), 'white')
probe.paste(frames[0], (0, 0)); probe.paste(frames[-1], (Q, 0))
adaptive = probe.quantize(colors=NC - 3, method=Image.MEDIANCUT).getpalette()[:3 * (NC - 3)]
entries = [0, 0, 0, 255, 255, 255, 128, 128, 128] + adaptive
pal = Image.new('P', (1, 1))
pal.putpalette(entries + [0] * (768 - len(entries)))

qf  = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]

out = D + 'logo_composite.gif'
qf[0].save(out, save_all=True, append_images=qf[1:], duration=durs, loop=0, optimize=False)
print(f'{len(qf)} frames, {Q}x{Q}, {NC} colors -> '
      f'{os.path.getsize(out)/1e6:.2f} MB, cycle {sum(durs)/1000:.1f}s')
