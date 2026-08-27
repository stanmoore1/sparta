"""Build the logo composite: the three animations one after another in a single
looping gif.  Each plays its whole arc - no trimming - sampled down to a common
frame count, with a hold on the intact emblem as each one begins."""

import glob, os, sys
from PIL import Image, ImageDraw, ImageFont

D    = '/home/user/sparta/logo/'
Q    = int(sys.argv[1]) if len(sys.argv) > 1 else 200   # frame size
NF   = int(sys.argv[2]) if len(sys.argv) > 2 else 26    # frames per segment
NC   = int(sys.argv[3]) if len(sys.argv) > 3 else 10
DUR  = int(sys.argv[4]) if len(sys.argv) > 4 else 140
HOLD = 900                                              # ms on each intact emblem
LBL  = 24
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)

SEGMENTS = [('piston shock', 'pis'), ('hypersonic wind', 'bs'), ('free expansion', 'exp')]

frames, durs = [], []
for name, prefix in SEGMENTS:
    fs = sorted(glob.glob(D + prefix + '.*.ppm'))
    fs = [fs[round(i * (len(fs) - 1) / (NF - 1))] for i in range(NF)]
    for k, path in enumerate(fs):
        c = Image.new('RGB', (Q, Q + LBL), 'white')
        d = ImageDraw.Draw(c)
        d.rectangle([0, 0, Q - 1, LBL - 1], fill=(28, 28, 32))
        d.text((8, 5), name, font=font, fill=(238, 232, 220))
        c.paste(Image.open(path).convert('RGB').resize((Q, Q), Image.LANCZOS), (0, LBL))
        frames.append(c)
        durs.append(HOLD if k == 0 else DUR)

pal = frames[0].quantize(colors=NC, method=Image.MEDIANCUT)
qf  = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]

out = D + 'logo_composite.gif'
qf[0].save(out, save_all=True, append_images=qf[1:], duration=durs, loop=0, optimize=False)
print(f'{len(qf)} frames, {Q}x{Q+LBL}, {NC} colors -> '
      f'{os.path.getsize(out)/1e6:.2f} MB, cycle {sum(durs)/1000:.1f}s')
