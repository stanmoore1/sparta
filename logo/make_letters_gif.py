"""Build the lettering bow shock gif.

Forward only - the shot fills an empty box, so running it backwards would
suck the gas back out - with a second held on the empty box before the gas
arrives.
"""
import glob, os, sys
from PIL import Image

D   = '/home/user/sparta/logo/'
Q   = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NF  = int(sys.argv[2]) if len(sys.argv) > 2 else 34
NC  = int(sys.argv[3]) if len(sys.argv) > 3 else 16
DUR = int(sys.argv[4]) if len(sys.argv) > 4 else 130

fs = sorted(glob.glob(D + 'let.*.ppm'))
fs = [fs[round(i * (len(fs) - 1) / (NF - 1))] for i in range(NF)]
ims = [Image.open(f).convert('RGB').resize((Q, Q), Image.LANCZOS) for f in fs]

# Build the palette from an empty frame and a full one together.  The last
# frame alone holds no white, so quantising on it turns the background gold.
probe = Image.new('RGB', (Q * 2, Q), 'white')
probe.paste(ims[0], (0, 0)); probe.paste(ims[-1], (Q, 0))
pal = probe.quantize(colors=NC, method=Image.MEDIANCUT)

qf  = [im.quantize(palette=pal, dither=Image.NONE) for im in ims]
dur = [DUR] * len(qf); dur[0] = 1000        # hold the empty box

out = D + 'logo_letters.gif'
qf[0].save(out, save_all=True, append_images=qf[1:], duration=dur, loop=0, optimize=False)
print(f'{len(qf)} frames, {Q}x{Q} -> {os.path.getsize(out)/1e6:.2f} MB, cycle {sum(dur)/1000:.1f}s')
