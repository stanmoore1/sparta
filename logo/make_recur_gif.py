"""Build the recurrence gif.

Forward only, and the last frame is dropped: the run returns exactly to its
initial state, so frame 3000 is pixel-identical to frame 0 and keeping both
would stutter.  No reversal anywhere - the emblem reassembles because the
dynamics bring it back, not because the frames are played backwards.
"""
import glob, os, sys
from PIL import Image

D   = '/home/user/sparta/logo/'
Q   = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NF  = int(sys.argv[2]) if len(sys.argv) > 2 else 40
NC  = int(sys.argv[3]) if len(sys.argv) > 3 else 16
DUR = int(sys.argv[4]) if len(sys.argv) > 4 else 110

fs = sorted(glob.glob(D + 'rec.*.ppm'))[:-1]        # drop the repeat of frame 0
fs = [fs[round(i * (len(fs) - 1) / (NF - 1))] for i in range(NF)]
ims = [Image.open(f).convert('RGB').resize((Q, Q), Image.LANCZOS) for f in fs]

probe = Image.new('RGB', (Q * 2, Q), 'white')
probe.paste(ims[0], (0, 0)); probe.paste(ims[len(ims)//2], (Q, 0))
pal = probe.quantize(colors=NC, method=Image.MEDIANCUT)
qf  = [im.quantize(palette=pal, dither=Image.NONE) for im in ims]

dur = [DUR] * len(qf); dur[0] = 1000                # hold the emblem it returns to
out = D + 'logo_recur.gif'
qf[0].save(out, save_all=True, append_images=qf[1:], duration=dur, loop=0, optimize=False)
print(f'{len(qf)} frames, {Q}x{Q} -> {os.path.getsize(out)/1e6:.2f} MB, cycle {sum(dur)/1000:.1f}s')
