"""Encode the composite as mp4, from the 500x500 render frames rather than from
the gif - the gif is 150px and 10 colours, which is a size compromise an mp4
does not have to make.

Timing is carried by repeating frames at a constant rate rather than by
per-frame durations, since that is how video works.
"""
import glob, os, subprocess, sys
import imageio_ffmpeg
from PIL import Image

D    = '/home/user/sparta/logo/'
TMP  = '/tmp/claude-0/-home-user-sparta/72874b4a-41d7-593e-9ac0-4a207f4143ed/scratchpad/mp4/'
Q    = int(sys.argv[1]) if len(sys.argv) > 1 else 500
NF   = int(sys.argv[2]) if len(sys.argv) > 2 else 24
FPS  = 25
DUR  = 4                                    # frames held per animation step
HOLD = 25                                   # frames held on the crisp emblem

SEGMENTS = [('slam', NF, True), ('bs', NF, True), ('exp', NF, True), ('rec', 90, False)]

os.makedirs(TMP, exist_ok=True)
for f in glob.glob(TMP + '*.png'): os.remove(f)

n = 0
for prefix, nf, pingpong in SEGMENTS:
    fs = sorted(glob.glob(D + prefix + '.*.ppm'))
    if not pingpong: fs = fs[:-1]           # last frame repeats the first exactly
    fs = [fs[round(i * (len(fs) - 1) / (nf - 1))] for i in range(nf)]
    ims = [Image.open(f).convert('RGB').resize((Q, Q), Image.LANCZOS) for f in fs]
    leg = ims + ims[-2:0:-1] if pingpong else ims
    for k, im in enumerate(leg):
        for _ in range(HOLD if k == 0 else DUR):
            im.save(f'{TMP}f{n:06d}.png'); n += 1

exe = imageio_ffmpeg.get_ffmpeg_exe()
out = D + 'logo_composite.mp4'
subprocess.run([exe, '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', TMP + 'f%06d.png',
                # Particle noise is expensive to encode.  crf 18 put this at
                # 41 MB; 28 is visually indistinguishable here at a third of it.
                # The lever that actually matters is the number of DISTINCT
                # frames - the repeats that carry the timing cost almost
                # nothing, being identical.
                '-c:v', 'libx264', '-preset', 'slow', '-crf', '28',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], check=True)
print(f'{n} frames at {FPS} fps = {n/FPS:.1f}s, {Q}x{Q} -> {os.path.getsize(out)/1e6:.2f} MB')
