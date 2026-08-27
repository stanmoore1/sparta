import glob, os, sys
from PIL import Image, ImageDraw, ImageFont

D    = '/home/user/sparta/logo/'
Q    = int(sys.argv[1]) if len(sys.argv)>1 else 200   # panel size
NF   = int(sys.argv[2]) if len(sys.argv)>2 else 24    # frames per leg
NC   = int(sys.argv[3]) if len(sys.argv)>3 else 12
DUR  = int(sys.argv[4]) if len(sys.argv)>4 else 140
LBL  = 24
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)

def load(prefix, frac=1.0):
    # frac trims each run to a comparable arc: the piston dissolves its emblem
    # a quarter of the way through 16000 steps, so using all of it would leave
    # that panel finished while the others had barely started
    fs = sorted(glob.glob(D+prefix+'.*.ppm'))
    last = max(1, int(round((len(fs)-1)*frac)))
    fs = [fs[round(i*last/(NF-1))] for i in range(NF)]
    return [Image.open(f).convert('RGB').resize((Q,Q), Image.LANCZOS) for f in fs]

panels = [('piston shock',    load('pis', 0.32)),
          ('hypersonic wind', load('bs',  1.0)),
          ('free expansion',  load('exp', 1.0))]

W, H = Q*3, Q+LBL
frames=[]
for k in range(NF):
    c = Image.new('RGB',(W,H),'white')
    d = ImageDraw.Draw(c)
    for i,(name,ims) in enumerate(panels):
        x = i*Q
        d.rectangle([x,0,x+Q-1,LBL-1], fill=(28,28,32))
        d.text((x+8,5), name, font=font, fill=(238,232,220))
        c.paste(ims[k],(x,LBL))
    frames.append(c)

pal = frames[0].quantize(colors=NC, method=Image.MEDIANCUT)
qf  = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
qf  = qf + qf[-2:0:-1]                    # ping-pong: ends back on the intact emblem
dur = [DUR]*len(qf); dur[0] = 1000        # hold the logo before it goes
out = D+'logo_composite.gif'
qf[0].save(out, save_all=True, append_images=qf[1:], duration=dur, loop=0, optimize=False)
print(f'{len(qf)} frames, {W}x{H}, {NC} colors -> {os.path.getsize(out)/1e6:.2f} MB, cycle {sum(dur)/1000:.1f}s')
