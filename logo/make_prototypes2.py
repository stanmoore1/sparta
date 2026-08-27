import glob, os
from PIL import Image, ImageDraw, ImageFont
D='/home/user/sparta/logo/'
Q, NF, NC, DUR, LBL = 148, 18, 12, 150, 20
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 11)
P = [('collide (Kn low)','pair'), ('ghost (Kn high)','kn'), ('shock focusing','foc'),
     ('spinning','spin'), ('3d medallion','d3'), ('temperature','tmp')]

def load(prefix):
    fs = sorted(glob.glob(D+prefix+'.*.ppm'))
    fs = [fs[round(i*(len(fs)-1)/(NF-1))] for i in range(NF)]
    out=[]
    for f in fs:
        im = Image.open(f).convert('RGB'); im.thumbnail((Q,Q), Image.LANCZOS)
        c = Image.new('RGB',(Q,Q),'white'); c.paste(im, ((Q-im.width)//2,(Q-im.height)//2))
        out.append(c)
    return out

data=[(n,load(p)) for n,p in P]
W,H = Q*3, (Q+LBL)*2
frames=[]
for k in range(NF):
    c=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(c)
    for i,(name,ims) in enumerate(data):
        x,y = (i%3)*Q, (i//3)*(Q+LBL)
        d.rectangle([x,y,x+Q-1,y+LBL-1], fill=(28,28,32))
        d.text((x+5,y+4), name, font=font, fill=(238,232,220))
        c.paste(ims[k],(x,y+LBL))
    frames.append(c)
pal=frames[0].quantize(colors=NC, method=Image.MEDIANCUT)
qf=[f.quantize(palette=pal, dither=Image.NONE) for f in frames]
qf=qf+qf[-2:0:-1]
dur=[DUR]*len(qf); dur[0]=900
out=D+'logo_prototypes2.gif'
qf[0].save(out, save_all=True, append_images=qf[1:], duration=dur, loop=0, optimize=False)
print(f'{len(qf)} frames, {W}x{H} -> {os.path.getsize(out)/1e6:.2f} MB')
