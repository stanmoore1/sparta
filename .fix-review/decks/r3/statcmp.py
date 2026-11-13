# usage: statcmp.py logA logB -> prints SAME or first differing stats row (CPU column dropped)
import sys
def rows(p):
    out=[]; on=False; hdr=None
    for l in open(p, errors="replace"):
        w=l.split()
        if w and w[0]=="Step": on=True; hdr=w; out.append(" ".join(w)); continue
        if l.startswith("Loop time"): on=False; continue
        if on and w:
            if hdr and "CPU" in hdr and len(w)==len(hdr): w=[x for i,x in enumerate(w) if hdr[i]!="CPU"]
            out.append(" ".join(w))
    return out
a,b=rows(sys.argv[1]),rows(sys.argv[2])
if a==b: print("SAME", len(a), "rows")
else:
    for x,y in zip(a,b):
        if x!=y: print("DIFF\n ",x,"\n ",y); break
    else: print("DIFF length", len(a), len(b))
