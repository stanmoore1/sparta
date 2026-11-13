#!/usr/bin/env python3
# compare thermo output of two run_examples.sh output dirs
import sys,glob,os,re
a,b=sys.argv[1],sys.argv[2]
def thermo(f):
    out=[];on=False
    for l in open(f,errors='replace'):
        if l.startswith('Step') or l.strip().startswith('Step '): on=True; out.append(l.split()); continue
        if l.startswith('Loop time'): on=False; continue
        if on and re.match(r'^\s*-?\d',l): out.append(l.split()[:1]+l.split()[2:])  # drop CPU col
    return out
for fa in sorted(glob.glob(a+'/*.log')):
    fb=os.path.join(b,os.path.basename(fa))
    name=os.path.basename(fa)[:-4]
    if not os.path.exists(fb): print(name,'MISSING'); continue
    ta,tb=thermo(fa),thermo(fb)
    if ta==tb: print(name,'IDENTICAL',len(ta)); continue
    n=min(len(ta),len(tb)); first=next((i for i in range(n) if ta[i]!=tb[i]),n)
    print(name,'DIFF at row',first,'of',len(ta),len(tb))
