#!/usr/bin/env python3
"""Per-step kernel launches, kernel time and host-device copies from two
   kp_count reports of the same input run for N and 2N steps:
   per step = (count_2N - count_N) / N, which cancels setup and post-run.

   usage: perstep.py report_N.txt report_2N.txt N
"""
import re,sys
def load(path):
    d = {}
    for line in open(path):
        if line[0] not in "KC": continue
        p = line.split(None,3)
        d[(p[0],p[3].strip())] = (int(p[1]),float(p[2]))
    return d
a,b = load(sys.argv[1]),load(sys.argv[2])
n = int(sys.argv[3])
def short(name):
    name = re.sub(r'std::__cxx11::basic_string<[^>]*>, ','',name)
    name = re.sub(r'Kokkos::RangePolicy<[^>]*>, ','',name)
    return name.replace('SPARTA_NS::','').replace('Kokkos::','')[:100]
rows = []
for k in b:
    c1,t1 = a.get(k,(0,0.0)); c2,t2 = b[k]
    rows.append((k[0],(c2-c1)/n,(t2-t1)/n,k[1]))
tot = 0.0
print("=== kernels per step: launches, ms (host wall time between the begin/end callbacks)")
for kind,per,tper,name in sorted(rows,key=lambda r:-r[2]):
    if kind != 'K' or (per < 0.01 and tper < 1e-4): continue
    tot += tper
    print(f"{per:7.2f}  {tper*1000:9.3f}  {short(name)}")
print(f"total kernel time per step {tot*1000:.2f} ms")
print("=== copies per step: count, MB")
for kind,per,tper,name in sorted(rows,key=lambda r:-r[2]):
    if kind != 'C' or per < 0.01: continue
    print(f"{per:7.2f}  {tper:9.3f}  {short(name)}")
