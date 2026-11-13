#!/bin/bash
# runfft.sh: run examples/fft in.fft.2d on fin_serial (double), fin_single, mst_serial, mst_single; compare dump c_2 columns
source /home/user/sparta/.fix-review/tools/env.sh
O=$SP/v7/r1/fft
for e in $BIN/fin_serial $O/bin/fin_single $BIN/mst_serial $O/bin/mst_single; do
  n=$(basename $e); w=$O/run.$n; rm -rf $w; mkdir -p $w; cd $w; cp $FR/decks/r1/T2b/{in.fft.2d,ar.species,ar.vss} .
  timeout 300 $e -in in.fft.2d -echo none > log 2>&1; echo "$n rc=$? $(grep -m1 ERROR log)"
done
cd $O; python3 - <<'PY'
import os
def rd(n):
    p=f"run.{n}/dump.grid.2d"
    if not os.path.exists(p): return None
    L=open(p).read().split("ITEM: CELLS")[-1].strip().split("\n")[1:]
    return {int(l.split()[0]):[float(x) for x in l.split()[1:]] for l in L}
d=rd("fin_serial"); s=rd("fin_single"); md=rd("mst_serial"); ms=rd("mst_single")
def cmp(a,b,lab):
    if a is None or b is None: print(lab,"missing"); return
    mx=max(abs(x) for v in a.values() for x in v) or 1
    err=max(abs(x-y) for k in a for x,y in zip(a[k],b[k]))
    print(f"{lab}: ncell {len(a)}/{len(b)} max|diff| {err:.3e} max|val| {mx:.3e} rel {err/mx:.3e}")
cmp(d,s,"F double vs F single"); cmp(d,md,"F double vs M double"); cmp(md,ms,"M double vs M single")
PY
