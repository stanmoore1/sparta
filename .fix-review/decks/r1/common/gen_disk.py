# gen_disk.py <n> <out>: 2d read_isurf binary file, (n+1)x(n+1) corner values, smooth disk centered, radius n/3
import struct, sys, math
n = int(sys.argv[1]); out = sys.argv[2]; m = n + 1; c = n / 2.0; r = n / 3.0
vals = []
for j in range(m):
    for i in range(m):
        d = math.hypot(i - c, j - c)
        vals.append(max(0, min(255, int(255 * (r + 2 - d) / 4.0))))
with open(out, "wb") as f:
    f.write(struct.pack("ii", m, m)); f.write(bytes(vals))
