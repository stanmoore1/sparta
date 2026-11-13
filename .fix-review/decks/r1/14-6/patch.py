# patch.py <base> <lev2> <out> <value>: maxlevel int offset = first 4-byte-aligned-or-not position where base has int 1 and lev2 int 2
import struct, sys
a = open(sys.argv[1], "rb").read(); b = open(sys.argv[2], "rb").read()
cands = [i for i in range(min(len(a), len(b)) - 4)
         if struct.unpack_from("i", a, i)[0] == 1 and struct.unpack_from("i", b, i)[0] == 2]
# first divergence region between the files where 1 -> 2
off = cands[0]
print("offset", off, "ncands", len(cands))
out = bytearray(a); struct.pack_into("i", out, off, int(sys.argv[4])); open(sys.argv[3], "wb").write(out)
