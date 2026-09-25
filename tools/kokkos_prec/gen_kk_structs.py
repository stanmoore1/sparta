#!/usr/bin/env python3
"""Generate src/KOKKOS/kokkos_structs.h from the host struct definitions.

The per-particle, per-grid-cell and per-surf structs of the host classes
(Particle::OnePart, Grid::ChildCell, Surf::Line, ...) are always double
precision.  The KOKKOS package stores a copy of them on the device in KK
precision.  This script parses the host struct definitions and writes, for
each precision mode, either a typedef of the host struct (when no field
changes type in that mode) or a KK struct whose double fields are retyped per
the "structs" section of precision_map.json, plus kk_convert() overloads used
by TransformView to convert between the two.

The script fails if a host struct has a double field with no mapping, so the
KK structs cannot silently drift from the host structs.

usage: gen_kk_structs.py [--check]
  --check   exit 1 if the generated file differs from the one on disk
"""

import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precision_map.json")
OUT = os.path.join(ROOT, "src", "KOKKOS", "kokkos_structs.h")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def parse_struct(header, name):
    """Return (attr, [(type, fieldname, dims)]) for 'struct [attr] name {...};'"""
    text = strip_comments(open(os.path.join(ROOT, header)).read())
    m = re.search(r"struct\s+(SPARTA_ALIGN\(\d+\)\s+)?" + name + r"\s*\{", text)
    if not m:
        sys.exit("gen_kk_structs: struct %s not found in %s" % (name, header))
    depth, i = 1, m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    body = text[m.end():i - 1]
    fields = []
    for decl in body.split(";"):
        decl = " ".join(decl.split())
        if not decl:
            continue
        dm = re.match(r"((?:const\s+)?[A-Za-z_][\w:]*)\s+(.*)$", decl)
        if not dm:
            sys.exit("gen_kk_structs: cannot parse '%s' in %s" % (decl, name))
        ctype, rest = dm.group(1), dm.group(2)
        for item in rest.split(","):
            item = item.strip()
            ptr = ""
            while item.startswith("*"):
                ptr += "*"
                item = item[1:].strip()
            im = re.match(r"(\w+)((?:\[[^\]]+\])*)$", item)
            if not im:
                sys.exit("gen_kk_structs: cannot parse '%s' in %s" % (item, name))
            fields.append((ctype + ptr, im.group(1), im.group(2)))
    return (m.group(1) or "").strip(), fields


def dims_of(dimstr):
    return re.findall(r"\[([^\]]+)\]", dimstr)


def main():
    cfg = json.load(open(MAP))
    modes = cfg["modes"]
    ctypes = cfg["precision_classes"]
    structs = []
    for name, info in cfg["structs"].items():
        attr, fields = parse_struct(info["header"], name)
        fmap = info["fields"]
        for ctype, fname, _ in fields:
            if ctype == "double" and fname not in fmap:
                sys.exit("gen_kk_structs: double field %s::%s has no mapping in "
                         "precision_map.json" % (name, fname))
            if fname in fmap and ctype != "double":
                sys.exit("gen_kk_structs: mapped field %s::%s is not double" % (name, fname))
        structs.append((name, info, attr, fields))

    out = []
    w = out.append
    w("/* -*- c++ -*- ----------------------------------------------------------")
    w("   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer")
    w("   http://sparta.github.io")
    w("   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov")
    w("   Sandia National Laboratories")
    w("")
    w("   Copyright (2014) Sandia Corporation.  Under the terms of Contract")
    w("   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains")
    w("   certain rights in this software.  This software is distributed under")
    w("   the GNU General Public License.")
    w("")
    w("   See the README file in the top-level SPARTA directory.")
    w("------------------------------------------------------------------------- */")
    w("")
    w("// GENERATED FILE, DO NOT EDIT: run tools/kokkos_prec/gen_kk_structs.py")
    w("//   after changing a host struct or tools/kokkos_prec/precision_map.json")
    w("")
    w("// KK precision copies of the host structs shared with the device, and")
    w("//   the element conversions TransformView uses on host/device sync.")
    w("//   In a mode where no field of a struct changes type, the KK struct is")
    w("//   the host struct itself.")
    w("")
    w("#ifndef SPARTA_KOKKOS_STRUCTS_H")
    w("#define SPARTA_KOKKOS_STRUCTS_H")
    w("")
    w("namespace SPARTA_NS {")
    first = True
    for mode, mtypes in modes.items():
        w("")
        w("#%s defined(%s)" % ("if" if first else "elif", mode))
        first = False
        for name, info, attr, fields in structs:
            host = "%s::%s" % (info["scope"], name)
            kk = info["kk_name"]
            changed = any(mtypes[c] != "double" for c in info["fields"].values())
            w("")
            if not changed:
                w("typedef %s %s;" % (host, kk))
                continue
            w("struct %s%s {" % (attr + " " if attr else "", kk))
            for ctype, fname, dims in fields:
                if fname in info["fields"]:
                    ctype = ctypes[info["fields"][fname]]
                w("  %s %s%s;" % (ctype, fname, dims))
            w("};")
            for dst, src, dname, sname in ((kk, host, "KK", "legacy"),
                                          (host, kk, "legacy", "KK")):
                w("")
                w("KOKKOS_INLINE_FUNCTION")
                w("void kk_convert(%s &dst, const %s &src)" % (dst, src))
                w("{")
                for ctype, fname, dims in fields:
                    d = dims_of(dims)
                    if fname in info["fields"]:
                        dtype = "double" if dst == host else ctypes[info["fields"][fname]]
                        if not d:
                            w("  dst.%s = static_cast<%s>(src.%s);" % (fname, dtype, fname))
                        else:
                            if len(d) != 1:
                                sys.exit("gen_kk_structs: only 1d array fields supported")
                            w("  for (int k = 0; k < %s; k++)" % d[0])
                            w("    dst.%s[k] = static_cast<%s>(src.%s[k]);" % (fname, dtype, fname))
                    elif d:
                        w("  for (int k = 0; k < %s; k++) dst.%s[k] = src.%s[k];"
                          % (d[0], fname, fname))
                    else:
                        w("  dst.%s = src.%s;" % (fname, fname))
                w("}")
    w("")
    w("#endif")
    w("")
    w("// in a double precision build the KK structs are the host structs")
    w("")
    w("#if defined(SPARTA_KOKKOS_DOUBLE_DOUBLE)")
    for name, info, attr, fields in structs:
        w("static_assert(std::is_same_v<%s,%s::%s>);" % (info["kk_name"], info["scope"], name))
    w("#endif")
    w("")
    w("}")
    w("")
    w("#endif")
    text = "\n".join(out) + "\n"

    if "--check" in sys.argv:
        old = open(OUT).read() if os.path.exists(OUT) else ""
        if old != text:
            print("gen_kk_structs: %s is out of date" % os.path.relpath(OUT, ROOT))
            return 1
        return 0
    open(OUT, "w").write(text)
    print("gen_kk_structs: wrote %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
