#!/usr/bin/env python3
"""Lint the KOKKOS package for code that defeats reduced precision builds.

Fails (exit 1) if, in device code (KOKKOS_INLINE_FUNCTION/KOKKOS_FUNCTION
bodies and signatures, KOKKOS_LAMBDA bodies) of src/KOKKOS:

  - a 'double' appears on a line without a '// KK_DOUBLE' comment saying
    why double precision is needed there
  - a floating point literal has an 'f' suffix (use static_cast<KK_FLOAT>)
  - kk_prec_convert.py would still change the file, i.e. there is a bare
    C math call, an unwrapped floating point literal in an arithmetic
    expression, or a legacy SPARTA_FLOAT/F_FLOAT/t_float_* type

Run it after adding or changing KOKKOS code, and fix what it reports by
running kk_prec_convert.py --apply or by hand.

usage: kk_prec_audit.py [-v] [files]
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kkprec as K  # noqa: E402
from kk_prec_convert import Converter, KEEP_MARK  # noqa: E402


def audit_file(conv, fname, verbose):
    text = open(fname).read()
    base = os.path.basename(fname)
    if base in conv.skip_files:
        return []
    problems = []
    mask = K.code_mask(text)
    regions = K.device_regions(text, mask)
    for m in re.finditer(r"\bdouble\b", text):
        i = m.start()
        if not mask[i] or not K.in_spans(regions, i):
            continue
        eol = text.find("\n", i)
        if KEEP_MARK in text[i:eol if eol >= 0 else len(text)]:
            continue
        problems.append((K.line_of(text, i), "double in device code without // %s"
                         % KEEP_MARK))
    for m in K.FLOAT_LITERAL.finditer(text):
        if m.group(2) in ("f", "F") and mask[m.start()] and K.in_spans(regions, m.start()):
            problems.append((K.line_of(text, m.start()), "float literal with f suffix"))
    conv.review = []
    new = conv.convert(fname, text, ["types", "double", "math", "literals", "casts", "rng"])
    if new != text:
        a, b = text.splitlines(), new.splitlines()
        for ln, (x, y) in enumerate(zip(a, b), 1):
            if x != y:
                problems.append((ln, "not converted: %s" % y.strip()[:90]))
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    conv = Converter(K.load_map())
    conv.build_param_table(K.kokkos_files())
    total = 0
    for f in K.kokkos_files(args.files):
        probs = audit_file(conv, f, args.verbose)
        total += len(probs)
        rel = os.path.relpath(f, K.ROOT)
        if probs and args.verbose:
            for line, why in probs:
                print("%s:%d: %s" % (rel, line, why))
        elif probs:
            print("%s: %d problems" % (rel, len(probs)))
    print("kk_prec_audit: %d problems" % total)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
