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

With --promotions BUILD_LOG it instead checks a build log of a reduced
precision build compiled with -Wdouble-promotion: double arithmetic in
device code, where a KK precision value meets a double (typically a double
member such as a host scalar read in a kernel).  Most of it is on purpose
(masses, accumulators, attempt counts, mixed mode positions), so the count
per file and mode is compared with promotions_allowlist.json and the check
fails if a file has more than recorded.  After a reviewed change, record
the new counts with --update-allowlist.

usage: kk_prec_audit.py [-v] [files]
       kk_prec_audit.py --promotions BUILD_LOG --mode mixed|single [--update-allowlist]
"""

import json

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


ALLOWLIST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "promotions_allowlist.json")


def device_promotions(log):
    """unique device code -Wdouble-promotion sites per KOKKOS file in a
    build log, not counting lines justified with a KK_DOUBLE comment"""
    sites = {}
    pat = re.compile(r"^(\S+?):(\d+):(\d+): warning: .*\[-Wdouble-promotion\]")
    for line in open(log, errors="replace"):
        m = pat.match(line.strip())
        if not m:
            continue
        path = m.group(1)
        base = os.path.basename(path)
        full = os.path.join(K.KOKKOS_DIR, base)
        if not os.path.exists(full):
            continue
        sites.setdefault(base, set()).add((int(m.group(2)), int(m.group(3))))
    counts = {}
    for base, locs in sites.items():
        text = open(os.path.join(K.KOKKOS_DIR, base)).read()
        mask = K.code_mask(text)
        regions = K.device_regions(text, mask)
        lines = text.split("\n")
        starts = [0]
        for ln in lines:
            starts.append(starts[-1] + len(ln) + 1)
        n = 0
        for ln, col in locs:
            if ln > len(lines) or KEEP_MARK in lines[ln - 1]:
                continue
            if K.in_spans(regions, starts[ln - 1] + col - 1):
                n += 1
        if n:
            counts[base] = n
    return counts


def check_promotions(log, mode, update):
    counts = device_promotions(log)
    allow = json.load(open(ALLOWLIST)) if os.path.exists(ALLOWLIST) else {}
    if update:
        allow[mode] = dict(sorted(counts.items()))
        with open(ALLOWLIST, "w") as f:
            json.dump(allow, f, indent=2, sort_keys=True)
            f.write("\n")
        print("kk_prec_audit: recorded %d device double promotions in %d files (%s)"
              % (sum(counts.values()), len(counts), mode))
        return 0
    rec = allow.get(mode, {})
    bad = 0
    for base in sorted(counts):
        if counts[base] > rec.get(base, 0):
            bad += 1
            print("%s: %d double promotions in device code (%s allows %d)"
                  % (base, counts[base], os.path.basename(ALLOWLIST), rec.get(base, 0)))
    print("kk_prec_audit: %d device double promotions in %d files (%s), %d files over the allowlist"
          % (sum(counts.values()), len(counts), mode, bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--promotions", metavar="BUILD_LOG")
    ap.add_argument("--mode", choices=["mixed", "single"])
    ap.add_argument("--update-allowlist", action="store_true")
    args = ap.parse_args()
    if args.promotions:
        if not args.mode:
            ap.error("--promotions needs --mode")
        return check_promotions(args.promotions, args.mode, args.update_allowlist)
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
