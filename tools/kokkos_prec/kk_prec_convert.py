#!/usr/bin/env python3
"""Bulk conversion of the SPARTA KOKKOS package to precision-generic code.

Rewrites src/KOKKOS/*.{h,cpp} following the LAMMPS KOKKOS conventions:

  types     rename the legacy SPARTA_FLOAT/F_FLOAT types and the t_float_*
            view typedefs to the KK_FLOAT/KK_ACC_FLOAT families, and the host
            structs used on the device (Particle::OnePart, ...) to their KK
            versions (OnePartKK, ...)
  double    in device code, retype each 'double' declaration as KK_FLOAT,
            KK_POS_FLOAT, KK_ACC_FLOAT, or keep it double, chosen by the
            declared identifier (see precision_map.json)
  math      in device code, bare C math calls sqrt(), exp(), ... become
            Kokkos::sqrt(), Kokkos::exp(), ... which have float overloads
  literals  in device code, a floating point literal (or MathConst constant,
            or a #define'd floating constant) that takes part in an
            arithmetic expression is wrapped as static_cast<T>(literal) so
            it does not promote a float expression to double; T is the
            precision of the enclosing statement
  casts     in device code, C casts (double) become (T)
  rng       in device code, random draws (generated in double) are narrowed
            to the precision T of their statement, except for particle IDs
            and 1-drand(), which could round to 0
  post      per-file literal substitutions from precision_map.json, for the
            few cases the heuristics above get wrong

Device code is the signature and body of every KOKKOS_INLINE_FUNCTION,
KOKKOS_FUNCTION and KOKKOS_LAMBDA/SPARTA_LAMBDA; host code is never changed
except by the 'types' and 'post' passes.

The conversion is idempotent.  By default it is a dry run that prints a
unified diff; --apply writes the files.  It also writes a report of the
constructs that need review by hand (sizeof(double), MPI_DOUBLE on Kokkos
data, memcpy of structs, floating point atomics, ...) to
tools/kokkos_prec/review.md.

usage: kk_prec_convert.py [--apply] [--pass NAME ...] [--report FILE] [files]
"""

import argparse
import difflib
import fnmatch
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kkprec as K  # noqa: E402

CLASS_TYPE = {"float": "KK_FLOAT", "pos": "KK_POS_FLOAT", "acc": "KK_ACC_FLOAT",
              "double": "double"}
TYPE_CLASS = {v: k for k, v in CLASS_TYPE.items()}
RANK = {"float": 0, "acc": 1, "pos": 2, "double": 3}
DECL_TOKENS = ("double", "float", "KK_FLOAT", "KK_POS_FLOAT", "KK_ACC_FLOAT", "const",
               "constexpr", "define", "auto")

# a 'double' on a line with this marker comment is deliberate and kept

KEEP_MARK = "KK_DOUBLE"


class Converter:
    def __init__(self, cfg):
        self.cfg = cfg
        conv = cfg["conversion"]
        self.pos_names = set(conv["pos_identifiers"])
        self.acc_names = set(conv["acc_identifiers"])
        self.double_names = set(conv["keep_double_identifiers"])
        self.file_defaults = conv.get("file_default_class", {})
        self.file_overrides = conv.get("file_identifier_class", {})
        self.post_subs = conv.get("post_substitutions", {})
        self.type_renames = conv["type_renames"]
        self.file_type_renames = conv.get("file_type_renames", {})
        self.struct_renames = conv["struct_renames"]
        self.math_consts = set(conv["float_constants"])
        self.skip_files = set(conv.get("skip_files", []))
        self.review = []
        self.param_table = {}
        self.view_class = {}
        self.int_names = set()

    # ------------------------------------------------------------------
    # classification of an identifier to a precision class

    def classify(self, name, fname):
        base = os.path.basename(fname)
        ov = self.file_overrides_for(base)
        if name in ov:
            return ov[name]
        if name in self.double_names:
            return "double"
        if name in self.pos_names:
            return "pos"
        if name in self.acc_names:
            return "acc"
        for pattern, cls in self.file_defaults.items():
            if fnmatch.fnmatch(base, pattern):
                return cls
        return "float"

    def file_overrides_for(self, base):
        """identifier classes for a file; keys of file_identifier_class are
        file names or glob patterns, later matches win"""
        out = {}
        for pattern, ov in self.file_overrides.items():
            if fnmatch.fnmatch(base, pattern):
                out.update(ov)
        return out

    # ------------------------------------------------------------------

    def convert(self, fname, text, passes):
        base = os.path.basename(fname)
        if base in self.skip_files:
            return text
        edits = []
        mask = K.code_mask(text)
        regions = K.device_regions(text, mask)
        self.fname = fname

        if "types" in passes:
            self.pass_types(text, mask, regions, edits)
        decl_class = {}
        if "double" in passes:
            self.pass_double(text, mask, regions, edits, decl_class)
        if "math" in passes:
            self.pass_math(text, mask, regions, edits)
        consts = self.float_defines(text, fname)
        self.int_names = self.int_declarations(text, mask)
        self.view_class = self.view_classes(text, fname)
        if "literals" in passes:
            self.pass_literals(text, mask, regions, edits, decl_class, consts)
        if "casts" in passes:
            self.pass_casts(text, mask, regions, edits, decl_class)
        if "rng" in passes:
            self.pass_rng(text, mask, regions, edits, decl_class)
        self.report(text, mask, regions)

        new = apply_edits(text, edits, fname)
        if "post" in passes:
            for old, rep in self.post_subs.get(base, []):
                if old in new:
                    new = new.replace(old, rep)
                elif rep not in new:
                    self.review.append((base, 0, "post-substitution not applied",
                                        old.strip()[:80]))
        return new

    # ------------------------------------------------------------------
    # types: whole file

    def pass_types(self, text, mask, regions, edits):
        base = os.path.basename(self.fname)
        renames = []
        for pattern, subs in self.file_type_renames.items():
            if fnmatch.fnmatch(base, pattern):
                renames += subs
        renames += self.type_renames
        done = set()
        for old, new in renames:
            for m in re.finditer(old, text):
                if mask[m.start()] and m.start() not in done:
                    done.add(m.start())
                    edits.append((m.start(), m.end(), m.expand(new)))
        # host structs used on the device: only in device code
        for old, new in self.struct_renames:
            for m in re.finditer(old, text):
                if mask[m.start()] and K.in_spans(regions, m.start()):
                    edits.append((m.start(), m.end(), new))

    # ------------------------------------------------------------------
    # double declarations in device code

    def build_param_table(self, files):
        """precision class of each double parameter of each device function
        definition, by (function name, number of parameters, index), so a
        declaration with unnamed or differently named parameters gets the
        same types as its definition"""
        self.param_table = {}
        for f in files:
            if os.path.basename(f) in self.skip_files:
                continue
            text = open(f).read()
            mask = K.code_mask(text)
            self.fname = f
            for sig in signatures(text, mask, K.device_regions(text, mask)):
                if not sig["body"]:
                    continue
                classes = []
                for ps, pe in sig["params"]:
                    ptxt = text[ps:pe]
                    if not re.search(r"\bdouble\b", ptxt):
                        classes.append(None)
                        continue
                    ids = [x for x in K.IDENT.findall(ptxt)
                           if x not in K.CPP_KEYWORDS and x not in ("restrict",)]
                    if not ids:
                        classes.append(None)
                        continue
                    cls = self.classify(ids[-1], f)
                    if ids[-1] in self.double_names:
                        cls = "double"
                    classes.append(cls)
                key = (sig["name"], len(sig["params"]))
                self.param_table.setdefault(key, classes)

    def param_class(self, sigs, i):
        """class for a double at position i if it is in a function parameter
        list with a known definition, else None"""
        for sig in sigs:
            if sig["popen"] < i < sig["pclose"]:
                key = (sig["name"], len(sig["params"]))
                if key not in self.param_table:
                    return None
                for k, (ps, pe) in enumerate(sig["params"]):
                    if ps <= i < pe:
                        return self.param_table[key][k]
        return None

    def pass_double(self, text, mask, regions, edits, decl_class):
        sigs = signatures(text, mask, regions)
        # declarations converted by an earlier run
        for m in re.finditer(r"\b(KK_FLOAT|KK_POS_FLOAT|KK_ACC_FLOAT)\b", text):
            if not mask[m.start()] or not K.in_spans(regions, m.start()):
                continue
            nxt = next_code(text, mask, m.end())
            if nxt in ")>(":
                continue
            _, names = declared_names(text, mask, m.end())
            for nm in names:
                decl_class.setdefault(nm, TYPE_CLASS[m.group(1)])
        for m in re.finditer(r"\bdouble\b", text):
            i = m.start()
            if not mask[i] or not K.in_spans(regions, i):
                continue
            if KEEP_MARK in text[i:text.find("\n", i)]:
                continue
            pcls = self.param_class(sigs, i)
            # skip casts/template args: (double), <double>, sizeof(double)
            prev = prev_code(text, mask, i)
            nxt_i = next_code_idx(text, mask, m.end())
            nxt = text[nxt_i] if nxt_i < len(text) else ""
            if pcls is None and (nxt in ")>" or (prev == "(" and nxt == ")")):
                continue
            name, names = declared_names(text, mask, m.end())
            if pcls is not None:
                if pcls != "double":
                    edits.append((i, m.end(), CLASS_TYPE[pcls]))
                for nm in names:
                    decl_class.setdefault(nm, pcls)
                continue
            if name is None:
                continue
            classes = set()
            for nm in names:
                c = self.classify(nm, self.fname)
                classes.add(c)
                decl_class.setdefault(nm, c)
            cls = self.classify(name, self.fname)
            if len(classes) > 1:
                split = self.split_declaration(text, mask, i, m.end())
                if split:
                    edits.append(split)
                    for nm in names:
                        decl_class[nm] = self.classify(nm, self.fname)
                    if "double" in classes:
                        eol = text.find("\n", split[1])
                        eol = len(text) if eol < 0 else eol
                        edits.append((eol, eol, "  // %s: precision_map.json "
                                      "keep_double_identifiers" % KEEP_MARK))
                    continue
                self.review.append((os.path.basename(self.fname), K.line_of(text, i),
                                    "multi-declarator with mixed precision classes",
                                    ", ".join(names)))
            # reduction value argument of a functor: keep double
            reason = "precision_map.json keep_double_identifiers"
            if self.is_reduction_arg(text, mask, i, m.end()):
                cls = "double"
                reason = "reduction value"
                for nm in names:
                    decl_class[nm] = "double"
            if cls != "double":
                edits.append((i, m.end(), CLASS_TYPE[cls]))
            else:
                eol = text.find("\n", i)
                eol = len(text) if eol < 0 else eol
                edits.append((eol, eol, "  // %s: %s" % (KEEP_MARK, reason)))

    def split_declaration(self, text, mask, i, j):
        """'double a, *b, c[3];' whose declarators have different precision
        classes: return an edit that splits it into one declaration per
        declarator, or None if this is not a simple declaration statement"""
        start = i
        before = text[max(0, i - 12):i]
        cm = re.search(r"\bconst\s+$", before)
        prefix = ""
        if cm:
            start = i - len(cm.group(0))
            prefix = "const "
        if prev_code(text, mask, start) not in (";", "{", "}", ""):
            return None
        decls = []
        k = j
        cur = j
        n = len(text)
        while k < n:
            if mask[k]:
                c = text[k]
                if c in "([{":
                    k = K.match_close(text, mask, k)
                    continue
                if c == ",":
                    decls.append(text[cur:k])
                    cur = k + 1
                elif c == ";":
                    decls.append(text[cur:k])
                    break
                elif c == ")":
                    return None
            k += 1
        if k >= n:
            return None
        parts = []
        for d in decls:
            nm = K.IDENT.search(d.replace("*", " ").replace("&", " "))
            if not nm:
                return None
            cls = self.classify(nm.group(0), self.fname)
            parts.append("%s%s %s;" % (prefix, CLASS_TYPE[cls], d.strip()))
        return (start, k + 1, " ".join(parts))

    def is_reduction_arg(self, text, mask, i, j):
        """'double &name' as the last, non-const parameter of an operator()
        or KOKKOS_LAMBDA with more than one parameter"""
        k = next_code_idx(text, mask, j)
        if k >= len(text) or text[k] != "&":
            return False
        # const?
        before = text[max(0, i - 20):i]
        if re.search(r"\bconst\s*$", before):
            return False
        # find enclosing '(' of the parameter list
        depth = 0
        p = i - 1
        while p >= 0:
            if mask[p]:
                if text[p] == ")":
                    depth += 1
                elif text[p] == "(":
                    if depth == 0:
                        break
                    depth -= 1
                elif text[p] in ";{}" and depth == 0:
                    return False
            p -= 1
        if p < 0:
            return False
        head = text[max(0, p - 40):p]
        if not re.search(r"(operator\s*\(\s*\)|_LAMBDA|join|init)\s*$", head):
            return False
        close = K.match_close(text, mask, p)
        params = text[p + 1:close - 1]
        if "," not in params:
            return False
        # must be the last parameter
        rest = text[j:close - 1]
        return "," not in rest

    # ------------------------------------------------------------------

    def pass_math(self, text, mask, regions, edits):
        pat = re.compile(r"(?<![\w:.>])(" + "|".join(K.MATH_FUNCS) + r")\s*\(")
        for m in pat.finditer(text):
            if mask[m.start()] and K.in_spans(regions, m.start()):
                edits.append((m.start(), m.start(), "Kokkos::"))

    # ------------------------------------------------------------------

    def float_defines(self, text, fname):
        """names of #define'd floating point constants in this file and in
        its companion header"""
        names = set()
        srcs = [text]
        if fname.endswith(".cpp"):
            h = fname[:-4] + ".h"
            if os.path.exists(h):
                srcs.append(open(h).read())
        for s in srcs:
            for m in re.finditer(r"^\s*#define\s+(\w+)\s+([-+]?[\d.]+[eE]?[-+]?\d*)\s*(//.*)?$",
                                 s, re.M):
                if "." in m.group(2) or "e" in m.group(2).lower():
                    names.add(m.group(1))
        return names

    def int_declarations(self, text, mask):
        names = set()
        for m in re.finditer(r"\b(int|bigint|cellint|surfint)\b", text):
            if not mask[m.start()]:
                continue
            nxt = next_code(text, mask, m.end())
            if nxt in ")>,&*":
                continue
            _, nms = declared_names(text, mask, m.end())
            names.update(nms)
        return names

    def view_classes(self, text, fname):
        """precision class of each Kokkos view declared with a KK view type
        in this file or its companion header"""
        srcs = [text]
        if fname.endswith(".cpp"):
            h = fname[:-4] + ".h"
            if os.path.exists(h):
                srcs.append(open(h).read())
        base = os.path.basename(fname)
        renames = []
        for pattern, subs in self.file_type_renames.items():
            if fnmatch.fnmatch(base, pattern) or fnmatch.fnmatch(os.path.basename(
                    fname[:-4] + ".h" if fname.endswith(".cpp") else fname), pattern):
                renames += subs
        renames += self.type_renames
        out = {}
        for src in srcs:
            # view types as they will be named after the 'types' pass
            for old, new in renames:
                src = re.sub(old, new, src)
            for m in re.finditer(r"\bt_kk(float|pos|acc)_\w+\s+(\w+)\s*[;,=]", src):
                out.setdefault(m.group(2), m.group(1))
        return out

    def statement_class(self, text, mask, i, decl_class):
        """precision class of the statement containing position i"""
        # statement start: previous ';', '{', '}' in code
        s = i - 1
        while s >= 0 and not (mask[s] and text[s] in ";{}"):
            s -= 1
        e = i
        while e < len(text) and not (mask[e] and text[e] in ";{}"):
            e += 1
        stmt = "".join(c if mask[s + 1 + k] else " "
                       for k, c in enumerate(text[s + 1:e]))
        st = stmt.strip()
        # an expression converted to an integer stays double: truncation
        #   of a float can differ by one, e.g. int(n*drand()) can reach n
        if re.search(r"static_cast<\s*(int|bigint|cellint|surfint)\s*>|"
                     r"\(\s*(int|bigint|cellint|surfint)\s*\)", st):
            return "double"
        if re.match(r"(?:const\s+)?(int|bigint|cellint|surfint)\b", st):
            return "double"
        for am in re.finditer(r"(\w+)\s*(?:\[[^\]]*\]\s*)*[-+*/]?=(?!=)", st):
            if am.group(1) in self.int_names:
                return "double"
        # declaration with an explicit type
        m = re.match(r"(?:const\s+)?(double|KK_FLOAT|KK_POS_FLOAT|KK_ACC_FLOAT|float)\b"
                     r"\s*[*&]?\s*(\w+)", st)
        if m:
            if m.group(1) == "double":
                return self.lookup(m.group(2), decl_class)
            if m.group(1) == "float":
                return "float"
            return TYPE_CLASS[m.group(1)]
        m = re.match(r"return\b", st)
        # assignment: class of the last identifier of the lhs
        am = re.match(r"([^=<>!]*?)(?<![=<>!+\-*/])([-+*/]?=)(?!=)", st)
        if am and not re.match(r"(if|while|for|return)\b", st):
            lhs = re.sub(r"\[[^\]]*\]", "", am.group(1))
            lhs = re.sub(r"\([^()]*\)", "", lhs)
            ids = [x for x in K.IDENT.findall(lhs) if x not in K.CPP_KEYWORDS]
            if ids:
                return self.lookup(ids[-1], decl_class)
        # otherwise the widest class of any identifier in the statement
        best = "float"
        for x in K.IDENT.findall(stmt):
            if x in K.CPP_KEYWORDS:
                continue
            c = decl_class.get(x) or self.view_class.get(x)
            ov = self.file_overrides_for(os.path.basename(self.fname))
            if c is None and (x in self.pos_names or x in self.double_names
                              or x in self.acc_names or x in ov):
                c = self.classify(x, self.fname)
            if c and RANK[c] > RANK[best]:
                best = c
        return best

    def lookup(self, name, decl_class):
        return (decl_class.get(name) or self.view_class.get(name) or
                self.classify(name, self.fname))

    def pass_literals(self, text, mask, regions, edits, decl_class, consts):
        cands = []
        for m in K.FLOAT_LITERAL.finditer(text):
            cands.append((m.start(), m.end(), m.group(0)))
        idpat = re.compile(r"(?<![\w.])(?:MathConst::)?(" +
                           "|".join(sorted(self.math_consts | consts) or ["NO_CONST_"]) +
                           r")\b")
        for m in idpat.finditer(text):
            cands.append((m.start(), m.end(), m.group(0)))
        # literals already wrapped by an earlier run: re-evaluate the class,
        #   so a changed rule or map entry also corrects converted code
        wpat = re.compile(r"static_cast<(KK_FLOAT|KK_POS_FLOAT|KK_ACC_FLOAT)>\(\s*(" +
                          K.FLOAT_LITERAL.pattern + r"|(?:MathConst::)?\w+)\s*\)")
        for m in wpat.finditer(text):
            s, e = m.start(), m.end()
            inner = m.group(2)
            if not mask[s] or not K.in_spans(regions, s):
                continue
            if not (K.FLOAT_LITERAL.fullmatch(inner) or
                    inner.split("::")[-1] in (self.math_consts | consts)):
                continue
            cls = self.statement_class(text, mask, s, decl_class)
            if cls == "double":
                edits.append((s, e, inner))
            elif CLASS_TYPE[cls] != m.group(1):
                edits.append((s, e, "static_cast<%s>(%s)" % (CLASS_TYPE[cls], inner)))
        for s, e, lit in cands:
            if not mask[s] or not K.in_spans(regions, s):
                continue
            if lit[0].isalpha() and prev_code_token(text, mask, s) in DECL_TOKENS:
                continue
            if self.skip_literal(text, mask, s, e):
                continue
            cls = self.statement_class(text, mask, s, decl_class)
            if cls == "double":
                continue
            if lit[-1] in "fF" and lit[0].isdigit():
                lit = lit[:-1]
            edits.append((s, e, "static_cast<%s>(%s)" % (CLASS_TYPE[cls], lit)))

    def skip_literal(self, text, mask, s, e):
        before = text[max(0, s - 40):s]
        # already wrapped
        if re.search(r"static_cast<\s*\w+\s*>\(\s*$", before):
            return True
        # preprocessor line
        ls = text.rfind("\n", 0, s) + 1
        if text[ls:s].lstrip().startswith("#"):
            return True
        p = prev_code_token(text, mask, s)
        n = next_code(text, mask, e)
        # the literal is the entire right hand side, initializer, argument
        # or return value: no promotion can happen
        if p in ("=", "return", "(", ",", "{", "?", ":") and n in (";", ")", ",", "}", ":"):
            # but not an argument of a math function or MIN/MAX, whose
            #   other arguments it would promote
            if p in ("(", ",") or n in (",", ")"):
                if self.in_promoting_call(text, mask, s):
                    return False
            if p == "?" or n == ":" or p == ":":
                return False
            return True
        # unary minus of a whole expression: '= -1.0;'
        if p == "-" and n in (";", ")", ",", "}"):
            pp = prev_code_token(text, mask, text.rfind("-", 0, s))
            if pp in ("=", "return", "(", ","):
                if pp in ("(", ",") and self.in_promoting_call(text, mask, s):
                    return False
                return True
        return False

    def in_promoting_call(self, text, mask, s):
        """True if position s is an argument of a call to a math function or
        MIN/MAX (or any call with more than one argument)"""
        depth = 0
        p = s - 1
        while p >= 0:
            if mask[p]:
                c = text[p]
                if c == ")":
                    depth += 1
                elif c == "(":
                    if depth == 0:
                        break
                    depth -= 1
                elif c in ";{}" and depth == 0:
                    return False
            p -= 1
        if p < 0:
            return False
        head = re.search(r"([\w:]+)\s*$", text[max(0, p - 60):p])
        if not head:
            return False
        fn = head.group(1).split("::")[-1]
        return fn in K.MATH_FUNCS or fn in ("MIN", "MAX")

    # ------------------------------------------------------------------

    def pass_casts(self, text, mask, regions, edits, decl_class):
        for m in re.finditer(r"\(\s*(double|KK_FLOAT|KK_POS_FLOAT|KK_ACC_FLOAT)\s*\)", text):
            if not mask[m.start()] or not K.in_spans(regions, m.start()):
                continue
            prev = prev_code_token(text, mask, m.start())
            if prev == "sizeof":
                continue
            # a C cast is followed by an operand, not an operator or ';'
            if next_code(text, mask, m.end()) in ";,)=*/+<>?:":
                continue
            cls = self.statement_class(text, mask, m.start(), decl_class)
            if CLASS_TYPE[cls] != m.group(1):
                edits.append((m.start(), m.end(), "(%s)" % CLASS_TYPE[cls]))

    # ------------------------------------------------------------------
    # random draws are generated in double; in device code narrow each draw
    #   to the precision of its statement so it does not promote the
    #   expression to double.  A draw that is not narrowed safely (1-drand()
    #   can round to 0 in float, and particle IDs need all 31 bits) is kept.

    def pass_rng(self, text, mask, regions, edits, decl_class):
        for m in re.finditer(r"\b(\w+)(\.|->)(drand|normal)\s*\(\s*\)", text):
            s, e = m.start(), m.end()
            if not mask[s] or not K.in_spans(regions, s):
                continue
            wm = re.search(r"static_cast<\s*(\w+)\s*>\(\s*$", text[max(0, s - 40):s])
            if wm:
                # already narrowed by an earlier run: undo if now double
                ws = s - len(wm.group(0))
                we = next_code_idx(text, mask, e)
                if (wm.group(1) in TYPE_CLASS and text[we:we + 1] == ")" and
                        self.statement_class(text, mask, s, decl_class) == "double"):
                    edits.append((ws, we + 1, m.group(0)))
                continue
            ls = s
            while ls > 0 and not (mask[ls - 1] and text[ls - 1] in ";{}"):
                ls -= 1
            le = e
            while le < len(text) and not (mask[le] and text[le] in ";{}"):
                le += 1
            stmt = text[ls:le]
            if re.search(r"MAXSMALLINT|MAXBIGINT|\bid\b", stmt):
                continue
            if re.search(r"\b1\.?0*\s*-\s*$", text[max(0, s - 12):s]):
                self.review.append((os.path.basename(self.fname), K.line_of(text, s),
                                    "1-drand(): kept double", stmt.strip()[:80]))
                continue
            if re.match(r"\s*(int|bigint|cellint)\b", stmt):
                continue
            # a draw into a double kept on purpose, e.g. one that selects an
            #   index from a cumulative distribution, where a draw rounded
            #   to 1 in float runs off the end
            line = text[text.rfind("\n", 0, s) + 1:text.find("\n", s)]
            code = re.sub(r"//[^\n]*|/\*.*?\*/", "", stmt, flags=re.S)
            if re.match(r"\s*(const\s+)?double\b", code) and KEEP_MARK in line:
                continue
            cls = self.statement_class(text, mask, s, decl_class)
            if cls != "double":
                edits.append((s, e, "static_cast<%s>(%s)" % (CLASS_TYPE[cls], m.group(0))))

    # ------------------------------------------------------------------
    # constructs that need review by hand

    def report(self, text, mask, regions):
        base = os.path.basename(self.fname)
        checks = [
            (r"sizeof\s*\(\s*(double|Particle::OnePart|OnePart|Grid::ChildCell|"
             r"Surf::Line|Surf::Tri)\s*\)",
             "sizeof of a type whose Kokkos copy may have a different size"),
            (r"\bMPI_DOUBLE\b", "MPI_DOUBLE: check the buffer is double, not a Kokkos KK view"),
            (r"\bmemcpy\s*\(", "memcpy: check element types/sizes of source and destination"),
            (r"\batomic_(add|max|min|fetch_add|sub)\b",
             "atomic on a view: check floating point atomics use the intended precision"),
            (r"\bd_ubuf\b", "d_ubuf packs integers in a double: check the buffer is double"),
            (r"\bdrand\s*\(\s*\)\s*\*\s*MAXSMALLINT|MAXSMALLINT\s*\*\s*\w+\.drand",
             "particle ID from a random draw: must stay double"),
            (r"\bKokkos::pow\s*\([^,]+,\s*\d+\s*\)", "pow with an integer exponent promotes to double"),
            (r"\b\d+\.?\d*[eE]-(1[3-9]|[2-9]\d)\b",
             "tiny absolute tolerance, below float resolution"),
        ]
        for pat, why in checks:
            for m in re.finditer(pat, text):
                if not mask[m.start()]:
                    continue
                dev = "device" if K.in_spans(regions, m.start()) else "host"
                line = text[text.rfind("\n", 0, m.start()) + 1:text.find("\n", m.start())]
                self.review.append((base, K.line_of(text, m.start()),
                                    "%s (%s)" % (why, dev), line.strip()[:100]))


# ----------------------------------------------------------------------
# helpers

def prev_code(text, mask, i):
    j = i - 1
    while j >= 0 and (not mask[j] or text[j].isspace()):
        j -= 1
    return text[j] if j >= 0 else ""


def prev_code_token(text, mask, i):
    j = i - 1
    while j >= 0 and (not mask[j] or text[j].isspace()):
        j -= 1
    if j < 0:
        return ""
    if text[j].isalnum() or text[j] == "_":
        k = j
        while k >= 0 and (text[k].isalnum() or text[k] == "_"):
            k -= 1
        return text[k + 1:j + 1]
    # two-char operators
    two = text[j - 1:j + 1] if j > 0 else ""
    if two in ("==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "&&", "||", "->", "::"):
        return two
    return text[j]


def next_code_idx(text, mask, i):
    n = len(text)
    while i < n and (not mask[i] or text[i].isspace()):
        i += 1
    return i


def next_code(text, mask, i):
    i = next_code_idx(text, mask, i)
    return text[i] if i < len(text) else ""


def signatures(text, mask, regions):
    """function signatures of device regions: name, parameter list span,
    parameter spans, and whether the function has a body"""
    sigs = []
    for rs, re_ in regions:
        head = text[rs:rs + 40]
        if any(head.startswith(mk) for mk in K.LAMBDA_MARKERS):
            continue
        # end of signature: first '{' or ';' outside parens
        i = rs
        groups = []
        while i < re_:
            if mask[i]:
                if text[i] == "(":
                    j = K.match_close(text, mask, i)
                    groups.append((i, j))
                    i = j
                    continue
                if text[i] in "{;":
                    break
            i += 1
        body = i < re_ and text[i] == "{"
        for gs, ge in groups:
            before = text[rs:gs].rstrip()
            if before.endswith("operator"):
                continue
            nm = re.search(r"(operator\s*\(\s*\)|\w+)\s*$", before)
            if not nm:
                continue
            name = re.sub(r"\s", "", nm.group(1))
            if name in K.CPP_KEYWORDS and name != "operator()":
                continue
            params = []
            ps = gs + 1
            k = gs + 1
            depth = 0
            while k < ge - 1:
                if mask[k]:
                    c = text[k]
                    if c in "([{<":
                        depth += 1
                    elif c in ")]}>":
                        depth -= 1
                    elif c == "," and depth == 0:
                        params.append((ps, k))
                        ps = k + 1
                k += 1
            if text[ps:ge - 1].strip() and text[ps:ge - 1].strip() != "void":
                params.append((ps, ge - 1))
            sigs.append({"name": name, "popen": gs, "pclose": ge - 1,
                         "params": params, "body": body})
            break
    return sigs


def declared_names(text, mask, i):
    """After a 'double' keyword at text[:i], return (first name, all names) of
    the declarators, or (None, []) if this is not a declaration."""
    n = len(text)
    names = []
    j = i
    depth = 0
    expect_name = True
    while j < n:
        if not mask[j]:
            j += 1
            continue
        c = text[j]
        if expect_name:
            if c.isspace() or c in "*&":
                j += 1
                continue
            m = K.IDENT.match(text, j)
            if not m:
                break
            if m.group(0) in ("const", "volatile", "restrict", "__restrict__"):
                j = m.end()
                continue
            # qualified name: Class::name, Class<T>::name
            q = m.end()
            while True:
                k = next_code_idx(text, mask, q)
                if text.startswith("<", k):
                    k2 = K.match_close(text, mask, k)
                    if text.startswith("::", next_code_idx(text, mask, k2)):
                        k = next_code_idx(text, mask, k2)
                if text.startswith("::", k):
                    m2 = K.IDENT.match(text, next_code_idx(text, mask, k + 2))
                    if m2:
                        m = m2
                        q = m2.end()
                        continue
                break
            names.append(m.group(0))
            expect_name = False
            j = m.end()
            continue
        if c in "([{":
            j = K.match_close(text, mask, j)
            continue
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif c == "," and depth <= 0:
            # a new declarator only in a declaration statement, not in a
            #   parameter list: peek whether the next token is a type
            k = next_code_idx(text, mask, j + 1)
            m = K.IDENT.match(text, k)
            if m and m.group(0) in ("const", "double", "int", "float", "bool",
                                     "KK_FLOAT", "KK_POS_FLOAT", "KK_ACC_FLOAT",
                                     "bigint", "cellint", "surfint", "auto"):
                break
            if m and re.match(r"(t_|DAT|HAT|Particle|Grid|Surf|rand_type|OnePart|"
                              r"ChildCell|Line|Tri|Kokkos)", m.group(0)):
                break
            expect_name = True
        elif c in ";)":
            break
        elif c == "=":
            # skip initializer to next ',' or ';' at depth 0
            k = j + 1
            while k < n:
                if mask[k]:
                    if text[k] in "([{":
                        k = K.match_close(text, mask, k)
                        continue
                    if text[k] in ",;)":
                        break
                k += 1
            j = k
            continue
        j += 1
    if not names:
        return None, []
    return names[0], names


def apply_edits(text, edits, fname):
    edits = sorted(set(edits), key=lambda e: (e[0], e[1]))
    out = []
    last = 0
    for s, e, r in edits:
        if s < last:
            # overlapping edit: keep the first, which is from an earlier pass
            continue
        out.append(text[last:s])
        out.append(r)
        last = e
    out.append(text[last:])
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--apply", action="store_true", help="write the files")
    ap.add_argument("--pass", dest="passes", action="append",
                    choices=["types", "double", "math", "literals", "casts", "rng", "post"],
                    help="run only these passes (default all)")
    ap.add_argument("--report", default=os.path.join(K.HERE, "review.md"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    passes = args.passes or ["types", "double", "math", "literals", "casts", "rng", "post"]

    conv = Converter(K.load_map())
    conv.build_param_table(K.kokkos_files())
    nchanged = 0
    for f in K.kokkos_files(args.files):
        text = open(f).read()
        new = conv.convert(f, text, passes)
        if new != text:
            nchanged += 1
            if args.apply:
                open(f, "w").write(new)
            elif not args.quiet:
                rel = os.path.relpath(f, K.ROOT)
                sys.stdout.writelines(difflib.unified_diff(
                    text.splitlines(True), new.splitlines(True), rel, rel))

    with open(args.report, "w") as r:
        r.write("# KOKKOS precision conversion: items to review by hand\n\n")
        r.write("Generated by tools/kokkos_prec/kk_prec_convert.py. Each item is a\n")
        r.write("construct the conversion cannot decide automatically.\n\n")
        r.write("| file | line | issue | code |\n|---|---|---|---|\n")
        for f, line, why, code in sorted(set(conv.review)):
            r.write("| %s | %d | %s | `%s` |\n" % (f, line, why, code.replace("|", "\\|")))
    print("kk_prec_convert: %d files %s, %d review items in %s"
          % (nchanged, "changed" if args.apply else "would change",
             len(set(conv.review)), os.path.relpath(args.report, K.ROOT)), file=sys.stderr)


if __name__ == "__main__":
    main()
