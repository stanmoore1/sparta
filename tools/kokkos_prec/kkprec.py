"""Shared helpers for the KOKKOS precision conversion scripts.

A very small C++ lexer: enough to find comments, string literals, balanced
braces/parentheses, and the "device regions" of a KOKKOS source file, i.e.
the signatures and bodies of KOKKOS_INLINE_FUNCTION/KOKKOS_FUNCTION functions
and of KOKKOS_LAMBDA/SPARTA_LAMBDA lambdas, which is the code that runs on the
device and must be written in KK precision.
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KOKKOS_DIR = os.path.join(ROOT, "src", "KOKKOS")

FUNC_MARKERS = ("KOKKOS_INLINE_FUNCTION", "KOKKOS_FUNCTION",
                "KOKKOS_FORCEINLINE_FUNCTION")
LAMBDA_MARKERS = ("KOKKOS_LAMBDA", "SPARTA_LAMBDA", "KOKKOS_CLASS_LAMBDA",
                  "SPARTA_CLASS_LAMBDA")

# C math functions that have Kokkos:: overloads for float and double

MATH_FUNCS = ("sqrt", "cbrt", "exp", "log", "log10", "pow", "fabs", "sin",
              "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh",
              "tanh", "erf", "erfc", "tgamma", "lgamma", "floor", "ceil",
              "round", "fmod", "hypot", "expm1", "log1p")

FLOAT_LITERAL = re.compile(
    r"(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)([fFlL]?)(?![\w.])")

IDENT = re.compile(r"[A-Za-z_]\w*")

CPP_KEYWORDS = {
    "if", "else", "for", "while", "do", "return", "switch", "case", "break",
    "continue", "const", "static", "sizeof", "int", "double", "float", "bool",
    "void", "auto", "char", "long", "unsigned", "true", "false", "this",
    "static_cast", "reinterpret_cast", "const_cast", "template", "typename",
    "struct", "class", "operator", "volatile", "new", "delete", "inline",
    "constexpr", "default", "goto", "namespace", "using", "typedef"}


def load_map():
    with open(os.path.join(HERE, "precision_map.json")) as f:
        return json.load(f)


def code_mask(text):
    """Return a bytearray, 1 where text[i] is code, 0 in comments/strings.
    Preprocessor lines are marked as code."""
    n = len(text)
    mask = bytearray(b"\x01") * n
    i = 0
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            mask[i:j] = b"\x00" * (j - i)
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            mask[i:j] = b"\x00" * (j - i)
            i = j
        elif c == '"' or (c == "'" and not (i > 0 and text[i - 1].isalnum())):
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            mask[i:j] = b"\x00" * (j - i)
            i = j
        else:
            i += 1
    return mask


def brace_mask(text, mask):
    """code mask for bracket matching: of the branches of a preprocessor
    conditional only the last one is kept, so a construct that opens a
    brace in two branches (e.g. alternative 'for (...) {' lines) is counted
    once.  The last branch is kept since SPARTA style headers put the whole
    class in the #else branch of '#ifdef XXX_CLASS'."""
    bm = bytearray(mask)
    lines = text.splitlines(True)
    starts = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line)
    starts.append(pos)
    # each open group: list of branch start line indices
    stack = []
    drop = []
    for k, line in enumerate(lines):
        st = line.lstrip()
        if not st.startswith("#"):
            continue
        d = st[1:].lstrip()
        if d.startswith(("ifdef", "ifndef", "if")):
            stack.append([k])
        elif d.startswith(("else", "elif")) and stack:
            stack[-1].append(k)
        elif d.startswith("endif") and stack:
            branches = stack.pop()
            for b0, b1 in zip(branches[:-1], branches[1:]):
                drop.append((b0 + 1, b1))
    for a, b in drop:
        bm[starts[a]:starts[b]] = b"\x00" * (starts[b] - starts[a])
    return bm


def match_close(text, mask, i):
    """text[i] is an opening bracket; return index just past its match."""
    open_c = text[i]
    close_c = {"(": ")", "{": "}", "[": "]", "<": ">"}[open_c]
    depth = 0
    n = len(text)
    while i < n:
        if mask[i]:
            if text[i] == open_c:
                depth += 1
            elif text[i] == close_c:
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return n


def find_code(text, mask, chars, i):
    """index of the next code char in chars at or after i"""
    n = len(text)
    while i < n and not (mask[i] and text[i] in chars):
        i += 1
    return i


def device_regions(text, mask):
    """Return a sorted list of (start, end) spans of device code."""
    spans = []
    code = mask
    mask = brace_mask(text, code)
    for m in re.finditer(r"\b(" + "|".join(FUNC_MARKERS + LAMBDA_MARKERS) + r")\b", text):
        if not code[m.start()]:
            continue
        ls = text.rfind("\n", 0, m.start()) + 1
        if text[ls:m.start()].lstrip().startswith("#"):
            continue
        start = m.start()
        if m.group(1) in LAMBDA_MARKERS:
            p = find_code(text, mask, "(", m.end())
            if p >= len(text):
                continue
            q = match_close(text, mask, p)
            b = find_code(text, mask, "{", q)
            if b >= len(text):
                continue
            spans.append((start, match_close(text, mask, b)))
            continue
        # function: signature runs to the first '{' or ';' outside parens
        i = m.end()
        n = len(text)
        while i < n:
            if mask[i]:
                if text[i] == "(":
                    i = match_close(text, mask, i)
                    continue
                if text[i] in "{;":
                    break
                if text[i] == "#":  # preprocessor line inside a marker? stop
                    break
            i += 1
        if i < n and text[i] == "{":
            spans.append((start, match_close(text, mask, i)))
        else:
            spans.append((start, i))
    # merge nested/overlapping spans
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def in_spans(spans, i):
    lo, hi = 0, len(spans)
    while lo < hi:
        mid = (lo + hi) // 2
        if spans[mid][1] <= i:
            lo = mid + 1
        else:
            hi = mid
    return lo < len(spans) and spans[lo][0] <= i < spans[lo][1]


def kokkos_files(paths=None):
    if paths:
        return [os.path.abspath(p) for p in paths]
    files = []
    for f in sorted(os.listdir(KOKKOS_DIR)):
        if f.endswith((".h", ".cpp")) and f != "kokkos_structs.h":
            files.append(os.path.join(KOKKOS_DIR, f))
    return files


def line_of(text, i):
    return text.count("\n", 0, i) + 1
