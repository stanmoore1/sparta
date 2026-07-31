#!/usr/bin/env python3
"""One-shot post-processing for the txt2html -> reST migration.

txt2rst emits one .rst per .txt and knows nothing about SPARTA's equation
images or figure conventions. This script applies the transformations that
have to happen on the converted output, and is idempotent so it can be re-run
after any re-conversion.

Once doc/*.txt is retired this script has no further use; it is kept with the
converter for the record.

Usage:  python3 utils/converters/postconvert.py [--src src] [--eqs Eqs]
"""

import argparse
import pathlib
import re
import sys

# Full-size figures whose pages display only the _small thumbnail via a
# one-argument image(), leaving the large version unreachable.
ORPHAN_FIGURES = [
    "implicit_bend", "implicit_bend_uneven", "implicit_corner",
    "implicit_pointy", "multipoint_decrement", "multivalues",
]

# lambda_old.jpg has no .tex source; transcribed from the image.
# Bird94 equation 4.65, the single-species VHS mean free path.
HAND_TRANSCRIBED = {
    "lambda_old":
        r"\lambda = \left \{ \sqrt{2} \pi D_{\rm ref}^2 n "
        r"(T_{\rm ref}/T)^{\omega - 1/2} \right \}^{-1}",
}

# GS_list.tex and PS_list.tex are LaTeX tables rendered as images, not
# equations, so they become list-tables rather than math blocks.
GS_LIST = """.. list-table:: Gas-surface (GS) reaction types, with examples
   :header-rows: 1
   :widths: 10 35 55

   * - Symbol
     - Reaction type
     - Examples
   * - AA
     - Associative Adsorption
     - :math:`O(g) \\longrightarrow O(s)`

       :math:`O_{2}(g) \\longrightarrow O_{2}(s)`
   * - DA
     - Dissociative Adsorption
     - :math:`O_{2}(g) \\longrightarrow O(s) + O(g)`

       :math:`O_{2}(g) \\longrightarrow 2O(s)`
   * - LH1
     - Langmuir-Hinshelwood type 1
     - :math:`O(g) + O(s) \\longrightarrow O_{2}(g)`

       :math:`O(g) + C(b) \\longrightarrow CO(g)`
   * - LH3
     - Langmuir-Hinshelwood type 3
     - :math:`O(g) + O(s) \\longrightarrow O_{2}(s)`

       :math:`O(g) + C(b) \\longrightarrow CO(s)`
   * - CD
     - Condensation
     - :math:`C_{3}(g) \\longrightarrow 3C(b)`
   * - ER
     - Eley-Rideal
     - :math:`CO(g) + O(s) \\longrightarrow CO_{2}(g)`
   * - CI
     - Collision Induced
     - :math:`O(g) + CO(s) \\longrightarrow CO(g) + O(s)`

       :math:`Ar(g) + O(s) \\longrightarrow O(g) + Ar(g)`
"""

PS_LIST = """.. list-table:: Pure-surface (PS) reaction types, with examples
   :header-rows: 1
   :widths: 10 35 55

   * - Symbol
     - Reaction type
     - Examples
   * - DS
     - Desorption
     - :math:`O(s) \\longrightarrow O(g)`

       :math:`O_{2}(s) \\longrightarrow O_{2}(g)`
   * - LH2
     - Langmuir-Hinshelwood type 2
     - :math:`N(s) + O(s) \\longrightarrow NO(g)`

       :math:`O(s) + C(b) \\longrightarrow CO(g)`
   * - LH4
     - Langmuir-Hinshelwood type 4
     - :math:`N(s) + O(s) \\longrightarrow NO(s)`

       :math:`O(s) + C(b) \\longrightarrow CO(s)`
   * - SB
     - Sublimation
     - :math:`3C(b) \\longrightarrow C_{3}(g)`
"""

TABLE_REPLACEMENTS = {"GS_list": GS_LIST, "PS_list": PS_LIST}

IMAGE_RE = re.compile(
    r'^(?P<indent>[ ]*)\.\. image:: Eqs/(?P<name>[A-Za-z0-9_]+)'
    r'\.(?:jpg|JPG|png|gif)\n'
    r'(?:^[ ]*   :[a-z]+:.*\n)*',
    re.M)


def tex_body(eqs_dir, name):
    """Math body of Eqs/<name>.tex, or None if there is no usable source."""
    p = eqs_dir / (name + '.tex')
    if not p.exists():
        return None
    s = p.read_text()
    m = re.search(r'\$\$(.*?)\$\$', s, re.S)
    if m:
        return m.group(1)
    m = re.search(r'\\begin\{equation\}(.*?)\\end\{equation\}', s, re.S)
    if m:
        return m.group(1)
    return None


def as_math_block(body, indent):
    lines = [ln.strip() for ln in body.strip().split('\n') if ln.strip()]
    out = indent + '.. math::\n\n'
    return out + ''.join('%s   %s\n' % (indent, ln) for ln in lines)


def convert_equations(src, eqs_dir):
    converted = unresolved = 0

    for f in sorted(src.glob('*.rst')):
        s = f.read_text()
        if 'Eqs/' not in s:
            continue
        stats = {'ok': 0, 'bad': []}

        def repl(m):
            name, indent = m.group('name'), m.group('indent')
            if name in TABLE_REPLACEMENTS:
                stats['ok'] += 1
                return TABLE_REPLACEMENTS[name]
            body = HAND_TRANSCRIBED.get(name) or tex_body(eqs_dir, name)
            if body is None:
                stats['bad'].append(name)
                return m.group(0)
            stats['ok'] += 1
            return as_math_block(body, indent)

        new = IMAGE_RE.sub(repl, s)
        if new != s:
            f.write_text(new)
        converted += stats['ok']
        for n in stats['bad']:
            print("  no equation source for %s (in %s)" % (n, f.name),
                  file=sys.stderr)
            unresolved += 1

    print("equations converted: %d" % converted)
    if unresolved:
        print("unresolved: %d" % unresolved, file=sys.stderr)
    return unresolved


def link_orphan_figures(src):
    """Give thumbnails a :target: so the full-size figure is reachable."""
    n = 0
    for f in src.glob('*.rst'):
        s = orig = f.read_text()
        for name in ORPHAN_FIGURES:
            pat = re.compile(
                r'(\.\. image:: JPG/%s_small\.png\n)(?!   :target:)'
                % re.escape(name))
            s = pat.sub(r'\1   :target: JPG/%s.png\n' % name, s)
        if s != orig:
            f.write_text(s)
            n += 1
    print("figure pages given full-size targets: %d" % n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src', type=pathlib.Path)
    ap.add_argument('--eqs', default='Eqs', type=pathlib.Path)
    args = ap.parse_args()

    unresolved = convert_equations(args.src, args.eqs)
    link_orphan_figures(args.src)
    return 1 if unresolved else 0


if __name__ == '__main__':
    sys.exit(main())
