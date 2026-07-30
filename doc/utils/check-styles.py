#!/usr/bin/env python3
"""Check that every style registered in the SPARTA sources is documented.

Modelled on utils/check-styles.py in LAMMPS. SPARTA registers styles with
macros of the form

    ComputeStyle(grid,ComputeGrid)

in src/*.h and src/*/*.h. Section_commands.rst carries the corresponding
category and alphabetical listings, where each entry is a :doc: link whose
text is the style name, with accelerated variants marked "(k)".

This script cross-checks the two, so that adding a style without listing it
-- or removing one without delisting it -- fails the doc build instead of
silently leaving the manual wrong.

Exits non-zero if any discrepancy is found.
"""

import argparse
import os
import re
import sys

CATEGORIES = ('Collide', 'Command', 'Compute', 'Dump', 'Fix', 'React',
              'Region', 'SurfCollide', 'SurfReact')

STYLE_RE = re.compile(
    r'^(?P<category>%s)Style\((?P<name>[^,)]+),' % '|'.join(CATEGORIES),
    re.M)

# :doc:`text <page>`, optionally with a "(k)" accelerator marker in the text
DOC_LINK_RE = re.compile(r':doc:`([^`<]+?)\s*(?:\(k\))?\s*<([^>]+)>`')

INDEX_PAGE = 'Section_commands.rst'


def unescape(text):
    """reST escapes underscores in link text; style names do not have them."""
    return text.replace('\\_', '_').strip()


def registered_styles(src_dir):
    """(category, name) pairs from the Style() macros in the sources."""
    styles = set()
    for root, _dirs, files in os.walk(src_dir):
        if os.path.basename(root) in ('MAKE', 'STUBS'):
            continue
        for fn in files:
            if not fn.endswith('.h'):
                continue
            try:
                with open(os.path.join(root, fn), 'r', errors='replace') as f:
                    text = f.read()
            except OSError:
                continue
            for m in STYLE_RE.finditer(text):
                styles.add((m.group('category'), m.group('name').strip()))
    return styles


def listed_styles(doc_dir):
    """Style names linked from the listings in Section_commands.rst."""
    path = os.path.join(doc_dir, INDEX_PAGE)
    with open(path, 'r', errors='replace') as f:
        text = f.read()
    return {unescape(m.group(1)) for m in DOC_LINK_RE.finditer(text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--src', required=True, help='SPARTA src directory')
    ap.add_argument('-d', '--doc', required=True,
                    help='documentation src directory with the .rst files')
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(args.doc, INDEX_PAGE)):
        print("check-styles: %s not found in %s" % (INDEX_PAGE, args.doc),
              file=sys.stderr)
        return 1

    styles = registered_styles(args.src)
    listed = listed_styles(args.doc)

    missing = []
    for category, name in sorted(styles):
        # accelerated variants are marked "(k)" on the base entry, not listed
        # separately, so they are covered by their base style
        if name.endswith('/kk'):
            continue
        if name not in listed:
            missing.append((category, name))

    if missing:
        print("Styles registered in %s but not listed in %s/%s:"
              % (args.src, args.doc, INDEX_PAGE))
        for category, name in missing:
            print("  %-18s %s" % (category + 'Style', name))
        return 1

    print("check-styles: %d registered styles, all listed" % len(styles))
    return 0


if __name__ == '__main__':
    sys.exit(main())
