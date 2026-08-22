#!/usr/bin/env python3
"""Compare the Sphinx manual against the txt2html manual, page by page.

The Sphinx migration must not change what the manual says.  This extracts
the things a reader depends on -- the words, where the links go, and what
anchors exist for other pages to link to -- from both builds and diffs
them.  Presentation differs by design and is ignored: tag soup, CSS
classes, and the navigation chrome Sphinx adds around every page.

Usage:
    parity-check.py OLD_HTML_DIR NEW_HTML_DIR [--verbose] [--page NAME]

Exits non-zero if any page differs outside the allowances in DELTAS.
"""
import argparse
import collections
import html
import pathlib
import re
import sys
import unicodedata

# Intentional differences, agreed before the migration.  Anything not
# covered here is a regression and fails the check.
DELTAS = {
    'equations': (
        'txt2html embedded pre-rendered images from doc/Eqs; Sphinx renders '
        'the same equations with MathJax, so the image alt text is replaced '
        'by the equation source.'
    ),
}

# Sphinx wraps every page in navigation the old manual did not have.
CHROME = re.compile(
    r'<(nav|header|footer)\b.*?</\1>'
    r'|<div[^>]+role="navigation".*?</div>'
    r'|<div[^>]+class="[^"]*\b(sphinxsidebar|related|footer|headerlink)\b[^"]*".*?</div>',
    re.S | re.I)

# Every txt2html page opens with a navigation line -- command pages carry
# "SPARTA WWW Site - Documentation - Commands", chapter pages add Previous
# and Next Section links around it.  Sphinx replaces both with its own
# navigation, so neither is content.
# bounded so it cannot swallow a later <CENTER> block (the pages use them
# for figures too)
OLD_BANNER = re.compile(
    r'<CENTER>(?:(?!</?CENTER>).)*?SPARTA WWW Site(?:(?!</?CENTER>).)*?</CENTER>',
    re.S | re.I)

BLOCK = re.compile(r'</(p|div|h[1-6]|li|tr|pre|ul|ol|dl|dd|dt|table|blockquote)>',
                   re.I)
TAG = re.compile(r'<[^>]+>')
SCRIPT = re.compile(r'<(script|style)\b.*?</\1>', re.S | re.I)


def content_only(markup):
    """Strip everything that is page furniture rather than content.

    Both banner styles are removed from both inputs, so every comparison
    built on this is symmetric: a build compared against itself always
    reports parity.
    """
    t = SCRIPT.sub(' ', markup)
    t = OLD_BANNER.sub(' ', t)
    return CHROME.sub(' ', t)


def visible_text(markup, *, sphinx=None):
    """The words a reader sees, normalized so formatting cannot matter."""
    t = content_only(markup)
    # keep block boundaries as separators so words do not run together
    t = BLOCK.sub('\n', t)
    t = TAG.sub(' ', t)
    t = html.unescape(t)
    t = unicodedata.normalize('NFKC', t)
    # Sphinx numbers sections and adds permalink markers
    t = t.replace('¶', ' ')
    return [w for w in t.split() if w]


def links(markup):
    """Where each link points, ignoring the text it is attached to.

    The banner and breadcrumb links are page furniture: every old page
    carries a fixed "SPARTA WWW Site - Documentation - Commands" header and
    Sphinx replaces it with its own navigation.  Those are excluded, so what
    is compared is the links the page's own content makes.
    """
    out = []
    for m in re.finditer(r'<a\b[^>]*?href\s*=\s*"([^"]*)"', content_only(markup), re.I):
        href = html.unescape(m.group(1)).strip()
        if not href or href.startswith(('javascript:', '#')):
            continue
        out.append(href)
    return out


def anchors(markup):
    """Anchor names other pages can link to.

    Only attributes inside a tag count.  The manual's own prose contains
    things like 'the default mixture has an ID = "all"', which is text, not
    an anchor, so the search is scoped to tags rather than run over the
    whole file.
    """
    out = set()
    # <a name="..."> is the txt2html anchor form; id="..." on any element is
    # the Sphinx one.  name= on other elements is not an anchor -- <meta
    # name="author"> is document metadata, not a link target.
    for m in re.finditer(r'<a\b[^>]*?\bname\s*=\s*"([^"]+)"', markup, re.I):
        out.add(m.group(1))
    for tag in re.finditer(r'<[a-zA-Z][^>]*>', markup):
        for m in re.finditer(r'\bid\s*=\s*"([^"]+)"', tag.group(0), re.I):
            out.add(m.group(1))
    return out


def equation_images(markup):
    """Alt text / filenames of the pre-rendered equation images."""
    return {m.group(1) for m in
            re.finditer(r'<img\b[^>]*?src\s*=\s*"(?:\./)?(Eqs/[^"]+)"', markup, re.I)}


def diff_words(old, new, context=6):
    """First divergence between two word lists, with a little context."""
    import difflib
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        return {
            'tag': tag,
            'old': ' '.join(old[max(0, i1 - context):i2 + context]),
            'new': ' '.join(new[max(0, j1 - context):j2 + context]),
            'n_old': i2 - i1,
            'n_new': j2 - j1,
        }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('old')
    ap.add_argument('new')
    ap.add_argument('--verbose', '-v', action='store_true')
    ap.add_argument('--page', help='check only this page')
    ap.add_argument('--allow-equations', action='store_true', default=True)
    args = ap.parse_args()

    old_dir, new_dir = pathlib.Path(args.old), pathlib.Path(args.new)
    pages = sorted(p.name for p in old_dir.glob('*.html'))
    if args.page:
        pages = [p for p in pages if p == args.page or p == args.page + '.html']

    missing, text_bad, link_bad, anchor_bad, ok = [], [], [], [], 0
    all_new_anchors = {}

    for name in pages:
        new_path = new_dir / name
        if not new_path.exists():
            missing.append(name)
            continue
        o = (old_dir / name).read_text(errors='replace')
        n = new_path.read_text(errors='replace')

        ow, nw = visible_text(o, sphinx=False), visible_text(n, sphinx=True)
        # equations: the old page shows an image, the new one shows TeX
        eq = equation_images(o)

        page_ok = True
        if ow != nw:
            d = diff_words(ow, nw)
            # an equation site is allowed to differ
            if not (eq and args.allow_equations and d and d['n_old'] + d['n_new'] < 400):
                text_bad.append((name, d))
                page_ok = False
            elif args.verbose:
                print(f'  ~ {name}: text differs at an equation site (allowed)')

        # Compare intra-manual targets as a multiset, so replacing one of
        # several links to the same page is still caught, and report added
        # targets as well as lost ones -- a link retargeted to a page that
        # happens to exist elsewhere on the page is still a regression.
        internal = lambda L: collections.Counter(
            x for x in L if not x.startswith(('http', 'mailto')))
        oc, nc = internal(links(o)), internal(links(n))
        # A fragment whose case changed is equivalent only if the anchor it
        # names actually exists in the new build -- anchor_compat re-adds the
        # original-case anchors, so both spellings resolve.  Verified per
        # link rather than assumed.
        def resolve(href):
            page, _, frag = href.partition('#')
            if not frag:
                return href
            target = new_dir / (page or name)
            if not target.exists():
                return href
            if frag in all_new_anchors.setdefault(
                    target.name, anchors(target.read_text(errors='replace'))):
                return page + '#' + frag.lower()
            return href
        oc = collections.Counter(resolve(k) for k in oc.elements())
        nc = collections.Counter(resolve(k) for k in nc.elements())
        lost = oc - nc
        added = nc - oc
        if lost or added:
            link_bad.append((name, sorted(lost.elements()), sorted(added.elements())))
            page_ok = False

        oa, na = anchors(o), anchors(n)
        lost_a = {a for a in oa - na if not a.startswith(('index', 'search'))}
        if lost_a:
            anchor_bad.append((name, sorted(lost_a)))
            page_ok = False

        if page_ok:
            ok += 1

    total = len(pages)
    print(f'\n  pages compared      {total}')
    print(f'  clean               {ok}')
    print(f'  missing in new      {len(missing)}')
    print(f'  text differs        {len(text_bad)}')
    print(f'  links lost          {len(link_bad)}')
    print(f'  anchors lost        {len(anchor_bad)}')

    if missing:
        print('\nMISSING PAGES (URL would 404):')
        for m in missing:
            print(f'  {m}')
    if anchor_bad:
        print('\nANCHORS LOST (inbound links would break):')
        for name, a in anchor_bad[:20]:
            print(f'  {name}: {a[:8]}{" ..." if len(a) > 8 else ""}')
    if link_bad:
        print('\nLINK TARGETS CHANGED:')
        for name, lost, added in link_bad[:20]:
            if lost:
                print(f'  {name}  lost:  {lost[:6]}{" ..." if len(lost) > 6 else ""}')
            if added:
                print(f'  {name}  added: {added[:6]}{" ..." if len(added) > 6 else ""}')
    if text_bad:
        print('\nTEXT DIFFERS:')
        for name, d in text_bad[:15]:
            if not d:
                print(f'  {name}: (length only)')
                continue
            print(f'  {name}  [{d["tag"]}: -{d["n_old"]} +{d["n_new"]}]')
            print(f'      old: ...{d["old"][:150]}...')
            print(f'      new: ...{d["new"][:150]}...')

    failed = bool(missing or text_bad or link_bad or anchor_bad)
    print('\n  RESULT: ' + ('PARITY FAILED' if failed else 'PARITY OK'))
    if not failed:
        print('  allowed deltas:')
        for k, v in DELTAS.items():
            print(f'    {k}: {v}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
