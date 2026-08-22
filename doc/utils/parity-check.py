#!/usr/bin/env python3
"""Compare the Sphinx manual against the txt2html manual, page by page.

The Sphinx migration must not change what the manual says.  This extracts
the things a reader depends on -- the words, where the links go, and what
anchors exist for other pages to link to -- from both builds and diffs
them.  Presentation differs by design and is ignored: tag soup, CSS
classes, and the navigation chrome Sphinx adds around every page.

Usage:
    parity-check.py OLD_HTML_DIR NEW_HTML_DIR [--verbose] [--page NAME]
                    [--no-allow-equations]

Exits non-zero if any page differs outside the agreed deltas, which are
DELTAS for the ones that recur and DECLARED for the individual fixes.

OLD_HTML_DIR has to be a complete doc/ tree -- the pages *and* the Eqs and
JPG directories beside them -- because the check also verifies that a link
which resolved in the old manual still resolves in the new one.  Generate
it by running txt2html over the .txt sources of the commit being compared
against, rather than reusing the .html committed next to them: two of those
had drifted from their own sources by the time of the migration, and a
baseline regenerated from the branch's own edited sources proves nothing.
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
    'unescaped_lt': (
        'txt2html emits a literal "<" from the source without escaping it, '
        'so a browser parses everything up to the next ">" as a tag and '
        'hides it.  Text like "if the ratio < 1, then the incoming particle '
        'may be deleted" is in the published file but invisible.  Sphinx '
        'escapes it, so the text appears.  Each occurrence is verified: the '
        'run is only allowed if the old markup really does contain it.'
    ),
    'equations': (
        'txt2html embedded pre-rendered images from doc/Eqs.  Equations are '
        'now typeset by MathJax from their LaTeX source, and the two images '
        'that were tables of surface reactions are now real tables, so the '
        'image alt text is replaced by the text it depicted.  Every one was '
        'compared against a screenshot of the image it replaces.  Allowed '
        'only where the old page had an equation image, and only for the '
        'text that replaced that image.'
    ),
    'rst_markup_in_source': (
        'variable.txt was edited at some point with reStructuredText markup '
        'rather than txt2html markup, so txt2html printed ":doc:`clear `" '
        'and "**not**" to the page as literal text.  Sphinx renders them, '
        'which is what they were written to mean, so four links the old page '
        'showed as raw markup are now real links.'
    ),
    'markup_typos_fixed': (
        'Markup typos in the txt2html sources that made a page render '
        'wrongly, fixed in the reST.  Each one is listed individually in '
        'DECLARED below with the source line it came from, so the diff it '
        'produces is reviewed rather than waved through.'
    ),
}

# Differences that are individually agreed, page by page and run by run.
#
# A heuristic cannot tell an intended fix from a regression, so the fixes
# this migration makes are written out: the exact words the old page showed,
# the exact words the new one shows, and why.  Anything else is a
# regression.  Each entry is (old run, new run, delta key, reason).
DECLARED = {
    'dump_modify.html': [
        ('(image', 'image', 'markup_typos_fixed',
         'dump_modify.txt:41 wrote "(image}" where it meant "{image}", so '
         'txt2html printed the paren and italicised the wrong span.'),
    ],
    'Section_python.html': [
        ('site https://github.com/sparta/sparta,', 'site,',
         'markup_typos_fixed',
         'Section_python.txt put the link target of "GitHub site"_ on the '
         'next line, so txt2html could not pair them and printed the URL as '
         'text.  It is a link now, and the URL is no longer duplicated.'),
    ],
    'variable.html': [
        (':doc:`python `', 'python', 'rst_markup_in_source', ''),
        (':doc:`clear `', 'clear', 'rst_markup_in_source', ''),
        ('**not**', 'not', 'rst_markup_in_source', ''),
        ('* :doc:`create_particles ` * :doc:`python `',
         'create_particles python', 'rst_markup_in_source', ''),
    ],
}

# Where an equation image stood in the old page.  Marking the site lets the
# equation allowance be scoped to the text that replaced that image, rather
# than to any smallish difference on a page that happens to contain one.
EQ_MARK = '\x01EQIMG\x01'
EQ_IMG = re.compile(r'<img\b[^>]*?src\s*=\s*"(?:\./)?Eqs/[^"]+"[^>]*>', re.I)

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

# Tags that separate one run of text from the next.  Everything else is
# inline and must not introduce a space: Pygments wraps each token of a
# literal block in its own <span>, so replacing inline tags with a space
# would split "/path/to/x" into "/ path / to / x".
BLOCK = re.compile(
    r'</(p|div|h[1-6]|li|tr|td|th|pre|ul|ol|dl|dd|dt|table|blockquote)>'
    r'|<(br|hr)\b[^>]*>', re.I)
TAG = re.compile(r'<[^>]+>')
SCRIPT = re.compile(r'<(script|style)\b.*?</\1>', re.S | re.I)


HEAD = re.compile(r'<head\b.*?</head>', re.S | re.I)
# the theme appends a permalink anchor to every heading
HEADERLINK = re.compile(r'<a\b[^>]*class="[^"]*headerlink[^"]*"[^>]*>.*?</a>',
                        re.S | re.I)


def content_only(markup):
    """Strip everything that is page furniture rather than content.

    Both banner styles are removed from both inputs, so every comparison
    built on this is symmetric: a build compared against itself always
    reports parity.
    """
    # <head> is the browser tab title and document metadata, not content
    t = HEAD.sub(' ', markup)
    t = HEADERLINK.sub(' ', t)
    t = SCRIPT.sub(' ', t)
    t = OLD_BANNER.sub(' ', t)
    return CHROME.sub(' ', t)


def visible_text(markup, *, sphinx=None):
    """The words a reader sees, normalized so formatting cannot matter."""
    t = content_only(markup)
    # An equation image shows text but contributes no words, so leave a
    # marker where it stood; without one the words that replaced it look
    # like an insertion at an arbitrary point in the page.
    t = EQ_IMG.sub(' ' + EQ_MARK + ' ', t)
    # keep block boundaries as separators so words do not run together
    t = BLOCK.sub('\n', t)
    t = TAG.sub('', t)
    t = html.unescape(t)
    t = unicodedata.normalize('NFKC', t)
    # Icon glyphs live in the Unicode private use area; the theme uses one
    # for the permalink marker.  They are decoration, not words.
    t = ''.join(' ' if '\ue000' <= c <= '\uf8ff' else c for c in t)
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
        if not href or href.startswith('javascript:'):
            continue
        if href == '#':
            # a link to the top of the page it is on; txt2html spelled the
            # same thing as "thispage.html"
            out.append('#')
            continue
        if href.startswith('#'):
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



def hidden_by_unescaped_lt(raw, old_run, new_run):
    """Is this one difference text a browser could not show?

    True only if the new words contain the "<" that caused the swallowing
    *and* are present verbatim in the old file's raw markup.  That second
    check is what makes it safe: it proves the text was published and
    hidden, rather than being new text.
    """
    if not new_run or not any('<' in w for w in new_run):
        return False
    # Where the hidden span ends mid-paragraph the browser joins the word
    # before it to the word after -- "x" + "These" becomes the one token
    # "xThese" -- so the run picks up trailing words that belong to the
    # visible text.  Trim them, but only if what is trimmed is really the
    # tail of that merged token.
    k = len(new_run)
    while k > 0 and ' '.join(new_run[:k]) not in raw:
        k -= 1
    if k == 0 or not any('<' in w for w in new_run[:k]):
        return False
    tail = ''.join(new_run[k:])
    return not tail or ''.join(old_run).endswith(tail)


def classify(name, raw, old_run, new_run, allow_equations):
    """Name the agreed delta this one difference falls under, or None.

    Every difference is judged on its own.  That cuts both ways and both
    ways matter: one unexplained difference no longer discards the
    verification of the others on the page, and an explained one no longer
    covers for an unexplained one somewhere else.
    """
    o, n = ' '.join(old_run), ' '.join(new_run)
    for d_old, d_new, key, _why in DECLARED.get(name, ()):
        if o == d_old and n == d_new:
            return key
    # Scoped to the marker: the words have to be what replaced that image,
    # not merely a difference somewhere on a page that has one.
    if allow_equations and old_run and all(w == EQ_MARK for w in old_run):
        return 'equations'
    if hidden_by_unescaped_lt(raw, old_run, new_run):
        return 'unescaped_lt'
    return None


def compare_text(name, old_markup, old_words, new_words, allow_equations):
    """Classify every difference between two pages.

    Returns (Counter of agreed deltas, list of unexplained differences).
    """
    import difflib
    raw = ' '.join(html.unescape(old_markup).split())
    agreed = collections.Counter()
    unexplained = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            a=old_words, b=new_words, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        old_run, new_run = old_words[i1:i2], new_words[j1:j2]
        key = classify(name, raw, old_run, new_run, allow_equations)
        if key:
            agreed[key] += 1
        else:
            unexplained.append({
                'tag': tag,
                'old': ' '.join(old_words[max(0, i1 - 6):i2 + 6]),
                'new': ' '.join(new_words[max(0, j1 - 6):j2 + 6]),
                'n_old': i2 - i1,
                'n_new': j2 - j1,
            })
    return agreed, unexplained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('old')
    ap.add_argument('new')
    ap.add_argument('--verbose', '-v', action='store_true')
    ap.add_argument('--page', help='check only this page')
    ap.add_argument('--no-allow-equations', dest='allow_equations',
                    action='store_false',
                    help='treat the MathJax equations as regressions too')
    args = ap.parse_args()

    old_dir, new_dir = pathlib.Path(args.old), pathlib.Path(args.new)
    pages = sorted(p.name for p in old_dir.glob('*.html'))
    if args.page:
        pages = [p for p in pages if p == args.page or p == args.page + '.html']

    missing, text_bad, link_bad, anchor_bad, ok = [], [], [], [], 0
    dangling = []
    all_new_anchors = {}
    agreed_total = collections.Counter()
    agreed_pages = collections.Counter()

    for name in pages:
        new_path = new_dir / name
        if not new_path.exists():
            missing.append(name)
            continue
        o = (old_dir / name).read_text(errors='replace')
        n = new_path.read_text(errors='replace')

        ow, nw = visible_text(o, sphinx=False), visible_text(n, sphinx=True)

        page_ok = True
        agreed = collections.Counter()
        if ow != nw:
            agreed, unexplained = compare_text(
                name, o, ow, nw, args.allow_equations)
            agreed_total.update(agreed)
            for key in agreed:
                agreed_pages[key] += 1
            if args.verbose and agreed:
                detail = ', '.join(f'{n_} {k}' for k, n_ in sorted(agreed.items()))
                print(f'  ~ {name}: {detail} (agreed)')
            if unexplained:
                text_bad.append((name, unexplained))
                page_ok = False

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
            # txt2html writes a self-reference as "thispage.html"; Sphinx
            # writes it as "#".  Same destination.
            if href == name:
                return '#'
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
        # Links the old page did not make are not automatically a fault: the
        # unescaped "<" swallowed some of them, and variable.txt's stray reST
        # left others as literal text, so recovering the words recovers the
        # link with them.  Allowed only where this page's text differences
        # were themselves agreed for one of those two reasons, and never
        # where a link was also lost.
        recoverable = agreed['unescaped_lt'] or agreed['rst_markup_in_source']
        if added and not lost and recoverable:
            agreed_total['links_recovered'] += len(list(added.elements()))
            agreed_pages['links_recovered'] += 1
            if args.verbose:
                print(f'  ~ {name}: {len(list(added.elements()))} link(s) the '
                      f'old page hid, now visible (agreed)')
        elif lost or added:
            link_bad.append((name, sorted(lost.elements()), sorted(added.elements())))
            page_ok = False

        # A link that still points where it always did is only half the
        # check: the file it points at has to be in the build.  Sphinx
        # copies only the images a page displays, so a link to a file the
        # page merely points at -- the full-size version of a thumbnail --
        # silently 404s unless something else puts it there.
        #
        # Judged against the old tree rather than absolutely, so this
        # reports what the migration broke and not what was already broken.
        # Manual.pdf, for one, is built separately and dropped in beside the
        # manual; it is absent from both trees and is not this check's
        # business.
        for href in {h for h in nc.elements() if h != '#'}:
            target = href.split('#', 1)[0]
            if not target or target.startswith(('http', 'mailto', 'ftp')):
                continue
            if not (new_dir / target).exists() and (old_dir / target).exists():
                dangling.append((name, target))
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
    print(f'  link targets moved  {len(link_bad)}')
    print(f'  links that 404      {len(dangling)}')
    print(f'  anchors lost        {len(anchor_bad)}')
    if agreed_total:
        print('\n  agreed deltas:')
        for k in sorted(agreed_total):
            print(f'    {agreed_total[k]:4d} on {agreed_pages[k]:3d} page(s)  {k}')

    if dangling:
        print('\nLINKS THAT 404 (target not in the build):')
        seen = collections.Counter(t for _, t in dangling)
        for target, n_ in seen.most_common(20):
            print(f'  {target}  ({n_} link(s))')
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
        print('\nTEXT DIFFERS (not covered by any agreed delta):')
        for name, diffs in text_bad[:15]:
            for d in diffs[:4]:
                print(f'  {name}  [{d["tag"]}: -{d["n_old"]} +{d["n_new"]}]')
                print(f'      old: ...{d["old"][:150]}...')
                print(f'      new: ...{d["new"][:150]}...')
            if len(diffs) > 4:
                print(f'      ... and {len(diffs) - 4} more on this page')

    failed = bool(missing or text_bad or link_bad or anchor_bad or dangling)
    print('\n  RESULT: ' + ('PARITY FAILED' if failed else 'PARITY OK'))
    if not failed:
        print('  agreed deltas, in full:')
        for k, v in DELTAS.items():
            print(f'    {k}: {v}')
        for name, entries in sorted(DECLARED.items()):
            for d_old, d_new, key, why in entries:
                if why:
                    print(f'    {name} [{key}]: {why}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
