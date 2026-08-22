"""Keep the anchor names the txt2html manual published working.

An anchor name is part of a URL.  The SPARTA web site, papers, mailing list
posts and bookmarks link to things like Section_start.html#start_7 and
collide.html#Bird94, so an anchor that disappears is a broken inbound link
even though the page still exists.

docutils lowercases every reference name while parsing, so an explicit
".. _Bird94:" produces id="bird94".  The original case cannot be recovered
from the doctree, and the lowercasing happens too early to intercept
cleanly.  This extension instead works from a manifest -- the list of
anchors the old manual actually published -- and appends an empty span for
any of them the built page does not already contain.

The manifest is doc/utils/sphinx-config/legacy_anchors.txt, generated from
the txt2html HTML at the time of the migration.  It is a compatibility
shim: once nothing links to the old names it can be deleted, and the
parity checker is what proves whether that is true.
"""
import os
import re

MANIFEST = 'legacy_anchors.txt'


def _load(app):
    path = os.path.join(os.path.dirname(__file__), MANIFEST)
    wanted = {}
    if not os.path.exists(path):
        return wanted
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            page, _, anchor = line.partition(' ')
            if anchor:
                wanted.setdefault(page, []).append(anchor)
    return wanted


def _on_page(app, pagename, templatename, context, doctree):
    anchors = app.env.sparta_legacy_anchors.get(pagename)
    if not anchors:
        return
    body = context.get('body')
    if not body:
        return
    have = set(re.findall(r'\bid="([^"]+)"', body))
    missing = [a for a in anchors if a not in have]
    if not missing:
        return
    spans = ''.join(f'<span id="{a}"></span>' for a in missing)
    context['body'] = body + (
        '\n<!-- anchors from the txt2html manual, kept so inbound links '
        'still resolve -->\n' + spans + '\n')
    app.env.sparta_legacy_added += len(missing)


def _on_finish(app, exception):
    if exception is None and getattr(app.env, 'sparta_legacy_added', 0):
        app.env.config  # noqa: B018  - touch, keeps linters quiet
        print(f'anchor_compat: re-added {app.env.sparta_legacy_added} '
              f'anchor names from the txt2html manual')


def _on_init(app):
    app.env.sparta_legacy_anchors = _load(app)
    app.env.sparta_legacy_added = 0


def setup(app):
    app.connect('builder-inited', lambda a: _on_init(a))
    app.connect('html-page-context', _on_page)
    app.connect('build-finished', _on_finish)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
