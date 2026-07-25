#!/usr/bin/env python3
"""Locate and press widgets by name, using accessibility rather than pixels.

Driving a GUI by hardcoded screen coordinates is guesswork: it breaks when a
layout changes, when a window manager places a window differently, or when a
toolbar gains a button, and it fails silently -- the click lands somewhere
harmless and the test still passes.

Qt publishes its whole widget tree over AT-SPI, including each widget's role,
its accessible name (taken from the button text, or the tooltip when a button
is icon-only, which most of this application's toolbar buttons are) and its
exact position on screen. So instead of guessing, ask: find the button called
"Zoom in by 10 percent" and click its centre.

This module needs an interpreter with pyatspi; on this system that is
/usr/bin/python3.12 rather than the default python3, and it needs a session bus
plus QT_ACCESSIBILITY=1 in the application's environment, both of which
guidrive.Gui sets up.
"""
import sys
import time

try:
    import pyatspi
except ImportError:  # pragma: no cover - reported by the caller
    pyatspi = None


def available():
    """True if this interpreter can talk to the accessibility bus."""
    return pyatspi is not None


def _app(name_fragment, timeout=20):
    """Find the running application by name; it may take a moment to register."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        desktop = pyatspi.Registry.getDesktop(0)
        for app in desktop:
            try:
                if app and name_fragment.lower() in (app.name or "").lower():
                    return app
            except Exception:
                continue
        time.sleep(0.5)
    return None


def _walk(node, depth=0, limit=40):
    """Yield every accessible descendant, depth-first."""
    if node is None or depth > limit:
        return
    yield node
    try:
        for i in range(node.childCount):
            child = node.getChildAtIndex(i)
            yield from _walk(child, depth + 1, limit)
    except Exception:
        return


def controls(app_fragment="SPARTA"):
    """Every actionable control, as (name, role, x, y, w, h).

    Name is the accessible name: a button's text, or its tooltip when it has no
    text -- which is how the icon-only toolbar buttons in this application are
    identified.
    """
    app = _app(app_fragment)
    if app is None:
        return []
    found = []
    # "label" is included because Qt-ADS renders dock tabs as custom widgets
    # whose only named part is their title label -- without it a tabbed panel
    # cannot be raised, and every control behind it stays unreachable.
    # "image" is included so a capture can crop to the render area's real
    # geometry instead of a hand-measured rectangle that silently clips
    # whatever is drawn outside it.
    wanted = {"push button", "toggle button", "check box", "radio button",
              "combo box", "spin button", "slider", "page tab", "text",
              "menu item", "entry", "label", "image"}
    for node in _walk(app):
        try:
            role = node.getRoleName()
            if role not in wanted:
                continue
            comp = node.queryComponent()
            x, y = comp.getPosition(pyatspi.DESKTOP_COORDS)
            w, h = comp.getSize()
            if w <= 0 or h <= 0:
                continue
            name = node.name or ""
            desc = ""
            try:
                desc = node.description or ""
            except Exception:
                pass
            found.append((name or desc, role, x, y, w, h))
        except Exception:
            continue
    return found


def find(name_fragment, app_fragment="SPARTA", role=None):
    """Centre of the first control whose name contains @p name_fragment."""
    for name, r, x, y, w, h in controls(app_fragment):
        if role and r != role:
            continue
        if name_fragment.lower() in name.lower():
            return (x + w // 2, y + h // 2)
    return None


if __name__ == "__main__":
    # Used as a helper process: dump the control list as TSV so a harness
    # running under a different interpreter can consume it.
    if not available():
        print("pyatspi unavailable", file=sys.stderr)
        sys.exit(2)
    frag = sys.argv[1] if len(sys.argv) > 1 else "SPARTA"
    rows = controls(frag)
    for name, role, x, y, w, h in rows:
        print(f"{name}\t{role}\t{x}\t{y}\t{w}\t{h}")
    sys.exit(0 if rows else 1)
