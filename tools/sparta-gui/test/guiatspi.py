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
"Camera zoom in by 10 percent" and click its centre. (Names lead with the
family they belong to -- "Camera ..." moves the point of view and re-renders,
"Displayed image ..." rearranges the pixels already on screen -- so a lookup
by name cannot land on the wrong one of the two.)

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


def _walk_paths(node, path=(), depth=0, limit=40):
    """Yield (node, path) for every descendant, path being child indices.

    A path lets a separate process name a control precisely enough to activate
    it later, which coordinates cannot do for anything that is not currently
    on screen -- a menu item's reported geometry is where it *would* be, so
    clicking it while its menu is closed hits whatever is underneath.
    """
    if node is None or depth > limit:
        return
    yield node, path
    try:
        for i in range(node.childCount):
            yield from _walk_paths(node.getChildAtIndex(i), path + (i,), depth + 1, limit)
    except Exception:
        return


def _resolve(app, path):
    """The node at @p path under @p app, or None if the tree has changed."""
    node = app
    for i in path:
        try:
            if i >= node.childCount:
                return None
            node = node.getChildAtIndex(i)
        except Exception:
            return None
    return node


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


# Roles that can be activated through the accessibility action interface
# rather than by clicking where they are drawn. Menu items are the reason this
# exists: they are in the tree whether or not their menu is open, so asking
# them to do their action reaches them all, while a click only ever reaches
# the handful that happen to be visible.
ACTIONABLE = {"push button", "toggle button", "check box", "radio button", "menu item"}

# activate() outcomes, also used as the helper process's exit status
ACTIVATE_OK = 0      # the action ran and returned
ACTIVATE_FAILED = 1  # no action interface, or the call failed
ACTIVATE_MODAL = 2   # the action ran and put a modal dialog on screen


def set_reply_timeout(msec, startup_msec=-1):
    """Shorten the wait for an accessibility reply.

    The default is around 25 seconds. A control that opens a modal blocks its
    reply for as long as the dialog is up, and the application has dozens of
    those, so at the default a sweep of every control would spend most of an
    hour waiting for timeouts it already knows are coming.

    Applied around the action call only. Setting it globally also throttles
    finding the application and walking its tree, and those legitimately take
    longer than a second on a first call -- which made every lookup fail and
    every control report as dead.
    """
    try:
        from gi.repository import Atspi
        Atspi.set_timeout(msec, startup_msec if startup_msec >= 0 else msec)
        return True
    except Exception:
        return False


def actionable(app_fragment="SPARTA"):
    """Every control that can be activated, as (name, role, path, enabled).

    `path` is a slash-separated list of child indices from the application
    root, which activate() below turns back into a node.
    """
    app = _app(app_fragment)
    if app is None:
        return []
    found = []
    for node, path in _walk_paths(app):
        try:
            role = node.getRoleName()
            if role not in ACTIONABLE:
                continue
            name = node.name or ""
            if not name:
                continue
            try:
                node.queryAction()
            except Exception:
                continue  # no action interface: nothing to ask it to do
            state = node.getState()
            enabled = state.contains(pyatspi.STATE_ENABLED)
            found.append((name, role, "/".join(str(i) for i in path), enabled))
        except Exception:
            continue
    return found


def activate(path, expect_name="", app_fragment="SPARTA", action_timeout=1500):
    """Perform the default action of the control at @p path.

    Falls back to a search by name when the tree has shifted under us, which
    it does whenever a previous action opened or closed something. Returns
    True if an action was performed.
    """
    app = _app(app_fragment)
    if app is None:
        return ACTIVATE_FAILED
    node = _resolve(app, [int(i) for i in path.split("/") if i != ""])
    if node is None or (expect_name and (node.name or "") != expect_name):
        node = None
        if expect_name:
            for cand, _p in _walk_paths(app):
                try:
                    if (cand.name or "") == expect_name and cand.getRoleName() in ACTIONABLE:
                        node = cand
                        break
                except Exception:
                    continue
    if node is None:
        return ACTIVATE_FAILED
    try:
        action = node.queryAction()
        if action.nActions < 1:
            return ACTIVATE_FAILED
    except Exception:
        return ACTIVATE_FAILED

    # Only now shorten the wait: the lookup above needs the default.
    set_reply_timeout(action_timeout, 15000)
    try:
        action.doAction(0)
        return ACTIVATE_OK
    except Exception as exc:
        # A control that opens a modal dialog does not return from its action
        # until the dialog closes, so the accessibility call times out waiting
        # for a reply. The action still happened -- the dialog is on screen.
        # Reporting that as a failure would mean every menu entry that asks
        # the user something counts as a dead control.
        if "Did not receive a reply" in str(exc):
            return ACTIVATE_MODAL
        return ACTIVATE_FAILED


if __name__ == "__main__":
    # Used as a helper process: the harness runs under an interpreter that
    # cannot import pyatspi, so it shells out to this one and reads TSV back.
    if not available():
        print("pyatspi unavailable", file=sys.stderr)
        sys.exit(2)

    if len(sys.argv) > 1 and sys.argv[1] == "actions":
        frag = sys.argv[2] if len(sys.argv) > 2 else "SPARTA"
        rows = actionable(frag)
        for name, role, path, enabled in rows:
            print(f"{name}\t{role}\t{path}\t{int(enabled)}")
        sys.exit(0 if rows else 1)

    if len(sys.argv) > 1 and sys.argv[1] == "do":
        path = sys.argv[2] if len(sys.argv) > 2 else ""
        want = sys.argv[3] if len(sys.argv) > 3 else ""
        frag = sys.argv[4] if len(sys.argv) > 4 else "SPARTA"
        sys.exit(activate(path, want, frag))

    frag = sys.argv[1] if len(sys.argv) > 1 else "SPARTA"
    rows = controls(frag)
    for name, role, x, y, w, h in rows:
        print(f"{name}\t{role}\t{x}\t{y}\t{w}\t{h}")
    sys.exit(0 if rows else 1)
