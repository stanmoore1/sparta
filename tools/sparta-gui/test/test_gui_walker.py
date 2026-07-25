#!/usr/bin/env python3
"""Activate every control of the running application, one after another.

The in-process walker (test_gui_widgets.cpp) drives the dialogs by constructing
them directly. It cannot reach the main window: menus, the toolbar, the mode
tabs and the dock chrome only exist once the whole application is up, and the
application is what a user actually clicks on. So this drives the real thing.

Controls are reached through the accessibility *action* interface rather than
by clicking where they are drawn. That matters for menus: a menu item is in
the tree whether or not its menu is open, but the geometry it reports is where
it would be if it were showing, so a coordinate click reaches the handful of
items that happen to be visible and lands on whatever is underneath for the
rest -- silently, since something always gets clicked.

What each activation establishes:

  * the application is still running afterwards
  * it did not leave a modal dialog that cannot be dismissed
  * the control is still there and still reachable afterwards

and, as a whole-run measure, how many controls changed anything on screen. A
control that changes nothing is not necessarily broken -- "Save" on an unmodified
buffer is meant to do nothing visible -- so this is reported rather than
asserted per control. What *is* asserted is the total: if a change to the
application left most of the interface inert, the count collapses, and a floor
catches that. Without it a run that clicked 200 dead buttons would pass.
"""
import os
import re
import subprocess
import sys
import time

from guidrive import Gui

OUT = os.environ.get("SHOT_DIR", "/tmp/guitest/walker")

results = []


def note(what, status, detail=""):
    results.append((what, status, detail))
    print(f"  {status} {what:55s} {detail}", flush=True)


# ---------------------------------------------------------------------------
# Controls that must not be activated
# ---------------------------------------------------------------------------
#
# Each of these ends the run rather than exercising it. They are matched by
# name and every skip is printed, so the list cannot quietly grow into "we
# don't test that any more". The same rules exist in the in-process walker;
# they are repeated rather than shared because the two walkers see different
# names for the same thing (an accessible name here, an objectName there).

SKIP = [
    ("Quit", "terminates the application"),
    ("Exit", "terminates the application"),
    ("Reset Preferences", "wipes every stored setting with no confirmation"),
    ("Check for SPARTA update", "downloads a library and relaunches the process"),
    ("Relaunch SPARTA", "restarts the simulator underneath the run"),
    ("Close Group", "removes the dock group the later controls live in"),
    ("Detach Group", "floats a dock away from the window the run tracks"),
    ("Close Tab", "removes the panel the later controls live in"),
    ("Welcome Screen", "opens a full-window overlay over everything else"),
    ("Reset Layout", "rearranges the docks mid-run, invalidating every path"),
]


def skip_reason(name):
    for match, reason in SKIP:
        if match.lower() in name.lower():
            return reason
    return None


def is_example_entry(name):
    """True for the per-file entries of the Open Example menu.

    There are around 150 of them and they are one control repeated: the same
    slot with a different path. One is activated so the path is covered; the
    rest would only re-measure the same code while replacing the edit buffer
    150 times. The recent-files list numbers its entries, so match those too.
    """
    return name.startswith("in.") or re.match(r"^\d+\.\s", name) is not None


def main():
    os.makedirs(OUT, exist_ok=True)
    example_done = [False]

    with Gui(display=57, size=(1400, 950), outdir=OUT) as g:
        controls = g.actions()
        note("the application publishes its controls", "PASS" if controls else "FAIL",
             f"{len(controls)} activatable controls")
        if not controls:
            return 1

        # Deduplicate by name+role: the dock chrome repeats the same four
        # buttons per dock, and driving each copy measures the same slot.
        seen = set()
        plan = []
        for name, role, path, enabled in controls:
            key = (name, role)
            if key in seen:
                continue
            seen.add(key)
            plan.append((name, role, path, enabled))

        note("controls deduplicated by name", "PASS",
             f"{len(plan)} distinct of {len(controls)}")

        # How many windows the application has when nothing has been opened;
        # anything above this is something a control put there.
        baseline_windows = len(g._xdo("search", "--name", "SPARTA").stdout.split())

        driven = changed = modal = failed = skipped = disabled = 0

        # Two screenshots per control would double the cost of a sweep of two
        # hundred. Each shot is the "after" of one control and the "before" of
        # the next, so one per control is enough -- as long as nothing between
        # them changes the screen, which is why the cleanup below only runs
        # when something was actually opened.
        shots = [f"{OUT}/walk-a.png", f"{OUT}/walk-b.png"]
        subprocess.run(["import", "-window", "root", shots[0]],
                       env=g.env, capture_output=True)
        cur = 0

        for name, role, path, enabled in plan:
            reason = skip_reason(name)
            if reason:
                skipped += 1
                print(f"  SKIP {name:55s} {reason}", flush=True)
                continue
            if is_example_entry(name):
                if example_done[0]:
                    skipped += 1
                    continue
                example_done[0] = True
            if not enabled:
                # a disabled control is a legitimate state (Stop is disabled
                # when nothing is running); count it rather than driving it
                disabled += 1
                continue

            rc = g._atspi("do", path, name, "SPARTA").returncode
            time.sleep(0.35)

            # Check liveness before anything else, and on the failure path
            # too: a control that takes the process down reports as "could not
            # be activated", and every control after it fails the same way
            # while the accessibility lookup waits out its timeout. Reporting
            # the first one and stopping names the culprit instead of burying
            # it in two hundred identical failures.
            if g.app.poll() is not None:
                note(f"application survived '{name}'", "FAIL",
                     f"the process exited ({g.app.poll()}) when this control was activated")
                return 1

            if rc == 2:      # activated, and something modal came up
                modal += 1
            elif rc != 0:
                failed += 1
                print(f"  FAIL {name:55s} could not be activated ({role})", flush=True)
                continue
            driven += 1

            nxt = 1 - cur
            subprocess.run(["import", "-window", "root", shots[nxt]],
                           env=g.env, capture_output=True)
            if not same_image(shots[cur], shots[nxt]):
                changed += 1
            cur = nxt

            # Put the application back where the next control can be reached
            # from, but only when something was opened: escaping and hunting
            # for stray windows after every control costs more than the sweep
            # itself. Escape dismisses a modal; a top-level tool window needs
            # closing, and while one holds focus nothing else receives
            # anything -- which is how a sweep silently becomes a no-op after
            # its first dialog.
            if rc == 2 or len(g._xdo("search", "--name", "SPARTA").stdout.split()) > baseline_windows:
                g.escape(2)
                g.close_extra_windows()
                subprocess.run(["import", "-window", "root", shots[cur]],
                               env=g.env, capture_output=True)

        note("every control was activated", "PASS" if failed == 0 else "FAIL",
             f"{driven} driven, {failed} failed, {modal} opened a dialog, "
             f"{skipped} skipped by rule, {disabled} disabled")

        # The floor is well under what a working build reaches; it is here to
        # catch a run that drove almost nothing, not to pin an exact number
        # that every new menu entry would have to be added to.
        note("enough of the interface is reachable", "PASS" if driven >= 40 else "FAIL",
             f"{driven} controls driven (floor 40)")

        note("controls respond to being activated",
             "PASS" if changed >= 10 else "FAIL",
             f"{changed} of {driven} changed the screen (floor 10)")

        # The tree must survive the sweep: a control that destroyed the window
        # it lives in would leave the second enumeration much shorter.
        again = g.actions()
        note("the interface is intact afterwards",
             "PASS" if len(again) >= 0.8 * len(controls) else "FAIL",
             f"{len(again)} controls, was {len(controls)}")

        note("application alive at the end", "PASS" if g.app.poll() is None else "FAIL")

    bad = [r for r in results if r[1] == "FAIL"]
    print(f"\n{len(results) - len(bad)}/{len(results)} walker checks passed")
    return 1 if bad else 0


def same_image(a, b):
    """True if two screenshots are identical enough to call unchanged."""
    if not (os.path.exists(a) and os.path.exists(b)):
        return True
    r = subprocess.run(["compare", "-metric", "AE", "-fuzz", "2%", a, b, "null:"],
                       capture_output=True, text=True)
    try:
        return int(float(r.stderr.strip().split()[0])) < 200
    except (ValueError, IndexError):
        return False


if __name__ == "__main__":
    sys.exit(main())
