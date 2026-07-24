#!/usr/bin/env python3
"""Exercise every menu action and record what actually happened.

Each case is judged objectively rather than by eye: a screenshot taken after
the action is compared against an "idle" reference of the untouched main
window. An action that is supposed to open something but leaves the screen
byte-identical did nothing, and is reported as such.

Dialogs are modal, so every case ends by dismissing whatever it opened and
confirming the app is still alive -- a case that kills the app is the most
important thing this pass can find.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guidrive import Gui

OUT = os.environ.get("SHOT_DIR", "gui-actions-shots")
EXAMPLE = os.environ.get("SPARTA_EXAMPLE", "examples/circle/in.circle")

# (id, description, keys to send, seconds to wait, expects_visible_change)
CASES = [
    # --- File -----------------------------------------------------------
    ("file-view-text",   "File > View Text File",        "ctrl+shift+f", 4, True),
    ("file-view-image",  "File > View Image/Movie",      "ctrl+shift+j", 4, True),
    ("file-plot-data",   "File > Plot Data File",        "ctrl+shift+p", 4, True),
    ("file-inspect",     "File > Inspect Restart File",  "ctrl+shift+r", 4, True),
    ("file-save-as",     "File > Save Input File As",    "ctrl+shift+s", 4, True),
    # --- Edit -----------------------------------------------------------
    ("edit-find",        "Edit > Find and Replace",      "ctrl+f",       3, True),
    ("edit-prefs",       "Edit > Preferences",           "ctrl+p",       4, True),
    # --- Run ------------------------------------------------------------
    ("run-check-input",  "Run > Check Input",            "ctrl+k",       4, True),
    ("run-set-vars",     "Run > Set Variables",          "ctrl+shift+v", 3, True),
    ("run-create-image", "Run > Create Image",           "ctrl+i",       8, True),
    # --- Tools ----------------------------------------------------------
    ("tools-import-surf","Tools > Import Surface",       "ctrl+shift+t", 6, True),
    ("tools-paraview",   "Tools > Export to ParaView",   "ctrl+shift+e", 5, True),
    # --- View: workspace modes -------------------------------------------
    ("view-mode-setup",  "View > Setup workspace",       "ctrl+1",       3, None),
    ("view-mode-run",    "View > Run workspace",         "ctrl+2",       3, None),
    ("view-mode-analyze","View > Analyze workspace",     "ctrl+3",       3, None),
    # --- View: panel toggles ---------------------------------------------
    ("view-output",      "View > Output Window",         "ctrl+shift+l", 3, None),
    ("view-charts",      "View > Charts Window",         "ctrl+shift+c", 3, None),
    ("view-image",       "View > Image Window",          "ctrl+shift+i", 3, None),
    ("view-slideshow",   "View > Slide Show Window",     "ctrl+l",       3, None),
    ("view-variables",   "View > Variables Window",      "ctrl+shift+w", 3, None),
    # --- About ------------------------------------------------------------
    ("about-about",      "About > About SPARTA-GUI",     "ctrl+shift+a", 3, True),
    ("about-help",       "About > Quick Help",           "ctrl+shift+h", 3, True),
]


def differs(a, b):
    """True if two PNGs differ at all (ImageMagick compare, absolute error)."""
    r = subprocess.run(["compare", "-metric", "AE", a, b, "null:"],
                       capture_output=True, text=True)
    txt = (r.stderr or "0").strip().split()[0]
    try:
        return float(txt.replace(",", "")) > 0
    except ValueError:
        return True


def main():
    os.makedirs(OUT, exist_ok=True)
    results = []
    with Gui(display=55, outdir=OUT, args=[EXAMPLE]) as g:
        g.shot("00-idle")
        idle = f"{OUT}/00-idle.png"

        for cid, desc, keys, wait, expect_change in CASES:
            if g.app.poll() is not None:
                results.append((cid, desc, "CRASH", "app died before this case"))
                break
            g.focus_main()          # a leftover window must not swallow the keys
            g.key(keys, wait)
            shot = f"{cid}.png"
            g.shot(cid)
            path = f"{OUT}/{shot}"

            note, verdict = "", "PASS"
            if not os.path.exists(path):
                verdict, note = "FAIL", "no screenshot captured"
            elif expect_change is True:
                if differs(path, idle):
                    note = "opened something"
                else:
                    verdict, note = "FAIL", "screen unchanged - action did nothing"
            else:
                note = "changed" if differs(path, idle) else "no visible change"

            # dismiss whatever opened, then make sure we are back to a usable app
            g.escape(3)
            g.close_extra_windows()
            time.sleep(0.5)
            if g.app.poll() is not None:
                verdict, note = "CRASH", note + " | app died after this action"
            results.append((cid, desc, verdict, note))
            print(f"  {verdict:5s} {desc:34s} {note}")

        alive = g.app.poll() is None

    print(f"\napp alive at end: {alive}")
    npass = sum(1 for r in results if r[2] == "PASS")
    print(f"{npass}/{len(results)} passed")
    with open(f"{OUT}/results.tsv", "w") as f:
        for cid, desc, verdict, note in results:
            f.write(f"{cid}\t{desc}\t{verdict}\t{note}\n")
    return 0 if npass == len(results) and alive else 1


if __name__ == "__main__":
    sys.exit(main())
