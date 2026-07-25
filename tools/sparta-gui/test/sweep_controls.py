#!/usr/bin/env python3
"""Photograph what every control does, as a before/after pair, for review by eye.

The screen sweep (sweep_capture.py) answers "does every screen come up looking
right". This answers the other half: "does every control on those screens do
something, and is what it does the right thing".

For each control it photographs the screen, asks the control to perform its
action through the accessibility interface, photographs the screen again, and
crops both to the region that changed. The crops are montaged into labelled
contact sheets so a few hundred of them can be read in a sitting, and anything
suspicious pulled out at full size.

Two things make this worth doing over the automated walk in test_gui_walker.py.
That walk establishes that a control does not crash and that its own state
changed; it cannot tell whether the right thing happened -- clicking "zoom in"
and not crashing says nothing about whether the picture zoomed. And a control
that changes nothing at all on screen is reported here as exactly that, which
is how "View > Charts Window" and "Slide Show in Viewer" were found to be
doing nothing whatsoever.

Activation goes through the control's own action, not a click at its
coordinates. Coordinates only reach what is currently on screen: the viewer's
settings column is taller than its panel, so half its buttons sit below the
fold, and a coordinate-driven pass reported them as unreachable when they were
merely scrolled.

Output (nothing is committed):
    $CONTROLS_DIR/pairs/<n>-<slug>.png    one before|after crop per control
    $CONTROLS_DIR/sheet-<k>.png           contact sheets, 12 crops each
    $CONTROLS_DIR/report.tsv              name, role, verdict, changed area
"""
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guidrive import Gui, EXAMPLE

OUT = os.environ.get("CONTROLS_DIR", "/tmp/sweep-controls")
PAIRS = os.path.join(OUT, "pairs")
PER_SHEET = 12

# Controls that must not be driven unattended, and why. Same list as the
# walker's, for the same reasons: each one either ends the process, throws away
# state the rest of the pass depends on, or rearranges the docks so that every
# path collected up front stops resolving.
SKIP = [
    ("Quit", "terminates the application"),
    ("Exit", "terminates the application"),
    ("Reset Preferences", "wipes every stored setting with no confirmation"),
    ("Check for SPARTA update", "downloads a library and relaunches the process"),
    ("Relaunch SPARTA", "restarts the simulator underneath the run"),
    ("Close Group", "removes the dock group the later controls live in"),
    ("Detach Group", "floats a dock away from the window"),
    ("Close Tab", "removes the panel the later controls live in"),
    ("Welcome Screen", "opens a full-window overlay over everything else"),
    ("Reset Layout", "rearranges the docks, invalidating every path"),
    ("New Input File", "discards the buffer the rest of the pass reads"),
    ("Open Input File", "opens a modal file browser over everything"),
]


def skip_reason(name):
    for match, reason in SKIP:
        if match.lower() in name.lower():
            return reason
    return None


def is_repeat(name):
    """The data-driven repeats: one control shown once per file.

    Around 150 Open Example entries and five recent-file slots are the same
    slot with a different path. Driving one covers the code; driving the rest
    would replace the edit buffer 150 times and bury the sheets in identical
    crops. sweep_capture.py checks the other 149 as data.
    """
    return name.startswith("in.") or re.match(r"^\d+\.\s", name) is not None


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "unnamed"


def changed_box(before, after):
    """Bounding box of the pixels that differ, as (x, y, w, h), or None.

    A plain difference composite: identical pixels come out black, changed ones
    do not, and trimming the black border leaves exactly the region that
    changed. So the crop is measured rather than guessed, and a control whose
    effect is one repainted line is not cropped away.

    Not `compare`: its output image is the second picture lowlighted with the
    differences drawn over it, not a mask, so nothing about it is black and
    -trim returns the whole frame for every control alike.
    """
    r = subprocess.run(
        ["sh", "-c",
         f"convert '{before}' '{after}' -compose difference -composite "
         f"-colorspace gray -fuzz 12% -trim -format '%wx%h%X%Y' info: 2>/dev/null"],
        capture_output=True, text=True)
    m = re.match(r"^(\d+)x(\d+)([-+]\d+)([-+]\d+)$", (r.stdout or "").strip())
    if not m:
        return None
    w, h, x, y = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    if w <= 1 or h <= 1:
        return None
    return x, y, w, h


def pair_image(before, after, box, out, label):
    """One before|after crop, side by side, with the control's name under it."""
    x, y, w, h = box
    pad = 24
    x, y = max(0, x - pad), max(0, y - pad)
    w, h = min(w + 2 * pad, 900), min(h + 2 * pad, 600)
    geom = f"{w}x{h}+{x}+{y}"
    subprocess.run(
        ["sh", "-c",
         f"montage -label 'before' '{before}[{geom}]' "
         f"-label 'after' '{after}[{geom}]' "
         f"-tile 2x1 -geometry +4+4 -background '#dddddd' miff:- | "
         f"montage -label '{label}' miff:- -geometry +2+2 "
         f"-background '#f4f4f4' -pointsize 13 '{out}'"],
        capture_output=True, text=True)
    return os.path.exists(out)


def contact_sheets(paths):
    made = []
    for k in range(0, len(paths), PER_SHEET):
        chunk = paths[k:k + PER_SHEET]
        sheet = os.path.join(OUT, f"sheet-{k // PER_SHEET + 1:02d}.png")
        subprocess.run(["montage", *chunk, "-tile", "2x6", "-geometry", "+6+6",
                        "-background", "#ffffff", sheet], capture_output=True)
        if os.path.exists(sheet):
            made.append(sheet)
    return made


def main():
    os.makedirs(PAIRS, exist_ok=True)
    rows = []
    pairs = []

    with Gui(display=62, size=(1400, 950), outdir=OUT, args=[EXAMPLE]) as g:
        # Run first, so the panels the controls belong to have real content:
        # driving a chart's smoothing control against an empty chart changes
        # nothing on screen and would be reported as a dead control.
        g.key("ctrl+Return", 8)
        time.sleep(25)
        g.focus_main()

        controls = g.actions()
        print(f"{len(controls)} activatable controls", flush=True)
        if not controls:
            return 1

        # Every path that some other control's path descends from.
        parent_paths = set()
        for _, _, p, _ in controls:
            parts = p.split("/")
            for k in range(1, len(parts)):
                parent_paths.add("/".join(parts[:k]))

        seen = set()
        example_done = [False]
        n = 0
        for name, role, path, enabled in controls:
            if not name:
                continue
            key = (name, role)
            if key in seen:
                continue        # the dock chrome repeats the same buttons per dock
            seen.add(key)

            if is_repeat(name):
                if example_done[0]:
                    rows.append((name, role, "sampled", "one of this repeated control driven"))
                    continue
                example_done[0] = True

            reason = skip_reason(name)
            if reason:
                rows.append((name, role, "skipped", reason))
                print(f"  skip  {name[:44]:44s} {reason}", flush=True)
                continue
            if not enabled:
                rows.append((name, role, "disabled", "greyed out in this state"))
                continue

            n += 1
            before = os.path.join(PAIRS, f"{n:03d}-before.png")
            after = os.path.join(PAIRS, f"{n:03d}-after.png")
            subprocess.run(["import", "-window", "root", before],
                           env=g.env, capture_output=True)
            g.activate(path, name, pause=1.0)
            subprocess.run(["import", "-window", "root", after],
                           env=g.env, capture_output=True)

            box = changed_box(before, after)
            if box is None:
                # A submenu parent has an action, but performing it while the
                # menu that holds it is closed opens nothing on screen. That is
                # this harness's limit, not a defect: sweep_capture.py opens
                # those submenus for real and photographs them.
                #
                # Accessibility gives a submenu parent the same role as a leaf
                # entry ("menu item"), so the tree decides instead: a parent is
                # a control other controls' paths descend from.
                verdict = "submenu parent" if path in parent_paths else "NO VISIBLE EFFECT"
                rows.append((name, role, verdict, "nothing on screen changed"))
                print(f"  none  {name[:44]:44s} {verdict}", flush=True)
            else:
                out = os.path.join(PAIRS, f"{n:03d}-{slug(name)}.png")
                ok = pair_image(before, after, box, out, f"{name}  [{role}]")
                rows.append((name, role, "changed", f"{box[2]}x{box[3]}"))
                if ok:
                    pairs.append(out)
                print(f"  ok    {name[:44]:44s} {box[2]}x{box[3]} changed", flush=True)

            os.remove(before)
            os.remove(after)

            # Whatever it opened has to go before the next one, or every later
            # pair photographs that same dialog rather than its own control.
            g.escape(2)
            g.close_extra_windows()
            g.focus_main()

    with open(os.path.join(OUT, "report.tsv"), "w") as f:
        f.write("name\trole\tverdict\tdetail\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    sheets = contact_sheets(pairs)
    dead = [r for r in rows if r[2] == "NO VISIBLE EFFECT"]
    parents = [r for r in rows if r[2] == "submenu parent"]
    print(f"\n{len(pairs)} before/after pairs, {len(sheets)} contact sheets", flush=True)
    print(f"{len(dead)} controls changed nothing on screen "
          f"({len(parents)} submenu parents excluded)", flush=True)
    for r in dead:
        print(f"    {r[0]}  [{r[1]}]", flush=True)
    print(f"pairs:  {PAIRS}", flush=True)
    print(f"sheets: {OUT}/sheet-*.png", flush=True)
    print(f"report: {OUT}/report.tsv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
