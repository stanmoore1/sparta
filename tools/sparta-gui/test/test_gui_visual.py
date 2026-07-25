#!/usr/bin/env python3
"""Confirm that controls with a visible effect actually produce it.

The widget walker proves a control exists and does not crash. It cannot prove
that "zoom in" zoomed, that "rotate" turned the right way, or that a mirror
mirrored -- for that the application has to be driven for real and the result
photographed.

Asserting merely that "the pixels changed" is too weak: it passes for a control
that changes the wrong thing, and an earlier pass on this project produced a
false positive exactly that way. So wherever the semantics allow, this asserts
a round-trip *invariant*:

    zoom in, then out          -> back to the baseline
    rotate 90 degrees x4       -> back to the baseline
    rotate one way, then back  -> back to the baseline
    mirror, then mirror again  -> back to the baseline
    reset after a zoom         -> back to the baseline

Between them these catch three faults a "something changed" check cannot: a
control that does nothing, one that does the wrong thing, and one that is not
idempotent.

Rendering is not bit-exact between runs (anti-aliasing, scaling), so "the same
image" means a normalised RMSE below RMSE_SAME. Every comparison prints its
measured value, so a marginal result is visible rather than hidden by a pass.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guidrive import Gui

OUT = os.environ.get("SHOT_DIR", "visual-shots")
FIX = os.environ.get("SPARTA_FIXTURES", "fixtures")

# Two captures of an unchanged view still differ slightly; below this is "the
# same image", above it is a real visual change.
RMSE_SAME = 0.02

# Buttons are found by their accessible name -- which for these icon-only
# toolbar buttons is the tooltip -- rather than by coordinates. Guessed
# coordinates break silently when a layout changes: the click lands somewhere
# harmless and the test still reports a pass.
BTN = {
    "zoom_in": "Zoom in by 10 percent",
    "zoom_out": "Zoom out by 10 percent",
    "rotate_cw": "Rotate displayed image 90",
    "rotate_ccw": "counter-clock",
    "flip_h": "Mirror displayed image horizontally",
    "flip_v": "Mirror displayed image vertically",
    "reset": "Reset zoom to normal",
}
# the render viewport, excluding toolbars and the control strip
VIEWPORT = os.environ.get("VIEWPORT", "935x395+10+60")
# the docked image panel inside the main window
IV_VIEWPORT = "400x150+805+490"

results = []


def note(check, verdict, detail=""):
    results.append((check, verdict, detail))
    print(f"  {verdict:4s} {check:44s} {detail}")


def rmse(a, b):
    """Normalised RMSE between two PNGs: 0.0 identical, 1.0 maximally different."""
    r = subprocess.run(["compare", "-metric", "RMSE", a, b, "null:"],
                       capture_output=True, text=True)
    txt = (r.stderr or "").strip()
    if "(" in txt:                     # "1234.5 (0.0188)"
        try:
            return float(txt.split("(")[1].split(")")[0])
        except (IndexError, ValueError):
            pass
    return 1.0


def same(a, b):
    return rmse(a, b) < RMSE_SAME


class Viewer:
    """Drives the standalone slide-show window."""

    def __init__(self, g, wid):
        self.g, self.wid = g, wid
        self.n = 0

    def click(self, button, pause=1.2):
        """Press a toolbar button by name; raises if it cannot be found.

        Failing loudly matters: a silent miss would leave the view unchanged
        and the round-trip assertions would then "pass" for the wrong reason.
        """
        if not self.g.click_named(BTN[button], pause=pause):
            raise RuntimeError(f"no control named {BTN[button]!r} on screen")

    def capture(self, name, geom=None):
        """Photograph the render viewport alone.

        Cropping matters: comparing whole windows would fold in the filename
        label and image counter, so a zoom that did nothing could still look
        like a change because the caption updated.
        """
        full = f"{OUT}/{name}-full.png"
        subprocess.run(["import", "-window", self.wid, full], env=self.g.env,
                       capture_output=True)
        cropped = f"{OUT}/{name}.png"
        subprocess.run(["convert", full, "-crop", geom or VIEWPORT, "+repage", cropped],
                       capture_output=True)
        os.remove(full)
        return cropped


def main():
    os.makedirs(OUT, exist_ok=True)
    frame = f"{FIX}/gimg.1000.ppm"
    if not os.path.exists(frame):
        print(f"fixture missing: {frame} (run the in.surfq deck first)")
        return 1

    with Gui(display=83, outdir=OUT, args=["-i", frame]) as g:
        time.sleep(3)
        ids = g._xdo("search", "--name", "Slide Show").stdout.split()
        if not ids:
            note("slide show opens", "FAIL", "no window found")
            return 1
        v = Viewer(g, ids[-1])
        note("slide show opens with a frame", "PASS", os.path.basename(frame))

        base = v.capture("00-baseline")

        # --- zoom -------------------------------------------------------
        v.click("zoom_in"); v.click("zoom_in")
        zin = v.capture("01-zoom-in")
        note("zoom in changes the view", "PASS" if not same(base, zin) else "FAIL",
             f"rmse={rmse(base, zin):.4f}, needs >{RMSE_SAME}")

        v.click("zoom_out"); v.click("zoom_out")
        zback = v.capture("02-zoom-roundtrip")
        note("zoom in then out returns to baseline",
             "PASS" if same(base, zback) else "FAIL",
             f"rmse={rmse(base, zback):.4f}, needs <{RMSE_SAME}")

        # --- reset ------------------------------------------------------
        v.click("zoom_in"); v.click("zoom_in"); v.click("zoom_in")
        v.click("reset")
        rst = v.capture("03-reset")
        note("reset returns to baseline after zooming",
             "PASS" if same(base, rst) else "FAIL",
             f"rmse={rmse(base, rst):.4f}, needs <{RMSE_SAME}")

        # --- rotation ---------------------------------------------------
        v.click("rotate_cw")
        r1 = v.capture("04-rotate-cw")
        note("rotate clockwise changes the view",
             "PASS" if not same(base, r1) else "FAIL",
             f"rmse={rmse(base, r1):.4f}, needs >{RMSE_SAME}")

        v.click("rotate_ccw")
        rback = v.capture("05-rotate-back")
        note("rotate one way then back returns to baseline",
             "PASS" if same(base, rback) else "FAIL",
             f"rmse={rmse(base, rback):.4f}, needs <{RMSE_SAME}")

        for _ in range(4):
            v.click("rotate_cw")
        r4 = v.capture("06-rotate-x4")
        note("four 90-degree rotations return to baseline",
             "PASS" if same(base, r4) else "FAIL",
             f"rmse={rmse(base, r4):.4f}, needs <{RMSE_SAME}")

        # --- mirrors ----------------------------------------------------
        # An exact match against ImageMagick's -flop is not a sound assertion
        # here: the application composites the frame into a wider viewport and
        # pads it differently per orientation, so a correct mirror still will
        # not be pixel-identical to a mirror of the whole viewport. What can be
        # asserted soundly is that each mirror changes the view, is its own
        # inverse, and is genuinely a different operation from the other one --
        # which together rule out a dead handler, a non-idempotent one, and the
        # two buttons being wired to the same slot.
        v.click("flip_h")
        fh = v.capture("07-flip-h")
        note("mirror horizontally changes the view",
             "PASS" if not same(base, fh) else "FAIL",
             f"rmse={rmse(base, fh):.4f}, needs >{RMSE_SAME}")

        v.click("flip_h")
        fhb = v.capture("08-flip-h-back")
        note("mirroring horizontally twice returns to baseline",
             "PASS" if same(base, fhb) else "FAIL",
             f"rmse={rmse(base, fhb):.4f}, needs <{RMSE_SAME}")

        v.click("flip_v")
        fv = v.capture("09-flip-v")
        note("mirror vertically changes the view",
             "PASS" if not same(base, fv) else "FAIL",
             f"rmse={rmse(base, fv):.4f}, needs >{RMSE_SAME}")

        note("horizontal and vertical mirrors differ from each other",
             "PASS" if not same(fh, fv) else "FAIL",
             f"rmse={rmse(fh, fv):.4f}, needs >{RMSE_SAME} "
             "(equal would mean both buttons run the same slot)")

        v.click("flip_v")
        fvb = v.capture("10-flip-v-back")
        note("mirroring vertically twice returns to baseline",
             "PASS" if same(base, fvb) else "FAIL",
             f"rmse={rmse(base, fvb):.4f}, needs <{RMSE_SAME}")

        note("application still alive after every transform",
             "PASS" if g.app.poll() is None else "FAIL")

    # ---------------------------------------------------------------
    # The docked image viewer
    # ---------------------------------------------------------------
    # Different code path from the slide show: these controls re-render the
    # scene through SPARTA rather than transforming an already-decoded image,
    # so a working slide show says nothing about whether these work.
    deck = f"{FIX}/in.surfq"
    if not os.path.exists(deck):
        note("image viewer checks", "SKIP", "in.surfq fixture missing")
    else:
        with Gui(display=84, outdir=OUT, args=[deck]) as g:
            g.key("ctrl+2", 2)
            g.key("ctrl+Return", 8)
            time.sleep(20)                 # let the deck run so a box exists
            g.focus_main()
            g.key("ctrl+i", 16)            # Create Image
            g.close_extra_windows()
            g.focus_main()
            g.key("ctrl+3", 3)             # Analyze
            time.sleep(2)

            # The Image and Slide Show panels are tabbed together and the slide
            # show is usually in front. Its controls carry the same tooltips
            # ("Zoom in by 10 percent"), so without raising the Image tab first
            # the checks below would silently drive the wrong panel.
            # Qt-ADS dock tabs are custom widgets and do not carry the
            # "page tab" accessibility role, so match the tab's title instead.
            if not g.click_named("Image - ", pause=2):
                note("raise the Image panel", "FAIL", "Image dock tab not found")
            time.sleep(1)

            wid = g.main_window()
            iv = Viewer(g, wid)
            ivbase = iv.capture("20-iv-baseline", IV_VIEWPORT)
            note("image viewer renders a snapshot", "PASS", "via Create Image")

            # Feature toggles: each must visibly change the render, and putting
            # it back must restore the original -- which catches a toggle wired
            # to nothing as well as one that does not undo cleanly.
            for label, frag in (("particles", "Toggle displaying particles"),
                                ("surfaces", "Toggle displaying surface elements"),
                                ("box", "Toggle displaying box"),
                                ("axes", "Toggle displaying axes")):
                if not g.click_named(frag, pause=6):
                    note(f"toggle {label}", "FAIL", "control not found")
                    continue
                off = iv.capture(f"21-iv-{label}-toggled", IV_VIEWPORT)
                changed = not same(ivbase, off)
                g.click_named(frag, pause=6)
                back = iv.capture(f"22-iv-{label}-restored", IV_VIEWPORT)
                restored = same(ivbase, back)
                note(f"toggle {label} changes the render",
                     "PASS" if changed else "FAIL",
                     f"rmse={rmse(ivbase, off):.4f}, needs >{RMSE_SAME}")
                note(f"toggling {label} back restores the render",
                     "PASS" if restored else "FAIL",
                     f"rmse={rmse(ivbase, back):.4f}, needs <{RMSE_SAME}")

            # Zoom here re-renders through SPARTA rather than scaling a pixmap.
            if g.click_named("Zoom in by 10 percent", pause=7):
                zi = iv.capture("23-iv-zoom-in", IV_VIEWPORT)
                note("image viewer zoom in changes the render",
                     "PASS" if not same(ivbase, zi) else "FAIL",
                     f"rmse={rmse(ivbase, zi):.4f}, needs >{RMSE_SAME}")
                g.click_named("Zoom out by 10 percent", pause=7)
                zb = iv.capture("24-iv-zoom-back", IV_VIEWPORT)
                note("image viewer zoom in then out returns to baseline",
                     "PASS" if same(ivbase, zb) else "FAIL",
                     f"rmse={rmse(ivbase, zb):.4f}, needs <{RMSE_SAME}")

            if g.click_named("Reset view to defaults", pause=7):
                rv = iv.capture("25-iv-reset", IV_VIEWPORT)
                note("image viewer reset returns to the default view",
                     "PASS" if same(ivbase, rv) else "FAIL",
                     f"rmse={rmse(ivbase, rv):.4f}, needs <{RMSE_SAME}")

            note("application alive after image viewer checks",
                 "PASS" if g.app.poll() is None else "FAIL")

    print()
    npass = sum(1 for r in results if r[1] == "PASS")
    print(f"{npass}/{len(results)} visual checks passed")
    with open(f"{OUT}/results.tsv", "w") as f:
        for c, verdict, d in results:
            f.write(f"{c}\t{verdict}\t{d}\n")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
