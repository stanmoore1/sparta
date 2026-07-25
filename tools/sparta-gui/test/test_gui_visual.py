#!/usr/bin/env python3
"""Confirm that controls with a visible effect produce the *correct* effect.

Counting changed pixels only proves that something happened. It cannot tell a
clockwise rotation from a counter-clockwise one, a zoom in from a zoom out, or
a horizontal mirror from a vertical one -- each of those is a one-character
mistake in the source, each leaves the pixel count identical, and a
"the pixels changed" assertion passes all of them. This project has already
shipped one such bug (the slide show's zoom-out scaled by 0.9 where zoom-in
scaled by 1.1, so the pair was not a round trip), so the checks here are built
to answer "did it do the right thing", not "did it do a thing".

Two kinds of evidence are used, depending on what a reference is available for.

The slide show displays an image *file*. ImageMagick's transform of that same
file is therefore the exact expected result, down to the pixel, and every
transform is compared against it -- and against the transform it would have
been had the direction been wrong, which must not match. Accessibility reports
the image widget's true geometry, so the capture is the displayed image and
nothing else: no toolbar, no caption, no padding.

The docked image viewer has no reference file: it re-renders the scene through
SPARTA. What can be checked there is *where* the render changed. Toggling the
axes must alter a small L at one corner; toggling the box must alter a thin
frame around the whole scene; toggling the particles must alter the filled
interior. A toggle wired to the wrong feature moves the changed region
somewhere else, which the region checks catch and a pixel count does not.
"""
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guidrive import Gui

OUT = os.environ.get("SHOT_DIR", "visual-shots")
FIX = os.environ.get("SPARTA_FIXTURES", "fixtures")

# Two captures of an unchanged view still differ slightly (anti-aliasing,
# scaling); below this is "the same image", above it a real visual change.
RMSE_SAME = 0.02

# How far a transform must be from a reference before it counts as "not that
# transform". The wrong-direction references measure ~0.3 away on the fixture,
# so this is not a close call.
RMSE_DIFFERENT = 0.10

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

results = []


def note(check, verdict, detail=""):
    results.append((check, verdict, detail))
    print(f"  {verdict:4s} {check:52s} {detail}")


def rmse(a, b):
    """Normalised RMSE between two images: 0.0 identical, 1.0 maximally different."""
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


def transform(src, dst, *ops):
    subprocess.run(["convert", src, *ops, dst], capture_output=True)
    return dst


def geometry_of(spec):
    """(w, h) from an ImageMagick crop spec such as '400x400+347+196'."""
    m = re.match(r"(\d+)x(\d+)", spec or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def changed_region(a, b):
    """Bounding box and pixel count of everything that differs between a and b.

    Returns (x, y, w, h, count), or None when nothing differs. The box is what
    makes the check semantic: it says *where* the control acted, so a button
    wired to the wrong feature fails even though it changed exactly as many
    pixels as the right one would have.
    """
    r = subprocess.run(["compare", "-metric", "AE", "-fuzz", "3%", a, b, "null:"],
                       capture_output=True, text=True)
    try:
        count = int(float((r.stderr or "0").strip().split()[0].replace(",", "")))
    except (ValueError, IndexError):
        count = -1
    if count <= 0:
        return None
    # Build the mask by subtracting the two images rather than by picking the
    # highlight colour out of `compare`'s visualisation: that visualisation
    # marks differences in red over a faded copy of the original, and this
    # scene is full of red particles, so colour-matching selected the whole
    # render and every feature came back with the same bounding box.
    mask = f"{OUT}/.mask.png"
    subprocess.run(["convert", a, b, "-compose", "difference", "-composite",
                    "-colorspace", "gray", "-threshold", "8%", mask],
                   capture_output=True)
    out = subprocess.run(["convert", mask, "-trim", "-format", "%wx%h%X%Y",
                          "info:"], capture_output=True, text=True).stdout
    m = re.match(r"(\d+)x(\d+)([+-]\d+)([+-]\d+)", out.strip())
    if not m:
        return None
    return (int(m.group(3)), int(m.group(4)), int(m.group(1)), int(m.group(2)), count)


def describe(region):
    if not region:
        return "no pixels differ"
    x, y, w, h, n = region
    return f"{n} px in a {w}x{h} region at +{x}+{y}"


class Viewer:
    """Drives a viewer, photographing the render widget itself."""

    def __init__(self, g):
        self.g = g

    def click(self, button, pause=1.2):
        """Press a toolbar button by name; raise if it cannot be found.

        Failing loudly matters: a silent miss would leave the view unchanged
        and the round-trip assertions would then "pass" for the wrong reason.
        """
        if not self.g.click_named(BTN[button], pause=pause):
            raise RuntimeError(f"no control named {BTN[button]!r} on screen")

    def capture(self, name, geom=None):
        path = self.g.capture_render(f"{OUT}/{name}.png", geom)
        if not path:
            raise RuntimeError(f"could not photograph the render for {name}")
        return path


def check_slideshow(g, src):
    """The slide show shows a file, so the file itself is the reference."""
    wid = g._xdo("search", "--name", "Slide Show").stdout.split()
    if not wid:
        note("slide show opens", "FAIL", "no window found")
        return
    # Give the window room for the image at full size plus three zoom steps.
    # In a window too small to hold it the view is clipped, and a clipped view
    # cannot be compared against a whole-image reference.
    g._xdo("windowsize", wid[-1], "1200", "900")
    time.sleep(2)

    v = Viewer(g)
    note("slide show opens with a frame", "PASS", os.path.basename(src))

    base = v.capture("00-baseline")
    note("the frame is displayed unaltered", "PASS" if same(src, base) else "FAIL",
         f"rmse={rmse(src, base):.4f} against the source file itself")

    w0, h0 = geometry_of(g.render_geometry())

    # --- zoom: check the scale factor, not merely that something changed ----
    v.click("zoom_in"); v.click("zoom_in")
    v.capture("01-zoom-in")
    w2, h2 = geometry_of(g.render_geometry())
    want_w, want_h = round(w0 * 1.21), round(h0 * 1.21)
    note("zoom in twice scales by 1.1 twice",
         "PASS" if abs(w2 - want_w) <= 2 and abs(h2 - want_h) <= 2 else "FAIL",
         f"{w0}x{h0} -> {w2}x{h2}, expected {want_w}x{want_h}")

    v.click("zoom_out"); v.click("zoom_out")
    zback = v.capture("02-zoom-roundtrip")
    wb, hb = geometry_of(g.render_geometry())
    note("zoom out is the exact inverse of zoom in",
         "PASS" if (wb, hb) == (w0, h0) and same(src, zback) else "FAIL",
         f"back to {wb}x{hb} (was {w0}x{h0}), rmse={rmse(src, zback):.4f}")

    v.click("zoom_in"); v.click("zoom_in"); v.click("zoom_in")
    v.click("reset")
    rst = v.capture("03-reset")
    wr, hr = geometry_of(g.render_geometry())
    note("reset returns to 1:1 after zooming",
         "PASS" if (wr, hr) == (w0, h0) and same(src, rst) else "FAIL",
         f"back to {wr}x{hr}, rmse={rmse(src, rst):.4f}")

    # --- rotation: compare against both directions -------------------------
    cw = transform(src, f"{OUT}/ref-rotate-cw.png", "-rotate", "90")
    ccw = transform(src, f"{OUT}/ref-rotate-ccw.png", "-rotate", "-90")

    v.click("rotate_cw")
    r1 = v.capture("04-rotate-cw")
    note("rotate clockwise matches a 90 degree clockwise rotation",
         "PASS" if same(cw, r1) else "FAIL", f"rmse={rmse(cw, r1):.4f}")
    note("rotate clockwise is not a counter-clockwise rotation",
         "PASS" if rmse(ccw, r1) > RMSE_DIFFERENT else "FAIL",
         f"rmse={rmse(ccw, r1):.4f} against the wrong direction")

    v.click("rotate_ccw")
    rback = v.capture("05-rotate-back")
    note("rotate counter-clockwise undoes it exactly",
         "PASS" if same(src, rback) else "FAIL", f"rmse={rmse(src, rback):.4f}")

    for _ in range(4):
        v.click("rotate_cw")
    r4 = v.capture("06-rotate-x4")
    note("four 90 degree rotations return to the original",
         "PASS" if same(src, r4) else "FAIL", f"rmse={rmse(src, r4):.4f}")

    # --- mirrors: horizontal must be -flop, vertical must be -flip ---------
    flop = transform(src, f"{OUT}/ref-flop.png", "-flop")
    flip = transform(src, f"{OUT}/ref-flip.png", "-flip")

    v.click("flip_h")
    fh = v.capture("07-flip-h")
    note("mirror horizontally matches a left-right mirror",
         "PASS" if same(flop, fh) else "FAIL", f"rmse={rmse(flop, fh):.4f}")
    note("mirror horizontally is not a top-bottom mirror",
         "PASS" if rmse(flip, fh) > RMSE_DIFFERENT else "FAIL",
         f"rmse={rmse(flip, fh):.4f} against the wrong axis")

    v.click("flip_h")
    note("mirroring horizontally twice returns to the original",
         "PASS" if same(src, v.capture("08-flip-h-back")) else "FAIL")

    v.click("flip_v")
    fv = v.capture("09-flip-v")
    note("mirror vertically matches a top-bottom mirror",
         "PASS" if same(flip, fv) else "FAIL", f"rmse={rmse(flip, fv):.4f}")
    note("mirror vertically is not a left-right mirror",
         "PASS" if rmse(flop, fv) > RMSE_DIFFERENT else "FAIL",
         f"rmse={rmse(flop, fv):.4f} against the wrong axis")

    v.click("flip_v")
    note("mirroring vertically twice returns to the original",
         "PASS" if same(src, v.capture("10-flip-v-back")) else "FAIL")

    note("application still alive after every transform",
         "PASS" if g.app.poll() is None else "FAIL")


# What each feature looks like when it appears or disappears: "fill" covers the
# interior, "frame" hugs the whole scene without filling it (an outline),
# "blob" is a bounded interior shape, "corner" is small and off to one side.
FEATURES = (
    ("particles", "Toggle displaying particles", "fill"),
    ("surfaces", "Toggle displaying surface elements", "blob"),
    ("box", "Toggle displaying box", "frame"),
    ("axes", "Toggle displaying axes", "corner"),
)


def check_shape(kind, region, size):
    """Does the changed region have the shape and place this feature should?

    The expectations are read off the committed fixture deck, whose scene is a
    circle centred in a square box: the particles fill the box, the surface is
    the circle at the middle, the box is an outline around everything, and the
    axes are a small L outside the box's lower-left corner. Judging position as
    well as extent is the point -- a button wired to the wrong feature still
    changes a plausible number of pixels, but not in the right place.
    """
    if not region:
        return False, "nothing changed"
    x, y, w, h, n = region
    sw, sh = size
    span = (w / max(sw, 1), h / max(sh, 1))   # how much of the render it spans
    density = n / max(w * h, 1)               # how solidly it fills its own box
    # how far the region's middle sits from the render's middle
    off = (abs(x + w / 2 - sw / 2) / max(sw, 1), abs(y + h / 2 - sh / 2) / max(sh, 1))
    where = f"spans {span[0]:.0%}x{span[1]:.0%}, {density:.0%} dense, {max(off):.0%} off-centre"

    if kind == "fill":                 # particles: broad and solid
        return (span[0] > 0.3 and span[1] > 0.3 and density > 0.15), where
    if kind == "frame":                # box: broad but hollow
        return (span[0] > 0.3 and span[1] > 0.3 and density < 0.15), where
    if kind == "blob":                 # surface: bounded, and around the middle
        return (0.05 < span[0] < 0.95 and 0.05 < span[1] < 0.95 and max(off) < 0.15), where
    if kind == "corner":               # axes: small, and away from the middle
        return (span[0] < 0.6 and span[1] < 0.6 and max(off) > 0.10), where
    return False, "unknown shape"


def check_image_viewer(g, deck):
    """The docked viewer re-renders through SPARTA, so check *where* it changed."""
    g.key("ctrl+2", 2)
    g.key("ctrl+Return", 8)
    time.sleep(20)                     # let the deck run so a box exists
    g.focus_main()
    g.key("ctrl+i", 16)                # Create Image
    g.close_extra_windows()
    g.focus_main()
    g.key("ctrl+3", 3)                 # Analyze
    time.sleep(2)

    # The Image and Slide Show panels are tabbed together and the slide show is
    # usually in front. Its controls carry the same tooltips, so without raising
    # the Image tab the checks below would silently drive the wrong panel.
    # Qt-ADS tabs are custom widgets without the "page tab" role, so the tab is
    # matched by its title instead.
    if not g.click_named("Image - ", pause=2):
        note("raise the Image panel", "FAIL", "Image dock tab not found")
        return
    time.sleep(1)

    v = Viewer(g)
    geom = g.render_geometry()
    size = geometry_of(geom)
    base = v.capture("20-iv-baseline", geom)
    note("image viewer renders a snapshot", "PASS", f"{size[0]}x{size[1]} via Create Image")

    regions = {}
    for label, frag, kind in FEATURES:
        if not g.click_named(frag, pause=6):
            note(f"toggle {label}", "FAIL", "control not found")
            continue
        off = v.capture(f"21-iv-{label}-toggled", geom)
        region = changed_region(base, off)
        regions[label] = region

        ok, why = check_shape(kind, region, size)
        note(f"toggling {label} changes the {kind} it should",
             "PASS" if ok else "FAIL", f"{describe(region)}; {why}")

        g.click_named(frag, pause=6)
        back = v.capture(f"22-iv-{label}-restored", geom)
        note(f"toggling {label} back restores the render",
             "PASS" if same(base, back) else "FAIL", f"rmse={rmse(base, back):.4f}")

    # Two buttons wired to the same slot would each pass their own check while
    # being one control; identical changed regions catch that.
    named = [(k, r) for k, r in regions.items() if r]
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            (ka, ra), (kb, rb) = named[i], named[j]
            note(f"{ka} and {kb} are different controls",
                 "PASS" if ra[:4] != rb[:4] else "FAIL",
                 "identical changed regions would mean one slot" if ra[:4] == rb[:4] else "")

    # Zoom here moves the camera and re-renders through SPARTA rather than
    # scaling a pixmap, so it is a different code path from the slide show.
    if g.click_named("Zoom in by 10 percent", pause=7):
        zi = v.capture("23-iv-zoom-in", geom)
        note("image viewer zoom in changes the render",
             "PASS" if not same(base, zi) else "FAIL", f"rmse={rmse(base, zi):.4f}")
        g.click_named("Zoom out by 10 percent", pause=7)
        zb = v.capture("24-iv-zoom-back", geom)
        note("image viewer zoom out is the inverse of zoom in",
             "PASS" if same(base, zb) else "FAIL", f"rmse={rmse(base, zb):.4f}")

    if g.click_named("Reset view to defaults", pause=7):
        rv = v.capture("25-iv-reset", geom)
        note("image viewer reset returns to the default view",
             "PASS" if same(base, rv) else "FAIL", f"rmse={rmse(base, rv):.4f}")

    note("application alive after image viewer checks",
         "PASS" if g.app.poll() is None else "FAIL")


def main():
    os.makedirs(OUT, exist_ok=True)

    frame = f"{FIX}/gimg.1000.ppm"
    if not os.path.exists(frame):
        print(f"fixture missing: {frame} (run the in.surfq deck first)")
        return 1

    with Gui(display=83, outdir=OUT, args=["-i", frame]) as g:
        time.sleep(3)
        check_slideshow(g, frame)

    deck = f"{FIX}/in.surfq"
    if not os.path.exists(deck):
        note("image viewer checks", "SKIP", "in.surfq fixture missing")
    else:
        with Gui(display=84, outdir=OUT, args=[deck]) as g:
            check_image_viewer(g, deck)

    print()
    npass = sum(1 for r in results if r[1] == "PASS")
    print(f"{npass}/{len(results)} visual checks passed")
    with open(f"{OUT}/results.tsv", "w") as f:
        for c, verdict, d in results:
            f.write(f"{c}\t{verdict}\t{d}\n")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
