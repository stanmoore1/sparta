#!/usr/bin/env python3
"""Photograph every screen the application can show, for review by eye.

This is not a pass/fail test. It is the capture half of a one-time exhaustive
review: it opens every menu, every dialog, every tab, every panel and every
viewer state in turn, photographs each one, and writes a manifest saying what
each image is supposed to show. Something then has to *look* at them.

That division is the point. The automated suites answer "did anything change"
and "did it survive", which is all a program can judge without being told what
correct looks like. They cannot tell a settings tab that came up blank from one
that came up right, and that is exactly the class of defect that survives a
green test run.

Data-driven repeats are sampled rather than enumerated. The Open Example menu
has around 150 entries that are one control repeated with a different path, so
one is opened for real and the rest are checked as data (does the file exist,
is it readable) by check_example_links() below. Photographing 150 near-identical
screens would bury the ones that matter.

Output (nothing is committed):
    $SWEEP_DIR/screens/<id>.png     one image per state
    $SWEEP_DIR/manifest.tsv         id, phase, description, what it should show
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guidrive import Gui, EXAMPLE, SPARTA_EXAMPLES

OUT = os.environ.get("SWEEP_DIR", "/tmp/sweep")
SCREENS = os.path.join(OUT, "screens")
EXAMPLES = SPARTA_EXAMPLES
DECK = EXAMPLE

manifest = []


class Sweep:
    """Captures named states and records what each one is supposed to show."""

    def __init__(self, gui, phase):
        self.g = gui
        self.phase = phase
        self.captured = 0
        self.missing = []

    # -- opening a state ---------------------------------------------------

    def menu(self, key, pause=1.2):
        """Open a top-level menu by its Alt mnemonic and leave it open."""
        self.g.key(f"alt+{key}", pause)

    def action(self, name, pause=1.5):
        """Activate a control by accessible name (works for closed menus)."""
        for n, role, path, enabled in self.g.actions():
            if n == name:
                self.g._atspi("do", path, name, "SPARTA")
                time.sleep(pause)
                return True
        return False

    def action_like(self, fragment, pause=1.5):
        """Activate the first control whose accessible name contains @p fragment.

        Asking the control to perform its action, rather than clicking where it
        is drawn, is what makes this work for anything scrolled out of view.
        The viewer's eight settings buttons live in a column taller than the
        panel, so five of them sit below the fold; a coordinate click lands
        wherever those coordinates happen to be, which is how they first showed
        up here as "could not be opened".
        """
        for n, role, path, enabled in self.g.actions():
            if fragment.lower() in n.lower():
                self.g._atspi("do", path, n, "SPARTA")
                time.sleep(pause)
                return True
        return False

    def tab(self, label, pause=0.8):
        """Raise a tab by its visible label."""
        ok = self.g.click_named(label, role="page tab", pause=pause)
        return ok

    # -- capturing ---------------------------------------------------------

    def shot(self, ident, description, expectation):
        """Photograph the screen as it stands and record what it should show."""
        path = os.path.join(SCREENS, f"{ident}.png")
        subprocess.run(["import", "-window", "root", path],
                       env=self.g.env, capture_output=True)
        ok = os.path.exists(path) and os.path.getsize(path) > 1000
        if not ok:
            self.missing.append(ident)
        else:
            self.captured += 1
        manifest.append((ident, self.phase, description, expectation))
        print(f"  {'ok ' if ok else 'MISS'} {ident:38s} {description}", flush=True)
        return ok

    def reset(self):
        """Return to the plain main window so the next state starts clean."""
        self.g.escape(2)
        self.g.close_extra_windows()
        self.g.focus_main()

    def capture(self, ident, description, expectation, opener, reset=True):
        """Open a state, photograph it, and go back to where we started."""
        try:
            opened = opener()
            if opened is False:
                print(f"  SKIP {ident:38s} could not be opened", flush=True)
                self.missing.append(ident)
                manifest.append((ident, self.phase, description,
                                 "NOT CAPTURED: could not be opened"))
                return False
            return self.shot(ident, description, expectation)
        finally:
            if reset:
                self.reset()


def check_example_links():
    """Every Open Example entry must point at a file that is really there.

    The menu is built by scanning the examples tree, so a broken entry means a
    file moved or the scan is wrong -- and the failure only shows up when
    someone picks that one entry. Checking the data behind all 150 costs
    nothing; opening all 150 windows would cost the whole sweep.
    """
    bad, total = [], 0
    if not os.path.isdir(EXAMPLES):
        return ["examples directory not found: " + EXAMPLES], 0
    for case in sorted(os.listdir(EXAMPLES)):
        d = os.path.join(EXAMPLES, case)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.startswith("in."):
                continue
            total += 1
            p = os.path.join(d, f)
            if not (os.path.isfile(p) and os.access(p, os.R_OK)):
                bad.append(p)
            elif os.path.getsize(p) == 0:
                bad.append(p + " (empty)")
    return bad, total


def write_manifest():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "manifest.tsv"), "w") as f:
        f.write("id\tphase\tdescription\texpected\n")
        for row in manifest:
            f.write("\t".join(row) + "\n")


def main():
    os.makedirs(SCREENS, exist_ok=True)
    total_missing = []

    # ---------------------------------------------------------------- phase 1
    # The application as it comes up, before anything has been run.
    with Gui(display=58, size=(1400, 950), outdir=OUT, args=[DECK]) as g:
        s = Sweep(g, "idle")
        print("== phase 1: idle application ==", flush=True)

        s.capture("00-startup", "the application at startup",
                  "editor on the left with the loaded deck syntax-highlighted, "
                  "Output panel on the right, mode tabs along the bottom, "
                  "status bar reading Ready", lambda: True)

        # -- menus ---------------------------------------------------------
        for key, name, expect in [
            ("f", "File", "New/Open/Save/Save As, Open Example submenu, View Text, "
                          "Plot Data, Inspect Restart, Quit -- each with its shortcut"),
            ("e", "Edit", "Undo/Redo/Cut/Copy/Paste, Insert Snippet, Find and Replace, "
                          "Preferences, Reset Preferences"),
            ("r", "Run", "Run from buffer/file, Stop, Check Input, Relaunch, "
                         "Set Variables, Insert Restart Commands, Create Image, 3D Snapshot"),
            ("t", "Tools", "Import Surface, Export to ParaView, Surface Quantities Report, "
                           "and a Studies submenu"),
            ("v", "View", "four workspace entries, then eight panel entries named as "
                          "windows (Output Window, Charts Window, ...), Reset Layout"),
            ("a", "About", "About SPARTA-GUI, Quick Help, both documentation links, "
                           "Check for SPARTA update"),
        ]:
            s.capture(f"01-menu-{name.lower()}", f"the {name} menu, open",
                      expect, lambda k=key: s.menu(k))

        # Two submenus, one level down.  Reach them by mnemonic rather than by
        # counting Down presses: arrow navigation skips disabled entries (Open
        # Example is disabled until the examples tree has been probed), and a
        # Right press on an entry that has no submenu walks the menu bar to the
        # next menu instead.  Both together silently captured Edit in place of
        # File > Open Example and View in place of Tools > Studies.
        s.capture("01-menu-file-examples", "File > Open Example submenu",
                  "a list of example categories (ablation, adapt, circle, ...), each "
                  "with its own submenu of in.* decks",
                  lambda: (s.menu("f"), g.key("e", 0.8)))
        s.capture("01-menu-tools-studies", "Tools > Studies submenu",
                  "Parametric Sweep and Run History",
                  lambda: (s.menu("t"), g.key("s", 0.8)))

        # -- workspace modes ------------------------------------------------
        for n, (key, mode, expect) in enumerate([
            ("ctrl+1", "Setup", "editor and Output only; Setup tab highlighted"),
            ("ctrl+2", "Run", "editor, Output and Variables; Run tab highlighted"),
            ("ctrl+3", "Analyze", "Charts and Viewer panels present; Analyze tab highlighted"),
            ("ctrl+4", "Visualize", "the Viewer panel dominant; Visualize tab highlighted"),
        ]):
            s.capture(f"02-mode-{mode.lower()}", f"{mode} workspace",
                      expect, lambda k=key: g.key(k, 2.0))

        # -- panels ---------------------------------------------------------
        for panel, expect in [
            ("Output Window", "a docked Output panel with the log text"),
            ("Charts Window", "a docked Charts panel"),
            ("Viewer Window", "a docked Viewer panel with its source tab bar"),
            ("Variables Window", "a docked Variables panel"),
            ("Parametric Sweep Window", "the parametric sweep panel with its table and controls"),
            ("Run History Window", "the run history panel with its table"),
            ("Diagnostics Window", "the diagnostics panel"),
            ("Project Files Window", "the project files panel listing files next to the deck"),
        ]:
            ident = "03-panel-" + panel.split()[0].lower()
            s.capture(ident, f"View > {panel}", expect,
                      lambda p=panel: s.action(p), reset=False)
            s.action(panel)      # toggle it back off
            s.reset()

        # -- dialogs that need no run ---------------------------------------
        for name, ident, expect in [
            ("About SPARTA-GUI", "04-about",
             "version, Qt version, build details and attribution to LAMMPS-GUI"),
            ("Quick Help", "04-quickhelp", "a scrollable summary of what the GUI does"),
            ("Find and Replace...", "04-findreplace",
             "Find and Replace fields, case/whole-word/regex options, and the action buttons"),
            ("Set Variables...", "04-setvariables",
             "a table of index variables with add/delete controls"),
            ("Insert Snippet...", "04-snippets",
             "a list of input-deck snippets with a preview"),
            ("Insert Restart Commands...", "04-restartcmds",
             "controls for building read_restart / write_restart commands"),
            ("Export to ParaView...", "04-paraview",
             "export mode selection, output directory, and format options"),
            ("Parametric Sweep...", "04-sweep",
             "the parametric sweep dialog: parameter table, ranges, run controls"),
            ("Run History...", "04-runhistory",
             "the archived-run table, empty on a fresh profile, with report buttons"),
            ("Preferences...", "04-preferences",
             "a five-tab dialog opening on General Settings"),
        ]:
            s.capture(ident, name, expect, lambda n=name: s.action(n))

        # -- preferences, one screen per tab --------------------------------
        # Full tab labels, not the leading word. "Snapshot" also names the
        # viewer's page tab in the main window behind the dialog, and that is
        # the one the lookup found: the Preferences dialog stayed on General
        # Settings and the capture recorded the wrong tab as the right one.
        for label, ident, expect in [
            ("General Settings", "05-prefs-general",
             "library path, examples path, session options"),
            ("Accelerators", "05-prefs-accel", "None / Kokkos choice, thread and GPU settings"),
            ("Snapshot Image", "05-prefs-snapshot",
             "default image size, anti-aliasing, background"),
            ("Editor Settings", "05-prefs-editor",
             "font, tab width, autocompletion, colour scheme"),
            ("Charts Settings", "05-prefs-charts",
             "chart colours, line width, smoothing defaults"),
        ]:
            s.capture(ident, f"Preferences > {label}", expect,
                      lambda l=label: (s.action("Preferences...") and s.tab(l)))

        total_missing += s.missing
        print(f"phase 1: {s.captured} captured, {len(s.missing)} missing", flush=True)

    # ---------------------------------------------------------------- phase 2
    # After a completed run: the viewers, charts and reports have content.
    with Gui(display=59, size=(1400, 950), outdir=OUT, args=[DECK]) as g:
        s = Sweep(g, "after-run")
        print("== phase 2: after a completed run ==", flush=True)

        g.key("ctrl+Return", 8)
        time.sleep(25)
        g.focus_main()
        s.shot("10-run-finished", "the application after a completed run",
               "log text in Output, a populated chart, status bar reporting the run "
               "finished, no error badge")

        s.capture("11-createimage", "Tools > Create Image",
                  "a rendered snapshot of the circle flow in the Viewer panel",
                  lambda: (g.key("ctrl+i", 14), g.close_extra_windows(),
                           g.focus_main(), g.key("ctrl+3", 2)), reset=False)
        s.reset()

        # The image viewer has eight settings buttons, one per tab of the
        # settings dialog; each carries the tab it should open on. Driving them
        # by name checks that mapping too -- a button that opens the dialog on
        # the wrong tab is exactly the sort of thing only a look will catch.
        #
        # By button *text*, not tooltip: these buttons have labels, so that is
        # their accessible name. (The icon-only toolbar buttons elsewhere are
        # the ones named after their tooltips.) And by action rather than by
        # click, because the column is taller than the panel and five of the
        # eight sit below the fold -- a coordinate click would land wherever
        # those coordinates happened to be.
        for label, ident, expect in [
            ("Particles...", "12-ivs-particles",
             "the Particles tab: mixture selector, colour-by attribute, diameter "
             "controls, per-species colour rows"),
            ("Grid...", "12-ivs-grid",
             "the Grid tab: grid enable, colour-by source, grid-line controls"),
            ("Grid Planes...", "12-ivs-planes",
             "the Grid Planes tab: per-axis gridx/gridy/gridz enables with coordinate sliders"),
            ("Surfaces...", "12-ivs-surfaces",
             "the Surfaces tab: enable, colour mode, element diameter, surface lines"),
            ("Box & Axes...", "12-ivs-box",
             "the Box/Axes tab: box, axes and subbox toggles with colour and diameter"),
            ("Camera...", "12-ivs-camera",
             "the Camera tab: theta/phi, centre, up vector, zoom; persp greyed out"),
            ("Quality...", "12-ivs-quality",
             "the Quality tab: SSAO, shiny, FSAA, background colour and gradient, lights"),
            ("Color Maps...", "12-ivs-colormaps",
             "the Color Maps tab: a map selector per mode with preview swatches"),
        ]:
            s.capture(ident, f"Image Viewer settings, opened by '{label}'", expect,
                      lambda l=label: (g.focus_main(), g.key("ctrl+3", 1.5),
                                       g.click_named("Snapshot", role="page tab", pause=1.0),
                                       s.action_like(l, pause=2.0))[-1])

        for name, ident, expect in [
            ("Surface Quantities Report...", "13-surfreport",
             "per-surface quantities read from the live instance, not an empty table"),
            ("Import Surface (STL / SPARTA)...", "13-stlwizard",
             "the STL import wizard on its first page, with file selection"),
        ]:
            s.capture(ident, name, expect, lambda n=name: s.action(n))

        s.capture("14-slideshow", "View > Slide Show in Viewer",
                  "the slide-show source showing a rendered frame, with playback and "
                  "display-transform buttons",
                  lambda: s.action("Slide S&how in Viewer") or s.action("Slide Show in Viewer"))

        s.capture("15-chartwindow", "the chart panel after a run",
                  "a line chart of the thermo output with axes labelled and a legend",
                  lambda: g.key("ctrl+3", 2.5), reset=False)
        s.reset()

        total_missing += s.missing
        print(f"phase 2: {s.captured} captured, {len(s.missing)} missing", flush=True)

    # ---------------------------------------------------------------- phase 3
    # States that only appear when something is wrong.
    print("== phase 3: error states ==", flush=True)
    with Gui(display=60, size=(1200, 800), outdir=OUT, args=[], no_library=True) as g:
        s = Sweep(g, "errors")
        s.shot("20-no-library", "startup with no SPARTA library configured",
               "a modal explaining no suitable SPARTA shared library was found, "
               "offering Browse / Exit / Download")
        total_missing += s.missing

    write_manifest()

    bad, total = check_example_links()
    print(f"\nOpen Example entries: {total} checked, {len(bad)} broken", flush=True)
    for b in bad[:20]:
        print("  BROKEN " + b, flush=True)

    print(f"\n{len(manifest)} states recorded, {len(total_missing)} not captured")
    if total_missing:
        print("not captured: " + ", ".join(total_missing))
    print(f"images: {SCREENS}\nmanifest: {os.path.join(OUT, 'manifest.tsv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
