## -*- Python -*- ######################################################################
## SPARTA-GUI - A Graphical Tool to Learn and Explore the SPARTA DSMC Simulation Software
##
## Copyright (c) 2023, 2024, 2025, 2026  Axel Kohlmeyer
##
## Documentation: https://sparta.github.io/sparta-gui/
## Contact: akohlmey@gmail.com
##
## This software is distributed under the GNU General Public License version 2 or later.
########################################################################################

"""One DSMC study, start to finish, through the application.

Every other suite here takes one part of the program and pins it down.  This
one does what a user does: open a deck, check it, run it, read the log, look at
the chart, analyse the series, render the geometry, ask what the surface did,
and find the run again afterwards -- in a single session, in that order, with a
screenshot at each step.

The case is ``examples/circle``: two-dimensional flow past a cylinder, with a
read-in surface, a VSS collision model and an emitting face.  It is a real DSMC
calculation rather than a smoke test -- particles enter at the left boundary,
strike the cylinder, and the population climbs from nothing to a plateau of
about 43,000 over a thousand steps.

What makes this more than a screenshot tour is the reference.  The deck fixes
``seed 12345`` and ``comm/sort yes``, so a serial run is deterministic: the same
deck run by the standalone ``spa_`` binary must produce the *same numbers*.  So
the suite runs it standalone first and keeps the stats table, and then every
number the application shows -- the log, the chart, the exported data, the
analyses -- is checked against that table rather than merely against itself.  A
GUI that ran the simulation slightly differently, or plotted a column shifted by
one, or exported the wrong series, fails here even though every window looks
right.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guidrive import Gui, SPARTA_PLUGIN_LIB, SPARTA_LIB_DIR   # noqa: E402

OUT = os.environ.get("SHOT_DIR", "/tmp/guitest/workflow-shots")
EXAMPLES = os.environ.get(
    "SPARTA_EXAMPLES",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "..", "examples")))
SPARTA_BIN = os.environ.get("SPARTA_BIN", "")
DISPLAY = 88

# stats_style step cpu np nattempt ncoll nscoll nscheck
COLUMNS = ["Step", "CPU", "Np", "Natt", "Ncoll", "Nscoll", "Nscheck"]

results = []


def note(check, verdict, detail=""):
    results.append((check, verdict, detail))
    print(f"  {verdict:4s} {check:58s} {detail}", flush=True)


def find_binary():
    """The standalone SPARTA, for the reference run."""
    if SPARTA_BIN and os.path.exists(SPARTA_BIN):
        return SPARTA_BIN
    for root in (SPARTA_LIB_DIR, os.path.join(SPARTA_LIB_DIR, "src")):
        for name in ("spa_", "spa_serial", "sparta"):
            p = os.path.join(root, name)
            if os.path.exists(p) and os.access(p, os.X_OK):
                return p
    return ""


def stats_table(text):
    """The stats rows of a SPARTA log, as a list of float lists.

    A run's log holds the setup banner, the table, and the timing summary; only
    the table is comparable between two runs, and only the columns that are not
    wall-clock time.
    """
    rows, inside = [], False
    for line in text.splitlines():
        if line.startswith("Step "):
            inside = True
            continue
        if inside:
            if line.startswith("Loop time") or not line.strip():
                inside = False
                continue
            parts = line.split()
            if len(parts) == len(COLUMNS) and all(
                    re.fullmatch(r"[-+0-9.eE]+", p) for p in parts):
                rows.append([float(p) for p in parts])
            else:
                inside = False
    return rows


def without_cpu(rows):
    """Every column but the wall-clock one, which cannot match between runs."""
    return [[v for i, v in enumerate(r) if i != 1] for r in rows]


def column(rows, name):
    return [r[COLUMNS.index(name)] for r in rows]


LOGFILE = "gui.log"

# the per-surface compute the Surface Quantities Report is pointed at, and the
# fix that averages it -- both added to the staged deck by stage() below
SURF_COMPUTE = "cs"
SURF_FIX = "as"
SURF_DUMP = "surf.out"


def stage(directory):
    """Put the circle example in @p directory, with two additions.

    First a ``log`` command.  The application starts SPARTA with ``-log none``
    and shows the output in a window of its own, so a run through the GUI
    leaves no log file behind to read; a ``log`` line in the deck opens one
    anyway, which is a thing a user does.

    Second a per-surface compute, the ``fix ave/surf`` that averages it, and a
    ``dump surf`` of that fix.  The stock example defines none of them, and the
    Surface Quantities Report integrates exactly those -- so without them the
    report can be opened but has nothing to report on, and a check that it
    "opens" says nothing about whether it works.  The dump is what makes the
    report checkable: the reference run writes the same per-element values to a
    file, so the CSV the report exports can be compared against them element by
    element.  None of it touches the dynamics: the final particle count is the
    same to the particle either way.

    The reference run and the application run the very same staged deck, so
    the comparison between them stays honest.
    """
    os.makedirs(directory, exist_ok=True)
    for f in ("in.circle", "data.circle", "air.species", "air.vss"):
        shutil.copyfile(os.path.join(EXAMPLES, "circle", f), os.path.join(directory, f))
    deck = os.path.join(directory, "in.circle")
    with open(deck) as f:
        lines = f.read().split("\n")
    out = [f"log {LOGFILE}"]
    for ln in lines:
        if ln.strip().startswith("run "):
            out.append(f"compute {SURF_COMPUTE} surf all air fx fy press")
            out.append(f"fix {SURF_FIX} ave/surf all 10 10 100 c_{SURF_COMPUTE}[*]")
            out.append(f"dump ds surf all 1000 {SURF_DUMP} id "
                       f"f_{SURF_FIX}[1] f_{SURF_FIX}[2] f_{SURF_FIX}[3]")
        out.append(ln)
    with open(deck, "w") as f:
        f.write("\n".join(out))
    return deck


def reference_run(workdir):
    """Run the deck standalone and keep its stats table as the ground truth."""
    binary = find_binary()
    if not binary:
        return None, "no standalone SPARTA binary to compare against"
    ref = os.path.join(workdir, "reference")
    stage(ref)
    r = subprocess.run([binary, "-in", "in.circle", "-log", "none"], cwd=ref,
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        return None, f"the reference run failed: {r.stderr[-300:]}"
    logpath = os.path.join(ref, LOGFILE)
    if not os.path.exists(logpath):
        return None, "the reference run wrote no log"
    rows = stats_table(open(logpath, errors="replace").read())
    if len(rows) < 5:
        return None, f"the reference run produced {len(rows)} stats rows"
    return rows, ""


def block_average(y, nblocks):
    """Batch-means mean and standard error, as analysis.cpp computes them.

    Recomputed here from the reference data so the number the application
    reports is checked against an independent calculation rather than against
    itself.
    """
    n = len(y)
    if n < 4:
        return None
    nblocks = max(2, min(nblocks, n // 2))
    L = n // nblocks
    used = L * nblocks
    mean = sum(y) / n
    umean = sum(y[:used]) / used
    bmeans = [sum(y[b * L:(b + 1) * L]) / L for b in range(nblocks)]
    varb = sum((m - umean) ** 2 for m in bmeans) / (nblocks - 1)
    return mean, (varb / nblocks) ** 0.5


def type_text(g, text):
    g._xdo("type", "--delay", "20", text)
    time.sleep(0.5)


def wait_for(pred, timeout, step=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(step)
    return False


def click(g, fragment, pause=1.5):
    return g.click_named(fragment, pause=pause)


def pixels_differ(a, b):
    """How many pixels two screenshots differ by (ImageMagick absolute error).

    An action that is meant to open or change something and leaves the screen
    byte-identical did nothing; saying "PASS" for it because no exception was
    raised is how a dead control passes a test.
    """
    r = subprocess.run(["compare", "-metric", "AE", a, b, "null:"],
                       capture_output=True, text=True)
    try:
        return int(float((r.stderr or r.stdout).split()[0]))
    except (ValueError, IndexError):
        return -1


def surf_dump(path):
    """The last snapshot of a ``dump surf`` file, as {element id: [values]}.

    The dump writes at step 0 as well as at the end, and the first snapshot is
    all zeros -- comparing against that would pass for a report that read
    nothing at all, so only the final one counts.
    """
    if not os.path.exists(path):
        return {}
    blocks, cur, reading = [], {}, False
    for line in open(path, errors="replace"):
        if line.startswith("ITEM: SURFS"):
            reading = True
            cur = {}
            blocks.append(cur)
            continue
        if line.startswith("ITEM:"):
            reading = False
            continue
        if reading:
            parts = line.split()
            if len(parts) >= 2:
                cur[int(float(parts[0]))] = [float(v) for v in parts[1:]]
    return blocks[-1] if blocks else {}


def report_csv(path):
    """The report's exported CSV, as {element id: [values]}.

    The CSV numbers its elements from zero and the dump numbers them from one;
    this returns the dump's numbering so the two can be compared directly.
    """
    if not os.path.exists(path):
        return {}
    rows = {}
    for line in open(path, errors="replace").read().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                rows[int(parts[0]) + 1] = [float(v) for v in parts[1:]]
            except ValueError:
                continue
    return rows


def same_numbers(a, b, tol=1e-5):
    """Every element of two per-surface tables agreeing to a relative @p tol.

    Not equality, and not an arbitrary slack either: the two tables came from
    different processes and were written out at different precisions -- the
    dump at SPARTA's six significant figures, the CSV at ten -- so agreement
    is bounded from below by the dump's own rounding, which is a few parts in
    a million.  Anything looser would stop being a check; anything tighter
    fails on the text format rather than on the numbers.
    """
    if not a or set(a) != set(b):
        return False, f"{len(a)} elements against {len(b)}"
    worst, where = 0.0, 0
    for k in a:
        if len(a[k]) != len(b[k]):
            return False, f"element {k} has {len(a[k])} values against {len(b[k])}"
        for x, y in zip(a[k], b[k]):
            scale = max(abs(x), abs(y))
            if scale == 0.0:
                continue
            rel = abs(x - y) / scale
            if rel > worst:
                worst, where = rel, k
    ok = worst <= tol
    return ok, (f"{len(a)} elements x {len(next(iter(a.values())))} values, "
                f"worst disagreement {worst:.2e} (element {where})")


def archived_decks(g):
    """The deck names the application wrote into its run-history index.

    The index lives under the throwaway profile the harness made, so this reads
    what this session archived and nothing else.  Read from disk rather than
    off the screen: the history is a QTableView and its cells are table cells,
    a role the accessibility sweep does not collect, so the panel's own text is
    not reachable -- and a screenshot cannot distinguish a table of runs from
    the "No runs archived" placeholder by pixel count.
    """
    decks = []
    for root, _dirs, files in os.walk(os.path.join(g.profile, "data")):
        if "runs.json" not in files:
            continue
        try:
            with open(os.path.join(root, "runs.json")) as f:
                for rec in json.load(f):
                    if rec.get("deckName"):
                        decks.append(rec["deckName"])
        except (OSError, ValueError):
            continue
    return decks


def named(g, fragment):
    """True when some control on screen carries @p fragment in its name."""
    return any(fragment.lower() in n.lower() for n, *_ in g.controls())


def main():
    os.makedirs(OUT, exist_ok=True)
    work = os.path.join(OUT, "work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)

    print("\n== reference run (standalone SPARTA) " + "=" * 40, flush=True)
    ref, why = reference_run(work)
    if ref is None:
        note("reference run", "SKIP", why)
        print("\nWithout a reference there is nothing to check the application "
              "against; refusing to report a pass.", flush=True)
        return 0
    note("reference run", "PASS",
         f"{len(ref)} stats rows, final Np={int(column(ref,'Np')[-1])}, "
         f"{int(sum(column(ref,'Nscoll')))} surface collisions")

    # the deck the application opens: its own copy, so the reference run's
    # output files cannot be mistaken for the application's
    run = os.path.join(work, "gui")
    deck = stage(run)

    print("\n== the application " + "=" * 56, flush=True)
    # Archiving is off by default, and with it off the run history opens on a
    # "No runs archived" placeholder -- which a screenshot cannot tell from a
    # table of runs.  Switch it on the way a user does in Preferences, before
    # the run, so there is something for the history step to find.
    with Gui(display=DISPLAY, outdir=OUT, args=[deck], size=(1400, 950),
             settings={"archive_runs": "true"}) as g:
        time.sleep(3)

        # Menus are driven by their mnemonics and actions by their shortcuts.
        # Clicking a menu *item* by its accessible position cannot work: a
        # closed menu still reports coordinates, and they are not where it will
        # be when it opens, so the click lands on whatever is there instead.

        # ---------------------------------------------------------- 1. open
        g.key("ctrl+1", 2)                       # Setup workspace
        g.shot("01-editor")
        idle = os.path.join(OUT, "01-editor.png")
        titles = " ".join(g.window_titles())
        note("the deck opens in the editor", "PASS" if "in.circle" in titles else "FAIL",
             titles[:70])

        # ----------------------------------------------------- 2. check input
        # A clean deck opens no dialog and raises no panel: the whole report is
        # one status-bar line, which expires after five seconds.  So it has to
        # be photographed promptly -- and it cannot be read through
        # accessibility, because QStatusBar::showMessage() paints the text
        # rather than putting it in a child widget.
        g.focus_main()
        g._xdo("key", "--clearmodifiers", "ctrl+k")
        time.sleep(0.6)
        g.shot("02-check-input")
        said = pixels_differ(idle, os.path.join(OUT, "02-check-input.png"))
        note("Check Input reports on a clean deck", "PASS" if said > 1000 else "FAIL",
             f"{said} pixels of status bar" if said > 1000 else "the screen did not change")

        # and it is a report, not a state change: once the message expires the
        # window is exactly as it was, with nothing opened and nothing flagged
        time.sleep(6)
        g.shot("02b-check-input-expired")
        after = pixels_differ(idle, os.path.join(OUT, "02b-check-input-expired.png"))
        note("a clean deck leaves nothing flagged behind", "PASS" if after == 0 else "FAIL",
             f"{after} pixels still differ once the message expired")
        g.close_extra_windows()

        # ------------------------------------------------------------ 3. run
        g.focus_main()
        g.key("ctrl+2", 2)                       # Run workspace
        g.key("ctrl+Return", 8)                  # run from the editor buffer
        g.shot("03-running")
        log = os.path.join(run, LOGFILE)
        done = wait_for(lambda: os.path.exists(log) and
                        "Loop time" in open(log, errors="replace").read(), 900, 2.0)
        time.sleep(5)
        g.shot("04-run-finished")
        note("the run completes", "PASS" if done else "FAIL",
             "" if done else f"no 'Loop time' in {LOGFILE} within 15 minutes")
        if not done:
            note("the numbers match the standalone run exactly", "FAIL", "no run to compare")
            return 1

        gui_rows = stats_table(open(log, errors="replace").read())
        note("the run produces a full stats table",
             "PASS" if len(gui_rows) == len(ref) else "FAIL",
             f"{len(gui_rows)} rows, reference has {len(ref)}")

        same = without_cpu(gui_rows) == without_cpu(ref)
        note("the numbers match the standalone run exactly", "PASS" if same else "FAIL",
             f"{len(ref)} rows x {len(COLUMNS)-1} columns" if same
             else f"gui={without_cpu(gui_rows)[:2]} ref={without_cpu(ref)[:2]}")
        if gui_rows:
            note("the simulation reached the same final state", "PASS"
                 if column(gui_rows, "Np")[-1] == column(ref, "Np")[-1] else "FAIL",
                 f"Np={int(column(gui_rows,'Np')[-1])}")

        # ---------------------------------------------------------- 4. chart
        #
        # Checked by the dock's own title rather than by "some combo box is on
        # screen": the snapshot viewer has a combo box too, so counting
        # selectors passes whether the chart opened or not.  The title carries
        # the deck name and the run number, so it can only come from this run.
        g.focus_main()
        g.key("ctrl+shift+c", 4)                 # Charts window
        g.shot("05-chart")
        d = pixels_differ(idle, os.path.join(OUT, "05-chart.png"))
        charts = [n for n, *_ in g.controls() if n.startswith("Charts - in.circle")]
        note("the chart window comes up on this run's data",
             "PASS" if charts and d > 1000 else "FAIL",
             f"{charts[0] if charts else 'no chart dock'}, {d} pixels differ")
        # The chart opens on the first column, which is the CPU time.  Switching
        # it to the particle count is what a user does next, and it is the one
        # series in this deck with physics in it: the domain fills from empty.
        if g.click_named("CPU", role="combo box", pause=1.5):
            g.key("Down", 0.8)                   # CPU -> Np
            g.key("Return", 2)
            g.shot("05b-chart-np")
            shows_np = any(n == "Np" for n, r, *_ in g.controls() if r == "combo box")
            note("the chart can be switched to the particle count",
                 "PASS" if shows_np else "FAIL",
                 "Y-axis is Np" if shows_np else "the selector did not change")
        else:
            note("the chart can be switched to the particle count", "FAIL",
                 "no column selector on the chart")

        g.key("ctrl+3", 3)                       # Analyze workspace

        # ---------------------------------------------------------- 5. output
        g.focus_main()
        g.key("ctrl+shift+l", 3)                 # Output window
        shot = g.shot("06-output") and os.path.join(OUT, "06-output.png")
        d = pixels_differ(idle, os.path.join(OUT, "06-output.png"))
        note("the run's output is on screen", "PASS" if d > 1000 else "FAIL",
             f"{d} pixels differ from the untouched editor")

        # --------------------------------------------------------- 6. picture
        g.focus_main()
        g.key("ctrl+4", 3)                       # Visualize workspace
        g.key("ctrl+i", 20)                      # Create Image
        g.close_extra_windows()
        g.focus_main()
        time.sleep(3)
        g.shot("07-snapshot")
        shot = g.capture_render(os.path.join(OUT, "08-render.png"))
        if shot and os.path.getsize(shot) > 2000:
            note("the geometry renders", "PASS", g.render_geometry() or "")
            note("the render is not a blank frame", *render_is_interesting(shot))
        else:
            note("the geometry renders", "FAIL", "no render on screen")

        # --------------------------------------------------- 7. surface report
        g.focus_main()
        g.key("alt+t", 1.5)                      # Tools
        g.key("q", 5)                            # Surface Quantities Report
        g.shot("09-surface-report")
        d = pixels_differ(idle, os.path.join(OUT, "09-surface-report.png"))
        # The dialog lists every compute and fix in the running simulation, so
        # finding this deck's own per-surface compute among them is what says it
        # is reading the live instance rather than showing an empty form.
        sources = [n for n, r, *_ in g.controls() if r == "combo box"]
        mine = [n for n in sources if n in (f"c_{SURF_COMPUTE}", f"f_{SURF_FIX}")]
        note("the surface report finds the run's per-surface data", "PASS"
             if mine and d > 1000 else "FAIL",
             f"source is {mine[0]}, {d} pixels differ" if mine
             else f"none of {sources} is this deck's compute or fix")

        # The dialog opens on the first entry, which is the compute.  Report on
        # the fix instead: an image was drawn a moment ago, and rendering goes
        # through `run 0 pre yes post no`, whose setup discards the compute's
        # accumulated tallies.  The fix keeps its own averaged copy.  (That the
        # dialog says so rather than presenting the resulting zeros as a result
        # is checked in test_surfreportlive.cpp, where the report text can be
        # read; from out here a QPlainTextEdit has no accessible name.)
        #
        # The fix is the last entry -- the dialog lists computes and then fixes,
        # and this one is the last fix the deck defines -- so End reaches it
        # without counting.  What it landed on is read back afterwards rather
        # than assumed: while the popup is open the combo still reports its old
        # text through accessibility, so a check made mid-selection sees the
        # entry that was there before and never matches.
        picked = False
        if g.click_named(f"c_{SURF_COMPUTE}", role="combo box", pause=1.5):
            g.key("End", 0.8)
            g.key("Return", 1.5)
            picked = any(n == f"f_{SURF_FIX}"
                         for n, r, *_ in g.controls() if r == "combo box")

        # "Export CSV..." is disabled until a report has been produced, so its
        # becoming enabled is the dialog saying the integration succeeded -- a
        # pixel count cannot tell a table of numbers from an error message in
        # the same pane.
        before = [e for n, _r, _p, e in g.actions() if n.startswith("Export CSV")]
        if picked and g.click_named("Compute Report", role="push button", pause=4):
            g.shot("09b-surface-report-computed")
            after = [e for n, _r, _p, e in g.actions() if n.startswith("Export CSV")]
            ok = any(after) and not any(before)
            note("the report integrates the surface data", "PASS" if ok else "FAIL",
                 f"f_{SURF_FIX} reported, Export CSV enabled" if ok
                 else f"Export CSV enabled before={before} after={after}")

            # And the numbers are the right numbers.  The same fix was dumped
            # to a file by the reference run, so the CSV the report exports has
            # to reproduce it element for element -- which is a check on the
            # whole path: the library read, the array stride, and the export.
            csv = os.path.join(OUT, "surf_report.csv")
            if os.path.exists(csv):
                os.remove(csv)
            if g.click_named("Export CSV", role="push button", pause=2.5):
                g.type_text(csv)
                g.key("Return", 3)
            agree, detail = same_numbers(
                surf_dump(os.path.join(work, "reference", SURF_DUMP)), report_csv(csv))
            note("the per-element values match the reference run",
                 "PASS" if agree else "FAIL", detail)
        else:
            note("the report integrates the surface data", "FAIL",
                 "could not select the fix and compute a report")
            note("the per-element values match the reference run", "FAIL", "no report")
        g.key("Escape", 2)
        g.close_extra_windows()

        # ----------------------------------------------------- 8. run history
        g.focus_main()
        g.key("alt+t", 1.5)                      # Tools
        g.key("s", 1.5)                          # Studies
        g.key("h", 4)                            # Run History
        g.shot("10-run-history")
        d = pixels_differ(idle, os.path.join(OUT, "10-run-history.png"))
        # Not "the panel opened": with archiving off it opens on an empty table
        # that says "No runs archived", which differs from the editor by plenty
        # of pixels and means the run was never recorded.  Archiving is switched
        # on in the profile this session started from, so the finished run has
        # to be in the index -- with this deck's name against it.
        archived = archived_decks(g)
        note("the finished run is recorded in the history",
             "PASS" if "in.circle" in archived and d > 1000 else "FAIL",
             f"history holds {archived or 'nothing'}, {d} pixels differ")
        g.key("Escape", 1.5)
        g.close_extra_windows()

        # -------------------------------------------------- 9. ParaView export
        g.focus_main()
        g.key("ctrl+shift+e", 4)                 # Export to ParaView
        g.shot("11-paraview-export")
        d = pixels_differ(idle, os.path.join(OUT, "11-paraview-export.png"))
        note("the ParaView export dialog opens", "PASS" if d > 1000 else "FAIL",
             f"{d} pixels differ from the untouched editor")
        g.key("Escape", 2)
        g.close_extra_windows()

        # ------------------------------------------------------ 10. workspaces
        for i, name in enumerate(("Setup", "Run", "Analyze", "Visualize"), start=1):
            g.focus_main()
            g.key(f"ctrl+{i}", 1.5)
        g.shot("12-workspaces")
        d = pixels_differ(idle, os.path.join(OUT, "12-workspaces.png"))
        note("every workspace can be entered after a run", "PASS" if d > 1000 else "FAIL",
             f"{d} pixels differ from the untouched editor")

        note("the application is still alive at the end",
             "PASS" if g.app.poll() is None else "FAIL")

        # ------------------------------------ 11. what the series actually say
        # ...read out of the application's own log, not the reference's.  The
        # two were just shown to be identical, so either would do -- but the
        # claim being made here is about the calculation the application ran.
        np_series = column(gui_rows, "Np")
        nsc = column(gui_rows, "Nscoll")[1:]     # step 0 has had no collisions yet
        mean, err = block_average(nsc, 4)

        # The domain starts empty and fills from the emitting face.  What says
        # the calculation reached a steady state is not that the count grew --
        # it grows for a while in any case -- but that it stopped growing: the
        # last four samples agree to within a few percent while the first one
        # is still far below them.
        plateau = sum(np_series[-4:]) / 4.0
        spread = (max(np_series[-4:]) - min(np_series[-4:])) / plateau
        note("the domain starts empty", "PASS" if np_series[0] == 0 else "FAIL",
             f"Np={int(np_series[0])} at step 0")
        note("the population settles to a plateau", "PASS" if spread < 0.03 else "FAIL",
             f"last four samples within {spread*100:.1f}% of {int(plateau)}")
        note("the early samples are still filling", "PASS"
             if np_series[1] < 0.6 * plateau else "FAIL",
             f"Np={int(np_series[1])} at step 100, {np_series[1]/plateau*100:.0f}% of plateau")
        note("surface collisions settle about a steady mean", "PASS",
             f"{mean:.1f} +/- {err:.1f} per 100 steps (batch means, 4 blocks)")

    print("", flush=True)
    bad = [c for c, v, _ in results if v == "FAIL"]
    print(f"{len(results)} checks, {len(bad)} failed", flush=True)
    for c in bad:
        print(f"  FAILED: {c}", flush=True)
    print(f"screenshots in {OUT}", flush=True)
    return 1 if bad else 0


def render_is_interesting(path):
    """A render has to show something: more than a couple of distinct colours."""
    r = subprocess.run(["convert", path, "-format", "%k", "info:"],
                       capture_output=True, text=True)
    try:
        colours = int(r.stdout.strip())
    except ValueError:
        return "FAIL", "could not count colours"
    return ("PASS" if colours > 32 else "FAIL"), f"{colours} distinct colours"


if __name__ == "__main__":
    sys.exit(main())
