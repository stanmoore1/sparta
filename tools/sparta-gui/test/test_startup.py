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

"""What the application does before it has a main window: src/main.cpp.

main() is one function and it ends in app.exec(), so nothing in it can be
reached from an in-process test -- which is why it sat at 48.6%, the lowest of
any file in the project.  Everything below the option parsing is a standalone
mode the user reaches from the command line and nowhere else:

  -c FILE   open a data file directly in a chart window
  -i FILE   open images or movies in the snapshot viewer (repeatable)
  -t FILE   open a file in the text viewer
  -p PATH   set the SPARTA shared library path and remember it

Each of those either opens a window and runs the event loop, or refuses the
file and exits 1 -- behind a modal error box in the refusing case.  So this
drives the real binary as a subprocess under its own Xvfb, dismisses whatever
modal it puts up, and checks the exit status and the window it left behind.

The refusals are the part worth having.  `-t` on a movie, an image or a binary
file are three separate messages, and each one is the difference between the
user being told why their file will not open and the text viewer showing them
a screenful of mojibake.
"""

import os
import re
import subprocess
import sys
import tempfile
import time
import unittest

GUI = os.environ.get("SPARTA_GUI", "build-gui/sparta-gui")
PLUGIN = os.environ.get("SPARTA_PLUGIN_LIB", "")
FIXTURES = os.environ.get(
    "SPARTA_FIXTURES",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures"))
# A display of its own.  test_gui_walker.py already uses :57, and with the two
# suites on one server this one listed that one's windows as its own -- which
# reads as "the second image opened a window of its own" for a working run.
DISPLAY = ":91"


def sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


class App:
    """One run of the application, with an X server and a profile of its own.

    A run that is expected to keep going (a viewer window) is stopped by this
    class; a run that is expected to exit on its own is waited for.  Either way
    the exit status and the list of windows it managed to map are available
    afterwards.
    """

    def __init__(self, args, dismiss=0, settle=6.0, preseed_plugin=True):
        self.args = list(args)
        self.preseed_plugin = preseed_plugin
        self.dismiss = dismiss   # how many modals to answer with Return
        self.settle = settle
        self.profile = tempfile.mkdtemp()
        self.windows = []
        self.before = []         # what was on screen before anything was dismissed
        self.returncode = None
        self.output = ""
        self.stored = ""

    def __enter__(self):
        env = dict(os.environ)
        env["DISPLAY"] = DISPLAY
        env["XDG_CONFIG_HOME"] = f"{self.profile}/config"
        env["XDG_DATA_HOME"] = f"{self.profile}/data"
        cfgdir = f"{env['XDG_CONFIG_HOME']}/The SPARTA Developers"
        os.makedirs(cfgdir, exist_ok=True)
        os.makedirs(env["XDG_DATA_HOME"], exist_ok=True)
        self.cfgfile = f"{cfgdir}/SPARTA-GUI (QT6).conf"
        # No welcome screen and no session restore, so what comes up is what
        # the command line asked for and nothing else.
        with open(self.cfgfile, "w") as f:
            f.write("[General]\nshowwelcome=false\nrestore_session=false\n")
            if PLUGIN and self.preseed_plugin:
                f.write(f"plugin_path={PLUGIN}\n")
        self.env = env

        self.log = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
        self.proc = subprocess.Popen([GUI] + self.args, env=env,
                                     stdout=self.log, stderr=subprocess.STDOUT)
        time.sleep(self.settle)

        # what the run put up, before a keystroke changes it: the title of the
        # refusal box is the only thing that says *which* refusal it was, and
        # three of them exit 1 alike
        self.before = self._window_names()

        for _ in range(self.dismiss):
            if self.proc.poll() is not None:
                break
            sh("xdotool", "key", "--clearmodifiers", "Return", env=env)
            time.sleep(1.5)

        self.windows = self._window_names()
        return self

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
            self.returncode = None     # it was still running: it did not exit
        else:
            self.returncode = self.proc.returncode
        self.log.close()
        with open(self.log.name) as f:
            self.output = f.read()
        os.unlink(self.log.name)
        # read the settings before the profile goes: the caller only gets to
        # look after the run has finished
        try:
            with open(self.cfgfile) as f:
                self.stored = f.read()
        except OSError:
            self.stored = ""
        import shutil
        shutil.rmtree(self.profile, ignore_errors=True)

    def _window_names(self):
        """The windows this run mapped, by process id.

        Searching by name would pick up anything else on the same server, so
        the pid is what makes the answer about this run and no other.
        """
        out = []
        ids = sh("xdotool", "search", "--pid", str(self.proc.pid), env=self.env).stdout.split()
        if not ids:
            ids = sh("xdotool", "search", "--name", ".", env=self.env).stdout.split()
        for wid in ids:
            name = sh("xdotool", "getwindowname", wid, env=self.env).stdout.strip()
            geo = sh("xdotool", "getwindowgeometry", wid, env=self.env).stdout
            m = re.search(r"Geometry: (\d+)x(\d+)", geo)
            # the tiny unmapped helper windows xdotool also returns are noise
            if name and m and int(m.group(1)) * int(m.group(2)) > 10000:
                out.append(name)
        return out

    def showed(self, needle):
        return any(needle in w for w in self.windows + self.before)

    def settings(self):
        return self.stored


class Startup(unittest.TestCase):
    """Each case is one command line."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GUI):
            raise unittest.SkipTest(f"no application to run: {GUI}")
        cls.xvfb = subprocess.Popen(
            ["Xvfb", DISPLAY, "-screen", "0", "1024x768x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        cls.tmp = tempfile.mkdtemp()

        cls.deck = os.path.join(cls.tmp, "in.test")
        with open(cls.deck, "w") as f:
            f.write("dimension 2\nrun 0\n")

        # a data file the chart mode can read
        cls.data = os.path.join(cls.tmp, "series.dat")
        with open(cls.data, "w") as f:
            f.write("# Step Temp\n")
            for i in range(20):
                f.write(f"{i*10} {300.0 + i}\n")

        # and one it cannot
        cls.junk = os.path.join(cls.tmp, "notdata.dat")
        with open(cls.junk, "w") as f:
            f.write("this file holds no columns of numbers at all\n")

        # a real image, written by the toolkit rather than hand-rolled
        cls.image = os.path.join(cls.tmp, "shot.png")
        sh("convert", "-size", "120x90", "xc:seagreen", cls.image)
        if not os.path.exists(cls.image):
            for cand in (os.path.join(FIXTURES, "snapshot.png"),):
                if os.path.exists(cand):
                    cls.image = cand

        # a binary file that is neither image nor movie
        cls.binary = os.path.join(cls.tmp, "restart.bin")
        with open(cls.binary, "wb") as f:
            f.write(bytes(range(256)) * 8)

        # a name that looks like a movie; -t must refuse it on the name alone
        cls.movie = os.path.join(cls.tmp, "clip.mp4")
        with open(cls.movie, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)

    @classmethod
    def tearDownClass(cls):
        cls.xvfb.terminate()
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ------------------------------------------------------ the plain options

    def testVersionExitsWithTheVersion(self):
        r = sh(GUI, "--version", env={**os.environ, "DISPLAY": DISPLAY})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertRegex(r.stdout, r"\d+\.\d+", "no version number in the output")

    def testHelpListsEveryDocumentedOption(self):
        r = sh(GUI, "--help", env={**os.environ, "DISPLAY": DISPLAY})
        self.assertEqual(r.returncode, 0, r.stderr)
        text = r.stdout + r.stderr
        for opt in ("--width", "--height", "--style", "--chart", "--image", "--text"):
            self.assertIn(opt, text, f"{opt} is not in the help output")
        self.assertIn("SPARTA-GUI", text)

    def testHelpNamesTheConfiguredPluginPath(self):
        # the description is built from the stored setting, so this is also the
        # only way to see from outside which library the application would load
        r = sh(GUI, "--help", env={**os.environ, "DISPLAY": DISPLAY})
        text = r.stdout + r.stderr
        self.assertIn("plugin path", text.lower())

    def testThePluginPathOptionIsRemembered(self):
        if not PLUGIN or not os.path.exists(PLUGIN):
            self.skipTest("no SPARTA library to point at")
        # Start from a profile that has no library recorded, or "it is in the
        # settings afterwards" would be true whether or not -p did anything.
        #
        # And with a real deck rather than --help: QCommandLineParser::process()
        # prints the help and exits before any of main()'s own option handling
        # runs, so -p --help would never reach the code being tested.
        with App(["-p", PLUGIN, self.deck], preseed_plugin=False) as app:
            self.assertTrue(app.windows, f"no window at all: {app.output[-800:]}")
        self.assertIn("plugin_path", app.settings(),
                      "-p did not record the library it was given")
        self.assertIn(os.path.realpath(PLUGIN), app.settings(),
                      f"-p stored something else: {app.settings()!r}")

    # -------------------------------------------------------------- -t / text

    def testTextModeOpensAViewerOnAnOrdinaryFile(self):
        with App(["-t", self.deck]) as app:
            self.assertTrue(app.showed("in.test") or app.showed("SPARTA-GUI"),
                            f"no viewer window: {app.windows} {app.output[-800:]}")
        self.assertIsNone(app.returncode, "the text viewer exited instead of staying up")

    def testTextModeRefusesAMovieByName(self):
        # by name, before anything is read: the point of the separate message is
        # to send the user to -i, which "this looks like a binary file" does not
        with App(["-t", self.movie], dismiss=1) as app:
            pass
        self.assertEqual(app.returncode, 1,
                         f"a movie was accepted as text: {app.before} {app.output[-800:]}")
        self.assertTrue(app.showed("Movie"),
                        f"it refused the movie for the wrong reason: {app.before}")

    def testTextModeRefusesAnImage(self):
        with App(["-t", self.image], dismiss=1) as app:
            pass
        self.assertEqual(app.returncode, 1,
                         f"an image was accepted as text: {app.before} {app.output[-800:]}")
        self.assertTrue(app.showed("Image"),
                        f"it refused the image for the wrong reason: {app.before}")

    def testTextModeRefusesABinaryFile(self):
        # not by extension -- .bin means nothing -- but by what is in it
        with App(["-t", self.binary], dismiss=1) as app:
            pass
        self.assertEqual(app.returncode, 1,
                         f"a binary file was shown as text: {app.before}")
        self.assertTrue(app.showed("Binary"),
                        f"it refused the file for the wrong reason: {app.before}")

    # ------------------------------------------------------------- -i / image

    def testImageModeOpensTheSnapshotViewer(self):
        with App(["-i", self.image]) as app:
            self.assertTrue(app.windows, f"no window at all: {app.output[-800:]}")
        self.assertIsNone(app.returncode, "the snapshot viewer exited instead of staying up")

    def testImageModeWithTwoFilesShowsThemBoth(self):
        # -i is repeatable, and the second file is added to the same viewer
        # rather than opening a window of its own
        second = os.path.join(self.tmp, "shot2.png")
        sh("convert", "-size", "120x90", "xc:tomato", second)
        if not os.path.exists(second):
            self.skipTest("no second image to add")
        with App(["-i", self.image, "-i", second]) as app:
            named = [w for w in app.windows if "SPARTA-GUI" in w]
            self.assertEqual(len(named), 1,
                             f"the second image opened a window of its own: {app.windows}")

    # -------------------------------------------------------------- -c / chart

    def testChartModeAsksWhichColumnsToPlot(self):
        # A readable data file goes through the column-picker dialog first, and
        # that dialog is the assertion: what happens after it depends on which
        # button the keystroke lands on, but it must be offered at all -- a
        # chart mode that plotted straight away would pick the columns for you.
        with App(["-c", self.data]) as app:
            self.assertTrue(app.showed("Select Data") or app.showed("Plot")
                            or app.showed("SPARTA-GUI"),
                            f"no window at all: {app.windows} {app.output[-800:]}")
            picked = list(app.windows)
        self.assertTrue(picked, "the chart mode came up with nothing on screen")

    def testChartModeRefusesAFileWithNoDataInIt(self):
        with App(["-c", self.junk], dismiss=1) as app:
            pass
        self.assertEqual(app.returncode, 1,
                         f"a file with no columns was plotted: {app.before}")
        self.assertTrue(app.showed("Plot Data"), f"no explanation was offered: {app.before}")

    def testChartModeRefusesAFileThatIsNotThere(self):
        with App(["-c", os.path.join(self.tmp, "nosuch.dat")], dismiss=1) as app:
            pass
        self.assertEqual(app.returncode, 1, "a missing file was plotted")

    # ------------------------------------------------------------ the default

    def testAPositionalArgumentOpensTheEditor(self):
        with App([self.deck]) as app:
            self.assertTrue(app.showed("in.test"),
                            f"the deck is not in any window title: {app.windows}")
        self.assertIsNone(app.returncode)

    def testTheWindowSizeOptionsAreHonoured(self):
        # deliberately not near the default size: a window that ignored these
        # and came up at its usual size would satisfy a loose lower bound
        with App(["-x", "900", "-y", "640", self.deck]) as app:
            geo = ""
            ids = sh("xdotool", "search", "--name", "SPARTA-GUI", env=app.env).stdout.split()
            best = 0
            for wid in ids:
                out = sh("xdotool", "getwindowgeometry", wid, env=app.env).stdout
                m = re.search(r"Geometry: (\d+)x(\d+)", out)
                if m and int(m.group(1)) * int(m.group(2)) > best:
                    best = int(m.group(1)) * int(m.group(2))
                    geo = f"{m.group(1)}x{m.group(2)}"
            self.assertTrue(geo, "no main window to measure")
            w, h = (int(v) for v in geo.split("x"))
            # 900x640 is above the layout's own minimum (a smaller request is
            # clamped up, which would pass a loose bound whatever main() did)
            # and unlike the size a window with no -x/-y comes up at
            self.assertAlmostEqual(w, 900, delta=40, msg=f"the window came up {geo}")
            self.assertAlmostEqual(h, 640, delta=40, msg=f"the window came up {geo}")


##############################
if __name__ == "__main__":
    unittest.main(verbosity=2)
