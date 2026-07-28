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

"""The first thing a new user sees: SpartaGui::setupPlugin().

Without a SPARTA shared library the application cannot do anything at all, so
before the main window exists the constructor puts up a dialog offering three
ways out -- download one, browse for one, or give up -- and loops on it until
one of them works.  77 lines, and none of it had ever run under test.

It cannot: every branch ends in exit(1) or in relaunchApplication(), which
replaces the process image outright.  An in-process test would take the whole
test runner with it.  So this drives the real binary as a subprocess and
presses the buttons through accessibility rather than at guessed coordinates,
which is also the only way to press a specific one of the three -- Enter hits
the default (Download, which would go to the network) and Escape hits Exit,
so the middle button is unreachable from the keyboard alone.

What makes this worth the trouble rather than a coverage exercise: this dialog
is the entire first-run experience, it is the one place the application can
strand a user with no way forward, and the recovery path through it -- browse,
pick a library, come back up working -- is a relaunch of the process, which
nothing else in the suite exercises.
"""

import os
import shutil
import subprocess
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guidrive import Gui, SPARTA_PLUGIN_LIB, SPARTA_LIB_DIR   # noqa: E402

DISPLAY = 92
OUT = os.environ.get("SHOT_DIR", "/tmp/guitest/plugin-shots")

# The dialog's three buttons, by the text they carry.
DOWNLOAD = "Download Library"
BROWSE = "Browse Filesystem"
EXIT = "Exit"

DIALOG = "No SPARTA Shared Library"


def library():
    """A real libsparta to point the browse dialog at."""
    if SPARTA_PLUGIN_LIB and os.path.exists(SPARTA_PLUGIN_LIB):
        return SPARTA_PLUGIN_LIB
    found = subprocess.run(["find", SPARTA_LIB_DIR, "-name", "libsparta*.so*"],
                           capture_output=True, text=True).stdout.split("\n")[0]
    return found if found and os.path.exists(found) else ""


class PluginSetup(unittest.TestCase):
    """Each case is one way through the missing-library dialog."""

    def gui(self, **kw):
        kw.setdefault("no_library", True)
        kw.setdefault("display", DISPLAY)
        kw.setdefault("outdir", OUT)
        return Gui(**kw)

    def assertDialogUp(self, g, why=""):
        titles = g.window_titles()
        self.assertTrue(any(DIALOG in t for t in titles),
                        f"the missing-library dialog is not on screen{why}: {titles}")

    def buttons(self, g):
        return [n for n, r, *_ in g.controls() if r in ("push button", "button")]

    # ------------------------------------------------------------ what it offers

    def testItExplainsItselfAndOffersThreeWaysOut(self):
        """A user with no library must be told why, and given something to do."""
        with self.gui() as g:
            self.assertDialogUp(g)
            names = self.buttons(g)
            for want in (DOWNLOAD, BROWSE, EXIT):
                self.assertTrue(any(want in n for n in names),
                                f"no '{want}' button among {names}")
            g.shot("plugin-dialog", DIALOG)
            # and no main window behind it: there is nothing to edit yet
            self.assertFalse(any("Editor" in t for t in g.window_titles()),
                             "the editor came up behind a dialog that says it cannot run")

    # ------------------------------------------------------------------- Exit

    def testExitEndsTheApplicationWithAFailureStatus(self):
        """Giving up has to be a real exit, and a failure one.

        A user who presses Exit and is left with a process still running -- or
        one that reports success -- has no way to tell that nothing happened.
        """
        with self.gui() as g:
            self.assertDialogUp(g)
            self.assertTrue(g.click_named(EXIT), "no Exit button to press")
            deadline = time.time() + 20
            while time.time() < deadline and g.app.poll() is None:
                time.sleep(0.25)
            self.assertIsNotNone(g.app.poll(), "Exit left the application running")
        self.assertEqual(g.app.returncode, 1,
                         "Exit reported success from a run that never started")

    # ----------------------------------------------------------------- Browse

    def testCancellingTheBrowseComesBackToTheDialog(self):
        """Changing your mind must not strand the user.

        The browse branch has no way forward of its own: if a cancelled file
        dialog fell through, the application would sit there with no library,
        no dialog and no main window.
        """
        with self.gui() as g:
            self.assertDialogUp(g)
            self.assertTrue(g.click_named(BROWSE), "no Browse button to press")
            time.sleep(2.0)
            titles = g.window_titles()
            self.assertTrue(any("Select SPARTA shared library" in t or "shared library" in t.lower()
                                for t in titles),
                            f"no file dialog appeared: {titles}")
            g.key("Escape", 2.0)
            self.assertDialogUp(g, " after cancelling the file chooser")
            self.assertIsNone(g.app.poll(), "cancelling the file chooser ended the run")

    def testAFileThatIsNotNamedLikeASpartaLibraryIsRefused(self):
        """The file is judged by its name before anything is stored.

        The decoy is a *copy of the real library* under another name, which is
        the only way to tell the check from its absence: a file that would fail
        to load anyway ends up back at this dialog either way -- stored, tried
        on the relaunch, rejected, forgotten -- so the end state cannot say
        whether the name was ever looked at.  A library that would load can
        only be refused by the name check.
        """
        lib = library()
        if not lib:
            self.skipTest("no SPARTA library to copy")
        decoy = os.path.join(OUT, "notalibrary.so")
        os.makedirs(OUT, exist_ok=True)
        shutil.copyfile(lib, decoy)

        with self.gui() as g:
            self.assertDialogUp(g)
            self.assertTrue(g.click_named(BROWSE))
            time.sleep(2.0)
            g.type_text(decoy)
            g.key("Return", 3.0)
            self.assertDialogUp(g, " after choosing a file whose name is not libsparta*")
            self.assertFalse(any("Editor" in t for t in g.window_titles()),
                             "a file the check should have refused was loaded anyway")
            self.assertNotIn("plugin_path", g.stored_settings(),
                             "a file whose name is not libsparta* was recorded anyway")

    def testChoosingALibraryRestartsTheApplicationWithIt(self):
        """The recovery path, end to end.

        Picking a library stores it and re-execs, because a library cannot be
        loaded cleanly into a process that has already decided it has none.
        What has to be true afterwards is that the editor is up and the choice
        was written down -- otherwise the next start asks again.
        """
        lib = library()
        if not lib:
            self.skipTest("no SPARTA library to choose")

        with self.gui() as g:
            self.assertDialogUp(g)
            self.assertTrue(g.click_named(BROWSE))
            time.sleep(2.0)
            g.type_text(lib)
            g.key("Return", 3.0)

            # the re-exec keeps the process id, so what says it worked is the
            # window: the dialog is gone and the editor is up
            deadline = time.time() + 45
            while time.time() < deadline:
                titles = g.window_titles()
                if any("Editor" in t for t in titles):
                    break
                time.sleep(1.0)
            titles = g.window_titles()
            self.assertTrue(any("Editor" in t for t in titles),
                            f"the application did not come back up with a library: {titles}")
            self.assertFalse(any(DIALOG in t for t in titles),
                             "it came up and asked for a library again")
            g.shot("plugin-recovered")

            stored = g.stored_settings()
            self.assertIn("plugin_path", stored, "the chosen library was not remembered")
            self.assertIn("libsparta", stored["plugin_path"], stored["plugin_path"])

    # --------------------------------------------------------------- Download

    def testADownloadThatFailsSaysSoAndComesBack(self):
        """The third button, on a machine that cannot fetch the file.

        The download itself is not what is being tested -- there is no
        pre-compiled library at that URL for this checkout, so the request
        fails and that is the point.  What has to happen is that the user is
        told, and is returned to the dialog rather than dropped into an
        application with no simulator and no explanation.

        The whole attempt takes well under a second here, so the error box has
        to be waited for rather than sampled: it is modal and stays until it is
        answered, but a poll that starts late enough can still find only the
        window behind it.
        """
        with self.gui() as g:
            self.assertDialogUp(g)
            self.assertTrue(g.click_named(DOWNLOAD), "no Download button to press")

            deadline = time.time() + 90
            while time.time() < deadline:
                if any("Error" in t for t in g.window_titles()):
                    break
                time.sleep(0.5)
            self.assertTrue(any("Error" in t for t in g.window_titles()),
                            "a download that could not be fetched said nothing at all")
            g.shot("plugin-download-failed")

            g.key("Return", 2.5)          # dismiss the error
            self.assertDialogUp(g, " after the download attempt")
            self.assertIsNone(g.app.poll(),
                              "a failed download took the application down with it")
            self.assertNotIn("plugin_path", g.stored_settings(),
                             "a download that did not produce a usable library was recorded")

    # ------------------------------------------------- a stored path gone bad

    def testAStoredPathThatNoLongerLoadsIsForgotten(self):
        """An upgrade that moves the library must not wedge the application.

        The stored path is tried first; when it will not load the key has to be
        dropped, or the next start reads the same bad path and asks again with
        no way to get past it.
        """
        decoy = os.path.join(OUT, "libsparta.so.gone")
        os.makedirs(OUT, exist_ok=True)
        with open(decoy, "wb") as f:
            f.write(b"\x7fELF" + b"\x00" * 64)

        # Recorded rather than contorted around: removing *this* removal alone
        # changes nothing observable, because the loop that shows the dialog
        # removes the key again on every pass ("so we won't get stuck in a loop
        # reading a bad file").  The two are redundant with each other; what is
        # checked here is the behaviour they jointly provide.
        with self.gui(settings={"plugin_path": decoy}) as g:
            self.assertDialogUp(g, " for a stored library that does not load")
            self.assertNotIn("plugin_path", g.stored_settings(),
                             "a stored path that will not load was kept, so the next "
                             "start reads it again")


##############################
if __name__ == "__main__":
    unittest.main(verbosity=2)
