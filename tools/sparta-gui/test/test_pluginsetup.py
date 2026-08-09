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

"""The first thing a new user sees: SpartaGui::setupPlugin() and SetupCard.

Without a SPARTA shared library the application cannot *run* a deck.  It used
to conclude from that that it could not start either: the constructor put up a
modal offering download, browse or exit, and looped on it, so a user with a
deck to read and no simulator had the choice of fetching one or leaving.

It comes up now, with the editor working and a card above it saying what is
missing and offering the two ways to fix it.  These cases are about that
promise on both sides -- that the application really is usable without a
library, and that acquiring one really does light the rest of it up without a
restart.

Driven as a subprocess through accessibility rather than in-process: the
Preferences route still re-execs, the file chooser is a native modal, and the
buttons are found by name rather than at guessed coordinates.

SPARTA_GUI_FORCE_NO_PLUGIN is what puts the application in the no-library
state.  Without it these cases would depend on the machine not having a
library anywhere the automatic probe looks, which on a developer's machine is
exactly backwards.
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

# The card's buttons, by the text they carry.
DOWNLOAD = "Download"
BROWSE = "Browse"
WHATIS = "What is this?"

# The card has no window of its own, so it is recognized by its heading.
CARD = "No SPARTA shared library yet"


def library():
    """A real libsparta to point the browse dialog at."""
    if SPARTA_PLUGIN_LIB and os.path.exists(SPARTA_PLUGIN_LIB):
        return SPARTA_PLUGIN_LIB
    found = subprocess.run(["find", SPARTA_LIB_DIR, "-name", "libsparta*.so*"],
                           capture_output=True, text=True).stdout.split("\n")[0]
    return found if found and os.path.exists(found) else ""


class PluginSetup(unittest.TestCase):
    """Each case is one way through the no-library state."""

    def setUp(self):
        os.environ["SPARTA_GUI_FORCE_NO_PLUGIN"] = "1"

    def tearDown(self):
        os.environ.pop("SPARTA_GUI_FORCE_NO_PLUGIN", None)

    def gui(self, **kw):
        kw.setdefault("no_library", True)
        kw.setdefault("display", DISPLAY)
        kw.setdefault("outdir", OUT)
        return Gui(**kw)

    def labels(self, g):
        return [n for n, r, *_ in g.controls()]

    def assertCardUp(self, g, why=""):
        names = self.labels(g)
        self.assertTrue(any(CARD in n for n in names),
                        f"the setup card is not on screen{why}: {names}")

    def assertCardGone(self, g, why=""):
        names = self.labels(g)
        self.assertFalse(any(CARD in n for n in names),
                         f"the setup card is still on screen{why}: {names}")

    def assertEditorUp(self, g, why=""):
        titles = g.window_titles()
        self.assertTrue(any("Editor" in t for t in titles),
                        f"the editor is not up{why}: {titles}")

    def buttons(self, g):
        return [n for n, r, *_ in g.controls() if r in ("push button", "button")]

    # ------------------------------------------------------------ what it offers

    def testItComesUpUsableAndSaysWhatIsMissing(self):
        """The whole point: no library is not a reason to refuse to start."""
        with self.gui() as g:
            self.assertEditorUp(g, " on a start with no library")
            self.assertCardUp(g)
            names = self.buttons(g)
            for want in (DOWNLOAD, BROWSE, WHATIS):
                self.assertTrue(any(want in n for n in names),
                                f"no '{want}' button among {names}")
            g.shot("plugin-card", CARD)

    def testEditingWorksWithoutALibraryAndRunningDoesNot(self):
        """The card's claim, checked rather than taken on faith.

        It says decks can be written and saved but not run.  A card that said
        so while Run sat there enabled would be worse than no card: the user
        would press it and find out the hard way.
        """
        with self.gui() as g:
            self.assertCardUp(g)
            g.type_text("# a deck written with no simulator behind it")
            time.sleep(0.5)
            self.assertEditorUp(g, " while being typed into")

            # actions() carries the enabled flag, so this can tell an action the
            # application is offering from one it is refusing -- which is the
            # whole difference the card is claiming.
            state = {name: enabled for name, role, path, enabled in g.actions()}
            # By the entry's own words, not by "Run": the Run *menu* is called
            # that too, and a menu is enabled whatever is inside it.
            offered = [n for n in state if "Run SPARTA from Editor Buffer" in n
                       or "Create Image" in n.replace("&", "")]
            self.assertTrue(offered, f"no Run action to check: {sorted(state)}")
            for name in offered:
                self.assertFalse(state[name],
                                 f"'{name}' is offered while the card says nothing can run")

            saving = [n for n in state if "Save Input File" in n]
            self.assertTrue(saving, f"no Save action to check: {sorted(state)}")
            for name in saving:
                self.assertTrue(state[name],
                                f"'{name}' is refused, but the card says decks can be saved")

            g.shot("plugin-card-run-disabled", "Run greyed out with no library")
            self.assertIsNone(g.app.poll(), "the application died while being typed into")

    # ----------------------------------------------------------------- Browse

    def testCancellingTheBrowseLeavesEverythingAsItWas(self):
        """Changing your mind must not strand the user."""
        with self.gui() as g:
            self.assertCardUp(g)
            self.assertTrue(g.click_named(BROWSE), "no Browse button to press")
            time.sleep(2.0)
            titles = g.window_titles()
            self.assertTrue(any("Select SPARTA shared library" in t or "shared library" in t.lower()
                                for t in titles),
                            f"no file dialog appeared: {titles}")
            g.key("Escape", 2.0)
            self.assertCardUp(g, " after cancelling the file chooser")
            self.assertIsNone(g.app.poll(), "cancelling the file chooser ended the run")

    def testAFileThatIsNotNamedLikeASpartaLibraryIsRefused(self):
        """The file is judged by its name before anything is stored.

        The decoy is a *copy of the real library* under another name, which is
        the only way to tell the check from its absence: a file that would fail
        to load anyway leaves the card up either way.  A library that would
        load can only be refused by the name check.
        """
        lib = library()
        if not lib:
            self.skipTest("no SPARTA library to copy")
        decoy = os.path.join(OUT, "notalibrary.so")
        os.makedirs(OUT, exist_ok=True)
        shutil.copyfile(lib, decoy)

        with self.gui() as g:
            self.assertCardUp(g)
            self.assertTrue(g.click_named(BROWSE))
            time.sleep(2.0)
            g.type_text(decoy)
            g.key("Return", 3.0)

            # The name check explains itself rather than silently bouncing back,
            # so there is a question to answer first.  Escape is the reject role
            # of a Yes/No box -- click_named("No") would match half the
            # accessible names on screen.
            titles = g.window_titles()
            self.assertTrue(any("Unexpected File Name" in t for t in titles),
                            f"the name check said nothing about the file: {titles}")
            g.key("Escape", 1.5)

            self.assertCardUp(g, " after refusing a file whose name is not libsparta*")
            self.assertNotIn("plugin_path", g.stored_settings(),
                             "a file whose name is not libsparta* was recorded anyway")

    def testChoosingALibraryLightsUpTheApplicationInPlace(self):
        """The recovery path, end to end, without a restart.

        Nothing has been loaded on this path, so the chosen library can simply
        be opened and used -- the relaunch the old flow did here was ceremony.
        What has to be true afterwards is that the card is gone, the choice was
        written down, and the application is the same one the user was already
        typing into.
        """
        lib = library()
        if not lib:
            self.skipTest("no SPARTA library to choose")

        with self.gui() as g:
            self.assertCardUp(g)
            before = g.app.pid
            self.assertTrue(g.click_named(BROWSE))
            time.sleep(2.0)
            g.type_text(lib)
            g.key("Return", 4.0)

            deadline = time.time() + 30
            while time.time() < deadline:
                if not any(CARD in n for n in self.labels(g)):
                    break
                time.sleep(1.0)

            self.assertCardGone(g, " after choosing a library that loads")
            self.assertEditorUp(g)
            self.assertEqual(g.app.pid, before,
                             "the application restarted rather than adopting the library")
            g.shot("plugin-recovered")

            stored = g.stored_settings()
            self.assertIn("plugin_path", stored, "the chosen library was not remembered")
            self.assertIn("libsparta", stored["plugin_path"], stored["plugin_path"])

    # --------------------------------------------------------------- Download

    def testADownloadThatFailsSaysSoOnTheCard(self):
        """The other button, on a machine that cannot fetch the file.

        The download itself is not what is being tested -- there is no
        pre-compiled library at that URL for this checkout, so the request
        fails and that is the point.  What has to happen is that the user is
        told, on the card they pressed the button on, and that the application
        carries on.
        """
        with self.gui() as g:
            self.assertCardUp(g)
            self.assertTrue(g.click_named(DOWNLOAD), "no Download button to press")

            deadline = time.time() + 90
            while time.time() < deadline:
                if any("failed" in n.lower() for n in self.labels(g)):
                    break
                time.sleep(0.5)
            names = self.labels(g)
            self.assertTrue(any("failed" in n.lower() for n in names),
                            f"a download that could not be fetched said nothing at all: {names}")
            g.shot("plugin-download-failed")

            self.assertCardUp(g, " after the download attempt")
            self.assertIsNone(g.app.poll(),
                              "a failed download took the application down with it")
            self.assertNotIn("plugin_path", g.stored_settings(),
                             "a download that did not produce a usable library was recorded")

    # ------------------------------------------------- a stored path gone bad

    def testAStoredPathThatNoLongerLoadsIsForgotten(self):
        """An upgrade that moves the library must not wedge the application.

        The stored path is tried first; when it will not load the key has to be
        dropped, or the next start reads the same bad path and asks again.
        """
        decoy = os.path.join(OUT, "libsparta.so.gone")
        os.makedirs(OUT, exist_ok=True)
        with open(decoy, "wb") as f:
            f.write(b"\x7fELF" + b"\x00" * 64)

        # This one needs the real code path rather than the forced one, or the
        # stored setting would never be read at all.
        os.environ.pop("SPARTA_GUI_FORCE_NO_PLUGIN", None)
        with self.gui(settings={"plugin_path": decoy}) as g:
            self.assertEditorUp(g, " for a stored library that does not load")
            self.assertNotIn(decoy, str(g.stored_settings()),
                             "a stored path that will not load was kept, so the next "
                             "start reads it again")


##############################
if __name__ == "__main__":
    unittest.main(verbosity=2)
