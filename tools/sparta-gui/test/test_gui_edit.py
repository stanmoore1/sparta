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

import os
import pyautogui
import shutil
import subprocess
import tempfile
import time
import unittest
from PIL import Image

pyautogui.PAUSE = 0.25

def tuple_compare(one,two,delta):
    """Compare two RGB triples to determine if they differ less than delta"""
    if abs(one[0]-two[0]) < delta and \
       abs(one[1]-two[1]) < delta and \
       abs(one[2]-two[2]) < delta:
        return True
    else:
        return False

def screenshot(fname):
    """Create a screenshot and save to the given file name fname"""
    return subprocess.check_output(["shooter", fname], stderr=subprocess.STDOUT)

def check_image(fname, color, delta):
    """
    Check if a square of four pixels of an image have the expected color
    with a given delta.  Delete the image if the pixels match.
    """
    with Image.open(fname) as im:
        im = im.convert("RGB")
        px = im.load()
        # check format, size, and that it is all one color
        if tuple_compare(px[60,75], color, delta) and \
           tuple_compare(px[930,75], color, delta) and \
           tuple_compare(px[930,400], color, delta) and \
           tuple_compare(px[60,400], color, delta):
            os.remove(fname)
            return True
        else:
            print(px[60,75], px[930,75], px[930,400], px[60,400], color)
            return False

def focus():
    """Give focus to the editor window by positioning the mouse and clicking"""
    pyautogui.click(button='left', x=100, y=100)

class GUIEditorChecks(unittest.TestCase):
    """This set of tests exercises some basic editor tasks"""

    # handle for SPARTA-GUI process
    gui=None

    def setUp(self):
        """Launch SPARTA-GUI on a scratch profile and give it focus"""
        helptxt = subprocess.check_output([os.environ['SPARTA_GUI'], '--platform', 'offscreen', '-h'],
                                          shell=False, stderr=subprocess.STDOUT).decode()
        # get exact path of the SPARTA-GUI binary from environment variable
        cmdline = [os.environ['SPARTA_GUI'], '-x', '1000', '-y', '500']
        # Point the plugin loader at the library the build found. The name used
        # to be a hardcoded "libsparta.so.0", which is not a soname this
        # project has ever produced: the app came up on a modal "no suitable
        # SPARTA shared library found" dialog, which swallowed every keystroke
        # afterwards, so the exit tests below timed out against a window that
        # was never listening.
        plugin = os.environ.get("SPARTA_PLUGIN_LIB", "")
        if 'pluginpath' in helptxt and plugin:
            cmdline.append('-p')
            cmdline.append(plugin)

        # A scratch profile, so a stored session, a remembered window layout or
        # the first-run welcome screen from an earlier run cannot decide what
        # this one sees.
        self.profile = tempfile.mkdtemp()
        env = dict(os.environ)
        env['XDG_CONFIG_HOME'] = os.path.join(self.profile, 'config')
        env['XDG_DATA_HOME'] = os.path.join(self.profile, 'data')
        cfgdir = os.path.join(env['XDG_CONFIG_HOME'], 'The SPARTA Developers')
        os.makedirs(cfgdir, exist_ok=True)
        os.makedirs(env['XDG_DATA_HOME'], exist_ok=True)
        with open(os.path.join(cfgdir, 'SPARTA-GUI (QT6).conf'), 'w') as f:
            f.write("[General]\nshowwelcome=false\nrestore_session=false\n")
            if plugin:
                f.write("plugin_path=%s\n" % plugin)

        # launch SPARTA-GUI and give it time to come up; a second is not
        # enough on a cold cache, and a test that starts typing at a window
        # that is not there yet fails somewhere else entirely
        self.gui=subprocess.Popen(cmdline, env=env,
                                  stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        time.sleep(4.0)
        for f in ['hello.txt', 'hello1.txt', 'hello2.txt', 'empty.txt', 'complete.txt',
                  'shot1.png', 'shot2.png']:
            if os.path.exists(f):
                os.remove(f)
        focus()

    def exited(self, timeout=20.0):
        """The application's exit status, waiting up to @p timeout for it.

        poll() asked once is a race: the keystroke that ends the run has only
        just been delivered, and under a loaded machine -- four suites at once,
        an instrumented build -- the process has not always finished tearing
        down by the time the next line executes.  A single poll() then reports
        None and the case fails for the machine's speed rather than the
        application's behaviour.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            rc = self.gui.poll()
            if rc is not None:
                return rc
            time.sleep(0.25)
        return self.gui.poll()

    def tearDown(self):
        """Stop SPARTA-GUI"""
        if self.gui.poll() is None:
            self.gui.terminate()
        shutil.rmtree(getattr(self, 'profile', ''), ignore_errors=True)

    def testExitShortcut(self):
        """Exit SPARTA-GUI immediately via keyboard shortcut"""
        screenshot("shot1.png")
        pyautogui.hotkey('ctrl','q')
        screenshot("shot2.png")
        self.assertEqual(self.exited(), 0)
        self.assertTrue(check_image('shot1.png', (255,255,255), 10))
        self.assertTrue(check_image('shot2.png', (0,0,0), 1))

    def testExitMenu(self):
        """Exit SPARTA-GUI immediately via the menu"""
        screenshot("shot1.png")
        pyautogui.hotkey('alt','f')
        pyautogui.press('q')
        screenshot("shot2.png")
        self.assertEqual(self.exited(), 0)
        self.assertTrue(check_image('shot1.png', (255,255,255), 10))
        self.assertTrue(check_image('shot2.png', (0,0,0), 1))

    def testExitModCancelNo(self):
        """Exit SPARTA-GUI with a modified buffer without saving"""
        pyautogui.typewrite("Hello, World!")
        pyautogui.press('enter')
        # cancel on first attempt to exit and we'll drop back to the editor
        pyautogui.hotkey('ctrl','q')
        pyautogui.hotkey('alt','c')
        # try to exit again and this time say "no"
        focus()
        pyautogui.hotkey('ctrl','q')
        pyautogui.hotkey('alt','n')
        self.assertEqual(self.exited(), 0)

    @unittest.skip("unreliable interaction with the modal save dialog; needs rework")
    def testExitModSave(self):
        """Exit SPARTA-GUI with a modified buffer and save it to a file"""
        # First enter some text
        pyautogui.typewrite("Hello, World!")
        pyautogui.press('enter')
        # try to exit and select "yes" to save
        pyautogui.hotkey('ctrl','q')
        pyautogui.hotkey('alt','y')
        # enter file name and save
        pyautogui.typewrite('hello.txt')
        time.sleep(0.2)
        pyautogui.press('enter')
        pyautogui.press('enter')
        time.sleep(0.2)
        self.assertTrue(os.path.exists('hello.txt'))
        with open('hello.txt','r') as f:
            lines = f.read().splitlines()
            self.assertEqual(lines[1],"Hello, World!")
            f.close()
            os.remove('hello.txt')
        self.assertEqual(self.exited(), 0)

    @unittest.skip("unreliable interaction with the modal file dialogs; needs rework")
    def testEditSaveLoad(self):
        """Exercise various Load/Save/Save As/New file options"""
        pyautogui.hotkey('ctrl','a')
        pyautogui.press('delete')
        pyautogui.typewrite("Hello, World!\n")
        pyautogui.typewrite("Hello, SPARTA!\n")
        # save edit to file
        pyautogui.hotkey('ctrl','s')
        pyautogui.typewrite('hello.txt')
        time.sleep(0.2)
        pyautogui.hotkey('enter')
        time.sleep(0.2)
        pyautogui.hotkey('enter')
        self.assertTrue(os.path.exists('hello.txt'))
        with open('hello.txt','r') as f:
            lines = f.read().splitlines()
            self.assertEqual(lines[0],"Hello, World!")
            self.assertEqual(lines[1],"Hello, SPARTA!")
            f.close()

        # start new file and remove comment
        focus()
        pyautogui.hotkey('ctrl','n')
        pyautogui.hotkey('ctrl','a')
        pyautogui.press('delete')
        # save empty buffer to file
        pyautogui.hotkey('ctrl','s')
        pyautogui.typewrite('empty.txt')
        time.sleep(0.2)
        pyautogui.hotkey('enter')
        time.sleep(0.2)
        pyautogui.hotkey('enter')
        self.assertTrue(os.path.exists('empty.txt'))
        with open('empty.txt','r') as f:
            lines = f.read().splitlines()
            self.assertEqual(lines[0],"")
            f.close()
            os.remove('empty.txt')
        focus()
        # abandon buffer and open existing file
        pyautogui.hotkey('ctrl','o')
        pyautogui.typewrite('hello.txt')
        pyautogui.press('enter')
        focus()
        pyautogui.press('down')
        pyautogui.press('down')
        pyautogui.press('down')
        pyautogui.typewrite("Hello, SPARTA-GUI!")
        pyautogui.press('enter')
        focus()
        # update current file on disk
        pyautogui.hotkey('ctrl','s')
        focus()
        time.sleep(0.2)
        # write same content to new name
        pyautogui.hotkey('ctrl','shift','s')
        time.sleep(0.2)
        pyautogui.typewrite('hello1.txt')
        pyautogui.press('enter')
        pyautogui.press('enter')
        time.sleep(0.2)

        # check files and content
        self.assertTrue(os.path.exists('hello.txt'))
        self.assertTrue(os.path.exists('hello1.txt'))

        with open('hello.txt','r') as f:
            lines = f.read().splitlines()
            self.assertEqual(lines[0],"Hello, World!")
            self.assertEqual(lines[1],"Hello, SPARTA!")
            self.assertEqual(lines[2],"Hello, SPARTA-GUI!")
            f.close()
            os.remove('hello.txt')

        with open('hello1.txt','r') as f:
            lines = f.read().splitlines()
            self.assertEqual(lines[0],"Hello, World!")
            self.assertEqual(lines[1],"Hello, SPARTA!")
            self.assertEqual(lines[2],"Hello, SPARTA-GUI!")
            f.close()
            os.remove('hello1.txt')

        # close SPARTA-GUI
        focus()
        pyautogui.hotkey('ctrl','q')
        time.sleep(0.2)
        self.assertEqual(self.exited(), 0)

    @unittest.skip("depends on completion popup timing; needs rework")
    def testEditCompletion(self):
        """Exercise the editor auto-completion popup"""
        # clear buffer
        pyautogui.hotkey('ctrl','a')
        pyautogui.press('delete')
        pyautogui.typewrite('ato')
        pyautogui.press('down')
        pyautogui.press('down')
        pyautogui.press('enter')
        pyautogui.press('space')
        pyautogui.typewrite('mol')
        pyautogui.press('enter')
        pyautogui.press('tab')
        pyautogui.press('enter')
        pyautogui.typewrite('uni')
        pyautogui.press('enter')
        pyautogui.press('space')
        pyautogui.typewrite('met')
        pyautogui.press('enter')
        pyautogui.press('tab')
        pyautogui.press('enter')
        pyautogui.typewrite('reg')
        pyautogui.press('enter')
        pyautogui.press('space')
        pyautogui.typewrite('box bl')
        pyautogui.press('enter')
        pyautogui.typewrite(' 0 1 0 1 0 1')
        pyautogui.press('tab')
        pyautogui.press('enter')

        # save edit to file
        pyautogui.hotkey('ctrl','s')
        pyautogui.typewrite('complete.txt')
        time.sleep(0.2)
        pyautogui.hotkey('enter')
        time.sleep(0.2)
        pyautogui.hotkey('enter')
        self.assertTrue(os.path.exists('complete.txt'))
        with open('complete.txt','r') as f:
            lines = f.read().splitlines()
            self.assertEqual(lines[0],"atom_style      molecular")
            self.assertEqual(lines[1],"units           metal")
            self.assertEqual(lines[2],"region          box block 0 1 0 1 0 1")
            f.close()
            os.remove('complete.txt')
        # close SPARTA-GUI
        focus()
        pyautogui.hotkey('ctrl','q')
        time.sleep(0.2)
        self.assertEqual(self.exited(), 0)

##############################
if __name__ == "__main__":
    unittest.main(verbosity=2)
