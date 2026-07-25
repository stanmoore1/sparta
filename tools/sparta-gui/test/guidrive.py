#!/usr/bin/env python3
"""Drive SPARTA-GUI under Xvfb and capture screenshots of menus and dialogs.

Used to exercise the UI without a desktop and to capture the screenshots the
manual needs (see doc/JPG/README.md for the list still outstanding).

Menu navigation goes through the menus' own mnemonics (Alt+F, then the item's
underlined letter) rather than counting arrow-key presses: item positions shift
whenever a menu is edited, and a miscounted Down lands on the wrong action
silently. Mnemonics are stable and are what a keyboard user would press.

Each captured window is grabbed by its own X window id where one can be found,
so dialogs are cropped to themselves rather than photographed against whatever
happens to be behind them.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

# Point these at a built tree. SPARTA_GUI is what the CMake test harness
# already exports; the others follow the same convention.
GUI = os.environ.get("SPARTA_GUI") or os.environ.get("GUI_BIN", "build-gui/sparta-gui")
SPARTA_LIB_DIR = os.environ.get("SPARTA_LIB_DIR", "build-lib")
# An explicit path to the shared libsparta, which is what the app actually
# needs. Searching a directory for one is the fallback: the GUI's own build
# directory never contains a libsparta, so when that was all this had, the app
# came up with no simulator, every check that needed a run quietly found no
# window to drive, and the suite reported a missing control rather than a
# missing library.
SPARTA_PLUGIN_LIB = os.environ.get("SPARTA_PLUGIN_LIB", "")
# The examples tree, and the deck the suites open by default. Absolute, because
# the application resolves a relative deck against its own working directory
# rather than the harness's: a relative default loaded nothing and left a
# "cannot open file" modal sitting over every screen that followed.
SPARTA_EXAMPLES = os.environ.get(
    "SPARTA_EXAMPLES",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "..", "examples")))
EXAMPLE = os.environ.get("SPARTA_EXAMPLE",
                         os.path.join(SPARTA_EXAMPLES, "circle", "in.circle"))

# Interpreter that can import pyatspi; on this system the default python3 is a
# local build without the gobject bindings, so the distribution one is used.
ATSPI_PYTHON = os.environ.get("ATSPI_PYTHON", "/usr/bin/python3.12")


def sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


class Gui:
    def __init__(self, display=50, size=(1280, 900), outdir="/tmp/guitest/shots", args=None,
                 no_library=False):
        self.display = f":{display}"
        self.w, self.h = size
        self.outdir = outdir
        self.args = args if args is not None else [EXAMPLE]
        # Start with no simulator on purpose. Normally that is a mistake and
        # __enter__ refuses, because every later check would fail for the wrong
        # reason; but the "no suitable SPARTA shared library" dialog is itself a
        # state worth photographing, and this is the only way to reach it.
        self.no_library = no_library
        self.profile = tempfile.mkdtemp()
        os.makedirs(outdir, exist_ok=True)
        self.captured = []

    def __enter__(self):
        env = dict(os.environ)
        env["DISPLAY"] = self.display
        env["XDG_CONFIG_HOME"] = f"{self.profile}/config"
        env["XDG_DATA_HOME"] = f"{self.profile}/data"
        cfgdir = f"{env['XDG_CONFIG_HOME']}/The SPARTA Developers"
        os.makedirs(cfgdir, exist_ok=True)
        os.makedirs(env["XDG_DATA_HOME"], exist_ok=True)
        plugin = ""
        if not self.no_library:
            plugin = SPARTA_PLUGIN_LIB
            if not plugin:
                plugin = subprocess.run(
                    ["find", SPARTA_LIB_DIR, "-name", "libsparta*.so.*"],
                    capture_output=True, text=True).stdout.split("\n")[0]
            if not plugin or not os.path.exists(plugin):
                raise RuntimeError(
                    "no SPARTA library to load: set SPARTA_PLUGIN_LIB (or point "
                    f"SPARTA_LIB_DIR at a build tree; looked in {SPARTA_LIB_DIR!r}). "
                    "Without one the application starts but cannot run anything, "
                    "and every check below would fail for the wrong reason.")
        # Every argument that looks like a path has to exist. A deck that does
        # not puts up a "Cannot open file ... will create new file on saving"
        # modal over the whole window before the harness takes its first
        # screenshot; every later shot then shows that same modal, so a
        # screen-changed comparison against the first one reports no change and
        # the run comes back as "the action did nothing" for a working action.
        for arg in self.args:
            if arg.startswith("-"):
                continue
            if not os.path.exists(arg):
                raise RuntimeError(
                    f"no such file to open: {arg!r}. The application would come "
                    "up behind a modal error dialog and every check below would "
                    "fail for the wrong reason.")
        # preseed so no modal dialog blocks an unattended run
        with open(f"{cfgdir}/SPARTA-GUI (QT6).conf", "w") as f:
            f.write("[General]\n"
                    "showwelcome=false\n"
                    "restore_session=false\n")
            if plugin:
                f.write(f"plugin_path={plugin}\n")
            f.write(f"examples_path={SPARTA_EXAMPLES}\n")
        self.env = env

        self.xvfb = subprocess.Popen(
            ["Xvfb", self.display, "-screen", "0", f"{self.w}x{self.h}x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)

        # A session bus plus QT_ACCESSIBILITY lets the app publish its widget
        # tree over AT-SPI, so buttons can be found by name instead of by
        # guessed coordinates (see guiatspi.py). Harmless if the bus is absent:
        # the app simply does not export accessibility and callers fall back.
        try:
            out = subprocess.run(["dbus-launch"], capture_output=True, text=True).stdout
            for line in out.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k] = v.rstrip(";").strip("'")
            self.dbus = env.get("DBUS_SESSION_BUS_PID")
        except Exception:
            self.dbus = None
        env["QT_ACCESSIBILITY"] = "1"
        env["QT_LINUX_ACCESSIBILITY_ALWAYS_ON"] = "1"
        subprocess.Popen(["/usr/libexec/at-spi-bus-launcher", "--launch-immediately"],
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)

        # A bare X server has no window manager, so nothing ever assigns
        # keyboard focus: the app receives shortcuts only until the first
        # dialog takes focus away, after which every later keystroke goes
        # nowhere and each test silently becomes a no-op. Run a minimal WM so
        # focus behaves the way it does on a real desktop.
        self.wm = subprocess.Popen(["openbox"], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        self.app = subprocess.Popen([GUI] + self.args, env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
        if self.app.poll() is not None:
            raise RuntimeError("app exited during startup")
        self.focus_main()
        return self

    def __exit__(self, *exc):
        alive = self.app.poll() is None
        for p in (self.app, getattr(self, "wm", None), self.xvfb):
            if p is None:
                continue
            try:
                p.terminate(); p.wait(timeout=5)
            except Exception:
                p.kill()
        shutil.rmtree(self.profile, ignore_errors=True)
        self.alive_at_end = alive

    # -- X helpers ---------------------------------------------------------
    def _xdo(self, *args):
        return sh("xdotool", *args, env=self.env)

    def main_window(self):
        """The largest window matching the app name: xdotool also returns
        tiny hidden helper windows that would otherwise be picked first."""
        ids = self._xdo("search", "--name", "SPARTA-GUI").stdout.split()
        best, best_area = None, 0
        for wid in ids:
            g = self._xdo("getwindowgeometry", wid).stdout
            for line in g.split("\n"):
                if "Geometry:" in line:
                    try:
                        w, h = line.split()[1].split("x")
                        a = int(w) * int(h)
                    except ValueError:
                        continue
                    if a > best_area:
                        best_area, best = a, wid
        return best

    def focus_main(self):
        wid = self.main_window()
        if wid:
            self._xdo("windowactivate", wid)
            time.sleep(0.5)

    def key(self, seq, pause=0.8):
        self._xdo("key", "--clearmodifiers", seq)
        time.sleep(pause)

    def shot(self, name, window_name=None):
        """Grab a named window if given (and findable), else the whole root."""
        path = f"{self.outdir}/{name}.png"
        target = None
        if window_name:
            ids = self._xdo("search", "--name", window_name).stdout.split()
            if ids:
                target = ids[-1]
        cmd = ["import", "-window", target or "root", path]
        r = subprocess.run(cmd, env=self.env, capture_output=True, text=True)
        ok = r.returncode == 0 and os.path.exists(path)
        self.captured.append((name, ok, bool(target)))
        return ok

    # -- menu driving ------------------------------------------------------
    def menu(self, menu_key, item_key=None, pause=1.0):
        """Open a menu by its Alt mnemonic, optionally activating an item by
        its own mnemonic letter."""
        self.key(f"alt+{menu_key}", pause)
        if item_key:
            self.key(item_key, pause)

    def controls(self):
        """Every actionable control as (name, role, x, y, w, h).

        Runs the AT-SPI helper under an interpreter that has pyatspi, which is
        not necessarily the one running this harness.
        """
        r = self._atspi("SPARTA")
        out = []
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) == 6:
                out.append((parts[0], parts[1], int(parts[2]), int(parts[3]),
                            int(parts[4]), int(parts[5])))
        return out

    def _atspi(self, *args):
        """Run the accessibility helper under an interpreter that has pyatspi."""
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guiatspi.py")
        return subprocess.run([ATSPI_PYTHON, script, *args],
                              capture_output=True, text=True, env=self.env)

    def actions(self):
        """Every activatable control as (name, role, path, enabled).

        Unlike controls(), this reaches menu items: they are in the
        accessibility tree whether or not their menu is open, and asking a
        control to perform its action does not require it to be on screen.
        Clicking coordinates does, which is why a coordinate-driven sweep can
        only ever reach the few controls that happen to be visible -- and why
        clicking a closed menu's item lands on whatever is underneath it.
        """
        out = []
        for line in self._atspi("actions", "SPARTA").stdout.splitlines():
            parts = line.split("\t")
            if len(parts) == 4:
                out.append((parts[0], parts[1], parts[2], parts[3] == "1"))
        return out

    def activate(self, path, expect_name="", pause=0.6):
        """Perform the default action of the control at @p path."""
        r = self._atspi("do", path, expect_name, "SPARTA")
        time.sleep(pause)
        return r.returncode == 0

    def render_geometry(self, index=-1):
        """ImageMagick crop spec for the largest rendered image on screen.

        Hand-measured rectangles silently clip: an axis or a label drawn just
        outside one makes a working control look like a dead one.
        """
        best = None
        for name, role, x, y, w, h in self.controls():
            if role != "image" or w <= 0 or h <= 0:
                continue
            if best is None or w * h > best[2] * best[3]:
                best = (x, y, w, h)
        if best is None:
            return None
        x, y, w, h = best
        return f"{w}x{h}+{x}+{y}"

    def capture_render(self, path, geom=None):
        """Photograph the rendered image alone, cropped to the render widget.

        The crop always comes off a capture of the *root* window. Accessibility
        reports geometry in screen coordinates, so cropping those coordinates
        out of a per-window capture shifts everything by the window's position:
        the result still looks like a plausible screenshot, which is how a
        working control ends up being reported as dead.
        """
        geom = geom or self.render_geometry()
        if not geom:
            return None
        root = f"{path}.root.png"
        subprocess.run(["import", "-window", "root", root], env=self.env,
                       capture_output=True)
        subprocess.run(["convert", root, "-crop", geom, "+repage", path],
                       capture_output=True)
        if os.path.exists(root):
            os.remove(root)
        return path if os.path.exists(path) else None

    def click_named(self, fragment, role=None, pause=1.2):
        """Click the centre of the control whose accessible name contains @p fragment.

        Returns False when no such control exists, so a test can report a
        missing button rather than clicking empty space and passing.
        """
        for name, r, x, y, w, h in self.controls():
            if role and r != role:
                continue
            if fragment.lower() in name.lower():
                self._xdo("mousemove", str(x + w // 2), str(y + h // 2), "click", "1")
                time.sleep(pause)
                return True
        return False

    def escape(self, n=2):
        for _ in range(n):
            self.key("Escape", 0.4)

    def close_extra_windows(self):
        """Close every mapped window that is not the main one.

        Escape dismisses a modal QDialog but not a top-level tool window such
        as the text viewer, and while one of those holds focus no keystroke
        reaches the main window -- which silently turns every later step of a
        test run into a no-op."""
        main = self.main_window()
        ids = self._xdo("search", "--name", "SPARTA-GUI").stdout.split()
        for wid in ids:
            if wid == main:
                continue
            g = self._xdo("getwindowgeometry", wid).stdout
            # skip the tiny hidden helper windows xdotool also returns
            if "Geometry:" not in g:
                continue
            try:
                w, h = g.split("Geometry:")[1].split()[0].split("x")
                if int(w) * int(h) < 10000:
                    continue
            except (ValueError, IndexError):
                continue
            self._xdo("windowclose", wid)
        time.sleep(0.6)
        self.focus_main()
