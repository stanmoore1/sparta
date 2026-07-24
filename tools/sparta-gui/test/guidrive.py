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
EXAMPLE = os.environ.get("SPARTA_EXAMPLE", "examples/circle/in.circle")


def sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


class Gui:
    def __init__(self, display=50, size=(1280, 900), outdir="/tmp/guitest/shots", args=None):
        self.display = f":{display}"
        self.w, self.h = size
        self.outdir = outdir
        self.args = args if args is not None else [EXAMPLE]
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
        plugin = subprocess.run(
            ["find", SPARTA_LIB_DIR, "-name", "libsparta*.so.*"],
            capture_output=True, text=True).stdout.split("\n")[0]
        # preseed so no modal dialog blocks an unattended run
        with open(f"{cfgdir}/SPARTA-GUI (QT6).conf", "w") as f:
            f.write("[General]\n"
                    "showwelcome=false\n"
                    "restore_session=false\n"
                    f"plugin_path={plugin}\n"
                    "examples_path=/home/user/sparta/examples\n")
        self.env = env

        self.xvfb = subprocess.Popen(
            ["Xvfb", self.display, "-screen", "0", f"{self.w}x{self.h}x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
