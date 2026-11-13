"""Helpers shared by the steps: find the SPARTA executables, run a deck,
read the reaction tallies and the stats table of a log, and report
PASS/FAIL lines in one format.

Executables (set by build.sh, or by environment variables):
  SPARTA_EXE       release build of this branch            (required)
  SPARTA_CMP_EXE   comparison build with the earlier TCE treatments
                   (vib_energy dof, elec_energy yes/no/micro); optional
  SPARTA_NP        MPI ranks per run (default 4)
  SPARTA_MPIRUN    MPI launcher (default "mpirun")
"""
import glob
import math
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "bin")


def exe(kind="release"):
    if kind == "release":
        e = os.environ.get("SPARTA_EXE") or os.path.join(BIN, "spa_release")
    elif kind == "compare":
        e = os.environ.get("SPARTA_CMP_EXE") or os.path.join(BIN, "spa_compare")
    elif kind == "precap":
        e = os.environ.get("SPARTA_PRECAP_EXE") or os.path.join(BIN, "spa_precap")
    else:
        raise ValueError(kind)
    return e if os.path.exists(e) else None


def nproc():
    return int(os.environ.get("SPARTA_NP", "4"))


def run(kind, deck, workdir, variables, tag, files=(), np=None):
    """run deck in workdir with -var settings; skip if the log already
    exists and finished (delete the log to rerun). Returns the log path."""
    e = exe(kind)
    if e is None:
        raise FileNotFoundError("no %s executable: run ./build.sh first"
                                % kind)
    os.makedirs(workdir, exist_ok=True)
    log = os.path.join(workdir, "log." + tag)
    if os.path.exists(log) and "Loop time" in open(log).read():
        return log
    cmd = [os.environ.get("SPARTA_MPIRUN", "mpirun"), "-np",
           str(np or nproc()), e, "-in", deck, "-log", log, "-screen", "none"]
    for k, v in variables.items():
        cmd += ["-var", k, str(v)]
    env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1",
               OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                       env=env)
    if r.returncode != 0 or "Loop time" not in open(log).read():
        sys.stderr.write("run failed: %s\n%s\n%s\n"
                         % (" ".join(map(shlex.quote, cmd)), r.stdout[-2000:],
                            r.stderr[-2000:]))
        raise RuntimeError("SPARTA run failed: " + tag)
    return log


def tallies(log):
    """reaction tallies printed at the end of a run: {label: count}"""
    out = {}
    for line in open(log):
        m = re.match(r"\s*reaction (.*): (\d+)\s*$", line)
        if m:
            out[" ".join(m.group(1).split())] = int(m.group(2))
    return out


def stats(log):
    """last stats table of a log: (header list, rows of floats)"""
    lines = open(log).read().splitlines()
    head, rows, cur = None, [], None
    for ln in lines:
        w = ln.split()
        if w and w[0] == "Step":
            cur, rows = w, []
            head = cur
            continue
        if cur is not None:
            if ln.startswith("Loop time"):
                cur = None
                continue
            try:
                rows.append([float(x) for x in w])
            except ValueError:
                pass
    return head, rows


class Report:
    """collects PASS/FAIL lines of one step and writes them to results.txt"""

    def __init__(self, step, path):
        self.step, self.path, self.lines, self.nfail = step, path, [], 0

    def info(self, text):
        self.lines.append("  " + text)
        print("  " + text)

    def check(self, ok, text):
        line = "  [%s] %s" % ("PASS" if ok else "FAIL", text)
        self.lines.append(line)
        print(line)
        self.nfail += 0 if ok else 1

    def write(self):
        with open(self.path, "w") as fh:
            fh.write("%s\n" % self.step)
            fh.write("\n".join(self.lines) + "\n")
            fh.write("failed checks: %d\n" % self.nfail)
        return self.nfail


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd
