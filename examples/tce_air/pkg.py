"""Helpers shared by the steps of the tce_air package.

Every step defines a list of runs (Run objects) in its runs.py, in two
sizes: "quick" (CPU, minutes; checks that everything works) and "paper"
(the statistics the paper quotes; meant for GPU nodes).  The same run list
is used to run locally, to write batch job scripts, and to analyze, so the
analysis never depends on where a run happened.

Executables and launch settings come from environment variables, or from
bin/ as written by build.sh:
  SPARTA_EXE        executable of this branch (default bin/spa_release)
  SPARTA_EXE_ARGS   extra arguments, e.g. "-k on g 1 -sf kk" for a GPU build
  SPARTA_CMP_EXE    comparison build with the earlier TCE treatments
                    (react_modify vib_energy dof, elec_energy yes|no|micro);
                    optional (default bin/spa_compare)
  SPARTA_CMP_ARGS   extra arguments of the comparison build
  SPARTA_NP         MPI ranks per run for local CPU runs (default 4)
  SPARTA_MPIRUN     launcher for local runs (default "mpirun")
  TCE_AIR_RESULTS   results directory (default ./results next to this file)
"""
import math
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "bin")
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data"))


def results_dir():
    return os.environ.get("TCE_AIR_RESULTS") or os.path.join(HERE, "results")


# ------------------------------------------------------------------ runs
class Run:
    """one SPARTA run: deck (path relative to the step directory), -var
    settings, executable kind ('release' or 'compare'), and a unique tag.
    files: extra files the deck reads, copied into the run directory."""

    def __init__(self, step, tag, deck, variables, kind="release",
                 files=()):
        self.step, self.tag, self.deck = step, tag, deck
        self.variables, self.kind, self.files = dict(variables), kind, files

    @property
    def stepdir(self):
        return os.path.join(HERE, self.step)

    @property
    def workdir(self):
        return os.path.join(results_dir(), self.step, self.tag)

    @property
    def log(self):
        return os.path.join(self.workdir, "log.sparta")

    def done(self):
        return os.path.exists(self.log) and "Loop time" in open(self.log).read()

    def command(self, launcher=None, exe_override=None):
        """argument list of the SPARTA call (without the MPI launcher)"""
        e, extra = exe_and_args(self.kind)
        if exe_override:
            e = exe_override
        cmd = [e] + extra + ["-in", os.path.join(self.stepdir, self.deck),
                             "-log", self.log, "-screen", "none"]
        for k, v in self.variables.items():
            cmd += ["-var", k, str(v)]
        return cmd


def exe_and_args(kind="release"):
    if kind == "release":
        e = os.environ.get("SPARTA_EXE") or os.path.join(BIN, "spa_release")
        a = os.environ.get("SPARTA_EXE_ARGS", "")
    elif kind == "compare":
        e = os.environ.get("SPARTA_CMP_EXE") or os.path.join(BIN, "spa_compare")
        a = os.environ.get("SPARTA_CMP_ARGS", "")
    elif kind == "precap":
        e = os.environ.get("SPARTA_PRECAP_EXE") or os.path.join(BIN,
                                                                "spa_precap")
        a = os.environ.get("SPARTA_PRECAP_ARGS", "")
    else:
        raise ValueError(kind)
    return e, shlex.split(a)


def have(kind):
    return os.path.exists(exe_and_args(kind)[0])


def prepare(run):
    os.makedirs(run.workdir, exist_ok=True)
    for f in run.files:
        src = f if os.path.isabs(f) else os.path.join(run.stepdir, f)
        dst = os.path.join(run.workdir, os.path.basename(f))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)


def run_local(run, np=None):
    """run one Run on this machine; skip it if its log already finished"""
    if run.done():
        return run.log
    if not have(run.kind):
        raise FileNotFoundError("no '%s' executable: run ./build.sh first"
                                % run.kind)
    prepare(run)
    cmd = [os.environ.get("SPARTA_MPIRUN", "mpirun"), "-np",
           str(np or int(os.environ.get("SPARTA_NP", "4")))] + run.command()
    env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1",
               OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    r = subprocess.run(cmd, cwd=run.workdir, capture_output=True, text=True,
                       env=env)
    if r.returncode != 0 or not run.done():
        sys.stderr.write("run failed: %s\n%s\n%s\n"
                         % (" ".join(map(shlex.quote, cmd)),
                            r.stdout[-2000:], r.stderr[-2000:]))
        raise RuntimeError("SPARTA run failed: %s/%s" % (run.step, run.tag))
    return run.log


def start_gpu(run, gpu):
    """start one run on GPU number gpu of this node, without waiting;
    the launcher is $TCE_AIR_GPU_LAUNCH (default: run the executable
    directly, as a single MPI rank), the GPU is selected through the
    visible-device variables of CUDA and ROCm"""
    prepare(run)
    launch = shlex.split(os.environ.get("TCE_AIR_GPU_LAUNCH", ""))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               ROCR_VISIBLE_DEVICES=str(gpu), HIP_VISIBLE_DEVICES=str(gpu))
    out = open(os.path.join(run.workdir, "stdout"), "w")
    return subprocess.Popen(launch + run.command(), cwd=run.workdir, env=env,
                            stdout=out, stderr=subprocess.STDOUT)


# ------------------------------------------------------------ log parsing
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
    head, rows, cur = None, [], None
    for ln in open(log).read().splitlines():
        w = ln.split()
        if w and w[0] == "Step":
            cur, rows, head = w, [], w
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


def warnings(log):
    return [ln.strip() for ln in open(log) if ln.startswith("WARNING")]


# ------------------------------------------------------------- reporting
class Report:
    """PASS/FAIL lines of one step, written to results/<step>/results.txt"""

    def __init__(self, step):
        self.step, self.lines, self.nfail = step, [], 0
        self.path = os.path.join(results_dir(), step, "results.txt")

    def info(self, text):
        self.lines.append("  " + text)
        print("  " + text)

    def check(self, ok, text):
        line = "  [%s] %s" % ("PASS" if ok else "FAIL", text)
        self.lines.append(line)
        print(line)
        self.nfail += 0 if ok else 1

    def write(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as fh:
            fh.write("%s\n%s\nfailed checks: %d\n"
                     % (self.step, "\n".join(self.lines), self.nfail))
        return self.nfail


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd
