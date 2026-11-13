#!/usr/bin/env python3
"""Driver of the tce_air reproduction package.

  ./tce_air.py list    STEP [--size quick|paper]      runs and their status
  ./tce_air.py run     STEP [--size ...] [--group G] [--gpus N] [--shard i/n]
                                                      run here: CPU (MPI) or
                                                      one run per GPU
  ./tce_air.py jobs    STEP --size paper --system slurm|flux [...]
                                                      write batch job scripts
  ./tce_air.py analyze STEP [--size ...]              figures, tables, checks

STEP is a step directory name (e.g. step1_rates) or 'all'.  Runs whose log
already finished are skipped, so 'run' can be repeated after an
interruption.  Results go to results/<step>/ (or $TCE_AIR_RESULTS).
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pkg  # noqa: E402

STEPS = sorted(d for d in os.listdir(HERE)
               if d.startswith("step") and os.path.isdir(os.path.join(HERE, d)))


def load(step, name):
    path = os.path.join(HERE, step, name + ".py")
    spec = importlib.util.spec_from_file_location("%s_%s" % (step, name), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def steps(arg):
    return STEPS if arg == "all" else [arg]


def selected(step, size, group):
    rl = load(step, "runs").runs(size)
    return [r for r in rl if group is None or r.tag.startswith(group + "_")]


def cmd_list(a):
    for st in steps(a.step):
        rl = selected(st, a.size, a.group)
        ndone = sum(r.done() for r in rl)
        print("%s (%s): %d runs, %d done" % (st, a.size, len(rl), ndone))
        if a.verbose:
            for r in rl:
                print("  %-4s %-8s %s" % ("done" if r.done() else "-",
                                          r.kind, r.tag))


def shard(rl, spec):
    """keep every n-th run starting at i, for spec 'i/n'"""
    if not spec:
        return rl
    i, n = (int(x) for x in spec.split("/"))
    return [r for k, r in enumerate(rl) if k % n == i]


def cmd_run(a):
    if a.gpus:
        return cmd_run_gpus(a)
    for st in steps(a.step):
        rl = shard(selected(st, a.size, a.group), a.shard)
        todo = [r for r in rl if not r.done()]
        skip = [r for r in todo if not pkg.have(r.kind)]
        todo = [r for r in todo if pkg.have(r.kind)]
        if skip:
            print("%s: skipping %d runs that need the '%s' executable"
                  % (st, len(skip), "/".join(sorted({r.kind for r in skip}))))
        failed = []
        for i, r in enumerate(todo):
            print("%s: [%d/%d] %s" % (st, i + 1, len(todo), r.tag), flush=True)
            try:
                pkg.run_local(r, np=a.np)
            except RuntimeError:
                failed.append(r.tag)
        if failed:
            print("%s: %d runs FAILED (see their log.sparta): %s"
                  % (st, len(failed), " ".join(failed)))


def cmd_run_gpus(a):
    """run the pending runs of a step concurrently, one run per GPU, as many
    at a time as there are GPUs; each run uses one MPI rank"""
    import time
    pending = []
    for st in steps(a.step):
        rl = shard(selected(st, a.size, a.group), a.shard)
        pending += [r for r in rl if not r.done() and pkg.have(r.kind)]
    print("%d runs on %d GPUs" % (len(pending), a.gpus), flush=True)
    busy = {}
    while pending or busy:
        for g in [g for g in range(a.gpus) if g not in busy]:
            if not pending:
                break
            r = pending.pop(0)
            busy[g] = (r, pkg.start_gpu(r, g), time.time())
            print("GPU %d: start %s/%s" % (g, r.step, r.tag), flush=True)
        time.sleep(2)
        for g, (r, proc, t0) in list(busy.items()):
            if proc.poll() is not None:
                ok = proc.returncode == 0 and r.done()
                print("GPU %d: %s %s/%s (%.0f s)"
                      % (g, "done" if ok else "FAILED", r.step, r.tag,
                         time.time() - t0), flush=True)
                del busy[g]


def cmd_analyze(a):
    nfail = 0
    for st in steps(a.step):
        print("== %s" % st)
        nfail += load(st, "analyze").analyze(a.size)
    sys.exit(min(nfail, 255))


def cmd_jobs(a):
    import jobs
    for st in steps(a.step):
        rl = [r for r in selected(st, a.size, a.group) if not r.done()]
        jobs.write(st, rl, a)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "run", "analyze", "jobs"):
        s = sub.add_parser(name)
        s.add_argument("step")
        s.add_argument("--size", default="quick", choices=("quick", "paper"))
        s.add_argument("--group", default=None)
        if name == "list":
            s.add_argument("-v", "--verbose", action="store_true")
        if name == "run":
            s.add_argument("--np", type=int, default=None,
                           help="MPI ranks per run for CPU runs")
            s.add_argument("--gpus", type=int, default=0,
                           help="run concurrently, one run per GPU")
            s.add_argument("--shard", default=None,
                           help="i/n: only every n-th run starting at i")
        if name == "jobs":
            s.add_argument("--system", required=True,
                           choices=("slurm", "flux"))
            s.add_argument("--gpus-per-node", type=int, default=4)
            s.add_argument("--runs-per-job", type=int, default=16)
            s.add_argument("--hours", type=float, default=2.0)
            s.add_argument("--outdir", default=os.path.join(HERE, "jobs"))
    a = p.parse_args()
    {"list": cmd_list, "run": cmd_run, "analyze": cmd_analyze,
     "jobs": cmd_jobs}[a.cmd](a)


if __name__ == "__main__":
    main()
