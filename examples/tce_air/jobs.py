"""Batch job scripts for the GPU nodes.

Each job takes one node and runs one shard of a step's pending runs with
'tce_air.py run STEP --size SIZE --gpus G --shard i/n': G runs at a time,
one per GPU.  Runs are resumable, so a job that hits its time limit can
simply be submitted again.

The machine-specific part (modules, executable, launcher) lives in
hpc/env.sh, which each job sources; copy hpc/env_h100.sh or
hpc/env_mi300a.sh to hpc/env.sh and edit it once.
"""
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

HEAD = {
    "slurm": """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={gpus}
#SBATCH --time={hhmm}:00
#SBATCH --output={out}/{name}.%j.out
# edit or pass on the command line: sbatch -A ACCOUNT -p PARTITION ...
""",
    "flux": """#!/bin/bash
#flux: --job-name={name}
#flux: -N 1
#flux: -t {minutes}m
#flux: --output={out}/{name}.{{{{id}}}}.out
# edit or pass on the command line: flux batch -q QUEUE ...
""",
}

BODY = """
set -e
cd {pkg}
source hpc/env.sh
./tce_air.py run {step} --size {size} --gpus {gpus} --shard {i}/{n}
"""


def write(step, runs, a):
    os.makedirs(a.outdir, exist_ok=True)
    n = max(1, math.ceil(len(runs) / a.runs_per_job))
    h, m = divmod(int(round(a.hours * 60)), 60)
    paths = []
    for i in range(n):
        name = "tce_%s_%d_of_%d" % (step, i, n)
        text = HEAD[a.system].format(name=name, gpus=a.gpus_per_node,
                                     hhmm="%02d:%02d" % (h, m),
                                     minutes=h * 60 + m, out=a.outdir)
        text += BODY.format(pkg=HERE, step=step, size=a.size,
                            gpus=a.gpus_per_node, i=i, n=n)
        path = os.path.join(a.outdir, name + ".sh")
        with open(path, "w") as fh:
            fh.write(text)
        os.chmod(path, 0o755)
        paths.append(path)
    submit = "sbatch" if a.system == "slurm" else "flux batch"
    print("%s: %d pending runs in %d job scripts under %s"
          % (step, len(runs), n, a.outdir))
    print("  submit with:  for f in %s/tce_%s_*.sh; do %s $f; done"
          % (a.outdir, step, submit))
    return paths
