# Runbook for an agent on a GPU node

You are running the reproduction package of the paper *A rate-preserving
microcanonical total collision energy model for discrete internal states
and detailed-balance reverse reactions* on one node with 4 GPUs (NVIDIA
H100 or AMD MI300A). Your job: build, check correctness, run, analyze, and
push results. You do not change the code, the decks, or the analysis. If
something fails, stop and report; do not work around it.

## 0. What you need

- Read access to `https://github.com/stanmoore1/sparta`, branch
  `claude/sparta-backward-rates-r9cb99`.
- Push access to `https://github.com/sparta/overleaf`, only to your
  results branch: `results-h100` or `results-mi300a`. Never push to
  `main`, `paper-artifacts` or any other branch.
- cmake, an MPI C++ compiler, CUDA (H100) or ROCm with Cray MPICH
  (MI300A), Python 3 with numpy, scipy and matplotlib.

## 1. Check out and set up

```bash
git clone --branch claude/sparta-backward-rates-r9cb99 https://github.com/stanmoore1/sparta
cd sparta/examples/tce_air
cp hpc/env_h100.sh hpc/env.sh        # or hpc/env_mi300a.sh on MI300A
# edit the module lines of hpc/env.sh for this site, then:
source hpc/env.sh
```

Record the commit you run: `git rev-parse HEAD`. Every result you push
must name it.

## 2. Build

```bash
./build.sh --compare                  # CPU: bin/spa_release, bin/spa_compare
hpc/build_gpu.sh --exact              # GPU: bin/spa_gpu, bin/spa_gpu_exact
```

`spa_compare` is a CPU build of the earlier TCE treatments; the runs that
need it are small and run on the CPU cores of the node.

## 3. Correctness gates (stop if any fails)

Create the results directory first: `mkdir -p results`.

The GPU code of this branch has not been run on a GPU before. Do not
start production runs until all three gates pass.

**Gate A: physics checks on the GPU.** Every check of the validator,
with the GPU build as the main executable:

```bash
cd ../reverse
python3 validate_reverse.py --exe ../tce_air/bin/spa_gpu --exe-args "-k on g 1 -sf kk" \
    > ../tce_air/results/gate_A_validator_gpu.txt 2>&1; echo "failures: $?"
cd ../tce_air
```

Expected: `0 failures`. A check that needs numpy is skipped if numpy is
missing; install it rather than accept skips.

**Gate B: CPU against GPU.** The same validator with the CPU build first
and the GPU build second, comparing tallies within statistics (GPU
floating point is not bit-identical to CPU):

```bash
cd ../reverse
python3 validate_reverse.py --exe ../tce_air/bin/spa_release \
    --exe2 ../tce_air/bin/spa_gpu --exe2-args "-k on g 1 -sf kk" --parity statistical \
    > ../tce_air/results/gate_B_parity.txt 2>&1; echo "failures: $?"
cd ../tce_air
```

**Gate C: the package itself on GPUs.** Step 1 at the quick size, one run
per GPU, must pass all its checks:

```bash
SPARTA_CMP_EXE=$PWD/bin/spa_compare ./tce_air.py run step1_rates --size quick --gpus 4
./tce_air.py analyze step1_rates --size quick
```

Runs of the comparison build (kind `compare`) start on the CPU even when
`--gpus` is given; that is intended.

If a gate fails: keep the outputs, push them (step 6) with a short
`REPORT.md` that quotes the failing lines, and stop.

## 4. Production runs

For each step, in order, at the paper size:

```bash
./tce_air.py list STEP --size paper
./tce_air.py run  STEP --size paper --gpus 4
./tce_air.py analyze STEP --size paper
```

Runs are resumable: if you are interrupted, run the same command again
and finished runs are skipped. To use the batch system instead of the
interactive node, `./tce_air.py jobs STEP --size paper --system slurm`
(or `flux`) writes job scripts and prints the submit command.

Watch `results/STEP/*/stdout` and `log.sparta` of running jobs. A run
that fails leaves its `log.sparta`; read its last lines, report them, and
continue with the other runs.

## 5. Analysis

`analyze` writes, in `results/STEP/`: the figures (`fig*.pdf`), tables
(`table*.tex`), `numbers.json` (every number the paper quotes from that
step) and `results.txt` (PASS/FAIL checks against independent
references). A FAIL is a finding, not something to fix: report it.

## 6. Push the results

Push only small files: per step `results.txt`, `numbers.json`, `*.pdf`,
`*.tex`, `*.txt`, and every run's `log.sparta` (compressed). Do not push
dumps larger than 5 MB unless a step asks for them.

```bash
git clone --branch results-h100 https://github.com/sparta/overleaf out \
  || (git clone https://github.com/sparta/overleaf out && cd out && git checkout --orphan results-h100 && git rm -rq . && cd ..)
mkdir -p out/$(date +%F)
rsync -a --include='*/' --include='results.txt' --include='numbers.json' \
      --include='*.pdf' --include='*.tex' --include='*.txt' --include='log.sparta' \
      --exclude='*' results/ out/$(date +%F)/
find out/$(date +%F) -name log.sparta -exec gzip -f {} \;
```

Add `out/$(date +%F)/MACHINE.md` with: node type, GPU model and count,
driver and CUDA/ROCm versions, compiler and MPI modules, the sparta commit,
and the commands you ran. Commit with a message that names the steps and
says how many checks passed and failed, then push to your results branch
only.

## 7. Report

Write a short summary for Stan: gates passed or failed, steps run, checks
passed and failed per step, run times, and anything unexpected. Quote
numbers from `results.txt`; do not interpret beyond them.
