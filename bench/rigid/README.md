# fix rigid benchmark decks

2d decks of N mobile rigid circles in a gas, each circle its own body of
32 line segments, driven by the gas and by push-off contacts with its
neighbours and the box walls.  Every deck runs 40 steps and prints a
stats table whose columns are chosen to be sensitive to the trajectory,
so two runs can be compared bit for bit.

## Getting the data files

The surf and body files are generated, not stored (the 4000-body surf
file is 8.5 MB):

    sh mkdata.sh [/path/to/sparta]

writes `circle.surf`, `nc{1000,2000,4000}.surf`,
`nc{1000,2000,4000}.bodies`, `air.species` and `air.vss` here.  It drives
`tools/rigid/replicate.py`, and is deterministic: the seeds are fixed, so
the same files come out every time.

## The decks

| deck | bodies | grid | particles | what it is for |
|---|---|---|---|---|
| `in.bench40` | 1000 | 1131^2 | 1.8M | the main benchmark, `bodies replicated` |
| `in.bench40.owned` | 1000 | 1131^2 | 1.8M | the same, `bodies owned` |
| `in.w1k.rep` / `.owned` | 1000 | 566^2 | 1.8M | weak series, 1 rank |
| `in.w2k.rep` / `.owned` | 2000 | 800^2 | 3.6M | weak series, 2 ranks |
| `in.w4k.rep` / `.owned` | 4000 | 1132^2 | 7.2M | weak series, 4 ranks |
| `in.w4k.owndist` | 4000 | 1132^2 | 7.2M | as `in.w4k.owned` with `global surfs explicit/distributed` and `gridcut 0.02` |

The `w*` decks hold bodies and particles per rank constant, so they are a
weak series when run at 1, 2 and 4 ranks respectively.  `in.bench40*` is
a fixed problem for strong scaling.

## Running

    export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
    SPARTA_RIGID_TIMING=1 mpirun -np 4 .../spa_mpi -in in.bench40.owned
    SPARTA_RIGID_TIMING=1 mpirun -np 4 .../spa_kokkos_mpi_only -in in.bench40.owned -k on -sf kk

`SPARTA_RIGID_TIMING=1` prints the per-stage table with min/avg/max over
procs, the sub-timers, and the per-run work counters.  Without it the fix
adds no timing overhead.

`mpirun` refuses to run as root without those two variables and produces
an **empty log**, which a naive diff of two empty logs reports as
identical.  Always check the stats table has rows.

## Comparing two runs

    ./st.sh a.log > a.txt; ./st.sh b.log > b.txt; diff a.txt b.txt

`st.sh` prints the stats table with the CPU-time column stripped, so only
the physics is compared.  Every optimization on this branch is required
to leave this empty.
