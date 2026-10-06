# AB12 — MPI / bounds-check addendum (closes multi-rank and DEBUG_BOUNDS_CHECK gaps of AB1-AB11)
Binaries: A_mpi = $S/bmpi_base/src/spa_kokkos_omp (e071055f), C_mpi = $S/spa_C3_mpi (== bmpi_fixed, HEAD source f9743740, all fixes incl. follow-ups); both real MPI (OpenMPI) + Kokkos OpenMP with Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON. CPU reference = same binary without -sf kk.
Runner: $S/ab/AB12/r.sh <A|C> <np> <kk1|kk2|kk4|cpu> <deck> <log>  (mpirun --allow-run-as-root --oversubscribe, timeout 90 s, ulimit -v 8 GB). Work dirs $S/ab/AB12/<ID>/.

