#!/bin/bash
# test-only: make the OpenMP build run the GPU move dispatch (two-pass + reduce kernels)
set -e
cd "$1"
f=src/KOKKOS/update_kokkos.cpp
sed -i 's/^#if defined SPARTA_KOKKOS_GPU$/#if defined SPARTA_KOKKOS_GPU || defined SPARTA_TEST_FORCE_GPU/' $f
sed -i 's/^\( *\)const int team_size = 128;/#ifdef SPARTA_TEST_FORCE_GPU\n\1const int team_size = 1;\n#else\n\1const int team_size = 128;\n#endif/' $f
sed -i 's/^\( *\)if (can_recheck_two_pass \&\& particle_count >= concurrency) {/&\n\1  if (comm->me == 0 \&\& getenv("SPARTA_TEST_TRACE")) printf("TWOPASS step %ld\\n",(long) update->ntimestep);/' $f
sed -i '1i #include <cstdlib>' $f
k=src/KOKKOS/kokkos.cpp
# line 211-212: CPU branch "comm_serial = 1; atomic_reduction = 0;"
awk 'prev ~ /comm_serial = 1;/ && $0 ~ /^ *atomic_reduction = 0;/ && !done {print "#ifdef SPARTA_TEST_FORCE_GPU"; print "    atomic_reduction = 1;"; print "#else"; print; print "#endif"; done=1; prev=$0; next} {print; prev=$0}' $k > $k.tmp && mv $k.tmp $k
sed -i "1i #define SPARTA_TEST_FORCE_GPU" $f $k
grep -c SPARTA_TEST_FORCE_GPU $f $k
