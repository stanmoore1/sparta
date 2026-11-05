# source me. Scratch area for binaries/runs (lost on container replacement; rebuilt by ensure_builds.sh)
export FR=/home/user/sparta/.fix-review
export SP=${SP:-/tmp/claude-0/-home-user-sparta/6123e18b-95dc-57b3-8d79-a474c7818aa8/scratchpad}
export BIN=$SP/ab/bin
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 OMPI_MCA_mpi_yield_when_idle=1
export LSAN_OPTIONS=suppressions=$FR/tools/lsan.supp ASAN_OPTIONS=detect_leaks=1
mkdir -p $BIN
