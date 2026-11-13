#!/usr/bin/env bash
# GPU correctness and performance check for sparta/sparta#683 and the review fixes.
#
# Builds these variants from the same repo and compares them:
#   base    e071055f  PR base (master before #683)
#   pr      21763d55  #683 head
#   fix1    e7151f27  #683 head + first round of review fixes
#   fix     9ff7c43d  + simplified two-pass move (57cb662c: no mid-move host wait,
#                       RangePolicy second pass, two-pass re-enabled each run) and
#                       verified cleanups (9ff7c43d) -- the candidate
#   fixvol  9ff7c43d with the per-branch "volume" reload in the collision kernel
#           put back.  This checks whether removing it (finding 13) costs anything.
#   mrg     b48f7c2f  9ff7c43d merged with sparta/sparta#700 (bug fixes)
#   pr700   4863e6ec  #700 alone (needs: git fetch https://github.com/sparta/sparta pull/700/head:pr700)
#
# Usage, from the top of a sparta clone that has the branches fetched:
#   git fetch https://github.com/sparta/sparta pull/683/head:pr683
#   git fetch https://github.com/stanmoore1/sparta claude/code-review-pr-683-29lhfg
#   PRESET=kokkos_cuda EXTRA_CMAKE="-DKokkos_ARCH_HOPPER90=ON" ./pr683_gpu_check.sh
#
# Environment:
#   PRESET       kokkos_cuda (default) or kokkos_hip
#   EXTRA_CMAKE  extra cmake args, e.g. a different -DKokkos_ARCH_*=ON
#   RUN          launcher prefix, e.g. "mpirun -np 1" (default: none)
#   KKARGS       SPARTA Kokkos args (default "-k on g 1 -sf kk")
#   SIZES        bench grid edge lengths (default "50 100": 1.25M and 10M particles)
#   REPEAT       timing repeats per case, the minimum is reported (default 3)
#   VARIANTS     subset to build and run (default "base pr fix1 fix fixvol mrg pr700")
#   WORK         scratch dir (default ./pr683_check)

set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
PRESET=${PRESET:-kokkos_cuda}
EXTRA_CMAKE=${EXTRA_CMAKE:-}
RUN=${RUN:-}
KKARGS=${KKARGS:-"-k on g 1 -sf kk"}
SIZES=${SIZES:-"50 100"}
REPEAT=${REPEAT:-3}
VARIANTS=${VARIANTS:-"base pr fix1 fix fixvol mrg pr700"}
WORK=${WORK:-$PWD/pr683_check}
JOBS=${JOBS:-$(nproc --all)}  # nproc alone honours OMP_NUM_THREADS

declare -A REV=([base]=e071055f [pr]=21763d55 [fix1]=e7151f27 [fix]=9ff7c43d [fixvol]=9ff7c43d [mrg]=b48f7c2f [pr700]=4863e6ec)

mkdir -p "$WORK"

# ---------------------------------------------------------------- build

build() {
  local v=$1 src=$WORK/src_$v bld=$WORK/build_$v
  if [ ! -d "$src" ]; then
    git -C "$REPO" worktree add --detach "$src" "${REV[$v]}" >/dev/null
  fi
  if [ "$v" = fixvol ] && ! grep -q "PR683_VOLUME_RELOAD" "$src/src/KOKKOS/collide_vss_kokkos.cpp"; then
    # restore the author's reload of volume in the first recombination branch
    awk '!done && /\/\/react->recomb_density = np \* update->fnum \/ volume;/ {
           print; print "        const double volume = grid_kk_copy.obj.k_cinfo.view_device()[icell].volume / grid_kk_copy.obj.k_cinfo.view_device()[icell].weight; // PR683_VOLUME_RELOAD"; done=1; next } {print}' \
      "$src/src/KOKKOS/collide_vss_kokkos.cpp" > "$src/tmp.cpp"
    mv "$src/tmp.cpp" "$src/src/KOKKOS/collide_vss_kokkos.cpp"
  fi
  if [ ! -x "$bld/src/spa_$PRESET" ] && [ ! -x "$bld/spa_$PRESET" ]; then
    cmake -S "$src/cmake" -B "$bld" -C "$src/cmake/presets/$PRESET.cmake" \
      -DCMAKE_BUILD_TYPE=Release $EXTRA_CMAKE > "$WORK/cmake_$v.log" 2>&1
    cmake --build "$bld" -j "$JOBS" > "$WORK/build_$v.log" 2>&1 \
      || { echo "build $v failed, see $WORK/build_$v.log"; exit 1; }
  fi
}

exe() {
  local bld=$WORK/build_$1
  if [ -x "$bld/src/spa_$PRESET" ]; then echo "$bld/src/spa_$PRESET"; else echo "$bld/spa_$PRESET"; fi
}

for v in $VARIANTS; do echo "building $v (${REV[$v]})"; build "$v"; done

# ---------------------------------------------------------------- inputs

IN=$WORK/inputs
mkdir -p "$IN"
cp "$REPO"/bench/in.free "$REPO"/bench/in.collide "$REPO"/bench/in.sphere \
   "$REPO"/bench/ar.species "$REPO"/bench/ar.vss "$REPO"/bench/data.sphere "$IN"/
cp "$REPO"/examples/chem/air.species "$IN"/air_chem.species
cp "$REPO"/examples/chem/air.vss "$REPO"/examples/chem/air.tce "$IN"/
cp "$REPO"/examples/axi/air.species "$IN"/air_axi.species

# finding 1 and 4: move counters with two-pass mode on (large free-flow problem)
cat > "$IN/in.free3ds" <<'EOF'
seed 12345
dimension 3
global gridcut 0.0 comm/sort yes
boundary rr rr oo
create_box 0 1 0 1 0 1
create_grid 40 40 40
balance_grid rcb cell
species ar.species Ar
mixture air Ar vstream 0.0 0.0 200.0 temp 273.15
global nrho 1.0 fnum 0.0000025
create_particles air n 4000000
fix in emit/face air zlo
timestep 0.00001
compute temp temp
compute rz reduce ave z
stats 100
stats_style step np ntouch nexit nbound ncomm c_rz c_temp
run 500
EOF

# finding 2: axisymmetric radial motion (big enough to reach the two-pass threshold)
cat > "$IN/in.axir" <<'EOF'
seed 12345
dimension 2
global gridcut 0.0 comm/sort yes
boundary p ar p
create_box 0 1 0 1 -0.5 0.5
create_grid 100 100 1
balance_grid rcb cell
global nrho 1.e20 fnum 1.e14
species air_axi.species N2
mixture air N2 vstream 0.0 1000.0 0.0 temp 10.0
create_particles air n 2000000
timestep 2e-6
compute ry reduce ave y
compute rvy reduce ave vy
stats 50
stats_style step np ntouch nbound c_ry c_rvy
run 200
EOF

# finding 3: nearcp + reaction retry widens plist (run with react/retry)
cat > "$IN/in.chemnn3" <<'EOF'
seed 12345
dimension 3
global gridcut 1.0e-5 comm/sort yes
boundary rr rr rr
create_box 0 0.0001 0 0.0001 0 0.0001
create_grid 1 1 1
balance_grid rcb part
species air_chem.species N2 N
mixture air N2 N vstream 0.0 0.0 0.0 temp 60000.0
mixture air N2 frac 1.0
mixture air N frac 0.0
global nrho 7.07043E24
global fnum 7.07043E7
collide vss air air.vss
react tce air.tce
collide_modify nearcp yes 10
create_particles air n 2000 twopass
compute temp temp
stats 1
stats_style step cpu np nattempt ncoll nreact c_temp
timestep 3.0E-8
run 30
EOF

# #700 positive control: particles grow between runs with a custom attribute
#   defined; on GPU, master and #683 can abort ("DualView claimed on both
#   sides") in sync(Device,CUSTOM_MASK); pr700 and mrg must not
cat > "$IN/in.customgrow" <<'EOF'
seed 12345
dimension 3
global gridcut 0.0 comm/sort yes
boundary pp pp pp
create_box 0 1 0 1 0 1
create_grid 10 10 10
balance_grid rcb cell
species ar.species Ar
mixture air Ar temp 273.15
global nrho 1.0 fnum 0.0001
create_particles air n 20000
timestep 0.00001
variable pdvec particle id/100000+3.5
custom particle create dvec float 0 set dvec v_pdvec all NULL
compute s reduce sum p_dvec
stats 10
stats_style step np ntouch c_s
run 20
create_particles air n 60000
run 20
create_particles air n 120000
run 20
EOF

# ---------------------------------------------------------------- correctness

OUT=$WORK/out
mkdir -p "$OUT"
cd "$IN"

echo
echo "== correctness (compare last stats rows across variants; exit status must be 0)"
for t in free3ds axir chemnn3 customgrow; do
  extra=""
  [ "$t" = chemnn3 ] && extra="-pk kokkos react/retry yes react/extra 1.0"
  for v in $VARIANTS; do
    log=$OUT/$t.$v.log
    set +e
    $RUN "$(exe "$v")" $KKARGS $extra -in in.$t -log none > "$log" 2>&1
    st=$?
    set -e
    last=$(awk '/^ *Step/ {h=1; next} /^Loop time/ {h=0} h && $1 ~ /^[0-9]+$/ {row=$0} END {print row}' "$log")
    printf "%-8s %-6s exit=%-3s %s\n" "$t" "$v" "$st" "$last"
  done
done

# ---------------------------------------------------------------- performance

echo
echo "== performance: ms/step of the final 100-step run, min of $REPEAT"
printf "%-10s %6s" case size
for v in $VARIANTS; do printf " %10s" "$v"; done
echo
for b in free collide sphere; do
  for s in $SIZES; do
    printf "%-10s %6s" "$b" "$s"
    for v in $VARIANTS; do
      best=""
      for r in $(seq "$REPEAT"); do
        log=$OUT/bench.$b.$s.$v.$r.log
        $RUN "$(exe "$v")" $KKARGS -var x "$s" -var y "$s" -var z "$s" \
          -in in.$b -log none > "$log" 2>&1 || true   # pr aborts at exit (status 134) after the run
        t=$(grep "Loop time of" "$log" | tail -1 | awk '{printf "%.4f", $4/$9*1000}')
        [ -n "$t" ] || { best=FAIL; break; }
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best=$t; fi
      done
      printf " %10s" "$best"
    done
    echo
  done
done

echo
echo "logs: $OUT"
