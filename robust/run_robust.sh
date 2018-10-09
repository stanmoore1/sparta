#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=standard
####SBATCH --partition=knl
####SBATCH -C quad,cache
#SBATCH --time=00:30:00
#SBATCH --job-name=run_robust

# DataWarp striped scratch allocation
#DW persistentdw name=STAMOORS_1
export CHECKPOINT_READ_DIR=$DW_PERSISTENT_STRIPED_STAMOORS_1/restart
export CHECKPOINT_WRITE_DIR=$DW_PERSISTENT_STRIPED_STAMOORS_1/restart
export PFS_STAGEOUT_DIR=./restart_$SLURM_JOB_ID
mkdir -p $PFS_STAGEOUT_DIR

####export CHECKPOINT_READ_DIR=./restart_419725

if [ -f ./NO_DW ]; then
  export CHECKPOINT_READ_DIR=`python choose_latest.py`
  if [ "$CHECKPOINT_READ_DIR" == "failed" ]; then
    echo Stopping to prevent total checkpoint loss
    exit 1
  fi
  export CHECKPOINT_WRITE_DIR=$PFS_STAGEOUT_DIR
fi

echo $CHECKPOINT_READ_DIR
echo $CHECKPOINT_WRITE_DIR
mkdir -p $CHECKPOINT_WRITE_DIR
####rm -rf $CHECKPOINT_READ_DIR/*

module list
ulimit -c unlimited

NODES=1
MPIPNODE=64
THREADS=1

EXE=../src/spa_kokkos_omp

INPUT_START=in.collide_start
INPUT=in.collide_restart

NFILES=64

export OMP_PLACES=threads
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=4

echo !!!!!!!!! Restarted $i times !!!!!!!!!
RESTART_NUMBER=`python choose_restart.py ${CHECKPOINT_READ_DIR}`
echo Picked Restart $RESTART_NUMBER

if [ $RESTART_NUMBER -eq -2 ]
then
  touch ./NO_DW
  export CHECKPOINT_READ_DIR=`python choose_latest.py`
  if [ "$CHECKPOINT_READ_DIR" == "failed" ]; then
    echo Stopping to prevent total checkpoint loss
    exit 1
  fi
  export CHECKPOINT_WRITE_DIR=$PFS_STAGEOUT_DIR
  echo Switching To Lustre
  RESTART_NUMBER=`python choose_restart.py ${CHECKPOINT_READ_DIR}`
  echo Picked Restart $RESTART_NUMBER
fi

if [ $RESTART_NUMBER -eq -1 ]
then
  #rm -rf $CHECKPOINT_READ_DIR/*
  # write out initial restart file
  time -p srun -u --cpu_bind=cores -n $(($NODES*$MPIPNODE)) --ntasks-per-node ${MPIPNODE} ${EXE} -in ${INPUT_START} -k on t ${THREADS} -sf kk -pk kokkos comm classic -v DIR_READ ${CHECKPOINT_READ_DIR} -v DIR_WRITE ${CHECKPOINT_WRITE_DIR} -v NFILES ${NFILES}
  RESTART_NUMBER=1
fi

time -p srun -u --cpu_bind=cores -n $(($NODES*$MPIPNODE)) --ntasks-per-node ${MPIPNODE} ${EXE} -in ${INPUT} -k on t ${THREADS} -sf kk -pk kokkos comm classic -v DIR_READ ${CHECKPOINT_READ_DIR} -v DIR_WRITE ${CHECKPOINT_WRITE_DIR} -v RESTART_NUMBER ${RESTART_NUMBER} -v NFILES ${NFILES}

