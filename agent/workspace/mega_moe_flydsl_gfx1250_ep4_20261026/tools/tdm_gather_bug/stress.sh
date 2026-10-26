#!/bin/bash
# stress.sh COMMIT ITERS [EXTRA_ARGS]: stress_ep.py at EP4 on ep4val/src_COMMIT (repo for COMMIT=repo)
# -> ep4c/logs/stress_COMMIT_ITERS.log; prints the summary and the first bad iterations.
E=/home/xiaompen/mega_scratch/ep4c
V=/home/xiaompen/mega_scratch/ep4val
C=$1; N=$2; shift 2
if [ "$C" = repo ]; then S=/home/xiaompen/Turbo; CACHE=${CACHE:-$E/cache}; else S=$V/src_$C; CACHE=$V/cache_$C; fi
SCRIPT=${SCRIPT:-stress_ep.py}; LOG=$E/logs/${SCRIPT%.py}_${C}_${N}${TAG:+_$TAG}.log
docker exec -w $S -e TURBO_ROOT=$S -e PYTHONPATH=$S -e TURBO_SRC=$S -e MEGA_IDX_BITS=${MEGA_IDX_BITS:-0} -e MEGA_B_WAVE=${MEGA_B_WAVE:-1} -e MEGA_ET=${MEGA_ET:-1} -e MEGA_SLEEP=${MEGA_SLEEP:-0} -e MEGA_HOLD=${MEGA_HOLD:-0} -e FLYDSL_DUMP_IR=${FLYDSL_DUMP_IR:-0} -e FLYDSL_DUMP_DIR=${FLYDSL_DUMP_DIR:-/tmp/fdump} -e FORCE_DUMP_IT=${FORCE_DUMP_IT:--1} -e FLYDSL_RUNTIME_CACHE_DIR=$CACHE \
    -e TRITON_HIP_USE_EXPERT_SCHEDULING=0 -e HIP_VISIBLE_DEVICES=0,1,2,3 xiaompen_mega_gfx1250_ep4 bash -c \
    "ulimit -c 0; /home/xiaompen/mega_scratch/capped.sh 1800 $LOG torchrun --nproc-per-node 4 $E/${SCRIPT:-stress_ep.py} --iters $N $*" \
    >/dev/null
echo "== $C: $(grep -E '^T=' $LOG)"
grep -E "^max rel|^iterations with|BAD|KILLED|^  it |Error" $LOG | head -24
