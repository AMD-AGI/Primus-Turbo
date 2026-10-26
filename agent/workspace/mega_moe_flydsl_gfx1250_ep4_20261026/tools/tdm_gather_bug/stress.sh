#!/bin/bash
# stress.sh COMMIT|repo ITERS [EXTRA_ARGS]: run $SCRIPT (default stress_ep.py) under torchrun against the repo tree
# (COMMIT=repo) or an overlay tree $V/src_COMMIT; log -> $E/logs/<script>_<COMMIT>_<ITERS>[_TAG].log, then print
# the summary and the first bad iterations.
# Overridable: E (scripts + logs, default this dir), TURBO (repo root), V (overlay trees), CACHE (FlyDSL cache),
# CONTAINER (docker exec target; set it empty to run directly, e.g. inside the container), GPUS, NPROC, LIMIT (s).
HERE=$(cd "$(dirname "$0")" && pwd)
E=${E:-$HERE}
TURBO=${TURBO:-$(cd "$HERE/../../../../.." && pwd)}
V=${V:-/home/xiaompen/mega_scratch/ep4val}
CONTAINER=${CONTAINER-xiaompen_mega_gfx1250_ep4}
C=$1; N=$2; shift 2
if [ "$C" = repo ]; then S=$TURBO; CACHE=${CACHE:-$E/cache}; else S=$V/src_$C; CACHE=${CACHE:-$V/cache_$C}; fi
SCRIPT=${SCRIPT:-stress_ep.py}; LOG=$E/logs/${SCRIPT%.py}_${C}_${N}${TAG:+_$TAG}.log
mkdir -p "$E/logs"
ENVS=(TURBO_ROOT=$S PYTHONPATH=$S TURBO_SRC=$S
      MEGA_IDX_BITS=${MEGA_IDX_BITS:-0} MEGA_B_WAVE=${MEGA_B_WAVE:-1} MEGA_ET=${MEGA_ET:-1} MEGA_SLEEP=${MEGA_SLEEP:-0}
      MEGA_HOLD=${MEGA_HOLD:-0} MEGA_TDM_GATHER=${MEGA_TDM_GATHER:-0} MEGA_A_SPREAD=${MEGA_A_SPREAD:-0}
      FLYDSL_DUMP_IR=${FLYDSL_DUMP_IR:-0} FLYDSL_DUMP_DIR=${FLYDSL_DUMP_DIR:-/tmp/fdump} FORCE_DUMP_IT=${FORCE_DUMP_IT:--1}
      FLYDSL_RUNTIME_CACHE_DIR=$CACHE TRITON_HIP_USE_EXPERT_SCHEDULING=0 HIP_VISIBLE_DEVICES=${GPUS:-0,1,2,3})
RUN="ulimit -c 0; cd $S; $E/capped.sh ${LIMIT:-1800} $LOG torchrun --nproc-per-node ${NPROC:-4} $E/$SCRIPT --iters $N $*"
if [ -n "$CONTAINER" ]; then
    docker exec -w "$S" $(printf -- '-e %s ' "${ENVS[@]}") "$CONTAINER" bash -c "$RUN" >/dev/null
else
    env "${ENVS[@]}" bash -c "$RUN" >/dev/null
fi
echo "== $C: $(grep -E '^T=' "$LOG")"
grep -E "^max rel|^iterations with|BAD|KILLED|^  it |Error" "$LOG" | head -24
