#!/bin/bash
# One torchtitan run on this single-GPU box. Adapted from output/0914__campaign/bin/e2e.sh,
# which fenced a 4-GPU box with HIP_VISIBLE_DEVICES=3 at `docker run` time and used a
# separate fa-e2e container -- neither applies here.
#
# compile stays DISABLED. The config's own comment records that inductor's triton autotune
# over TransformerBlock raises hipErrorLaunchFailure and takes the GPU down with it. Do not
# turn it on to chase the documented 1.2-1.5x without a plan for losing the card.
#   e2e.sh <run_tag> <config_basename> [extra primus-cli args...]
set -u
TAG=${1:?run tag}; CFG=${2:?config yaml basename}; shift 2
PRIMUS=/home/lihuzhan/code/2026_0828__primus/Primus
OUT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0915__opt
RUNDIR=/home/lihuzhan/_dbg_l8b/$TAG
mkdir -p "$RUNDIR" "$OUT/logs"
timeout ${E2E_TIMEOUT:-1800} docker exec \
  -e GPU=0 -e HIP_VISIBLE_DEVICES=0 \
  -e PYTHONPATH=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo:/home/lihuzhan/code/aiter-src \
  -e GPUS_PER_NODE=1 -e NNODES=1 -e NODE_RANK=0 -e PRIMUS_GPU_MODEL=MI455X \
  -e PRIMUS_EXP_NAME="$TAG" -e TRITON_CACHE_DIR=/tmp/triton_cache_e2e \
  ${E2E_ENV:-} fa-repro bash -lc "ulimit -c 0; cd $PRIMUS && exec timeout -k 20 ${E2E_INNER:-1700} \
      bash runner/primus-cli direct --log_file $RUNDIR/launcher.log \
      -- train pretrain --config examples/torchtitan/configs/MI455X/$CFG $*" \
  > "$OUT/logs/e2e.$TAG.log" 2>&1
echo "rc=$? tag=$TAG log=$OUT/logs/e2e.$TAG.log"
