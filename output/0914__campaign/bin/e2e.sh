#!/bin/bash
# One end-to-end torchtitan run in the GPU3-fenced container.
#   e2e.sh <run_tag> <config_basename> [extra primus-cli args...]
#
# The container is fenced with HIP_VISIBLE_DEVICES=3 at `docker run` time, so nothing here
# can stray onto a card another stream owns.
set -u
TAG=${1:?run tag}; CFG=${2:?config yaml basename}; shift 2
REPO=/home/lihuzhan/code/2026_0828__primus/Primus
OUT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0914__campaign
RUNDIR=/home/lihuzhan/_dbg_l8b/$TAG
mkdir -p "$RUNDIR"
timeout 3600 docker exec ${E2E_ENV:-} \
  -e GPUS_PER_NODE=1 -e NNODES=1 -e NODE_RANK=0 -e PRIMUS_GPU_MODEL=MI455X \
  -e PRIMUS_EXP_NAME="$TAG" \
  fa-e2e bash -lc "cd $REPO && bash runner/primus-cli direct \
      --log_file $RUNDIR/launcher.log \
      -- train pretrain --config examples/torchtitan/configs/MI455X/$CFG $*" \
  > "$OUT/logs/e2e.$TAG.log" 2>&1
echo "rc=$? tag=$TAG"
