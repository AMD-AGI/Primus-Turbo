#!/bin/bash
# One torchtitan run on this single-GPU box. Adapted from output/0914__campaign/bin/e2e.sh,
# which fenced a 4-GPU box with HIP_VISIBLE_DEVICES=3 at `docker run` time and used a
# separate fa-e2e container -- neither applies here.
#
# compile stays DISABLED. The config's own comment records that inductor's triton autotune
# over TransformerBlock raises hipErrorLaunchFailure and takes the GPU down with it. Do not
# turn it on to chase the documented 1.2-1.5x without a plan for losing the card.
# BLAS_ENV overrides the default. hipBLASLt is usable after all -- the image just has it
# mis-pathed (see output/0915__opt/BLAS-FINDING.md), and pointing HIPBLASLT_TENSILE_LIBPATH
# at library/gfx1250/ gives 68.74 TFLOP/s against rocBLAS's 27.64 on an 8192^3 bf16 GEMM.
#
# TORCH_BLAS_PREFER_HIPBLASLT=0 is NOT optional as the default. Without it step 1 dies with
# HIPBLAS_STATUS_INVALID_VALUE out of hipblasLtMatmulAlgoGetHeuristic, because the image is
# missing TensileLibrary_lazy_gfx1250.dat. tune_attention.py sets the same variable itself
# before importing torch, which is why the op-level measurements never saw this and the
# first e2e run did. torch reads it at backend-selection time, so it has to be in the
# environment before the process starts -- it cannot be set from inside the training script.
#   e2e.sh <run_tag> <config_basename> [extra primus-cli args...]
set -u
TAG=${1:?run tag}; CFG=${2:?config yaml basename}; shift 2
PRIMUS=/home/lihuzhan/code/2026_0828__primus/Primus
OUT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0915__opt
RUNDIR=/home/lihuzhan/_dbg_l8b/$TAG
mkdir -p "$RUNDIR" "$OUT/logs"
timeout ${E2E_TIMEOUT:-1800} docker exec \
  -e GPU=0 -e HIP_VISIBLE_DEVICES=0 \
  ${BLAS_ENV:--e TORCH_BLAS_PREFER_HIPBLASLT=0} \
  -e PYTHONPATH=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo:/home/lihuzhan/code/aiter-src \
  -e GPUS_PER_NODE=1 -e NNODES=1 -e NODE_RANK=0 -e PRIMUS_GPU_MODEL=MI455X \
  -e PRIMUS_EXP_NAME="$TAG" -e TRITON_CACHE_DIR=/tmp/triton_cache_e2e \
  ${E2E_ENV:-} fa-repro bash -lc "ulimit -c 0; cd $PRIMUS && exec timeout -k 20 ${E2E_INNER:-1700} \
      bash runner/primus-cli direct --log_file $RUNDIR/launcher.log \
      -- train pretrain --config examples/torchtitan/configs/MI455X/$CFG $*" \
  > "$OUT/logs/e2e.$TAG.log" 2>&1
echo "rc=$? tag=$TAG log=$OUT/logs/e2e.$TAG.log"
