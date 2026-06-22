#!/bin/bash
# Launch the fused BF16 dispatch + grouped GEMM benchmark inside the dev_primus
# container. Default case: DeepSeek-V3, EP8, 8192 tokens/rank (the bench defaults:
# H=7168 I=2048 E=256 topk=8). Any extra args are passed straight through, e.g.:
#   ./run_bench_dispatch_grouped_gemm.sh --mode load_balanced --iters 50
set -euo pipefail

CONTAINER=${CONTAINER:-dev_primus}
REPO=${REPO:-/apps/zhuang12/MegaKernel/Primus-Turbo}
NP=${NP:-8}
PORT=${MASTER_PORT:-8501}

docker exec "$CONTAINER" bash -lc "
  set -e
  cd '$REPO/benchmark/ops'
  ulimit -l unlimited 2>/dev/null || true
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export ROCSHMEM_HEAP_SIZE=8GB PYTORCH_ROCM_ARCH=gfx950
  export MASTER_PORT=$PORT
  export PYTHONPATH='$REPO'
  python bench_dispatch_grouped_gemm.py --num-processes $NP $*
"
