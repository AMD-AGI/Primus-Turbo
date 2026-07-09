#!/bin/bash
# Run fused BF16 grouped GEMM + combine bench in dev_primus (extra args passthrough).
set -euo pipefail

# dev_primus is machine-specific; override via CONTAINER=... if named differently.
CONTAINER=${CONTAINER:-dev_primus}
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
NP=${NP:-8}
PORT=${MASTER_PORT:-8503}

docker exec "$CONTAINER" bash -lc "
  set -e
  cd '$REPO/benchmark/ops'
  ulimit -l unlimited 2>/dev/null || true
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export ROCSHMEM_HEAP_SIZE=8GB PYTORCH_ROCM_ARCH=gfx950
  export MASTER_PORT=$PORT
  export PYTHONPATH='$REPO'
  python bench_grouped_gemm_combine.py --num-processes $NP $*
"
