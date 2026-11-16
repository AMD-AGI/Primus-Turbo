#!/bin/bash
# run_e2e_gqa.sh <tag> <model> <cp> <ep> <seq> <ulysses: 1|-1|0> [KEY=VAL ...]
# One Primus/Megatron pretraining run on 8 GPUs, inside the container; log -> $OUT/logs/<tag>.log.
#   ulysses  1: MegaUlysses (fused GemmA2A / A2AGemm)   -1: Ulysses unfused (hipBLASLt + pack + RCCL)
#            0: stock Megatron CP (reference only, different attention path)
# Modes -1 and 1 call the same primus_turbo flash_attn_func on the same local-head shapes.
# Env: TURBO_ROOT (default: this checkout), PRIMUS_ROOT (Primus checkout, branch
# cursor/ulysses-mla-gqa-proxy-configs-43f1), OUT (default: ./e2e_out), E2E_TIMEOUT (s, default 5400).
# Models: qwen3_30B_A3B, glm4_7_4layer_proxy, minimax_m2.5_4layer_proxy, qwen3_235B_A22B_4layer_proxy.
HERE=$(cd "$(dirname "$0")" && pwd)
TURBO_ROOT=${TURBO_ROOT:-$(cd "$HERE/../../../.." && pwd)}
: "${PRIMUS_ROOT:?set PRIMUS_ROOT to the Primus checkout}"
OUT=${OUT:-$PWD/e2e_out}
tag=$1 model=$2 cp=$3 ep=$4 seq=$5 uly=$6
shift 6
for kv in "$@"; do export "$kv"; done
ulimit -c 0
mkdir -p "$OUT/logs" "$OUT/output" "$OUT/shim"
# Primus' launcher puts site-packages ahead of PYTHONPATH; a sitecustomize puts this checkout first.
cat > "$OUT/shim/sitecustomize.py" <<'EOF'
import os
import sys

for _p in reversed([os.environ.get("TURBO_ROOT", "")] + os.environ.get("EXTRA_SYS_PATH", "").split(":")):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
EOF
export TURBO_ROOT PYTHONPATH=$OUT/shim
export EXTRA_SYS_PATH=$PRIMUS_ROOT/third_party/Emerging-Optimizers
export FLYDSL_EXTRA_SOURCE_DIRS=$TURBO_ROOT/primus_turbo/flydsl
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=${PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32:-0}
export PRIMUS_MODEL=$model PRIMUS_CP=$cp PRIMUS_EP=$ep PRIMUS_SEQ_LENGTH=$seq PRIMUS_MAX_POSITION_EMBEDDINGS=$seq
export PRIMUS_TRAIN_ITERS=${PRIMUS_TRAIN_ITERS:-12} PRIMUS_GBS=${PRIMUS_GBS:-16}
export PRIMUS_EXP_NAME=$tag PRIMUS_WORKSPACE=$OUT/output
cd "$PRIMUS_ROOT"
# Official tuned qwen3_30B_A3B-BF16-pretrain.yaml with only MBS/GBS/iters and CP changed (mock data).
timeout "${E2E_TIMEOUT:-5400}" ./runner/primus-cli direct -- train pretrain \
    --config examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-cp4-pretrain.yaml \
    --turbo_ulysses_fused_a2a "$uly" > "$OUT/logs/$tag.log" 2>&1
rc=$?
pkill -9 -f '[t]orchrun'
echo "$tag rc=$rc"
