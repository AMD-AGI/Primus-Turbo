#!/bin/bash
# 快速 benchmark attention kernel（仅测关键配置）
# 用法: bash .cursor/skills/triton-attention-optimize/scripts/bench_quick.sh

set -e

PROJ_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$PROJ_ROOT"

echo "=========================================="
echo " Primus-Turbo Attention Quick Benchmark"
echo " GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo " Time: $(date)"
echo "=========================================="

echo ""
echo "--- Accuracy Check ---"
python3 benchmark/accuracy/eval_attention_accuracy.py 2>&1 | tail -5

echo ""
echo "--- Performance Benchmark ---"
python3 benchmark/ops/bench_attention_turbo.py 2>&1 | tail -30

echo ""
echo "=========================================="
echo " Done. Check output CSV for full results."
echo "=========================================="
