#!/bin/bash
# 快速 profiling 脚本
# 用法: bash quick_profile.sh <python_script.py> [args...]
# 示例: bash quick_profile.sh benchmark/ops/bench_attention_turbo.py

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <python_script.py> [args...]"
    echo "Example: $0 benchmark/ops/bench_attention_turbo.py"
    exit 1
fi

SCRIPT="$1"
shift
ARGS="$@"

PROJ_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$PROJ_ROOT"

echo "=========================================="
echo " Quick Kernel Profile"
echo " Script: $SCRIPT"
echo " GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo " Time: $(date)"
echo "=========================================="

echo ""
echo "--- Step 1: rocprof kernel stats ---"
if command -v rocprof &> /dev/null; then
    rocprof --stats python3 "$SCRIPT" $ARGS 2>&1 | tail -30
    echo ""
    echo "Full stats in: results.stats.csv"
else
    echo "rocprof not found, skipping."
fi

echo ""
echo "--- Step 2: PyTorch Profiler summary ---"
python3 -c "
import torch
from torch.profiler import profile, ProfilerActivity
import importlib.util, sys

# Quick PyTorch profiler run
print('PyTorch profiler: run the script with torch.profiler for detailed analysis')
print('Example:')
print('  from torch.profiler import profile, ProfilerActivity')
print('  with profile(activities=[ProfilerActivity.CUDA]) as prof:')
print('      # your kernel call')
print('  print(prof.key_averages().table(sort_by=\"cuda_time_total\", row_limit=10))')
"

echo ""
echo "=========================================="
echo " Done. Use omniperf for deeper analysis:"
echo "   omniperf profile -n run -- python3 $SCRIPT"
echo "   omniperf analyze -p workloads/run/"
echo "=========================================="
