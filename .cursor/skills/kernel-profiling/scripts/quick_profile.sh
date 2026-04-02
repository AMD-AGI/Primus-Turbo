#!/bin/bash
# 快速 profiling 脚本（增强版）
# 用法: bash quick_profile.sh <python_script.py> [args...]
# 示例: bash quick_profile.sh benchmark/ops/bench_attention_turbo.py
#
# 输出:
#   1. rocprof kernel 统计 (CSV)
#   2. PyTorch profiler top-20 kernels
#   3. 简要瓶颈诊断

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

GPU_NAME=$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="profiling_output/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo " Quick Kernel Profile (Enhanced)"
echo " Script: $SCRIPT"
echo " GPU: $GPU_NAME"
echo " Time: $(date)"
echo " Output: $OUTPUT_DIR/"
echo "=========================================="

# --- Step 1: rocprof kernel stats ---
echo ""
echo "--- Step 1: rocprof kernel stats ---"
if command -v rocprof &> /dev/null; then
    rocprof --stats python3 "$SCRIPT" $ARGS 2>&1 | tail -30
    # Move results to output dir
    for f in results.stats.csv results.csv; do
        [ -f "$f" ] && mv "$f" "$OUTPUT_DIR/"
    done
    echo "Stats saved to: $OUTPUT_DIR/results.stats.csv"
else
    echo "rocprof not found, skipping."
fi

# --- Step 2: PyTorch Profiler top-20 ---
echo ""
echo "--- Step 2: PyTorch Profiler summary ---"
python3 -c "
import torch, sys, os

if not torch.cuda.is_available():
    print('No GPU available')
    sys.exit(0)

print('GPU:', torch.cuda.get_device_name(0))
print('HBM:', f'{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print('CUs:', torch.cuda.get_device_properties(0).multi_processor_count)
print()
print('For detailed profiling, run:')
print(f'  rocprof --stats python3 $SCRIPT')
print(f'  omniperf profile -n run -- python3 $SCRIPT')
print(f'  omniperf analyze -p workloads/run/')
"

# --- Step 3: Quick bottleneck hints ---
echo ""
echo "--- Step 3: Bottleneck analysis hints ---"
echo ""
echo "If rocprof stats available, check $OUTPUT_DIR/results.stats.csv:"
echo "  - Sort by TotalDurationNs (descending) to find hottest kernels"
echo "  - Kernels with 'mfma' in name → compute-bound path"
echo "  - High call count + low duration → launch overhead concern"
echo ""
echo "MI300X Roofline reference:"
echo "  BF16 peak: 1307.4 TFLOPS | HBM BW: 5.3 TB/s"
echo "  Balance point: ~247 FLOP/Byte"
echo "  FP8 peak: 2614.9 TFLOPS | Balance: ~494 FLOP/Byte"
echo ""
echo "=========================================="
echo " Done."
echo " For deeper analysis:"
echo "   omniperf profile -n run -- python3 $SCRIPT"
echo "   omniperf analyze -p workloads/run/"
echo "=========================================="
