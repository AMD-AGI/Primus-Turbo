#!/bin/bash
# 快速验证脚本: 编译 → 精度测试 → 性能 benchmark
# 用法: bash tools/quick_verify.sh [operator]
# 示例:
#   bash tools/quick_verify.sh attention
#   bash tools/quick_verify.sh gemm
#   bash tools/quick_verify.sh grouped_gemm
#   bash tools/quick_verify.sh all

set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

OP="${1:-all}"
FAILED=0

echo "=========================================="
echo " Primus-Turbo Quick Verify"
echo " Operator: $OP"
echo " Time: $(date)"
echo "=========================================="

# --- Step 1: Build ---
echo ""
echo "--- Step 1: Build ---"
pip3 install --no-build-isolation -e . -v 2>&1 | tail -5
if [ $? -ne 0 ]; then
    echo "❌ Build FAILED"
    exit 1
fi
echo "✅ Build OK"

# --- Step 2: Accuracy Tests ---
echo ""
echo "--- Step 2: Accuracy Tests ---"

run_test() {
    local name="$1"
    local cmd="$2"
    echo -n "  $name: "
    if eval "$cmd" > /dev/null 2>&1; then
        echo "✅ PASS"
    else
        echo "❌ FAIL"
        FAILED=$((FAILED + 1))
    fi
}

case "$OP" in
    attention)
        run_test "attention unit" "pytest tests/pytorch/ops/test_attention.py -x -q"
        run_test "attention accuracy" "python3 benchmark/accuracy/eval_attention_accuracy.py"
        ;;
    gemm)
        run_test "gemm unit" "pytest tests/pytorch/ops/test_gemm.py -x -q"
        run_test "gemm fp8 unit" "pytest tests/pytorch/ops/test_gemm_fp8.py -x -q"
        run_test "gemm accuracy" "python3 benchmark/accuracy/eval_gemm_accuracy.py"
        ;;
    grouped_gemm)
        run_test "grouped_gemm unit" "pytest tests/pytorch/ops/test_grouped_gemm.py -x -q"
        run_test "grouped_gemm fp8 unit" "pytest tests/pytorch/ops/test_grouped_gemm_fp8.py -x -q"
        ;;
    moe)
        run_test "moe router unit" "pytest tests/pytorch/ops/test_fused_moe_router.py -x -q"
        run_test "permutation unit" "pytest tests/pytorch/ops/test_permutation.py -x -q"
        ;;
    all)
        run_test "attention" "pytest tests/pytorch/ops/test_attention.py -x -q"
        run_test "gemm" "pytest tests/pytorch/ops/test_gemm.py -x -q"
        run_test "gemm fp8" "pytest tests/pytorch/ops/test_gemm_fp8.py -x -q"
        run_test "grouped_gemm" "pytest tests/pytorch/ops/test_grouped_gemm.py -x -q"
        run_test "moe router" "pytest tests/pytorch/ops/test_fused_moe_router.py -x -q"
        run_test "activation" "pytest tests/pytorch/ops/test_activation.py -x -q"
        run_test "normalization" "pytest tests/pytorch/modules/test_normalization.py -x -q"
        run_test "quantization" "pytest tests/pytorch/ops/test_quantization.py -x -q"
        ;;
    *)
        echo "Unknown operator: $OP"
        echo "Supported: attention, gemm, grouped_gemm, moe, all"
        exit 1
        ;;
esac

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "❌ $FAILED test(s) FAILED. Fix before proceeding to benchmark."
    exit 1
fi

# --- Step 3: Performance Benchmark ---
echo ""
echo "--- Step 3: Performance Benchmark ---"

case "$OP" in
    attention)
        python3 benchmark/ops/bench_attention_turbo.py 2>&1 | tail -20
        ;;
    gemm)
        python3 benchmark/ops/bench_gemm_turbo.py 2>&1 | tail -20
        ;;
    grouped_gemm)
        python3 benchmark/ops/bench_grouped_gemm_turbo.py 2>&1 | tail -20
        ;;
    moe)
        echo "MoE benchmark: run grouped_gemm benchmark as proxy"
        python3 benchmark/ops/bench_grouped_gemm_turbo.py 2>&1 | tail -20
        ;;
    all)
        echo "Running full suite..."
        python3 benchmark/ops/run_suite.py -d output/ 2>&1 | tail -30
        ;;
esac

echo ""
echo "=========================================="
echo " Quick Verify Complete"
echo " Operator: $OP"
echo " Status: ✅ ALL PASSED"
echo "=========================================="
