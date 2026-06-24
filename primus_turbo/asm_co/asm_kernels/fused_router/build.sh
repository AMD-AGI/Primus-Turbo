#!/usr/bin/env bash
# build.sh -- Compile patch_v1.s -> final.co (fused_scaling_group_sum_routing_kernel)
#
# Kernel:   fused_scaling_group_sum_routing_kernel
# Base:     base.hsaco  (Triton-generated reference)
#
# Single-step patch chain:
#   base.hsaco  --(patch_v1.s)--> final.co
#
#   patch_v1.s  -- hand-optimized assembly (softmax + top-k routing)
#
# Usage:
#   bash build.sh             # build final.co
#   bash build.sh --check     # build then run correctness check
#   bash build.sh --bench     # build then run benchmark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS="${SCRIPT_DIR}/../tools"
PATCH_CO="${TOOLS}/patch_co.py"
LLVM_MC="${LLVM_MC:-/opt/rocm/llvm/bin/llvm-mc}"

BASE="${SCRIPT_DIR}/base.hsaco"
PATCH1="${SCRIPT_DIR}/patch_v1.s"
FINAL="${SCRIPT_DIR}/final.co"

[[ -f "$PATCH_CO" ]] || { echo "ERROR: patch_co.py not found: $PATCH_CO" >&2; exit 1; }
[[ -f "$BASE" ]]     || { echo "ERROR: base.hsaco not found: $BASE" >&2; exit 1; }
[[ -f "$PATCH1" ]]   || { echo "ERROR: patch_v1.s not found: $PATCH1" >&2; exit 1; }

echo "=== fused_router build ==="
echo "  base:   base.hsaco"
echo "  patch:  patch_v1.s"
echo ""

python3 "$PATCH_CO" "$BASE" "$PATCH1" "$FINAL" --llvm-mc "$LLVM_MC"
echo "  -> final.co"
echo ""

case "${1:-}" in
    --check)
        echo "--- Correctness check ---"
        python3 "${SCRIPT_DIR}/check.py" --co "$FINAL"
        ;;
    --bench)
        echo "--- Benchmark ---"
        python3 "${SCRIPT_DIR}/bench.py" --co "$FINAL"
        ;;
    *)
        echo "=== Done ==="
        echo "  Check:  python3 check.py --co final.co"
        echo "  Bench:  python3 bench.py --co final.co"
        ;;
esac
