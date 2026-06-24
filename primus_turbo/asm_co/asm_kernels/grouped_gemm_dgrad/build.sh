#!/usr/bin/env bash
# build.sh -- Compile patch .s files into final .hsaco binaries (DGRAD)
#
# Kernel:   _grouped_fp8_persistent_gemm_kernel  (trans_b=True, DGRAD)
# Shapes:
#   down_dgrad:    a[131072, 2880] @ b[32, 2880, 2880]^T -> out[131072, 2880]
#   gate_up_dgrad: a[131072, 5760] @ b[32, 2880, 5760]^T -> out[131072, 2880]
#
# Patch chain (one step per shape):
#   base_down_2880.hsaco     + patch_down_2880.s     -> final_down_2880.hsaco
#   base_gate_up_5760.hsaco  + patch_gate_up_5760.s  -> final_gate_up_5760.hsaco
#
# Usage:
#   bash build.sh                  # build both shapes
#   bash build.sh --down-only      # down_dgrad shape only
#   bash build.sh --gate-up-only   # gate_up_dgrad shape only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS="${SCRIPT_DIR}/../tools"
PATCH_CO="${TOOLS}/patch_co.py"
LLVM_MC="${LLVM_MC:-/opt/rocm/llvm/bin/llvm-mc}"

[[ -f "$PATCH_CO" ]] || { echo "ERROR: patch_co.py not found: $PATCH_CO" >&2; exit 1; }

build_shape() {
    local base="$1" patch="$2" out="$3" label="$4"
    [[ -f "$base" ]]  || { echo "ERROR: base not found: $base" >&2; return 1; }
    [[ -f "$patch" ]] || { echo "ERROR: patch not found: $patch" >&2; return 1; }
    echo "--- $label ---"
    python3 "$PATCH_CO" "$SCRIPT_DIR/$base" "$SCRIPT_DIR/$patch" "$SCRIPT_DIR/$out" --llvm-mc "$LLVM_MC"
    echo "  -> $out"
}

echo "=== grouped_gemm_dgrad build ==="
echo ""

MODE="${1:-all}"

if [[ "$MODE" != "--gate-up-only" ]]; then
    build_shape "base_down_2880.hsaco" "patch_down_2880.s" "final_down_2880.hsaco" \
                "down_dgrad (K=2880, N=2880)"
    echo ""
fi

if [[ "$MODE" != "--down-only" ]]; then
    build_shape "base_gate_up_5760.hsaco" "patch_gate_up_5760.s" "final_gate_up_5760.hsaco" \
                "gate_up_dgrad (K=5760, N=2880)"
    echo ""
fi

echo "=== Done ==="
echo "  Check:  python3 check.py"
echo "  Bench:  python3 bench.py"
