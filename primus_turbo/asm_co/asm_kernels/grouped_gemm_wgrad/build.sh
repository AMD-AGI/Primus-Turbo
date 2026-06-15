#!/usr/bin/env bash
# build.sh -- Compile patch .s files into final.co
#
# Kernel:   grouped_variable_k_dot_scaled_kernel
# Base:     base.co  = dot_scaled_v2 (ggemm-asm load-hoisting optimization applied)
#                      built from: patch_co.py dot_scaled_compiled.co dot_scaled_v2.s
#
# Two-step patch chain on top of the v2 load-hoisted base:
#   base.co  --(patch_cbsz_fix.s)--> intermediate.co  --(patch_beta1.s)--> final.co
#
#   patch_cbsz_fix.s  -- fixes cbsz:1 -> cbsz:0 on all 16 MFMA instructions
#                        (corrects FP8 OCP E4M3 systematic 1/8 scaling error)
#   patch_beta1.s     -- adds .L_BETA1 trampoline before s_endpgm for fused
#                        gradient accumulation (beta=1)
#
# Full optimization stack in final.co:
#   1. RHS buffer_load hoisting    (+0.8%)  from dot_scaled_v2.s
#   2. cbsz:0 fix                  (2x correctness, ~0%)  from patch_cbsz_fix.s
#   3. beta=1 accumulation trampoline       from patch_beta1.s
#
# To rebuild base.co from scratch:
#   python3 ../tools/patch_co.py \
#     ../../ggemm-asm/kernels/dot_scaled_compiled.co \
#     ../../ggemm-asm/kernels/dot_scaled_v2.s \
#     base.co --llvm-mc /opt/rocm/llvm/bin/llvm-mc
#
# Usage:
#   bash build.sh                # build both patches -> final.co
#   bash build.sh --cbsz-only    # stop after cbsz fix (intermediate.co only)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS="${SCRIPT_DIR}/../tools"
PATCH_CO="${TOOLS}/patch_co.py"
LLVM_MC="${LLVM_MC:-/opt/rocm/llvm/bin/llvm-mc}"

BASE="${SCRIPT_DIR}/base.co"
PATCH1="${SCRIPT_DIR}/patch_cbsz_fix.s"
PATCH2="${SCRIPT_DIR}/patch_beta1.s"
INTER="${SCRIPT_DIR}/intermediate.co"
FINAL="${SCRIPT_DIR}/final.co"

[[ -f "$PATCH_CO" ]] || { echo "ERROR: patch_co.py not found: $PATCH_CO" >&2; exit 1; }
[[ -f "$BASE" ]]     || { echo "ERROR: base.co not found: $BASE" >&2; exit 1; }
[[ -f "$PATCH1" ]]   || { echo "ERROR: patch_cbsz_fix.s not found: $PATCH1" >&2; exit 1; }
[[ -f "$PATCH2" ]]   || { echo "ERROR: patch_beta1.s not found: $PATCH2" >&2; exit 1; }

echo "=== grouped_gemm_wgrad build ==="
echo "  base:   base.co"
echo "  patch1: patch_cbsz_fix.s"
echo "  patch2: patch_beta1.s"
echo ""

echo "--- Step 1: patch_cbsz_fix.s -> intermediate.co ---"
python3 "$PATCH_CO" "$BASE" "$PATCH1" "$INTER" --llvm-mc "$LLVM_MC"
echo "  -> intermediate.co"
echo ""

[[ "${1:-}" == "--cbsz-only" ]] && {
    echo "=== Done (--cbsz-only) ==="; echo "  $INTER"; exit 0
}

echo "--- Step 2: patch_beta1.s -> final.co ---"
python3 "$PATCH_CO" "$INTER" "$PATCH2" "$FINAL" --llvm-mc "$LLVM_MC"
echo "  -> final.co"
echo ""

echo "=== Done ==="
echo "  $FINAL"
echo ""
echo "Check:  python3 check.py"
echo "Bench:  python3 bench.py"
