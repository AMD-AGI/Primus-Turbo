#!/usr/bin/env bash
# build.sh -- Compile patch .s files into final.co (swiglu_with_mask_bwd_kernel)
#
# Kernel:   swiglu_with_mask_bwd_kernel
# Base:     base.co  (Triton-extracted)
#
# Two-step patch chain:
#   base.co  --(patch_v1.s)--> intermediate.co  --(patch_v2.s)--> final.co
#
#   patch_v1.s  -- phase-1 optimization: instruction hoisting, scheduling
#   patch_v2.s  -- phase-2 optimization: register pressure, VGPR tuning
#
# Usage:
#   bash build.sh               # build both phases -> final.co
#   bash build.sh --phase1-only # stop after patch_v1.s -> intermediate.co
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS="${SCRIPT_DIR}/../tools"
PATCH_CO="${TOOLS}/patch_co.py"
LLVM_MC="${LLVM_MC:-/opt/rocm/llvm/bin/llvm-mc}"

BASE="${SCRIPT_DIR}/base.co"
PATCH1="${SCRIPT_DIR}/patch_v1.s"
PATCH2="${SCRIPT_DIR}/patch_v2.s"
INTER="${SCRIPT_DIR}/intermediate.co"
FINAL="${SCRIPT_DIR}/final.co"

[[ -f "$PATCH_CO" ]] || { echo "ERROR: patch_co.py not found: $PATCH_CO" >&2; exit 1; }
[[ -f "$BASE" ]]     || { echo "ERROR: base.co not found: $BASE" >&2; exit 1; }
[[ -f "$PATCH1" ]]   || { echo "ERROR: patch_v1.s not found: $PATCH1" >&2; exit 1; }
[[ -f "$PATCH2" ]]   || { echo "ERROR: patch_v2.s not found: $PATCH2" >&2; exit 1; }

echo "=== swiglu_bwd build ==="
echo "  base:   base.co"
echo "  patch1: patch_v1.s  (phase-1)"
echo "  patch2: patch_v2.s  (phase-2)"
echo ""

echo "--- Step 1: patch_v1.s -> intermediate.co ---"
python3 "$PATCH_CO" "$BASE" "$PATCH1" "$INTER" --llvm-mc "$LLVM_MC"
echo "  -> intermediate.co"
echo ""

[[ "${1:-}" == "--phase1-only" ]] && {
    echo "=== Done (--phase1-only) ==="; echo "  $INTER"; exit 0
}

echo "--- Step 2: patch_v2.s -> final.co ---"
python3 "$PATCH_CO" "$INTER" "$PATCH2" "$FINAL" --llvm-mc "$LLVM_MC"
echo "  -> final.co"
echo ""

echo "=== Done ==="
echo "  $FINAL"
echo ""
echo "Check:  python3 check.py final.co"
echo "Bench:  python3 bench.py final.co"
