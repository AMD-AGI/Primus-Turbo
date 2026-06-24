#!/usr/bin/env bash
# build.sh -- Build optimized .hsaco for FWD grouped GEMM (no patches yet).
#
# Kernel:   _grouped_fp8_persistent_gemm_kernel  (trans_b=False, FWD)
# Shapes:
#   gate_up_fwd: a[131072, 2880] @ b[32, 2880, 5760] -> out[131072, 5760]  (N=5760)
#   down_fwd:    a[131072, 5760] @ b[32, 5760, 2880] -> out[131072, 2880]  (N=2880)
#
# No hand-tuned patches exist yet. This script provides:
#   --round-trip  Disassemble -> reassemble base binaries to verify the toolchain
#
# When .s patches are added:
#   patch_gate_up_5760.s  ->  base_gate_up_5760.hsaco  ->  final_gate_up_5760.hsaco
#   patch_down_2880.s     ->  base_down_2880.hsaco      ->  final_down_2880.hsaco
#
# Usage:
#   bash build.sh               # build from opt patches (if any)
#   bash build.sh --round-trip  # round-trip toolchain validation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS="${SCRIPT_DIR}/../tools"
PATCH_CO="${TOOLS}/patch_co.py"
DISASM_TO_ASM="${TOOLS}/disasm_to_asm.py"
LLVM_OBJDUMP="${LLVM_OBJDUMP:-/opt/rocm/llvm/bin/llvm-objdump}"
LLVM_MC="${LLVM_MC:-/opt/rocm/llvm/bin/llvm-mc}"

[[ -f "$PATCH_CO" ]] || { echo "ERROR: patch_co.py not found: $PATCH_CO" >&2; exit 1; }

# -- Round-trip: disassemble -> reassemble -> verify ---------------------------
if [[ "${1:-}" == "--round-trip" ]]; then
    echo "=== FWD grouped GEMM round-trip validation ==="
    for variant in "gate_up_5760" "down_2880"; do
        base="${SCRIPT_DIR}/base_${variant}.hsaco"
        raw_dis="/tmp/fwd_${variant}_raw.dis"
        rt_s="/tmp/fwd_${variant}_rt.s"
        rt_co="${SCRIPT_DIR}/rt_${variant}.hsaco"

        [[ -f "$base" ]] || { echo "  SKIP: $base not found"; continue; }
        echo "  [$variant] Disassembling..."
        ${LLVM_OBJDUMP} -d --mcpu=gfx950 "$base" > "$raw_dis"
        echo "  [$variant] Converting to .s..."
        python3 "$DISASM_TO_ASM" "$raw_dis" "$rt_s"
        echo "  [$variant] Reassembling..."
        python3 "$PATCH_CO" "$base" "$rt_s" "$rt_co" --llvm-mc "$LLVM_MC"
        echo "  [$variant] -> $rt_co"
    done
    echo ""
    echo "=== Round-trip done. Run: python3 check.py to validate. ==="
    exit 0
fi

# -- Optimization build: compile opt patches -> .hsaco -------------------------
built=0
for pair in "gate_up_5760" "down_2880"; do
    patch="${SCRIPT_DIR}/patch_${pair}.s"
    base="${SCRIPT_DIR}/base_${pair}.hsaco"
    out="${SCRIPT_DIR}/final_${pair}.hsaco"
    if [[ ! -f "$patch" ]]; then
        echo "  SKIP: patch_${pair}.s not found (no FWD opt yet)"
        continue
    fi
    echo "--- Building final_${pair}.hsaco from patch_${pair}.s ---"
    python3 "$PATCH_CO" "$base" "$patch" "$out" --llvm-mc "$LLVM_MC"
    echo "  -> $out"
    built=$((built + 1))
done

echo ""
if [[ $built -eq 0 ]]; then
    echo "No patch .s files found. Base binaries are unchanged."
    echo ""
    echo "To validate the base binaries:   bash build.sh --round-trip"
    echo "To add FWD optimizations:"
    echo "  1. Disassemble:  llvm-objdump -d --mcpu=gfx950 base_gate_up_5760.hsaco"
    echo "  2. Clean:        python3 ../tools/disasm_to_asm.py raw.dis patch_gate_up_5760.s"
    echo "  3. Edit:         patch_gate_up_5760.s"
    echo "  4. Build:        bash build.sh"
    echo "  5. Check:        python3 check.py"
else
    echo "=== Done ($built kernel(s) built) ==="
    echo "  Check:  python3 check.py"
    echo "  Bench:  python3 bench.py"
fi
