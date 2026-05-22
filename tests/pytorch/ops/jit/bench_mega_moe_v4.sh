#!/bin/bash
###############################################################################
# Mega MoE benchmark script for DeepSeek-V4-Flash and DeepSeek-V4-Pro
# parameter configurations (from DeepGEMM PR #316).
#
# Both models are benchmarked under EP8 (8-way expert parallelism).
# Batch sizes = number of tokens per rank: 1, 512, 8192, 32768.
#
# When the kernel shape is JIT-instantiated, actual Time / TFLOPS / HBM /
# Interconnect metrics are printed.  Otherwise the theoretical metric
# inputs (num_recv_tokens, num_touched_experts, etc.) are shown so the
# layout + dispatch pipeline can be validated.
#
# Usage:
#   # Single GPU (default — exercises layout + kernel launch path):
#   bash bench_mega_moe_v4.sh
#
#   # Full EP8 (requires 8 GPUs):
#   NUM_PROCESSES=8 bash bench_mega_moe_v4.sh
#
#   # Layout parity only (no kernel launch):
#   bash bench_mega_moe_v4.sh --parity-only
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PY="${SCRIPT_DIR}/test_mega_moe_jit_perf.py"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
EP_RANKS="${EP_RANKS:-8}"
EXTRA_ARGS=("$@")

# --------------------------------------------------------------------------
# DeepSeek-V4-Flash
#   256 experts, top-k=6, hidden=4096, intermediate_hidden=2048
# --------------------------------------------------------------------------
V4_FLASH_EXPERTS=256
V4_FLASH_TOPK=6
V4_FLASH_HIDDEN=4096
V4_FLASH_INTERMEDIATE=2048

# --------------------------------------------------------------------------
# DeepSeek-V4-Pro
#   384 experts, top-k=6, hidden=7168, intermediate_hidden=3072
# --------------------------------------------------------------------------
V4_PRO_EXPERTS=384
V4_PRO_TOPK=6
V4_PRO_HIDDEN=7168
V4_PRO_INTERMEDIATE=3072

# Batch sizes (tokens per rank) from PR #316
BATCH_SIZES=(1 512 8192 32768)
MAX_BATCH_SIZE=32768

separator() {
    printf '=%.0s' {1..72}
    echo
}

run_config() {
    local model_name="$1"
    local batch_size="$2"
    local num_experts="$3"
    local num_topk="$4"
    local hidden="$5"
    local intermediate_hidden="$6"

    separator
    echo "[${model_name}] batch_size=${batch_size}, experts=${num_experts}, topk=${num_topk}, hidden=${hidden}, intermediate=${intermediate_hidden}, EP=${NUM_PROCESSES}"
    separator

    python "${TEST_PY}" \
        --num-processes "${NUM_PROCESSES}" \
        --ep-ranks "${EP_RANKS}" \
        --num-max-tokens-per-rank "${MAX_BATCH_SIZE}" \
        --num-tokens "${batch_size}" \
        --hidden "${hidden}" \
        --intermediate-hidden "${intermediate_hidden}" \
        --num-experts "${num_experts}" \
        --num-topk "${num_topk}" \
        --skip-parity \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

    echo ""
}

echo ""
echo "############################################################"
echo "#  Mega MoE Benchmark: DeepSeek-V4-Flash & DeepSeek-V4-Pro #"
echo "#  (Configurations from DeepGEMM PR #316)                  #"
echo "#  NUM_PROCESSES=${NUM_PROCESSES}, EP_RANKS=${EP_RANKS}                              #"
echo "############################################################"
echo ""

# --- DeepSeek-V4-Flash ---
echo "============================================================"
echo "  DeepSeek-V4-Flash"
echo "    experts=${V4_FLASH_EXPERTS}, topk=${V4_FLASH_TOPK}, hidden=${V4_FLASH_HIDDEN}, intermediate=${V4_FLASH_INTERMEDIATE}"
echo "============================================================"
echo ""

for bs in "${BATCH_SIZES[@]}"; do
    run_config "V4-Flash" "${bs}" \
        "${V4_FLASH_EXPERTS}" "${V4_FLASH_TOPK}" \
        "${V4_FLASH_HIDDEN}" "${V4_FLASH_INTERMEDIATE}"
done

# --- DeepSeek-V4-Pro ---
echo "============================================================"
echo "  DeepSeek-V4-Pro"
echo "    experts=${V4_PRO_EXPERTS}, topk=${V4_PRO_TOPK}, hidden=${V4_PRO_HIDDEN}, intermediate=${V4_PRO_INTERMEDIATE}"
echo "============================================================"
echo ""

for bs in "${BATCH_SIZES[@]}"; do
    run_config "V4-Pro" "${bs}" \
        "${V4_PRO_EXPERTS}" "${V4_PRO_TOPK}" \
        "${V4_PRO_HIDDEN}" "${V4_PRO_INTERMEDIATE}"
done

separator
echo "All configurations complete."
separator
