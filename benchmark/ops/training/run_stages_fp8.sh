#!/usr/bin/env bash
###############################################################################
# Run each mega-MoE fp8 stage bench in ISOLATION, cleaning the environment
# before (and after) every stage.
#
# A hung / SIGABRT'd prior stage can leave torch.multiprocessing spawn workers
# (and their GPU allocations / faulted contexts) behind, which then poisons the
# next stage. This runner kills any leftover bench procs and waits for VRAM to
# fall back to baseline before each stage, so every stage starts from a clean
# slate and gets its own fresh MASTER_PORT.
#
# Run INSIDE the dev container (it pkills container procs + reads rocm-smi):
#   srun --jobid <J> --overlap docker exec <ctr> bash \
#     /perf_apps/xiaoming/MegaMoE/benchmark/ops/training/run_stages_fp8.sh
#
# Env overrides: REPO T MODE WARMUP ITERS TIMEOUT STAGES VRAM_BASE_MB
###############################################################################
set -u

REPO=${REPO:-/perf_apps/xiaoming/MegaMoE}
cd "$REPO" || { echo "no repo at $REPO"; exit 1; }
export PYTHONPATH="$REPO"

T=${T:-8192}
MODE=${MODE:-load_balanced}
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-10}
TIMEOUT=${TIMEOUT:-400}                 # hard wall per stage (s)
VRAM_BASE_MB=${VRAM_BASE_MB:-1024}      # a GPU is "busy" above this many MB used
BENCH="benchmark/ops/training/bench_mega_moe_fp8.py"
# default: every stage, one at a time (l1/l2/fwd individually, not "both")
STAGES=${STAGES:-"l1 l2 fwd dispatch_fc2_dgrad fc2_wgrad fc1_wgrad fc1_dgrad_combine"}

cleanup_procs() {
    # kill any leftover bench trees: the launcher (has the script name) + the
    # spawn workers / resource-tracker (whose cmdline is only 'spawn_main').
    pkill -9 -f "$BENCH"                         2>/dev/null
    pkill -9 -f "multiprocessing.spawn"          2>/dev/null
    pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null
    sleep 2
}

wait_gpu_idle() {
    # block until every GPU's used VRAM is back under VRAM_BASE_MB (or give up).
    local limit=$((VRAM_BASE_MB * 1024 * 1024)) busy
    for _ in $(seq 1 30); do
        busy=$(rocm-smi --showmeminfo vram 2>/dev/null \
            | awk -v L="$limit" '/Used Memory/ { if ($NF+0 > L) c++ } END { print c+0 }')
        if [ "${busy:-1}" -eq 0 ]; then return 0; fi
        sleep 2
    done
    echo "  [runner] WARN: GPU still busy (>$VRAM_BASE_MB MB) after wait"
    rocm-smi --showpids 2>/dev/null | grep -E "PID|UNKNOWN" | head
}

echo "[runner] repo=$REPO T=$T mode=$MODE warmup=$WARMUP iters=$ITERS timeout=${TIMEOUT}s"
for st in $STAGES; do
    echo ""
    echo "############################## STAGE=$st ##############################"
    cleanup_procs
    wait_gpu_idle
    port=$((20000 + RANDOM % 20000))
    MEGA_BENCH_TIMEOUT_S=$((TIMEOUT - 50)) timeout --signal=KILL "$TIMEOUT" \
        python3 "$BENCH" --num-tokens "$T" --stage "$st" --mode "$MODE" \
        --warmup "$WARMUP" --iters "$ITERS" 2>&1 \
        | grep -vE "UserWarning|warnings.warn|_ensure_sig|FutureWarning|return func\(|jf\(\*args\)"
    rc=${PIPESTATUS[0]}
    echo "----- STAGE=$st rc=$rc ($([ "$rc" = 137 ] && echo HANG/KILLED || ([ "$rc" = 0 ] && echo OK || echo FAIL))) -----"
    cleanup_procs
done
wait_gpu_idle
echo ""
echo "[runner] all stages done."
