#!/bin/bash
# run_round.sh N [full|quick]: full = focused tests + bench, quick = bench only (it checks SNR per shape).
# Runs in the xiaompen_mega_gfx1250_ep4 container on GPUs 0,1; logs go to rounds/round-N/artifacts/.
C=$(cd "$(dirname "$0")" && pwd)
N=$1; LEVEL=${2:-quick}
A=$C/rounds/round-$N/artifacts
mkdir -p "$A"
CAP=/home/xiaompen/mega_scratch/capped.sh
EXEC="docker exec -w /home/xiaompen/Turbo -e PYTHONPATH=/home/xiaompen/Turbo -e TRITON_HIP_USE_EXPERT_SCHEDULING=0"
if [ "$LEVEL" = full ]; then
    $EXEC -e HIP_VISIBLE_DEVICES=0,1 xiaompen_mega_gfx1250_ep4 bash -c "ulimit -c 0; $CAP 1500 $A/test_ep.log \
        python3 -m pytest tests/pytorch/ops/test_mega_bf16_gfx1250_fused_moe_ep.py -q"
    $EXEC -e HIP_VISIBLE_DEVICES=0 xiaompen_mega_gfx1250_ep4 bash -c "ulimit -c 0; $CAP 1500 $A/test_ep1.log \
        python3 -m pytest tests/pytorch/ops/test_mega_bf16_gfx1250_gemm_combine.py \
        tests/pytorch/ops/test_mega_bf16_gfx1250_dispatch_gemm.py tests/pytorch/ops/test_mega_bf16_gfx1250_fused_moe.py -q"
    tail -n 2 "$A/test_ep.log" "$A/test_ep1.log"
fi
$EXEC -e HIP_VISIBLE_DEVICES=0,1 xiaompen_mega_gfx1250_ep4 bash -c "ulimit -c 0; $CAP 900 $A/quick_validation.log \
    torchrun --nproc-per-node 2 $C/quick_test_bench.py --csv $A/bench.csv"
grep -E "PASS|FAIL|AGGREGATE|Error|KILLED" "$A/quick_validation.log"
