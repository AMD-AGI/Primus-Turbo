#!/bin/bash
# bench_only.sh N TAG: quick bench (per-shape SNR + timing) on GPUs 0,1 -> rounds/round-N/artifacts/quick_TAG.log
C=$(cd "$(dirname "$0")" && pwd)
A=$C/rounds/round-$1/artifacts
mkdir -p "$A"
docker exec -w /home/xiaompen/Turbo -e PYTHONPATH=/home/xiaompen/Turbo -e TRITON_HIP_USE_EXPERT_SCHEDULING=0 \
    -e HIP_VISIBLE_DEVICES=0,1 xiaompen_mega_gfx1250_ep4 bash -c "ulimit -c 0; /home/xiaompen/mega_scratch/capped.sh 900 \
    $A/quick_$2.log torchrun --nproc-per-node 2 $C/quick_test_bench.py --csv $A/bench_$2.csv"
grep -E "PASS|FAIL|AGGREGATE|Error|KILLED" "$A/quick_$2.log"
