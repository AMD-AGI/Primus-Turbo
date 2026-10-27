#!/bin/bash
# run_tests.sh TAG: run_round.sh's focused tests (EP4 one test per process, EP2 on GPUs 0,1, EP1 on GPU 0)
# against the repo tree, logs -> logs/test_TAG_*.log, one fresh FlyDSL cache per call.
# CONTAINER as in stress.sh (empty = run directly).
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=$1
TURBO=${TURBO:-$(cd "$HERE/../../../../.." && pwd)}
CONTAINER=${CONTAINER-xiaompen_mega_gfx1250_ep4}
CACHE=${CACHE:-/tmp/fly_test_$TAG-$(date +%s)}
L=$HERE/logs; mkdir -p "$L"
EPT=tests/pytorch/ops/test_mega_bf16_gfx1250_fused_moe_ep.py
run() {  # GPUS LIMIT LOG cmd...
    local gpus=$1 limit=$2 log=$3; shift 3
    local envs=(PYTHONPATH=$TURBO TRITON_HIP_USE_EXPERT_SCHEDULING=0 FLYDSL_RUNTIME_CACHE_DIR=$CACHE HIP_VISIBLE_DEVICES=$gpus)
    local cmd="ulimit -c 0; cd $TURBO; $HERE/capped.sh $limit $log $*"
    if [ -n "$CONTAINER" ]; then
        docker exec -w "$TURBO" $(printf -- '-e %s ' "${envs[@]}") "$CONTAINER" bash -c "$cmd" > /dev/null
    else
        env "${envs[@]}" bash -c "$cmd" > /dev/null
    fi
}
for t in test_deepseek_v3 test_ragged_skewed test_one_token test_staged_bias_gpt_oss; do
    run 0,1,2,3 900 $L/test_${TAG}_ep4_$t.log python3 -m pytest $EPT -q -p no:cacheprovider -k $t
    echo "EP4 $t: $(grep -E 'passed|failed|KILLED' $L/test_${TAG}_ep4_$t.log | tail -n 1)"
done
run 0,1 1500 $L/test_${TAG}_ep2.log python3 -m pytest $EPT -q -p no:cacheprovider
echo "EP2: $(grep -E 'passed|failed|KILLED' $L/test_${TAG}_ep2.log | tail -n 1)"
run 0 1500 $L/test_${TAG}_ep1.log python3 -m pytest tests/pytorch/ops/test_mega_bf16_gfx1250_gemm_combine.py \
    tests/pytorch/ops/test_mega_bf16_gfx1250_dispatch_gemm.py tests/pytorch/ops/test_mega_bf16_gfx1250_fused_moe.py \
    -q -p no:cacheprovider
echo "EP1: $(grep -E 'passed|failed|KILLED' $L/test_${TAG}_ep1.log | tail -n 1)"
