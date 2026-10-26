#!/bin/bash
# ab.sh TAG REPS SHAPES...: same-session A/B of the baseline tree (1149c02b) vs the repo tree, alternating per rep and
# shape, campaign bench (one torchrun per shape) -> logs/ab_TAG_{base,new}_SHAPE_rREP.log; prints fwd+bwd lines.
E=/home/xiaompen/mega_scratch
TAG=$1; REPS=$2; shift 2
BENCH=/home/xiaompen/Turbo/agent/workspace/mega_moe_flydsl_gfx1250_ep4_20261026/quick_test_bench.py
declare -A TREE=([base]=$E/ep4val/src_1149c02b [new]=/home/xiaompen/Turbo)
declare -A CACHE=([base]=$E/ep4val/cache_1149c02b [new]=${NEW_CACHE:-$E/ep4c/cache})
for rep in $(seq 1 $REPS); do
    for s in "$@"; do
        for v in base new; do
            L=$E/ep4c/logs/ab_${TAG}_${v}_${s}_r$rep.log
            docker exec -w ${TREE[$v]} -e PYTHONPATH=${TREE[$v]} -e TRITON_HIP_USE_EXPERT_SCHEDULING=0 \
                -e FLYDSL_RUNTIME_CACHE_DIR=${CACHE[$v]} -e ONLY_SHAPE=$s -e HIP_VISIBLE_DEVICES=0,1,2,3 \
                xiaompen_mega_gfx1250_ep4 bash -c "ulimit -c 0; $E/capped.sh 600 $L torchrun --nproc-per-node 4 $BENCH" \
                > /dev/null
            echo "r$rep $v $(grep -E '^dsv3|KILLED' $L | sed -E 's/snr\S+=\S+ //')"
        done
    done
done
