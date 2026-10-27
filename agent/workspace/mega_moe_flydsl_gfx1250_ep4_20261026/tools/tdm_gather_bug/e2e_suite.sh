#!/bin/bash
# e2e_suite.sh TAG [ARMS] [ROUTINGS]: DSV3 proxy EP4 training (Primus run_deepseek_v3_proxy_mi455x.sh, Mega MoE on)
# for every arm x routing, alternating arms per routing; run inside the container as root (the launcher patches
# the container's Triton). ARMS are NAME=TURBO_DIR pairs (default: tdm = the 1149c02b tree, async = this repo),
# ROUTINGS from even / uniform / real (default all three; even / uniform 10 steps, real 50).
# Logs -> $OUT/e2e_TAG_<arm>_<routing>.log, FlyDSL cache $OUT/fly_e2e_TAG_<arm>; compare with parse_train.py.
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=$1
TURBO=${TURBO:-$(cd "$HERE/../../../../.." && pwd)}
PRIMUS=${PRIMUS:-/home/xiaompen/mega_scratch/primus_a8abb804}
OUT=${OUT:-/home/xiaompen/mega_scratch}
ARMS=(${2:-tdm=/home/xiaompen/mega_scratch/ep4val/src_1149c02b async=$TURBO})
ROUTINGS=(${3:-even uniform real})
LIMIT=${LIMIT:-2400}
cd "$PRIMUS"
for r in "${ROUTINGS[@]}"; do
    case $r in
        even) iters=10; extra=() ;;
        uniform) iters=10; extra=(--moe_router_force_load_balancing_type uniform) ;;
        real) iters=50; extra=(--moe_router_force_load_balancing False) ;;
        *) echo "unknown routing $r" >&2; exit 2 ;;
    esac
    for arm in "${ARMS[@]}"; do
        name=${arm%%=*}; dir=${arm#*=}
        log=$OUT/e2e_${TAG}_${name}_${r}.log
        FLYDSL_RUNTIME_CACHE_DIR=$OUT/fly_e2e_${TAG}_${name} \
        PRIMUS_EXP_NAME=e2e_${TAG}_${name}_${r}_$(date +%s) TRAIN_ITERS=$iters PRIMUS_LAUNCHER=direct TURBO_DIR=$dir \
            timeout -k 30 "$LIMIT" bash examples/megatron/models/deepseek-v3/run_deepseek_v3_proxy_mi455x.sh \
            "${extra[@]}" --use_turbo_mega_moe True > "$log" 2>&1
        rc=$?
        pkill -9 -f "pretrain --config" 2>/dev/null
        echo "$name $r: exit $rc, $(grep -E -c 'iteration +[0-9]+/' "$log") steps, \
$(grep -oE 'lm loss: [^|]+' "$log" | grep -ciE 'nan|inf') with nan/inf loss, \
$(grep -oE 'number of nan iterations: +[1-9]' "$log" | wc -l) with nan iterations"
    done
done
