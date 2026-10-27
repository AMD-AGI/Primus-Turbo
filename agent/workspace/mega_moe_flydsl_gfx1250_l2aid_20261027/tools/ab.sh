#!/bin/bash
# ab.sh TAG REPS "SHAPES" VARIANT...: same-session A/B, campaign bench (one EP4 torchrun per shape), the variants
# alternating inside every rep and shape (odd reps in the given order, even reps reversed).
# VARIANT = name=TREE[:ENV=V,ENV=V]; TREE `acc` = the last accepted round's copy, `repo` = /home/xiaompen/Turbo.
# Each variant compiles into its own FlyDSL cache. Logs: mega_scratch/l2c/logs/ab_TAG_<name>_<shape>_r<rep>.log
E=/home/xiaompen/mega_scratch/l2c
# GPUs 0,1 belong to another user from 2026-10-27 15:30: default to EP2 on GPUs 2,3, own rendezvous port
GPUS=${GPUS:-2,3}; NP=$(echo $GPUS | tr ',' ' ' | wc -w); MPORT=${MPORT:-29617}
BENCH=/home/xiaompen/Turbo/agent/workspace/mega_moe_flydsl_gfx1250_l2aid_20261027/quick_test_bench.py
TAG=$1; REPS=$2; SHAPES=$3; shift 3
VARS=("$@")
mkdir -p $E/logs
for rep in $(seq 1 $REPS); do
    order=("${VARS[@]}")
    if (( rep % 2 == 0 )); then order=(); for ((i=${#VARS[@]}-1; i>=0; i--)); do order+=("${VARS[$i]}"); done; fi
    for s in $SHAPES; do
        for spec in "${order[@]}"; do
            name=${spec%%=*}; rest=${spec#*=}; tree=${rest%%:*}; envs=""
            [[ $rest == *:* ]] && envs=${rest#*:}
            case $tree in acc) T=$E/accepted ;; repo) T=/home/xiaompen/Turbo ;; *) T=$tree ;; esac
            EX=""; IFS=, read -ra kv <<< "$envs"; for x in "${kv[@]}"; do [ -n "$x" ] && EX="$EX -e $x"; done
            L=$E/logs/ab_${TAG}_${name}_${s}_r$rep.log
            docker exec -w $T -e PYTHONPATH=$T -e TURBO_SRC=$T -e TRITON_HIP_USE_EXPERT_SCHEDULING=0 \
                -e FLYDSL_RUNTIME_CACHE_DIR=$E/cache/ab_${TAG}_$name $EX -e ONLY_SHAPE=$s -e HIP_VISIBLE_DEVICES=$GPUS \
                xiaompen_mega_gfx1250_ep4 bash -c "ulimit -c 0; /home/xiaompen/mega_scratch/capped.sh 600 $L \
                torchrun --nproc-per-node $NP --master-port $MPORT $BENCH" > /dev/null
            echo "r$rep $name $(grep -E '^dsv3|KILLED' $L | sed -E 's/snr\S+=\S+ //')"
        done
    done
done
