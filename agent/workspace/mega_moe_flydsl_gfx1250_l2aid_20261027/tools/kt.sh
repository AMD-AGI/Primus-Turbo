#!/bin/bash
# kt.sh TAG REPS T VARIANT...: same-session kernel-trace A/B (rocprofv3 --kernel-trace only, never --pmc on this
# node), tools/prof_ss.py T under EP4, variants alternating (even reps reversed). VARIANT as in ab.sh.
# Output: mega_scratch/l2c/kt/TAG_<name>_r<rep>/r_<pid>_kernel_trace.csv (one per rank); compare with kt_compare.py.
E=/home/xiaompen/mega_scratch/l2c
GPUS=${GPUS:-2,3}; NP=$(echo $GPUS | tr ',' ' ' | wc -w); MPORT=${MPORT:-29617}
W=/home/xiaompen/Turbo/agent/workspace/mega_moe_flydsl_gfx1250_l2aid_20261027
TAG=$1; REPS=$2; TOK=$3; shift 3
VARS=("$@")
for rep in $(seq 1 $REPS); do
    order=("${VARS[@]}")
    if (( rep % 2 == 0 )); then order=(); for ((i=${#VARS[@]}-1; i>=0; i--)); do order+=("${VARS[$i]}"); done; fi
    for spec in "${order[@]}"; do
        name=${spec%%=*}; rest=${spec#*=}; tree=${rest%%:*}; envs=""
        [[ $rest == *:* ]] && envs=${rest#*:}
        case $tree in acc) T=$E/accepted ;; repo) T=/home/xiaompen/Turbo ;; *) T=$tree ;; esac
        EX=""; IFS=, read -ra kv <<< "$envs"; for x in "${kv[@]}"; do [ -n "$x" ] && EX="$EX -e $x"; done
        D=$E/kt/${TAG}_${name}_r$rep
        rm -rf $D; mkdir -p $D
        docker exec -w $T -e PYTHONPATH=$T -e TURBO_SRC=$T -e TRITON_HIP_USE_EXPERT_SCHEDULING=0 \
            -e FLYDSL_RUNTIME_CACHE_DIR=$E/cache/kt_${TAG}_$name $EX -e HIP_VISIBLE_DEVICES=$GPUS \
            xiaompen_mega_gfx1250_ep4 bash -c "ulimit -c 0; /home/xiaompen/mega_scratch/capped.sh 600 $D/run.log \
            rocprofv3 --kernel-trace --output-format csv -d $D -o r_%pid% -- \
            torchrun --nproc-per-node $NP --master-port $MPORT $W/tools/prof_ss.py $TOK" > /dev/null
        echo "r$rep $name $(find $D -name '*kernel_trace.csv' | wc -l) traces $(grep -c KILLED $D/run.log) killed"
    done
done
