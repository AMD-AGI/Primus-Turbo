#!/bin/bash
# stage_bench.sh OUTDIR TREE [ENV=V,...]: tools/bench_mega_gfx1250.py (bench_mega_moe.py's gemm_only / comm_only /
# fused / overlap view on the gfx1250 kernels) over its 4 shapes, one EP4 torchrun per shape.
# TREE `acc` = the last accepted round's copy, `repo` = /home/xiaompen/Turbo. Writes OUTDIR/<shape>.{log,csv} and
# OUTDIR/summary.txt (tools/stage_summary.py).
E=/home/xiaompen/mega_scratch/l2c
W=/home/xiaompen/Turbo/agent/workspace/mega_moe_flydsl_gfx1250_l2aid_20261027
OUT=$1; TREE=$2; ENVS=$3
case $TREE in acc) T=$E/accepted ;; repo) T=/home/xiaompen/Turbo ;; *) T=$TREE ;; esac
EX=""; IFS=, read -ra kv <<< "$ENVS"; for x in "${kv[@]}"; do [ -n "$x" ] && EX="$EX -e $x"; done
mkdir -p $OUT
tag=$(echo "$T$ENVS" | md5sum | cut -c1-8)
for s in bench_dsv3 dsv3_t8192_random dsv3_t8192_skew1 dsv3_t4096_random; do
    docker exec -w $T -e PYTHONPATH=$T -e TURBO_SRC=$T -e TRITON_HIP_USE_EXPERT_SCHEDULING=0 \
        -e FLYDSL_RUNTIME_CACHE_DIR=$E/cache/stage_$tag $EX -e ONLY_SHAPE=$s -e HIP_VISIBLE_DEVICES=0,1,2,3 \
        xiaompen_mega_gfx1250_ep4 bash -c "ulimit -c 0; /home/xiaompen/mega_scratch/capped.sh 900 $OUT/$s.log \
        torchrun --nproc-per-node 4 $W/tools/bench_mega_gfx1250.py --csv $OUT/$s.csv" > /dev/null
    echo "$s $(grep -cE 'PASS\]' $OUT/$s.log) PASS $(grep -cE 'FAIL\]' $OUT/$s.log) FAIL $(grep -c KILLED $OUT/$s.log) killed"
done
python3 $W/tools/stage_summary.py $OUT | tee $OUT/summary.txt
