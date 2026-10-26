#!/bin/bash
# sweep.sh TAG SHAPE "ENV=V ENV=V ..." : one EP4 torchrun of sweep_dcu_bench.py (repo tree) with the knob
# env vars -> ep4c/logs/TAG_SHAPE.log; prints the result line.
E=/home/xiaompen/mega_scratch/ep4c
mkdir -p $E/logs
TAG=$1; SHAPE=$2; KNOBS=$3
ENVS=""; for kv in $KNOBS; do ENVS="$ENVS -e $kv"; done
R=/home/xiaompen/Turbo
docker exec -w $R -e PYTHONPATH=$R -e TRITON_HIP_USE_EXPERT_SCHEDULING=0 -e ONLY_SHAPE=$SHAPE $ENVS \
    -e FLYDSL_RUNTIME_CACHE_DIR=$E/cache -e HIP_VISIBLE_DEVICES=0,1,2,3 xiaompen_mega_gfx1250_ep4 bash -c "ulimit -c 0; \
    /home/xiaompen/mega_scratch/capped.sh 600 $E/logs/${TAG}_$SHAPE.log \
    torchrun --nproc-per-node 4 $E/sweep_dcu_bench.py" > /dev/null
echo "$TAG [$KNOBS] $(grep -E '^dsv3|KILLED' $E/logs/${TAG}_$SHAPE.log | sed -E 's/snr\S+=\S+ //')"
