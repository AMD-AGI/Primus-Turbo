#!/bin/bash
# ep2_info.sh N [TAG]: EP2 bench on GPUs 0,1 (all shapes in one process) for the record, not the accept
# gate -> rounds/round-N/artifacts/ep2_bench[_TAG].csv. Fresh FlyDSL cache dir per call.
C=$(cd "$(dirname "$0")" && pwd)
N=$1; TAG=${2:+_$2}
A=$C/rounds/round-$N/artifacts
mkdir -p "$A"
R=/home/xiaompen/Turbo
CACHE=/home/xiaompen/mega_scratch/ep4c_cache/ep2-round-$N$TAG-$(date +%s)
docker exec -w $R -e PYTHONPATH=$R -e TRITON_HIP_USE_EXPERT_SCHEDULING=0 -e FLYDSL_RUNTIME_CACHE_DIR=$CACHE \
    -e HIP_VISIBLE_DEVICES=0,1 xiaompen_mega_gfx1250_ep4 bash -c "ulimit -c 0; \
    /home/xiaompen/mega_scratch/capped.sh 900 $A/ep2_quick$TAG.log \
    torchrun --nproc-per-node 2 $C/quick_test_bench.py --csv $A/ep2_bench$TAG.csv" > /dev/null
grep -E "^dsv3|AGGREGATE|KILLED" "$A/ep2_quick$TAG.log" | sed -E 's/snr\S+=\S+ //'
